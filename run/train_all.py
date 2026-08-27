"""Train and score every model on every dataset. One process per job, `n_workers` at a time.

The matrix is (model x dataset x seed). Every job writes two files and reads none from any
other job, so the run is resumable: a job whose result exists is skipped, and killing the run
loses at most the jobs in flight.

    artifacts/ckpt/<name>.pt        weights + the kwargs to rebuild the model + loss history
    artifacts/results/<name>.json   one row of the head-to-head table

`--score-only` re-runs the rulers against the checkpoints already on disk without touching a
weight. That is the mode to use when a ruler changes: re-scoring 124 models costs a fraction
of retraining them, and it keeps the change to the measurement provably separate from any
change to the models.

Same training budget for every model -- the comparison is loss-against-loss, not
tuning-against-tuning. Where that is unfair to a model, its notebook says so. It is a budget
in ITERATIONS only: the transformer carries 100k parameters against the MLP's 17k, so it is a
control on architecture and not a matched-capacity comparison.

Run:  PYTHONPATH=. python run/train_all.py     [--only 03] [--data sde015] [--workers 6]
                                               [--force] [--score-only]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pool

import torch
from torch import nn

from l63 import ARTIFACTS
from l63 import evaluate as E
from l63.data import make_datasets
from l63.models import (FlowMatchingPredictor, ForecastModel, GaussianPredictor,
                        LeadTimePredictor, Predictor, RecurrentPredictor, RolloutPredictor,
                        TransformerPredictor)
from l63.train import train

CKPT, RESULTS = ARTIFACTS / 'ckpt', ARTIFACTS / 'results'
SEEDS = (0, 1, 2, 3, 4)
N_ITERS = 3000
ACTS = {'SiLU': nn.SiLU, 'Tanh': nn.Tanh, 'ReLU': nn.ReLU}

# Dataset keys, defined in run/ground_truth.py. `ode` and `sde` get the whole ladder; the
# starter's noise level gets the four rungs that answer whether it was wide enough to run on
# at all -- the two extremes of the deterministic side and both probabilistic rungs.
FULL = ('ode', 'sde')
NARROW = 'sde015'
NARROW_MODELS = ('02', '03', '06', '06f')

REGISTRY: dict[str, type[ForecastModel]] = {
    'Predictor': Predictor, 'RolloutPredictor': RolloutPredictor,
    'LeadTimePredictor': LeadTimePredictor, 'RecurrentPredictor': RecurrentPredictor,
    'GaussianPredictor': GaussianPredictor, 'TransformerPredictor': TransformerPredictor,
    'FlowMatchingPredictor': FlowMatchingPredictor,
}


@dataclass(frozen=True)
class Job:
    topic: str
    label: str
    cls: str
    data: str            # a key of ground_truth.json's `datasets`, not an integrator kind
    seed: int
    kwargs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f'{self.topic}_{self.label}_{self.data}_s{self.seed}'


def matrix() -> list[Job]:
    """Every job. Slowest first, so the pool has no long tail waiting on one straggler."""
    jobs: list[Job] = []
    for data in FULL + (NARROW,):
        narrow = data == NARROW
        for seed in SEEDS:
            def add(topic, *a, **kw):
                if not narrow or topic in NARROW_MODELS:
                    jobs.append(Job(topic, *a, **kw))

            # AR arms -- the same nets, trained through the COMPOSED map instead of on true
            # states only. Deterministic rungs only: 02's AR counterpart already exists as
            # topic 03, and 06 is trained per-step on purpose (see GaussianPredictor's
            # docstring -- unrolling an MSE through a sampled chain collapses its sigma head).
            # k = 4 for both -- topic 03's own five-seed sweep is what puts the optimum there,
            # and holding k equal across the rungs is what lets the poster read them side by
            # side. Neither rung is in NARROW_MODELS, so `add` drops these on sde015 itself.
            # Both unroll from EVERY valid start, as `RolloutPredictor` does; see
            # `RecurrentPredictor.ar_loss` for why that is the fairness condition.
            add('05', 'lstm_ar4', 'RecurrentPredictor', data, seed,
                dict(cell='lstm', ar=True, k=4))
            add('04', 'leadtime_ar4', 'LeadTimePredictor', data, seed, dict(ar=True, k=4))

            # transformer and flow matching are the two expensive rungs (330 s and 220 s
            # against 80 s for the MLP), so they go first and the cheap ones fill in behind.
            add('05t', 'transformer', 'TransformerPredictor', data, seed)
            add('06f', 'flow', 'FlowMatchingPredictor', data, seed)
            for k in (16, 8, 4, 1):
                add('03', f'rollout_k{k}', 'RolloutPredictor', data, seed, dict(k=k))
            for cell in ('lstm', 'rnn'):
                add('05', cell, 'RecurrentPredictor', data, seed, dict(cell=cell))
            add('06', 'gaussian', 'GaussianPredictor', data, seed)
            add('04', 'leadtime', 'LeadTimePredictor', data, seed)
            add('02', 'mlp', 'Predictor', data, seed)

        # architecture sweep: one seed. Its held-out loss column is the interpretable one,
        # not its horizon -- the across-seed spread of a single configuration covers most of
        # the horizon column's range.
        if not narrow:
            for w in (32, 64, 256):
                jobs.append(Job('02s', f'width{w}', 'Predictor', data, 0, dict(hidden_dim=w)))
            for nh in (1, 4):
                jobs.append(Job('02s', f'depth{nh}', 'Predictor', data, 0, dict(n_hidden=nh)))
            for a in ('Tanh', 'ReLU'):
                jobs.append(Job('02s', f'act{a}', 'Predictor', data, 0, dict(act=a)))

    return jobs


def build(job: Job) -> ForecastModel:
    kw = dict(job.kwargs)
    if 'act' in kw:
        kw['act'] = ACTS[kw['act']]
    return REGISTRY[job.cls](**kw)


def rulers(m: ForecastModel, d, gt: dict, curves: dict, key: str) -> dict:
    """All four rulers plus the gate. Identical code for every model, called once per model
    -- twice for the sampled rungs, whose controlled test needs the same weights both ways.
    """
    ref = gt['datasets'][key]
    floor, bar = curves[key]['floor'], curves[key]['bar']

    _, err = E.rollout(m, d.eval, d.std, steps=gt['eval_steps'])
    curve = E.median_curve(err)
    h = E.horizon(curve, floor, bar, d.scale)

    long_raw = d.raw(m.forecast(d.eval[:gt['n_long'], :m.history], gt['long_steps']))

    # against long TRUE rollouts of the same shape, not the evaluation set -- see
    # evaluate.climate_ratio. The reference is frozen in ground_truth.pt so every model is
    # scored against the identical truth sample.
    climate, _, _ = E.climate_ratio(long_raw, curves[key]['climate_ref'])
    chaos, chaos_sd = E.chaos_ratio(m, d, ref['lambda_true'], n_ic=32, n_steps=8000)
    spread = E.spread_ratio(m, d, curves[key]['truth_sd'], ref['spread_lead'])

    return dict(
        horizon=h['model_steps'], horizon_detail=h,
        early=E.early_errors(curve, 8),
        vpt=E.vpt(err, d.scale, d.dt),
        spread=spread, climate=climate, chaos=chaos, chaos_sd=chaos_sd,
        # divided by what truth itself scores, so 1.00 means "as close to truth as an
        # independent draw of truth is". The raw ratio does not read 1.00 at truth.
        climate_vs_truth=climate / ref['truth_climate'],
        alive=E.alive_frac(long_raw, d.scale),
        lobe=float(E.lobe_frac(long_raw, d.dt, ref['lambda_true']).mean()),
        curve=[float(v) for v in curve[::5]],       # every 5th step, for the figure
    )


def evaluate_model(m: ForecastModel, d, gt: dict, curves: dict, key: str) -> dict:
    row = dict(**rulers(m, d, gt, curves, key), note='')

    # Topic 04 answers "does forecasting directly to lead time s beat s repeated steps?", and
    # that is what `direct` measures. Its `forecast` is the s = 1 head iterated, not chained
    # blocks: chaining blocks makes this rung's `chaos` describe a different map from its
    # `horizon`. Iterating keeps every column here about one object, and the rung's result is
    # the direct-vs-autoregressive table.
    if isinstance(m, LeadTimePredictor):
        direct_err, auto_err = E.rollout_direct(m, d.eval, d.std)
        row['direct'] = E.early_errors(E.median_curve(direct_err), m.s_max)
        row['autoregressive'] = E.early_errors(E.median_curve(auto_err), m.s_max)
        row['note'] = ('read `direct` vs `autoregressive`; every other column is the s = 1 '
                       'map iterated, which is not what this rung was trained to be good at')

    # The controlled test: the SAME weights rolled out using the mean instead of a sample.
    # One difference, nothing else -- which is what isolates whether sampling itself helps or
    # hurts. It also measures how large the learned sigma is: on the ODE the true conditional
    # is a point mass, so any nonzero sigma is spurious width that `forecast` then injects at
    # every step.
    #
    # `chaos` is dropped from the control: chaos_ratio switches sampling off itself, so the
    # two rows would carry the identical number by construction and could never differ.
    if hasattr(m, 'sampling'):
        m.sampling = False
        drop = ('curve', 'horizon_detail', 'early', 'chaos', 'chaos_sd')
        row['mean_only'] = {k: v for k, v in rulers(m, d, gt, curves, key).items()
                            if k not in drop}
        m.sampling = True

        # sigma is a Gaussian-specific readout: flow matching has no scale parameter, its
        # width is whatever the learned transport produces.
        if isinstance(m, GaussianPredictor):
            with torch.no_grad():
                mu, log_sigma = m(d.eval[:, 0])
                resid = ((mu - d.eval[:, 1]) * d.std).norm(dim=-1).median()
            row['sigma_raw'] = float(log_sigma.median().exp() * d.std.norm())
            row['residual_raw'] = float(resid)
            row['sigma_over_residual'] = row['sigma_raw'] / row['residual_raw']

    return row


def _load(job: Job):
    """Rebuild the trained model and its history from the checkpoint. Self-describing."""
    blob = torch.load(CKPT / f'{job.name}.pt', weights_only=False)
    m = build(job)
    m.load_state_dict(blob['state'])
    m.eval()
    return m, blob


def run_job(args: tuple[Job, bool]) -> tuple[str, str]:
    """Train (or load), score, and write. Returns (name, status) -- never raises into the pool."""
    job, score_only = args
    torch.set_num_threads(1)
    t0 = time.time()

    try:
        gt = json.load(open(ARTIFACTS / 'ground_truth.json'))
        curves = torch.load(ARTIFACTS / 'ground_truth.pt', weights_only=False)
        ref = gt['datasets'][job.data]
        d = make_datasets(seed=0, kind=ref['kind'], b=ref['b'])

        if score_only:
            m, blob = _load(job)
            hist_train, hist_val = blob['train'], blob['val']
        else:
            torch.manual_seed(job.seed)
            m = build(job)
            m, hist = train(m, d.train, d.val, n_iters=N_ITERS, progress=False)
            hist_train, hist_val = hist.train, hist.val

        # Scoring DRAWS, for two of the rungs: `forecast` samples on the Gaussian and on flow
        # matching, and `spread_ratio` builds an ensemble out of repeated calls. This seed
        # must stay outside the train/score branch above. Seeding only inside the training
        # branch makes `--score-only` return a different answer on every run for exactly the
        # two rungs whose numbers are drawn rather than computed, and makes a re-score
        # disagree with the score taken at training time. Deterministic rungs are unaffected:
        # they draw nothing.
        torch.manual_seed(job.seed)

        row = dict(name=job.name, topic=job.topic, label=job.label, cls=job.cls,
                   data=job.data, kind=ref['kind'], b=ref['b'], seed=job.seed,
                   kwargs=job.kwargs, n_params=m.n_params, history=m.history,
                   train_loss=hist_train[-1], val_loss=hist_val[-1] if hist_val else None,
                   **evaluate_model(m, d, gt, curves, job.data))
        row['seconds'] = round(time.time() - t0, 1)
        row['scored_only'] = score_only

        CKPT.mkdir(parents=True, exist_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        if not score_only:
            torch.save(dict(cls=job.cls, kwargs=job.kwargs, state=m.state_dict(),
                            train=hist_train, val=hist_val), CKPT / f'{job.name}.pt')
        json.dump(row, open(RESULTS / f'{job.name}.json', 'w'), indent=2)

        return job.name, (f'{row["seconds"]:6.1f}s  horizon {row["horizon"]:4d}  '
                          f'chaos {row["chaos"]:6.2f}  climate {row["climate"]:6.2f}  '
                          f'alive {row["alive"]:.2f}')
    except Exception:
        return job.name, 'FAILED\n' + traceback.format_exc()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--only', default=None, help='run one topic, e.g. 03')
    p.add_argument('--data', default=None, help='run one dataset key, e.g. sde015')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--force', action='store_true', help='rerun jobs that already have results')
    p.add_argument('--score-only', action='store_true',
                   help='re-run the rulers against existing checkpoints; trains nothing')
    a = p.parse_args()

    jobs = [j for j in matrix()
            if (a.only is None or j.topic == a.only) and (a.data is None or j.data == a.data)]

    if a.score_only:
        missing = [j for j in jobs if not (CKPT / f'{j.name}.pt').exists()]
        jobs = [j for j in jobs if (CKPT / f'{j.name}.pt').exists()]
        if missing:
            print(f'{len(missing)} jobs have no checkpoint and are skipped '
                  f'(train them first): {missing[0].name} ...')
    elif not a.force:
        jobs = [j for j in jobs if not (RESULTS / f'{j.name}.json').exists()]

    if not jobs:
        print('nothing to do -- every result already exists (pass --force to rerun)')
        return

    verb = 're-scoring' if a.score_only else 'training'
    print(f'{verb} {len(jobs)} jobs on {a.workers} workers\n')
    t0 = time.time()
    done = failed = 0
    with Pool(a.workers) as pool:
        for name, status in pool.imap_unordered(run_job, [(j, a.score_only) for j in jobs]):
            done += 1
            failed += status.startswith('FAILED')
            print(f'[{done:3d}/{len(jobs)}] {name:34s} {status}', flush=True)

    print(f'\n{done} jobs in {(time.time() - t0) / 60:.1f} min'
          + (f'   {failed} FAILED' if failed else ''))


if __name__ == '__main__':
    main()
