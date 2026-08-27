"""Assemble every result into the standing figures and one summary the Typst sources read.

`artifacts/results/*.json` is one row per (model, dataset, seed). This aggregates them over
seeds, banks the figures per topic, and writes `artifacts/summary.json` -- which is the ONLY
number source the .typ files read. No number is typed into a .typ file by hand.

SEEDS. Every ruler is reported as a median over the five seeds with its full range, `horizon`
included. One seed is not enough: across the ladder a single seed lands between 45 % and
284 % of its own median, and for two rungs seed 0 is the best of the five. The seed-0 value
is still written out as `horizon_s0` so the figures, which show one model, can be checked
against the row.

FIGURES. Each topic's figures come from the seed whose horizon is closest to the median, not
from seed 0, so the picture and the table describe the same model.

CONTROLS. Topic 03 at k = 1 is topic 02's loss through the rollout code path -- the same
weights to within float summation order. Scoring both and printing the gap gives every ruler
a numerical noise floor that owes nothing to seeds, which bounds how small a difference this
suite can resolve.

Run:  PYTHONPATH=. python run/report.py            [--topic 02] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

from l63 import ARTIFACTS, evaluate as E, plots as P
from l63.data import make_datasets
from l63.models import ForecastModel
from l63.train import History
from run.train_all import CKPT, RESULTS, REGISTRY, ACTS

# the model shown in each topic's figures
REPRESENTATIVE = {'02': 'mlp', '03': 'rollout_k8', '04': 'leadtime', '05': 'lstm',
                  '05t': 'transformer', '06': 'gaussian', '06f': 'flow'}

TITLES = {'02': 'MLP one-step predictor', '03': 'MLP with a rollout loss ($k=8$)',
          '04': 'Lead-time predictor', '05': 'LSTM', '05t': 'Transformer',
          '06': 'Gaussian predictor', '06f': 'Flow matching'}

RULERS = ('spread', 'climate', 'climate_vs_truth', 'chaos', 'alive', 'lobe', 'horizon')


def load_rows() -> list[dict]:
    return [json.load(open(p)) for p in sorted(RESULTS.glob('*.json'))]


def clean(o):
    """NaN and +-inf to null.

    Python's json.dump writes bare `NaN`, which is not valid JSON and which Typst does not
    parse -- so an undefined ruler silently breaks the whole .typ build instead of rendering
    as a dash. Undefined columns are real here (`spread` on the ODE), so they have to survive
    the round trip as null. The notebooks call this too, so they print "—" instead of "nan".
    """
    if isinstance(o, float):
        return None if (o != o or o in (float('inf'), float('-inf'))) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def load_model(name: str) -> tuple[ForecastModel, History]:
    """Rebuild a trained model from its checkpoint. Self-describing: class + kwargs are in it."""
    blob = torch.load(CKPT / f'{name}.pt', weights_only=False)
    kw = dict(blob['kwargs'])
    if 'act' in kw:
        kw['act'] = ACTS[kw['act']]
    m = REGISTRY[blob['cls']](**kw)
    m.load_state_dict(blob['state'])
    m.eval()

    h = History(); h.train, h.val = blob['train'], blob['val']
    return m, h


def agg(rows: list[dict], key: str) -> dict:
    """median and (min, max) over seeds, ignoring NaN."""
    v = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return dict(median=float('nan'), lo=float('nan'), hi=float('nan'), n=0)
    return dict(median=float(np.median(v)), lo=float(v.min()), hi=float(v.max()), n=len(v))


def median_seed(rows: list[dict]) -> int:
    """The seed whose horizon is nearest the median -- the one the figures should show."""
    med = np.median([r['horizon'] for r in rows])
    return min(rows, key=lambda r: (abs(r['horizon'] - med), r['seed']))['seed']


def summarise(rows: list[dict]) -> dict:
    """One entry per (topic, label, dataset), aggregated over seeds."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r['topic'], r['label'], r.get('data', r.get('kind')))].append(r)

    out = {}
    for (topic, label, data), rs in sorted(by.items()):
        rs.sort(key=lambda r: r['seed'])
        s0 = rs[0]
        rep = median_seed(rs)
        r_rep = next(r for r in rs if r['seed'] == rep)

        # A job records only the kwargs it was CONSTRUCTED with, so a default never appears --
        # `LeadTimePredictor()` leaves s_max out entirely. The checkpoint is self-describing,
        # so this recovers the model's own resolved config from it rather than duplicating it
        # into the job table, where it would be a second copy to keep in step.
        cfg = {}
        try:
            cfg = dict(getattr(load_model(f'{topic}_{label}_{data}_s{rep}')[0], 'cfg', {}))
        except Exception:
            pass

        entry = dict(
            topic=topic, label=label, data=data, kind=s0.get('kind'), b=s0.get('b'),
            n_seeds=len(rs), rep_seed=rep,
            n_params=s0['n_params'], history=s0['history'],
            kwargs={**cfg, **s0['kwargs']},
            note=s0.get('note', ''),
            # every ruler: median over seeds with the full range
            **{k: agg(rs, k) for k in RULERS},
            # and the representative model's own numbers, which are what the figures show
            horizon_s0=s0['horizon'], horizon_rep=r_rep['horizon'],
            horizon_detail=r_rep['horizon_detail'],
            early=r_rep['early'], vpt=r_rep['vpt'], seconds=s0.get('seconds'),
            train_loss=r_rep['train_loss'], val_loss=r_rep['val_loss'],
        )
        entry['horizon_seeds'] = entry['horizon']         # alias, for readers of the old key

        if 'mean_only' in s0:
            # the controlled test gets the same five seeds as everything else -- it is read
            # on `alive` and `climate`, the two most seed-unstable columns in the suite
            mo = [r['mean_only'] for r in rs if 'mean_only' in r]
            entry['mean_only'] = {k: agg(mo, k) for k in RULERS if k in mo[0]}
            for k in ('sigma_raw', 'residual_raw', 'sigma_over_residual'):
                if k in s0:
                    entry[k] = agg(rs, k)
        if 'direct' in s0:
            entry['direct'], entry['autoregressive'] = r_rep['direct'], r_rep['autoregressive']

        out[f'{topic}_{label}_{data}'] = entry
    return out


def controls(rows: list[dict]) -> dict:
    """The k = 1 control, per seed and per dataset: the suite's numerical noise floor.

    `Predictor` and `RolloutPredictor(k=1)` are the same model and the same loss. They are not
    bit-identical only because one reshapes to (N, 3) and the other keeps (batch, T, 3), so
    the sums happen in a different order; after 3000 Adam steps the weights differ by ~7e-7.
    Whatever a ruler does with that difference is what it cannot resolve, before any seed,
    any architecture and any training choice enters. Free to compute, since both rows already
    exist, and it bounds any claim that two models differ.
    """
    by = {(r['topic'], r['label'], r.get('data', r.get('kind')), r['seed']): r for r in rows}
    out = {}
    for (topic, label, data, seed), r in by.items():
        if (topic, label) != ('02', 'mlp'):
            continue
        k1 = by.get(('03', 'rollout_k1', data, seed))
        if k1 is None:
            continue
        out[f'{data}_s{seed}'] = {
            k: dict(mlp=r[k], rollout_k1=k1[k],
                    rel=abs(r[k] - k1[k]) / max(abs(r[k]), 1e-12))
            for k in ('horizon', 'climate', 'chaos', 'alive', 'lobe', 'vpt')
            if r.get(k) is not None and k1.get(k) is not None
        }
    return out


# ============================================================ figures

def figures_ground_truth(gt: dict) -> None:
    """Topic 00: what the data is, and the Lorenz map of the truth itself."""
    for key, ref in gt['datasets'].items():
        d = make_datasets(seed=0, kind=ref['kind'], b=ref['b'])
        P.data_figure(d, f'00a_data_{key}.png')
        P.lorenz_map_figure(d.raw(d.eval), d.raw(d.eval), f'00f_lorenz_map_{key}.png',
                            label='truth')


def figures_for(topic: str, key: str, gt: dict, curves: dict, summary: dict) -> None:
    """The standing figures for one topic, from its representative (median-horizon) model."""
    label = REPRESENTATIVE[topic]
    s = summary.get(f'{topic}_{label}_{key}')
    if s is None:
        return
    name = f'{topic}_{label}_{key}_s{s["rep_seed"]}'
    if not (CKPT / f'{name}.pt').exists():
        print(f'  skip {name}: no checkpoint')
        return

    m, hist = load_model(name)
    ref = gt['datasets'][key]
    d = make_datasets(seed=0, kind=ref['kind'], b=ref['b'])

    ylab = {'06': 'Gaussian NLL   (normalised units)',
            '06f': 'flow-matching velocity loss'}.get(
                topic, 'one-step MSE   (normalised units)')
    P.loss_figure(hist, f'{topic}b_loss_{key}.png', n_val_traj=8, ylabel=ylab)
    P.arch_figure(m.spec(), TITLES[topic], m.n_params, f'{topic}c_arch_{key}.png',
                  note=f'Adam, 3000 iterations, lr 1e-3   ·   seed {s["rep_seed"]}')

    _, err = E.rollout(m, d.eval, d.std, steps=gt['eval_steps'])
    curve = E.median_curve(err)
    P.error_figure(curve, curves[key]['floor'], curves[key]['bar'], d.scale,
                   s['horizon_detail'], d.dt, f'{TITLES[topic]} (median of 128)',
                   f'{topic}d_error_{key}.png')

    long_raw = d.raw(m.forecast(d.eval[:gt['n_long'], :m.history], gt['long_steps']))
    P.lorenz_map_figure(long_raw, d.raw(d.eval), f'{topic}f_lorenz_map_{key}.png',
                        label=TITLES[topic])

    # No `e` figure. The rulers go into a Typst table instead, because a table can show the
    # median AND its range across five seeds -- and the range is the part that decides whether
    # two models differ at all. `plots.ruler_figure` remains available for notebook use.


def figures_head_to_head(gt: dict, summary: dict) -> None:
    """Topic 07: every model against truth, on every dataset."""
    for key, ref in gt['datasets'].items():
        d = make_datasets(seed=0, kind=ref['kind'], b=ref['b'])

        # the truth row, measured through the same functions the models go through -- not
        # three hardcoded 1.00s. climate does not read 1.00 at truth, and alive is a
        # proportion of finite rollouts, so both are calibration and not targets.
        rows = {'truth': dict(horizon=ref['floor_steps'],
                              spread=ref.get('truth_spread', float('nan')),
                              climate=ref['truth_climate'], climate_vs_truth=1.0,
                              chaos=1.0, alive=ref['truth_alive'], lobe=ref['truth_lobe'])}
        states = {'truth': d.raw(d.eval)}

        for topic, label in REPRESENTATIVE.items():
            s = summary.get(f'{topic}_{label}_{key}')
            if s is None:
                continue
            rows[TITLES[topic]] = {k: s[k]['median'] for k in
                                   ('spread', 'climate', 'climate_vs_truth', 'chaos',
                                    'alive', 'lobe')}
            rows[TITLES[topic]]['horizon'] = int(s['horizon']['median'])

            name = f'{topic}_{label}_{key}_s{s["rep_seed"]}'
            if (CKPT / f'{name}.pt').exists():
                m, _ = load_model(name)
                states[TITLES[topic]] = d.raw(m.forecast(d.eval[:4, :m.history], 8000))

        P.attractor_grid(states, f'07a_attractors_{key}.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--topic', default=None)
    p.add_argument('--no-figures', action='store_true')
    a = p.parse_args()

    gt = json.load(open(ARTIFACTS / 'ground_truth.json'))
    curves = torch.load(ARTIFACTS / 'ground_truth.pt', weights_only=False)
    rows = load_rows()
    print(f'{len(rows)} results')

    summary = summarise(rows)
    json.dump(clean(dict(ground_truth=gt, models=summary, controls=controls(rows))),
              open(ARTIFACTS / 'summary.json', 'w'), indent=2, allow_nan=False)
    print(f'wrote artifacts/summary.json  ({len(summary)} model x dataset entries)')

    if a.no_figures:
        return

    figures_ground_truth(gt)
    for topic in REPRESENTATIVE:
        if a.topic and topic != a.topic:
            continue
        for key in gt['datasets']:
            figures_for(topic, key, gt, curves, summary)
    figures_head_to_head(gt, summary)


if __name__ == '__main__':
    main()
