"""Topic 00 and 01: the ground truth every model inherits, computed once.

This script does three things, and nothing else may recompute them:

  1. Reproduces the standard results (Strogatz ch. 9). If they do not come out, the
     integrator is wrong and nothing downstream is worth running.
  2. Generates the frozen datasets -- the deterministic ODE control and the SDE at both
     the noise level `run/width_sweep.py` selected and the starter notebook's own.
  3. Measures the reference values every ruler is read against: the attractor scale, the
     two integrator curves that define "too large" FOR THAT SYSTEM, lambda of the true map,
     truth's own ensemble spread, and what truth itself scores on every ruler.

The truth row is measured through the same functions the models are, not asserted. It is the
calibration: `climate` does not read 1.00 at truth and `alive` is a proportion of finite
rollouts, so a model column is read against the truth row and never against 1.

Writes artifacts/ground_truth.pt (curves) and artifacts/ground_truth.json (scalars, which
run/report.py and the notebooks read).

Run:  PYTHONPATH=. python run/ground_truth.py            [--only sde015]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from l63 import ARTIFACTS, known as K
from l63.data import (CONFIG, DT, conditional_width, integrate, make_datasets, lorenz,
                      references, solve_ode)
from l63.lyapunov import lambda_map, spectrum, true_step
from l63 import evaluate as E

# The datasets, by the key that names every result file. A key is a (integrator, noise pair):
#
#   ode     the deterministic control. The experiment runs on stochastic trajectories, so
#           this is not the experiment -- it is the b -> 0 limit, where p(u_{n+1}|u_n) is a
#           point mass and a probabilistic model has nothing to recover. It shows that the
#           SDE comparison is real and not an artefact of the ruler.
#   sde     b = 0.6, chosen from the five-point sweep in run/width_sweep.py: conditional
#           width 16.9 % of attractor scale, with the attractor itself only 7 % wider than
#           the deterministic one. 0.3 also passes the sweep's own test; 0.6 is the wider of
#           the two that still leaves the attractor undistorted.
#   sde015  the starter notebook's own b. Included because a change of noise level needs a
#           trained answer and not just a width measurement: at 4.2 % of attractor scale the
#           claim is that every rung scores the same, and this is what tests it.
DATASETS = {
    'ode':    dict(kind='ode', b=0.0),
    'sde':    dict(kind='sde', b=0.6),
    'sde015': dict(kind='sde', b=0.15),
}

# 890, not 900: the recurrent models need 8 warm-up states before they can forecast at all,
# so 890 is the longest window EVERY model can produce. Same axis for all of them matters
# more than ten extra steps at the far end, where all the curves are flat anyway.
EVAL_STEPS = 890
LONG_STEPS = 20_000       # long rollouts, for climate / chaos / alive
N_LONG = 32               # how many of them, for truth and for every model alike
N_ANCHOR = 5              # independent truth blocks scored on climate, to get its resolution


def known_results() -> dict:
    """The standard results. Each one printed against what the textbook says it is.

    Two kinds of row, and they are not the same kind of evidence. C+-, div f and rho_H are
    closed forms evaluated -- they check that the formulas were transcribed correctly and
    cannot fail for any other reason. The spectrum, its sum against div f, Kaplan-Yorke, the
    symmetry residual and the Lorenz map's slope are numerical and CAN fail; those are the
    integrator check. `measured_divergence` promotes div f into the second group by taking
    the trace of the Jacobian at real states rather than quoting the constant.
    """
    out = {}
    o, cp, _ = K.fixed_points()
    out['C_plus'] = cp.tolist()
    out['divergence'] = K.divergence()
    out['rho_hopf'] = K.rho_hopf()
    out['symmetry_residual'] = K.symmetry_residual(torch.randn(2000, 3) * 15)
    out['eigs'] = {k: [complex(v).__repr__() for v in ev]
                   for k, ev in K.stability().items()}

    lams, _ = spectrum(torch.tensor([0., 1., 0.]), t_total=400., dt_int=2.5e-4)
    out['spectrum'] = lams.tolist()
    out['spectrum_sum'] = float(lams.sum())
    out['kaplan_yorke'] = K.kaplan_yorke(lams.tolist())

    long = solve_ode(lorenz, torch.tensor([[0., 1., 0.]]), 0., 400., 1_600_000)[::4, 0]
    zn, zn1 = K.lorenz_map(long[:, 2])
    _, slopes = K.map_slope(zn, zn1)
    out['lorenz_map_peaks'] = len(zn)
    out['lorenz_map_min_slope'] = float(abs(slopes).min())

    # trace(Df) at 2000 real attractor states: div f as a measurement, not as a constant
    tr = K.measured_divergence(long[::800])
    out['divergence_measured'] = float(tr.mean())
    out['divergence_measured_max_dev'] = float((tr - K.divergence()).abs().max())

    print(f'  --- closed forms, transcription checks (cannot fail) ---')
    print(f'  C+                  {cp.tolist()}          expected (+-8.485, +-8.485, 27)')
    print(f'  div f (formula)     {out["divergence"]:.4f}                     expected -13.6667')
    print(f'  rho_H               {out["rho_hopf"]:.4f}                      expected 24.7368')
    print(f'  --- numerical, these are the integrator check ---')
    print(f'  div f (measured)    {out["divergence_measured"]:.6f}   max deviation '
          f'{out["divergence_measured_max_dev"]:.1e} over 2000 attractor states')
    print(f'  symmetry residual   {out["symmetry_residual"]:.1e}                       expected 0')
    print(f'  spectrum            {[round(v, 4) for v in out["spectrum"]]}   expected (0.906, 0, -14.57)')
    print(f'  sum of spectrum     {out["spectrum_sum"]:.4f}   vs div f {out["divergence"]:.4f}  <- the identity')
    print(f'  Kaplan-Yorke dim    {out["kaplan_yorke"]:.4f}                      Strogatz measured ~2.05')
    print(f'  Lorenz map |f\'|min  {out["lorenz_map_min_slope"]:.3f}  over {len(zn)} maxima   must be > 1')
    return out


def truth_rollouts(d, steps: int, n: int, seed: int = 0) -> torch.Tensor:
    """`n` long TRUE trajectories in raw units, generated exactly like a model's rollout.

    Same length and same starting states as the models get -- so `alive`, `lobe` and
    `climate` can be measured on truth through the identical function a model goes through.
    Without this the truth row is three hardcoded 1.00s, which is an assertion and not a
    calibration.

    This needs more rollouts than the evaluation set holds (the climate reference needs a
    pool plus several held-out blocks), so past the first 128 the starts are fresh draws put
    on the attractor by the same spin-up `gen_data` uses. Taking `d.eval[:n, 0]` instead
    would silently return 128 rollouts and halve every sample this feeds.
    """
    z0 = d.raw(d.eval[:, 0])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if n > len(z0):
            extra = torch.randn(n - len(z0), 3) * 10.
            spin = CONFIG['spinup'] * 100
            extra = integrate(d.kind, extra, 0., CONFIG['spinup'] * d.dt, spin,
                              b=d.b, keep_every=spin)[-1]
            z0 = torch.cat([z0, extra])
        zs = integrate(d.kind, z0[:n], 0., steps * d.dt, steps * 100, b=d.b,
                       keep_every=100)
    return zs.permute(1, 0, 2)


def reference_for(key: str, lam_true: float) -> tuple[dict, dict]:
    """Everything a model trained on `key` is scored against. Returns (scalars, curves)."""
    kind, b = DATASETS[key]['kind'], DATASETS[key]['b']
    d = make_datasets(seed=0, kind=kind, b=b)
    print(f'\n[{key}] {d}')

    # The floor and the bar, matched to the system that made the data. On the SDE the floor
    # is realisation-vs-realisation -- irreducible, and nothing to do with the step size --
    # and the bar shares its Brownian path with the fine reference so it measures method and
    # not noise. Using the deterministic pair here makes every SDE model read "loses to Euler
    # at step 1", which is an artefact of the reference, not a result.
    z0 = d.raw(d.eval[:, 0])
    floor_c, bar_c = references(kind, z0, EVAL_STEPS, d.dt, b=b)
    floor, bar = E.median_curve(floor_c), E.median_curve(bar_c)

    # Truth, generated exactly the way a model's rollout is: same count, same length, same
    # starting states. The evaluation set is NOT usable as the climate reference: 901 steps
    # is 20 Lyapunov times, far too short to sample the invariant measure, and scoring
    # 500-tau model rollouts against it penalises a model for mixing properly.
    #
    # The first 2 * N_LONG are the frozen denominator pool every model is scored against.
    # The rest are N_ANCHOR independent held-out blocks, each the size of one model's
    # rollout set, scored through the identical call a model gets. That is the anchor -- and
    # it has to be several blocks, because ONE of them is not an anchor, it is one draw from
    # a distribution with ~50 % spread. Quoting a single draw makes this ruler look biased
    # (0.565) when it is unbiased and merely coarse.
    long_truth = truth_rollouts(d, LONG_STEPS, (2 + N_ANCHOR) * N_LONG)
    climate_ref = long_truth[:2 * N_LONG]

    anchors, ref_w1, ref_sd = [], 0., 0.
    for k in range(N_ANCHOR):
        block = long_truth[(2 + k) * N_LONG:(3 + k) * N_LONG]
        r, ref_w1, ref_sd = E.climate_ratio(block, climate_ref)
        anchors.append(r)
    truth_climate = float(np.median(anchors))

    fresh = long_truth[2 * N_LONG:3 * N_LONG]
    truth_sd, lead = E.truth_spread(d)
    truth_alive = E.alive_frac(fresh, d.scale)
    truth_lobe = float(E.lobe_frac(fresh, d.dt, lam_true).mean())

    # the attractor's own scale, from a well-mixed sample, beside the evaluation set's.
    # The rulers use the evaluation set's -- that is the data the error is measured on --
    # but the two differ by ~5 %, so both are reported.
    scale_long = float(long_truth.reshape(-1, 3).std(dim=0).norm())

    # and truth's spread ruler against ITSELF: a second, independent truth ensemble scored
    # through spread_ratio's own arithmetic. 1.00 is the definition; this is the measurement,
    # and the gap is the ruler's noise at this ensemble size.
    truth_spread = float('nan')
    if lead > 0:
        second, _ = E.truth_spread(d, seed=1)
        truth_spread = float(second[lead] / truth_sd[lead])

    width = conditional_width(z0[:40], d.dt, b=b).mean().item() if kind == 'sde' else 0.

    scal = dict(
        key=key, kind=kind, b=b, dt=d.dt,
        attractor_scale=d.scale, attractor_scale_long=scale_long,
        lambda_true=lam_true,
        floor_steps=E.first_above(floor, d.scale),
        euler_bar_steps=E.first_above(bar, d.scale),
        truth_climate=truth_climate, truth_climate_lo=float(min(anchors)),
        truth_climate_hi=float(max(anchors)), truth_climate_draws=anchors,
        climate_ref_w1=ref_w1, climate_ref_sd=ref_sd,
        truth_alive=truth_alive, truth_lobe=truth_lobe, truth_spread=truth_spread,
        spread_lead=lead,
        conditional_width=width,
        conditional_width_pct=100 * width / d.scale,
    )
    print(f'  attractor scale        {d.scale:.2f} on the eval set, {scale_long:.2f} well-mixed')
    print(f'  same-cost solver usable {scal["euler_bar_steps"]} steps')
    print(f'  ground truth usable    {scal["floor_steps"]} steps'
          + ('   <- realisation noise, not discretisation' if kind == 'sde' else ''))
    print(f'  truth scores on climate {truth_climate:.3f}  over {N_ANCHOR} independent draws, '
          f'range {min(anchors):.3f}-{max(anchors):.3f}  <- the ruler\'s resolution')
    print(f'  truth alive {truth_alive:.2f}   lobe {truth_lobe:.3f} switches/tau'
          + (f'   spread {truth_spread:.3f}' if lead > 0 else ''))
    if kind == 'sde':
        print(f'  conditional width      {width:.3f} = {scal["conditional_width_pct"]:.1f}% of scale')
        print(f'  spread read at lead    {lead} steps')

    # climate_ref travels with the curves because every model has to be scored against the
    # SAME truth sample -- regenerating it per job would put ruler noise in every column.
    return scal, dict(floor=floor, bar=bar, truth_sd=truth_sd, climate_ref=climate_ref)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--only', default=None, help='recompute one dataset key, e.g. sde015')
    a = p.parse_args()

    torch.manual_seed(0)
    ARTIFACTS.mkdir(exist_ok=True)

    out_path, curve_path = ARTIFACTS / 'ground_truth.json', ARTIFACTS / 'ground_truth.pt'
    have = json.load(open(out_path)) if (a.only and out_path.exists()) else None
    curves = torch.load(curve_path, weights_only=False) if (have and curve_path.exists()) else {}

    if have:
        known, lam_true, lam_sd = have['known'], have['lambda_true'], have['lambda_true_sd']
        scalars = dict(have['datasets'])
        print(f'reusing known results and lambda_true from {out_path.name}')
    else:
        print('=== the standard results (Strogatz ch. 9) ===')
        known = known_results()

        # lambda of the TRUE MAP at dt, through the same estimator every model gets, at the
        # same n_steps. Using the model's own estimator rather than a literature constant is
        # what makes that ratio like-for-like.
        print('\n=== lambda of the true map, same estimator the models get ===')
        z = solve_ode(lorenz, torch.randn(64, 3) * 10., 0., 5., 20_000)[-1]
        lm = lambda_map(true_step(DT), z, DT, n_steps=8000)
        lam_true, lam_sd = float(lm.mean()), float(lm.std())
        print(f'  lambda_map  {lam_true:.4f} +- {lam_sd:.4f} over 64 ICs '
              f'(literature 0.906, spectrum {known["spectrum"][0]:.4f})')
        scalars = {}

    print('\n=== the datasets ===')
    for key in DATASETS:
        if a.only and key != a.only:
            continue
        scalars[key], curves[key] = reference_for(key, lam_true)

    out = dict(known=known, lambda_true=lam_true, lambda_true_sd=lam_sd,
               datasets=scalars, eval_steps=EVAL_STEPS, long_steps=LONG_STEPS,
               n_long=N_LONG)
    json.dump(out, open(out_path, 'w'), indent=2)
    torch.save(curves, curve_path)
    print('\nwrote artifacts/ground_truth.json and ground_truth.pt')


if __name__ == '__main__':
    main()
