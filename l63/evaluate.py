"""The four rulers.

    horizon   time steps until ||u-hat_n - u_n|| is too large        plain steps
    spread    sigma_ensemble(model) / sigma_ensemble(truth)          1 = truth
    climate   W1(model, truth) / W1(truth, truth), on the marginals  1 = truth
    chaos     lambda_1(learned map) / lambda_1(true map)             1 = truth

plus `alive`, the fraction of long rollouts still moving like the system -- which the brief
names as one of its two tests, so it is a headline number and not a footnote.

Read as: *how long is it right* -- *is its uncertainty calibrated* -- *does it live in the
right place* -- *does it move the right way* -- *is it still moving at all*.

Nothing here is tuned against a model. Every level a number is read against is measured from
the data and printed beside it, and the levels `horizon` uses are the integrator run against
itself -- `l63.data.references` picks the deterministic or the stochastic pair to match the
dataset. Two constants ARE chosen and neither is threshold-free: `alive_frac`'s 0.25 / 5000
(insensitive, see its docstring) and `vpt`'s 0.4 (the published NRMSE convention, quoted so
outside numbers are comparable and never mixed with the step counts).

Two rulers do not read 1.00 at truth and must be read against the measured truth row instead:
`climate` (see `climate_ratio`) and, on the SDE, anything involving the noise floor.

All errors are in RAW Lorenz units.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from l63.data import Data, raw_err
from l63.lyapunov import lambda_map
from l63.models import ForecastModel


# --- rolling out ------------------------------------------------------------------------

def rollout(model: ForecastModel, xs: Tensor, std: Tensor,
            steps: int | None = None) -> tuple[Tensor, Tensor]:
    """Forecast every trajectory in `xs` from its own first state(s).

    Returns (pred, err): pred is (n, steps + 1, 3) normalised, err is (n, steps + 1) in raw
    units, and both are aligned with `xs[:, k - 1 : k + steps]` for a model needing k warm-up
    states.
    """
    # A model needing k warm-up states can only forecast len - k steps from a trajectory of
    # that length. Capping here rather than trusting the caller keeps every model on the same
    # axis without the caller having to know any model's history length.
    k = model.history
    steps = min(xs.shape[1] - k, steps if steps is not None else xs.shape[1] - k)

    model.eval()
    with torch.no_grad():
        pred = model.forecast(xs[:, :k], steps)

    truth = xs[:, k - 1: k + steps]
    err = raw_err(pred, truth, std)

    assert err.isfinite().all(), 'forecast produced a non-finite state'
    assert err[:, 0].max() < 1e-4, 'step 0 IS the initial condition: there is no error there'

    return pred, err


def rollout_direct(model, xs: Tensor, std: Tensor) -> tuple[Tensor, Tensor]:
    """Forecast every lead time 1..s_max DIRECTLY, with no autoregression. Topic 04 only.

    The comparison this exists for: the same trained network, asked for lead time s in one
    call, versus asked for it as s repeated single steps. The two make different mistakes --
    one approximation of a hard map, or s approximations of an easy map compounding -- and
    which wins says where the long-horizon error actually comes from.
    """
    model.eval()
    s = model.s_max
    with torch.no_grad():
        direct = model.forecast_direct(xs[:, :1], s)
        auto = model.forecast_autoregressive(xs[:, :1], s)

    truth = xs[:, :s + 1]
    return raw_err(direct, truth, std), raw_err(auto, truth, std)


def ensemble(model: ForecastModel, x0: Tensor, steps: int, n_samples: int) -> Tensor:
    """`n_samples` independent forecasts -> (n_samples, n, steps + 1, 3).

    Pointless for a deterministic model, which is why sampling is not baked into `forecast`:
    the probabilistic rungs get their spread from calling it more than once.
    """
    model.eval()
    with torch.no_grad():
        return torch.stack([model.forecast(x0, steps) for _ in range(n_samples)])


def median_curve(e: Tensor) -> Tensor:
    """Median error over initial conditions.

    quantile(0.5), not median(): with an even number of ICs torch.median returns the lower of
    the two middle values, which near the flat ceiling shifts the crossing by tens of steps.
    Median and not mean: chaos makes a few ICs saturate far earlier than the rest, and a mean
    would track those rather than the typical run.
    """
    return e.quantile(0.5, dim=0)


# --- ruler 1: horizon -------------------------------------------------------------------

def first_above(curve: Tensor, level: Tensor | float) -> int:
    """First step at which `curve` exceeds `level` (a scalar or a curve); -1 if never."""
    over = curve > (level if torch.is_tensor(level) else torch.as_tensor(level))
    return int(over.float().argmax()) if bool(over.any()) else -1


def horizon(curve: Tensor, floor: Tensor, bar: Tensor, scale: float) -> dict:
    """How many time steps each of three methods stays usable. Medians, raw units.

    The first three entries are the same question asked of three different methods, so they
    are directly comparable and answer "how many steps can I take" on one axis:

      model_steps       the model's error reaches the attractor scale. Past here the forecast
                        is no better than naming a random point on the attractor.
      euler_bar_steps   the same, for ONE explicit-Euler step of dt per time step -- a
                        classical solver of exactly the same cost as one model call.
      floor_steps       the same, for the gap between Euler at dt_int and at dt_int/2. Past
                        here "tracks the truth" can only mean "tracks OUR numerical ground
                        truth", so this bounds what any claim in this project can say.

    Two crossings, which say where the model sits between those references:

      beyond_noise      the model leaves the ground truth's own uncertainty. Expect n ~ 1;
                        it says the model is not a drop-in for the integrator at any horizon.
      loses_to_euler    where the model first becomes worse than the same-cost solver, or -1.
                        Only meaningful while both curves are still below the ceiling -- once
                        both have saturated the crossing is an artefact of two flat lines, so
                        it is reported as -1 rather than as a step count when that happens.
    """
    n = min(len(curve), len(floor), len(bar))
    model_steps = first_above(curve, scale)
    crossing = first_above(curve[:n], bar[:n])

    # a crossing that happens after the model has already saturated compares two flat lines
    if crossing >= 0 and 0 <= model_steps <= crossing:
        crossing = -1

    return {
        'model_steps': model_steps,
        'euler_bar_steps': first_above(bar, scale),
        'floor_steps': first_above(floor, scale),
        'beyond_noise': first_above(curve[:n], floor[:n]),
        'loses_to_euler': crossing,
        'e1': float(curve[1]),
    }


def early_errors(curve: Tensor, n: int = 8) -> list[float]:
    """||u-hat_n - u_n|| at n = 1 .. n: the first few steps' errors as plain numbers."""
    return [float(v) for v in curve[1:n + 1]]


def vpt(err: Tensor, scale: float, dt: float, lam: float = 0.906,
        thresh: float = 0.4) -> float:
    """Valid Prediction Time in Lyapunov times, the literature convention (NRMSE 0.4).

    Reported so the result can be compared with published numbers. It is measured at a
    stricter bar than the step counts above and the two are NOT interchangeable, so it is
    never reported beside them.
    """
    per_ic = torch.stack([torch.as_tensor(float(first_above(e, thresh * scale)))
                          for e in err])
    per_ic[per_ic < 0] = float(err.shape[1])
    return float(per_ic.median()) * dt * lam


# --- ruler 2: spread --------------------------------------------------------------------

def _ens_sd(ens: Tensor) -> Tensor:
    """Ensemble spread at each lead time: RMS distance of members from the ensemble mean."""
    return ((ens - ens.mean(0)) ** 2).mean(0).sum(-1).sqrt().mean(0)


def truth_spread(d: Data, n_ic: int = 32, steps: int = 200,
                 n_members: int = 16, seed: int = 0) -> tuple[Tensor, int]:
    """Truth's own ensemble spread curve, and the lead time to read `spread` at.

    On the SDE two realisations from the *same* state differ, because the noise is part of
    the system -- so truth has a real ensemble spread and "how uncertain should a forecast
    be here" has an answer measured from the data.

    The lead time is not chosen either: it is the step at which truth's spread first reaches
    half its saturated value, which is where there is most to distinguish. Computed once per
    dataset and passed to every model, both because it is the expensive part and because
    every model must be read at the same lead time for the column to mean anything.

    Returns (nan curve, -1) on the ODE, where truth's ensemble spread is identically zero.

    Seeded, because this curve is the denominator of every model's `spread` and a reference
    that moves between runs is not a reference. Pass a different `seed` to draw a second,
    independent truth ensemble -- scoring that one against the first is how the ruler's own
    noise at this ensemble size gets measured instead of assumed.
    """
    if d.kind == 'ode':
        return torch.full((steps + 1,), float('nan')), -1

    from l63.data import integrate
    raw0 = d.raw(d.eval[:n_ic, 0])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        real = torch.stack([integrate('sde', raw0, 0., steps * d.dt, steps * 100, b=d.b,
                                      keep_every=100).permute(1, 0, 2)
                            for _ in range(n_members)])

    sd = _ens_sd(real)
    return sd, first_above(sd, 0.5 * float(sd[-1]))


def spread_ratio(model: ForecastModel, d: Data, truth_sd: Tensor, lead: int,
                 n_ic: int = 32, steps: int = 200, n_members: int = 16,
                 seed: int = 0) -> float:
    """sigma_ensemble(model) / sigma_ensemble(truth), at the lead time `truth_spread` pinned.

    An ensemble is repeated forecasts from ONE initial condition. A model whose spread is
    below truth's is overconfident; one above it is vague. 1.00 is calibrated.

    NaN on the ODE, where the ratio is 0/0 -- an undefined column rather than a score of 0.
    """
    if lead <= 0:
        return float('nan')

    with torch.random.fork_rng():            # a model's ensemble is its own sampling noise
        torch.manual_seed(seed)              # -- seeded so the column is reproducible
        ens = d.raw(ensemble(model, d.eval[:n_ic, :model.history], steps, n_members))
    return float(_ens_sd(ens)[lead] / truth_sd[lead])


# --- ruler 3: climate -------------------------------------------------------------------

def w1(a: Tensor, b: Tensor, n_q: int = 2000) -> float:
    """1-Wasserstein distance between two 1-D samples, via their inverse CDFs.

    On the line the optimal transport cost has a closed form: the mean absolute gap between
    the two quantile functions. Evaluating both on a shared grid of probabilities is what
    makes it correct for samples of DIFFERENT sizes -- comparing sorted values elementwise
    only works when the two samples are the same length, and silently compares the wrong
    parts of the two distributions when they are not.

    Midpoint probabilities (1/2n, ..., 1 - 1/2n) rather than [0, 1], so the estimate is not
    dominated by the two most extreme order statistics.
    """
    qs = torch.linspace(0.5 / n_q, 1 - 0.5 / n_q, n_q)
    return float((a.flatten().quantile(qs) - b.flatten().quantile(qs)).abs().mean())


def _segments(x: Tensor, seg: int) -> Tensor:
    """(n, T, 3) -> (n * (T // seg), seg, 3): cut trajectories into equal-length pieces.

    This is what puts a model's few long rollouts and truth's many short trajectories on the
    same footing. W1's sampling error is set by the number of *independent* pieces, not by
    the number of states, so 32 rollouts of 20 000 steps and 64 trajectories of 901 are not
    comparable samples however many states each is thinned to. Cut to a common length and
    the two become the same kind of object; `climate_ratio` then draws the same number of
    pieces from each. At 901 steps a piece is 20 Lyapunov times, so consecutive pieces from
    one rollout are decorrelated and count as independent.
    """
    x = x if x.dim() == 3 else x[None]
    k = x.shape[1] // seg
    assert k >= 1, f'need at least {seg} states per trajectory, got {x.shape[1]}'
    return x[:, :k * seg].reshape(-1, seg, 3)


def climate_ratio(model_states: Tensor, truth_states: Tensor, n_splits: int = 20,
                  seed: int = 0) -> tuple[float, float, float]:
    """W1(model, truth) / W1(truth, truth), averaged over the x, y, z marginals.

    Returns (ratio, reference distance, sd of the reference distance).

    The denominator is truth against *itself*, measured between two disjoint halves of the
    true sample. That is what turns an absolute distance into a ratio with a meaning: 1.0 is
    as close as two samples of the true system are to each other, so it is the floor, not a
    target anyone can beat.

    `truth_states` must be a WELL-SAMPLED picture of the attractor and must keep its
    trajectory axis, shape (n_ic, T, 3) -- pass the 128 held-out evaluation trajectories
    (115k states), not the single reference trajectory.

    WHAT THIS GETS RIGHT, each of which silently breaks the ratio if ignored:

    `truth_states` MUST BE LONG TRUE ROLLOUTS OF THE SAME SHAPE AS THE MODEL'S -- pass what
    `ground_truth.truth_rollouts` generates, never the evaluation set. Three things have to
    match or the ruler measures sampling instead of distribution:

      1. Split the truth by *trajectory*, not by state. States along one trajectory are
         strongly correlated; a random split compares a pool against itself and makes the
         denominator far too small.
      2. Match the NUMBER OF INDEPENDENT PIECES, not the number of states. W1's sampling
         error is set by the former. Thinning a model's 32 x 20 000 rollout and truth's
         64 x 901 to the same state count leaves the model with ~11x the effective sample:
         measured, an independent draw of the truth scored 0.565 at 128 trajectories, 0.665
         at 64 and 0.972 at 16, so most of the "score" was sample size.
      3. Match the LENGTH, because 901 steps is only 20 Lyapunov times and that is not
         enough to sample the invariant measure. A single 901-step true trajectory has
         ||std|| = 13.1 +- 3.1 against the attractor's own 14.77. Scored against a reference
         that under-mixed, a model rolled out for 500 Lyapunov times is penalised for
         sampling the attractor properly: truth in the model's own shape scored 1.69.

    So both sides arrive as long rollouts, are cut to a common length, and the same number of
    pieces is drawn from each. With the defaults that is 32 pieces of 20 000 steps per side.

    The split is repeated `n_splits` times and BOTH sides are averaged. The returned sd is
    the per-split spread of the denominator; the sd of the average is smaller by sqrt(n),
    so do not read it as the ratio's own error bar.

    Ratio of means, not mean of ratios: a ratio of two noisy quantities is biased upward.

    WHERE TRUTH LANDS IS MEASURED, NOT ASSUMED. The denominator is two disjoint halves of one
    finite pool, so an outlier in one half is necessarily absent from the other and the two
    are pushed slightly apart; an independent draw is under no such constraint. The anchor is
    therefore the measured `truth_climate` in ground_truth.json, and `climate_vs_truth`
    divides by it -- that ratio cancels the effect, because it acts identically on truth and
    on every model. Read a model against the truth row, never against 1.

    ITS RESOLUTION. Coarse. The numerical noise floor alone is several per cent (see the
    k = 1 control in report.py's `controls`). It separates gross failures confidently and
    fine distributional differences not at all.

    KNOWN BLIND SPOT: this is marginals-only. It does not measure joint structure -- a model
    can match all three marginals and still put the mass in the wrong shape. The attractor
    grid and the Lorenz map cover that by eye, with no number behind them.
    """
    m = model_states if model_states.dim() == 3 else model_states[None]
    t = truth_states if truth_states.dim() == 3 else truth_states[None]
    half = t.shape[0] // 2
    assert half >= 1, 'climate needs a truth sample with at least two trajectories'

    seg = min(m.shape[1], t.shape[1])
    ms = _segments(m, seg)

    nums, dens = [], []
    for s in range(n_splits):
        g = torch.Generator().manual_seed(seed + s)
        perm = torch.randperm(t.shape[0], generator=g)
        a = _segments(t[perm[:half]], seg)
        b = _segments(t[perm[half:2 * half]], seg)

        n = min(len(a), len(b), len(ms))
        pick = lambda x: x[torch.randperm(len(x), generator=g)[:n]]
        A, B, M = pick(a), pick(b), pick(ms)

        nums.append(np.mean([w1(M[..., i], B[..., i]) for i in range(3)]))
        dens.append(np.mean([w1(A[..., i], B[..., i]) for i in range(3)]))

    return float(np.mean(nums) / np.mean(dens)), float(np.mean(dens)), float(np.std(dens))


# --- ruler 4: chaos ---------------------------------------------------------------------

def chaos_ratio(model: ForecastModel, d: Data, lam_true: float,
                n_ic: int = 32, n_steps: int = 2000) -> tuple[float, float]:
    """lambda_1 of the learned map divided by lambda_1 of the true map. Returns (ratio, sd).

    The same Benettin estimator, the same time step, the same number of steps -- only the map
    differs. That is what makes it a fair ratio rather than a comparison against a literature
    constant measured some other way.

    Sampling is switched OFF for the measurement. With it on, the map is not a function of
    the state alone and the estimator would measure the injected noise rather than the
    dynamics -- the number would come out large and mean nothing. The consequence is that
    for a sampled model this column describes the mean/zero-base map, not the map the other
    rulers roll out, and it is read as such. It is also why `mean_only` carries no `chaos`:
    it would be this same number by construction, so the two could never differ.

    THE MAP IS THE MODEL'S OWN, via `map_state` / `map_step`. Whatever object `forecast`
    iterates is what gets measured -- the window for a Markov model, (u, h, c) for a
    recurrent one. Building the map here instead, out of repeated one-step `forecast` calls,
    measures a memoryless system for the recurrent rungs: 05_lstm_ode_s0 then reads horizon
    440 (chaotic, wing-switching for 20 000 steps) beside chaos 0.11 (not chaotic at all),
    because those two numbers are about different systems.

    Reading it: 1.0 means the model stretches phase space at the true rate. Below 1 it is too
    smooth -- the classic failure of an MSE-trained net, which regresses toward the mean and
    damps the very sensitivity that defines the system. At or below 0 the model is not
    chaotic at all: it has found a fixed point or a limit cycle.

    `lam_true` is the DETERMINISTIC flow map's exponent on both datasets. The SDE has no
    deterministic map to take an exponent of; its drift is the same system, so the same
    denominator is the sensible comparison, but the ratio on the SDE is "against the
    underlying deterministic dynamics", not "against the process that made the data".
    """
    k = model.history
    was_sampling = getattr(model, 'sampling', None)
    if was_sampling is not None:
        model.sampling = False

    try:
        net = model.double()
        with torch.no_grad():
            h0 = net.map_state(d.eval[:n_ic, :k].double())
        lam = lambda_map(net.map_step, h0, d.dt, n_steps=n_steps)
    finally:
        model.float()
        if was_sampling is not None:
            model.sampling = was_sampling

    return float(lam.mean() / lam_true), float(lam.std())


# --- the gate ---------------------------------------------------------------------------

def alive_frac(states: Tensor, truth_std: float, window: int = 5000,
               ratio: float = 0.25) -> float:
    """Fraction of long rollouts whose last `window` steps still move like the system.

    A rollout is dead if the spread of its final window has collapsed below `ratio` of the
    truth's -- it has fallen into a fixed point or a tight cycle. Measured on the FINAL
    window and nowhere else: a trailing-window criterion has no false "recovered" state and
    needs no persistence tuning, unlike a probe that slides along the trajectory.

    `ratio` and `window` ARE chosen numbers -- the only two in the ruler suite. They are not
    tuned against any model: the quantity is bimodal (a rollout keeps most of the attractor's
    spread or almost none of it), so anything between roughly 0.05 and 0.6 gives the same
    partition, and the truth row is measured through this same function as the check. This
    ruler is insensitive to those two constants, not threshold-free, and the truth row is
    printed beside it.

    The brief names "stays on the attractor" as one of its two tests. Report it with its
    across-seed range, which on the SDE is the whole interval [0, 1] for the deterministic
    rungs.
    """
    sd = states[:, -window:].std(dim=1).norm(dim=-1)
    return float((sd >= ratio * truth_std).float().mean())


def lobe_frac(states: Tensor, dt: float, lam: float = 0.906) -> Tensor:
    """Wing switches per Lyapunov time, per rollout. Blind spot cover for `alive_frac`.

    Spread-based collapse detection does not detect a model that keeps moving at full
    amplitude but inside ONE wing -- a period-p orbit reads as alive. Counting sign changes
    of x catches exactly that, and truth's own value is the reference.
    """
    sign = states[..., 0].sign()
    switches = (sign[:, 1:] != sign[:, :-1]).sum(dim=1).float()
    return switches / ((states.shape[1] - 1) * dt * lam)
