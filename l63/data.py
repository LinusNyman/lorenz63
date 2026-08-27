"""The true system, both integrators, and the datasets every model is judged on.

Two ground truths, and every model is trained and scored on both:

    ODE   du = f(u) dt                    explicit Euler          the deterministic control
    SDE   du = f(u) dt + g(u) dW          Euler-Maruyama          what the brief prescribes

Both are integrated on a fine grid of dt_int = 2.5e-4 and every 100th step is kept, so one
*time step* -- the step a model predicts, and the unit of every step count in this project --
is dt = 0.025 time units. The 100 substeps buy accuracy in u_{n+1}; they are not withheld
data. The kept series is u_0, u_1, ..., u_N: consecutive, complete, one training pair per
consecutive pair.

Everything here is in RAW Lorenz units unless a name says `normalised`. Training happens in
normalised units for conditioning; `raw_err` is what converts an error back before it is
reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from l63 import ARTIFACTS

A = (10., 28., 8 / 3)      # sigma, rho, beta -- Lorenz's own values
B = 0.15                   # multiplicative volatility; only the SDE uses it


def lorenz(x: Tensor, t: Tensor | float | None = None, a: Sequence = A) -> Tensor:
    """The Lorenz 63 vector field. (n, 3) -> (n, 3). Autonomous, so `t` is ignored."""
    x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
    a1, a2, a3 = a

    f1 = a1 * (x2 - x1)
    f2 = a2 * x1 - x2 - x1 * x3
    f3 = x1 * x2 - a3 * x3

    return torch.cat([f1, f2, f3], dim=1)


def lorenz_vol(x: Tensor, t: Tensor | float | None = None, b: float | Sequence = B) -> Tensor:
    """State-multiplicative volatility g(u) = b * u, for the SDE.

    Multiplicative and not additive: the noise scales with the state, so it does not
    swamp the origin or vanish out on the wings. `b` is the one knob that sets how wide
    the conditional p(u_{n+1} | u_n) is, and therefore how much room the deterministic
    model has to fail -- see `conditional_width`.
    """
    b = torch.as_tensor(b, dtype=x.dtype, device=x.device)
    return x * b


def solve_ode(
        ode: Callable[[Tensor, Tensor], Tensor],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        keep_every: int = 1,
) -> Tensor:
    """Explicit Euler. Returns (n_steps // keep_every + 1, *z.shape), starting at `z`.

    `keep_every` drops the intermediate substeps as it goes instead of afterwards. The
    arithmetic is identical -- it only decides what is stored -- but it is the difference
    between running and not: a 20 000-step rollout at 100 substeps over 224 trajectories is
    2 million states, which is terabytes kept and 54 MB dropped.
    """
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1]
    dt = (tf - ts) / n_steps

    path = [z]
    for i, t in enumerate(tt):
        z = z + ode(z, t) * dt

        if (i + 1) % keep_every == 0:
            path.append(z)

    return torch.stack(path)


def solve_sde(
        sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        keep_every: int = 1,
) -> Tensor:
    """Euler-Maruyama; `sde(z, t)` returns (drift, volatility). `keep_every` as in solve_ode.

    The sqrt(dt) on the noise term is not a choice: a Wiener increment over dt has
    standard deviation sqrt(dt), so anything else would make the limit depend on the grid.
    """
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = [z]
    for i, t in enumerate(tt):
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        if (i + 1) % keep_every == 0:
            path.append(z)

    return torch.stack(path)


def sde_of(b: float = B) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """The (drift, volatility) pair `solve_sde` wants, at noise level `b`."""
    return lambda z, t: (lorenz(z, t), lorenz_vol(z, t, b))


def integrate(kind: str, z: Tensor, ts: float, tf: float, n_steps: int,
              b: float = B, keep_every: int = 1) -> Tensor:
    """`solve_ode` or `solve_sde`, chosen by name. The one place `kind` is dispatched."""
    if kind == 'ode':
        return solve_ode(lorenz, z, ts, tf, n_steps, keep_every)
    if kind == 'sde':
        return solve_sde(sde_of(b), z, ts, tf, n_steps, keep_every)
    raise ValueError(f"kind must be 'ode' or 'sde', got {kind!r}")


# --- units ------------------------------------------------------------------------------
# Models train on normalised states because unit-variance inputs condition the optimisation.
# Nothing is REPORTED in those units: a reader asked to interpret ||u-hat - u|| should not
# have to hold a normalisation in their head, so every error goes back to raw x, y, z first.

def raw_err(pred: Tensor, truth: Tensor, std: Tensor) -> Tensor:
    """||u-hat_n - u_n|| in raw Lorenz units, from normalised tensors.

    The mean cancels in a difference, so only the per-component std is needed:
    (a - b)_raw = (a - b)_normalised * std, componentwise, and then the Euclidean norm.
    """
    return ((pred - truth) * std).norm(dim=-1)


def attractor_scale(zs: Tensor) -> tuple[Tensor, float]:
    """Per-component std of the attractor, and the size of a typical state vector.

    ||std|| is the number a forecast error is finally compared against: once the error is
    that big the forecast is no better than naming a random point on the attractor.
    """
    s = zs.reshape(-1, 3).std(dim=0)
    return s, s.norm().item()


# --- the reference curves that define "too large" ---------------------------------------
# Neither is a chosen threshold. Both are the integrator run against itself, which is what
# makes them answerable: one says how good the ground truth itself is, the other says what
# a classical solver of the SAME NUMBER OF FUNCTION EVALUATIONS as the model achieves.
#
# There are two versions of each because a stochastic system has a different irreducible
# uncertainty from a deterministic one. Using the ODE curves on the SDE compares a noise-free
# reference against a noisy target and makes every SDE model look like it loses to Euler at
# step 1. Use `references(kind=...)` and let it dispatch; do not call the ODE pair directly
# on SDE data.

def euler_floor(z0: Tensor, n: int, dt: float, n_inner: int = 100) -> Tensor:
    """The ground truth's own uncertainty: Euler at dt_int against Euler at dt_int/2.

    Returns (batch, n + 1) in raw units. Past the step where this reaches the attractor
    scale, "the model tracks the truth" can only mean "tracks OUR numerical ground truth" --
    which is why it is measured rather than assumed.
    """
    ref = solve_ode(lorenz, z0, 0., n * dt, n * n_inner, keep_every=n_inner)
    half = solve_ode(lorenz, z0, 0., n * dt, n * n_inner * 2, keep_every=n_inner * 2)
    return (ref - half).norm(dim=-1).T


def euler_bar(z0: Tensor, n: int, dt: float, n_inner: int = 100) -> Tensor:
    """A classical solver of the same cost as the model: ONE Euler step of dt per time step.

    Returns (batch, n + 1) in raw units, measured against the fine reference. This is the
    bar the model has to clear to be worth using at all -- one call, one step, same budget.
    """
    ref = solve_ode(lorenz, z0, 0., n * dt, n * n_inner, keep_every=n_inner)
    one = solve_ode(lorenz, z0, 0., n * dt, n)
    return (ref - one).norm(dim=-1).T


def sde_references(z0: Tensor, n: int, dt: float, b: float, n_inner: int = 100,
                   seed: int = 0) -> tuple[Tensor, Tensor]:
    """The SDE's own floor and bar, from one shared Brownian path. Returns (floor, bar).

    Both are (batch, n + 1) in raw units, and both are the stochastic analogues of
    `euler_floor` / `euler_bar` -- which must NOT be used here, because they are computed
    from the deterministic drift alone and so ignore the thing that actually limits
    prediction on this dataset.

      floor  two INDEPENDENT realisations started from the same state. This is the
             irreducible uncertainty of a stochastic system: it is not discretisation error,
             it does not shrink if the step is refined, and no model -- and no solver --
             can get below it. It bounds any "tracks the truth" claim here.

      bar    Euler-Maruyama at dt against Euler-Maruyama at dt_int driven by *the same*
             Brownian path, so the difference is discretisation and nothing else. Sharing
             the path is required: two independent coarse and fine runs would differ by the
             floor above, and the bar would measure the noise instead of the method.
    """
    h, dt_2 = dt / n_inner, (dt / n_inner) ** 0.5
    g = lambda z: lorenz_vol(z, None, b)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        fine = coarse = alt = z0
        out_f, out_c, out_a = [fine], [coarse], [alt]

        for _ in range(n):
            dw = torch.zeros_like(z0)
            for _ in range(n_inner):
                w = torch.randn_like(z0) * dt_2
                fine = fine + lorenz(fine) * h + g(fine) * w
                dw = dw + w                              # the coarse step's own increment
                alt = alt + lorenz(alt) * h + g(alt) * (torch.randn_like(z0) * dt_2)
            coarse = coarse + lorenz(coarse) * dt + g(coarse) * dw
            out_f.append(fine); out_c.append(coarse); out_a.append(alt)

    f = torch.stack(out_f)
    return ((f - torch.stack(out_a)).norm(dim=-1).T,
            (f - torch.stack(out_c)).norm(dim=-1).T)


def references(kind: str, z0: Tensor, n: int, dt: float, b: float = B,
               n_inner: int = 100) -> tuple[Tensor, Tensor]:
    """(floor, bar) for whichever system generated the data. The one place this is chosen.

    Non-finite values become +inf rather than NaN, and that case is real, not defensive: on
    the SDE at b = 0.6 the same-cost solver -- one Euler-Maruyama step of dt -- **diverges**.
    A multiplicative increment of 0.6 |u| sqrt(dt) is around 10 % of the state every step and
    the coarse scheme does not survive it. So the bar on that dataset is not "worse than the
    model", it is "unusable", which is the stronger statement. +inf keeps
    `first_above` and the figure's truncation working; both stop at the attractor scale long
    before the blow-up anyway.
    """
    if kind == 'ode':
        out = euler_floor(z0, n, dt, n_inner), euler_bar(z0, n, dt, n_inner)
    elif kind == 'sde':
        out = sde_references(z0, n, dt, b, n_inner)
    else:
        raise ValueError(f"kind must be 'ode' or 'sde', got {kind!r}")
    inf = float('inf')
    return tuple(torch.nan_to_num(c, nan=inf, posinf=inf, neginf=inf) for c in out)


def conditional_width(z0: Tensor, dt: float, b: float = B, n_samples: int = 500,
                      n_inner: int = 100, seed: int = 0) -> Tensor:
    """Width of p(u_{n+1} | u_n) for the SDE, at each anchor state in `z0`.

    Freeze a state, integrate `n_samples` independent realisations for exactly one time
    step, and take the standard deviation of the resulting cloud. Returns (batch,) as
    ||std|| in raw units.

    This is the experiment's independent variable. A deterministic model trained by MSE
    targets the MEAN of this cloud (see models.Predictor); if the cloud is a point, the
    mean is the flow map and nothing is lost. Only once the cloud is wide does a
    probabilistic model have anything to recover.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        widths = []
        for z in z0:
            cloud = integrate('sde', z[None].repeat(n_samples, 1), 0., dt, n_inner,
                              b=b, keep_every=n_inner)[-1]
            widths.append(cloud.std(dim=0).norm())
    return torch.stack(widths)


# --- the frozen datasets ----------------------------------------------------------------
# Every model is trained on the same pairs and scored on the same 128 held-out initial
# conditions. Changing anything here changes every model's numbers, so it lives in one place
# and the notebooks only ever call make_datasets().

CONFIG = dict(
    ts=0., tf=10., n_steps=400,           # train / validation: 400 time steps
    train_size=2 ** 10, val_size=2 ** 3,
    # The eval trajectories are LONGER (same time step, 25 time units instead of 10). A
    # one-step error of ~1e-2 needs roughly 1000x growth to reach the size of the attractor,
    # which at this system's growth rate takes ~8 time units -- so a 10-time-unit window
    # would censor the slow initial conditions and the 90th percentile would read "never"
    # when it means "not yet".
    eval_size=128, eval_tf=25., eval_n_steps=1000,
    ref_tf=60., ref_n_steps=2400,         # long truth, for backdrops and for climate
    n_inner_steps=100, spinup=100,
    kind='ode', b=B,
)

DT = (CONFIG['tf'] - CONFIG['ts']) / CONFIG['n_steps']    # 0.025 time units per time step


def gen_data(
        batch_size: int,
        ts: float,
        tf: float,
        n_steps: int,
        kind: str = 'ode',
        b: float = B,
        init_std: float = 10.,
        n_inner_steps: int = 100,
        spinup: int = 100,
        stats: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """`batch_size` trajectories, integrated finely and sampled every `n_inner_steps`.

    Returns (normalised states, (mean, std)). Shape (batch, n_steps + 1 - spinup, 3).
    """
    z0 = torch.randn(batch_size, 3) * init_std
    zs = integrate(kind, z0, ts, tf, n_steps * n_inner_steps, b=b, keep_every=n_inner_steps)
    zs = zs.permute(1, 0, 2)

    # Drop the spin-up BEFORE measuring the units: the transient starts far off the attractor
    # and would inflate std, so the normalised data would not have unit variance.
    zs = zs[:, spinup:]

    # The units are frozen by the first caller and passed to every other set, so training,
    # validation and evaluation data are all in the same units.
    mean, std = stats if stats is not None else (zs.mean(dim=(0, 1)), zs.std(dim=(0, 1)))

    return (zs - mean) / std, (mean, std)


@dataclass(frozen=True)
class Data:
    """Everything the notebooks generate once, in one object. States are normalised."""
    train: Tensor          # (1024, 301, 3)  defines the units
    val: Tensor            # (8, 301, 3)     disjoint initial conditions
    eval: Tensor           # (128, 901, 3)   held-out, longer
    ref: Tensor            # (2301, 3)       one long true trajectory
    mean: Tensor
    std: Tensor
    kind: str = 'ode'
    b: float = B
    dt: float = DT

    @property
    def stats(self) -> tuple[Tensor, Tensor]:
        return self.mean, self.std

    @property
    def scale(self) -> float:
        """||std|| of the attractor in raw units: the level at which a forecast is a guess."""
        return self.std.norm().item()

    def raw(self, xs: Tensor) -> Tensor:
        """Normalised states back to raw Lorenz units."""
        return xs * self.std + self.mean

    def __repr__(self) -> str:
        return (f'Data[{self.kind}] train {tuple(self.train.shape)}  val {tuple(self.val.shape)}'
                f'  eval {tuple(self.eval.shape)}  ref {tuple(self.ref.shape)}  dt {self.dt}'
                f'  scale {self.scale:.2f}')


def make_datasets(seed: int = 0, kind: str = 'ode', b: float = B,
                  cache: bool = True, c: dict | None = None) -> Data:
    """Generate (or load) the frozen train / validation / eval / reference sets.

    Deterministic in `seed`, so the cache is a speed-up and never a source of truth; it is
    keyed on the full config and regenerated if anything in it changes.
    """
    c = {**CONFIG, **(c or {}), 'kind': kind, 'b': b}
    # `b` is in the filename, not only in the config that is checked after loading: two noise
    # levels of the same kind are two different datasets and must not share a cache slot, or
    # running them in one session makes each regenerate over the other's file.
    path = ARTIFACTS / f'data_{kind}_b{b:g}_seed{seed}.pt'

    if cache and path.exists():
        blob = torch.load(path, weights_only=False)
        if blob['config'] == c and blob['seed'] == seed:
            return blob['data']

    torch.manual_seed(seed)
    kw = dict(kind=kind, b=b, n_inner_steps=c['n_inner_steps'], spinup=c['spinup'])

    train, stats = gen_data(c['train_size'], c['ts'], c['tf'], c['n_steps'], **kw)
    val, _ = gen_data(c['val_size'], c['ts'], c['tf'], c['n_steps'], stats=stats, **kw)
    eval_, _ = gen_data(c['eval_size'], c['ts'], c['eval_tf'], c['eval_n_steps'],
                        stats=stats, **kw)
    ref, _ = gen_data(1, c['ts'], c['ref_tf'], c['ref_n_steps'], stats=stats, **kw)

    dt = (c['tf'] - c['ts']) / c['n_steps']
    mean, std = stats

    # Same units everywhere, and unit variance -- which is what every ruler assumes.
    assert torch.allclose(train.mean(dim=(0, 1)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(train.std(dim=(0, 1)), torch.ones(3), atol=1e-5)
    assert torch.allclose(eval_.std(dim=(0, 1)), torch.ones(3), atol=0.15)
    assert abs(c['eval_tf'] / c['eval_n_steps'] - dt) < 1e-12   # eval MUST use the same step

    data = Data(train=train, val=val, eval=eval_, ref=ref[0],
                mean=mean, std=std, kind=kind, b=b, dt=dt)

    if cache:
        ARTIFACTS.mkdir(exist_ok=True)
        torch.save({'config': c, 'seed': seed, 'data': data}, path)

    return data
