"""Lyapunov exponents, for the true system and for a learned map.

Strogatz 9.3 gives three qualifications on "the Lyapunov exponent", and this module is built
around them:

  1. The divergence is not exactly exponential -- its strength varies over the attractor. So
     every number here is an average with a stated spread, never a single-pair slope fit.
  2. It saturates once the separation reaches the attractor's diameter. So the perturbation
     is renormalised back to d0 at every step, and we never measure across a saturation.
  3. An n-dimensional system has n exponents. `spectrum` returns all three; `lambda_map`
     returns only the largest, the one normally quoted as *the* Lyapunov exponent.

Everything runs in float64. In float32 the round-off at d0 = 1e-8 is a sizeable fraction of
the separation itself and inflates lambda badly.

One useful invariance: the largest exponent is unchanged by any fixed invertible linear
change of coordinates, because the condition number of the change of basis contributes a
bounded factor that vanishes in the long-time average. So a model working in normalised
units and the true system in raw units are directly comparable, with no conversion.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from l63.data import A, lorenz
from l63.known import jacobian

Step = Callable[[Tensor], Tensor]


def true_step(dt: float, n_inner: int = 100, a: tuple = A) -> Step:
    """The true flow map over one time step, as a plain function u -> u.

    Passing this to `lambda_map` gives the truth's lambda through *the same estimator* the
    models are measured with, which is what makes the `chaos` ratio a fair comparison rather
    than a comparison against a literature constant.
    """
    h = dt / n_inner

    def step(u: Tensor) -> Tensor:
        for _ in range(n_inner):
            u = u + lorenz(u, None, a) * h
        return u

    return step


@torch.no_grad()
def lambda_map(step: Step, u0: Tensor, dt: float, n_steps: int = 2000,
               d0: float = 1e-8, warmup: int = 100, seed: int = 0) -> Tensor:
    """Benettin's renormalising estimator for a discrete map. Returns (n_ic,) exponents.

    Two trajectories are started a distance d0 apart, stepped together, and the separation
    is measured and then pulled back to d0 -- every step, so the measurement always happens
    in the linear regime and never across a saturation. lambda is the mean log growth per
    unit time:

        lambda = (1 / (N dt)) sum_n ln( |d_n| / d0 )

    `warmup` steps are renormalised but not accumulated, which lets the perturbation rotate
    onto the most unstable direction first. Without it the estimate is biased low, because
    a random initial direction has only a small component along that direction.

    Works unchanged for a learned map: `step` is any u -> u function.
    """
    u = u0.double().clone()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        v = torch.randn_like(u)
    w = u + v / v.norm(dim=-1, keepdim=True) * d0

    total = torch.zeros(u.shape[0], dtype=torch.float64)
    for n in range(n_steps):
        u, w = step(u), step(w)

        sep = w - u
        d = sep.norm(dim=-1, keepdim=True)
        if n >= warmup:
            total += torch.log(d.squeeze(-1) / d0)

        w = u + sep * (d0 / d)          # back to d0, same direction

    return total / ((n_steps - warmup) * dt)


@torch.no_grad()
def spectrum(u0: Tensor, t_total: float = 200., dt_int: float = 2.5e-4,
             renorm_every: int = 40, transient: float = 20., a: tuple = A
             ) -> tuple[Tensor, Tensor]:
    """All three exponents of the true system, by tangent-space QR. Returns (lams, history).

    Integrate the state together with a matrix Q whose columns span the tangent space, each
    column carried by the linearised flow  dQ/dt = Df(u) Q. The columns all collapse onto
    the most unstable direction, so Q is re-orthonormalised by QR every `renorm_every`
    steps and the diagonal of R records how much each direction stretched in between:

        lambda_i = (1 / T) sum ln |R_ii|

    Expected at Lorenz's parameters: (0.906, 0, -14.57). The check worth doing is not any
    one of them but their **sum**, which must equal div f = -(sigma + 1 + beta) = -13.67 --
    volume contraction is exactly the sum of the exponents, and it is an identity, so any
    disagreement is integration error and nothing else.

    `u0` is a single state (3,). The transient is integrated away first, without accumulating.
    """
    u = u0.double().reshape(1, 3).clone()
    q = torch.eye(3, dtype=torch.float64)

    for _ in range(int(transient / dt_int)):
        u = u + lorenz(u, None, a) * dt_int

    n_steps = int(t_total / dt_int)
    acc = torch.zeros(3, dtype=torch.float64)
    hist = []

    for n in range(n_steps):
        j = jacobian(u, a).double().squeeze(0)
        u = u + lorenz(u, None, a) * dt_int
        q = q + (j @ q) * dt_int

        if (n + 1) % renorm_every == 0:
            q, r = torch.linalg.qr(q)
            d = r.diagonal()
            q = q * torch.sign(d)                 # keep the frame right-handed
            acc = acc + torch.log(d.abs())
            hist.append(acc / ((n + 1) * dt_int))

    return acc / (n_steps * dt_int), torch.stack(hist)
