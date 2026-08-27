"""The standard results for the Lorenz system, each as one function.

These are not our findings. They are what Strogatz, *Nonlinear Dynamics and Chaos* (2nd ed.)
ch. 9 says the system does, and the point of computing them here is that they check the
integrator and the data pipeline before any model exists. If the Lyapunov exponents do not
sum to the divergence, nothing downstream is worth running.

Section numbers below refer to that chapter.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from l63.data import A, lorenz

# The symmetry of the system: (x, y, z) -> (-x, -y, z)   [9.2]
S = torch.diag(torch.tensor([-1., -1., 1.]))


def fixed_points(a: tuple = A) -> tuple[Tensor, Tensor, Tensor]:
    """The origin and the pair C+-.   [9.2]

        C+-  :  x* = y* = +-sqrt(beta (rho - 1)),   z* = rho - 1

    Born out of the origin in a pitchfork bifurcation at rho = 1; at Lorenz's rho = 28 they
    are the left- and right-turning convection rolls, one at the centre of each wing. They
    are *unstable* here, which is why the trajectory keeps switching wings instead of
    settling into one.
    """
    sigma, rho, beta = a
    xy = float(np.sqrt(beta * (rho - 1)))
    origin = torch.zeros(3)
    c_plus = torch.tensor([xy, xy, rho - 1.])
    c_minus = torch.tensor([-xy, -xy, rho - 1.])
    return origin, c_plus, c_minus


def jacobian(u: Tensor, a: tuple = A) -> Tensor:
    """Df at a state. (3,) -> (3, 3), or (n, 3) -> (n, 3, 3).

        [ -sigma    sigma     0   ]
        [ rho - z    -1      -x   ]
        [    y        x    -beta  ]
    """
    sigma, rho, beta = a
    u = u[None] if u.dim() == 1 else u
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    o, i = torch.zeros_like(x), torch.ones_like(x)

    j = torch.stack([
        torch.stack([-sigma * i, sigma * i, o], dim=-1),
        torch.stack([rho - z, -i, -x], dim=-1),
        torch.stack([y, x, -beta * i], dim=-1),
    ], dim=-2)
    return j.squeeze(0) if j.shape[0] == 1 else j


def divergence(a: tuple = A) -> float:
    """div f = -(sigma + 1 + beta), a constant.   [9.2]

    Volume contraction: V(t) = V(0) exp(-(sigma + 1 + beta) t). At Lorenz's parameters the
    exponent is -13.67, so a blob of initial conditions halves its volume every
    ln(2)/13.67 = 0.051 time units. Phase-space volume collapses to zero.

    Two consequences follow: there are no quasiperiodic solutions (an invariant torus would
    enclose a constant volume), and there are no repellers (a repeller is a source of
    volume). Every fixed point is a sink or a saddle, and all trajectories end up in a set
    of zero volume.
    """
    sigma, _, beta = a
    return -(sigma + 1 + beta)


def measured_divergence(u: Tensor, a: tuple = A) -> Tensor:
    """trace(Df) at each state -- should equal `divergence(a)` everywhere, exactly."""
    return jacobian(u, a).diagonal(dim1=-2, dim2=-1).sum(-1)


def rho_hopf(a: tuple = A) -> float:
    """rho_H = sigma (sigma + beta + 3) / (sigma - beta - 1), where C+- lose stability. [9.2]

    Valid while sigma - beta - 1 > 0. At Lorenz's sigma, beta this is 24.74, so rho = 28 sits
    just past it -- he chose the parameters knowing something strange had to happen there.

    The bifurcation is *subcritical*: below rho_H each of C+- is stable but encircled by a
    saddle cycle, which shrinks onto the fixed point as rho -> rho_H and is absorbed by it.
    So above rho_H there is no small stable limit cycle to fall into -- there is nothing
    attracting anywhere nearby.
    """
    sigma, _, beta = a
    assert sigma - beta - 1 > 0, 'the formula needs sigma - beta - 1 > 0'
    return sigma * (sigma + beta + 3) / (sigma - beta - 1)


def stability(a: tuple = A) -> dict[str, np.ndarray]:
    """Eigenvalues of Df at the origin and at C+.   [9.2]

    Expected at rho = 28: the origin is a saddle (one positive, two negative real
    eigenvalues); C+ has one negative real eigenvalue and a complex pair with *positive*
    real part -- an unstable spiral, which is why trajectories spiral outward on a wing
    instead of falling into its centre.
    """
    origin, c_plus, _ = fixed_points(a)
    return {name: np.linalg.eigvals(jacobian(u, a).numpy())
            for name, u in [('origin', origin), ('C+', c_plus)]}


def symmetry_residual(u: Tensor, a: tuple = A) -> float:
    """max |f(Su) - S f(u)|, which is zero: the system is unchanged by (x,y) -> (-x,-y). [9.2]

    Every solution is therefore either symmetric itself or has a symmetric partner. **This
    is why the attractor has two wings**, and why C+ and C- are a pair rather than two
    unrelated points.
    """
    su = u @ S.T
    return (lorenz(su, None, a) - lorenz(u, None, a) @ S.T).abs().max().item()


def kaplan_yorke(lams: Sequence[float]) -> float:
    """D_KY = 2 + (l1 + l2) / |l3| for a 3-D flow with l1 > 0 > l3.   [9.3]

    The dimension at which the sum of the exponents first turns negative, interpolated: the
    largest ball of directions whose volume does not shrink. Strogatz quotes a numerically
    measured attractor dimension of ~2.05; this formula gives ~2.06 from the spectrum alone,
    and the agreement is the check that the spectrum is right.
    """
    l1, l2, l3 = sorted(lams, reverse=True)
    assert l1 > 0 > l3, f'expected l1 > 0 > l3, got {(l1, l2, l3)}'
    return 2 + (l1 + l2) / abs(l3)


def horizon_time(a_tol: float, d0: float, lam: float = 0.906) -> float:
    """t ~ (1/lam) ln(a/d0): how long before an error d0 grows past tolerance a.  [Ex 9.3.1]

    The logarithm is the sting. Strogatz's example: tolerance 1e-3 and initial uncertainty
    1e-7 give 4 ln(10)/lam. Improve the measurement a *millionfold*, to 1e-13, and you get
    10 ln(10)/lam -- a factor 1e6 in measurement buys 2.5x in prediction time.

    This is also why a better one-step fit has sharply diminishing returns for us: the model
    only sets d0. Everything after that is the system's, not the model's.
    """
    return float(np.log(a_tol / d0) / lam)


def horizon_steps(a_tol: float, d0: float, dt: float, lam: float = 0.906) -> float:
    """The same horizon counted in time steps: ln(a/d0) / (lam dt)."""
    return horizon_time(a_tol, d0, lam) / dt


def lorenz_map(z: Tensor, refine: bool = True) -> tuple[Tensor, Tensor]:
    """Successive local maxima of z(t): returns (z_n, z_{n+1}).   [9.4]

    Lorenz's trick for extracting order from chaos. The pairs from a long chaotic time series
    fall almost exactly on a single curve, z_{n+1} = f(z_n) -- a one-dimensional map you can
    iterate. It works only because the attractor is nearly flat (~2-D); it is *not* a
    Poincare map, which would need two coordinates per return.

    `refine` fits a parabola through each peak and its two neighbours and takes the vertex.
    Without it the peak is quantised to the sampling grid and the curve comes out visibly
    thick for a reason that has nothing to do with the dynamics.

    Strogatz's caveat stands: the graph does have some thickness, so f is not strictly a
    well-defined function, and conclusions from it are plausible rather than rigorous.
    """
    z = z.flatten()
    i = torch.nonzero((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])).flatten() + 1

    if refine:
        a_, b_, c_ = z[i - 1], z[i], z[i + 1]
        # vertex of the parabola through three equally spaced points, in units of the offset
        denom = a_ - 2 * b_ + c_
        peaks = torch.where(denom.abs() > 1e-12, b_ - (c_ - a_) ** 2 / (8 * denom), b_)
    else:
        peaks = z[i]

    return peaks[:-1], peaks[1:]


def map_slope(zn: Tensor, zn1: Tensor, n_bins: int = 24, min_count: int = 8
              ) -> tuple[np.ndarray, np.ndarray]:
    """Local slope |f'| of the Lorenz map, by binned linear fit. Returns (centres, slopes).

    The result: |f'| > 1 *everywhere*.   [9.4]

    A fixed point of the map is a closed orbit of the flow; perturbing it gives
    eta_{n+1} ~ f'(z*) eta_n, so |f'| > 1 makes it unstable. The same argument over p
    iterations makes every period-p orbit unstable, since every factor in the product
    exceeds 1. That is Lorenz's answer to the objection that the integration might simply
    have been too short to reveal an enormous period: there are no stable closed orbits to
    find.
    """
    zn_, zn1_ = zn.numpy(), zn1.numpy()
    edges = np.linspace(zn_.min(), zn_.max(), n_bins + 1)
    idx = np.digitize(zn_, edges) - 1

    centres, slopes = [], []
    for k in range(n_bins):
        m = idx == k
        if m.sum() < min_count:
            continue
        centres.append(zn_[m].mean())
        slopes.append(np.polyfit(zn_[m], zn1_[m], 1)[0])

    return np.array(centres), np.array(slopes)
