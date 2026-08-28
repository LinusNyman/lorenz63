"""Measure the conditional width p(u_{n+1} | u_n) against the SDE noise level b.

This sweep selects b, the experiment's independent variable.

A deterministic model trained by MSE learns the conditional mean of the next state. If the
conditional is narrow, that mean is the flow map to within the width, the deterministic model
has no room to fail, and every rung scores the same. The width therefore has to be large
enough to matter, measured as a fraction of the attractor's own scale.

Run:  PYTHONPATH=. python run/width_sweep.py
"""

from __future__ import annotations

import torch

from l63.data import DT, attractor_scale, conditional_width, gen_data, integrate, lorenz, solve_ode

B_VALUES = (0.15, 0.3, 0.6, 1.0, 1.5)
N_ANCHORS = 60


def main() -> None:
    torch.manual_seed(0)

    # anchors: real states on the deterministic attractor
    z = solve_ode(lorenz, torch.randn(N_ANCHORS, 3) * 10., 0., 5., 20_000)[-1]
    _, ode_scale = attractor_scale(solve_ode(lorenz, z[:8], 0., 50., 200_000)[::100])
    print(f'deterministic attractor scale ||sigma|| = {ode_scale:.2f}\n')

    print(f'{"b":>6} {"width":>10} {"% of scale":>12} {"SDE scale":>11} {"|mean z|":>10}  verdict')
    print('-' * 68)

    for b in B_VALUES:
        w = conditional_width(z, DT, b=b, n_samples=400).mean().item()

        # does the attractor survive this much forcing? measured on a real SDE sample
        xs, stats = gen_data(32, 0., 30., 1200, kind='sde', b=b)
        raw = (xs * stats[1] + stats[0]).reshape(-1, 3)
        sde_scale = raw.std(dim=0).norm().item()

        pct = 100 * w / ode_scale
        ok = 'usable' if 8 <= pct <= 40 and sde_scale < 2.5 * ode_scale else (
             'too narrow' if pct < 8 else 'attractor distorted')
        print(f'{b:>6.2f} {w:>10.3f} {pct:>11.1f}% {sde_scale:>11.2f} '
              f'{raw[:, 2].mean().item():>10.2f}  {ok}')

    print('\nWidth scales like b * |u| * sqrt(dt); the check that matters is the last two '
          'columns,\nwhere a b large enough to matter starts to deform the attractor itself.')


if __name__ == '__main__':
    main()
