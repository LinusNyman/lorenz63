"""Every figure that ends up in `figures/`.

Four per model, and the letter always means the same thing:

    a  data           what it was trained on           (topic 00 only -- one per dataset)
    b  loss           training and held-out loss
    c  architecture   the net, as a diagram
    d  error          ||u-hat_n - u_n|| vs time step, with the two integrator references
    f  lorenz_map     successive maxima of z, model over truth

There is no `e`: the four rulers are reported as a table rather than a figure, because a table
can carry the across-seed range and a bar chart of medians cannot. `ruler_figure` below still
draws the bar chart for notebook use.

A figure is named `<topic><letter>_<name>_<dataset>.png`, with no separator between the topic
and the letter, so `04d_error.png` is the lead-time model's error curve. Two topics end in a
letter themselves: `06f_lorenz_map` is topic 06 with letter `f`, while `06fd_error` is topic
06f with letter `d`. No two figures share a filename, but `06f*` matches both topics; match
topics 05 and 06 on the letter that follows, and 05t and 06f on the letter after that.

Rule: red means the model. Grey means a reference. Nothing else is red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from torch import Tensor

from l63 import FIGDIR, ROOT
from l63.data import CONFIG, Data, gen_data, integrate, lorenz, solve_ode
from l63.known import lorenz_map
from l63.train import History

RED = '#d1264b'
GREY = '#5a6472'


def save(fig, path: str | Path, dpi: int = 200) -> Path:
    """Save under `figures/` (bare names) or at `path` (absolute), and say where."""
    path = Path(path)
    if not path.is_absolute():
        path = FIGDIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return path


# ============================================================ a  data

def data_figure(d: Data, path: str | Path | None = None, n_show: int = 10,
                n_demo: int = 5, seed: int = 1):
    """What the training data is: the trajectories, the series, and one step of it.

    Three panels: what was integrated, what was kept (every consecutive pair, no gaps), and
    what sits between two kept states (100 Euler substeps, which buy integrator accuracy
    rather than withholding data).
    """
    mean, std = d.stats

    # gen_data drops the spin-up before returning, so the discarded part has to be generated
    # again to be drawn. fork_rng so this draw cannot shift any other seeded stream.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        demo, _ = gen_data(n_demo, CONFIG['ts'], CONFIG['tf'], CONFIG['n_steps'],
                           kind=d.kind, b=d.b, spinup=0, stats=d.stats)

    # The fine grid behind the first n_show time steps of one real training trajectory: same
    # integrator, same step, restarted from a stored state, so on the ODE it must land back on
    # the stored states exactly; otherwise the zoom panel shows a different trajectory.
    #
    # On the SDE it cannot: restarting draws a fresh noise realisation, so the path between two
    # stored states is one sample of many. That is the property the SDE dataset exists for, so
    # the zoom panel shows the cloud instead of a single path.
    spinup = demo.shape[1] - d.train.shape[1]
    kept = d.train[0, :n_show + 1, 0].numpy()

    fine = solve_ode(lorenz, (d.train[0, 0] * std + mean)[None], 0., n_show * d.dt,
                     n_show * 100)
    fine_u = ((fine[:, 0] - mean) / std).numpy()

    if d.kind == 'ode':
        assert np.abs(fine_u[::100, 0] - kept).max() < 1e-4, \
            'restarting the integrator must land back on the stored states'
        bend = [np.abs(fine_u[i * 100:(i + 1) * 100 + 1, 0]
                       - np.linspace(kept[i], kept[i + 1], 101)).max() for i in range(n_show)]
        zoom = int(np.argmax(bend))
    else:
        zoom = n_show // 2
        # 40 independent one-step realisations from the same stored state
        z_anchor = (d.train[0, zoom] * std + mean)[None].repeat(40, 1)
        cloud = integrate('sde', z_anchor, 0., d.dt, 100, b=d.b)
        cloud_u = ((cloud - mean) / std)[..., 0].numpy()

    fig = plt.figure(figsize=(11.6, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1.25, 1],
                          hspace=0.62, wspace=0.10)
    ax3 = fig.add_subplot(gs[:, 0], projection='3d')
    axA = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1])

    for tr in demo:
        a = tr.numpy()
        ax3.plot(a[:spinup + 1, 0], a[:spinup + 1, 1], a[:spinup + 1, 2], lw=0.7, color='0.78')
        ax3.plot(a[spinup:, 0], a[spinup:, 1], a[spinup:, 2], lw=0.9, alpha=0.9)
        ax3.plot(*a[spinup, :, None], 'o', ms=4, color='k', zorder=5)
    ax3.view_init(elev=20, azim=-62)
    ax3.set_box_aspect((1, 1, 0.85), zoom=1.15)
    ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
    ax3.set_xlabel('$x$', labelpad=-12); ax3.set_ylabel('$y$', labelpad=-12)
    ax3.set_zlabel('$z$', labelpad=-12)
    ax3.set_title(f'{n_demo} trajectories, drawn exactly like the {d.train.shape[0]} '
                  f'training ones', fontsize=10.5, y=1.0)
    ax3.legend(handles=[Line2D([], [], color='0.78', lw=1.2,
                               label=f'{spinup} spin-up steps, discarded'),
                        Line2D([], [], color='C0', lw=1.2,
                               label=f'{d.train.shape[1]} states kept'),
                        Line2D([], [], color='k', marker='o', ls='none', ms=4,
                               label='first state kept')],
               fontsize=8.5, loc='upper left', bbox_to_anchor=(-0.05, 0.94),
               frameon=False, handlelength=1.6, labelspacing=0.3)

    # --- the series: consecutive, complete, and every pair is an example
    n = np.arange(len(fine_u)) / 100.0
    if d.kind == 'ode':
        axA.plot(n, fine_u[:, 0], lw=1.0, color='0.6', zorder=1)
    else:
        axA.plot(np.arange(n_show + 1), kept, lw=1.0, color='0.75', zorder=1)
    axA.plot(np.arange(n_show + 1), kept, 'o', ms=7, mfc='white', mec=RED, mew=1.7, zorder=3)
    lo, hi = min(kept.min(), fine_u[:, 0].min()), max(kept.max(), fine_u[:, 0].max())
    pad = hi - lo
    for i in range(3):
        y = max(kept[i], kept[i + 1]) + 0.15 * pad
        axA.annotate('', xy=(i + 0.93, y), xytext=(i + 0.07, y),
                     arrowprops=dict(arrowstyle='->', color=RED, lw=1.3,
                                     connectionstyle='arc3,rad=-0.35'))
    axA.text(-0.2, hi + 0.18 * pad, r'every consecutive pair is a training example',
             ha='left', va='bottom', color=RED, fontsize=9.5)
    for i in (0, 1, 2, 3):
        axA.text(i, kept[i] - 0.09 * pad, f'$\\mathbf{{u}}_{{{i}}}$',
                 fontsize=9, ha='center', va='top')
    axA.axvspan(zoom, zoom + 1, color='0.88', zorder=0)
    axA.set_ylim(lo - 0.35 * pad, hi + 0.60 * pad)
    axA.set_xlim(-0.45, n_show + 0.45)
    axA.set_xlabel('time step $n$', fontsize=9.5, labelpad=1)
    axA.set_ylabel('$x$ (normalised)', fontsize=9.5)
    axA.tick_params(labelsize=8.5)
    axA.set_title(r'The series is $\mathbf{u}_0, \mathbf{u}_1, \ldots$ — no gaps',
                  fontsize=10.5)
    axA.grid(alpha=0.25)

    # --- and what sits between two of those states
    if d.kind == 'ode':
        sl = slice(zoom * 100, zoom * 100 + 101)
        axB.plot(np.arange(101), fine_u[sl, 0], ls='none', marker='.', ms=2.6, color='0.4')
        axB.plot([0, 100], fine_u[sl, 0][[0, 100]], 'o', ms=7, mfc='white', mec=RED, mew=1.7)
        axB.set_xlabel(f'the 100 Euler substeps inside one time step '
                       f'($\\mathbf{{u}}_{{{zoom}}} \\rightarrow \\mathbf{{u}}_{{{zoom+1}}}$)',
                       fontsize=9.5, labelpad=1)
        axB.set_title('Zoom on the grey band: accuracy, not withheld data', fontsize=10.5)
    else:
        # cloud_u is (substep, realisation): matplotlib draws one line per column
        axB.plot(np.arange(101), cloud_u, lw=0.6, color='0.55', alpha=0.55)
        axB.plot([0], [cloud_u[0, 0]], 'o', ms=7, mfc='white', mec=RED, mew=1.7, zorder=4)
        axB.plot(np.full(cloud_u.shape[1], 100), cloud_u[-1], 'o', ms=4, mfc='none',
                 mec=RED, mew=1.0, alpha=0.8, zorder=4)
        axB.set_xlabel(f'100 Euler–Maruyama substeps inside one time step, '
                       f'40 independent realisations from $\\mathbf{{u}}_{{{zoom}}}$',
                       fontsize=9.5, labelpad=1)
        axB.set_title(r'The SDE has no single path between two states — it has a distribution',
                      fontsize=10.5)
    axB.set_xticks([0, 50, 100]); axB.tick_params(labelsize=8.5)
    axB.set_ylabel('$x$ (normalised)', fontsize=9.5)
    axB.set_facecolor('0.97')
    axB.grid(alpha=0.25)

    fig.subplots_adjust(left=0.0, right=0.985, bottom=0.12, top=0.92)
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ b  loss

def loss_figure(hist: History, path: str | Path | None = None, n_val_traj: int | None = None,
                w: int = 25, ylabel: str = 'one-step MSE   (normalised units)'):
    """Training and held-out loss on one log axis.

    The training curve is one mini-batch, so it is noisy by construction; the running mean is
    the curve to compare against the held-out one. A gap between them would be overfitting.
    With 307 200 training pairs against 900 to 100 000 parameters depending on the rung,
    there is none, and the figure shows it.

    The held-out curve is not comparable across rungs: it is whatever that model's `loss`
    returns, so an MSE, a Gaussian NLL and a flow-matching velocity loss all appear here
    under their own units. It is a within-rung diagnostic only, which is what the `ylabel`
    argument is for. For two of the rungs it is also stochastic (topic 04 draws a fresh lead
    time per call, 06f a fresh base sample and time), so its jitter comes from the loss
    definition rather than from the model.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    it = np.arange(len(hist.train))
    ax.semilogy(it, hist.train, lw=0.7, color='0.78', zorder=1, label='training (mini-batch)')
    if len(hist.train) >= w:
        smooth = np.convolve(hist.train, np.ones(w) / w, mode='valid')
        ax.semilogy(it[w - 1:], smooth, lw=1.4, color='0.35', zorder=2,
                    label=f'training, {w}-iteration mean')
    if hist.val:
        held = f' ({n_val_traj} held-out trajectories)' if n_val_traj else ' (held out)'
        ax.semilogy(it, hist.val, lw=1.4, color=RED, zorder=3, label=f'validation{held}')
        ax.set_title(f'final held-out loss {hist.val[-1]:.2e}', fontsize=11)
    ax.set_xlabel('Adam iteration')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ c  architecture

def arch_figure(spec: list[tuple[str, str]], title: str, n_params: int,
                path: str | Path | None = None, note: str = ''):
    """The net as a row of labelled blocks. `spec` is [(label, shape), ...] from model.spec().

    Every model returns its own spec and this function holds nothing model-specific, so the
    diagrams stay consistent across models.
    """
    n = len(spec)
    fig, ax = plt.subplots(figsize=(1.55 * n + 1.2, 2.5))

    for i, (label, shape) in enumerate(spec):
        first_last = i in (0, n - 1)
        ax.add_patch(plt.Rectangle((i * 1.5, 0), 1.1, 1.0,
                                   facecolor='white' if first_last else '#eef0f3',
                                   edgecolor=GREY if first_last else '#c3c9d2',
                                   lw=1.6 if first_last else 1.2, zorder=2))
        ax.text(i * 1.5 + 0.55, 0.62, label, ha='center', va='center', fontsize=10,
                weight='bold' if first_last else 'normal')
        ax.text(i * 1.5 + 0.55, 0.32, shape, ha='center', va='center', fontsize=9.5,
                color=GREY)
        if i < n - 1:
            ax.annotate('', xy=(i * 1.5 + 1.48, 0.5), xytext=(i * 1.5 + 1.12, 0.5),
                        arrowprops=dict(arrowstyle='->', color=GREY, lw=1.4))

    ax.set_xlim(-0.15, (n - 1) * 1.5 + 1.25)
    ax.set_ylim(-0.55, 1.5)
    ax.axis('off')
    ax.set_title(title, fontsize=11.5, weight='bold', y=0.94)
    ax.text((n - 1) * 0.75 + 0.55, -0.3, f'{n_params:,} parameters' + (f'   ·   {note}' if note else ''),
            ha='center', va='center', fontsize=9.5, color=GREY)
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ d  error vs time step

def error_figure(curve: Tensor, floor: Tensor, bar: Tensor, scale: float, h: dict,
                 dt: float, label: str = 'model', path: str | Path | None = None,
                 n_ic: int = 128):
    """||u-hat_n - u_n|| against time step, with the two explicit-Euler references.

    Three curves:

      red    the model
      grey   one explicit-Euler step of dt per time step, a classical solver of the same
             cost as one model call. The bar the model must clear to be worth using.
      grey   Euler at dt_int against dt_int/2: the ground truth's own uncertainty, and the
             floor below which no difference is resolvable.

    The dotted ceiling is the attractor scale: the distance between two unrelated true
    states, past which a forecast is no better than a guess. `data.attractor_scale` measures
    it from the data.
    """
    n = torch.arange(len(curve))

    # A reference says nothing once it has reached the ceiling: it is a flat line at the
    # same height as every other flat line. Draw each only while it is still informative.
    def upto(c: Tensor) -> tuple[Tensor, Tensor]:
        over = c > scale
        end = int(over.float().argmax()) + 1 if bool(over.any()) else len(c)
        return torch.arange(end), c[:end]

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.plot(*upto(bar), lw=1.2, color=GREY, ls='--',
            label=r'explicit Euler, one step of $\Delta t$ (same cost as the model)')
    ax.plot(*upto(floor), lw=1.2, color=GREY, ls=':',
            label=r'ground truth’s own uncertainty ($\Delta t_{int}$ vs $\Delta t_{int}/2$)')
    ax.plot(n, curve, lw=2.2, color=RED, label=label, zorder=4)
    ax.axhline(scale, ls=(0, (1, 2)), c='k', lw=1.2,
               label=f'{scale:.1f}  attractor scale = no information left')

    for key, colour, txt in [('model_steps', RED, label),
                             ('euler_bar_steps', GREY, 'Euler')]:
        s = h.get(key, -1)
        if s and s > 0:
            ax.plot([s], [scale], 'v', color=colour, ms=8, zorder=5, clip_on=False)
            ax.annotate(f'{s}', xy=(s, scale), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=10, weight='bold', color=colour)

    ax.set_yscale('log')
    ax.set_xlabel(r'time step $n$')
    ax.set_ylabel(r'$\|\hat{\mathbf{u}}_n - \mathbf{u}_n\|$   (Lorenz units)')
    ax.set_title(f'Forecast error against time step   (median over {n_ic} initial conditions)')
    ax.legend(fontsize=8.8, loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.22, which='both')

    sec = ax.secondary_xaxis('top', functions=(lambda v: v * dt, lambda v: v / dt))
    sec.set_xlabel('time units')

    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ e  the four rulers

def ruler_figure(rows: dict[str, dict], path: str | Path | None = None,
                 title: str = 'The four rulers'):
    """Bars for the three ratio rulers, with truth at 1.0 and the horizon printed as text.

    `rows` is {name: {'horizon':int, 'spread':float, 'climate':float, 'chaos':float,
    'alive':float, ...}}. The truth row is drawn first and its value is the line every other
    bar is read against; 1.0 is where truth sits rather than a target.
    """
    keys = [('chaos', r'chaos   $\lambda_1$ ratio'), ('climate', 'climate   $W_1$ ratio'),
            ('spread', r'spread   $\sigma$ ratio')]
    names = list(rows)

    fig, axes = plt.subplots(1, len(keys) + 1,
                             figsize=(3.1 * (len(keys) + 1), 0.46 * len(names) + 1.5),
                             gridspec_kw=dict(wspace=0.55))
    y = np.arange(len(names))[::-1]

    for ax, (k, lab) in zip(axes, keys):
        vals = [rows[nm].get(k, float('nan')) for nm in names]
        cols = ['0.55' if nm.lower().startswith('truth') else RED for nm in names]
        ax.barh(y, [0 if v != v else v for v in vals], color=cols, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(0.02, yi, '  —  not defined' if v != v else f'  {v:.2f}',
                    va='center', fontsize=9, color='k' if v == v else GREY)
        ax.axvline(1.0, color='k', lw=1.1, ls=':')
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_title(lab, fontsize=10.5)
        ax.set_xlim(0, max(1.35, *[v for v in vals if v == v] or [1.35]) * 1.15)
        ax.set_ylim(-0.65, len(names) - 0.35)
        ax.grid(axis='x', alpha=0.2)

    ax = axes[-1]
    ax.axis('off')
    ax.set_title('horizon   (time steps)', fontsize=10.5)
    for yi, nm in zip(y, names):
        r = rows[nm]
        ax.text(0.0, yi, f'{r.get("horizon", 0):>5d}', fontsize=11, weight='bold',
                va='center', family='monospace')
        ax.text(0.35, yi, f'alive {r.get("alive", float("nan")):.2f}', fontsize=9,
                va='center', color=GREY)
    ax.set_ylim(-0.65, len(names) - 0.35)
    ax.set_xlim(0, 1)

    fig.suptitle(title, fontsize=12.5, weight='bold', y=1.02)
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ f  the Lorenz map

def lorenz_map_figure(model_states: Tensor, truth_states: Tensor,
                      path: str | Path | None = None, label: str = 'model'):
    """Successive maxima of z: z_{n+1} against z_n, model over truth.   [Strogatz 9.4]

    Shows whether the dynamics were learned. Truth's pairs fall on one thin curve with
    |f'| > 1 everywhere, which is what rules out stable closed orbits. A model that has found
    a limit cycle collapses onto a few points; one that has smoothed the dynamics produces a
    fatter, flatter curve. Neither failure is visible in a marginal.
    """
    # Per trajectory, never across the concatenation: a peak finder run over stacked
    # trajectories invents a maximum at every join, and those show up as scatter off the
    # curve that looks like the system rather than like a bookkeeping mistake.
    def pairs(s: Tensor) -> tuple[Tensor, Tensor]:
        s = s if s.dim() == 3 else s[None]
        out = [lorenz_map(tr[:, 2]) for tr in s]
        return torch.cat([a for a, _ in out]), torch.cat([b for _, b in out])

    tz, mz = pairs(truth_states), pairs(model_states)

    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    ax.plot(tz[0], tz[1], '.', ms=2.2, color='0.62', label=f'truth ({len(tz[0])} maxima)')
    ax.plot(mz[0], mz[1], '.', ms=2.2, color=RED, alpha=0.75,
            label=f'{label} ({len(mz[0])} maxima)')

    lo = float(min(tz[0].min(), tz[1].min())) - 1
    hi = float(max(tz[0].max(), tz[1].max())) + 1
    ax.plot([lo, hi], [lo, hi], lw=0.9, color='k', ls=':', label='$z_{n+1}=z_n$')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('$z_n$   ($n$-th local maximum of $z$)')
    ax.set_ylabel('$z_{n+1}$')
    ax.set_title('The Lorenz map', fontsize=11.5)
    ax.legend(fontsize=8.8, loc='upper left')
    ax.grid(alpha=0.22)
    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig


# ============================================================ 07  attractor grid

def attractor_grid(states: dict[str, Tensor], path: str | Path | None = None,
                   n_show: int = 6000):
    """Truth beside every model, x against z. Covers what `climate` does not measure.

    `climate` is marginals-only: a model can match all three marginals and still put the mass
    in the wrong shape. This figure shows the shape, and carries no number.
    """
    names = list(states)
    fig, axes = plt.subplots(1, len(names), figsize=(2.5 * len(names), 2.9), sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, nm in zip(axes, names):
        # One line per trajectory, never across the concatenation. Truth arrives as 128
        # separate 901-step runs; drawing them as one array joins the end of each to the
        # start of the next and paints a straight line across the attractor for every join.
        s = states[nm]
        s = s if s.dim() == 3 else s[None]
        left = n_show
        for tr in s:
            if left <= 0:
                break
            piece = tr[:left]
            ax.plot(piece[:, 0], piece[:, 2], lw=0.35,
                    color='0.5' if nm.lower().startswith('truth') else RED, alpha=0.85)
            left -= len(piece)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel('$x$', fontsize=9.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('$z$', fontsize=9.5)

    fig.tight_layout()
    if path is not None:
        save(fig, path)
    return fig
