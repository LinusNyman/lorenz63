"""The poster's figures, re-rendered for A0.

The standing figures in `figures/` are styled for a 16:9 slide. Placed on an A0 poster their
tick labels land near 6 pt, against an 18 pt floor, and their 200 dpi rasters land near
100 dpi once scaled to the page width. Both are fixed here rather than in `l63/plots.py`,
because both are properties of the sheet and not of the result:

  * type is set from the placed width. A figure drawn 432 mm wide and placed 751 mm wide is
    scaled 1.74x, so an 11 pt label reads as 19 pt. Every size below follows from that
    arithmetic and the placed widths in `poster_final.typ`; changing one requires changing
    the other.
  * output is PDF. Vector has no dpi, so print size cannot degrade it.

Five figures are written; three are placed on the sheet. None exists anywhere else:

  p_hook         the opening figure: the butterfly in grey with one MLP rollout in red over
                 it, from the same start. Red on grey while they agree, red alone once they
                 do not. The window is 223 steps and is load-bearing; see `hook`.
  p_system       the Lorenz distance between two true trajectories 0.01 apart, median over
                 128 pairs, log axis, on both ground truths: the ODE needs 483 steps to reach
                 attractor scale and the SDE 23, and the slope of the straight stretch is
                 lambda_1. Written but not placed. Nothing on the sheet shows sensitive
                 dependence directly; placing this again is one line in poster_final.typ.
  p_conditional  the two ground truths as a picture rather than a table: one time step taken
                 500 times from the same state. A point on the ODE, a cloud on the SDE, and
                 the width of that cloud is the experiment's control parameter.
  p_attractors   truth beside every model, both datasets. Written but not placed. It is the
                 only view of joint structure on the SDE, so it is kept for reference.
  p_scorecard    §4. The four rulers as dots with the five-seed range drawn through them.
                 It replaces two 30-cell tables. The range matters because several of these
                 columns are bimodal, and a median without it invites comparisons the data
                 does not support.

Everything numeric comes from the same checkpoints, frozen datasets and
`artifacts/summary.json` as the rest of the repository. This script chooses sizes; it computes
nothing that `run/report.py` has not already computed, except the rollouts it draws.

Run:  PYTHONPATH=. ./.venv/bin/python poster/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import torch

from l63 import ARTIFACTS, plots as P
from l63.data import integrate, make_datasets
from run.report import load_model

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'

# The five models on the poster, pinned here rather than imported from `run/report.py`, whose
# REPRESENTATIVE carries seven: adding a rung upstream must not silently add a panel to a
# figure laid out for five.
# Keys are row ids -- the sheet's five slots, in order -- and values are the `topic_label`
# stem of the row's key in summary.json. The stem is written out rather than rebuilt from the
# row id, because in the AR set below two rows are both topic 03 (the MLP at two k).
MODELS = {'02': '02_mlp', '03': '03_rollout_k8', '04': '04_leadtime',
          '05': '05_lstm', '06': '06_gaussian'}

# Short names. The poster names each model once in its architecture block; a figure repeats
# the name so that a reader starting from the figure can find that block. Row 03 is spelled
# out rather than left as "rollout": the label has to say it is the same network as row 02,
# because that pair is the comparison.
SHORT = {'02': 'MLP, one step', '03': 'MLP, rollout $k{=}8$', '04': 'lead time',
         '05': 'LSTM', '06': 'Gaussian'}

# --- the AR variant, selected with `--ar` -------------------------------------------------
# The same five rows, with every deterministic model except the baseline trained through the
# composed map instead of on true states only. It draws into a separate `_ar` figure bank and
# is not what `poster_final.typ` places; it is kept because the AR checkpoints exist and the
# comparison is worth being able to redraw.
#
# Row 02 stays the one-step MLP and row 03 is the same network at k = 4, so the pair remains a
# measurement of what training through the composed map does. Replacing the baseline instead
# of pairing with it would leave nothing for the other rows to improve on.
#
# Only one k is drawn. Two rows of the same network at two unroll depths is a k-response that
# a poster has no room to explain; the sweep is named in the card's subtitle instead. Switching
# this to k = 8 requires re-picking `HOOK_IC_AR`, which is a property of one model's rollouts.
#
# Row 06 is unchanged: the Gaussian is the only model here whose rollout is stochastic, so
# unrolling its loss would have to score a distribution. An MSE through a sampled chain drives
# log-sigma to the floor and silently makes the model deterministic; an AR-NLL makes a one-step
# sigma head carry k-step uncertainty and then injects it at every step. Per-step training is
# the standard choice for a probabilistic forecaster (GenCast).
MODELS_AR = {'02': '02_mlp',       '03': '03_rollout_k4', '04': '04_leadtime_ar4',
             '05': '05_lstm_ar4',  '06': '06_gaussian'}
SHORT_AR = {'02': 'MLP, one step', '03': 'MLP, AR $k{=}4$', '04': 'lead time, AR',
            '05': 'LSTM, AR', '06': 'Gaussian (teacher-forced)'}

SUFFIX = ''        # '_ar' in AR mode, appended by `bank`, so nothing overwrites the original

# The initial condition `hook` draws, and the row it draws it from. Both are properties of the
# model rather than of the sheet, so both move with MODELS -- see the warnings in `hook`.
#
# Both banks draw the one-step MLP at IC 47, so `p_hook` and `p_hook_ar` are the same figure:
# the opening question is whether the butterfly effect can be predicted, and the model to show
# failing at it is the baseline the rest of the sheet is read against. IC 47 is the 54th
# percentile of that model's 128 starts. The IC moves with the model and with the window, so
# re-pick it with `pick_ic.py` when either changes rather than carrying one across.
HOOK_IC, HOOK_IC_AR = 47, 47
HOOK_TOPIC, HOOK_TOPIC_AR = '02', '02'

KINDS = ('ode', 'sde')
TITLE = {'ode': 'deterministic ODE,  $b = 0$', 'sde': 'stochastic SDE,  $b = 0.6$'}
ROW = {'ode': 'ODE', 'sde': 'SDE, $b\\,{=}\\,0.6$'}

SUMMARY = json.load(open(ARTIFACTS / 'summary.json'))
GT = SUMMARY['ground_truth']
DT_ODE = GT['datasets']['ode']['dt']

# Drawn width in inches, and the SMALLEST type each figure is allowed. Both are chosen from
# the width the figure is placed at in poster.typ, because that is what sets the scale factor
# and therefore what the reader actually reads:
#
#   figure        drawn    placed   scale   floor for 18 pt
#   hook          218 mm   365 mm   1.69x     10.7 pt   (wide panel, places 235 mm tall)
#   conditional   218 mm   365 mm   1.69x     10.7 pt   (places 170 mm tall)
#   system        144 mm     --     --         --      (written, not placed)
#   attractors    267 mm   494 mm   1.84x      9.8 pt   (written, not placed)
#   scorecard     440 mm   751 mm   1.71x     10.6 pt
#
# The how-to deck's floor is 18 pt inside a figure and NOTHING here is under it -- which is
# why the small text below is 10.6 pt rather than the 8 pt a screen figure would use. If a
# figure is ever placed at a different width, these have to move with it.
W3, W2, W1 = 17.0, 10.5, 5.6
# The hook has a width of its own: it is placed at HALF the content width (365 mm), which is
# neither the 237 mm one-column nor the 494 mm two-column span the others use. Drawn at 8.6 in
# it scales 1.69x, so its 11 pt type reads as 18.6 pt -- in line with the rest of the sheet.
# Drawn at W1 instead it would scale 2.57x and its labels would read at 28 pt, which is not a
# rule violation but would make the opening figure's furniture the largest type in any figure.
WH = 8.6

# The hook's outermost x tick. The butterfly reaches about -18 to +16, so 20 is the first round
# number that covers it; the frame goes two units wider still (see `hook`).
X_TICK = 20.0
F3, F2, F1 = 11.5, 10.5, 11.0
SMALL3, SMALL2, SMALL1 = 10.6, 10.0, 11.0


def bank(fig, name: str) -> None:
    for ext in ('pdf', 'png'):
        P.save(fig, OUT / f'{name}{SUFFIX}.{ext}', dpi=200)
    plt.close(fig)


def data(kind: str):
    return make_datasets(seed=0, kind=kind, b=GT['datasets'][kind]['b'])


def entry(topic: str, kind: str) -> dict:
    return SUMMARY['models'][f'{MODELS[topic]}_{kind}']


def rep(topic: str, kind: str) -> int:
    """The seed a figure should draw: the one whose horizon is nearest the median.

    `run/report.py` picks it, records it as `rep_seed`, and draws its own figures from it, and
    the numbers it puts in `horizon_detail` are that seed's. Drawing seed 0 here instead was
    fine while every rep_seed happened to be 0; after the rescore five of the ten are not, and
    a figure that plots one seed's curve under another seed's horizon marker is simply wrong.
    """
    return entry(topic, kind)['rep_seed']


# ============================================================ the hook

def hook(steps: int = 223, ic: int | None = None, topic: str | None = None,
         kind: str = 'ode') -> None:
    """The poster's opening panel: the butterfly, with one forecast drawn over it.

    Grey is a true trajectory in the x-z plane, the Lorenz attractor itself. Red is one MLP
    rollout from the same initial state, layered on top. For most of the window the red is
    hidden exactly under the grey; then it takes a loop the truth does not take, and from
    there the two lie on the same attractor along different paths -- the shape is learnable
    and the trajectory is not.

    `steps` is the parameter that matters, and it is chosen in Lyapunov times: 223 steps is
    5.00 tau at dt = 0.025 and lambda_1 = 0.897. Stating the window in tau makes it comparable
    to results outside this project, and it is the unit §1 prints beside the step convention.

    The window is a compromise. The butterfly needs enough steps to be drawn at all, and the
    divergence needs few enough that the two curves have not both covered the whole attractor.
    Past about 15 tau each curve traces the full shape and no disagreement is visible; the
    picture then says only that the model learned the attractor. At 5 tau the shape is drawn
    about seven times round and the forecast is 79 steps past its own crossing.

    The window and the IC have to be chosen together. Two of the filters that pick the IC --
    the lobe-change disagreement and the gap at the last drawn step -- are measured inside the
    window and do not survive a change to it. At 9.0 tau this figure used IC 113; at 5.0 tau
    that same IC has the two curves only 4.3 apart at the end, because they happen to be near
    each other again there, and the picture would read as a recovered forecast.

    No step count is printed on this figure. Any number here is either a different statistic
    from §4's horizon or an unrepresentative single draw, and both misread on an opening
    figure; §4 carries the measurement with its seed range.

    Both figure banks draw the one-step MLP (`HOOK_TOPIC_AR` is '02'), so `p_hook` and
    `p_hook_ar` are the same figure. The opening question is whether the butterfly effect can
    be predicted, and the model to show failing at it is the baseline the rest of the sheet is
    read against.

    IC 47 is typical, which is the point of the pick: its error reaches the attractor scale at
    156 steps against this model's median of 148 over the 128 evaluation starts -- the 54th
    percentile, where the crossing runs from 35 to 901 and 8 never cross inside 900 steps. An
    above-median rollout would flatter a model that already looks strong on this sheet.

    Legibility is a second filter and is not automatic. The truth makes 5 lobe changes inside
    the 223-step window and the forecast makes 4, so the red covers the butterfly while taking
    its own path, and the two are 24.0 apart at the last drawn step, past the attractor's own
    14.0. An IC that makes the same lobe changes as the truth keeps the red under the grey for
    nearly the whole window and shows only agreement. IC 47 is also not captured: its
    20 000-step rollout has a spread of 14.6 over its final 5000 steps against truth's ~14.4.

    Re-pick from the model rather than copying an IC when row 02 or the window changes.
    `pick_ic.py` applies the filter and takes the window as its argument: percentile 35-65,
    crossing inside the window, both sides changing wing at least 4 times, a lobe-count
    disagreement of at least 1, parted by more than 0.6 of the attractor scale at the last
    drawn step, and alive at 20 000 steps. One of the 128 starts passed all six at 223.

    156 is not §4's 336 and the two are not comparable: `evaluate.horizon` crosses the
    attractor scale with the median error curve over 128 ICs (`median_curve`), not with a
    single rollout, and the median curve crosses much later than the median single rollout
    does (336 against 148).

    Drawn for a half-content-width placement (365 mm), so it uses `WH` rather than sharing
    `W1` with the one-column figures, and places 235 mm tall. The panel is not equal-aspect
    and the butterfly is stretched about 1.8x in x; the frame block below gives the reasoning
    and the rejected alternatives.
    """
    ic = HOOK_IC if ic is None else ic
    topic = HOOK_TOPIC if topic is None else topic
    d = data(kind)
    name = f'{MODELS[topic]}_{kind}_s{rep(topic, kind)}'
    m, _ = load_model(name)
    print(f'  {name}   ic {ic}')

    with torch.no_grad():
        pred = d.raw(m.forecast(d.eval[ic:ic + 1, :m.history], steps))[0]
    truth = d.raw(d.eval[ic, :steps + 1])

    # This panel is not equal-aspect: a unit of x is about 1.8x longer on paper than a unit of
    # z, so the butterfly is drawn wider and flatter than it is. The panel is 365 mm wide and
    # 235 mm tall (1.56 : 1) while the butterfly spans about 35 in x against 43 in z, so it is
    # taller than wide. Holding a unit of x equal to a unit of z forces one of:
    #   * `adjustable='datalim'`, which widens the x limits to about +-37 to fill the panel:
    #     correct in shape, but two thirds of the frame is empty and the axis is ticked at
    #     +-30 where no trajectory goes;
    #   * a square panel about 224 mm wide, which is not the half content width of the layout;
    #   * a taller panel, about 390 mm, which §1 has no page height for.
    # The x limits are pinned to the data instead and the stretch is accepted. That trades the
    # butterfly's proportions for an axis whose range is the range of the data, and it should
    # not be copied into a figure whose claim is the shape.
    #
    # The z limits come from both curves. Framing on the truth alone would clip the model's
    # excursion, which is the part of the figure the reader is meant to notice.
    zlo = float(min(truth[:, 2].min(), pred[:, 2].min()))
    zhi = float(max(truth[:, 2].max(), pred[:, 2].max()))
    zpad = 0.08 * (zhi - zlo)

    # X TICKS AT +-20, X LIMITS AT +-22. The limits are deliberately two units wider than the
    # outermost tick so that the "-20" and "20" labels sit inside the frame with air around
    # them, the way the z labels do: z's ticks land on round decades (10 to 40) inside limits of
    # roughly 3 to 48, so they were never near a corner and x should not be either.
    #
    # Do not set the limits to +-20 to make the numbers match. That puts both outer labels in
    # the bottom corners, hard against the axis ends, which is what this fixes.
    frame = [(-X_TICK - 2.0, X_TICK + 2.0), (zlo - zpad, zhi + zpad)]

    with plt.rc_context({'font.size': F1, 'axes.titlesize': F1, 'axes.labelsize': F1,
                         'xtick.labelsize': SMALL1, 'ytick.labelsize': SMALL1}):
        # Square axes plus room for the axis furniture underneath and to the left of it;
        # `set_aspect('equal')` sizes the box and tight_layout trims what is left over.
        # The drawn aspect IS the placed aspect: 365 mm wide by 235 mm tall on the sheet, so
        # 8.6 in by 8.6 * 235/365 = 5.53. Everything else follows from `adjustable='datalim'`.
        fig, ax = plt.subplots(figsize=(WH, 5.53))

        # Truth thick and underneath, forecast thinner on top, so that "they are the same curve"
        # is visible as a grey halo around the red rather than as one colour hiding the other.
        ax.plot(truth[:, 0], truth[:, 2], lw=2.4, color='0.66', zorder=2, solid_capstyle='round')
        ax.plot(pred[:, 0], pred[:, 2], lw=1.3, color=P.RED, zorder=3, solid_capstyle='round')
        ax.plot([truth[0, 0]], [truth[0, 2]], 'o', ms=7.0, mfc='k', mec='white', mew=1.2,
                zorder=6)
        ax.annotate('same start', xy=(float(truth[0, 0]), float(truth[0, 2])),
                    xytext=(9, -2), textcoords='offset points', ha='left', va='top',
                    fontsize=SMALL1, color='0.25', zorder=6,
                    bbox=dict(fc='white', ec='none', pad=1.0))

        # A legend and not labels on the curves: for most of the window the two ARE the same
        # curve, so there is nowhere on the red that is not also the grey.
        ax.legend(handles=[Line2D([], [], color='0.66', lw=2.4, label='truth'),
                           Line2D([], [], color=P.RED, lw=1.8,
                                  label=f'{SHORT[topic]} forecast')],
                  loc='upper center', frameon=False, handlelength=1.5, handletextpad=0.6,
                  borderpad=0.1, labelspacing=0.35, fontsize=F1 + 0.5)

        # No `set_aspect`: the axes fill the panel and the data is stretched to fit it. See the
        # warning above the frame block for why, and for what that costs.
        ax.set_xlim(*frame[0])
        ax.set_ylim(*frame[1])
        ax.set_xlabel('$x$', labelpad=1)
        ax.set_ylabel('$z$', labelpad=1)
        # Ticks written out rather than left to MaxNLocator, which picked multiples of 8 and
        # printed -16 and 16 as the outermost labels on an axis that runs to +-20. The point of
        # pinning the limits was an axis whose range is the range of the data; the labels have
        # to say so.
        ax.set_xticks([-X_TICK, -X_TICK / 2, 0, X_TICK / 2, X_TICK])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3.5, pad=2)

        fig.tight_layout(pad=0.4)
        bank(fig, 'p_hook')


# ============================================================ the system itself

def system(steps: int = 600, delta: float = 1e-2, n_ic: int = 128, seed: int = 0) -> None:
    """How fast two true trajectories 0.01 apart come apart, on each ground truth.

    Plots the Lorenz distance ||u_1 - u_2|| against time step on a log axis, against the
    attractor scale. The norm rather than a single coordinate: x is the slowest of the three
    to show the divergence. The first step at which each gap passes 1.0 is 380 for the full
    state, 386 for y, 382 for z, and 449 for x, because after separation x carries only 0.27
    of the squared gap against y's 0.37 and z's 0.36, so a growing error hides inside x's own
    swings for about 70 steps longer. The norm also removes the choice of coordinate.

    Median over 128 pairs, which is what makes the exponential clean. A single pair does not:
    the eval set's IC 0 spends its first 400 steps spiralling near C-, where local expansion
    is weak, so its gap contracts from 0.010 to 0.005 before it grows and its average rate
    reads 0.58 rather than 0.897. Over 128 pairs the median fits 0.851 per time unit against a
    measured lambda_1 of 0.897, within 5 %, so the straight stretch on the log axis is the
    Lyapunov exponent and the reference line through it is justified. Drawing that line
    against one pair is not.

    The ODE median needs 483 steps to reach the attractor scale; the SDE median needs 23. On
    the SDE the 0.01 head start is irrelevant: the two trajectories draw different noise from
    the first step, so they are unrelated almost immediately and the curve starts at its
    ceiling rather than climbing to it.

    No red in this figure. Both curves are truth, and red marks a model everywhere else.
    """
    lam = GT['lambda_true']
    tone = {'ode': '#2b3440', 'sde': '#8a94a3'}
    label = {'ode': 'ODE,  $b = 0$', 'sde': 'SDE,  $b = 0.6$'}
    med, scales = {}, {}

    for kind in KINDS:
        b = GT['datasets'][kind]['b']
        d = data(kind)
        scales[kind] = GT['datasets'][kind]['attractor_scale']
        u0 = d.raw(d.eval[:n_ic, 0])
        pair = torch.cat([u0, u0 + torch.tensor([delta, 0.0, 0.0])])
        # The SDE draws fresh noise per trajectory, so this has to be seeded or the figure
        # moves every run -- same reason p_attractors seeds the Gaussian.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            tr = integrate(kind, pair, 0.0, steps * d.dt, steps * 100, b=b)[::100]
        med[kind] = (tr[:, n_ic:] - tr[:, :n_ic]).norm(dim=-1).quantile(0.5, dim=1)

    with plt.rc_context({'font.size': F1, 'axes.titlesize': F1, 'axes.labelsize': F1,
                         'xtick.labelsize': SMALL1, 'ytick.labelsize': SMALL1}):
        fig, ax = plt.subplots(figsize=(W1, 2.75))
        n = torch.arange(steps + 1)

        for kind in KINDS:
            ax.plot(n, med[kind], lw=2.3, color=tone[kind], zorder=4, solid_capstyle='round')
            # Each dataset saturates at ITS OWN attractor scale, and the two differ by 12 %
            # (14.0 against 15.8). One shared ceiling would put each curve against the other's.
            ax.axhline(scales[kind], ls=(0, (1, 2.6)), lw=1.2, color=tone[kind], zorder=2)
            c = int((med[kind] > scales[kind]).float().argmax())
            ax.plot([c], [scales[kind]], 'v', ms=7, color=tone[kind], zorder=6, clip_on=False)
            ax.annotate(f'{c}', xy=(c, scales[kind]), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=SMALL1,
                        weight='bold', color=tone[kind], zorder=6)

        # The slope IS lambda_1. Anchored on the ODE median inside the stretch that was fitted
        # (steps 100-350), and drawn only across it, so it is not extrapolated into the initial
        # transient or into saturation, where it would not hold.
        lo, hi, anch = 90, 360, 150
        t = torch.arange(lo, hi + 1)
        ref = float(med['ode'][anch]) * torch.exp(lam * (t - anch) * DT_ODE)
        ax.plot(t, ref, lw=1.3, ls='--', color='0.45', zorder=3)
        # Labelled below the line, not at its end: at its end it sits between the two dotted
        # ceilings and reads as belonging to one of them.
        ax.annotate(r'slope $= \lambda_1$', xy=(300, float(ref[300 - lo])),
                    xytext=(9, -6), textcoords='offset points', ha='left', va='top',
                    fontsize=SMALL1, color='0.35',
                    bbox=dict(fc='white', ec='none', pad=1.0), zorder=6)

        # The SDE label starts at 0.30 and not at the left edge: its crossing marker sits at
        # step 23, which is the left edge, and the two collided there.
        for kind, xy, va in (('sde', (0.30, 0.97), 'top'), ('ode', (0.56, 0.30), 'center')):
            ax.annotate(label[kind], xy=xy, xycoords='axes fraction', ha='left', va=va,
                        fontsize=F1, weight='bold', color=tone[kind], zorder=6,
                        bbox=dict(fc='white', ec='none', pad=1.2))

        ax.set_yscale('log')
        ax.set_xlim(0, steps)
        ax.set_ylim(4e-3, 90)
        ax.set_xlabel('time step $n$', labelpad=1)
        ax.set_ylabel(r'$\|\mathbf{u}_1 - \mathbf{u}_2\|$', labelpad=1)
        ax.grid(alpha=0.20, lw=0.6, which='major')
        ax.set_axisbelow(True)
        ax.tick_params(length=3.5, pad=2)

        fig.tight_layout(pad=0.4)
        bank(fig, 'p_system')


# ============================================================ the two ground truths

def _spread_anchors(states, k: int):
    """`k` indices into `states`, spread across the attractor and away from its extremes.

    The state whose x is nearest each of k interior quantiles of x: 1/6, 1/2, 5/6 for k = 3.
    That puts one anchor inside each wing and one near the crossing. Deterministic, no tuning.

    TWO REJECTED ALTERNATIVES, both of which were tried and produce a worse figure:

      * evenly spaced in TIME (the original). Even spacing along the trajectory puts anchors
        wherever the state happens to be, and at three anchors on the SDE all three landed on
        the same wing, where the clouds merged into a single blob.
      * farthest-point sampling. It separates them perfectly and picks the WING TIPS, which is
        exactly where the flow is fastest and the conditional widest -- the 27.7 %-of-scale end
        of a distribution whose other end is 9.0 %. Three extremal anchors would make the SDE
        cloud look uniformly huge, and they pushed the shared axes out far enough to shrink the
        ODE butterfly beside them.

    Quantiles of x rather than of the trajectory index for the same reason as the first bullet:
    the figure is drawn in x-z, so anchors have to be spread in the plane a reader is looking at.
    """
    x = states[:, 0]
    targets = torch.quantile(x, torch.linspace(0, 1, 2 * k + 1)[1::2])
    return sorted(int((x - t).abs().argmin()) for t in targets)


def conditional(n_samples: int = 500, n_anchor: int = 3, n_show: int = 800,
                seed: int = 3) -> None:
    """One time step, taken 500 times from the same state. A point, then a cloud.

    Three anchors and 800 steps of trajectory rather than seven anchors and the full 2301: at
    seven the panel is a dense grey tangle with red marks competing for the same space, and at
    A0 viewing distance each anchor has to be separable as a dot on the left and a smear on
    the right.

    The grey path is drawn at substep resolution (keep_every=1). Drawn from every 100th
    substep -- the stored dt the models work at -- the SDE renders as a polyline with long
    straight edges. Those edges are an artifact of the display sampling, not the geometry: the
    path is continuous and rough because of the multiplicative noise, which is what all 100
    substeps per stored step show. The models still see only the stored states; this changes
    the rendering alone. The display trajectory is integrated fresh from the frozen reference's
    first state, seeded so the figure is identical every run, and the anchors are picked from
    its every-100th samples so the red clouds sit on the drawn curve.

    What it shows, measured:
      * ODE. The spread is 0.000 in all three coordinates at every anchor -- exactly zero, not
        small, because 500 deterministic integrations from one state give one state. Each
        anchor is a point, which is the panel's content.
      * SDE. The clouds are anisotropic, sd(z) several times sd(x), most visibly at the
        wing-crossing anchor, which renders as a vertical smear. That is the
        state-multiplicative diffusion acting on z, the largest-magnitude coordinate.

    The width here is not the single 16.5 % figure quoted for the dataset: per-anchor rms
    radius runs 9.0 % to 27.7 % of attractor scale, mean about 17.5 %, because the
    conditional's width depends on position on the attractor. Three anchors make that
    variation less visible, not more.

    The figure also does not separate noise from motion. At some anchors one step moves the
    state further than the cloud is wide (11.6 against a radius of 4.4) and at others the
    reverse (0.71 against 2.2); both are drawn identically.

    Anchors are chosen by `_spread_anchors`, deterministically, so the picture is the same every
    run. They are NOT evenly spaced along the trajectory any more: even spacing in TIME puts
    them wherever the trajectory happens to be, and on the SDE at three anchors that landed all
    three on the same wing, where the clouds merged into one blob and the panel said nothing.
    """
    with plt.rc_context({'font.size': F1, 'axes.titlesize': F1 + 1,
                         'axes.labelsize': F1, 'xtick.labelsize': SMALL1,
                         'ytick.labelsize': SMALL1}):
        # Drawn at WH, like the hook, because it is placed at the same HALF CONTENT WIDTH
        # (365.5 mm) directly under it. At W1 it would scale 2.57x and its labels would read at
        # 28 pt; at WH it scales 1.69x and reads 18.6 pt, in line with the rest of the sheet.
        #
        # ⚠️ THESE PANELS ARE NOT EQUAL-ASPECT, and the butterfly above them is. A unit of x is
        # about 1.4x longer than a unit of z here, so both attractors are drawn wider and
        # flatter than they really are. That is deliberate and it is a HEIGHT decision, not a
        # design one: equal aspect at 365 mm wide needs about 215 mm of page against the 170 mm
        # this uses, and §1 does not have the 45 mm. It is tolerable because this figure makes a
        # WIDTH-OF-CONDITIONAL claim, not a shape claim -- the shape claim is the hook's, and
        # the hook is correctly proportioned. Do not copy this compromise into a shape figure.
        fig, axes = plt.subplots(1, 2, figsize=(WH, 4.0), sharex=True, sharey=True)

        for ax, kind in zip(axes, KINDS):
            b = GT['datasets'][kind]['b']
            d = data(kind)
            # Substep resolution for the DISPLAY path -- see the docstring. n_show stored
            # steps, all 100 substeps of each kept, from the frozen ref's first state.
            z0 = d.raw(d.ref)[0][None]
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                fine = integrate(kind, z0, 0.0, n_show * d.dt, n_show * 100, b=b,
                                 keep_every=1)[:, 0]
            coarse = fine[::100]
            ax.plot(fine[:, 0], fine[:, 2], lw=0.45, color='0.78', zorder=1)

            idx = _spread_anchors(coarse, n_anchor)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                for a in coarse[idx]:
                    nxt = integrate(kind, a[None].repeat(n_samples, 1), 0.0, d.dt, 100, b=b)[-1]
                    ax.plot(nxt[:, 0], nxt[:, 2], '.', ms=4.0, color=P.RED, alpha=0.35,
                            zorder=3, mec='none')

            ax.set_title(f'{"ODE" if kind == "ode" else "SDE"},  $b = {b:g}$', pad=6)
            ax.set_xlabel('$x$', labelpad=0)
            ax.tick_params(length=3, pad=2)
        axes[0].set_ylabel('$z$', labelpad=0)

        fig.tight_layout(pad=0.4, w_pad=1.0)
        bank(fig, 'p_conditional')
# ============================================================ rulers 2-4, in picture form

def attractors(n_show: int = 6000, n_ic: int = 4, steps: int = 8000) -> None:
    """Truth beside every model, x against z, both datasets in one figure.

    Two rows, one per ground truth, sharing axes within a row: the comparison the poster is
    making is between the rows, and a shared scale is what makes that comparison legible at 2 m.

    A previous version put each model's forecast-error curve in a row under its attractor. It
    was removed: two of the four rows were four flat lines at a ceiling, and `horizon` is
    already in the scorecard with its five-seed range, which a single curve cannot show.

    DRAWN FOR A TWO-COLUMN PLACEMENT, not the full width. Square panels and full width are
    incompatible at this row count: six panels across 751 mm are 120 mm wide each, so two
    square rows cannot be shorter than about 330 mm. Placed at 494 mm the same figure is
    218 mm, which is the height of the scorecard beside it. The type is sized for 494 mm --
    move it and the sizes have to move too -- and each axis is held to three ticks, because
    five tick labels do not fit a 41 mm panel.
    """
    cols = ['truth'] + [SHORT[t] for t in MODELS]

    # SQUARE, NOT STRETCHED, AND BOTH ROWS THE SAME SIZE. A butterfly drawn with a unit of x
    # wider than a unit of z is a different shape, so `set_aspect('equal')` is not optional.
    # The two truths do not have the same proportions -- the ODE spans about 40 in x and 46 in
    # z, the SDE 54 and 72 -- so framing each on its own data gives two rows of different
    # height. Widening the shorter axis of each frame until its two spans match makes every
    # panel a square of the same size. The cost is air: the SDE frame goes out to +-36 in x
    # where its data reaches +-27, so the SDE attractors sit smaller inside their squares.
    frames = {}
    for kind in KINDS:
        t = data(kind).raw(data(kind).eval).reshape(-1, 3)[:n_show]
        lim = []
        for lo, hi in ((t[:, 0].min(), t[:, 0].max()), (t[:, 2].min(), t[:, 2].max())):
            pad = 0.10 * float(hi - lo)
            lim.append((float(lo) - pad, float(hi) + pad))
        # SQUARE FRAMES, so both rows are the same size and neither butterfly is stretched.
        # The two truths do not have the same proportions -- the ODE spans about 40 in x and
        # 46 in z, the SDE 54 and 72 -- so framing each on its own data gives two rows of
        # different height. Widening the shorter axis of each frame until the two spans match
        # makes every panel a square of the same size, at the cost of air around the SDE, and
        # `set_aspect('equal')` then keeps a unit of x the same length as a unit of z.
        side = max(lim[0][1] - lim[0][0], lim[1][1] - lim[1][0])
        frames[kind] = [((lo + hi) / 2 - side / 2, (lo + hi) / 2 + side / 2) for lo, hi in lim]

    with plt.rc_context({'font.size': F2, 'axes.titlesize': F2, 'axes.labelsize': F2,
                         'xtick.labelsize': SMALL2, 'ytick.labelsize': SMALL2}):
        fig, axes = plt.subplots(len(KINDS), len(cols),
                                 figsize=(W2, 4.10), sharex='row', sharey='row')

        for r, kind in enumerate(KINDS):
            d = data(kind)
            states = {'truth': d.raw(d.eval)}
            for topic, stem in MODELS.items():
                name = f'{stem}_{kind}_s{rep(topic, kind)}'
                # The Gaussian samples, so its rollout is a draw and not a function of the
                # checkpoint alone. Seed it, or this figure changes every time it is run.
                m, _ = load_model(name)
                with torch.no_grad(), torch.random.fork_rng():
                    torch.manual_seed(0)
                    states[SHORT[topic]] = d.raw(m.forecast(d.eval[:n_ic, :m.history], steps))
                print(f'  {name}')

            for c, nm in enumerate(cols):
                ax = axes[r, c]
                s_ = states[nm].reshape(-1, 3)[:n_show]
                ax.plot(s_[:, 0], s_[:, 2], lw=0.35,
                        color='0.5' if nm == 'truth' else P.RED, alpha=0.85)
                ax.tick_params(length=2.5, pad=1.5)
                ax.set_aspect('equal', adjustable='box')
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if r == 0:
                    ax.set_title(nm, pad=5)
                if r == len(KINDS) - 1:
                    ax.set_xlabel('$x$', labelpad=0)

            # Each row is framed on ITS OWN truth, with 10 % of air. Sharing one frame across
            # both rows spends 40 % of the ODE row's height on a band the ODE never visits,
            # and the comparison this figure is making is truth-against-model inside a row.
            # A model that leaves the frame -- the Gaussian, which over-disperses -- is drawn
            # running off the top edge, which is the honest picture of leaving it.
            axes[r, 0].set_xlim(*frames[kind][0])
            axes[r, 0].set_ylim(*frames[kind][1])

            # The row label carries the dataset, so neither row needs a title of its own.
            axes[r, 0].set_ylabel(ROW[kind] + '\n$z$', labelpad=1)

        fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.6)
        bank(fig, 'p_attractors')


# ============================================================ every ruler, every model

# (key in summary.json, column head). `alive` is a gate and is drawn last for that reason.
RULERS = [('horizon_seeds', 'horizon   time steps'),
          ('spread', 'spread   $\\sigma$ ratio'),
          ('climate', 'climate   $W_1$ ratio'),
          ('chaos', 'chaos   $\\lambda_1$ ratio')]

# Axis per column, per dataset. `chaos` is the one column that clips: the lead-time model
# reaches -12 on the ODE, and an axis wide enough to hold that squashes the 0-to-1 range where
# every other model lives. A clipped point is drawn as an arrow at the edge with its value
# printed, which is honest and keeps the column readable.
LIMS = {'horizon_seeds': {'ode': (0, 560), 'sde': (0, 64)},
        'spread': {'ode': (-0.05, 1.34), 'sde': (-0.05, 1.34)},
        'climate': {'ode': (0.40, 300), 'sde': (0.40, 300)},
        'chaos': {'ode': (-1.35, 1.45), 'sde': (-1.35, 1.45)}}

FMT = {'horizon_seeds': lambda v: f'{v:.0f}'}


def scorecard() -> None:
    """Every ruler, every model in `MODELS`, both ground truths, with the five-seed range drawn.

    A dot is the median over five training seeds and the bar through it is the full range.
    That bar is not decoration: `horizon`, `climate` and `alive` are bimodal here, and where
    two bars overlap the two models are not distinguishable by this experiment. Two tables of
    thirty cells said the same thing in 150 printed numbers; this says it in 50 and shows the
    overlap, which no table can.

    THE ROW COUNT IS `len(MODELS)` AND THE FIGURE BOX IS NOT. Rows are laid out on a y axis
    that runs from -0.55 to `n + 0.80` inside a fixed 4.15 in height, so adding a model tightens
    the row pitch and changes nothing else: the drawn width, the scale factor at which
    `poster.typ` places it, and therefore every type size in the table stay put. At n = 6 the
    pitch is about 41 pt against 10.6 pt labels. That is what makes the one-step MLP row free.
    """
    topics = list(MODELS)
    n = len(topics)

    with plt.rc_context({'font.size': F3, 'axes.titlesize': F3, 'axes.labelsize': F3,
                         'xtick.labelsize': SMALL3, 'ytick.labelsize': F3}):
        fig, axes = plt.subplots(2, len(RULERS), figsize=(W3, 4.15),
                                 gridspec_kw=dict(width_ratios=[1.22, 1.0, 1.12, 1.12],
                                                  wspace=0.16, hspace=0.30))

        for r, kind in enumerate(KINDS):
            ds = GT['datasets'][kind]
            fi, f2 = lambda v: f'{v:.0f}', lambda v: f'{v:.2f}'
            refs = {'horizon_seeds': [(ds['euler_bar_steps'], 'Euler', fi),
                                      (ds['floor_steps'], 'truth', fi)],
                    'spread': [(1.0, 'truth', f2)],
                    'climate': [(ds['truth_climate'], 'truth', f2)],
                    'chaos': [(1.0, 'truth', f2)]}

            for c, (key, head) in enumerate(RULERS):
                ax = axes[r, c]
                x0, x1 = LIMS[key][kind]
                log = key == 'climate'
                if log:
                    ax.set_xscale('log')
                    ax.set_xticks([1, 10, 100])
                    ax.set_xticklabels(['1', '10', '100'])
                    ax.minorticks_off()

                # Every reference is drawn AND named AND given its value, in a band above the
                # five models. It is what the column is read against, it differs between the
                # two datasets, and a dashed line nobody labels is a line nobody trusts.
                drawn_fracs = []
                for x, name, f in refs[key]:
                    if x0 < x < x1:
                        ax.axvline(x, ls=(0, (2, 2)), lw=1.4, color='0.4', zorder=2)
                        frac = (_lg(x, x0, x1) if log else (x - x0) / (x1 - x0))
                        # two references closer than a label's width share a column of the
                        # header band, so the second one drops to a lane of its own
                        lane = (n - 0.14 if all(abs(frac - g) > 0.22 for g in drawn_fracs)
                                else n + 0.42)
                        drawn_fracs.append(frac)
                        ha = ('left' if frac < 0.2 else 'right' if frac > 0.8 else 'center')
                        ax.annotate(f'{name} {f(x)}', xy=(x, lane), fontsize=SMALL3,
                                    color='0.35', va='center', ha=ha, zorder=6,
                                    xytext=({'left': 4, 'right': -4, 'center': 0}[ha], 0),
                                    textcoords='offset points',
                                    bbox=dict(fc='white', ec='none', pad=1.0))

                drawn = False
                for i, t in enumerate(topics):
                    v = SUMMARY['models'][f'{MODELS[t]}_{kind}'][key]
                    if v['median'] is None:
                        continue
                    drawn = True
                    _mark(ax, n - 1 - i, v, x0, x1, FMT.get(key, f2), log)

                if not drawn:
                    ax.text(0.5, 0.40, 'not defined —\ntruth’s own\nspread is zero',
                            transform=ax.transAxes, ha='center', va='center',
                            color=P.GREY, fontsize=SMALL3)

                ax.set_xlim(x0, x1)
                ax.set_ylim(-0.55, n + 0.80)
                ax.set_yticks(range(n))
                ax.set_yticklabels([SHORT[t] for t in topics][::-1] if c == 0 else [])
                ax.tick_params(axis='y', length=0)
                ax.tick_params(axis='x', length=3, pad=2)
                ax.grid(axis='x', alpha=0.18, lw=0.6)
                ax.set_axisbelow(True)
                if r == 0:
                    ax.set_title(head, pad=9)
            axes[r, 0].set_ylabel(ROW[kind], labelpad=8)

        fig.subplots_adjust(left=0.062, right=0.995, top=0.925, bottom=0.075,
                            wspace=0.16, hspace=0.26)
        bank(fig, 'p_scorecard')


def _lg(x, x0, x1) -> float:
    import math
    return math.log(x / x0) / math.log(x1 / x0)


def _mark(ax, y, v, x0, x1, fmt, log) -> None:
    """One model on one ruler: the five-seed range as a bar, the median as a dot."""
    med, lo, hi = v['median'], v['lo'], v['hi']
    a, b = max(lo, x0), min(hi, x1)
    if b > a:
        ax.plot([a, b], [y, y], color=P.GREY, lw=3.4, alpha=0.45, solid_capstyle='butt',
                zorder=3)

    span = (0.06 * (x1 - x0)) if not log else 0
    if med <= x0:
        ax.plot([x0], [y], '<', ms=9, color=P.RED, zorder=5, clip_on=False)
        ax.annotate(fmt(med), xy=(x0, y), xytext=(11, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=SMALL3)
    elif med >= x1:
        ax.plot([x1], [y], '>', ms=9, color=P.RED, zorder=5, clip_on=False)
        ax.annotate(fmt(med), xy=(x1, y), xytext=(-11, 0), textcoords='offset points',
                    va='center', ha='right', fontsize=SMALL3)
    else:
        ax.plot([med], [y], 'o', ms=8.5, color=P.RED, zorder=5)
        # Print to the right of the dot, unless the dot is far enough right that the label
        # would run off the axis -- then print to its left.
        right = (med < x1 - 2.4 * span) if not log else (med < x1 ** 0.55 * x0 ** 0.45)
        ax.annotate(fmt(med), xy=(med, y), xytext=(11 if right else -11, 0),
                    textcoords='offset points', va='center',
                    ha='left' if right else 'right', fontsize=SMALL3)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--ar', action='store_true',
                   help='draw the AR-trained models instead, banking to p_*_ar.{pdf,png}')
    # Redrawing one figure is the common case when a row list changes, and redrawing all of
    # them is not free: `attractors` rolls every model out 8000 steps. `--only` also keeps a
    # figure that was deliberately hand-picked (the hook's IC) from being silently rebuilt.
    p.add_argument('--only', nargs='+', metavar='NAME', default=None,
                   help='draw only these figures (hook system conditional scorecard attractors)')
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    figures = [('hook', hook), ('system', system), ('conditional', conditional),
               ('scorecard', scorecard), ('attractors', attractors)]

    if a.ar:
        MODELS, SHORT, SUFFIX, HOOK_IC = MODELS_AR, SHORT_AR, '_ar', HOOK_IC_AR
        HOOK_TOPIC = HOOK_TOPIC_AR
        # `p_system` and `p_conditional` draw the ground truths and no model at all, so an AR
        # copy of them would be a byte-for-byte duplicate under a second name. Skipped.
        #
        # `p_hook` IS drawn, from row 02's AR model -- but on a different initial condition,
        # which is the whole reason HOOK_IC moves with MODELS. 122 is the 6th percentile for
        # the k=4 rollout model and the 69th for the one-step MLP, so reusing it would put a
        # near-worst rollout on the opening figure and read as a verdict the scorecard does
        # not support. See the second warning in `hook`.
        figures = [(n, f) for n, f in figures if n not in ('system', 'conditional')]

    if a.only:
        known = {n for n, _ in figures}
        unknown = [n for n in a.only if n not in known]
        if unknown:
            p.error(f'not drawable here: {" ".join(unknown)}  (have: {" ".join(sorted(known))})')
        figures = [(n, f) for n, f in figures if n in set(a.only)]

    for name, fn in figures:
        print(name + (' (AR)' if a.ar else ''))
        fn()
