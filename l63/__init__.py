"""Lorenz 63 forecasting: shared machinery for the notebooks in `notebooks/`.

Deliberately small. Seven modules, flat, one job each:

    data      the true system, both integrators, and the frozen datasets
    known     the standard results (Strogatz ch. 9), each as one function
    models    the ForecastModel contract + one class per model
    train     the Adam loop
    evaluate  the four rulers
    lyapunov  Benettin, for the true system and for a learned map
    plots     every figure written to `figures/`

Anything a second model would otherwise copy lives here; the per-model story stays in
its notebook.

Notation, fixed once and used everywhere (see README.md):

    u_n       the state at time step n, a vector in R^3
    u-hat_n   the model's state; a hat means the model's, no hat is the truth
    dt        = 0.025, THE time step: one model call advances the state by this much
    dt_int    = 2.5e-4, the integrator's substep; 100 of them make one dt
    n         the only step index

Errors are reported in RAW Lorenz units -- the same units as x, y, z. Normalisation is
an internal training detail and never appears on an axis label.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / 'figures'          # resolved from __file__, so it does not matter which
ARTIFACTS = ROOT / 'artifacts'     # directory a notebook or script runs from
