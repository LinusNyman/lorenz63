# Lorenz 63 — deterministic and probabilistic forecasting

Seven models, three datasets, four rulers and a gate. The question every topic answers is the
same one:

> **How many time steps can the model take before the result is too bad to use — and what
> counts as too bad?**

Both halves are answered from measurement. No level a number is read against is tuned to make
a model look good: each is either a property of the system or the integrator run against
itself, and it is printed beside the number it judges. Two constants *are* chosen and are
named as such — `alive_frac`'s 0.25 / 5000 and `vpt`'s literature 0.4.

---

## Notation

One alphabet, used everywhere — in the code, in the figures, and on the poster.

| symbol | meaning |
| --- | --- |
| **u**_n | the state at time step *n*. Bold, because it is a vector in ℝ³ |
| **û**_n | the model's state. A hat means the model's; no hat is the truth |
| Δt = 0.025 | **the time step.** One model call advances the state by this much |
| Δt_int = 2.5×10⁻⁴ | the integrator's substep. 100 of them make one Δt |
| n | the **only** step index |

Two conventions that remove a class of question before it is asked:

- **Errors are reported in raw Lorenz units** — the same units as x, y, z. So
  ‖**û**_n − **u**_n‖ means what a reader assumes: the distance between two points in state
  space. Normalisation to zero mean and unit variance happens inside training, for
  conditioning, and never appears on an axis.
- **The training series is u₀, u₁, …, u_N.** Consecutive, complete, one training
  pair per consecutive pair, nothing withheld. The 100 substeps between **u**_n and **u**_{n+1}
  are how the integrator makes **u**_{n+1} accurate; they are not skipped data.

## What one time step is, and why it matters

The integrator takes 100 substeps of 2.5×10⁻⁴ to advance the state by Δt = 0.025. The model
does it in **one** call. That is the whole reason a learned model is interesting here, and it
is what `evaluate.horizon` compares against: the reference is explicit Euler taking *one* step
of Δt — **the same number of function evaluations** as one model call.

Same call count is not same cost, and the difference is worth stating rather than glossing.
One Euler step is ~16 floating-point operations; the 128-wide MLP is ~35 000. Measured wall
clock on one core at batch 128: 57 µs for the model against 13.6 µs for the Euler step, and
1 340 µs for the 100-substep integrator it replaces. So the model is 4× a single coarse step
and 24× cheaper than the ground-truth integrator — the speed-up is real, and it is 24×, not
100×.

## The four rulers, and the gate

| ruler | reads as | truth | dead |
| --- | --- | --- | --- |
| **horizon** | time steps until ‖**û**_n − **u**_n‖ reaches the attractor scale | — | small |
| **spread** | σ_ensemble(model) / σ_ensemble(truth) | 1.00 | 0.00 |
| **climate** | W₁(model, truth) ÷ W₁(truth, truth), on the x/y/z marginals | see below | ≫1 |
| **chaos** | λ₁(learned map) ÷ λ₁(true map) | 1.00 | ≤0 |
| **alive** | fraction of long rollouts still moving like the system | 1.00 | 0.00 |

`alive` is one of the brief's two tests ("do long rollouts stay on the attractor"), so it is a
headline number reported with its across-seed range, not a footnote.

Read as: *how long is it right* · *does it admit when it is unsure* · *does it live in the
right place* · *does it move the right way* · *is it still moving at all*.

**Every ruler is measured on the map the model actually iterates.** For the recurrent rungs
that map is (**u**, h, c), not **u** — the hidden state is carried through the whole rollout —
and measuring the memoryless map instead produced an LSTM that tracked truth for 440 steps and
scored λ₁ ≈ 0 in the same row. `models.map_state` / `map_step` is where each model says what
its own state is.

**Every ruler needs five seeds — measured, and not what was planned.** The plan said `horizon`
would need one. Retrained on five seeds the same configuration gives horizons spanning
**45 % to 275 %** of their own median, because `horizon` is read at the attractor scale where
the error curve is already flat. Every number reported is a median over five seeds with its
full range printed beside it.

**`spread` is not defined on the ODE.** Truth's ensemble spread there is identically zero, so
the ratio is 0/0. It is printed as "—", which is the honest reading and not a score of zero.

**`climate` does not read 1.00 at truth, and its resolution is coarse.** Its denominator is two
disjoint halves of a finite truth pool, and the anchor is measured, not assumed: over five
independent truth draws it is **0.62 (range 0.56–1.89)** on the ODE and **0.66 (0.57–1.00)** on
the SDE. `climate_vs_truth` divides by it, so 1.00 there means "as close to truth as an
independent draw of truth is". Read a model against the truth row, never against 1.

## What this suite cannot resolve

Topic 03 at k = 1 is topic 02's loss through the rollout code path — the same model and the
same loss, differing only in that one reshapes to (N, 3) and the other keeps (batch, T, 3), so
the sums happen in a different order. After 3000 Adam steps the weights differ by ~7×10⁻⁷.
What each ruler does with that difference is its noise floor, before any seed enters:

| horizon | chaos | climate | alive |
| --- | --- | --- | --- |
| 0–7 % | 0.0–5.4 % | **0–80 %** | **0–50 %** |

So `horizon` and `chaos` are sharp, and `climate` and `alive` cannot see a factor-of-two
difference. This is computed in `report.controls` from rows that already exist, and it is the
number to quote when asked whether two models really differ.

## Layout

One number per topic, and the same number in every directory.

| # | topic | notebook | figures |
| --- | --- | --- | --- |
| 00 | ground truth, and the standard results | `00_ground_truth.ipynb` | `00*` |
| 01 | the four rulers | `01_rulers.ipynb` | `01*` |
| 02 | MLP one-step, MSE (+ architecture sweep) | `02_mlp.ipynb` | `02*` |
| 03 | MLP rollout-k | `03_rollout.ipynb` | `03*` |
| 04 | lead time, F(**u**, s) | `04_leadtime.ipynb` | `04*` |
| 05 | RNN / LSTM | `05_recurrent.ipynb` | `05*` |
| 05t | transformer | `05t_transformer.ipynb` | `05t*` |
| 06 | Gaussian, N(μ(**u**), Σ(**u**)) | `06_gaussian.ipynb` | `06*` |
| 06f | flow matching | `06f_flow.ipynb` | `06f*` |
| 07 | head-to-head | `07_head_to_head.ipynb` | `07*` |

Within a topic the letter always means the same thing:

`a` data · `b` loss · `c` architecture · `d` error vs time step · `f` Lorenz map

There is no `e`: the rulers are reported as a table rather than a bar chart, because a table
can carry the across-seed range and a bar chart of medians cannot.

⚠️ **A suffixed topic collides with a letter.** `06f_lorenz_map_ode.png` is topic **06**'s
Lorenz map; `06ff_lorenz_map_ode.png` is topic **06f**'s. Glob on the full `<topic>_<letter>`
prefix, never on `06f*`. Every figure name also carries its dataset key, so one topic has up to
three of each: `02d_error_ode.png`, `02d_error_sde.png`, `02d_error_sde015.png`.

```
l63/          the shared package — seven files, one job each
  data.py       the system, both integrators, both reference pairs, the frozen datasets
  known.py      the standard results (Strogatz ch. 9), each as one function
  models.py     the ForecastModel contract + one class per model + each model's own map
  train.py      the Adam loop
  evaluate.py   the four rulers and the gate
  lyapunov.py   Benettin, for the true system and for a learned map
  plots.py      every figure
notebooks/    one thin driver per topic, each closing with ## Findings
run/          width_sweep.py · ground_truth.py · train_all.py · report.py · make_notebooks.py
artifacts/    frozen datasets, checkpoints, results  (regenerable)
figures/      written by plots.save                  (regenerable)
poster/       the A0 poster: Typst source, its figures, and poster_final.pdf
```

Shared machinery lives in `l63/`; the per-model story stays in its notebook. Never copy
toolkit code back into a notebook — that is what breaks apples-to-apples comparisons.

## The three datasets

The brief prescribes stochastic trajectories throughout, so **the SDE is the experiment**. The
other two are controls, and each answers a distinct question about the result.

| key | what it is | why it is here |
| --- | --- | --- |
| `sde` | Euler–Maruyama at **b = 0.6** | the experiment |
| `ode` | explicit Euler, no noise | the b → 0 limit the brief itself describes, where p(**u**_{n+1}\|**u**_n) is a point mass and a probabilistic model has nothing to recover |
| `sde015` | Euler–Maruyama at **b = 0.15** | the starter notebook's own noise level — the evidence for having changed it |

The noise level is the experiment's independent variable, because it sets the width of
p(**u**_{n+1} | **u**_n), and an MSE-trained model learns the *mean* of that distribution.
Measured widths, from `run/width_sweep.py`:

| b | width | % of attractor scale | SDE attractor scale |
| --- | --- | --- | --- |
| 0.15 | 0.62 | 4.2 % | 14.50 |
| 0.30 | 1.24 | 8.4 % | 14.56 |
| **0.60** | **2.48** | **16.9 %** | **15.79** |
| 1.00 | 4.16 | 28.2 % | 19.49 |
| 1.50 | 6.29 | 42.7 % | 28.27 — attractor distorted |

Percentages are against the *deterministic* attractor's own ‖σ‖ = 14.72, not against the
column beside them. Scored on the frozen datasets at 500 samples rather than the sweep's 400,
b = 0.15 reads 4.4 % and b = 0.60 reads 16.5 %; the difference is the sample size.

The sweep narrows the choice to {0.30, 0.60} — both leave the attractor undistorted and both
are wide enough to matter. **0.60 was chosen from those two by hand**, as the wider one that
still leaves the attractor only 7 % broader than the deterministic case. The sweep did not
pick it on its own, and saying it did would be an overclaim.

**Why not the starter's 0.15, which the brief assumes is enough?** Because at 4.2 % of
attractor scale the deterministic and probabilistic rungs are not distinguishable, and that is
now trained rather than inferred from a width. On `sde015` the MLP scores alive 1.00 and chaos
0.94 against the Gaussian's 1.00 and 0.99, and the MLP has the *better* horizon (110 vs 89).
On `sde` the same MLP scores alive 0.19 and chaos 0.42 against the Gaussian's 1.00 and 1.00.
The comparison the project exists to make only exists at the wider noise level.

## Running

Set up the environment once:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Then run the stages in order — each reads the previous one's artifacts:

```bash
PYTHONPATH=. ./.venv/bin/python run/width_sweep.py
```
```bash
PYTHONPATH=. ./.venv/bin/python run/ground_truth.py
```
```bash
PYTHONPATH=. ./.venv/bin/python run/train_all.py --workers 6
```
```bash
PYTHONPATH=. ./.venv/bin/python run/report.py
```

`artifacts/` is not in the repository apart from two JSON summaries: the datasets and the
checkpoints are regenerable from their seeds, and `ground_truth.pt` is 154 MB. The two stages
above rebuild them. `artifacts/summary.json` is tracked, because the poster reads every number
it prints from it.

`train_all.py` is resumable: a job whose result already exists is skipped, so an interrupted
run continues where it stopped. `--only 03` runs one topic, `--data sde015` one dataset,
`--force` reruns.

**When a ruler changes, re-score instead of retraining.** This re-runs the whole suite against
the checkpoints already on disk and touches no weights — 159 models in 17 minutes, against
hours to retrain them, and it keeps a change in the measurement separate from any change in
the models:

```bash
PYTHONPATH=. ./.venv/bin/python run/train_all.py --score-only --workers 6
```

Notebooks are generated, and regenerating them keeps any Findings already written:

```bash
PYTHONPATH=. ./.venv/bin/python run/make_notebooks.py
```
```bash
PYTHONPATH=. ./.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/02_mlp.ipynb
```

Build the poster — A0, one page, and the figures it places:

```bash
PYTHONPATH=. ./.venv/bin/python poster/make_figures.py
```
```bash
typst compile --root .. poster/poster_final.typ poster/poster_final.pdf
```
