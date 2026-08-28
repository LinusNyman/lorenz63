# Lorenz 63 — deterministic and probabilistic forecasting

Seven neural forecasters, deterministic and probabilistic, trained on the Lorenz 63 system
and scored with one common set of measures. Three datasets, four rulers and a gate.

The question every topic answers: **how many time steps can a model take before the result is
too bad to use, and what counts as too bad?**

Every level a number is read against is either a property of the system or the integrator run
against itself, and each is printed beside the number it judges. Two constants are chosen
rather than measured: `alive_frac`'s 0.25 / 5000, and `vpt`'s 0.4, the published NRMSE
convention.

---

## Notation

One alphabet, in the code, the figures and the poster.

| symbol | meaning |
| --- | --- |
| **u**_n | the state at time step *n*. Bold, because it is a vector in ℝ³ |
| **û**_n | the model's state. A hat means the model's; no hat is the truth |
| Δt = 0.025 | **the time step.** One model call advances the state by this much |
| Δt_int = 2.5×10⁻⁴ | the integrator's substep. 100 of them make one Δt |
| n | the **only** step index |

Two conventions:

- **Errors are in raw Lorenz units**, the same units as x, y, z, so ‖**û**_n − **u**_n‖ is the
  distance between two points in state space. Training normalises to zero mean and unit
  variance for conditioning; that normalisation never reaches an axis.
- **The training series is u₀, u₁, …, u_N**: consecutive and complete, one training pair per
  consecutive pair. The 100 substeps between **u**_n and **u**_{n+1} are how the integrator
  reaches **u**_{n+1} accurately, not withheld data.

## One time step

The integrator takes 100 substeps of 2.5×10⁻⁴ to advance the state by Δt = 0.025; the model
does it in one call. `evaluate.horizon` therefore compares each model against explicit Euler
taking one step of Δt, which is the same number of function evaluations as one model call.

Equal call counts are not equal cost. One Euler step is about 16 floating-point operations
against the 128-wide MLP's 35 000. Measured wall clock on one core at batch 128: 57 µs for the
model, 13.6 µs for the Euler step, and 1 340 µs for the 100-substep integrator the model
replaces. The model costs 4× a single coarse step and 24× less than the ground-truth
integrator.

## The four rulers, and the gate

| ruler | reads as | truth | dead |
| --- | --- | --- | --- |
| **horizon** | time steps until ‖**û**_n − **u**_n‖ reaches the attractor scale | — | small |
| **spread** | σ_ensemble(model) / σ_ensemble(truth) | 1.00 | 0.00 |
| **climate** | W₁(model, truth) ÷ W₁(truth, truth), on the x/y/z marginals | see below | ≫1 |
| **chaos** | λ₁(learned map) ÷ λ₁(true map) | 1.00 | ≤0 |
| **alive** | fraction of long rollouts still moving like the system | 1.00 | 0.00 |

In words: how long a forecast stays usable · whether the model's uncertainty is calibrated ·
whether long rollouts visit the right places · whether the model stretches errors at the true
rate · whether long rollouts are still moving.

`alive` answers one of the project's two named tests, whether long rollouts stay on the
attractor, so it carries its across-seed range like the other four.

**Each ruler measures the map the model iterates.** For the recurrent rungs that map is
(**u**, h, c) rather than **u**, since the hidden state carries through the whole rollout.
Measuring the memoryless map instead gave an LSTM that tracked truth for 440 steps and scored
λ₁ ≈ 0 in the same row. Each model declares its own state in `models.map_state` / `map_step`.

**Each ruler takes five seeds.** Retrained on five seeds, one configuration gives horizons
spanning 45 % to 275 % of their own median, because `horizon` is read at the attractor scale
where the error curve is already flat. Every reported number is a median over five seeds with
its full range beside it.

**`spread` is undefined on the ODE**: truth's ensemble spread there is identically zero, so the
ratio is 0/0 and prints as "—". A zero there would read as a score.

**`climate` does not read 1.00 at truth, and its resolution is coarse.** Its denominator is two
disjoint halves of a finite truth pool. Over five independent truth draws the anchor is
**0.62 (range 0.56–1.89)** on the ODE and **0.66 (0.57–1.00)** on the SDE; `climate_vs_truth`
divides by it, so 1.00 means as close to truth as an independent draw of truth is. Compare a
model against the truth row rather than against 1.

## What this suite cannot resolve

Topic 03 at k = 1 is topic 02's loss through the rollout code path — the same model and the
same loss, differing only in that one reshapes to (N, 3) and the other keeps (batch, T, 3), so
the sums happen in a different order. After 3000 Adam steps the weights differ by ~7×10⁻⁷.
What each ruler does with that difference is its noise floor, before any seed enters:

| horizon | chaos | climate | alive |
| --- | --- | --- | --- |
| 0–7 % | 0.0–5.4 % | **0–80 %** | **0–50 %** |

`horizon` and `chaos` are therefore sharp, while `climate` and `alive` cannot resolve a factor
of two. `report.controls` computes this from rows that already exist, and it bounds any claim
that two models differ.

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

There is no `e`: the rulers go into a table rather than a bar chart, because a table carries
the across-seed range and a bar chart of medians does not.

A figure is named `<topic><letter>_<name>_<dataset>.png`, with no separator between the topic
and the letter. Every name carries its dataset key, so one topic has up to three of each:
`02d_error_ode.png`, `02d_error_sde.png`, `02d_error_sde015.png`.

⚠️ **Two topics end in a letter, so a prefix can be read two ways.** `06f_lorenz_map_ode.png`
is topic **06** with letter `f`; `06fd_error_ode.png` is topic **06f** with letter `d`. No two
figures share a filename, but `06f*` matches both topics: match topics 05 and 06 on the letter
that follows (`06[abcdf]_`), and topics 05t and 06f on the letter after that (`06f[abcdf]_`).

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

Shared machinery lives in `l63/` and the per-model story stays in its notebook. Copying
toolkit code back into a notebook breaks the like-for-like comparison between models.

## The three datasets

The SDE is the experiment; the other two are controls, and each answers a distinct question
about the result.

| key | what it is | why it is here |
| --- | --- | --- |
| `sde` | Euler–Maruyama at **b = 0.6** | the experiment |
| `ode` | explicit Euler, no noise | the b → 0 limit, where p(**u**_{n+1}\|**u**_n) is a point mass and a probabilistic model has nothing to recover |
| `sde015` | Euler–Maruyama at **b = 0.15** | the starter notebook's noise level, and the evidence for changing it |

The noise level is the independent variable: it sets the width of p(**u**_{n+1} | **u**_n),
and an MSE-trained model learns the mean of that distribution. Widths measured by
`run/width_sweep.py`:

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

The sweep narrows the choice to {0.30, 0.60}: both leave the attractor undistorted and both
are wide enough to separate the model families. **0.60 was chosen from those two by hand**, as
the wider one that still leaves the attractor only 7 % broader than the deterministic case. The
sweep bounds the choice; it does not make it.

At the starter's 0.15, which is 4.2 % of attractor scale, the deterministic and probabilistic
rungs are not distinguishable. On `sde015` the MLP scores alive 1.00 and chaos 0.94 against the
Gaussian's 1.00 and 0.99, and the MLP has the longer horizon (110 against 89). On `sde` the same
MLP scores alive 0.19 and chaos 0.42 against the Gaussian's 1.00 and 1.00. The comparison this
project makes exists only at the wider noise level.

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
