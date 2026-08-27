# Plan — Lorenz 63, exploration 2

What this repo is doing, in the order it does it. `README.md` is how to run it; this is why.

The question every topic answers:

> How many time steps can the model take before the result is too bad to use — and what counts
> as too bad?

---

## 0. The rule the whole thing is built on

**No level is tuned to make a model look good.** Every level a number is read against is either
a property of the system or the integrator run against itself, and it is printed beside the
number it judges. Where a ruler has its own noise, that noise is measured and printed too.

Two constants *are* chosen, and calling the suite threshold-free would be false: `alive_frac`'s
0.25 / 5000 (the quantity is bimodal, so anything from ~0.05 to ~0.6 gives the same partition)
and `vpt`'s 0.4 (the published NRMSE convention, quoted so outside numbers are comparable, and
never mixed with the step counts).

The second rule follows from the first: **the truth row goes on top of every table, measured
through the same code as every model** — long true rollouts of the same count and length, fed
through the identical functions. It used to be three hardcoded 1.00s, which is an assertion and
not a calibration. Truth does not land on 1.00 for `climate`, and that is information.

---

## 1. Topic 00 — ground truth, and the standard results

Reproduced before any model exists. If they do not come out, the integrator is wrong and
nothing downstream is worth running. `l63/known.py`, one function each.

Two kinds of row, and they are **not the same evidence**:

*Closed forms evaluated — these check transcription and cannot fail.*

| result | measured | expected |
| --- | --- | --- |
| fixed points C± | ±8.485, 27 | ±8.485, 27 |
| Hopf threshold ρ_H | 24.737 | 24.74, and ρ = 28 is past it |
| ∇·f from the formula | −13.667 | −13.667 |

*Numerical — these can fail, and they are the integrator check.*

| result | measured | expected |
| --- | --- | --- |
| ∇·f as trace(Df) at 2000 attractor states | −13.666668, max deviation 0 | −13.667 everywhere |
| symmetry residual | 0 | 0 |
| spectrum λ₁, λ₂, λ₃ | 0.909, −0.006, −14.579 | 0.906, 0, −14.57 |
| **Σλ vs ∇·f** | **−13.676 vs −13.667** | an identity — the sharpest check available |
| Kaplan–Yorke dimension | 2.062 | Strogatz measured ≈ 2.05 |
| Lorenz map, min \|f′\| | 1.15 over 534 maxima | > 1 everywhere |

λ is measured with Benettin, renormalised at every step, averaged along the trajectory, in
float64 — Strogatz 9.3's three qualifications are exactly why. λ₁ of the *true map at Δt*,
through the same estimator every model gets and at the same 8000 steps, is **0.897 ± 0.023**
over 64 initial conditions; that is the denominator of the `chaos` ruler, on every dataset.
The SDE has no deterministic map to take an exponent of, so on the stochastic datasets the
ratio means "against the underlying deterministic dynamics".

### The three datasets

The brief prescribes stochastic trajectories throughout, so the SDE at b = 0.6 is the
experiment. `ode` (b → 0) and `sde015` (the starter's b = 0.15) are controls. Measured widths
and the reason 0.6 was picked are in `README.md`; the short version is that the sweep narrows
it to {0.3, 0.6} and a human took the wider one.

The attractor's own scale differs from the evaluation set's by about 5 % on the ODE (14.76
against 13.98), because 901 steps is only 20 Lyapunov times and a single trajectory that short
does not sample the invariant measure — one has ‖std‖ = 13.1 ± 3.1. The rulers use the
evaluation set's scale, since that is the data the error is measured on, but both are recorded
and the gap is why `climate` compares long rollouts against long rollouts.

---

## 2. Topic 01 — the four rulers and the gate

| ruler | reads as | truth | dead |
| --- | --- | --- | --- |
| **horizon** | time steps until ‖**û**_n − **u**_n‖ reaches the attractor scale | — | small |
| **spread** | σ_ens(model) / σ_ens(truth) | 1.00 | 0.00 |
| **climate** | W₁(model, truth) ÷ W₁(truth, truth), on the marginals | measured, not 1 | ≫1 |
| **chaos** | λ₁(learned map) ÷ λ₁(true map) | 1.00 | ≤0 |
| **alive** | fraction of long rollouts still moving like the system | 1.00 | 0.00 |

`alive` is one of the brief's two named tests, so it is ranked and reported with its range —
it is not "a gate, never ranked", which was this plan's earlier position and does not match
what the brief asks for.

### "Too large" — the bars are measured, and they differ per dataset

| dataset | same-cost solver usable | ground truth usable |
| --- | --- | --- |
| `ode` | 32 steps | 362 steps |
| `sde` (b = 0.6) | 26 steps | **23 steps** |
| `sde015` (b = 0.15) | 29 steps | 104 steps |

On the ODE the floor is discretisation error — Euler at Δt_int against Δt_int/2. **On the SDE
it is not**, and using the deterministic curve there was wrong: a stochastic system's
irreducible uncertainty is realisation-to-realisation, it does not shrink if the step is
refined, and it reaches attractor scale at step 23. The old figure of 355 was the deterministic
discretisation error computed on stochastic data, and it made every SDE model read "loses to
the same-cost solver at step 1". The bar is now Euler–Maruyama at Δt against Δt_int **driven by
the same Brownian path**, so it measures method and not noise.

A consequence to state rather than bury: on the SDE the models' horizons (27–49 steps) sit
*above* the 23-step floor. A deterministic model shadows the conditional mean, which stays near
a given realisation longer than a second realisation does. That is real, and it is also why
`horizon` is not the ruler that decides anything on the SDE.

### What the rulers cannot see

- **`spread` is undefined on the ODE.** Truth's ensemble spread there is identically zero, so
  the ratio is 0/0. Printed as "—". Not a score of zero. On the SDE a second independent truth
  ensemble scores 1.02 against the first, which is the ruler's own noise at 16 members.
- **`climate` does not read 1.00 at truth and the anchor is measured.** Over five independent
  truth draws: 0.62 (0.56–1.89) on the ODE, 0.66 (0.57–1.00) on the SDE, 1.01 (0.78–1.42) on
  `sde015`. Quoting a single draw as the anchor is what previously made it look biased.
  `climate_vs_truth` divides by the median anchor.
- **Both `climate` and `alive` are coarse, and that is now quantified.** The k = 1 control —
  the same model and the same loss, differing only by float summation order, ~7×10⁻⁷ in the
  weights — moves `climate` by up to **80 %** and `alive` by up to **50 %**, while `horizon`
  moves ≤ 7 % and `chaos` ≤ 5.4 %. Neither `climate` nor `alive` can resolve a factor of two.

**Every ruler needs five seeds.** The plan said `horizon` would need one. Retrained on five
seeds the same configuration gives horizons spanning **45 % to 275 %** of their own median, and
for two rungs seed 0 was the best of the five. Every number is a median over five seeds with
its range beside it; seed 0's own value is kept as `horizon_s0`, and the figures are drawn from
the *median-horizon* seed so the picture and the table describe one model.

That correction is why the architecture sweep (§3) reports only its held-out loss column as
solid. Its `horizon` column spans 245–453 at one seed each, against a 242–399 five-seed spread
for a single configuration — comparable, not wider, but close enough that the column ranks
nothing.

---

## 3. Topics 02–06 — the seven models

All seven satisfy the same two-method contract (`loss`, `forecast`), so one trainer and one
ruler judge all of them. Every model runs on `ode` and `sde`, 5 seeds, 3000 Adam iterations;
four of them also run on `sde015`.

| # | model | what it is |
| --- | --- | --- |
| 02 | `Predictor` | MLP one-step map **u**_{n+1} = F(**u**_n), MSE |
| 02s | — | the same, at widths 32–256, depths 1–4, SiLU/Tanh/ReLU (1 seed) |
| 03 | `RolloutPredictor` | k steps unrolled inside the loss, k ∈ {1, 4, 8, 16} |
| 04 | `LeadTimePredictor` | F(**u**_n, s) — the net conditioned on the horizon |
| 05 | `RecurrentPredictor` | RNN and LSTM |
| 05t | `TransformerPredictor` | causal self-attention over an 8-state window |
| 06 | `GaussianPredictor` | **u**_{n+1} ~ N(μ(**u**_n), Σ(**u**_n)), maximum likelihood |
| 06f | `FlowMatchingPredictor` | learned transport from N(0,I) to p(**u**_{n+1}\|**u**_n) |

The shared budget is in **iterations, not capacity**: the transformer carries 100k parameters
against the MLP's 17k. Read it as an architecture control, not a matched comparison.

Each class docstring in `l63/models.py` is the textbook account of that model. Those docstrings
are the source text for each model's first slide, and where a measurement contradicts the
textbook expectation the docstring now says so — see `GaussianPredictor` on the ODE.

### The three controls, which are the point

- **Topic 03 at k = 1** is topic 02's loss through the rollout code path. It agrees to float
  summation order, which is both the control passing and the suite's noise floor (§2).
- **Topics 05 and 05t** cannot help: the state is fully observed and the flow is Markov, so
  there is nothing in the history to carry. They measure how much apparent improvement is
  architecture and how much is noise.
- **Topics 06 and 06f mean-vs-sampled** — the same weights rolled out with the mean instead of
  a sample. One difference, which isolates whether *sampling* is what changes the behaviour.
  It runs on all five seeds, because it is read on `alive` and `climate`. `chaos` is excluded:
  the estimator switches sampling off itself, so the two rows would be identical by
  construction and could never differ.

---

## 4. Deliverable

`poster/poster_final.typ`, Typst, A0 portrait, one page: the system, five of the seven models,
the four rulers, and one scorecard carrying every ruler on every model at both ground truths.
`run/report.py` also writes per-topic `.typ` fragments for a slide deck, which is not part of
this repository.

**No number is typed into a `.typ` file by hand.** Every one is read from
`artifacts/summary.json`, written by `run/report.py`. Every figure is regenerated from the same
run. A stale figure and a stale number are therefore the same failure, and both are fixed by
re-running one command.

---

## 5. Stated, not hidden

- `climate` is marginals-only. The attractor grid and the Lorenz map cover joint structure by
  eye, with no number behind them.
- `climate` and `alive` cannot resolve a factor of two — measured, §2. Do not rank on them
  without checking the ranges overlap.
- `spread` is 0.00 for every deterministic model by construction — the correct reading.
- The architecture sweep is single-seed; its `chaos`, `climate` and `alive` columns are not
  comparable across rows, and `alive` is the wildest of the three.
- Topic 04's `forecast` is the s = 1 head iterated, and every ruler on that rung describes that
  map. It is a bad map — λ₁ ≈ −12, dead within 15 steps — and the rung's actual result is the
  direct-vs-autoregressive table, which is what the brief asks for. An earlier version chained
  blocks of s_max to give it a long rollout; that was dropped because it made `chaos` describe
  one map while `horizon`, `climate` and `alive` described another.
- One training budget for every model, in iterations. That is what makes it loss-against-loss,
  and it is also what makes every number a floor rather than a best case — most visibly for
  topic 04, which spends the same budget learning 16 maps, and for 06f, which is learning a
  field over an extra dimension.
- `b` was chosen at one value from a five-point sweep, with `sde015` as the control at the
  starter's value. "At what conditional width does the deterministic model start to lose" is
  now bracketed between 4.2 % and 16.9 % — bounded by this work, not answered by it.

---

## 6. Next

1. The knee between 4.2 % and 16.9 % conditional width — a b-sweep of the MLP/Gaussian pair.
2. More seeds on the SDE. `alive` there is bimodal at n = 5; 15–20 seeds would make it a
   proportion with an interval instead of a median of five coin flips.
3. Jacobian eigenvalues of the learned map at C±, which is the mechanism behind the SDE
   collapse rather than a description of it.
