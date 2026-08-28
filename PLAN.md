# Design notes — Lorenz 63

What this repository does, in the order it does it. `README.md` covers how to run it; this
covers why each part is built the way it is.

The question every topic answers:

> How many time steps can the model take before the result is too bad to use — and what counts
> as too bad?

---

## 0. Two rules for the measurements

**Every level a number is read against is either a property of the system or the integrator run
against itself**, and each is printed beside the number it judges. Where a ruler has its own
noise, `run/report.py` measures and prints that too.

Two constants are chosen rather than measured: `alive_frac`'s 0.25 / 5000, where the quantity
is bimodal so anything from about 0.05 to 0.6 gives the same partition, and `vpt`'s 0.4, the
published NRMSE convention, kept separate from the step counts so outside numbers remain
comparable.

**The truth row goes on top of every table, measured through the same code as every model**:
long true rollouts of the same count and length, through identical functions. Hardcoding it to
1.00 would assert the calibration instead of measuring it. Truth does not land on 1.00 for
`climate`, which is itself a result.

---

## 1. Topic 00 — ground truth, and the standard results

`l63/known.py` reproduces these before any model exists, one function each. If they do not come
out, the integrator is wrong and nothing downstream is worth running.

The two tables are different kinds of evidence.

*Closed forms evaluated. These check transcription and cannot fail.*

| result | measured | expected |
| --- | --- | --- |
| fixed points C± | ±8.485, 27 | ±8.485, 27 |
| Hopf threshold ρ_H | 24.737 | 24.74, and ρ = 28 is past it |
| ∇·f from the formula | −13.667 | −13.667 |

*Numerical. These can fail, and they are the integrator check.*

| result | measured | expected |
| --- | --- | --- |
| ∇·f as trace(Df) at 2000 attractor states | −13.666668, max deviation 0 | −13.667 everywhere |
| symmetry residual | 0 | 0 |
| spectrum λ₁, λ₂, λ₃ | 0.909, −0.006, −14.579 | 0.906, 0, −14.57 |
| **Σλ vs ∇·f** | **−13.676 vs −13.667** | an identity: the tightest of these checks |
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

`alive` answers one of the project's two named tests, so it is ranked and reported with its
range rather than used only as a pass/fail gate.

### The bars for "too large", measured per dataset

| dataset | same-cost solver usable | ground truth usable |
| --- | --- | --- |
| `ode` | 32 steps | 362 steps |
| `sde` (b = 0.6) | 26 steps | **23 steps** |
| `sde015` (b = 0.15) | 29 steps | 104 steps |

On the ODE the floor is discretisation error: Euler at Δt_int against Δt_int/2. The SDE needs a
different floor, because a stochastic system's irreducible uncertainty is
realisation-to-realisation and does not shrink as the step is refined; it reaches attractor
scale at step 23. Applying the deterministic curve to stochastic data gives 355 and makes every
SDE model appear to lose to the same-cost solver at step 1. The SDE bar is Euler–Maruyama at Δt
against Δt_int **driven by the same Brownian path**, so it measures method rather than noise.

On the SDE the models' horizons (27–49 steps) sit above the 23-step floor. A deterministic model
shadows the conditional mean, which stays near a given realisation longer than a second
realisation does. `horizon` therefore decides nothing on the SDE.

### Limits of the rulers

- **`spread` is undefined on the ODE.** Truth's ensemble spread there is identically zero, so
  the ratio is 0/0 and prints as "—"; a zero would read as a score. On the SDE a second
  independent truth ensemble scores 1.02 against the first, the ruler's own noise at 16
  members.
- **`climate` does not read 1.00 at truth and the anchor is measured.** Over five independent
  truth draws: 0.62 (0.56–1.89) on the ODE, 0.66 (0.57–1.00) on the SDE, 1.01 (0.78–1.42) on
  `sde015`. `climate_vs_truth` divides by the median anchor; a single draw is too noisy to
  serve as one.
- **`climate` and `alive` are coarse, and the k = 1 control quantifies it.** That control is
  the same model and the same loss differing only by float summation order, ~7×10⁻⁷ in the
  weights, and it moves `climate` by up to **80 %** and `alive` by up to **50 %**, against
  ≤ 7 % for `horizon` and ≤ 5.4 % for `chaos`. Neither resolves a factor of two.

**Every ruler takes five seeds.** Retrained on five seeds, one configuration gives horizons
spanning **45 % to 275 %** of their own median, and for two rungs seed 0 was the best of the
five. Every number is a median over five seeds with its range beside it; seed 0's own value
stays as `horizon_s0`, and figures are drawn from the median-horizon seed so the picture and
the table describe one model.

For the same reason the architecture sweep (§3) reports only its held-out loss column as
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
describe each model's construction, and where a measurement contradicts the textbook
expectation the docstring states that; see `GaussianPredictor` on the ODE.

### The three controls

- **Topic 03 at k = 1** is topic 02's loss through the rollout code path. It agrees to float
  summation order, which is both the control passing and the suite's noise floor (§2).
- **Topics 05 and 05t** have nothing to gain from memory: the state is fully observed and the
  flow is Markov, so the history carries no extra information. They measure how much apparent
  improvement comes from architecture and how much from noise.
- **Topics 06 and 06f, mean against sampled**: the same weights rolled out with the mean
  instead of a sample. That one difference isolates whether sampling changes the behaviour. It
  runs on all five seeds because it is read on `alive` and `climate`. `chaos` is excluded,
  since the estimator switches sampling off itself and the two rows would be identical by
  construction.

---

## 4. Deliverable

`poster/poster_final.typ`, Typst, A0 portrait, one page: the system, five of the seven models,
the four rulers, and one scorecard carrying every ruler on every model at both ground truths.
`run/report.py` also writes per-topic `.typ` fragments for a slide deck, which is not part of
this repository.

**No number is typed into a `.typ` file by hand.** Every one is read from
`artifacts/summary.json`, which `run/report.py` writes, and every figure is regenerated from
the same run. A stale figure and a stale number are then the same failure, and one command
fixes both.

---

## 5. Limitations

- `climate` uses the marginals only. The attractor grid and the Lorenz map cover joint
  structure by eye, with no number behind them.
- `climate` and `alive` cannot resolve a factor of two (§2). Ranking on them requires checking
  that the ranges do not overlap.
- `spread` is 0.00 for every deterministic model by construction.
- The architecture sweep is single-seed, so its `chaos`, `climate` and `alive` columns are not
  comparable across rows; `alive` varies most.
- Topic 04's `forecast` iterates the s = 1 head, and every ruler on that rung describes that
  map, which has λ₁ ≈ −12 and dies within 15 steps. The rung's result is the
  direct-against-autoregressive table instead. Chaining blocks of s_max would give it a long
  rollout, but would make `chaos` describe one map while `horizon`, `climate` and `alive`
  describe another.
- One training budget for every model, counted in iterations. That makes the comparison
  loss-against-loss, and it makes every number a floor rather than a best case, most visibly
  for topic 04, which spends the budget learning 16 maps, and for 06f, which learns a field
  over an extra dimension.
- `b` takes one value from a five-point sweep, with `sde015` as the control at the starter's
  value. The conditional width at which the deterministic model starts to lose is bracketed
  between 4.2 % and 16.9 %, not located.

---

## 6. Next

1. The knee between 4.2 % and 16.9 % conditional width — a b-sweep of the MLP/Gaussian pair.
2. More seeds on the SDE. `alive` there is bimodal at n = 5; 15–20 seeds would make it a
   proportion with an interval instead of a median of five coin flips.
3. Jacobian eigenvalues of the learned map at C±, which is the mechanism behind the SDE
   collapse rather than a description of it.
