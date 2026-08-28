"""The contract every forecaster satisfies, and the seven models that satisfy it.

Training and evaluation only ever call `loss` and `forecast`, so an MLP, a recurrent net and
a Gaussian predictor are all trained and judged by the same code, and a comparison between
them holds everything but the model fixed.

One class per topic:

    02  Predictor            u_{n+1} = F(u_n),                MSE
    03  RolloutPredictor     the same F, unrolled k steps in the loss
    04  LeadTimePredictor    u_{n+s} = F(u_n, s)
    05  RecurrentPredictor   h_n = G(h_{n-1}, u_n),           RNN or LSTM
    05t TransformerPredictor causal self-attention over a window
    06  GaussianPredictor    u_{n+1} ~ N(mu(u_n), Sigma(u_n)), maximum likelihood
    06f FlowMatchingPredictor  learned transport from N(0,I) to p(u_{n+1} | u_n)

Each class docstring is the textbook account of that model: what object it approximates,
what its loss minimises, how it forecasts, and what it cannot do.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def mlp(in_dim: int = 3, out_dim: int = 3, hidden_dim: int = 128,
        n_hidden: int = 2, act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    """in -> (hidden, act) x n_hidden -> out."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


def as_history(x0: Tensor, k: int = 1) -> Tensor:
    """Normalise a forecast start to (n, k, 3): the last k states before the forecast.

    A single state (3,) and a batch of start states (n, 3) are accepted when k == 1; a model
    that needs warm-up must be handed a 3-D (n, k', 3) tensor, so the batch axis and the
    history axis cannot be confused for one another.
    """
    if x0.dim() == 1:
        x0 = x0[None]
    if x0.dim() == 2:
        x0 = x0[:, None]
    if x0.shape[1] < k:
        raise ValueError(f'model needs {k} warm-up states, got {x0.shape[1]}')
    return x0[:, x0.shape[1] - k:]


class ForecastModel(nn.Module, ABC):
    """`loss(xs)` for training, `forecast(x0, steps)` for evaluation. Nothing else."""

    history: int = 1        # warm-up states `forecast` needs; 1 for a Markov model

    # Two knobs, kept separate because collapsing them into one hides a case. `ar` picks the
    # loss: False is teacher forcing, where the net only ever sees true states, which is what
    # every rung except topic 03 does. True feeds the net its own output and compares every
    # intermediate to the truth, so the loss depends on the composed map, which a one-step
    # loss does not. `k` is how far it unrolls.
    #
    # Why not dispatch on `k > 1` alone: for the LSTM, AR at k = 1 differs from teacher
    # forcing, since it warms up on `history` states where teacher forcing carries the hidden
    # state across all 300. Dispatching on k alone would make that arm unreachable and would
    # silently retrain the existing model under a new name.
    #
    # Both are constructor kwargs, never `train()` arguments: `run/train_all.py` saves
    # `kwargs` into the checkpoint and rebuilds from it, so a training-time flag would put two
    # different models under one checkpoint name.
    #
    # Only 04 and 05 implement AR. Everything else leaves ar = False and is untouched.
    ar: bool = False
    k: int = 1

    @abstractmethod
    def loss(self, xs: Tensor) -> Tensor:
        """Training loss over a batch of trajectories, `xs` of shape (batch, time, 3)."""
        raise NotImplementedError

    @abstractmethod
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        """(n, k, 3) history -- or (n, 3) / (3,) when history == 1 -- to (n, steps + 1, 3).

        Element 0 of the output is the last state of `x0`, so a forecast from `xs[:, :k]`
        lines up with `xs[:, k - 1 : k + steps]`. A stochastic model draws a sample; call it
        repeatedly for an ensemble (`evaluate.ensemble`).
        """
        raise NotImplementedError

    # --- the model's own map, for the chaos ruler ----------------------------------------
    # `forecast` defines the dynamics, but Benettin needs a plain s -> s function on a
    # fixed-size state, and it has to be the same dynamics the other rulers roll out.
    # For a model that keeps nothing between steps, the state is the window and the default
    # below is exact. A model that carries something (an RNN's hidden state) has a bigger
    # state than its window and must override these two; otherwise `chaos` silently measures
    # a different system than `horizon`, `climate` and `alive` do.

    @property
    def map_dim(self) -> int:
        """Dimension of the state the model's map acts on."""
        return 3 * self.history

    def map_state(self, hist: Tensor) -> Tensor:
        """(n, history, 3) true warm-up states -> (n, map_dim) starting state of the map."""
        return hist.reshape(hist.shape[0], -1)

    def map_step(self, s: Tensor) -> Tensor:
        """(n, map_dim) -> (n, map_dim): one time step of the model's own dynamics."""
        w = s.reshape(s.shape[0], self.history, 3)
        nxt = self.forecast(w, 1)[:, 1]
        return torch.cat([w[:, 1:], nxt[:, None]], dim=1).reshape(s.shape[0], -1)

    def spec(self) -> list[tuple[str, str]]:
        """(label, shape) blocks for the architecture figure. Overridden per model."""
        return [('input', 'u  (3)'), ('net', repr(self)), ('output', '(3)')]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================ 02  one-step, MSE

class Predictor(ForecastModel):
    """MLP for the one-step map u_{n+1} = F(u_n), trained by mean squared error.

    THE OBJECT.  The Lorenz flow is deterministic and its vector field is smooth, so by
    Picard-Lindelof the state at any later time is a function of the state now, and nothing
    else. Fixing a time step dt therefore defines a single map

        Phi : R^3 -> R^3,        u_{n+1} = Phi(u_n)

    the flow map, which exists and is unique. The integrator approximates it with 100 Euler
    substeps. This model approximates the same object with one evaluation of a network:
    F_theta = an MLP 3 -> h -> ... -> 3. That is the idea, and the source of any speed-up:
    one matrix chain instead of a hundred substeps.

    THE DATA.  Consecutive pairs (u_n, u_{n+1}) from the reference trajectories, every pair,
    no gaps. The series is u_0, u_1, ..., u_N; the substeps between two of them belong to the
    integrator rather than being information withheld from the model.

    THE LOSS, AND ITS MINIMISER.  L(theta) = E || F_theta(u_n) - u_{n+1} ||^2.
    For a fixed input u, and over all functions F, write U' for the next state as a random
    variable:

        E[ ||F(u) - U'||^2 | u ]  =  ||F(u) - E[U'|u]||^2  +  Var(U'|u)

    Expanding, the cross term vanishes. Only the first term depends on F, so the minimiser is
    the conditional mean:

        F*(u) = E[ u_{n+1} | u_n = u ]

    This fact governs the whole ladder. On the ODE dataset the conditional is a point mass,
    so its mean is the flow map and MSE targets the right object; there is nothing for a
    probabilistic model to recover. On the SDE dataset the conditional has width, and the
    conditional mean is a smoothed, contracted version of the true map, so a deterministic
    model trained this way approximates the average of the dynamics rather than the dynamics.

    THE FORECAST.  Autoregressive: feed the output back in, so the n-step forecast is F_theta
    composed with itself n times.

    WHY THE ERROR GROWS.  With e_n = u-hat_n - u_n, to first order

        e_{n+1}  ~  DPhi(u_n) e_n  +  eps(u_n)

    where eps is the one-step model error. The homogeneous part grows like exp(lambda_1 n dt),
    which is the system's sensitivity rather than model error. Setting the tolerance a gives

        n_horizon  ~  ln(a / e_1) / (lambda_1 dt)

    which is Strogatz Ex. 9.3.1 counted in steps. The model only sets e_1, the size of its
    first mistake. Halving e_1 buys ln(2)/(lambda_1 dt) ~ 31 more steps and no more, however
    the halving was achieved. A better one-step fit therefore has sharply diminishing
    returns, which is a property of the system rather than of the network.
    """

    def __init__(self, hidden_dim: int = 128, n_hidden: int = 2,
                 act: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.net = mlp(3, 3, hidden_dim, n_hidden, act)
        self.cfg = dict(hidden_dim=hidden_dim, n_hidden=n_hidden, act=act.__name__)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    # --- model core ----------------------------------------------------------------------
    # MSE on every consecutive pair in the batch. Its minimiser is the conditional mean of
    # u_{n+1} given u_n; see the class docstring. The deterministic-vs-probabilistic
    # comparison rests on that result.
    def loss(self, xs: Tensor) -> Tensor:
        x_in = xs[:, :-1].reshape(-1, xs.shape[-1])
        x_target = xs[:, 1:].reshape(-1, xs.shape[-1])

        return torch.mean((self(x_in) - x_target) ** 2)
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        x = as_history(x0, 1)[:, 0]

        preds = [x]
        for _ in range(steps):
            x = self(x)
            preds.append(x)

        return torch.stack(preds, dim=1)

    def spec(self) -> list[tuple[str, str]]:
        h, n = self.cfg['hidden_dim'], self.cfg['n_hidden']
        return ([(r'$\mathbf{u}_n$', '3')] + [(self.cfg['act'], str(h))] * n
                + [(r'$\mathbf{u}_{n+1}$', '3')])


# ============================================================ 03  rollout loss

class RolloutPredictor(Predictor):
    """The same map F_theta, trained on k composed steps instead of one.

    THE DEFECT IT ADDRESSES.  One-step MSE only ever shows F_theta inputs that lie exactly on
    the true attractor. In a rollout the network is fed its own output, which drifts off that
    set almost immediately. So the map is trained on one distribution of inputs and evaluated
    on another, and nothing in the one-step loss says what F should do at a state slightly
    off the attractor. That gap is why a model with a tiny one-step error can still wander
    off, or fall into a place the true system never visits.

    THE LOSS.  Unroll k steps from a true state and compare every intermediate to the truth:

        L_k(theta) = E (1/k) sum_{j=1..k} || F_theta^(j)(u_n) - u_{n+j} ||^2

    where F^(j) is F composed j times. The 1/k keeps the loss on the same scale for every k,
    so one learning rate is right for all of them and k = 1 gives exactly topic 02's loss
    rather than k times it. Because the loss contains composed applications of F_theta, its
    gradient contains products of Jacobians DF_theta along the unrolled path. The loss
    therefore constrains both where the map sends a point and how it stretches the
    neighbourhood around it; a one-step loss constrains only the first.

    WHAT IS DIFFERENTIATED.  Backpropagation passes through the model, k times, and never
    through the solver: the targets u_{n+j} are fixed data, generated once. The integrator
    does not need to be differentiable.

    WHAT TO EXPECT.  At k = 1 this is topic 02's loss, the control that should reproduce
    it. As k grows, cost grows like k, and the gradient inherits the same exponential the
    forecast has: the product of Jacobians grows like exp(lambda_1 k dt), so beyond
    k dt ~ one Lyapunov time (~44 steps here) the gradient is dominated by the single most
    unstable direction and the optimisation gets harder, not more informative. The
    expectation is an improvement that saturates in k, not one that grows with it.

    SEED SENSITIVITY.  Training-seed variation distorts the rollout-k result more than any
    other number in this project: a two-point comparison of the extremes reads as a large
    effect that five seeds do not support. Measure it at five seeds with error bars, and do
    not quote a comparison of the two ends on its own.
    """

    def __init__(self, k: int = 4, hidden_dim: int = 128, n_hidden: int = 2,
                 act: type[nn.Module] = nn.SiLU):
        super().__init__(hidden_dim, n_hidden, act)
        self.k = k
        self.cfg['k'] = k

    # --- model core ----------------------------------------------------------------------
    # Unroll k steps, comparing every intermediate to the truth. The gradient runs back
    # through all k applications of the same weights, which is what makes the loss see the
    # composed map. `xs` is (batch, time, 3); windows are taken over the time axis.
    def loss(self, xs: Tensor) -> Tensor:
        k = min(self.k, xs.shape[1] - 1)

        x = xs[:, :-k]                             # (batch, T - k, 3) start states
        total = xs.new_zeros(())
        for j in range(1, k + 1):
            x = self(x)
            total = total + torch.mean((x - xs[:, j:xs.shape[1] - k + j]) ** 2)

        return total / k
    # --- end model core ------------------------------------------------------------------

    def spec(self) -> list[tuple[str, str]]:
        return super().spec()


# ============================================================ 04  lead time

class LeadTimePredictor(ForecastModel):
    """A family of maps indexed by the forecast horizon: u_{n+s} = F(u_n, s).

    THE OBJECT.  Instead of one map applied repeatedly, one network that has been handed the
    whole semigroup of flow maps as a function of two arguments. The true object is

        Phi_{s dt} = Phi_dt o Phi_dt o ... o Phi_dt   (s times)

    and F_theta(., s) approximates it directly, for each s = 1 ... S. The lead time enters as
    an extra input coordinate, so the network is R^4 -> R^3; it is scaled to O(1) so the
    state does not swamp it.

    THE LOSS.  Sample a start n and a lead time s together, uniformly:

        L(theta) = E_{n,s} || F_theta(u_n, s) - u_{n+s} ||^2

    THE QUESTION IT ANSWERS.  Does forecasting directly to lead time s beat taking s
    repeated single steps? The two fail differently. The direct forecast makes one
    approximation of a hard map: Phi_{s dt} stretches by exp(lambda_1 s dt), so for large s
    the target is a violently sensitive function and a smooth network must average over it.
    The autoregressive forecast makes s approximations of an easy map and lets them
    compound. There is no a priori winner, and which one wins says where the long-horizon
    error comes from: compounding, or the difficulty of the map itself.

    WHAT IT CANNOT DO.  A direct predictor forecasts only within its trained range. Asked for
    a horizon beyond S it has nothing trained to say.

    So `forecast` is the s = 1 head applied repeatedly. Chaining blocks of S (direct within a
    block, autoregressive between blocks) would give the rung a longer rollout, but then
    `chaos` would describe one map while `horizon`, `climate` and `alive` describe another,
    and the two diverge by a full attractor scale within 16 steps. Every ruler on this rung
    measures the same object, the s = 1 map iterated. The rung's result is `forecast_direct`
    against `forecast_autoregressive` at s <= S.
    """

    def __init__(self, s_max: int = 16, hidden_dim: int = 128, n_hidden: int = 2,
                 act: type[nn.Module] = nn.SiLU, ar: bool = False, k: int = 1):
        super().__init__()
        self.net = mlp(4, 3, hidden_dim, n_hidden, act)
        self.s_max = s_max
        self.ar, self.k = ar, k
        self.cfg = dict(s_max=s_max, hidden_dim=hidden_dim, n_hidden=n_hidden,
                        act=act.__name__, ar=ar, k=k)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """`s` in steps, broadcast to x's leading shape; scaled to O(1) before entering."""
        s = (s / self.s_max).expand(*x.shape[:-1], 1)
        return self.net(torch.cat([x, s], dim=-1))

    def loss(self, xs: Tensor) -> Tensor:
        return self.ar_loss(xs) if self.ar else self.tf_loss(xs)

    # --- model core ----------------------------------------------------------------------
    # One lead time per mini-batch, drawn uniformly from 1..s_max. Drawing one s per batch
    # rather than one per sample keeps the windowing a single slice and costs nothing:
    # over many iterations every s is seen equally often either way.
    def tf_loss(self, xs: Tensor) -> Tensor:
        s_max = min(self.s_max, xs.shape[1] - 1)
        s = int(torch.randint(1, s_max + 1, ()))

        x_in = xs[:, :-s]
        x_target = xs[:, s:]
        pred = self(x_in, torch.tensor(float(s), device=xs.device))

        return torch.mean((pred - x_target) ** 2)
    # --- end model core ------------------------------------------------------------------

    # --- model core ----------------------------------------------------------------------
    # AR: the same head, composed k times. Unrolling only the s = 1 head would leave heads
    # s = 2..s_max without any gradient, leave `forecast_direct` untrained, and remove the
    # rung's direct-vs-autoregressive result. Unrolling the drawn head instead trains every
    # head to compose, which is the semigroup property the true flow has:
    # Phi_s o Phi_s = Phi_2s. It reduces to `tf_loss` exactly at k = 1.
    #
    # Targets sit at n+s, n+2s, ..., n+ks, so a window needs k*s + 1 states, and that caps
    # s_max here rather than the flat `xs.shape[1] - 1` the one-step loss can use.
    def ar_loss(self, xs: Tensor) -> Tensor:
        t, k = xs.shape[1], self.k
        s_max = min(self.s_max, (t - 1) // k)
        s = int(torch.randint(1, s_max + 1, ()))
        lead = torch.tensor(float(s), device=xs.device)

        x = xs[:, :t - k * s]                       # (batch, T - k*s, 3) start states
        total = xs.new_zeros(())
        for j in range(1, k + 1):
            x = self(x, lead)
            total = total + torch.mean((x - xs[:, j * s:t - (k - j) * s]) ** 2)

        return total / k
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast_direct(self, x0: Tensor, steps: int) -> Tensor:
        """Direct prediction at every lead time 1..steps. Requires steps <= s_max."""
        x = as_history(x0, 1)[:, 0]
        if steps > self.s_max:
            raise ValueError(f'direct forecast is only trained to s_max={self.s_max}')

        def lead(s: int) -> Tensor:
            return torch.tensor(float(s), dtype=x.dtype, device=x.device)

        preds = [x] + [self(x, lead(s)) for s in range(1, steps + 1)]
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def forecast_autoregressive(self, x0: Tensor, steps: int) -> Tensor:
        """The same network asked only for s = 1, applied `steps` times.

        This is the other half of the rung's comparison. Against `forecast_direct` the
        network, the weights and the training are identical; only the way the horizon is
        reached differs.
        """
        x = as_history(x0, 1)[:, 0]
        one = torch.tensor(1., dtype=x.dtype, device=x.device)

        preds = [x]
        for _ in range(steps):
            x = self(x, one)
            preds.append(x)

        return torch.stack(preds, dim=1)

    forecast = forecast_autoregressive         # one map for every ruler; see the docstring

    def spec(self) -> list[tuple[str, str]]:
        h, n = self.cfg['hidden_dim'], self.cfg['n_hidden']
        return ([(r'$\mathbf{u}_n,\ s$', '3+1')] + [(self.cfg['act'], str(h))] * n
                + [(r'$\mathbf{u}_{n+s}$', '3')])


# ============================================================ 05  recurrent

class RecurrentPredictor(ForecastModel):
    """An RNN or LSTM over a window of past states.

    THE OBJECT.  A map carrying an internal state:

        h_n = G_theta(h_{n-1}, u_n),        u-hat_{n+1} = W h_n

    For the RNN, G is one affine layer and a tanh. The LSTM adds three gates (input, forget,
    output) whose purpose is to make the derivative dh_n / dh_{n-m} decay slowly in m, so
    that a gradient can survive being carried back over a long window. In a plain RNN that
    derivative is a product of m Jacobians and generically vanishes or explodes.

    WHAT IT IS DOING HERE.  The Lorenz state is fully observed and the flow is Markov:
    u_{n+1} depends on u_n alone. There is therefore nothing in the history for h to carry
    that is not already in the current state. In theory a recurrent model has no information
    advantage over the MLP.

    It is here for two reasons. The brief names it. And it is the control on the Markov
    property: if the LSTM beats the MLP by more than the seed spread, then either the flow is
    not Markov at this time step, which would be a finding about the data pipeline, or the
    extra parameters are doing ordinary capacity work. The architecture sweep in topic 02
    separates those two explanations, so the two are read together.

    WARM-UP AND `history`.  `history` is the number of states `forecast` is handed before it
    starts predicting. Training is teacher forcing over the whole trajectory in the
    mini-batch (300 states here), not over an 8-state window; and once a forecast starts, the
    hidden state is carried for the entire rollout, so the model's context is unbounded
    rather than 8. Those are three different lengths and they are not interchangeable.

    THE MAP IS BIGGER THAN THE WINDOW.  Because h is carried, the iterated object is
    (u, h, c) -> (u', h', c') rather than u -> u'. `map_state` / `map_step` below implement
    that on a 3 + 2*n_layers*hidden dimensional state, so the `chaos` ruler measures the same
    dynamics the rollout has.
    """

    def __init__(self, cell: str = 'lstm', hidden_dim: int = 64, n_layers: int = 1,
                 history: int = 8, ar: bool = False, k: int = 1, n_starts: int | None = None):
        super().__init__()
        rnn = {'rnn': nn.RNN, 'lstm': nn.LSTM}[cell.lower()]
        self.rnn = rnn(3, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3)
        self.history = history
        self.ar, self.k = ar, k
        self.n_starts = n_starts
        self.cfg = dict(cell=cell, hidden_dim=hidden_dim, n_layers=n_layers, history=history,
                        ar=ar, k=k, n_starts=n_starts)

    def forward(self, xs: Tensor, state=None) -> tuple[Tensor, object]:
        out, state = self.rnn(xs, state)
        return self.head(out), state

    def loss(self, xs: Tensor) -> Tensor:
        return self.ar_loss(xs) if self.ar else self.tf_loss(xs)

    # --- model core ----------------------------------------------------------------------
    # Teacher forcing: the whole true sequence goes in, and every output position predicts
    # the next true state. One pass over the window trains every position at once.
    def tf_loss(self, xs: Tensor) -> Tensor:
        pred, _ = self(xs[:, :-1])

        return torch.mean((pred - xs[:, 1:]) ** 2)
    # --- end model core ------------------------------------------------------------------

    # --- model core ----------------------------------------------------------------------
    # AR: warm up on `history` true states, then free-run k steps carrying (h, c), which is
    # the shape `forecast` has, truncated to k. This is rollout / multi-step training, the
    # standard way a deterministic forecaster is trained through its own composition (the
    # curriculum GraphCast uses). Scheduled sampling (Bengio et al. 2015) is the older
    # RNN-specific alternative, substituting the model's own output with probability eps
    # inside a full-sequence pass; this rung uses the k-step unroll instead, so that it is
    # comparable to the MLP's rollout loss and to the lead-time model.
    #
    # Every valid start, by default, which is the fairness condition against the other rungs.
    # `RolloutPredictor` unrolls from `xs[:, :-k]`, all T-k of them, and so does
    # `LeadTimePredictor.ar_loss`. Sampling a subset of windows instead (`n_starts=32`) puts
    # the LSTM on 32 supervised targets per trajectory against teacher forcing's 300, a 9x
    # handicap unrelated to AR that the rulers then report as AR making the model worse. Keep
    # the same starts and the same budget as the other rungs, or the rungs are not
    # comparable.
    #
    # `nn.LSTM` returns only the final (h, c), never one per position, so the warm-up cannot
    # be shared across starts the way the MLP shares its single forward: each window is
    # replayed from scratch. That is what makes this rung ~8x its teacher-forced cost, and it
    # follows from the API rather than from the method.
    def ar_loss(self, xs: Tensor) -> Tensor:
        t, h, k = xs.shape[1], self.history, self.k
        hi = t - h - k + 1                          # a window of h states AND k targets fit
        idx = (torch.arange(hi) if self.n_starts is None
               else torch.randint(0, hi, (min(self.n_starts, hi),)))

        win = torch.stack([xs[:, i:i + h] for i in idx], 1).reshape(-1, h, 3)
        tgt = torch.stack([xs[:, i + h:i + h + k] for i in idx], 1).reshape(-1, k, 3)

        # the same warm-up `forecast` does: keep the recurrent state, drop all but the last
        # prediction, since those positions are history rather than forecast
        out, state = self(win)
        x = out[:, -1:]

        preds = [x]
        for _ in range(k - 1):
            x, state = self(x, state)
            preds.append(x)

        return torch.mean((torch.cat(preds, dim=1) - tgt) ** 2)
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        h = as_history(x0, self.history)

        # warm up on the k given states; keep the recurrent state, drop all but the last
        # prediction, since those positions are history rather than forecast
        out, state = self(h)
        x = out[:, -1:]

        preds = [h[:, -1], x[:, 0]]
        for _ in range(steps - 1):
            x, state = self(x, state)
            preds.append(x[:, 0])

        return torch.stack(preds, dim=1)

    # --- the map: (u, h, c), because `forecast` carries the recurrent state -------------
    # The default in ForecastModel would re-run the warm-up on every step and so measure a
    # different, memoryless system. On 05_lstm_ode_s0 the two disagree by 4.3 Lorenz units at
    # step 50 and by 34.7 at step 200, more than the width of the attractor.

    @property
    def _n_gates(self) -> int:
        return 2 if isinstance(self.rnn, nn.LSTM) else 1

    @property
    def map_dim(self) -> int:
        c = self.cfg
        return 3 + self._n_gates * c['n_layers'] * c['hidden_dim']

    def _pack(self, x: Tensor, state) -> Tensor:
        parts = state if isinstance(state, tuple) else (state,)
        return torch.cat([x] + [p.permute(1, 0, 2).reshape(x.shape[0], -1) for p in parts],
                         dim=-1)

    def _unpack(self, s: Tensor):
        c, n = self.cfg, s.shape[0]
        x, rest = s[:, :3], s[:, 3:]
        state = tuple(p.reshape(n, c['n_layers'], c['hidden_dim']).permute(1, 0, 2).contiguous()
                      for p in rest.chunk(self._n_gates, dim=-1))
        return x, (state if len(state) > 1 else state[0])

    def map_state(self, hist: Tensor) -> Tensor:
        out, state = self(hist)                    # the same warm-up `forecast` does
        return self._pack(out[:, -1], state)

    def map_step(self, s: Tensor) -> Tensor:
        x, state = self._unpack(s)
        out, state = self(x[:, None], state)       # the same recurrence `forecast` iterates
        return self._pack(out[:, 0], state)

    def spec(self) -> list[tuple[str, str]]:
        c = self.cfg
        return [(r'$\mathbf{u}_{n-%d..n}$' % (c['history'] - 1), f'{c["history"]}×3'),
                (c['cell'].upper(), f'{c["hidden_dim"]} × {c["n_layers"]}'),
                ('linear', '3'), (r'$\mathbf{u}_{n+1}$', '3')]


# ============================================================ 06  Gaussian

class GaussianPredictor(ForecastModel):
    """The next state as a distribution: u_{n+1} | u_n ~ N(mu(u_n), Sigma(u_n)).

    THE OBJECT.  One network with two heads: a mean mu_theta: R^3 -> R^3 and a log standard
    deviation, so Sigma_theta = diag(sigma^2) is positive by construction. This is the
    smallest step from predicting a point to predicting a distribution, and the first rung
    that predicts a distribution at all.

    THE LOSS.  Negative log likelihood. For a diagonal Gaussian, per component,

        -log p  =  1/2 sum_i [ (u_i - mu_i)^2 / sigma_i^2  +  log sigma_i^2 ]  +  const

    The expression is a squared error divided by a learned variance, plus a penalty for a
    large variance. Where the model predicts well, sigma shrinks and the first term
    dominates; where it does not, sigma widens and the log term is paid instead. That is the
    difference from MSE: the predicted uncertainty can grow, and the log term prices it
    against the residual incurred.

    ITS RELATION TO TOPIC 02.  If sigma were held fixed and constant, the first term is MSE
    up to a constant factor and the second is a constant: minimising the NLL would then be
    minimising MSE, and mu_theta would converge to the same conditional mean. So none of the
    gain comes from the change of loss alone. It comes from the sigma head, and from sampling
    in the forecast.

    THE FORECAST.  Draw u-hat_{n+1} ~ N(mu(u-hat_n), Sigma(u-hat_n)) and feed the sample
    back. Each call gives a different trajectory; an ensemble is repeated calls. Sampling is
    not baked into the contract because a deterministic model called twice returns the same
    answer by design.

    WHAT HAPPENS ON EACH DATASET.  The ODE does not follow the theory.
      ODE. The conditional is a point mass, so sigma should go to zero. It does not: over five
      seeds sigma is 7.8x the model's own one-step residual (range 1.3-17.0). The predicted
      uncertainty is roughly an order of magnitude larger than the residual, and it costs
      `climate`: 18.6 against the MLP's 1.9, on a dataset where there is nothing to be
      uncertain about. That spurious width does not decide whether the rollout survives:
      mean-only and sampled both score alive 1.00 in the median. (Seed 0 alone shows mean-only
      collapsing to 0.00, which is why the control runs on all five seeds; one seed of `alive`
      is not a measurement.) `sigma_over_residual` is banked for this.
      SDE. The conditional has real width, sigma/residual = 0.90, and the model is calibrated
      (spread 1.00). The controlled test: the same weights rolled out with the mean instead of
      a sample go from climate 5.2 to 28.6 and from spread 1.00 to 0.00. Sampling recovers the
      statistics on the dataset that has statistics to recover, and the ODE row is the control
      on that comparison.

    THE LIMITATION.  A diagonal Gaussian cannot represent a skewed or multi-modal
    conditional, and it assumes the three components are conditionally independent. How mild
    that restriction is depends on how Gaussian the true conditional is at this time step,
    which topic 00's conditional-width experiment measures directly.
    """

    LOG_SIGMA_MIN, LOG_SIGMA_MAX = -12., 3.     # keeps the NLL finite early in training

    def __init__(self, hidden_dim: int = 128, n_hidden: int = 2,
                 act: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.net = mlp(3, 6, hidden_dim, n_hidden, act)      # 3 for mu, 3 for log sigma
        self.sampling = True                                 # flipped by the controlled test
        self.cfg = dict(hidden_dim=hidden_dim, n_hidden=n_hidden, act=act.__name__)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, log_sigma = self.net(x).chunk(2, dim=-1)
        return mu, log_sigma.clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)

    # --- model core ----------------------------------------------------------------------
    # Gaussian NLL, dropping the additive constant 3/2 log(2 pi): it shifts the loss but not
    # its gradient, and leaving it out keeps the number comparable with an MSE.
    def loss(self, xs: Tensor) -> Tensor:
        x_in = xs[:, :-1].reshape(-1, xs.shape[-1])
        x_target = xs[:, 1:].reshape(-1, xs.shape[-1])

        mu, log_sigma = self(x_in)
        nll = 0.5 * (((x_target - mu) / log_sigma.exp()) ** 2 + 2 * log_sigma)

        return nll.sum(-1).mean()
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        x = as_history(x0, 1)[:, 0]

        preds = [x]
        for _ in range(steps):
            mu, log_sigma = self(x)
            x = mu + log_sigma.exp() * torch.randn_like(mu) if self.sampling else mu
            preds.append(x)

        return torch.stack(preds, dim=1)

    def spec(self) -> list[tuple[str, str]]:
        h, n = self.cfg['hidden_dim'], self.cfg['n_hidden']
        return ([(r'$\mathbf{u}_n$', '3')] + [(self.cfg['act'], str(h))] * n
                + [(r'$\mathbf{\mu}\ |\ \log\mathbf{\sigma}$', '3+3'),
                   (r'sample $\mathbf{u}_{n+1}$', '3')])


# ============================================================ 05t  transformer

class TransformerPredictor(ForecastModel):
    """A transformer over a window of past states.

    THE OBJECT.  The same next-state map as every other rung, but computed by attention rather
    than by recurrence. Each state is embedded to a vector of width d, a learned position code
    is added, and a stack of self-attention blocks mixes the window:

        h = Encoder(W u_(n-k+1..n) + p),      u-hat_(n+1) = V h_n

    Attention replaces the hidden state: instead of carrying information forward step by
    step, every position looks at every earlier position directly. This is the property that
    made transformers useful for long sequences: the path length between two positions is 1
    rather than their separation, so a gradient does not have to survive a product of
    Jacobians the way it does in an RNN.

    THE MASK.  Causal, strictly lower-triangular. Position n may attend to positions <= n and
    no further. Without it the model would see u_(n+1) while predicting u_(n+1), and the
    training loss would go to zero while the forecast stayed useless.

    TRAINING.  Teacher forcing, exactly as for the recurrent models: the true sequence goes in
    and every position predicts the next true state, so one pass trains every position at
    once. The trajectory is cut into chunks first, because attention costs O(T^2) in the
    sequence length: over the full 300-step trajectory that is 90 000 pairs per head per
    layer, and it buys nothing, since at forecast time the model only ever sees `history`
    states. Chunking makes the cost linear in T at the price of one target per chunk
    boundary, so `chunk` is picked to divide the trajectory; at 75 the model trains on 296 of
    the 300 available pairs. A `chunk` that does not divide it drops the remainder: at
    chunk = 76, 301 // 76 is 3, so the last 73 states go unused and the model sees 225 pairs,
    75 % of what every other rung sees.

    WHAT TO EXPECT.  The same argument as for the RNN and the LSTM applies, and more
    strongly. The Lorenz state is fully observed and the flow is Markov: u_(n+1) depends on
    u_n alone. There is nothing in the window for attention to find. A transformer here is a
    large, expensive model solving a problem that has no sequence structure, and the
    expectation is that it matches the MLP at best.

    The brief names it. Together with the RNN and the LSTM it also puts three independent
    numbers on how much of any apparent improvement here is architecture rather than noise.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 dim_ff: int = 128, history: int = 8, chunk: int = 75):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.0,
                                           batch_first=True, norm_first=True)
        # enable_nested_tensor is incompatible with norm_first and only warns; off explicitly
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 3)
        self.history = history
        self.chunk = chunk
        self.cfg = dict(d_model=d_model, nhead=nhead, n_layers=n_layers, dim_ff=dim_ff,
                        history=history, chunk=chunk)

    def forward(self, xs: Tensor) -> Tensor:
        """(b, t, 3) -> (b, t, 3): position i predicts state i+1, attending to <= i only."""
        t = xs.shape[1]
        h = self.embed(xs) + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=xs.device), diagonal=1)
        return self.head(self.enc(h, mask=mask))

    # --- model core ----------------------------------------------------------------------
    # Teacher forcing under a causal mask, on chunks of the trajectory. The chunking is a
    # cost decision rather than a modelling one: it makes attention O(T * chunk) instead of
    # O(T^2). It costs one target per chunk boundary, so `chunk` must divide the trajectory;
    # check `n * c` against `t` before changing it.
    def loss(self, xs: Tensor) -> Tensor:
        b, t, _ = xs.shape
        c = min(self.chunk, t)
        n = max(1, t // c)

        w = xs[:, :n * c].reshape(b * n, c, 3)
        return torch.mean((self(w[:, :-1]) - w[:, 1:]) ** 2)
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        w = as_history(x0, self.history)

        preds = [w[:, -1]]
        for _ in range(steps):
            nxt = self(w)[:, -1]
            preds.append(nxt)
            w = torch.cat([w[:, 1:], nxt[:, None]], dim=1)

        return torch.stack(preds, dim=1)

    def spec(self) -> list[tuple[str, str]]:
        c = self.cfg
        return [(r'$\mathbf{u}_{n-7..n}$', f'{c["history"]}×3'),
                ('embed + pos', str(c['d_model'])),
                (f'causal attention ×{c["n_layers"]}', f'{c["nhead"]} heads'),
                ('linear', '3'), (r'$\mathbf{u}_{n+1}$', '3')]


# ============================================================ 06f  flow matching

class FlowMatchingPredictor(ForecastModel):
    """A generative model of p(u_{n+1} | u_n), learned as a transport from noise.

    THE OBJECT.  The Gaussian rung assumes the conditional has a shape (a diagonal normal)
    and fits its two moments. This rung assumes nothing about the shape. It learns to carry
    a simple base distribution onto the true conditional, by learning the velocity field of
    the flow that does the carrying:

        v_theta(x, t, u_n) : R^3 x [0,1] x R^3 -> R^3

    Sampling means starting at x ~ N(0, I) and integrating dx/dt = v_theta(x, t, u_n) from
    t = 0 to t = 1. Whatever distribution comes out the other end is the model's conditional.

    THE LOSS, AND WHY IT NEEDS NO SOLVER.  The target velocity field is not known: it depends
    on the distribution being learned. Conditional flow matching works around that. Pick a
    base sample x0 and a data sample x1 and connect them by a straight line,

        x_t = (1 - t) x0 + t x1,        d x_t / d t  =  x1 - x0

    Then regress the network on that per-pair velocity,

        L(theta) = E_(t, x0, x1) || v_theta(x_t, t, u_n) - (x1 - x0) ||^2

    with t uniform on [0,1]. Although each individual target is the velocity of one straight
    line, the minimiser is the marginal velocity field whose flow transports the base
    distribution onto the data distribution, so a loss made of per-pair regressions learns
    the transport. Nothing is backpropagated through a solver, and no likelihood is evaluated.

    THE FORECAST.  Draw a base sample, integrate the learned field with `n_steps` Euler steps,
    and feed the result back in. So one forecast step costs `n_steps` network evaluations
    rather than one, which makes this the most expensive model here to roll out; the cost
    buys not having to assume the conditional's shape.

    WHAT TO EXPECT.
      ODE. The conditional is a point mass. A model that can represent any shape must here
      represent a delta, which a finite-step Euler integration of a smooth field cannot do
      exactly. It should land near the deterministic models, not beat them.
      SDE. A real distribution to learn, and no shape assumption to be wrong about. This is
      where it should at least match the Gaussian, and where it could beat it if the true
      conditional is not Gaussian.

    THE CAVEAT.  It is not tuned: same width, same optimiser, same 3000 iterations as every
    other rung, on a strictly harder problem, since it has to learn a field over an extra
    time dimension. A poor number here means "not with this budget" rather than "flow
    matching does not work".

    The `sampling` switch freezes the base sample to zero, making the map deterministic so a
    Lyapunov exponent can be measured. That exponent describes the sensitivity of one
    particular path through the flow rather than of the ensemble, and `chaos` reports it as
    such.
    """

    def __init__(self, hidden_dim: int = 128, n_hidden: int = 3,
                 act: type[nn.Module] = nn.SiLU, n_steps: int = 16):
        super().__init__()
        self.net = mlp(3 + 3 + 1, 3, hidden_dim, n_hidden, act)   # x, condition, t
        self.n_steps = n_steps
        self.sampling = True
        self.cfg = dict(hidden_dim=hidden_dim, n_hidden=n_hidden, act=act.__name__,
                        n_steps=n_steps)

    def velocity(self, x: Tensor, u: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([x, u, t], dim=-1))

    # --- model core ----------------------------------------------------------------------
    # Conditional flow matching with the straight-line (rectified) path. One base sample and
    # one data sample per pair, joined by a line; the network regresses that line's constant
    # velocity at a random time along it.
    def loss(self, xs: Tensor) -> Tensor:
        u = xs[:, :-1].reshape(-1, 3)
        x1 = xs[:, 1:].reshape(-1, 3)

        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=xs.device)
        xt = (1 - t) * x0 + t * x1

        return torch.mean((self.velocity(xt, u, t) - (x1 - x0)) ** 2)
    # --- end model core ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(self, x0: Tensor, steps: int) -> Tensor:
        u = as_history(x0, 1)[:, 0]
        dt = 1.0 / self.n_steps

        preds = [u]
        for _ in range(steps):
            x = torch.randn_like(u) if self.sampling else torch.zeros_like(u)
            for i in range(self.n_steps):
                t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
                x = x + self.velocity(x, u, t) * dt
            u = x
            preds.append(u)

        return torch.stack(preds, dim=1)

    def spec(self) -> list[tuple[str, str]]:
        h, n = self.cfg['hidden_dim'], self.cfg['n_hidden']
        return ([(r'$\mathbf{x},\ \mathbf{u}_n,\ t$', '3+3+1')]
                + [(self.cfg['act'], str(h))] * n
                + [(r'velocity $\mathbf{v}_\theta$', '3'),
                   (f'{self.cfg["n_steps"]} Euler steps', r'$\mathbf{u}_{n+1}$')])
