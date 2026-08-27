"""The Adam loop. Model-agnostic: it only calls `model.loss`."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import trange

from l63.models import ForecastModel


@dataclass
class History:
    train: list[float] = field(default_factory=list)     # one mini-batch each, so noisy
    val: list[float] = field(default_factory=list)       # the full validation set

    def __repr__(self) -> str:
        v = f'   val {self.val[-1]:.3e}' if self.val else ''
        return f'History({len(self.train)} iterations, final train {self.train[-1]:.3e}{v})'


def train(
        model: ForecastModel,
        xs: Tensor,
        val_xs: Tensor | None = None,
        n_iters: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        progress: bool = True,
) -> tuple[ForecastModel, History]:
    """Mini-batches are sampled over the trajectory axis; `xs` is (batch, time, 3).

    `progress=False` silences tqdm -- pass it in notebooks that are run headless, where a
    3000-iteration bar is 3000 lines of committed output.
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = History()

    model.train()
    pbar = trange(n_iters, disable=not progress)
    for _ in pbar:
        idx = torch.randint(0, xs.shape[0], (min(batch_size, xs.shape[0]),), device=xs.device)
        loss = model.loss(xs[idx])

        pbar.set_description(f"train {loss.item():.4f}")
        hist.train.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        if val_xs is not None:
            model.eval()
            with torch.no_grad():
                val_loss = model.loss(val_xs)
            hist.val.append(val_loss.item())
            pbar.set_description(f"train {loss.item():.4f} val {val_loss.item():.4f}")
            model.train()

    model.eval()
    return model, hist
