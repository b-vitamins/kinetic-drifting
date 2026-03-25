"""MAE training utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from kdrifting.training.state import TrainState

BatchDict = dict[str, Tensor]
PreprocessFn = Callable[[BatchDict], BatchDict]


def input_dict(batch: BatchDict) -> dict[str, Tensor]:
    """Convert a preprocessed batch into MAE model kwargs."""
    return {"x": batch["images"], "labels": batch["labels"]}


def _step_generator(base_seed: int, step: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(base_seed) + int(step))
    return generator


def train_step(
    state: TrainState,
    batch: BatchDict,
    *,
    base_seed: int,
    forward_dict: dict[str, Any],
    learning_rate_fn: Callable[[int], float],
    preprocess_fn: PreprocessFn,
    max_grad_norm: float = 2.0,
) -> tuple[TrainState, dict[str, float]]:
    """Run one MAE optimization step."""
    batch = preprocess_fn(batch)
    generator = _step_generator(base_seed, state.step, batch["images"].device)
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    loss, metrics = state.model(
        **input_dict(batch),
        **forward_dict,
        generator=generator,
    )
    loss_mean = loss.mean()
    torch.autograd.backward(loss_mean)
    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm)
    state.optimizer.step()
    state.increment_step()
    state.update_ema()

    lr = float(learning_rate_fn(state.step - 1))
    metrics_out = {
        key: float(value.detach().float().mean().item()) for key, value in metrics.items()
    }
    metrics_out["loss"] = float(loss_mean.detach().item())
    metrics_out["g_norm"] = float(grad_norm.detach().item())
    metrics_out["lr"] = lr
    return state, metrics_out


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    batch: BatchDict,
    *,
    base_seed: int,
    step: int,
    forward_dict: dict[str, Any],
    preprocess_fn: PreprocessFn,
) -> dict[str, float]:
    """Run one MAE evaluation step."""
    batch = preprocess_fn(batch)
    generator = _step_generator(base_seed, step, batch["images"].device)
    model.eval()
    loss, metrics = model(
        **input_dict(batch),
        **forward_dict,
        generator=generator,
    )
    out = {key: float(value.detach().float().mean().item()) for key, value in metrics.items()}
    out["loss"] = float(loss.detach().float().mean().item())
    return out
