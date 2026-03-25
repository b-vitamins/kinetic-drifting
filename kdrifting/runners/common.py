"""Shared helpers for training and inference runners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from kdrifting.checkpointing import restore_checkpoint, restore_external_checkpoint
from kdrifting.distributed import world_size
from kdrifting.hf import load_generator_model, load_mae_model
from kdrifting.logging import WandbLogger
from kdrifting.training.state import TrainState

RawBatch = tuple[Tensor, Tensor]
PreparedBatch = dict[str, Tensor]
PreprocessFn = Callable[..., PreparedBatch]


def select_device(requested: str | torch.device | None = None) -> torch.device:
    """Select the runtime device, defaulting to the best available accelerator."""
    if isinstance(requested, torch.device):
        return requested
    if isinstance(requested, str) and requested.strip() and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device_tensor(value: Any, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    raise TypeError(f"Unsupported preprocessed value type: {type(value)!r}")


def prepare_preprocess_fn(preprocess_fn: PreprocessFn, device: torch.device) -> PreprocessFn:
    """Wrap a preprocess function so it always returns tensors on the target device."""

    def wrapped(batch: Any) -> PreparedBatch:
        prepared = preprocess_fn(batch)
        return {key: _to_device_tensor(value, device) for key, value in prepared.items()}

    return wrapped


def per_process_batch_size(total_batch_size: int) -> int:
    """Convert a global batch size to the current-process batch size."""
    processes = world_size()
    if total_batch_size % processes != 0:
        raise ValueError(
            f"Expected batch size {total_batch_size} to be divisible by world size {processes}.",
        )
    return total_batch_size // processes


def move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors onto the runtime device after restore."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def maybe_initialize_state(
    state: TrainState,
    *,
    kind: str,
    init_from: str,
    device: torch.device,
) -> TrainState:
    """Initialize a fresh training state from a local or HF artifact."""
    if not init_from or state.step != 0:
        return state
    if kind == "mae":
        model, _ = load_mae_model(init_from)
    elif kind == "gen":
        model, _ = load_generator_model(init_from)
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    state.model.load_state_dict(model.state_dict())
    state.ema_model.load_state_dict(model.state_dict())
    state.model.to(device)
    state.ema_model.to(device)
    return state


def create_or_restore_state(
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    ema_decay: float,
    workdir: str,
    init_from: str,
    kind: str,
    device: torch.device,
) -> TrainState:
    """Create a train state, restore checkpoints, and optionally seed from an artifact."""
    state = TrainState.create(model.to(device), optimizer, ema_decay=ema_decay)
    state = restore_checkpoint(state, workdir=workdir)
    if state.step == 0 and init_from:
        state = restore_external_checkpoint(state, init_from=init_from, kind=kind)
    move_optimizer_state_to_device(state.optimizer, device)
    return maybe_initialize_state(
        state,
        kind=kind,
        init_from=init_from,
        device=device,
    )


def average_metric_dicts(items: Sequence[tuple[dict[str, float], int]]) -> dict[str, float]:
    """Compute a weighted average over metric dictionaries."""
    totals: dict[str, float] = {}
    weights: dict[str, int] = {}
    for metric_dict, weight in items:
        for key, value in metric_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value) * weight
            weights[key] = weights.get(key, 0) + weight
    return {key: totals[key] / max(1, weights[key]) for key in totals}


def save_image_grid(images: Tensor, output_path: str | Path) -> Path:
    """Save an image batch as a single grid image."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image_array = WandbLogger.normalize_images(images)
    grid_image = WandbLogger.make_grid_image(image_array)
    grid_image.save(output, format="PNG")
    return output
