"""Checkpoint helpers for PyTorch training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.training.state import TrainState


def output_root(workdir: str | None = None) -> Path:
    """Return the root output directory."""
    if workdir is None:
        return Path("runs").resolve()
    return Path(workdir).resolve()


def checkpoint_dir(workdir: str | None = None) -> Path:
    """Return the checkpoint directory for a workdir."""
    return output_root(workdir) / "checkpoints"


def _checkpoint_path(step: int, workdir: str | None = None) -> Path:
    return checkpoint_dir(workdir) / f"step_{step:08d}.pt"


def restore_checkpoint(
    state: TrainState,
    *,
    workdir: str | None = None,
    step: int | None = None,
) -> TrainState:
    """Restore the latest or requested checkpoint into a state object."""
    ckpt_dir = checkpoint_dir(workdir)
    if not ckpt_dir.exists():
        return state

    if step is None:
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        if not checkpoints:
            return state
        path = checkpoints[-1]
    else:
        path = _checkpoint_path(step, workdir)
        if not path.exists():
            return state

    payload = torch.load(path, map_location="cpu")
    state.load_state_dict(payload)
    return state


def save_checkpoint(
    state: TrainState,
    *,
    workdir: str | None = None,
    keep: int = 2,
    keep_every: int | None = None,
) -> Path:
    """Save a training checkpoint and prune older snapshots."""
    ckpt_dir = checkpoint_dir(workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(state.step, workdir)
    torch.save(state.state_dict(), path)

    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    protected: set[Path] = set()
    if keep_every is not None and keep_every > 0:
        for candidate in checkpoints:
            step_value = int(candidate.stem.split("_")[-1])
            if step_value % keep_every == 0:
                protected.add(candidate)

    disposable = [candidate for candidate in checkpoints if candidate not in protected]
    for candidate in disposable[:-keep]:
        candidate.unlink(missing_ok=True)
    return path


def save_params_ema_artifact(
    state: TrainState,
    *,
    workdir: str | None = None,
    kind: str,
    model_config: dict[str, Any] | None = None,
) -> Path:
    """Save the EMA weights as a standalone artifact."""
    out_dir = output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = out_dir / "ema_model.pt"
    torch.save(state.ema_model.state_dict(), state_dict_path)

    metadata = {
        "format": "torch.state_dict",
        "kind": kind,
        "backend": "torch",
        "ema_decay": state.ema_decay,
        "step": state.step,
        "path": "params_ema/ema_model.pt",
        "model_config": dict(model_config or {}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir
