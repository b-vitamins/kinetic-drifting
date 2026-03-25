"""Checkpoint helpers for PyTorch training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.distributed import barrier, is_rank_zero
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


def _resolve_checkpoint_path(
    *,
    workdir: str | None = None,
    step: int | None = None,
) -> Path | None:
    ckpt_dir = checkpoint_dir(workdir)
    if not ckpt_dir.exists():
        return None

    if step is None:
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        if not checkpoints:
            return None
        return checkpoints[-1]

    path = _checkpoint_path(step, workdir)
    return path if path.exists() else None


def _load_checkpoint_payload(
    *,
    workdir: str | None = None,
    step: int | None = None,
) -> dict[str, Any] | None:
    path = _resolve_checkpoint_path(workdir=workdir, step=step)
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_checkpoint(
    state: TrainState,
    *,
    workdir: str | None = None,
    step: int | None = None,
) -> TrainState:
    """Restore the latest or requested checkpoint into a state object."""
    payload = _load_checkpoint_payload(workdir=workdir, step=step)
    if payload is None:
        return state
    state.load_state_dict(payload)
    return state


def restore_checkpoint_extra_state(
    *,
    workdir: str | None = None,
    step: int | None = None,
) -> dict[str, Any] | None:
    """Restore runner-specific extra state saved alongside a checkpoint."""
    payload = _load_checkpoint_payload(workdir=workdir, step=step)
    if payload is None:
        return None
    extra_state = payload.get("extra_state")
    if extra_state is None:
        return None
    return dict(extra_state)


def save_checkpoint(
    state: TrainState,
    *,
    workdir: str | None = None,
    keep: int = 2,
    keep_every: int | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint and prune older snapshots."""
    ckpt_dir = checkpoint_dir(workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(state.step, workdir)
    payload = dict(state.state_dict())
    if extra_state is not None:
        payload["extra_state"] = dict(extra_state)

    barrier()
    if is_rank_zero():
        torch.save(payload, path)

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
    barrier()
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
    barrier()
    if is_rank_zero():
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
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    barrier()
    return out_dir
