"""Checkpoint helpers for PyTorch training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from kdrifting.distributed import barrier, is_rank_zero
from kdrifting.jax_artifacts import (
    convert_generator_jax_optimizer_tensors,
    convert_generator_jax_params,
    convert_mae_jax_optimizer_tensors,
    convert_mae_jax_params,
    load_jax_checkpoint_entry,
    resolve_jax_checkpoint_dir,
)
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet
from kdrifting.training.state import TrainState


class ExternalCheckpointSource(NamedTuple):
    """Resolved external training-checkpoint source."""

    backend: str
    path: Path


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


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def _resolve_external_torch_checkpoint_path(source: Path) -> Path | None:
    resolved = source.expanduser().resolve()
    if resolved.is_file() and resolved.suffix == ".pt" and resolved.stem.startswith("step_"):
        return resolved
    if resolved.name == "checkpoints" and resolved.is_dir():
        return _latest_checkpoint_path(resolved)
    checkpoints_dir = resolved / "checkpoints"
    if checkpoints_dir.is_dir():
        return _latest_checkpoint_path(checkpoints_dir)
    return None


def _resolve_external_jax_checkpoint_dir(source: Path) -> Path | None:
    resolved = source.expanduser().resolve()
    try:
        checkpoint_dir = resolve_jax_checkpoint_dir(resolved)
    except FileNotFoundError:
        return None
    if _latest_checkpoint_path(checkpoint_dir) is not None:
        return None
    return checkpoint_dir


def resolve_external_checkpoint_source(init_from: str | Path) -> ExternalCheckpointSource | None:
    """Resolve a local external training-checkpoint source."""
    source = Path(init_from).expanduser()
    torch_checkpoint = _resolve_external_torch_checkpoint_path(source)
    if torch_checkpoint is not None:
        return ExternalCheckpointSource(backend="torch", path=torch_checkpoint)

    jax_checkpoint_dir = _resolve_external_jax_checkpoint_dir(source)
    if jax_checkpoint_dir is not None:
        return ExternalCheckpointSource(backend="jax", path=jax_checkpoint_dir)

    return None


def _extract_optax_adam_state(opt_state: Any) -> tuple[int, Any, Any]:
    adam_state: Any
    if isinstance(opt_state, dict):
        adam_state = cast(dict[str, Any], opt_state).get("0")
    elif isinstance(opt_state, tuple):
        opt_state_tuple = cast(tuple[Any, ...], opt_state)
        adam_state = opt_state_tuple[0] if opt_state_tuple else None
    else:
        adam_state = None

    if adam_state is None:
        raise ValueError("Expected an Optax AdamW optimizer state under slot 0.")

    if isinstance(adam_state, dict):
        adam_state_dict = cast(dict[str, Any], adam_state)
        count = adam_state_dict.get("count")
        mu = adam_state_dict.get("mu")
        nu = adam_state_dict.get("nu")
    else:
        count = getattr(adam_state, "count", None)
        mu = getattr(adam_state, "mu", None)
        nu = getattr(adam_state, "nu", None)

    if count is None or mu is None or nu is None:
        raise ValueError("Unsupported Optax AdamW state layout in JAX checkpoint.")

    count_value = int(np.asarray(count).reshape(-1)[0])
    return count_value, mu, nu


def _load_jax_model_state_dicts(
    *,
    state: TrainState,
    kind: str,
    params: Any,
    ema_params: Any,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if kind == "mae":
        model = cast(MAEResNet, state.model)
        ema_model = cast(MAEResNet, state.ema_model)
        return (
            convert_mae_jax_params(params, model),
            convert_mae_jax_params(ema_params, ema_model),
        )
    if kind == "gen":
        model = cast(DitGen, state.model)
        ema_model = cast(DitGen, state.ema_model)
        return (
            convert_generator_jax_params(params, model),
            convert_generator_jax_params(ema_params, ema_model),
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def _restore_jax_adamw_state(
    optimizer: AdamW,
    model: nn.Module,
    *,
    kind: str,
    opt_state: Any,
) -> None:
    step_value, mu_tree, nu_tree = _extract_optax_adam_state(opt_state)
    if kind == "mae":
        exp_avg = convert_mae_jax_optimizer_tensors(mu_tree, cast(MAEResNet, model))
        exp_avg_sq = convert_mae_jax_optimizer_tensors(nu_tree, cast(MAEResNet, model))
    elif kind == "gen":
        exp_avg = convert_generator_jax_optimizer_tensors(mu_tree, cast(DitGen, model))
        exp_avg_sq = convert_generator_jax_optimizer_tensors(nu_tree, cast(DitGen, model))
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    optimizer.state.clear()
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step_value), device=parameter.device, dtype=torch.float32),
            "exp_avg": exp_avg[name].to(device=parameter.device),
            "exp_avg_sq": exp_avg_sq[name].to(device=parameter.device),
        }


def _restore_external_torch_checkpoint(
    state: TrainState,
    *,
    checkpoint_path: Path,
) -> TrainState:
    payload = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    state.load_state_dict(payload)
    return state


def _restore_external_jax_checkpoint(
    state: TrainState,
    *,
    checkpoint_dir: Path,
    kind: str,
) -> TrainState:
    payload, _metadata = load_jax_checkpoint_entry(checkpoint_dir)
    if not isinstance(state.optimizer, AdamW):
        raise TypeError("JAX checkpoint restore currently supports torch.optim.AdamW only.")

    params = payload.get("params")
    if params is None:
        raise ValueError(f"JAX checkpoint is missing params: {checkpoint_dir}")
    ema_params = payload.get("ema_params", params)
    model_state, ema_state = _load_jax_model_state_dicts(
        state=state,
        kind=kind,
        params=params,
        ema_params=ema_params,
    )
    state.model.load_state_dict(model_state)
    state.ema_model.load_state_dict(ema_state)

    opt_state = payload.get("opt_state")
    if opt_state is None:
        raise ValueError(f"JAX checkpoint is missing opt_state: {checkpoint_dir}")
    _restore_jax_adamw_state(
        state.optimizer,
        state.model,
        kind=kind,
        opt_state=opt_state,
    )

    state.ema_decay = float(payload.get("ema_decay", state.ema_decay))
    state.step = int(payload.get("step", 0))
    return state


def restore_external_checkpoint(
    state: TrainState,
    *,
    init_from: str,
    kind: str,
) -> TrainState:
    """Restore an external torch or JAX training checkpoint into a state object."""
    source = resolve_external_checkpoint_source(init_from)
    if source is None:
        return state
    if source.backend == "torch":
        return _restore_external_torch_checkpoint(state, checkpoint_path=source.path)
    if source.backend == "jax":
        return _restore_external_jax_checkpoint(state, checkpoint_dir=source.path, kind=kind)

    return state


def write_run_metadata(
    *,
    workdir: str | None = None,
    kind: str,
    model_config: dict[str, Any] | None = None,
    optimizer_config: dict[str, Any] | None = None,
    train_config: dict[str, Any] | None = None,
    step: int | None = None,
    ema_decay: float | None = None,
    source_init_from: str | None = None,
) -> Path:
    """Write self-describing metadata at the run root."""
    root = output_root(workdir)
    root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "format": "torch.run",
        "kind": kind,
        "backend": "torch",
        "model_config": dict(model_config or {}),
    }
    if optimizer_config:
        metadata["optimizer_config"] = dict(optimizer_config)
    if train_config:
        metadata["train_config"] = dict(train_config)
    if step is not None:
        metadata["step"] = int(step)
    if ema_decay is not None:
        metadata["ema_decay"] = float(ema_decay)
    if source_init_from:
        metadata["source_init_from"] = source_init_from
    path = root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def write_state_dict_artifact(
    state_dict: dict[str, Tensor],
    *,
    workdir: str | None = None,
    kind: str,
    model_config: dict[str, Any] | None = None,
    step: int | None = None,
    ema_decay: float | None = None,
    source_init_from: str | None = None,
    source_backend: str | None = None,
    source_format: str | None = None,
) -> Path:
    """Write a canonical torch EMA artifact under ``params_ema/``."""
    out_dir = output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = out_dir / "ema_model.pt"
    metadata: dict[str, Any] = {
        "format": "torch.state_dict",
        "kind": kind,
        "backend": "torch",
        "path": "params_ema/ema_model.pt",
        "model_config": dict(model_config or {}),
    }
    if step is not None:
        metadata["step"] = int(step)
    if ema_decay is not None:
        metadata["ema_decay"] = float(ema_decay)
    if source_init_from:
        metadata["source_init_from"] = source_init_from
    if source_backend:
        metadata["source_backend"] = source_backend
    if source_format:
        metadata["source_format"] = source_format

    barrier()
    if is_rank_zero():
        torch.save(state_dict, state_dict_path)
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    barrier()
    return out_dir


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
    return write_state_dict_artifact(
        state.ema_model.state_dict(),
        workdir=workdir,
        kind=kind,
        model_config=model_config,
        step=state.step,
        ema_decay=state.ema_decay,
    )
