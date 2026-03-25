"""Conversion helpers for native torch artifacts and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import torch

from kdrifting.checkpointing import (
    resolve_external_checkpoint_source,
    restore_external_checkpoint,
    save_checkpoint,
    save_params_ema_artifact,
    write_run_metadata,
    write_state_dict_artifact,
)
from kdrifting.config import export_model_config
from kdrifting.hf import load_generator_model, load_mae_model
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet
from kdrifting.runners.common import move_optimizer_state_to_device, select_device
from kdrifting.schedules import create_learning_rate_fn
from kdrifting.training.state import TrainState

ModelKind = Literal["mae", "gen"]


def _normalize_kind(kind: str) -> ModelKind:
    if kind == "mae":
        return "mae"
    if kind in {"gen", "generator"}:
        return "gen"
    raise ValueError(f"Unsupported model kind: {kind}")


def _build_model_and_optimizer(
    config: dict[str, Any],
    *,
    kind: ModelKind,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    dataset_config = dict(config.get("dataset", {}) or {})
    model_config = dict(config.get("model", {}) or {})
    optimizer_config = dict(config.get("optimizer", {}) or {})
    lr_schedule = dict(optimizer_config.get("lr_schedule", {}) or {})

    if "num_classes" not in dataset_config:
        raise ValueError("Expected config.dataset.num_classes for checkpoint export.")

    num_classes = int(dataset_config["num_classes"])
    if kind == "mae":
        model = MAEResNet(num_classes=num_classes, **model_config)
    else:
        model = DitGen(num_classes=num_classes, **model_config)
    model.to(device)

    learning_rate_fn = create_learning_rate_fn(**lr_schedule)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate_fn(0)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        betas=(
            float(optimizer_config.get("adam_b1", 0.9)),
            float(optimizer_config.get("adam_b2", 0.999)),
        ),
    )
    return model, optimizer


def _result_dict(
    *,
    init_from: str,
    kind: ModelKind,
    workdir: str,
    artifact_dir: Path,
    metadata: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "init_from": init_from,
        "kind": kind,
        "workdir": str(Path(workdir).resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "metadata_path": str((artifact_dir / "metadata.json").resolve()),
        "step": int(metadata.get("step", 0)),
    }
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(checkpoint_path.resolve())
    return result


def export_model_artifact(
    *,
    init_from: str,
    kind: str,
    workdir: str,
    repo_id: str | None = None,
    prefix: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Export a model source into a canonical torch EMA artifact."""
    normalized_kind = _normalize_kind(kind)
    if normalized_kind == "mae":
        model, metadata = load_mae_model(
            init_from,
            repo_id=repo_id,
            prefix=prefix,
            output_root=output_root,
        )
    else:
        model, metadata = load_generator_model(
            init_from,
            repo_id=repo_id,
            prefix=prefix,
            output_root=output_root,
        )

    artifact_dir = write_state_dict_artifact(
        cast(dict[str, torch.Tensor], model.state_dict()),
        workdir=workdir,
        kind=normalized_kind,
        model_config=dict(metadata.get("model_config", {}) or {}),
        step=int(metadata.get("step", 0)),
        ema_decay=float(metadata["ema_decay"]) if "ema_decay" in metadata else None,
        source_init_from=init_from,
        source_backend=cast(str | None, metadata.get("backend")),
        source_format=cast(str | None, metadata.get("format")),
    )
    write_run_metadata(
        workdir=workdir,
        kind=normalized_kind,
        model_config=dict(metadata.get("model_config", {}) or {}),
        step=int(metadata.get("step", 0)),
        ema_decay=float(metadata["ema_decay"]) if "ema_decay" in metadata else None,
        source_init_from=init_from,
    )
    return _result_dict(
        init_from=init_from,
        kind=normalized_kind,
        workdir=workdir,
        artifact_dir=artifact_dir,
        metadata=metadata,
    )


def export_training_checkpoint(
    *,
    init_from: str,
    config: dict[str, Any],
    kind: str,
    workdir: str,
    device: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Export an external torch or JAX training checkpoint into native torch format."""
    normalized_kind = _normalize_kind(kind)
    source = resolve_external_checkpoint_source(init_from)
    if source is None:
        raise FileNotFoundError(f"Could not resolve an external checkpoint under {init_from}")

    runtime_device = select_device(device)
    model, optimizer = _build_model_and_optimizer(
        config,
        kind=normalized_kind,
        device=runtime_device,
    )
    train_config = dict(config.get("train", {}) or {})
    state = TrainState.create(
        model,
        optimizer,
        ema_decay=float(train_config.get("ema_decay", 0.999)),
    )
    state = restore_external_checkpoint(state, init_from=init_from, kind=normalized_kind)
    move_optimizer_state_to_device(state.optimizer, torch.device("cpu"))
    model_config = export_model_config(config)

    checkpoint_path = save_checkpoint(state, workdir=workdir, keep=1)
    artifact_dir = save_params_ema_artifact(
        state,
        workdir=workdir,
        kind=normalized_kind,
        model_config=model_config,
    )
    optimizer_config = dict(config.get("optimizer", {}) or {})
    write_run_metadata(
        workdir=workdir,
        kind=normalized_kind,
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        step=state.step,
        ema_decay=state.ema_decay,
        source_init_from=init_from,
    )
    metadata = {
        "step": state.step,
        "ema_decay": state.ema_decay,
        "backend": "torch",
        "format": "torch.checkpoint",
    }
    if source.backend:
        metadata["source_backend"] = source.backend
    return _result_dict(
        init_from=init_from,
        kind=normalized_kind,
        workdir=workdir,
        artifact_dir=artifact_dir,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
    )
