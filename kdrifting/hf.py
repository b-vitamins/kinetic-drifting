"""Artifact loading helpers for local and Hugging Face model bundles."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Final, Protocol, cast

import torch

from kdrifting.env import runtime_paths

HF_PREFIX: Final[str] = "hf://"


class _SnapshotDownload(Protocol):
    def __call__(self, repo_id: str, **kwargs: object) -> str: ...


class _HubModule(Protocol):
    snapshot_download: _SnapshotDownload


def read_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json`` from an artifact directory."""
    return cast(
        dict[str, Any],
        json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8")),
    )


def load_torch_ema_state_dict(artifact_dir: Path) -> dict[str, torch.Tensor]:
    """Load the saved EMA state dict from an artifact directory."""
    return cast(
        dict[str, torch.Tensor],
        torch.load(artifact_dir / "ema_model.pt", map_location="cpu", weights_only=False),
    )


def _download_artifact(
    *,
    repo_id: str,
    kind: str,
    backend: str,
    model_id: str,
    output_root: str,
    prefix: str | None,
) -> Path:
    hub_module = cast(_HubModule, importlib.import_module("huggingface_hub"))
    snapshot_download = hub_module.snapshot_download

    local_root = Path(output_root).resolve() / "models" / kind / backend / model_id
    local_root.mkdir(parents=True, exist_ok=True)
    repo_root = f"models/{kind}/{backend}/{model_id}"
    path_in_repo = f"{prefix.strip('/')}/{repo_root}" if prefix else repo_root
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(local_root),
    )
    nested_root = local_root / path_in_repo
    return nested_root if nested_root.exists() else local_root


def _resolve_local_artifact_dir(path: str | Path) -> Path:
    artifact_path = Path(path).expanduser().resolve()
    if artifact_path.is_file():
        artifact_path = artifact_path.parent
    if (artifact_path / "metadata.json").is_file():
        return artifact_path
    params_ema_dir = artifact_path / "params_ema"
    if (params_ema_dir / "metadata.json").is_file():
        return params_ema_dir
    raise FileNotFoundError(f"Could not find a model artifact under {artifact_path}")


def resolve_artifact_dir(
    init_from: str,
    *,
    kind: str,
    repo_id: str | None = None,
    prefix: str | None = None,
    output_root: str | None = None,
) -> Path:
    """Resolve a local or ``hf://`` artifact reference to a materialized directory."""
    if init_from.startswith(HF_PREFIX):
        paths = runtime_paths()
        model_id = init_from.removeprefix(HF_PREFIX).strip()
        if not model_id:
            raise ValueError("Expected a model id after 'hf://'.")
        return _download_artifact(
            repo_id=repo_id or paths.hf_repo_id,
            kind=kind,
            backend="torch",
            model_id=model_id,
            output_root=output_root or paths.hf_root,
            prefix=prefix,
        )
    return _resolve_local_artifact_dir(init_from)


def load_mae_model(
    init_from: str,
    *,
    repo_id: str | None = None,
    prefix: str | None = None,
    output_root: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a MAE model from a local or ``hf://`` artifact."""
    from kdrifting.models.mae import mae_from_metadata

    artifact_dir = resolve_artifact_dir(
        init_from,
        kind="mae",
        repo_id=repo_id,
        prefix=prefix,
        output_root=output_root,
    )
    metadata = read_metadata(artifact_dir)
    model = mae_from_metadata(metadata)
    model.load_state_dict(load_torch_ema_state_dict(artifact_dir))
    model.eval()
    return model, metadata


def load_generator_model(
    init_from: str,
    *,
    repo_id: str | None = None,
    prefix: str | None = None,
    output_root: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a generator model from a local or ``hf://`` artifact."""
    from kdrifting.models.generator import build_generator_from_config

    artifact_dir = resolve_artifact_dir(
        init_from,
        kind="gen",
        repo_id=repo_id,
        prefix=prefix,
        output_root=output_root,
    )
    metadata = read_metadata(artifact_dir)
    model_config = dict(metadata.get("model_config", {}) or {})
    if not model_config:
        raise ValueError(
            f"Generator artifact is missing metadata.model_config: {artifact_dir}",
        )
    model = build_generator_from_config(model_config)
    model.load_state_dict(load_torch_ema_state_dict(artifact_dir))
    model.eval()
    return model, metadata
