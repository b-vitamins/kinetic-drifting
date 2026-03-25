"""Artifact loading helpers for local and Hugging Face model bundles."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Final, Protocol, cast

import torch

from kdrifting.env import runtime_paths
from kdrifting.jax_artifacts import (
    convert_generator_jax_params,
    convert_mae_jax_params,
    load_jax_init_entry,
)

HF_PREFIX: Final[str] = "hf://"


class _SnapshotDownload(Protocol):
    def __call__(self, repo_id: str, **kwargs: object) -> str: ...


class _HubModule(Protocol):
    snapshot_download: _SnapshotDownload


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def read_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json`` from an artifact directory."""
    return _load_json(artifact_dir / "metadata.json")


def load_torch_ema_state_dict(artifact_dir: Path) -> dict[str, torch.Tensor]:
    """Load the saved EMA state dict from an artifact directory."""
    return cast(
        dict[str, torch.Tensor],
        torch.load(artifact_dir / "ema_model.pt", map_location="cpu", weights_only=False),
    )


def _looks_like_torch_artifact(artifact_dir: Path) -> bool:
    return (artifact_dir / "ema_model.pt").is_file()


def _looks_like_torch_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".pt" and path.stem.startswith("step_")


def _looks_like_direct_jax_artifact(artifact_dir: Path) -> bool:
    return any(
        (
            (artifact_dir / "ema_params.msgpack").is_file(),
            (artifact_dir / "ema_model.msgpack").is_file(),
        ),
    )


def _looks_like_jax_artifact(artifact_dir: Path) -> bool:
    return any(
        (
            _looks_like_direct_jax_artifact(artifact_dir),
            artifact_dir.name == "checkpoints",
            (artifact_dir / "checkpoints").is_dir(),
        ),
    )


def _checkpoint_step_key(path: Path) -> int:
    return int(path.stem.removeprefix("step_"))


def _latest_torch_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step_key)
    if not checkpoints:
        return None
    return checkpoints[-1]


def _torch_checkpoint_path(source: Path) -> Path | None:
    if _looks_like_torch_checkpoint_file(source):
        return source
    if source.name == "checkpoints" and source.is_dir():
        return _latest_torch_checkpoint_path(source)
    return None


def _metadata_candidates(artifact_dir: Path) -> tuple[Path, ...]:
    candidates = [artifact_dir / "metadata.json"]
    if artifact_dir.name == "checkpoints":
        candidates.extend(
            [
                artifact_dir.parent / "params_ema" / "metadata.json",
                artifact_dir.parent / "metadata.json",
            ],
        )
    return tuple(candidates)


def read_metadata_if_present(artifact_dir: Path) -> dict[str, Any]:
    """Read metadata from an artifact dir or its checkpoint siblings when available."""
    for candidate in _metadata_candidates(artifact_dir):
        if candidate.is_file():
            return _load_json(candidate)
    return {}


def _load_torch_checkpoint_entry(
    checkpoint_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    state_dict = payload.get("ema_model", payload.get("model"))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Torch checkpoint does not contain a model state dict: {checkpoint_path}")

    metadata = read_metadata_if_present(checkpoint_path.parent)
    if "step" not in metadata and "step" in payload:
        metadata["step"] = int(payload["step"])
    if "ema_decay" not in metadata and "ema_decay" in payload:
        metadata["ema_decay"] = float(payload["ema_decay"])
    metadata.setdefault("backend", "torch")
    metadata.setdefault("format", "torch.checkpoint")
    return cast(dict[str, torch.Tensor], state_dict), metadata


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
    if _looks_like_torch_artifact(artifact_path) or _looks_like_direct_jax_artifact(artifact_path):
        return artifact_path
    params_ema_dir = artifact_path / "params_ema"
    if _looks_like_torch_artifact(params_ema_dir) or _looks_like_direct_jax_artifact(
        params_ema_dir
    ):
        return params_ema_dir
    checkpoints_dir = artifact_path / "checkpoints"
    if checkpoints_dir.is_dir():
        return checkpoints_dir
    if (params_ema_dir / "metadata.json").is_file():
        return params_ema_dir
    if artifact_path.name == "checkpoints" and artifact_path.is_dir():
        return artifact_path
    raise FileNotFoundError(f"Could not find a model artifact under {artifact_path}")


def resolve_artifact_dir(
    init_from: str,
    *,
    kind: str,
    backend: str | None = None,
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
        backends = (backend,) if backend is not None else ("torch", "jax")
        for candidate_backend in backends:
            artifact_dir = _download_artifact(
                repo_id=repo_id or paths.hf_repo_id,
                kind=kind,
                backend=candidate_backend,
                model_id=model_id,
                output_root=output_root or paths.hf_root,
                prefix=prefix,
            )
            if _looks_like_torch_artifact(artifact_dir) or _looks_like_jax_artifact(artifact_dir):
                return artifact_dir
        raise FileNotFoundError(f"Could not resolve a {kind!r} artifact for {init_from}")
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

    local_path = Path(init_from).expanduser()
    if not init_from.startswith(HF_PREFIX):
        checkpoint_path = _torch_checkpoint_path(local_path.resolve())
        if checkpoint_path is not None:
            state_dict, metadata = _load_torch_checkpoint_entry(checkpoint_path)
            if not metadata.get("model_config"):
                raise ValueError(
                    f"Torch checkpoint is missing metadata.model_config: {checkpoint_path}",
                )
            model = mae_from_metadata(metadata)
            model.load_state_dict(state_dict)
            model.eval()
            return model, metadata

    artifact_dir = resolve_artifact_dir(
        init_from,
        kind="mae",
        repo_id=repo_id,
        prefix=prefix,
        output_root=output_root,
    )
    backend = "torch"
    metadata = read_metadata_if_present(artifact_dir)
    if metadata.get("backend") == "jax" or (
        _looks_like_jax_artifact(artifact_dir) and not _looks_like_torch_artifact(artifact_dir)
    ):
        backend = "jax"

    if backend == "jax":
        params, metadata = load_jax_init_entry(artifact_dir)
        if not metadata.get("model_config"):
            raise ValueError(
                f"MAE JAX artifact is missing metadata.model_config: {artifact_dir}",
            )
        model = mae_from_metadata(metadata)
        model.load_state_dict(convert_mae_jax_params(params, model))
    else:
        if _looks_like_torch_artifact(artifact_dir):
            metadata = read_metadata(artifact_dir)
            state_dict = load_torch_ema_state_dict(artifact_dir)
        else:
            checkpoint_path = _latest_torch_checkpoint_path(artifact_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"Could not find a torch checkpoint under {artifact_dir}")
            state_dict, metadata = _load_torch_checkpoint_entry(checkpoint_path)
        model = mae_from_metadata(metadata)
        model.load_state_dict(state_dict)
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

    local_path = Path(init_from).expanduser()
    if not init_from.startswith(HF_PREFIX):
        checkpoint_path = _torch_checkpoint_path(local_path.resolve())
        if checkpoint_path is not None:
            params, metadata = _load_torch_checkpoint_entry(checkpoint_path)
            model_config = dict(metadata.get("model_config", {}) or {})
            if not model_config:
                raise ValueError(
                    f"Torch checkpoint is missing metadata.model_config: {checkpoint_path}",
                )
            model = build_generator_from_config(model_config)
            model.load_state_dict(params)
            model.eval()
            return model, metadata

    artifact_dir = resolve_artifact_dir(
        init_from,
        kind="gen",
        repo_id=repo_id,
        prefix=prefix,
        output_root=output_root,
    )
    metadata = read_metadata_if_present(artifact_dir)
    backend = "torch"
    if metadata.get("backend") == "jax" or (
        _looks_like_jax_artifact(artifact_dir) and not _looks_like_torch_artifact(artifact_dir)
    ):
        backend = "jax"

    if backend == "jax":
        params, metadata = load_jax_init_entry(artifact_dir)
    else:
        if _looks_like_torch_artifact(artifact_dir):
            params = load_torch_ema_state_dict(artifact_dir)
            metadata = read_metadata(artifact_dir)
        else:
            checkpoint_path = _latest_torch_checkpoint_path(artifact_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"Could not find a torch checkpoint under {artifact_dir}")
            params, metadata = _load_torch_checkpoint_entry(checkpoint_path)

    model_config = dict(metadata.get("model_config", {}) or {})
    if not model_config:
        raise ValueError(
            f"Generator artifact is missing metadata.model_config: {artifact_dir}",
        )
    model = build_generator_from_config(model_config)
    if backend == "jax":
        model.load_state_dict(convert_generator_jax_params(params, model))
    else:
        model.load_state_dict(cast(dict[str, torch.Tensor], params))
    model.eval()
    return model, metadata
