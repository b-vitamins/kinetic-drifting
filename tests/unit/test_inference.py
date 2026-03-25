from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.inference import parse_labels, run_inference
from kdrifting.models.generator import DitGen


def _generator_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "generator_artifact"
    artifact_dir.mkdir()
    model_config: dict[str, Any] = {
        "cond_dim": 16,
        "num_classes": 11,
        "noise_classes": 4,
        "noise_coords": 2,
        "input_size": 8,
        "in_channels": 3,
        "patch_size": 2,
        "hidden_size": 16,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "out_channels": 3,
        "n_cls_tokens": 2,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
    }
    model = DitGen(**model_config)
    torch.save(model.state_dict(), artifact_dir / "ema_model.pt")
    metadata = {
        "kind": "gen",
        "backend": "torch",
        "model_config": model_config,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return artifact_dir


def test_parse_labels_repeats_and_truncates() -> None:
    assert parse_labels("", num_samples=3) == [0, 0, 0]
    assert parse_labels("1,2", num_samples=5) == [1, 2, 1, 2, 1]
    assert parse_labels("4,5,6", num_samples=2) == [4, 5]


def test_run_inference_saves_outputs(tmp_path: Path) -> None:
    artifact_dir = _generator_artifact(tmp_path)
    output_dir = tmp_path / "infer"
    json_copy = tmp_path / "report.json"

    result = run_inference(
        init_from=str(artifact_dir),
        workdir=str(output_dir),
        num_samples=4,
        labels="1,3",
        device="cpu",
        seed=5,
        json_out=str(json_copy),
    )

    assert result["num_samples"] == 4
    assert result["labels"] == [1, 3, 1, 3]
    assert Path(str(result["sample_tensor"])).is_file()
    assert Path(str(result["sample_grid"])).is_file()
    assert (output_dir / "result.json").is_file()
    assert json_copy.is_file()

    samples = torch.load(output_dir / "samples.pt", map_location="cpu", weights_only=False)
    assert tuple(samples.shape) == (4, 3, 8, 8)
