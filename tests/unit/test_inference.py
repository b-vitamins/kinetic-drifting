from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from pytest import MonkeyPatch

from kdrifting.inference import parse_labels, run_fid_evaluation, run_inference
from kdrifting.models.generator import DitGen

UPSTREAM_ROOT = Path("/home/b/projects/drifting")
FLAX_CHECKPOINTS = cast(Any, importlib.import_module("flax.training.checkpoints"))
JAX = cast(Any, importlib.import_module("jax"))


def _import_upstream(module_name: str) -> Any:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module_name)


def _write_jax_checkpoint_run(
    root: Path,
    *,
    metadata: dict[str, Any],
    params: Any,
    step: int = 7,
) -> Path:
    checkpoints_dir = root / "checkpoints"
    params_ema_dir = root / "params_ema"
    checkpoints_dir.mkdir(parents=True)
    params_ema_dir.mkdir()
    (params_ema_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    FLAX_CHECKPOINTS.save_checkpoint(
        ckpt_dir=str(checkpoints_dir),
        target={"params": params},
        step=step,
        overwrite=True,
    )
    return root


def _generator_model_config() -> dict[str, Any]:
    return {
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


def _expected_upstream_samples(
    *,
    params: Any,
    model_config: dict[str, Any],
    labels: torch.Tensor,
    noise_x: torch.Tensor,
    noise_labels: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    upstream_generator = _import_upstream("models.generator")
    upstream_dataset = _import_upstream("dataset.dataset")
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)

    upstream_model = upstream_generator.build_generator_from_config(model_config)
    labels_np = labels.detach().cpu().to(dtype=torch.int32).numpy()
    noise_x_np = noise_x.detach().cpu().numpy()
    noise_labels_np = noise_labels.detach().cpu().to(dtype=torch.int32).numpy()
    cond = upstream_model.apply(
        {"params": params},
        labels_np,
        cfg_scale,
        noise_labels_np,
        method=upstream_model.c_cfg_noise_to_cond,
    )
    raw_samples = upstream_model.apply(
        {"params": params},
        noise_x_np,
        cond,
        deterministic=True,
        method=upstream_model.generate_image,
    )
    postprocess_fn = upstream_dataset.get_postprocess_fn(
        use_aug=False,
        use_latent=False,
        use_cache=False,
    )
    expected = np.array(postprocess_fn(raw_samples), copy=True)
    return torch.tensor(expected, dtype=torch.float32)


def _generator_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "generator_artifact"
    artifact_dir.mkdir()
    model_config = _generator_model_config()
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


def test_run_inference_matches_upstream_jax_checkpoint_numerically(tmp_path: Path) -> None:
    model_config = _generator_model_config()
    metadata = {
        "kind": "gen",
        "backend": "jax",
        "model_config": model_config,
    }
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_generator = _import_upstream("models.generator")
    upstream_model = upstream_generator.build_generator_from_config(model_config)
    variables = upstream_model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **upstream_model.dummy_input(),
    )
    run_root = _write_jax_checkpoint_run(
        tmp_path / "generator_run",
        metadata=metadata,
        params=variables["params"],
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    noise_x = torch.linspace(-1.0, 1.0, steps=2 * 8 * 8 * 3, dtype=torch.float32).reshape(
        2,
        8,
        8,
        3,
    )
    noise_labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    result = run_inference(
        init_from=str(run_root),
        workdir=str(tmp_path / "infer"),
        cfg_scale=1.5,
        num_samples=2,
        labels="1,2",
        device="cpu",
        noise_x=noise_x,
        noise_labels=noise_labels,
    )
    expected = _expected_upstream_samples(
        params=variables["params"],
        model_config=model_config,
        labels=labels,
        noise_x=noise_x,
        noise_labels=noise_labels,
        cfg_scale=1.5,
    )
    actual = torch.load(result["sample_tensor"], map_location="cpu", weights_only=False)

    assert result["metadata"]["backend"] == "jax"
    assert torch.allclose(actual, expected, atol=5e-4, rtol=1e-4)


def test_run_fid_evaluation_wires_eval_stack(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    artifact_dir = _generator_artifact(tmp_path)
    output_dir = tmp_path / "fid"
    json_copy = tmp_path / "fid_report.json"
    captured: dict[str, Any] = {}

    class FakeLogger:
        def set_logging(self, **kwargs: Any) -> None:
            captured["logger_kwargs"] = kwargs

        def log_dict(self, values: dict[str, Any]) -> None:
            captured["logged_dict"] = values

        def log_image(self, name: str, images: Any) -> None:
            captured["logged_image"] = (name, images)

        def finish(self) -> None:
            captured["finished"] = True

    def fake_create_imagenet_split(
        **kwargs: Any,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], Any, Any]:
        captured["split_kwargs"] = kwargs
        loader = [(torch.zeros(2, 3, 8, 8), torch.tensor([1, 2], dtype=torch.int64))]

        def preprocess(batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
            images, labels = batch
            return {"images": images, "labels": labels}

        def postprocess(images: torch.Tensor) -> torch.Tensor:
            return images

        return loader, preprocess, postprocess

    def fake_evaluate_fid(**kwargs: Any) -> dict[str, float]:
        captured["eval_kwargs"] = kwargs
        batch = kwargs["eval_loader"][0]
        generated = kwargs["gen_func"](batch, 0)
        assert tuple(generated.shape) == (2, 3, 8, 8)
        return {"fid": 1.25, "isc_mean": 2.0, "isc_std": 0.1}

    import kdrifting.inference as inference_module

    monkeypatch.setattr(inference_module, "WandbLogger", FakeLogger)
    monkeypatch.setattr(inference_module, "create_imagenet_split", fake_create_imagenet_split)
    monkeypatch.setattr(inference_module, "evaluate_fid", fake_evaluate_fid)

    result = run_fid_evaluation(
        init_from=str(artifact_dir),
        workdir=str(output_dir),
        cfg_scale=1.5,
        num_samples=128,
        eval_batch_size=32,
        device="cpu",
        seed=7,
        json_out=str(json_copy),
        use_wandb=False,
    )

    assert result["fid"] == 1.25
    assert result["isc_mean"] == 2.0
    assert result["eval_batch_size"] == 32
    assert captured["split_kwargs"]["resolution"] == 256
    assert captured["eval_kwargs"]["log_prefix"] == "cfg_1.5"
    assert captured["logger_kwargs"]["use_wandb"] is False
    assert captured["finished"] is True
    assert (output_dir / "result.json").is_file()
    assert json_copy.is_file()


def test_run_fid_evaluation_gen_func_matches_upstream_jax_checkpoint(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    model_config = _generator_model_config()
    metadata = {
        "kind": "gen",
        "backend": "jax",
        "model_config": model_config,
    }
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_generator = _import_upstream("models.generator")
    upstream_model = upstream_generator.build_generator_from_config(model_config)
    variables = upstream_model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **upstream_model.dummy_input(),
    )
    run_root = _write_jax_checkpoint_run(
        tmp_path / "generator_run",
        metadata=metadata,
        params=variables["params"],
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    noise_x = torch.linspace(-1.0, 1.0, steps=2 * 8 * 8 * 3, dtype=torch.float32).reshape(
        2,
        8,
        8,
        3,
    )
    noise_labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    expected = _expected_upstream_samples(
        params=variables["params"],
        model_config=model_config,
        labels=labels,
        noise_x=noise_x,
        noise_labels=noise_labels,
        cfg_scale=1.5,
    )

    class FakeLogger:
        def set_logging(self, **kwargs: Any) -> None:
            del kwargs

        def finish(self) -> None:
            pass

    def fake_create_imagenet_split(
        **kwargs: Any,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], Any, Any]:
        del kwargs
        loader = [(torch.zeros(2, 3, 8, 8), labels)]

        def preprocess(batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
            images, batch_labels = batch
            return {"images": images, "labels": batch_labels}

        def postprocess(images: torch.Tensor) -> torch.Tensor:
            return images

        return loader, preprocess, postprocess

    import kdrifting.inference as inference_module

    original_generate_step = inference_module.generate_step

    def deterministic_generate_step(
        model: torch.nn.Module,
        *,
        labels: torch.Tensor,
        postprocess_fn: Any,
        base_seed: int,
        step: int,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        return original_generate_step(
            model,
            labels=labels,
            postprocess_fn=postprocess_fn,
            base_seed=base_seed,
            step=step,
            cfg_scale=cfg_scale,
            noise_x=noise_x,
            noise_labels=noise_labels,
        )

    def fake_evaluate_fid(**kwargs: Any) -> dict[str, float]:
        batch = kwargs["eval_loader"][0]
        generated = kwargs["gen_func"](batch, 0)
        assert torch.allclose(generated.cpu(), expected, atol=5e-4, rtol=1e-4)
        return {"fid": 1.25, "isc_mean": 2.0, "isc_std": 0.1}

    monkeypatch.setattr(inference_module, "WandbLogger", FakeLogger)
    monkeypatch.setattr(inference_module, "create_imagenet_split", fake_create_imagenet_split)
    monkeypatch.setattr(inference_module, "evaluate_fid", fake_evaluate_fid)
    monkeypatch.setattr(inference_module, "generate_step", deterministic_generate_step)

    result = run_fid_evaluation(
        init_from=str(run_root),
        workdir=str(tmp_path / "fid"),
        cfg_scale=1.5,
        num_samples=128,
        eval_batch_size=32,
        device="cpu",
        use_wandb=False,
    )

    assert result["metadata"]["backend"] == "jax"
    assert result["fid"] == 1.25
