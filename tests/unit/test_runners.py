from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from pytest import CaptureFixture, MonkeyPatch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from kdrifting.checkpointing import restore_checkpoint_extra_state
from kdrifting.logging import NullLogger
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet, build_activation_function
from kdrifting.runners.generator import train_generator
from kdrifting.runners.mae import train_mae
from kdrifting.schedules import create_learning_rate_fn
from kdrifting.training.state import TrainState


def _loader(
    batch_size: int,
    *,
    image_size: int,
    num_items: int = 8,
) -> DataLoader[tuple[Tensor, Tensor]]:
    images = torch.randn(num_items, image_size, image_size, 3)
    labels = torch.arange(num_items, dtype=torch.int64) % 5
    dataset = TensorDataset(images, labels)
    return cast(
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
    )


def _preprocess(batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
    images, labels = batch
    return {"images": images, "labels": labels}


def _postprocess(images: Tensor) -> Tensor:
    return ((images + 1.0) / 2.0).permute(0, 3, 1, 2).contiguous()


def _fake_eval(**_: object) -> dict[str, float]:
    return {"fid": 1.25, "isc_mean": 2.0, "isc_std": 0.1}


def _empty_feature_apply(*_args: object, **_kwargs: object) -> dict[str, Tensor]:
    return {}


def test_train_mae_runner_saves_checkpoints_and_artifacts(tmp_path: Path) -> None:
    model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=8,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)

    train_mae(
        model=model,
        optimizer=optimizer,
        logger=NullLogger(),
        eval_loader=_loader(2, image_size=32),
        train_loader=_loader(2, image_size=32),
        learning_rate_fn=schedule,
        preprocess_fn=_preprocess,
        model_config={
            "base_channels": 8,
            "patch_size": 2,
            "layers": [1, 1, 1, 1],
            "in_channels": 3,
        },
        workdir=str(tmp_path),
        device=torch.device("cpu"),
        total_steps=2,
        save_per_step=1,
        eval_per_step=1,
        eval_samples=4,
        ema_decay=0.9,
        seed=7,
        keep_every=1,
        keep_last=2,
        forward_dict={"mask_ratio_min": 0.5, "mask_ratio_max": 0.5, "lambda_cls": 0.1},
        eval_forward_dict={"mask_ratio_min": 0.5, "mask_ratio_max": 0.5, "lambda_cls": 0.1},
    )

    metadata = json.loads((tmp_path / "params_ema" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "mae"
    assert (tmp_path / "checkpoints" / "step_00000002.pt").is_file()


def test_train_generator_runner_saves_checkpoints_and_artifacts(tmp_path: Path) -> None:
    model = DitGen(
        cond_dim=16,
        num_classes=5,
        noise_classes=4,
        noise_coords=2,
        input_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        out_channels=3,
        n_cls_tokens=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)
    activation_fn = build_activation_function(None)

    train_generator(
        model=model,
        optimizer=optimizer,
        logger=NullLogger(),
        eval_loader=_loader(2, image_size=8),
        train_loader=_loader(2, image_size=8),
        learning_rate_fn=schedule,
        preprocess_fn=_preprocess,
        postprocess_fn=_postprocess,
        feature_apply=activation_fn,
        model_config={"cond_dim": 16, "num_classes": 5, "input_size": 8, "in_channels": 3},
        dataset_name="imagenet256",
        workdir=str(tmp_path),
        device=torch.device("cpu"),
        train_batch_size=2,
        total_steps=2,
        save_per_step=1,
        eval_per_step=1,
        eval_samples=4,
        ema_decay=0.9,
        seed=11,
        pos_per_sample=2,
        neg_per_sample=1,
        forward_dict={"gen_per_label": 2},
        positive_bank_size=4,
        negative_bank_size=8,
        cfg_list=[1.0],
        loss_kwargs={"r_list": (0.05, 0.2)},
        keep_every=1,
        keep_last=2,
        push_per_step=2,
        push_at_resume=2,
        evaluate_fn=_fake_eval,
    )

    metadata = json.loads((tmp_path / "params_ema" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "gen"
    assert (tmp_path / "checkpoints" / "step_00000002.pt").is_file()


def test_train_generator_resume_restores_memory_banks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    def fake_train_step(
        state: TrainState,
        *,
        labels: Tensor,
        samples: Tensor,
        negative_samples: Tensor,
        feature_apply: Any,
        learning_rate_fn: Any,
        base_seed: int,
        **kwargs: Any,
    ) -> tuple[TrainState, dict[str, float]]:
        del (
            labels,
            samples,
            negative_samples,
            feature_apply,
            learning_rate_fn,
            base_seed,
            kwargs,
        )
        state.increment_step()
        return state, {"loss": 0.0, "lr": 0.0}

    monkeypatch.setattr("kdrifting.runners.generator.train_step", fake_train_step)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)
    loader = _loader(2, image_size=8, num_items=8)

    train_generator(
        model=model,
        optimizer=optimizer,
        logger=NullLogger(),
        eval_loader=loader,
        train_loader=loader,
        learning_rate_fn=schedule,
        preprocess_fn=_preprocess,
        postprocess_fn=_postprocess,
        feature_apply=_empty_feature_apply,
        model_config={"kind": "dummy"},
        dataset_name="imagenet256",
        workdir=str(tmp_path),
        device=torch.device("cpu"),
        train_batch_size=2,
        total_steps=1,
        save_per_step=1,
        eval_per_step=100,
        eval_samples=4,
        ema_decay=0.9,
        seed=11,
        pos_per_sample=2,
        neg_per_sample=1,
        forward_dict={"gen_per_label": 1},
        positive_bank_size=4,
        negative_bank_size=8,
        cfg_list=[1.0],
        keep_every=1,
        keep_last=2,
        push_per_step=2,
        push_at_resume=2,
        evaluate_fn=_fake_eval,
    )
    train_generator(
        model=model,
        optimizer=optimizer,
        logger=NullLogger(),
        eval_loader=loader,
        train_loader=loader,
        learning_rate_fn=schedule,
        preprocess_fn=_preprocess,
        postprocess_fn=_postprocess,
        feature_apply=_empty_feature_apply,
        model_config={"kind": "dummy"},
        dataset_name="imagenet256",
        workdir=str(tmp_path),
        device=torch.device("cpu"),
        train_batch_size=2,
        total_steps=2,
        save_per_step=1,
        eval_per_step=100,
        eval_samples=4,
        ema_decay=0.9,
        seed=11,
        pos_per_sample=2,
        neg_per_sample=1,
        forward_dict={"gen_per_label": 1},
        positive_bank_size=4,
        negative_bank_size=8,
        cfg_list=[1.0],
        keep_every=1,
        keep_last=2,
        push_per_step=2,
        push_at_resume=2,
        evaluate_fn=_fake_eval,
    )

    extra_state = restore_checkpoint_extra_state(workdir=str(tmp_path), step=2)
    assert extra_state is not None

    positive_state = cast(dict[str, Any], extra_state["memory_bank_positive"])
    count = np.asarray(positive_state["count"], dtype=np.int32)

    np.testing.assert_array_equal(count[:5], np.array([1, 1, 1, 1, 0], dtype=np.int32))

    output = capsys.readouterr().out
    assert "Restored generator memory banks" in output
    assert "Generator resume warmup" not in output
