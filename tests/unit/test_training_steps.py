from __future__ import annotations

import torch

from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet, build_activation_function
from kdrifting.schedules import create_learning_rate_fn
from kdrifting.training.generator import train_step as train_generator_step
from kdrifting.training.mae import train_step as train_mae_step
from kdrifting.training.state import TrainState


def _identity_preprocess(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def test_mae_train_step_updates_state_and_returns_metrics() -> None:
    model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=8,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState.create(model, optimizer, ema_decay=0.9)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)
    batch = {
        "images": torch.randn(2, 32, 32, 3),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
    }

    state, metrics = train_mae_step(
        state,
        batch,
        base_seed=123,
        forward_dict={"mask_ratio_min": 0.5, "mask_ratio_max": 0.5, "lambda_cls": 0.1},
        learning_rate_fn=schedule,
        preprocess_fn=_identity_preprocess,
    )

    assert state.step == 1
    assert "loss" in metrics
    assert "g_norm" in metrics
    assert "lr" in metrics


def test_generator_train_step_runs_on_small_model() -> None:
    generator_model = DitGen(
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
    optimizer = torch.optim.AdamW(generator_model.parameters(), lr=1e-3)
    state = TrainState.create(generator_model, optimizer, ema_decay=0.9)
    feature_model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=8,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    activation_fn = build_activation_function(feature_model)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)

    labels = torch.tensor([1, 2], dtype=torch.int64)
    samples = torch.randn(2, 2, 8, 8, 3)
    negative_samples = torch.randn(2, 1, 8, 8, 3)

    state, metrics = train_generator_step(
        state,
        labels=labels,
        samples=samples,
        negative_samples=negative_samples,
        feature_apply=activation_fn,
        learning_rate_fn=schedule,
        base_seed=321,
        gen_per_label=2,
        activation_kwargs={
            "patch_mean_size": [2],
            "patch_std_size": [2],
            "use_std": True,
            "use_mean": True,
            "every_k_block": 1,
        },
        loss_kwargs={"r_list": (0.05, 0.2)},
    )

    assert state.step == 1
    assert "loss" in metrics
    assert "g_norm" in metrics
    assert "lr" in metrics
