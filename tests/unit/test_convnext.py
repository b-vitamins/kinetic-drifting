from __future__ import annotations

import torch

from kdrifting.models.convnext import ConvNextV2
from kdrifting.models.mae import MAEResNet, build_activation_function


def test_convnext_activations_have_expected_shapes() -> None:
    model = ConvNextV2(depths=(1, 1, 1, 1), dims=(16, 32, 64, 128))
    x = torch.randn(2, 64, 64, 3)

    activations = model.get_activations(x)

    assert activations["convenxt_stage_1"].shape[0] == 2
    assert activations["convenxt_stage_1_mean"].shape == (2, 1, 32)
    assert activations["convenxt_stage_2_mean"].shape == (2, 1, 64)
    assert activations["convenxt_stage_3_mean"].shape == (2, 1, 128)
    assert activations["global_mean"].shape == (2, 1, 128)
    assert activations["global_std"].shape == (2, 1, 128)


def test_activation_builder_merges_mae_and_convnext_features() -> None:
    mae_model = MAEResNet(
        num_classes=10,
        in_channels=3,
        base_channels=16,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    convnext_model = ConvNextV2(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
    activation_fn = build_activation_function(
        mae_model,
        convnext_model=convnext_model,
        postprocess_fn=lambda x: ((x + 1.0) / 2.0).permute(0, 3, 1, 2).contiguous(),
    )
    x = torch.randn(2, 32, 32, 3)

    activations = activation_fn(
        None,
        x,
        has_scale=True,
        patch_mean_size=[2],
        patch_std_size=[2],
        every_k_block=float("inf"),
    )

    assert "global" in activations
    assert "norm_x" in activations
    assert "conv1" in activations
    assert "convenxt_stage_1_mean" in activations
