from __future__ import annotations

import torch

from kdrifting.models.mae import MAEResNet, make_patch_mask, patch_input


def test_patch_input_rearranges_spatial_patches() -> None:
    x = torch.arange(1 * 4 * 4 * 1, dtype=torch.float32).reshape(1, 4, 4, 1)

    out = patch_input(x, 2)

    assert out.shape == (1, 2, 2, 4)


def test_make_patch_mask_expands_patchwise_mask() -> None:
    x = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    mask_ratio = torch.tensor([1.0, 0.0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(1)

    mask = make_patch_mask(x, mask_ratio, patch_size=4, generator=generator)

    assert mask.shape == (2, 8, 8, 1)
    assert torch.all(mask[0] == 1)
    assert torch.all(mask[1] == 0)


def test_mae_forward_and_activations_have_expected_shapes() -> None:
    model = MAEResNet(
        num_classes=10,
        in_channels=3,
        base_channels=32,
        patch_size=2,
        layers=(2, 2, 2, 2),
        input_patch_size=1,
    )
    x = torch.randn(2, 32, 32, 3)
    labels = torch.tensor([1, 2], dtype=torch.int64)
    generator = torch.Generator().manual_seed(7)

    loss, metrics = model(
        x,
        labels,
        lambda_cls=0.1,
        mask_ratio_min=0.5,
        mask_ratio_max=0.5,
        generator=generator,
    )
    activations = model.get_activations(x, patch_mean_size=[2], patch_std_size=[2], every_k_block=1)

    assert loss.shape == (2,)
    assert metrics["accuracy"].shape == (2,)
    assert "conv1" in activations
    assert activations["conv1"].ndim == 3
