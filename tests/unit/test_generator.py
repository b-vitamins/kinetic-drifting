from __future__ import annotations

import torch

from kdrifting.models.generator import DitGen, LightningDiT


def test_lightning_dit_preserves_expected_output_shape() -> None:
    model = LightningDiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_channels=3,
        n_cls_tokens=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    x = torch.randn(2, 8, 8, 3)
    cond = torch.randn(2, 32)

    out = model(x, cond, deterministic=True)

    assert out.shape == (2, 8, 8, 3)


def test_ditgen_forward_returns_samples_and_noise_dict() -> None:
    model = DitGen(
        cond_dim=32,
        num_classes=10,
        noise_classes=8,
        noise_coords=4,
        input_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_channels=3,
        n_cls_tokens=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    generator = torch.Generator().manual_seed(123)

    out = model(labels, cfg_scale=1.5, generator=generator)

    assert set(out.keys()) == {"samples", "noise"}
    assert isinstance(out["noise"], dict)
    assert out["samples"].shape == (2, 8, 8, 3)
    assert out["noise"]["x"].shape == (2, 8, 8, 3)
    assert out["noise"]["noise_labels"].shape == (2, 4)
