from __future__ import annotations

import math

import numpy as np
import torch

from kdrifting.losses import drift_loss


def _numpy_cdist(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    xydot = np.einsum("bnd,bmd->bnm", x, y)
    xnorms = np.einsum("bnd,bnd->bn", x, x)
    ynorms = np.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return np.sqrt(np.clip(sq_dist, a_min=eps, a_max=None))


def _numpy_drift_loss(
    gen: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_neg: np.ndarray,
    r_list: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float], float, np.ndarray]:
    weight_gen = np.ones_like(gen[:, :, 0], dtype=np.float32)
    weight_pos = np.ones_like(fixed_pos[:, :, 0], dtype=np.float32)
    weight_neg = np.ones_like(fixed_neg[:, :, 0], dtype=np.float32)
    _, channels_gen, feature_dim = gen.shape
    channels_pos = fixed_pos.shape[1]
    channels_neg = fixed_neg.shape[1]

    old_gen = gen.copy()
    targets = np.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    targets_w = np.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    info: dict[str, float] = {}
    dist = _numpy_cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    scale = float(weighted_dist.mean() / targets_w.mean())
    info["scale"] = scale

    scale_inputs = max(scale / math.sqrt(feature_dim), 1e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs
    dist_normed = dist / max(scale, 1e-3)

    diag_mask = np.eye(channels_gen, dtype=np.float32)
    block_mask = np.pad(diag_mask, ((0, 0), (0, channels_neg + channels_pos)))
    dist_normed = dist_normed + block_mask[None, ...] * 100.0

    force_across_r = np.zeros_like(old_gen_scaled)
    for radius in r_list:
        logits = -dist_normed / radius
        affinity = np.exp(logits - logits.max(axis=-1, keepdims=True))
        affinity = affinity / affinity.sum(axis=-1, keepdims=True)

        affinity_t = np.exp(logits - logits.max(axis=-2, keepdims=True))
        affinity_t = affinity_t / affinity_t.sum(axis=-2, keepdims=True)

        affinity = np.sqrt(np.clip(affinity * affinity_t, a_min=1e-6, a_max=None))
        affinity = affinity * targets_w[:, None, :]

        split_idx = channels_gen + channels_neg
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        sum_pos = aff_pos.sum(axis=-1, keepdims=True)
        sum_neg = aff_neg.sum(axis=-1, keepdims=True)
        coeff_neg = -aff_neg * sum_pos
        coeff_pos = aff_pos * sum_neg
        coeff = np.concatenate([coeff_neg, coeff_pos], axis=2)

        total_force = np.einsum("biy,byx->bix", coeff, targets_scaled)
        total_coeffs = coeff.sum(axis=-1)
        total_force = total_force - total_coeffs[..., None] * old_gen_scaled
        force_norm = float((total_force**2).mean())
        info[f"loss_{radius}"] = force_norm

        force_scale = math.sqrt(max(force_norm, 1e-8))
        force_across_r = force_across_r + total_force / force_scale

    goal_scaled = old_gen_scaled + force_across_r
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff**2).mean(axis=(-1, -2))
    return loss, info, scale_inputs, goal_scaled


def test_drift_loss_matches_independent_numpy_reference() -> None:
    gen = torch.tensor(
        [[[0.2, -0.3], [0.4, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    fixed_pos = torch.tensor(
        [[[0.1, -0.4], [0.6, 0.3]]],
        dtype=torch.float32,
    )
    fixed_neg = torch.tensor(
        [[[-0.2, 0.5]]],
        dtype=torch.float32,
    )

    torch_loss, torch_info = drift_loss(gen, fixed_pos, fixed_neg, r_list=(0.05, 0.2))
    numpy_loss, numpy_info, _, _ = _numpy_drift_loss(
        gen.detach().numpy(),
        fixed_pos.numpy(),
        fixed_neg.numpy(),
        (0.05, 0.2),
    )

    np.testing.assert_allclose(torch_loss.detach().numpy(), numpy_loss, rtol=1e-5, atol=1e-5)
    assert set(torch_info) == set(numpy_info)
    for key, value in numpy_info.items():
        assert math.isclose(torch_info[key].item(), value, rel_tol=1e-5, abs_tol=1e-5)


def test_drift_loss_gradient_matches_finite_difference() -> None:
    gen = torch.tensor(
        [[[0.3, -0.1], [0.2, 0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    fixed_pos = torch.tensor(
        [[[0.0, -0.2], [0.5, 0.1]]],
        dtype=torch.float32,
    )
    fixed_neg = torch.tensor(
        [[[0.2, -0.4]]],
        dtype=torch.float32,
    )

    loss, _ = drift_loss(gen, fixed_pos, fixed_neg, r_list=(0.05, 0.2))
    autodiff_grad = torch.autograd.grad(loss.sum(), gen)[0]

    _, _, scale_inputs, goal_scaled = _numpy_drift_loss(
        gen.detach().numpy(),
        fixed_pos.numpy(),
        fixed_neg.numpy(),
        (0.05, 0.2),
    )
    gen_scaled = gen.detach().numpy() / scale_inputs
    manual_grad = 2.0 * (gen_scaled - goal_scaled) / (gen.shape[1] * gen.shape[2] * scale_inputs)

    np.testing.assert_allclose(
        autodiff_grad.detach().numpy(),
        manual_grad,
        rtol=1e-5,
        atol=1e-5,
    )
