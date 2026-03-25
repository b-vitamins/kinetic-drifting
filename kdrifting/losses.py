"""Loss functions used by the PyTorch port."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise Euclidean distance for batched inputs."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def _default_weight(x: Tensor) -> Tensor:
    return torch.ones_like(x[:, :, 0], dtype=torch.float32)


def drift_loss(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    r_list: Sequence[float] = (0.02, 0.05, 0.2),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Faithful PyTorch implementation of the source drift loss."""
    _, channels_gen, feature_dim = gen.shape
    channels_pos = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])
    channels_neg = fixed_neg.shape[1]

    weight_gen = _default_weight(gen) if weight_gen is None else weight_gen
    weight_pos = _default_weight(fixed_pos) if weight_pos is None else weight_pos
    weight_neg = _default_weight(fixed_neg) if weight_neg is None else weight_neg

    gen = gen.to(dtype=torch.float32)
    fixed_pos = fixed_pos.to(dtype=torch.float32)
    fixed_neg = fixed_neg.to(dtype=torch.float32)
    weight_gen = weight_gen.to(dtype=torch.float32)
    weight_pos = weight_pos.to(dtype=torch.float32)
    weight_neg = weight_neg.to(dtype=torch.float32)

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        info: dict[str, Tensor] = {}
        dist = cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = torch.clamp(
            scale / torch.sqrt(torch.tensor(float(feature_dim), device=gen.device)),
            min=1e-3,
        )
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        mask_val = 100.0
        diag_mask = torch.eye(channels_gen, dtype=torch.float32, device=gen.device)
        block_mask = torch.nn.functional.pad(diag_mask, (0, channels_neg + channels_pos))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        force_across_r = torch.zeros_like(old_gen_scaled)
        for radius in r_list:
            logits = -dist_normed / radius
            affinity = torch.softmax(logits, dim=-1)
            affinity_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * affinity_t, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = channels_gen + channels_neg
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            coeff_neg = -aff_neg * sum_pos
            coeff_pos = aff_pos * sum_neg
            coeff = torch.cat([coeff_neg, coeff_pos], dim=2)

            total_force = torch.einsum("biy,byx->bix", coeff, targets_scaled)
            total_coeffs = coeff.sum(dim=-1)
            total_force = total_force - total_coeffs[..., None] * old_gen_scaled
            force_norm = (total_force.square()).mean()

            info[f"loss_{radius}"] = force_norm

            force_scale = torch.sqrt(torch.clamp(force_norm, min=1e-8))
            force_across_r = force_across_r + total_force / force_scale

        goal_scaled = old_gen_scaled + force_across_r

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = diff.square().mean(dim=(-1, -2))
    reduced_info = {key: value.mean() for key, value in info.items()}
    return loss, reduced_info
