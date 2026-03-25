"""Generator training utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor

from kdrifting.losses import drift_loss
from kdrifting.training.state import TrainState

FeatureApply = Callable[..., dict[str, Tensor]]


def _step_generator(base_seed: int, step: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(base_seed) + int(step))
    return generator


def _sample_cfg(
    batch_size: int,
    *,
    cfg_min: float,
    cfg_max: float,
    neg_cfg_pw: float,
    no_cfg_frac: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    frac = torch.rand((batch_size,), generator=generator, device=device)
    power = 1.0 - neg_cfg_pw
    if abs(power) < 1e-6:
        cfg = torch.exp(
            torch.log(torch.tensor(cfg_min, device=device))
            + frac
            * (
                torch.log(torch.tensor(cfg_max, device=device))
                - torch.log(torch.tensor(cfg_min, device=device))
            )
        )
    else:
        cfg = (cfg_min**power + frac * (cfg_max**power - cfg_min**power)) ** (1.0 / power)
    frac2 = torch.rand((batch_size,), generator=generator, device=device)
    return torch.where(frac2 < no_cfg_frac, torch.ones_like(cfg), cfg)


def train_step(
    state: TrainState,
    *,
    labels: Tensor,
    samples: Tensor,
    negative_samples: Tensor,
    feature_apply: FeatureApply,
    learning_rate_fn: Callable[[int], float],
    base_seed: int,
    cfg_min: float = 1.0,
    cfg_max: float = 4.0,
    neg_cfg_pw: float = 1.0,
    no_cfg_frac: float = 0.0,
    gen_per_label: int = 8,
    activation_kwargs: dict[str, Any] | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float = 2.0,
) -> tuple[TrainState, dict[str, float]]:
    """Run one generator optimization step."""
    activation_kwargs = dict(activation_kwargs or {})
    loss_kwargs = dict(loss_kwargs or {})
    generator = _step_generator(base_seed, state.step, labels.device)
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    cfg = _sample_cfg(
        samples.shape[0],
        cfg_min=cfg_min,
        cfg_max=cfg_max,
        neg_cfg_pw=neg_cfg_pw,
        no_cfg_frac=no_cfg_frac,
        generator=generator,
        device=labels.device,
    )
    uncond_w = (cfg - 1.0) * (gen_per_label - 1) / max(1, negative_samples.shape[1])
    n_pos = samples.shape[1]
    n_gen = gen_per_label
    n_uncond = negative_samples.shape[1]

    neg_samples_input = rearrange(
        torch.cat([samples, negative_samples], dim=1), "b x ... -> (b x) ..."
    )
    with torch.no_grad():
        sg_features = feature_apply(None, neg_samples_input, **activation_kwargs)
    sg_features = {
        key: rearrange(value, "(b x) ... -> b x ...", x=n_pos + n_uncond)
        for key, value in sg_features.items()
    }

    input_labels = repeat(labels, "b -> (b g)", g=gen_per_label)
    input_cfg = repeat(cfg, "b -> (b g)", g=gen_per_label)
    gen_out = state.model(
        input_labels,
        cfg_scale=input_cfg,
        deterministic=False,
        generator=generator,
    )
    gen_samples = gen_out["samples"]
    gen_features = feature_apply(None, gen_samples, **activation_kwargs)
    gen_features = {
        key: rearrange(value, "(b g) ... -> b g ...", g=n_gen)
        for key, value in gen_features.items()
    }

    total_loss = torch.tensor(0.0, device=labels.device)
    total_info: dict[str, float] = {}
    for key in sg_features:
        feature_pos = rearrange(sg_features[key][:, :n_pos], "b x f d -> (b f) x d")
        feature_gen = rearrange(gen_features[key], "b x f d -> (b f) x d")
        feature_uncond = rearrange(sg_features[key][:, n_pos:], "b x f d -> (b f) x d")
        feature_batch = feature_gen.shape[0]
        feature_loss, feature_info = drift_loss(
            gen=feature_gen,
            fixed_pos=feature_pos,
            fixed_neg=feature_uncond,
            weight_gen=torch.ones_like(feature_gen[:, :, 0]),
            weight_pos=torch.ones_like(feature_pos[:, :, 0]),
            weight_neg=repeat(
                uncond_w, "b -> (b f) k", f=feature_batch // uncond_w.shape[0], k=n_uncond
            ),
            **loss_kwargs,
        )
        total_loss = total_loss + feature_loss.mean()
        for info_key, info_value in feature_info.items():
            total_info[f"{info_key}/{key}"] = float(info_value.detach().item())

    torch.autograd.backward(total_loss)
    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_grad_norm)
    state.optimizer.step()
    state.increment_step()
    state.update_ema()

    metrics = {
        "loss": float(total_loss.detach().item()),
        "g_norm": float(grad_norm.detach().item()),
        "lr": float(learning_rate_fn(state.step - 1)),
    }
    metrics.update(total_info)
    return state, metrics
