"""Torch VAE helpers for latent encode/decode paths."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Protocol, Self, cast

import torch
from torch import Tensor


class _LatentDistribution(Protocol):
    def sample(self) -> Tensor: ...


class _EncodeOutput(Protocol):
    latent_dist: _LatentDistribution


class _DecodeOutput(Protocol):
    sample: Tensor


class _AutoencoderLike(Protocol):
    def eval(self) -> Self: ...

    def to(self, device: torch.device) -> Self: ...

    def encode(self, images: Tensor) -> _EncodeOutput: ...

    def decode(self, latents: Tensor) -> _DecodeOutput: ...


class _AutoencoderFactory(Protocol):
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> _AutoencoderLike: ...


@lru_cache(maxsize=4)
def _load_vae(model_id: str, device_spec: str) -> _AutoencoderLike:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    autoencoder_factory = cast(_AutoencoderFactory, AutoencoderKL)
    model = autoencoder_factory.from_pretrained(model_id)
    model.eval()
    return model.to(torch.device(device_spec))


def vae_enc_decode(
    *,
    model_id: str = "pcuenq/sd-vae-ft-mse",
    device: torch.device | str = "cpu",
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return encode/decode callables matching the source latent API."""
    torch_device = torch.device(device)
    vae = _load_vae(model_id, str(torch_device))

    @torch.no_grad()
    def encode_fn(images: Tensor) -> Tensor:
        images = images.to(device=torch_device)
        posterior = vae.encode(images).latent_dist
        latents = posterior.sample() * 0.18215
        return latents.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def decode_fn(latents: Tensor) -> Tensor:
        latents = latents.to(device=torch_device).permute(0, 3, 1, 2).contiguous()
        decoded = vae.decode(latents / 0.18215).sample
        return decoded

    return encode_fn, decode_fn
