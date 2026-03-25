"""ConvNeXt V2 feature backbone used by drift training."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import Tensor, nn

from kdrifting.models.common import bchw_to_bhwc, bhwc_to_bchw, safe_std


class ConvNextLayerNorm(nn.Module):
    """LayerNorm applied on the last channel of a BHWC tensor."""

    def __init__(self, normalized_shape: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x32 = x.to(dtype=torch.float32)
        mean = x32.mean(dim=-1, keepdim=True)
        var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x32 - mean) / torch.sqrt(var + self.eps)
        return (normed * self.weight + self.bias).to(dtype=input_dtype)


class ConvNextGRN(nn.Module):
    """Global response normalization on BHWC tensors."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x32 = x.to(dtype=torch.float32)
        gx = torch.sqrt((x32**2).sum(dim=(1, 2), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        out = self.gamma * (x32 * nx) + self.beta + x32
        return out.to(dtype=input_dtype)


class ConvNextBlock(nn.Module):
    """ConvNeXt V2 residual block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = ConvNextGRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = bchw_to_bhwc(self.dwconv(bhwc_to_bchw(x)))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = functional.gelu(x, approximate="none")
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNextStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.norm = ConvNextLayerNorm(out_channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = bchw_to_bhwc(self.conv(bhwc_to_bchw(x)))
        return self.norm(x)


class ConvNextDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = ConvNextLayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return bchw_to_bhwc(self.conv(bhwc_to_bchw(x)))


class ConvNextV2(nn.Module):
    """ConvNeXt V2 backbone with activation export."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNextV2 expects four stages.")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depths = tuple(int(depth) for depth in depths)
        self.dims = tuple(int(dim) for dim in dims)
        self.compute_dtype = dtype

        self.downsample_layers = nn.ModuleList(
            [
                ConvNextStem(in_channels, self.dims[0]),
                ConvNextDownsample(self.dims[0], self.dims[1]),
                ConvNextDownsample(self.dims[1], self.dims[2]),
                ConvNextDownsample(self.dims[2], self.dims[3]),
            ],
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*(ConvNextBlock(dim) for _ in range(depth)))
                for depth, dim in zip(self.depths, self.dims, strict=True)
            ],
        )
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = nn.Linear(self.dims[-1], num_classes)

    def _resize_input(self, x: Tensor) -> Tensor:
        x = bhwc_to_bchw(x.to(dtype=torch.float32))
        x = functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return bchw_to_bhwc(x).to(dtype=self.compute_dtype)

    @staticmethod
    def _normalize_features(x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x32 = x.to(dtype=torch.float32)
        mean = x32.mean(dim=-1, keepdim=True)
        std = x32.std(dim=-1, keepdim=True, unbiased=False)
        return ((x32 - mean) / (std + 1e-3)).to(dtype=input_dtype)

    def get_activations(self, x: Tensor) -> dict[str, Tensor]:
        x = self._resize_input(x)
        feature_dict: dict[str, Tensor] = {}
        for index, (downsample_layer, stage) in enumerate(
            zip(self.downsample_layers, self.stages, strict=True),
        ):
            x = downsample_layer(x)
            x = stage(x)
            x_normed = self._normalize_features(x)
            if index > 0:
                feature_dict[f"convenxt_stage_{index}"] = rearrange(
                    x_normed,
                    "b h w c -> b (h w) c",
                )
            feature_dict[f"convenxt_stage_{index}_mean"] = x_normed.mean(dim=(1, 2)).unsqueeze(1)
            feature_dict[f"convenxt_stage_{index}_std"] = safe_std(
                rearrange(x_normed, "b h w c -> b (h w) c"),
                dim=1,
            ).unsqueeze(1)

        feature_dict["global_mean"] = self.norm(x.mean(dim=(1, 2))).unsqueeze(1)
        feature_dict["global_std"] = safe_std(
            rearrange(self._normalize_features(x), "b h w c -> b (h w) c"),
            dim=1,
        ).unsqueeze(1)
        return feature_dict

    def forward_features(self, x: Tensor) -> Tensor:
        x = self._resize_input(x)
        for downsample_layer, stage in zip(self.downsample_layers, self.stages, strict=True):
            x = downsample_layer(x)
            x = stage(x)
        return self.norm(x.mean(dim=(1, 2)))

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)


ConvNextBase = partial(
    ConvNextV2,
    depths=(3, 3, 27, 3),
    dims=(128, 256, 512, 1024),
)
ConvNextTiny = partial(
    ConvNextV2,
    depths=(3, 3, 9, 3),
    dims=(96, 192, 384, 768),
)
