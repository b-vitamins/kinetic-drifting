"""MAE-style feature model ported to PyTorch."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import Tensor, nn

from kdrifting.models.common import bchw_to_bhwc, bhwc_to_bchw, safe_std
from kdrifting.models.convnext import ConvNextV2


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def patch_input(x: Tensor, input_patch_size: int) -> Tensor:
    return rearrange(
        x,
        "b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)",
        h2=input_patch_size,
        w2=input_patch_size,
    )


def make_patch_mask(
    x: Tensor,
    mask_ratio: Tensor,
    *,
    patch_size: int = 4,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch, height, width, _ = x.shape
    nh, nw = height // patch_size, width // patch_size
    noise = torch.rand((batch, nh, nw), device=x.device, dtype=x.dtype, generator=generator)
    mask = (noise < mask_ratio[:, None, None]).to(dtype=x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return mask.unsqueeze(-1)


class BasicBlock(nn.Module):
    """ResNet basic block implemented with PyTorch convolutions."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        *,
        stride: int = 1,
        gn_max_groups: int = 32,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            filters,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)
        self.drop = nn.Dropout(dropout_prob)
        self.proj_conv = nn.Conv2d(
            in_channels,
            filters,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.proj_gn = nn.GroupNorm(_choose_gn_groups(filters, gn_max_groups), filters)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = functional.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))

        if residual.shape != y.shape:
            residual = self.proj_gn(self.proj_conv(residual))
        return functional.relu(residual + y)


class ResNetEncoder(nn.Module):
    """Four-stage encoder producing multi-scale feature maps."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(_choose_gn_groups(base_channels, gn_max_groups), base_channels)

        stages: list[nn.Sequential] = []
        stage_norms: list[nn.GroupNorm] = []
        previous_channels = base_channels
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_channels = base_channels * (2**stage_idx)
            blocks: list[nn.Module] = []
            blocks.append(
                BasicBlock(
                    previous_channels,
                    out_channels,
                    stride=stride,
                    dropout_prob=dropout_prob,
                    gn_max_groups=gn_max_groups,
                ),
            )
            blocks.extend(
                [
                    BasicBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        dropout_prob=dropout_prob,
                        gn_max_groups=gn_max_groups,
                    )
                    for _ in range(1, num_blocks)
                ],
            )
            stages.append(nn.Sequential(*blocks))
            stage_norms.append(
                nn.GroupNorm(_choose_gn_groups(out_channels, gn_max_groups), out_channels),
            )
            previous_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(stage_norms)

    def forward(
        self,
        x: Tensor,
        *,
        return_block_outputs: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, list[Tensor]]] | dict[str, Tensor]:
        feats: dict[str, Tensor] = {}
        block_outputs: dict[str, list[Tensor]] = {}
        x = functional.relu(self.gn1(self.conv1(x)))
        feats["conv1"] = x

        for index, blocks in enumerate(self.stages):
            layer_name = f"layer{index + 1}"
            outputs: list[Tensor] = []
            for block in blocks.children():
                typed_block = cast(BasicBlock, block)
                x = typed_block(x)
                outputs.append(x)
            x = self.stage_norms[index](x)
            feats[layer_name] = x
            block_outputs[layer_name] = outputs

        if return_block_outputs:
            return feats, block_outputs
        return feats


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.gn = nn.GroupNorm(_choose_gn_groups(out_channels, 32), out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return functional.relu(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        total_channels = in_channels + skip_channels
        self.concat_norm = nn.GroupNorm(
            _choose_gn_groups(total_channels, 32),
            total_channels,
        )
        self.proj = ConvGNReLU(in_channels + skip_channels, out_channels)
        self.refine = ConvGNReLU(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm(x)
        x = self.proj(x)
        return self.refine(x)


class UNetDecoder(nn.Module):
    def __init__(self, *, base_channels: int, out_channels: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8
        self.bridge = ConvGNReLU(c5, c5)
        self.up43 = UpBlock(c5, c4, c4)
        self.up32 = UpBlock(c4, c3, c3)
        self.up21 = UpBlock(c3, c2, c2)
        self.up10 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, feats: dict[str, Tensor]) -> Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


class MAEResNet(nn.Module):
    """PyTorch port of the MAE-ResNet feature model."""

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Sequence[int] = (2, 2, 2, 2),
        use_bf16: bool = False,
        input_patch_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.patch_size = patch_size
        self.use_bf16 = use_bf16
        self.input_patch_size = input_patch_size
        if len(layers) != 4:
            raise ValueError(f"Expected 4 layer counts, got {layers}")
        normalized_layers = tuple(int(layer) for layer in layers)
        effective_in_channels = in_channels * input_patch_size * input_patch_size
        self.encoder = ResNetEncoder(
            base_channels=base_channels,
            layers=cast(tuple[int, int, int, int], normalized_layers),
            dropout_prob=dropout_prob,
        )
        self.decoder = UNetDecoder(
            base_channels=base_channels,
            out_channels=effective_in_channels,
        )
        self.input_proj = nn.Conv2d(effective_in_channels, base_channels, kernel_size=1, bias=False)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _encode(
        self,
        x_bhwc: Tensor,
        *,
        return_block_outputs: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, list[Tensor]]] | dict[str, Tensor]:
        x = bhwc_to_bchw(x_bhwc)
        x = self.input_proj(x)
        return self.encoder(x, return_block_outputs=return_block_outputs)

    def forward(
        self,
        x: Tensor,
        labels: Tensor,
        *,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        train: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x = patch_input(x, self.input_patch_size)
        if self.use_bf16:
            x = x.to(dtype=torch.bfloat16)
        mask_ratio = torch.rand((x.shape[0],), device=x.device, dtype=x.dtype, generator=generator)
        mask_ratio = mask_ratio * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_ratio, patch_size=self.patch_size, generator=generator)
        x_in = x * (1.0 - mask)

        feats = self._encode(x_in)
        assert isinstance(feats, dict)
        top = feats["layer4"]
        pooled = top.mean(dim=(-1, -2))
        logits = self.fc(pooled.to(dtype=torch.float32))
        recon = bchw_to_bhwc(self.decoder(feats))

        one_hot = functional.one_hot(labels, num_classes=self.num_classes).to(dtype=recon.dtype)
        cls_loss = -(one_hot * functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        mse = (recon - x) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (logits.argmax(dim=-1) == labels).to(dtype=recon.dtype),
            "mask_ratio": mask.mean(dim=(1, 2, 3)),
        }
        return loss, metrics

    def get_activations(
        self,
        x: Tensor,
        *,
        patch_mean_size: Sequence[int] | None = (2, 4),
        patch_std_size: Sequence[int] | None = (2, 4),
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> dict[str, Tensor]:
        patch_mean = list(patch_mean_size or [])
        patch_std = list(patch_std_size or [])
        x = patch_input(x, self.input_patch_size)
        need_blocks = not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            encoded = self._encode(x, return_block_outputs=True)
            assert isinstance(encoded, tuple)
            feats, block_outputs = encoded
        else:
            feats = self._encode(x)
            assert isinstance(feats, dict)
            block_outputs = {}

        out: dict[str, Tensor] = {}
        out["norm_x"] = torch.sqrt((x**2).mean(dim=(1, 2)) + 1e-6).unsqueeze(1)

        def process_feat(name: str, feat_bchw: Tensor) -> None:
            feat = bchw_to_bhwc(feat_bchw)
            batch, height, width, channels = feat.shape
            del batch, channels
            out[name] = rearrange(feat, "b h w c -> b (h w) c")
            if use_mean:
                out[f"{name}_mean"] = feat.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
            if use_std:
                out[f"{name}_std"] = safe_std(feat, dim=(1, 2)).unsqueeze(1)

            for size in patch_mean:
                if height % size == 0 and width % size == 0:
                    reshaped = rearrange(
                        feat,
                        "b (h s1) (w s2) c -> b (h w) (s1 s2) c",
                        s1=size,
                        s2=size,
                    )
                    out[f"{name}_mean_{size}"] = reshaped.mean(dim=2)

            for size in patch_std:
                if height % size == 0 and width % size == 0:
                    reshaped = rearrange(
                        feat,
                        "b (h s1) (w s2) c -> b (h w) (s1 s2) c",
                        s1=size,
                        s2=size,
                    )
                    out[f"{name}_std_{size}"] = safe_std(reshaped, dim=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            block_stride = int(every_k_block)
            for layer_index in range(1, 5):
                layer_name = f"layer{layer_index}"
                for block_index, feat in enumerate(block_outputs.get(layer_name, []), start=1):
                    if block_index % block_stride == 0:
                        process_feat(f"{layer_name}_blk{block_index}", feat)

        return out

    def dummy_input(self) -> dict[str, Any]:
        patch = self.input_patch_size
        return {
            "x": torch.zeros((1, 32 * patch, 32 * patch, self.in_channels), dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.int64),
            "lambda_cls": 0.0,
            "mask_ratio_min": 0.75,
            "mask_ratio_max": 0.75,
            "train": False,
        }


def mae_from_metadata(metadata: Mapping[str, Any]) -> MAEResNet:
    model_config = dict(metadata.get("model_config", {}) or {})
    num_classes = int(model_config.pop("num_classes", 1000))
    return MAEResNet(num_classes=num_classes, **model_config)


def build_activation_function(
    mae_model: MAEResNet | None,
    *,
    convnext_model: ConvNextV2 | None = None,
    postprocess_fn: Callable[[Tensor], Tensor] | None = None,
) -> Callable[..., dict[str, Tensor]]:
    """Build the feature extraction callable used by drift training."""

    def activation_fn(
        params: dict[str, Tensor] | None,
        x: Tensor,
        *,
        convnext_kwargs: dict[str, Any] | None = None,
        has_scale: bool = False,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        del params
        usual_feats = {"global": x.reshape(x.shape[0], 1, -1)}
        if has_scale:
            usual_feats["norm_x"] = torch.sqrt((x**2).mean(dim=(1, 2)) + 1e-6).unsqueeze(1)
        if mae_model is not None:
            usual_feats |= mae_model.get_activations(x, **kwargs)
        if convnext_model is not None:
            if postprocess_fn is None:
                raise ValueError("postprocess_fn is required when convnext_model is provided.")
            convnext_input = postprocess_fn(x)
            if convnext_input.ndim != 4:
                raise ValueError(
                    f"Expected 4D ConvNeXt input, got shape {tuple(convnext_input.shape)}",
                )
            if convnext_input.shape[1] in {1, 3} and convnext_input.shape[-1] not in {1, 3}:
                convnext_input = bchw_to_bhwc(convnext_input)
            imagenet_mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=convnext_input.device,
                dtype=convnext_input.dtype,
            )
            imagenet_std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=convnext_input.device,
                dtype=convnext_input.dtype,
            )
            convnext_input = (convnext_input - imagenet_mean) / imagenet_std
            usual_feats |= convnext_model.get_activations(convnext_input, **(convnext_kwargs or {}))
        return usual_feats

    return activation_fn
