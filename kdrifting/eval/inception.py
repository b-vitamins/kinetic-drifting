"""FID Inception backbone and feature extraction helpers."""

from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn.functional as functional
import torchvision
from torch import Tensor, nn
from torchvision.models.inception import Inception3, InceptionA, InceptionC, InceptionE

FID_WEIGHTS_URL = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/"
    "pt_inception-2015-12-05-6726825d.pth"
)


class _StateDictFromUrl(Protocol):
    def __call__(
        self,
        url: str,
        *,
        progress: bool = True,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Tensor]: ...


def _load_state_dict_from_url(
    url: str,
    *,
    progress: bool = True,
    map_location: str | torch.device | None = None,
) -> dict[str, Tensor]:
    from torch.hub import load_state_dict_from_url

    loader = cast(_StateDictFromUrl, load_state_dict_from_url)
    return loader(url, progress=progress, map_location=map_location)


def _inception_v3(*, num_classes: int, aux_logits: bool, weights: object | None) -> Inception3:
    version = tuple(int(part) for part in torchvision.__version__.split(".")[:2])
    kwargs: dict[str, object] = {
        "num_classes": num_classes,
        "aux_logits": aux_logits,
        "weights": weights,
    }
    if version >= (0, 6):
        kwargs["init_weights"] = False
    return torchvision.models.inception_v3(**kwargs)


class FIDInceptionA(InceptionA):
    """InceptionA block patched for FID computation."""

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = functional.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)


class FIDInceptionC(InceptionC):
    """InceptionC block patched for FID computation."""

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = functional.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)


class FIDInceptionE1(InceptionE):
    """First InceptionE block patched for FID computation."""

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, dim=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, dim=1)

        branch_pool = functional.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], dim=1)


class FIDInceptionE2(InceptionE):
    """Second InceptionE block patched for FID computation."""

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, dim=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, dim=1)

        branch_pool = functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], dim=1)


def fid_inception_v3() -> Inception3:
    """Build the FID-specific InceptionV3 network with published weights."""
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE1(1280)
    inception.Mixed_7c = FIDInceptionE2(2048)
    state_dict = _load_state_dict_from_url(FID_WEIGHTS_URL, progress=True, map_location="cpu")
    inception.load_state_dict(state_dict)
    return inception


def build_inception_extractor() -> FIDInceptionExtractor:
    """Construct the shared FID Inception feature extractor."""
    return FIDInceptionExtractor()


class FIDInceptionExtractor(nn.Module):
    """Feature extractor that returns pooled activations and logits."""

    def __init__(self) -> None:
        super().__init__()
        inception = fid_inception_v3()
        self.block0 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block1 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        self.block3 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.fc = inception.fc
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return pooled 2048-d activations and 1008-d logits."""
        x = self.block0(images)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        pooled = torch.flatten(x, start_dim=1)
        logits = self.fc(pooled)
        return pooled, logits
