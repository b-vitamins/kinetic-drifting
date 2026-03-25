"""ImageNet data pipeline for the PyTorch port."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from kdrifting.data.latent import LatentDataset
from kdrifting.data.vae import vae_enc_decode
from kdrifting.distributed import is_distributed, rank, world_size
from kdrifting.env import runtime_paths

LoaderBatch = tuple[Tensor, Tensor]
PreprocessFn = Callable[[LoaderBatch], dict[str, Tensor]]
PostprocessFn = Callable[[Tensor], Tensor]


class _ManualSeedModule(Protocol):
    def manual_seed(self, seed: int) -> torch.Generator: ...


class _LoaderWithSampler(Protocol):
    sampler: object


def _half_resolution(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return width // 2, height // 2


def _scaled_resolution(size: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = size
    return round(width * scale), round(height * scale)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop with the source repo's ADM-style preprocessing."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            _half_resolution(pil_image.size),
            resample=Image.Resampling.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        _scaled_resolution(pil_image.size, scale),
        resample=Image.Resampling.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _build_transforms(resolution: int, *, use_aug: bool, split: str) -> transforms.Compose:
    def crop(image: Image.Image) -> Image.Image:
        return center_crop_arr(image, resolution)

    if use_aug and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    resolution,
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ],
        )
    return transforms.Compose(
        [
            transforms.Lambda(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ],
    )


def _build_imagenet_dataset(
    *,
    resolution: int,
    use_aug: bool,
    use_cache: bool,
    split: str,
) -> torch.utils.data.Dataset[Any]:
    paths = runtime_paths()
    if use_cache:
        return LatentDataset(root=str(Path(paths.imagenet_cache_path) / split))
    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=str(Path(paths.imagenet_path) / split), transform=transform)


def worker_init_fn(worker_id: int, *, process_rank: int) -> None:
    """Initialize per-worker random seeds."""
    seed = int(worker_id + process_rank * 1000)
    seed_module = cast(_ManualSeedModule, torch)
    seed_module.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    local: bool | None = None,
    device: torch.device | str = "cpu",
) -> tuple[DataLoader[LoaderBatch], PreprocessFn, PostprocessFn]:
    """Create an ImageNet loader plus preprocess/postprocess helpers."""
    del local
    dataset = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
    )
    sampler: DistributedSampler[Any] | None = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=True,
        )

    loader = cast(
        DataLoader[LoaderBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            drop_last=(split == "train"),
            worker_init_fn=partial(worker_init_fn, process_rank=rank()),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    )

    if use_latent or use_cache:
        encode_fn, decode_fn = vae_enc_decode(device=device)

        def preprocess_cached_batch(batch: LoaderBatch) -> dict[str, Tensor]:
            cached_latent, label = batch
            return {
                "images": cached_latent.to(dtype=torch.float32),
                "labels": label.to(dtype=torch.int64),
            }

        def preprocess_latent_batch(batch: LoaderBatch) -> dict[str, Tensor]:
            image, label = batch
            return {
                "images": encode_fn(image),
                "labels": label.to(dtype=torch.int64),
            }

        def postprocess_latents(images: Tensor) -> Tensor:
            return torch.clamp((decode_fn(images) + 1.0) / 2.0, 0.0, 1.0)

        preprocess_fn = preprocess_cached_batch if use_cache else preprocess_latent_batch
        return loader, preprocess_fn, postprocess_latents

    def preprocess_pixels(batch: LoaderBatch) -> dict[str, Tensor]:
        image, label = batch
        return {
            "images": image.permute(0, 2, 3, 1).contiguous(),
            "labels": label.to(dtype=torch.int64),
        }

    def postprocess_pixels(images: Tensor) -> Tensor:
        return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0).permute(0, 3, 1, 2).contiguous()

    return loader, preprocess_pixels, postprocess_pixels


def get_postprocess_fn(
    *,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    has_clip: bool = True,
    device: torch.device | str = "cpu",
) -> PostprocessFn:
    """Return the generated-sample postprocess function for the chosen data mode."""
    if use_latent or use_cache:
        _, decode_fn = vae_enc_decode(device=device)

        def postprocess_latents(images: Tensor) -> Tensor:
            out = (decode_fn(images) + 1.0) / 2.0
            return torch.clamp(out, 0.0, 1.0) if has_clip else out

        return postprocess_latents

    if use_aug or (not use_latent and not use_cache):

        def postprocess_pixels(images: Tensor) -> Tensor:
            out = (images + 1.0) / 2.0
            out = torch.clamp(out, 0.0, 1.0) if has_clip else out
            return out.permute(0, 3, 1, 2).contiguous()

        return postprocess_pixels

    raise ValueError("Unsupported dataset flags.")


def _maybe_distributed_sampler(
    sampler: object,
) -> DistributedSampler[Any] | None:
    if isinstance(sampler, DistributedSampler):
        return cast(DistributedSampler[Any], sampler)
    return None


def infinite_sampler(loader: DataLoader[LoaderBatch], start_step: int = 0) -> Iterator[LoaderBatch]:
    """Yield batches forever, resuming from ``start_step``."""
    steps_per_epoch = len(loader)
    epoch_index = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
    skip_batches = start_step % steps_per_epoch if steps_per_epoch > 0 else 0
    loader_with_sampler = cast(_LoaderWithSampler, loader)
    distributed_sampler = _maybe_distributed_sampler(loader_with_sampler.sampler)
    if distributed_sampler is not None:
        distributed_sampler.set_epoch(epoch_index)
    while True:
        for index, batch in enumerate(loader):
            if skip_batches > 0 and index < skip_batches:
                continue
            yield batch
        skip_batches = 0
        epoch_index += 1
        if distributed_sampler is not None:
            distributed_sampler.set_epoch(epoch_index)


def epoch0_sampler(loader: DataLoader[LoaderBatch]) -> Iterator[LoaderBatch]:
    """Yield one deterministic epoch."""
    loader_with_sampler = cast(_LoaderWithSampler, loader)
    distributed_sampler = _maybe_distributed_sampler(loader_with_sampler.sampler)
    if distributed_sampler is not None:
        distributed_sampler.set_epoch(0)
    yield from loader
