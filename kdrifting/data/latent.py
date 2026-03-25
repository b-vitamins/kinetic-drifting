"""Latent cache dataset and cache builder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from kdrifting.env import runtime_paths

LatentArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CacheWriteItem:
    output_path: str
    moments: LatentArray
    moments_flip: LatentArray


def write_cache_file(item: CacheWriteItem) -> None:
    """Atomically write one latent cache file."""
    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save({"moments": item.moments, "moments_flip": item.moments_flip}, tmp_path)
    tmp_path.replace(output_path)


class LatentDataset(Dataset[tuple[LatentArray, int]]):
    """ImageFolder-style dataset for cached latent ``.pt`` files."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Latent cache root does not exist: {self.root}")
        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        self.classes = [path.name for path in class_dirs]
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        for class_dir in class_dirs:
            target = self.class_to_idx[class_dir.name]
            for file_path in sorted(class_dir.rglob("*.pt")):
                self.samples.append((str(file_path), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[LatentArray, int]:
        path, target = self.samples[index]
        data = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=False))
        moments_key = "moments" if torch.rand(1).item() < 0.5 else "moments_flip"
        moments = np.asarray(data[moments_key], dtype=np.float32)
        return moments, int(target)


def _half_resolution(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return width // 2, height // 2


def _scaled_resolution(size: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = size
    return round(width * scale), round(height * scale)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop."""
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


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


def _cache_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ],
    )


def create_cached_dataset(
    *,
    local_batch_size: int,
    target_path: str,
    data_path: str,
    device: torch.device | str = "cpu",
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> None:
    """Encode ImageNet images and write a latent cache."""
    from kdrifting.data.vae import vae_enc_decode

    encode_fn, _ = vae_enc_decode(device=device)
    transform = _cache_transform()

    for split in ("train", "val"):
        image_folder = ImageFolder(str(Path(data_path) / split), transform=transform)
        samples = image_folder.samples
        dataset = cast(Dataset[tuple[Tensor, int]], image_folder)
        loader = cast(
            DataLoader[tuple[Tensor, Tensor]],
            DataLoader(
                dataset,
                batch_size=local_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            ),
        )
        for batch_index, (images, _labels) in enumerate(loader):
            moments = encode_fn(images)
            moments_flip = encode_fn(torch.flip(images, dims=[3]))
            start_index = batch_index * local_batch_size
            end_index = start_index + images.shape[0]
            batch_paths = samples[start_index:end_index]
            for sample_index, (path, _) in enumerate(batch_paths):
                rel_path = Path(*Path(path).parts[-2:]).with_suffix(".pt")
                write_cache_file(
                    CacheWriteItem(
                        output_path=str(Path(target_path) / split / rel_path),
                        moments=moments[sample_index].detach().cpu().numpy(),
                        moments_flip=moments_flip[sample_index].detach().cpu().numpy(),
                    ),
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse latent-cache builder arguments."""
    paths = runtime_paths()
    parser = argparse.ArgumentParser(description="Build ImageNet latent cache files.")
    parser.add_argument("--data-path", default=paths.imagenet_path)
    parser.add_argument("--target-path", default=paths.imagenet_cache_path)
    parser.add_argument("--local-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for latent-cache creation."""
    args = parse_args(argv)
    create_cached_dataset(
        local_batch_size=args.local_batch_size,
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )
