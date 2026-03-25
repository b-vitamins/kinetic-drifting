"""Generator evaluation helpers for FID, Inception Score, and precision/recall."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from kdrifting.data import epoch0_sampler
from kdrifting.distributed import all_gather_objects, barrier
from kdrifting.env import runtime_paths
from kdrifting.logging import WandbLogger

from . import resize
from .fid import compute_frechet_distance
from .inception import FIDInceptionExtractor, build_inception_extractor
from .precision_recall import compute_precision_recall

RawBatch = tuple[Tensor, Tensor]
GeneratorFn = Callable[[RawBatch, int], Tensor]
MetricDict = dict[str, float]
StatDict = dict[str, np.ndarray]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]

_inception_extractor_cache: FIDInceptionExtractor | None = None
_pr_ref_features_cache: np.ndarray | None = None
_pr_ref_source_cache: str | None = None


class LoggerLike(Protocol):
    def log_dict(self, values: dict[str, Any]) -> None: ...

    def log_image(self, name: str, images: np.ndarray | torch.Tensor) -> None: ...


def _canonical_dataset_name(name: str) -> str:
    value = name.lower()
    if "imagenet256" in value:
        return "imagenet256"
    raise ValueError(f"Only ImageNet-256 is supported, got {name!r}.")


def _dataset_fid_stats_path(dataset_name: str) -> str:
    canon = _canonical_dataset_name(dataset_name)
    paths = runtime_paths()
    dataset_stats = {"imagenet256": paths.imagenet_fid_npz}
    return dataset_stats[canon]


def _load_reference_stats(dataset_name: str) -> StatDict:
    path = _dataset_fid_stats_path(dataset_name)
    data = np.load(path)
    if "ref_mu" in data:
        return {
            "mu": np.asarray(data["ref_mu"], dtype=np.float64),
            "sigma": np.asarray(data["ref_sigma"], dtype=np.float64),
        }
    return {
        "mu": np.asarray(data["mu"], dtype=np.float64),
        "sigma": np.asarray(data["sigma"], dtype=np.float64),
    }


def _get_inception_extractor(device: torch.device) -> FIDInceptionExtractor:
    global _inception_extractor_cache
    if _inception_extractor_cache is None:
        _inception_extractor_cache = build_inception_extractor()
    _inception_extractor_cache.to(device)
    _inception_extractor_cache.eval()
    return _inception_extractor_cache


def _to_uint8(samples: np.ndarray | Tensor) -> np.ndarray:
    return WandbLogger.normalize_images(samples)


def _stack_gathered_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [array for array in arrays if array.size > 0]
    if not nonempty:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)


def _covariance(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected features to be 2D, got shape {features.shape}.")
    if features.shape[0] <= 1:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.cov(features, rowvar=False)


@torch.no_grad()
def compute_inception_stats(
    samples_uint8: np.ndarray,
    num_samples: int,
    *,
    extractor: nn.Module,
    device: torch.device,
    compute_logits: bool,
    compute_features: bool,
    masks: np.ndarray | None = None,
    batch_size: int = 200,
) -> StatDict:
    """Run the FID Inception network over generated uint8 images and collect stats."""
    if samples_uint8.ndim != 4:
        raise ValueError(f"Expected a 4D sample array, got shape {samples_uint8.shape}.")
    if samples_uint8.shape[-1] != 3:
        raise ValueError(f"Expected NHWC uint8 samples, got shape {samples_uint8.shape}.")
    if masks is None:
        masks = np.ones((len(samples_uint8),), dtype=np.float32)

    feature_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    for start in range(0, len(samples_uint8), batch_size):
        stop = min(start + batch_size, len(samples_uint8))
        batch_array: Float32Array = np.ascontiguousarray(
            samples_uint8[start:stop].transpose(0, 3, 1, 2).astype(np.float32),
        )
        batch = torch.as_tensor(batch_array, device=device)
        batch = resize.forward(batch)
        pooled, logits = extractor(batch)
        feature_batches.append(pooled.detach().cpu().numpy())
        if compute_logits:
            logits_batches.append(logits.detach().cpu().numpy())

    features = _stack_gathered_arrays(feature_batches)
    valid_length = min(len(features), len(masks))
    valid_mask = np.asarray(masks[:valid_length] > 0.5)
    features = features[:valid_length][valid_mask][:num_samples]
    features64 = np.asarray(features, dtype=np.float64)
    if len(features64) == 0:
        raise ValueError("Expected at least one valid feature vector for evaluation.")

    stats: StatDict = {
        "mu": np.mean(features64, axis=0),
        "sigma": _covariance(features64),
    }
    if compute_features:
        stats["features"] = features
    if compute_logits:
        logits = _stack_gathered_arrays(logits_batches)
        logits = logits[:valid_length][valid_mask][:num_samples]
        stats["logits"] = logits
    return stats


def compute_inception_score(logits: np.ndarray, *, splits: int = 10) -> tuple[float, float]:
    """Compute the source-compatible Inception Score from raw logits."""
    rng = np.random.RandomState(2020)
    shuffled: Float64Array = np.asarray(logits, dtype=np.float64)[
        rng.permutation(logits.shape[0]),
        :,
    ]
    if shuffled.shape[0] == 0:
        raise ValueError("Expected at least one logit row for Inception Score.")

    effective_splits = min(splits, shuffled.shape[0])
    split_size = shuffled.shape[0] // effective_splits
    probs: Float64Array = np.asarray(
        torch.as_tensor(shuffled).softmax(dim=-1).cpu().numpy(),
        dtype=np.float64,
    )
    probs = probs[: split_size * effective_splits]

    scores: list[float] = []
    for index in range(effective_splits):
        part = probs[index * split_size : (index + 1) * split_size]
        py: Float64Array = np.asarray(np.mean(part, axis=0, keepdims=True), dtype=np.float64)
        kl: Float64Array = np.asarray(
            part * (np.log(part + 1e-10) - np.log(py + 1e-10)),
            dtype=np.float64,
        )
        score = float(np.mean(np.sum(kl, axis=1), dtype=np.float64))
        scores.append(float(np.exp(score)))

    scores_array = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores_array)), float(np.std(scores_array))


def _load_pr_reference_features(
    *,
    extractor: nn.Module,
    device: torch.device,
) -> np.ndarray:
    global _pr_ref_features_cache, _pr_ref_source_cache
    path = runtime_paths().imagenet_pr_npz
    if path == _pr_ref_source_cache and _pr_ref_features_cache is not None:
        return _pr_ref_features_cache
    data = np.load(path)
    if "arr_0" not in data:
        raise KeyError(f"Expected 'arr_0' in PR reference npz: {path}")
    reference_images: UInt8Array = np.asarray(data["arr_0"], dtype=np.uint8)
    stats = compute_inception_stats(
        reference_images,
        len(reference_images),
        extractor=extractor,
        device=device,
        compute_logits=False,
        compute_features=True,
    )
    _pr_ref_features_cache = np.asarray(stats["features"], dtype=np.float32)
    _pr_ref_source_cache = path
    return _pr_ref_features_cache


def evaluate_fid(
    *,
    dataset_name: str,
    gen_func: GeneratorFn,
    eval_loader: Any,
    logger: LoggerLike,
    num_samples: int = 5000,
    log_folder: str = "fid",
    log_prefix: str = "gen_model",
    eval_prc_recall: bool = False,
    eval_isc: bool = True,
    eval_fid: bool = True,
    device: torch.device,
    extractor: nn.Module | None = None,
) -> MetricDict:
    """Generate evaluation samples and compute release metrics."""
    start = time.perf_counter()
    inception_extractor = extractor if extractor is not None else _get_inception_extractor(device)

    local_samples: list[np.ndarray] = []
    current = 0
    for eval_index, batch in enumerate(epoch0_sampler(eval_loader)):
        generated = gen_func(batch, eval_index)
        local_batch = _to_uint8(generated)
        local_samples.append(local_batch)
        current += int(local_batch.shape[0])
        if current >= num_samples:
            break

    local_sample_array = _stack_gathered_arrays(local_samples)
    local_masks = np.ones((len(local_sample_array),), dtype=np.float32)

    local_stats = compute_inception_stats(
        local_sample_array,
        num_samples=num_samples,
        extractor=inception_extractor,
        device=device,
        compute_logits=eval_isc,
        compute_features=True,
        masks=local_masks,
    )

    gathered_features = all_gather_objects(np.asarray(local_stats["features"], dtype=np.float32))
    features = _stack_gathered_arrays(gathered_features)[:num_samples]
    if len(features) == 0:
        raise ValueError("Expected at least one generated sample for evaluation.")
    stats: StatDict = {
        "mu": np.mean(np.asarray(features, dtype=np.float64), axis=0),
        "sigma": _covariance(np.asarray(features, dtype=np.float64)),
        "features": features,
    }

    if eval_isc:
        gathered_logits = all_gather_objects(np.asarray(local_stats["logits"], dtype=np.float32))
        stats["logits"] = _stack_gathered_arrays(gathered_logits)[:num_samples]

    gathered_preview = all_gather_objects(local_sample_array[:64])
    preview_images = _stack_gathered_arrays(gathered_preview)[:64]

    metrics: MetricDict = {}
    if eval_fid:
        reference_stats = _load_reference_stats(dataset_name)
        metrics["fid"] = compute_frechet_distance(
            reference_stats["mu"],
            stats["mu"],
            reference_stats["sigma"],
            stats["sigma"],
        )
    if eval_isc and "logits" in stats:
        inception_mean, inception_std = compute_inception_score(stats["logits"])
        metrics["isc_mean"] = inception_mean
        metrics["isc_std"] = inception_std
    if eval_prc_recall:
        reference_features = _load_pr_reference_features(
            extractor=inception_extractor,
            device=device,
        )
        precision, recall = compute_precision_recall(reference_features, stats["features"], k=3)
        metrics["precision"] = precision
        metrics["recall"] = recall

    metrics["fid_time"] = float(time.perf_counter() - start)
    logger.log_dict({f"{log_folder}/{log_prefix}_{key}": value for key, value in metrics.items()})
    if len(preview_images) > 0:
        logger.log_image(f"{log_folder}/{log_prefix}_viz", preview_images)
    barrier()
    return metrics
