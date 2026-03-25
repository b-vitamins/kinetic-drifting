from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from kdrifting.eval.fid import compute_frechet_distance
from kdrifting.eval.generation import compute_inception_score, compute_inception_stats, evaluate_fid
from kdrifting.eval.precision_recall import compute_precision_recall

UPSTREAM_ROOT = Path("/home/b/projects/drifting")


def _import_upstream(module_name: str) -> Any:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module_name)


class _FakeExtractor(nn.Module):
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        channel_mean = images.mean(dim=(2, 3))
        pooled = torch.cat([channel_mean, channel_mean.mean(dim=1, keepdim=True)], dim=1)
        logits = torch.stack(
            [
                pooled[:, 0] + pooled[:, 1],
                pooled[:, 1] + pooled[:, 2],
                pooled[:, 2] + pooled[:, 3],
            ],
            dim=1,
        )
        return pooled, logits


class _RecordingLogger:
    def __init__(self) -> None:
        self.logged_dicts: list[dict[str, float]] = []
        self.logged_images: list[tuple[str, np.ndarray | torch.Tensor]] = []

    def log_dict(self, values: dict[str, float]) -> None:
        self.logged_dicts.append(values)

    def log_image(self, name: str, images: np.ndarray | torch.Tensor) -> None:
        self.logged_images.append((name, images))


def _samples_from_labels(labels: np.ndarray, *, image_size: int = 8) -> np.ndarray:
    base = np.zeros((len(labels), image_size, image_size, 3), dtype=np.uint8)
    base[..., 0] = (labels[:, None, None] * 17 + 23) % 255
    base[..., 1] = (labels[:, None, None] * 29 + 41) % 255
    base[..., 2] = (labels[:, None, None] * 37 + 59) % 255
    return base


def _loader(num_items: int, *, batch_size: int = 4) -> DataLoader[tuple[Tensor, Tensor]]:
    labels = torch.arange(num_items, dtype=torch.int64)
    images = torch.zeros(num_items, 8, 8, 3)
    dataset = TensorDataset(images, labels)
    return cast(
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
    )


def test_compute_frechet_distance_is_zero_for_matching_stats() -> None:
    mu = np.array([1.0, 2.0], dtype=np.float64)
    sigma = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)

    value = compute_frechet_distance(mu, mu, sigma, sigma)

    assert abs(value) < 1e-12


def test_compute_precision_recall_is_one_for_identical_features() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    precision, recall = compute_precision_recall(features, features, k=1)

    assert abs(precision - 1.0) < 1e-12
    assert abs(recall - 1.0) < 1e-12


def test_compute_inception_score_returns_finite_values() -> None:
    logits = np.array(
        [
            [4.0, 1.0, 0.5],
            [3.0, 1.5, 0.5],
            [0.5, 4.0, 1.0],
            [1.0, 3.0, 0.5],
            [0.5, 1.0, 4.0],
            [0.5, 1.5, 3.5],
            [4.5, 0.5, 1.0],
            [3.5, 0.5, 1.5],
            [1.0, 4.0, 0.5],
            [0.5, 3.5, 1.0],
        ],
        dtype=np.float64,
    )

    mean, std = compute_inception_score(logits, splits=5)

    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert mean > 1.0


def test_compute_frechet_distance_matches_upstream_jax_helper() -> None:
    upstream_fid = _import_upstream("utils.jax_fid.fid")
    mu_1 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    mu_2 = np.array([-0.5, 3.0, 1.5], dtype=np.float64)
    sigma_1 = np.array(
        [
            [2.0, 0.25, 0.1],
            [0.25, 1.5, -0.05],
            [0.1, -0.05, 0.8],
        ],
        dtype=np.float64,
    )
    sigma_2 = np.array(
        [
            [1.8, -0.15, 0.2],
            [-0.15, 2.2, 0.05],
            [0.2, 0.05, 1.1],
        ],
        dtype=np.float64,
    )

    expected = float(upstream_fid.compute_frechet_distance(mu_1, mu_2, sigma_1, sigma_2))
    actual = compute_frechet_distance(mu_1, mu_2, sigma_1, sigma_2)

    assert abs(actual - expected) < 1e-12


def test_compute_inception_score_matches_upstream_jax_helper() -> None:
    upstream_fid_util = _import_upstream("utils.fid_util")
    logits = np.array(
        [
            [4.0, 1.0, 0.5],
            [3.0, 1.5, 0.5],
            [0.5, 4.0, 1.0],
            [1.0, 3.0, 0.5],
            [0.5, 1.0, 4.0],
            [0.5, 1.5, 3.5],
            [4.5, 0.5, 1.0],
            [3.5, 0.5, 1.5],
            [1.0, 4.0, 0.5],
            [0.5, 3.5, 1.0],
        ],
        dtype=np.float64,
    )

    expected_mean, expected_std = upstream_fid_util._compute_inception_score(logits, splits=5)
    actual_mean, actual_std = compute_inception_score(logits, splits=5)

    assert abs(actual_mean - expected_mean) < 1e-7
    assert abs(actual_std - expected_std) < 1e-7


def test_compute_precision_recall_matches_upstream_jax_helper() -> None:
    jax_module = _import_upstream("jax")
    jax_module.config.update("jax_enable_x64", True)
    upstream_pr = _import_upstream("utils.jax_fid.precision_recall")
    features_real = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float64,
    )
    features_fake = np.array(
        [
            [0.1, 0.0],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.9],
            [0.45, 0.3],
        ],
        dtype=np.float64,
    )

    expected_precision, expected_recall = upstream_pr.compute_precision_recall(
        features_real,
        features_fake,
        k=3,
    )
    actual_precision, actual_recall = compute_precision_recall(features_real, features_fake, k=3)

    assert abs(actual_precision - float(expected_precision)) < 1e-12
    assert abs(actual_recall - float(expected_recall)) < 1e-12


def test_compute_inception_stats_collects_features_and_logits() -> None:
    labels = np.arange(6, dtype=np.int64)
    samples = _samples_from_labels(labels)

    stats = compute_inception_stats(
        samples,
        num_samples=len(samples),
        extractor=_FakeExtractor(),
        device=torch.device("cpu"),
        compute_logits=True,
        compute_features=True,
    )

    assert stats["mu"].shape == (4,)
    assert stats["sigma"].shape == (4, 4)
    assert stats["features"].shape == (6, 4)
    assert stats["logits"].shape == (6, 3)


def test_evaluate_fid_matches_reference_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.arange(20, dtype=np.int64)
    samples = _samples_from_labels(labels)
    extractor = _FakeExtractor()
    reference_stats = compute_inception_stats(
        samples,
        num_samples=len(samples),
        extractor=extractor,
        device=torch.device("cpu"),
        compute_logits=False,
        compute_features=True,
    )
    fid_path = tmp_path / "imagenet_fid.npz"
    np.savez(fid_path, mu=reference_stats["mu"], sigma=reference_stats["sigma"])
    pr_path = tmp_path / "imagenet_pr.npz"
    np.savez(pr_path, arr_0=samples)
    monkeypatch.setenv("IMAGENET_FID_NPZ", str(fid_path))
    monkeypatch.setenv("IMAGENET_PR_NPZ", str(pr_path))

    import kdrifting.eval.generation as generation_module

    monkeypatch.setattr(generation_module, "_pr_ref_features_cache", None)
    monkeypatch.setattr(generation_module, "_pr_ref_source_cache", None)

    logger = _RecordingLogger()

    def gen_func(batch: tuple[Tensor, Tensor], eval_index: int) -> Tensor:
        del eval_index
        _, batch_labels = batch
        generated = _samples_from_labels(batch_labels.cpu().numpy())
        return (
            torch.tensor(
                generated.tolist(),
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
            / 255.0
        )

    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_func,
        eval_loader=_loader(len(labels)),
        logger=logger,
        num_samples=len(labels),
        log_folder="CFG1",
        log_prefix="EMA_0.9",
        eval_prc_recall=True,
        eval_isc=True,
        eval_fid=True,
        device=torch.device("cpu"),
        extractor=extractor,
    )

    assert abs(metrics["fid"]) < 1e-7
    assert abs(metrics["precision"] - 1.0) < 1e-12
    assert abs(metrics["recall"] - 1.0) < 1e-12
    assert metrics["isc_mean"] > 1.0
    assert logger.logged_dicts
    assert "CFG1/EMA_0.9_fid" in logger.logged_dicts[0]
    assert logger.logged_images[0][0] == "CFG1/EMA_0.9_viz"
