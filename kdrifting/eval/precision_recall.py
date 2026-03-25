"""Precision and recall helpers for generator evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np


def _batch_pairwise_distances(features_u: np.ndarray, features_v: np.ndarray) -> np.ndarray:
    norm_u = np.sum(np.square(features_u), axis=1, dtype=np.float64).reshape(-1, 1)
    norm_v = np.sum(np.square(features_v), axis=1, dtype=np.float64).reshape(1, -1)
    distances = np.maximum(norm_u - 2.0 * np.matmul(features_u, features_v.T) + norm_v, 0.0)
    return distances


def _numpy_partition(
    array: np.ndarray,
    kth: np.ndarray,
    *,
    axis: int,
) -> list[np.ndarray]:
    workers = min(cpu_count(), len(array))
    chunk_size = len(array) // workers
    extra = len(array) % workers

    start_index = 0
    batches: list[np.ndarray] = []
    for worker_index in range(workers):
        size = chunk_size + (1 if worker_index < extra else 0)
        batches.append(array[start_index : start_index + size])
        start_index += size

    with ThreadPool(workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, axis=axis), batches))


class DistanceBlock:
    """Pairwise distance helper used by manifold precision/recall."""

    def pairwise_distances(self, features_u: np.ndarray, features_v: np.ndarray) -> np.ndarray:
        lhs = np.asarray(features_u, dtype=np.float64)
        rhs = np.asarray(features_v, dtype=np.float64)
        return _batch_pairwise_distances(lhs, rhs)

    def less_thans(
        self,
        batch_1: np.ndarray,
        radii_1: np.ndarray,
        batch_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lhs = np.asarray(batch_1, dtype=np.float64)
        rhs = np.asarray(batch_2, dtype=np.float64)
        lhs_radii = np.asarray(radii_1, dtype=np.float64)
        rhs_radii = np.asarray(radii_2, dtype=np.float64)
        distances = _batch_pairwise_distances(lhs, rhs)
        batch_1_in = np.any(distances[..., None] <= rhs_radii, axis=1)
        batch_2_in = np.any(distances[..., None] <= lhs_radii[:, None], axis=0)
        return batch_1_in, batch_2_in


class ManifoldEstimator:
    """Estimate feature manifolds for precision and recall."""

    def __init__(
        self,
        *,
        row_batch_size: int = 10000,
        col_batch_size: int = 10000,
        nhood_sizes: tuple[int, ...] = (3,),
        clamp_to_percentile: float | None = None,
        eps: float = 1e-5,
    ) -> None:
        self.distance_block = DistanceBlock()
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)
        radii = np.zeros((num_images, self.num_nhoods), dtype=np.float64)
        distance_batch = np.zeros((self.row_batch_size, num_images), dtype=np.float64)
        sequence = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin_1 in range(0, num_images, self.row_batch_size):
            end_1 = min(begin_1 + self.row_batch_size, num_images)
            row_batch = features[begin_1:end_1]

            for begin_2 in range(0, num_images, self.col_batch_size):
                end_2 = min(begin_2 + self.col_batch_size, num_images)
                col_batch = features[begin_2:end_2]
                distance_batch[0 : end_1 - begin_1, begin_2:end_2] = (
                    self.distance_block.pairwise_distances(row_batch, col_batch)
                )

            radii[begin_1:end_1, :] = np.concatenate(
                [
                    partition[:, self.nhood_sizes]
                    for partition in _numpy_partition(
                        distance_batch[0 : end_1 - begin_1, :],
                        sequence,
                        axis=1,
                    )
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0.0

        return radii

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        features_1_status = np.zeros((len(features_1), radii_2.shape[1]), dtype=bool)
        features_2_status = np.zeros((len(features_2), radii_1.shape[1]), dtype=bool)

        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = min(begin_1 + self.row_batch_size, len(features_1))
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = min(begin_2 + self.col_batch_size, len(features_2))
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1,
                    radii_1[begin_1:end_1],
                    batch_2,
                    radii_2[begin_2:end_2],
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in

        precision = np.mean(features_2_status.astype(np.float64), axis=0)
        recall = np.mean(features_1_status.astype(np.float64), axis=0)
        return precision, recall


def compute_precision_recall(
    features_real: np.ndarray,
    features_fake: np.ndarray,
    *,
    k: int | Sequence[int] = 3,
) -> tuple[float, float]:
    """Compute precision and recall over pooled Inception features."""
    real = np.asarray(features_real, dtype=np.float64)
    fake = np.asarray(features_fake, dtype=np.float64)
    nhood_sizes = (k,) if isinstance(k, int) else tuple(int(value) for value in k)
    estimator = ManifoldEstimator(nhood_sizes=nhood_sizes)
    radii_real = estimator.manifold_radii(real)
    radii_fake = estimator.manifold_radii(fake)
    precision, recall = estimator.evaluate_pr(real, radii_real, fake, radii_fake)
    return float(precision[0]), float(recall[0])
