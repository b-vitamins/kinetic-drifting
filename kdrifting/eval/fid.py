"""Frechet distance helpers for FID evaluation."""

from __future__ import annotations

import numpy as np


def compute_frechet_distance(
    mu1: np.ndarray,
    mu2: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute the source-compatible Frechet distance between two Gaussians."""
    mean_1 = np.atleast_1d(mu1).astype(np.float64)
    mean_2 = np.atleast_1d(mu2).astype(np.float64)
    cov_1 = np.atleast_2d(sigma1).astype(np.float64)
    cov_2 = np.atleast_2d(sigma2).astype(np.float64)

    if mean_1.shape != mean_2.shape:
        raise ValueError(
            f"Expected matching mean shapes, got {mean_1.shape} and {mean_2.shape}.",
        )
    if cov_1.shape != cov_2.shape:
        raise ValueError(
            f"Expected matching covariance shapes, got {cov_1.shape} and {cov_2.shape}.",
        )

    diff = mean_1 - mean_2
    eigvals = np.asarray(np.linalg.eigvals(cov_1.dot(cov_2)), dtype=np.complex128)
    trace_covmean = float(np.sum(np.real(np.sqrt(eigvals)), dtype=np.float64))
    fid = float(diff.dot(diff) + np.trace(cov_1) + np.trace(cov_2) - 2.0 * trace_covmean)
    return fid
