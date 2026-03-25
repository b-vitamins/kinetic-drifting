"""Runtime paths and artifact identifiers for the PyTorch port."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    imagenet_path: str = "/path/to/imagenet"
    imagenet_cache_path: str = ""
    imagenet_fid_npz: str = "/path/to/imagenet_256_fid_stats.npz"
    imagenet_pr_npz: str = "/path/to/imagenet_val_prc_arr0.npz"
    hf_repo_id: str = "Goodeat/drifting"
    hf_root: str = "/path/to/hf_cache"


def runtime_paths() -> RuntimePaths:
    """Build runtime paths from environment variables when available."""
    return RuntimePaths(
        imagenet_path=os.environ.get("IMAGENET_PATH", "/path/to/imagenet"),
        imagenet_cache_path=os.environ.get("IMAGENET_CACHE_PATH", ""),
        imagenet_fid_npz=os.environ.get(
            "IMAGENET_FID_NPZ",
            "/path/to/imagenet_256_fid_stats.npz",
        ),
        imagenet_pr_npz=os.environ.get(
            "IMAGENET_PR_NPZ",
            "/path/to/imagenet_val_prc_arr0.npz",
        ),
        hf_repo_id=os.environ.get("HF_REPO_ID", "Goodeat/drifting"),
        hf_root=os.environ.get("HF_ROOT", "/path/to/hf_cache"),
    )
