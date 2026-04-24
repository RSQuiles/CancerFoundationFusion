from __future__ import annotations
from __future__ import print_function

from pathlib import Path
import random
import anndata as ad
from anndata import AnnData
import numpy as np
import ftplib
import logging
import os
from typing import List, Optional, Union
import gc

import json
import os
import struct
import sys
import platform
import re
import time
import traceback
import requests
import socket
import random
import math
import numpy as np
import torch
import logging
import datetime
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


def sample_h5ad_subset_from_prefix(
    prefix: str | None,
    directory: Union[str, Path],
    n_subset: int,
    seed: int | None = None,
) -> AnnData:
    """
    Load .h5ad files in a directory, combine their observations virtually, and
    return a random subset of `n_subset` observations as a single AnnData object.

    This function avoids concatenating all cells into memory first. It samples
    row indices across the union of all matching files, then reads only the
    selected observations from each file.

    Args:
        prefix: Filename prefix to match. If None or empty, all .h5ad files
                in the directory are used.
        directory: Directory containing .h5ad files.
        n_subset: Number of observations (cells/rows) to sample.
        seed: Optional random seed for reproducibility.

    Returns:
        AnnData containing the sampled observations.

    Raises:
        FileNotFoundError: If no matching .h5ad files are found.
        ValueError: If n_subset is invalid or larger than total observations.
    """
    directory = Path(directory)
    pattern = f"{prefix}*.h5ad" if prefix else "*.h5ad"
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .h5ad files found in {directory} with prefix '{prefix}'."
        )

    if n_subset <= 0:
        raise ValueError("n_subset must be a positive integer.")

    # First pass: get number of observations in each file.
    file_infos: list[tuple[Path, int]] = []
    total_obs = 0

    for file in files:
        adata_backed = ad.read_h5ad(file, backed="r")
        n_obs = adata_backed.n_obs
        adata_backed.file.close()

        if n_obs > 0:
            file_infos.append((file, n_obs))
            total_obs += n_obs

    if total_obs == 0:
        raise ValueError("All matching .h5ad files are empty.")

    if n_subset > total_obs:
        raise ValueError(
            f"Requested n_subset={n_subset}, but only {total_obs} total observations "
            f"are available across {len(file_infos)} files."
        )

    rng = random.Random(seed)

    # Sample global row indices from the combined dataset.
    sampled_global_indices = sorted(rng.sample(range(total_obs), n_subset))

    # Map sampled global indices back to per-file local indices.
    per_file_indices: dict[Path, list[int]] = {file: [] for file, _ in file_infos}

    cursor = 0
    sample_ptr = 0
    for file, n_obs in file_infos:
        file_start = cursor
        file_end = cursor + n_obs

        local_indices = []
        while sample_ptr < len(sampled_global_indices):
            global_idx = sampled_global_indices[sample_ptr]
            if global_idx >= file_end:
                break
            local_indices.append(global_idx - file_start)
            sample_ptr += 1

        if local_indices:
            per_file_indices[file] = local_indices

        cursor = file_end

    # Second pass: load only selected rows from each file.
    subsets: list[AnnData] = []
    for file, _ in file_infos:
        print(f"File: {file}")
        local_indices = per_file_indices[file]
        if not local_indices:
            continue

        adata = ad.read_h5ad(file, backed="r")
        try:
            subset = adata[local_indices].to_memory()
            subset.obs["_source_file"] = file.name
            subsets.append(subset)
        finally:
            adata.file.close()

        del adata
        gc.collect()

    if len(subsets) == 1:
        return subsets[0]

    out_adata = ad.concat(subsets, join="outer", merge="same", index_unique=None)
    out_adata.obs_names_make_unique()
    return out_adata


def subsample_adata(adata, n_subset, seed=None, copy=True):
    """
    Randomly subsample n_subset observations (cells) from an AnnData object.

    Args:
        adata: AnnData object
        n_subset: number of observations to sample
        seed: random seed (optional)
        copy: whether to return a copy (recommended)

    Returns:
        Subsampled AnnData
    """
    if n_subset > adata.n_obs:
        raise ValueError(f"Requested {n_subset} but adata only has {adata.n_obs} cells")

    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n_subset, replace=False)

    return adata[idx].copy() if copy else adata[idx]



