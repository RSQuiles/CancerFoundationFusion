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


# Strip trailing _N or -N suffixes to derive a file's group key.
# e.g. "train_2" → "train", "liver-0" → "liver", "brain" → "brain".
_TRAILING_NUM_RE = re.compile(r"[_\-]?\d+$")


def _file_group(path: Path) -> str:
    """Return a grouping key for a file based on its stem (strips trailing numbers)."""
    stripped = _TRAILING_NUM_RE.sub("", path.stem)
    return stripped or path.stem  # fallback: keep full stem if nothing remains


def _select_representative_files(
    files: list[Path],
    max_files: int,
    seed: int | None,
) -> list[Path]:
    """
    Select up to ``max_files`` h5ad files with equal representation per group.

    Algorithm
    ---------
    1. Group files by their *natural prefix* (stem with trailing digits removed).
    2. Divide ``max_files`` slots equally across groups (floor division);
       any remainder is distributed one extra slot at a time, prioritising
       groups that still have unsampled files.
    3. Each group's allocation is capped at its actual size so a small group
       cannot be over-drawn.
    4. Within each group, files are chosen at random.

    The result is that every group contributes roughly the same number of
    files regardless of how many files it contains, preventing a large group
    from dominating the selection.
    """
    rng = random.Random(seed)

    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(_file_group(f), []).append(f)

    group_names = list(groups.keys())
    n_groups = len(group_names)

    base = max_files // n_groups
    remainder = max_files % n_groups

    # Groups are shuffled so the remainder bonus is not always given to the
    # same group(s) across calls with different seeds.
    order = list(range(n_groups))
    rng.shuffle(order)

    allocs = [base] * n_groups
    for i in order[:remainder]:
        allocs[i] += 1

    # Cap each allocation at the group's actual size.
    allocs = [min(a, len(groups[group_names[i]])) for i, a in enumerate(allocs)]

    selected: list[Path] = []
    for g, alloc in zip(group_names, allocs):
        pool = list(groups[g])
        rng.shuffle(pool)
        selected.extend(pool[:alloc])

    return sorted(selected, key=lambda p: p.name)


def sample_h5ad_subset_from_prefix(
    prefix: str | None,
    directory: Union[str, Path],
    n_subset: int,
    seed: int | None = None,
    max_files: int = 6,
) -> AnnData:
    """
    Load a representative subset of .h5ad files in a directory, combine their
    observations virtually, and return a random subset of ``n_subset``
    observations as a single AnnData object.

    Only ``max_files`` files are ever opened for cell counting and reading,
    keeping memory and I/O overhead low even when the directory contains many
    files.  Files are selected so that every distinct filename-prefix group
    (e.g. "brain", "liver") is represented by at least one file; remaining
    slots are filled with random picks from the full pool.

    Args:
        prefix:    Filename prefix filter.  If None or empty, all .h5ad files
                   in the directory are candidates.
        directory: Directory containing .h5ad files.
        n_subset:  Number of observations (cells/rows) to return.
        seed:      Random seed for reproducibility.
        max_files: Maximum number of files to load (default: 2).  Set to a
                   large number to disable the cap.

    Returns:
        AnnData containing the sampled observations.

    Raises:
        FileNotFoundError: If no matching .h5ad files are found.
        ValueError:        If n_subset is invalid or exceeds total available cells.
    """
    directory = Path(directory)
    pattern = f"{prefix}*.h5ad" if prefix else "*.h5ad"
    all_files = sorted(directory.glob(pattern))

    if not all_files:
        raise FileNotFoundError(
            f"No .h5ad files found in {directory}"
            + (f" with prefix '{prefix}'." if prefix else ".")
        )

    # Select a small representative set of files.
    files = _select_representative_files(all_files, max_files=max_files, seed=seed)
    if len(files) < len(all_files):
        print(
            f"Sampling from {len(files)}/{len(all_files)} files "
            f"({', '.join(f.name for f in files)})"
        )

    if n_subset <= 0:
        raise ValueError("n_subset must be a positive integer.")

    # First pass: get number of observations in each selected file.
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