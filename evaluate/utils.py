"""Shared utilities for evaluate/ scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad


def generate_pseudobulk_adata(
    sc_adata: ad.AnnData,
    group_column: str = "tissue_general",
    n_sc_per_pb: int = 10,
    agg_method: str = "sum",
    n_pb: int | None = None,
    seed: int = 0,
    is_log1p: bool = True,
    normalize: bool = False,
) -> ad.AnnData | None:
    """Aggregate SC expression within tissue groups to create pseudobulk profiles.

    For each pseudobulk, ``n_sc_per_pb`` cells are drawn from a single randomly
    chosen ``group_column`` group and their expression vectors are aggregated
    (element-wise mean or sum), mirroring ``BulkSCSampler``.

    Returns an AnnData with pseudobulk expression profiles sharing the same
    ``var`` as ``sc_adata``, or ``None`` if generation is not possible.
    """
    import scipy.sparse as sp

    if group_column not in sc_adata.obs.columns:
        return None

    rng = np.random.default_rng(seed)

    group_vals    = sc_adata.obs[group_column].astype(str).to_numpy()
    unique_groups = np.unique(group_vals)
    group_to_idx: dict[str, np.ndarray] = {
        g: np.where(group_vals == g)[0] for g in unique_groups
    }
    valid_groups = [g for g, idx in group_to_idx.items() if len(idx) >= n_sc_per_pb]

    if not valid_groups:
        return None

    if n_pb is None:
        n_pb = max(1, sc_adata.n_obs // n_sc_per_pb)

    X = sc_adata.X
    is_sparse = sp.issparse(X)

    chosen_groups: list[str] = rng.choice(valid_groups, size=n_pb, replace=True).tolist()
    pb_rows: list[np.ndarray] = []
    for g in chosen_groups:
        pool    = group_to_idx[g]
        sel_idx = rng.choice(pool, size=n_sc_per_pb, replace=len(pool) < n_sc_per_pb)
        expr    = X[sel_idx].toarray() if is_sparse else np.asarray(X[sel_idx])
        if is_log1p:
            expr = np.expm1(expr)
        agg = expr.sum(axis=0) if agg_method == "sum" else expr.mean(axis=0)
        if is_log1p and normalize:
            total = agg.sum()
            if total > 0:
                agg = agg / total * 1e4
            agg = np.log1p(agg)
        pb_rows.append(agg)

    pb_X = np.array(pb_rows, dtype=np.float32)
    pb_obs = pd.DataFrame(
        {"modality": ["pseudobulk"] * n_pb, group_column: chosen_groups},
        index=[f"pb_{i}" for i in range(n_pb)],
    )
    return ad.AnnData(X=pb_X, obs=pb_obs, var=sc_adata.var.copy())
