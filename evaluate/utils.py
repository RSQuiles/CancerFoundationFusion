"""Shared utilities for evaluate/ scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_looks_like_counts():
    """Import ``looks_like_counts`` without requiring the training stack.

    ``cancerfoundation/__init__.py`` eagerly imports the dataset (and therefore
    bionemo, tokenizers, pytorch_lightning), which the scIB evaluation environment
    does not have — that is why ``unified_metrics.py`` guards its CancerFoundation
    import. The helper itself only needs numpy/scipy, so fall back to loading its
    module by path rather than duplicating the function.
    """
    try:
        from cancerfoundation.data.preprocess import looks_like_counts

        return looks_like_counts
    except Exception:
        import importlib.util

        path = _PROJECT_ROOT / "cancerfoundation" / "data" / "preprocess.py"
        spec = importlib.util.spec_from_file_location("_cf_preprocess_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.looks_like_counts


looks_like_counts = _import_looks_like_counts()

# ---------------------------------------------------------------------------
# Canonical ``_eval_modality`` vocabulary
# ---------------------------------------------------------------------------
# Single source of truth for the labels build_eval_adata.py writes and that
# unified_metrics.py / plot/umaps.py / check/diagnose_scib.py match on. These used
# to be hard-coded strings in each script, which is how "subsampled" (a *filename
# prefix*, never a label) ended up in three consumers and silently suppressed the
# agg_synth_* metrics. Import these instead of writing the literals.
MOD_SC = "sc"
MOD_BULK = "bulk"
MOD_PB = "pseudobulk"
MOD_PAIRED_SC = "paired_sc"
MOD_PAIRED_PB = "paired_pb"
MOD_PAIRED_BULK = "paired_bulk"
MOD_SYNTH_PB = "synth_pb"

# Groupings used by the metric and plotting code.
SC_MODALITIES = (MOD_SC, MOD_PAIRED_SC)
BULK_MODALITIES = (MOD_BULK, MOD_PAIRED_BULK)
PB_MODALITIES = (MOD_PB, MOD_PAIRED_PB, MOD_SYNTH_PB)


def generate_pseudobulk_adata(
    sc_adata: ad.AnnData,
    group_column: str = "tissue_general",
    n_sc_per_pb: int = 10,
    agg_method: str = "sum",
    n_pb: int | None = None,
    seed: int = 0,
    is_log1p: bool | None = None,
    normalize: bool = False,
) -> ad.AnnData | None:
    """Aggregate SC expression within tissue groups to create pseudobulk profiles.

    For each pseudobulk, ``n_sc_per_pb`` cells are drawn from a single randomly
    chosen ``group_column`` group and their expression vectors are aggregated
    (element-wise mean or sum), mirroring ``BulkSCSampler``.

    Args:
        is_log1p: whether ``sc_adata.X`` holds log1p values, which must be mapped
            back to count space with ``expm1`` before aggregating. ``None``
            (default) auto-detects via :func:`looks_like_counts`. Passing ``True``
            for a raw-counts matrix saturates float32 and produces a badly
            distorted profile, so the detection result is logged.

    Returns an AnnData with pseudobulk expression profiles sharing the same
    ``var`` as ``sc_adata``, or ``None`` if generation is not possible.
    """
    import scipy.sparse as sp

    if group_column not in sc_adata.obs.columns:
        return None

    if is_log1p is None:
        is_log1p = not looks_like_counts(sc_adata.X)
        print(
            f"  [generate_pseudobulk_adata] auto-detected input as "
            f"{'log1p' if is_log1p else 'raw counts'} "
            f"({'expm1 before aggregating' if is_log1p else 'summing counts directly'})"
        )

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
