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
# obs column holding the labels below.
MODALITY_COL = "_eval_modality"

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


def detect_scale_by_modality(adata, modality_col: str = "_eval_modality") -> dict:
    """Per-modality verdict on whether the rows hold raw counts.

    A single global :func:`looks_like_counts` call over an eval AnnData is not
    trustworthy, because the modalities genuinely disagree on scale:
    ``pseudobulk_paired_generation.py`` log1p's its SC rows unconditionally while
    honouring ``use_counts`` for PB and bulk, so a ``*_counts`` dataset ends up with
    log1p ``paired_sc`` rows next to raw-count ``paired_pb`` / ``paired_bulk`` /
    ``sc`` / ``bulk`` rows. The global call then samples whichever group is largest
    and reports one answer for all of them, which is how ``expm1`` came to be
    applied to raw counts.

    Returns ``{modality_label: is_counts}``. Groups too small or all-zero come back
    ``False`` (treated as already-log1p, the conservative choice: it skips a
    transform rather than applying a wrong one).
    """
    if modality_col not in adata.obs.columns:
        return {}
    labels = adata.obs[modality_col].astype(str).to_numpy()
    out: dict[str, bool] = {}
    for label in dict.fromkeys(labels.tolist()):
        rows = np.where(labels == label)[0]
        if len(rows) == 0:
            continue
        out[label] = bool(looks_like_counts(adata.X[rows]))
    return out


def log_scale_table(scale: dict, log=None) -> None:
    """Log a one-line-per-modality summary of :func:`detect_scale_by_modality`."""
    if not scale:
        return
    emit = log.info if log is not None else print
    emit("  Expression scale per modality:")
    for label, is_counts in scale.items():
        emit(f"    {label:<14} {'raw counts' if is_counts else 'log1p'}")
    if len(set(scale.values())) > 1:
        emit(
            "    NOTE mixed scales in one AnnData - each group is handled "
            "separately; do not trust a single global counts/log1p verdict here."
        )


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
    # Preallocate and fill in place. Accumulating a list of n_pb row arrays and then
    # calling np.array() on it holds two full copies at once, which at
    # n_pb = 75_000 x 28_725 genes is ~17 GB instead of ~8.6 GB.
    pb_X = np.zeros((n_pb, sc_adata.n_vars), dtype=np.float32)
    for i, g in enumerate(chosen_groups):
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
        pb_X[i] = np.asarray(agg, dtype=np.float32).ravel()

    pb_obs = pd.DataFrame(
        {"modality": ["pseudobulk"] * n_pb, group_column: chosen_groups},
        index=[f"pb_{i}" for i in range(n_pb)],
    )
    # Sparse, so concatenating this onto a sparse eval AnnData cannot densify the
    # whole thing. Pseudobulk profiles are dense-ish but the caller's matrix is not.
    return ad.AnnData(X=sp.csr_matrix(pb_X), obs=pb_obs, var=sc_adata.var.copy())
