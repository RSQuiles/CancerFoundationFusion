"""Shared utilities for evaluate/ scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_looks_like_counts():
    """Import ``looks_like_counts`` without requiring the whole training stack.

    ``cancerfoundation/__init__.py`` eagerly imports the dataset (and therefore
    bionemo, tokenizers, pytorch_lightning), which the scIB evaluation environment
    does not have — that is why ``unified_metrics.py`` guards its CancerFoundation
    import. Falling back to loading ``preprocess.py`` by path skips that, though the
    module itself still imports torch.
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


_looks_like_counts_impl = None


def looks_like_counts(X):
    """Lazy proxy for ``cancerfoundation.data.preprocess.looks_like_counts``.

    Resolved on first call rather than at import, so that importing this module for
    the modality vocabulary alone — as ``check/check_modality_labels.py`` and the
    plain-environment steps do — does not require torch.
    """
    global _looks_like_counts_impl
    if _looks_like_counts_impl is None:
        _looks_like_counts_impl = _import_looks_like_counts()
    return _looks_like_counts_impl(X)

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

CANONICAL_MODALITIES = (
    MOD_SC, MOD_BULK, MOD_PB, MOD_PAIRED_SC, MOD_PAIRED_PB, MOD_PAIRED_BULK,
    MOD_SYNTH_PB,
)

# Filename prefix -> label, in the order build_eval_adata.py globs them. The glob is
# ``{prefix}*.h5ad`` and prefix-anchored, so "sc" never picks up "paired_sc_*.h5ad"
# and "bulk" never picks up "pseudo_bulk_*.h5ad"; the order still matters for the
# order modalities are concatenated in.
MODALITY_FILE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("subsampled",       MOD_SC),
    ("partition",        MOD_SC),
    ("sc",               MOD_SC),
    ("pretraining_sc",   MOD_SC),
    ("pretraining_bulk", MOD_BULK),
    ("pseudo_bulk",      MOD_PB),
    ("bulk",             MOD_BULK),
    ("paired_sc",        MOD_PAIRED_SC),
    ("paired_pb",        MOD_PAIRED_PB),
    ("paired_bulk",      MOD_PAIRED_BULK),
)

# Labels seen in the wild that are not canonical. Two sources:
#   * eval.h5ad files built before June 2026, when build_eval_adata.py wrote the
#     *filename prefix* into the obs column ("subsampled" for SC rows). Those files
#     are expensive to rebuild and are still the input to metrics/umaps/diagnose, so
#     they are translated on read rather than declared unreadable.
#   * hand-written or externally produced h5ads using an obvious synonym.
# Keys are matched case-insensitively with '-'/' ' folded to '_'; see
# :func:`canonical_modality`.
MODALITY_ALIASES: dict[str, str] = {
    # legacy prefix-as-label
    **{prefix: label for prefix, label in MODALITY_FILE_PREFIXES},
    # synonyms
    "single_cell": MOD_SC,
    "singlecell": MOD_SC,
    "scrna": MOD_SC,
    "sc_rna": MOD_SC,
    "cells": MOD_SC,
    "bulk_rna": MOD_BULK,
    "pb": MOD_PB,
    "pseudobulks": MOD_PB,
    "synthetic_pb": MOD_SYNTH_PB,
    "synth_pseudobulk": MOD_SYNTH_PB,
}


def canonical_modality(label: object) -> str:
    """Map one ``_eval_modality`` value onto the canonical vocabulary.

    Unknown labels are returned unchanged (lower-cased/underscored inputs that are
    already canonical pass through): a label this module has never heard of is not
    an error, it simply matches nothing, and silently renaming it to a canonical one
    would be worse than leaving it visible in the "rows available" line.
    """
    text = str(label).strip()
    if text in CANONICAL_MODALITIES:
        return text
    key = text.lower().replace("-", "_").replace(" ", "_")
    if key in CANONICAL_MODALITIES:
        return key
    return MODALITY_ALIASES.get(key, text)


def canonicalize_modality_column(
    adata, modality_col: str = MODALITY_COL, log=None
) -> dict[str, str]:
    """Rewrite ``adata.obs[modality_col]`` in place to canonical labels.

    Every consumer matches on the canonical labels, so an eval.h5ad carrying a legacy
    label silently loses whole metric families — SC rows written as "subsampled" make
    ``_get(MOD_SC)`` return None and every ``agg_synth_*`` metric is skipped with
    "missing 'sc' rows" while the modality is plainly listed as present. Call this
    once, immediately after reading the file.

    Returns ``{old_label: new_label}`` for the labels actually rewritten (empty when
    the file is already canonical), and logs both the rewrites and any label that
    matched nothing.
    """
    if modality_col not in adata.obs.columns:
        return {}

    old = adata.obs[modality_col].astype(str)
    renames = {
        lbl: canonical_modality(lbl)
        for lbl in dict.fromkeys(old.to_numpy().tolist())
    }
    unknown = sorted(
        lbl for lbl, new in renames.items() if new not in CANONICAL_MODALITIES
    )
    renames = {lbl: new for lbl, new in renames.items() if new != lbl}

    if renames:
        adata.obs[modality_col] = old.map(lambda v: renames.get(v, v)).astype(str)

    emit_info = log.info if log is not None else print
    emit_warn = log.warning if log is not None else print
    for lbl, new in renames.items():
        emit_info(
            f"  Canonicalized {modality_col} label {lbl!r} -> {new!r} "
            f"({int((old == lbl).sum())} rows)"
        )
    if unknown:
        emit_warn(
            f"  Unrecognised {modality_col} label(s): {unknown} - they match no "
            f"metric family. Known labels: {list(CANONICAL_MODALITIES)}"
        )
    return renames


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
    sc_adata: "ad.AnnData",
    group_column: str = "tissue_general",
    n_sc_per_pb: int = 10,
    agg_method: str = "sum",
    n_pb: int | None = None,
    seed: int = 0,
    is_log1p: bool | None = None,
    normalize: bool = False,
) -> "ad.AnnData | None":
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
    # Imported here, not at module scope: the modality vocabulary above is the one
    # thing every consumer shares, and check_modality_labels.py must be able to test
    # it in an environment with only numpy/pandas.
    import anndata as ad
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
