"""Deterministic gene-panel selection for embedding.

Two schemes:

  * single modality -- scanpy HVG for single-cell, MAD for bulk/pseudobulk
    (:func:`select_panel`). Appropriate when one modality is embedded on its own.
  * shared panel    -- ONE panel for every modality group present, via
    :func:`select_shared_panel`. Required for any cross-modality comparison: a
    per-modality panel makes the model's input distribution differ *with* modality,
    which no batch-integration metric can distinguish from a real batch effect.

Panel selection is independent of how the model's input is normalized. Each scorer
gets its own *scoring view* of the data (:func:`scoring_view`), transformed to meet
that scorer's assumption from whatever the values are:

  ==================  ============================  ===========================
  scorer              assumes                       raw counts become
  ==================  ============================  ===========================
  sc, seurat /        depth-normalized log values   CP10K + log1p
  cell_ranger HVG
  sc, seurat_v3 HVG   raw counts                    used as they are
  bulk / pseudobulk   log-scale, depth-normalized   CPM + log1p
  MAD
  ==================  ============================  ===========================

Linear-scale values that are not counts (TPM, CPM, CP10K) get log1p only; values
already on a log scale are used as they are. Library sizes come from
``obs[LIBRARY_SIZE_COL]`` when present, so a caller that has already dropped genes
(vocabulary intersection) still normalizes by the row's full total.
"""

from typing import List, Optional, Tuple

import numpy as np
import scanpy as sc

from cancerfoundation.data.preprocess import (
    library_size,
    looks_like_counts,
    normalize_expression,
    value_scale,
)

# obs column holding each row's total over ALL input genes (set by CancerFoundation.embed).
LIBRARY_SIZE_COL = "_cf_library_size"

_SC_MODALITY_NORM: frozenset = frozenset({
    "sc", "singlecell", "scrna", "scrnaseq", "subsampled", "pairedsc",
})
_PB_MODALITY_NORM: frozenset = frozenset({
    "pseudobulk", "synthpb", "pairedpb", "pseudo",
})

def _classify_modality(val: str) -> str:
    """Map a raw modality label to 'sc', 'pseudobulk', or 'bulk'."""
    norm = val.lower().replace(" ", "").replace("-", "").replace("_", "")
    if norm in _SC_MODALITY_NORM or norm.startswith("sc"):
        return "sc"
    if "pseudo" in norm or norm in _PB_MODALITY_NORM:
        return "pseudobulk"
    return "bulk"


def _dense(X) -> np.ndarray:
    return X if isinstance(X, np.ndarray) else X.toarray()


def _row_library_size(data) -> np.ndarray:
    if LIBRARY_SIZE_COL in data.obs.columns:
        return data.obs[LIBRARY_SIZE_COL].to_numpy(dtype=np.float64)
    return library_size(data.X)


def scoring_view(data, kind: str, flavor: str = "seurat", verbose: bool = True) -> Tuple[object, str]:
    """``(AnnData, flavor)`` whose X meets the assumption of the scorer for ``kind``.

    ``flavor`` may change: ``seurat_v3`` needs raw counts, so on anything else it
    falls back to ``seurat`` with a warning. See the module docstring for the rules.
    """
    import anndata as ad

    scale = value_scale(data.X)
    X = _dense(data.X)
    if kind == "sc" and flavor == "seurat_v3":
        if scale == "counts":
            how = "raw counts (seurat_v3)"
        else:
            print(f"  [panel] [WARNING] seurat_v3 needs raw counts but the values look {scale}; "
                  "using flavor 'seurat' instead")
            flavor = "seurat"
    if not (kind == "sc" and flavor == "seurat_v3"):
        if scale == "counts":
            method = "cp10k_log1p" if kind == "sc" else "cpm_log1p"
            X = normalize_expression(X, method, _row_library_size(data))
            how = ("CP10K" if kind == "sc" else "CPM") + " + log1p of the raw counts"
        elif scale == "linear":
            X = np.log1p(np.asarray(X, dtype=np.float32))
            how = "log1p of the (already depth-normalized) linear values"
        else:
            how = "values as they are (already log-scale)"
    if verbose:
        print(f"  [panel] scoring {kind} on {how}")
    return ad.AnnData(X=X, obs=data.obs[[]].copy(), var=data.var[[]].copy()), flavor


def _mad_scores(X) -> np.ndarray:
    """Per-gene MAD (median absolute deviation), higher = more variable.

    Robust to the composition-driven variance that dominates bulk/pseudobulk
    expression. Expects log-scale values (see :func:`scoring_view`); raw counts are
    still log1p'd here as a guard for direct callers.
    """
    X = _dense(X)
    if looks_like_counts(X):
        print("  [INFO] Non-log1p data detected. Normalizing for gene scoring!")
        X = np.log1p(X)

    med = np.median(X, axis=0)
    return np.median(np.abs(X - med), axis=0)


def _top_mad_genes(X, n_top: int) -> np.ndarray:
    """Return indices of the ``n_top`` genes (columns) with highest MAD."""
    return np.argsort(_mad_scores(X))[-n_top:]


def _dispersion_scores(view, flavor: str = "seurat") -> np.ndarray:
    """Per-gene variability score from scanpy's HVG routine, higher = more variable.

    For the ``seurat``/``cell_ranger`` flavors this is ``dispersions_norm``, the
    quantity ``sc.pp.highly_variable_genes`` ranks on, so taking its top ``n``
    reproduces that call's selection. For ``seurat_v3`` it is ``variances_norm``.
    ``view`` must already meet the flavor's assumption (:func:`scoring_view`).
    """
    work = view.copy()
    sc.pp.highly_variable_genes(work, flavor=flavor)
    for col in ("dispersions_norm", "variances_norm", "dispersions"):
        if col in work.var:
            return np.nan_to_num(
                work.var[col].to_numpy(dtype=np.float64), nan=-np.inf, neginf=-np.inf
            )
    raise RuntimeError(
        f"scanpy HVG (flavor={flavor!r}) produced no usable score column; "
        f"got {list(work.var.columns)}"
    )


def _gene_scores(data, kind: str, flavor: str = "seurat", verbose: bool = True) -> np.ndarray:
    """Per-gene variability score for one modality group, higher = more variable.

    Scanpy dispersion for single-cell, MAD for bulk/pseudobulk, each on its own
    scoring view of ``data`` (any scale; see :func:`scoring_view`).
    """
    view, flavor = scoring_view(data, kind, flavor=flavor, verbose=verbose)
    if kind == "sc":
        return _dispersion_scores(view, flavor=flavor)
    return _mad_scores(view.X)


def select_panel(data, kind: str, n_top: int, flavor: str = "seurat", verbose: bool = True) -> List[str]:
    """The ``n_top`` most variable genes of one modality, in ``data.var.index`` order."""
    scores = _gene_scores(data, kind, flavor=flavor, verbose=verbose)
    keep = np.sort(np.argsort(-scores, kind="stable")[:n_top])
    return data.var.index[keep].tolist()


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """Ranks of ``scores`` in descending order — rank 0 is the most variable gene."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    return ranks


def select_shared_panel(
    data,
    groups: np.ndarray,
    n_top: int,
    flavor: str = "seurat",
    strategy: str = "consensus",
    reference: Optional[str] = None,
    min_cells: int = 10,
    verbose: bool = True,
) -> List[str]:
    """Pick ONE gene panel for all modality groups in ``data``.

    Cross-modality comparisons (bulk vs pseudobulk alignment, pseudobulk vs mean-SC
    aggregation consistency) are only meaningful when both sides are embedded through
    the same genes. Fitting a panel per modality makes the input distribution differ
    *with* modality, which is indistinguishable from a batch effect.

    Strategies:
      ``consensus`` (default)
        Score genes *within* each group, convert to ranks, average the ranks, take
        the top ``n_top``. This is the principle behind
        ``sc.pp.highly_variable_genes(batch_key=...)`` and it matters here: a single
        fit over the pooled cells would rank bimodal genes highest, i.e. select
        precisely the genes that separate bulk from pseudobulk, biasing every
        alignment metric downward.
      ``reference``
        Use one named group's scores only. Simple to describe but asymmetric — genes
        variable in the reference may be all-zero elsewhere.
      ``pooled``
        One fit over all cells together. Kept for comparison; biased as above.

    Groups with fewer than ``min_cells`` cells are skipped, since their scores are
    noise. Returns gene names in ``data.var.index`` order.
    """
    groups = np.asarray(groups).astype(str)
    unique = list(dict.fromkeys(groups.tolist()))  # preserve first-seen order
    gene_names = data.var.index.to_numpy()

    if strategy == "pooled":
        kind = _classify_modality(unique[0]) if len(unique) == 1 else "bulk"
        scores = _gene_scores(data, kind, flavor=flavor, verbose=verbose)
        keep = np.argsort(-scores, kind="stable")[:n_top]
        if verbose:
            print(f"  [panel] pooled fit over {data.n_obs} cells -> {len(keep)} genes")
        return sorted(gene_names[np.sort(keep)].tolist())

    if strategy == "reference":
        ref = reference or unique[0]
        if ref not in unique:
            raise ValueError(
                f"panel_reference={ref!r} not among modality groups {unique}"
            )
        sub = data[groups == ref]
        scores = _gene_scores(sub, _classify_modality(ref), flavor=flavor, verbose=verbose)
        keep = np.argsort(-scores, kind="stable")[:n_top]
        if verbose:
            print(
                f"  [panel] reference fit on '{ref}' ({sub.n_obs} cells) "
                f"-> {len(keep)} genes, applied to all {len(unique)} group(s)"
            )
        return sorted(gene_names[np.sort(keep)].tolist())

    if strategy != "consensus":
        raise ValueError(
            f"Unknown panel strategy {strategy!r}; expected "
            "'consensus', 'reference', or 'pooled'."
        )

    # ── consensus: rank within each group, then average ranks ──────────────────
    rank_sum = np.zeros(data.n_vars, dtype=np.float64)
    n_used = 0
    own_panels: dict[str, set] = {}
    skipped: list[str] = []

    for g in unique:
        mask = groups == g
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            skipped.append(f"{g} ({n_cells} cells)")
            continue
        sub = data[mask]
        kind = _classify_modality(g)
        scores = _gene_scores(sub, kind, flavor=flavor, verbose=verbose)
        rank_sum += _rank_desc(scores)
        n_used += 1
        own_panels[g] = set(
            gene_names[np.argsort(-scores, kind="stable")[:n_top]].tolist()
        )
        if verbose:
            print(f"  [panel] scored '{g}' ({kind}): {n_cells} cells")

    if n_used == 0:
        # Every group was too small to score — fall back to a pooled fit rather
        # than returning an empty panel.
        if verbose:
            print(
                f"  [panel] no group reached min_cells={min_cells}; falling back to pooled"
            )
        return select_shared_panel(
            data, groups, n_top, flavor=flavor, strategy="pooled", verbose=verbose
        )

    keep = np.argsort(rank_sum / n_used, kind="stable")[:n_top]
    panel_names = gene_names[np.sort(keep)].tolist()

    if verbose:
        print(
            f"  [panel] consensus over {n_used} group(s) -> {len(panel_names)} genes"
            + (f"; skipped: {', '.join(skipped)}" if skipped else "")
        )
        # How much the retired per-modality scheme was diverging: the fraction of
        # each group's own panel that the shared panel retains.
        shared = set(panel_names)
        for g, own in own_panels.items():
            overlap = len(shared & own) / max(len(own), 1)
            print(f"  [panel]   retains {overlap:6.1%} of '{g}' own top-{n_top} panel")

    return sorted(panel_names)
