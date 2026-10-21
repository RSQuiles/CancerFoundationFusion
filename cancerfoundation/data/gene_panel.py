"""Deterministic gene-panel selection for embedding.

Two schemes, both assuming log1p input (raw counts are log1p'd internally for
scoring only):

  * per-modality  -- scanpy HVG for single-cell, log1p + MAD for bulk/pseudobulk.
    Appropriate when a single modality is being embedded on its own.
  * shared panel  -- ONE panel for every modality group present, via
    :func:`select_shared_panel`. Required for any cross-modality comparison: a
    per-modality panel makes the model's input distribution differ *with* modality,
    which no batch-integration metric can distinguish from a real batch effect.

Kept free of torch/lightning/bionemo imports so it can be exercised without a
checkpoint or the training stack.
"""

from typing import List, Optional

import numpy as np
import scanpy as sc

from cancerfoundation.data.preprocess import looks_like_counts

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


def _mad_scores(X) -> np.ndarray:
    """Per-gene MAD (median absolute deviation), higher = more variable.

    MAD is computed on the values as passed — in this pipeline the data is already
    log1p-transformed by the time gene selection runs, so this is log1p + MAD.
    Robust to the composition-driven variance that dominates bulk/pseudobulk
    expression. Raw counts are log1p'd first so the score is on a comparable scale.
    """
    X = X if isinstance(X, np.ndarray) else X.toarray()

    if looks_like_counts(X):
        print("  [INFO] Non-log1p data detected. Normalizing for gene scoring!")
        X = np.log1p(X)

    med = np.median(X, axis=0)
    return np.median(np.abs(X - med), axis=0)


def _top_mad_genes(X, n_top: int) -> np.ndarray:
    """Return indices of the ``n_top`` genes (columns) with highest MAD."""
    return np.argsort(_mad_scores(X))[-n_top:]


def _dispersion_scores(data, flavor: str = "seurat") -> np.ndarray:
    """Per-gene variability score from scanpy's HVG routine, higher = more variable.

    For the ``seurat``/``cell_ranger`` flavors this is ``dispersions_norm``, the
    quantity ``sc.pp.highly_variable_genes`` ranks on, so taking its top ``n``
    reproduces that call's selection. For ``seurat_v3`` it is ``variances_norm``.
    Returned as a plain array so it can be rank-aggregated against ``_mad_scores``.
    """
    work = data.copy()
    if looks_like_counts(work.X) and flavor != "seurat_v3":
        sc.pp.log1p(work)
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


def _gene_scores(data, kind: str, flavor: str = "seurat") -> np.ndarray:
    """Per-gene variability score for one modality group, higher = more variable.

    Dispatches on modality the same way ``_select_genes`` does: scanpy dispersion for
    single-cell, MAD for bulk/pseudobulk. Assumes ``data`` is already log1p (or will
    be log1p'd internally by ``_mad_scores``).
    """
    if kind == "sc":
        return _dispersion_scores(data, flavor=flavor)
    return _mad_scores(data.X)


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
        scores = _gene_scores(data, kind, flavor=flavor)
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
        scores = _gene_scores(sub, _classify_modality(ref), flavor=flavor)
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
        scores = _gene_scores(sub, kind, flavor=flavor)
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
