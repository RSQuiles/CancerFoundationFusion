"""Diagnose the scIB batch-correction numbers written by ``unified_metrics.py --scib``.

Reproduces the exact subset that ``run_scib_benchmark`` benchmarks (bulk vs
pseudobulk, ``n_max`` cells per side, same seed) and then reports, per model
embedding, the quantities the scIB metrics are built out of — so a
"metrics say separated / UMAP says overlapping" disagreement can be attributed
to a specific cause instead of guessed at.

What it prints
--------------
1. **Modality census** of the eval AnnData, including whether ``pseudobulk`` and
   ``synth_pb`` are the same underlying cells embedded twice.
2. **label x batch contingency** for the benchmarked subset.  BRAS and kBET are
   *label-conditioned*: they are computed inside each ``label_key`` group.  Any
   label that contains only one batch carries no batch-mixing information, so if
   bulk and pseudobulk barely share ``tissue_general`` values those two metrics
   are measuring almost nothing, while iLISI (label-free) still is.
3. **Geometry**: within-batch vs cross-batch mean L2, per-batch spread.  Two
   clouds can be co-located (overlapping in UMAP) yet have every k-NN of a
   pseudobulk point be another pseudobulk point, when one cloud is much tighter
   than the other.  That is the classic reason iLISI ~ 0 while a UMAP looks mixed.
4. **k-NN batch mixing** at several k, in the full embedding space: observed
   same-batch neighbour fraction vs the fraction expected under perfect mixing,
   plus a scaled inverse-Simpson (iLISI-equivalent) so the number can be compared
   to the scIB column directly.
5. **PCR**: the unscaled principal-component-regression values for the
   pre-integrated baseline (``X_pca``) and for the embedding.  ``pcr_comparison``
   reports ``max(0, (pcr_pre - pcr_post) / pcr_pre)``, so a column of exact zeros
   means ``pcr_post >= pcr_pre`` for every model — this shows by how much.

Usage
-----
    python diagnose_scib.py --eval-adata path/to/eval.h5ad
    python diagnose_scib.py --eval-adata path/to/eval.h5ad --group-column tissue_general
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.utils import (
    BULK_MODALITIES,
    MODALITY_COL as _MODALITY_COL,
    MOD_PB,
    MOD_SYNTH_PB,
    PB_MODALITIES,
    canonicalize_modality_column,
)


# ---------------------------------------------------------------------------
# Geometry / mixing helpers
# ---------------------------------------------------------------------------

def _pairwise_l2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    sq_x = (X ** 2).sum(axis=1)
    sq_y = (Y ** 2).sum(axis=1)
    return np.sqrt(np.clip(sq_x[:, None] + sq_y[None, :] - 2 * (X @ Y.T), 0, None))


def geometry_summary(emb: np.ndarray, batch: np.ndarray) -> dict:
    """Within- vs cross-batch mean L2 distance and per-batch spread."""
    labels = sorted(set(batch.tolist()))
    out: dict = {}
    for b in labels:
        E = emb[batch == b]
        if len(E) < 2:
            continue
        d = _pairwise_l2(E, E)
        m = ~np.eye(len(E), dtype=bool)
        out[f"within_{b}_l2"] = float(d[m].mean())
        out[f"norm_{b}"] = float(np.linalg.norm(E, axis=1).mean())
    if len(labels) == 2:
        A, B = emb[batch == labels[0]], emb[batch == labels[1]]
        out["cross_l2"] = float(_pairwise_l2(A, B).mean())
        within = [out.get(f"within_{b}_l2", np.nan) for b in labels]
        out["cross_over_within"] = float(out["cross_l2"] / np.nanmean(within))
        # centroid gap relative to the average cloud radius: >1 means the clouds
        # sit side by side, <<1 means they are concentric.
        gap = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        out["centroid_gap"] = gap
        out["centroid_gap_over_within"] = float(gap / np.nanmean(within))
        # Ratio of the tighter cloud's spread to the wider one's. <<1 means one
        # modality is a dense clump inside the other, which reads as "overlapping"
        # in a UMAP while every k-NN of a clump member is another clump member.
        w = [v for v in within if np.isfinite(v) and v > 0]
        out["spread_ratio"] = float(min(w) / max(w)) if len(w) == 2 else float("nan")
    return out


def knn_mixing(
    emb: np.ndarray,
    batch: np.ndarray,
    ks=(1, 15, 50, 90),
    metric: str = "euclidean",
) -> dict:
    """Same-batch neighbour fraction and scaled inverse-Simpson (iLISI-like) per k.

    ``metric`` matters: scIB's iLISI and a scanpy UMAP both use Euclidean
    neighbours, but BRAS uses cosine. Running both is how a
    "UMAP says mixed / BRAS says separated" disagreement gets attributed.
    """
    from sklearn.neighbors import NearestNeighbors

    codes, uniq = pd.factorize(pd.Series(batch))
    n_batches = len(uniq)
    shares = np.array([(codes == i).mean() for i in range(n_batches)])
    k_max = min(max(ks), len(emb) - 1)
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric=metric).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]  # drop self

    out: dict = {}
    for k in ks:
        if k > k_max:
            continue
        nb = codes[idx[:, :k]]
        same = (nb == codes[:, None]).mean()
        # expected same-batch fraction under perfect mixing
        exp_same = float((shares ** 2).sum() / shares.sum())
        # inverse Simpson per cell, scaled to [0, 1] as scib_metrics does
        props = np.stack([(nb == i).mean(axis=1) for i in range(n_batches)], axis=1)
        inv_simpson = 1.0 / np.clip((props ** 2).sum(axis=1), 1e-12, None)
        out[f"same_batch_frac_k{k}"] = float(same)
        out[f"expected_same_batch_k{k}"] = exp_same
        out[f"ilisi_like_k{k}"] = float(
            ((inv_simpson - 1.0) / (n_batches - 1)).mean()
        )
        # Fraction of cells whose whole neighbourhood is a single batch. This is
        # what drives iLISI, and it is the statistic the *mean* same-batch fraction
        # hides: two pure-but-opposite neighbourhoods average out to "looks mixed".
        out[f"pure_nbhd_frac_k{k}"] = float((props.max(axis=1) == 1.0).mean())
    return out


def batch_silhouette(
    emb: np.ndarray,
    batch: np.ndarray,
    label: np.ndarray,
    metric: str = "cosine",
) -> dict:
    """Overall and per-label silhouette w.r.t. batch (the family BRAS belongs to).

    1 - |ASW_batch| is the classic scIB ``silhouette_batch``: 1.0 = batches
    indistinguishable inside that label.  Labels holding a single batch are
    reported as NaN, because no batch-mixing statement can be made there — that is
    also exactly what BRAS and kBET do with them, which is why the count of usable
    labels is reported alongside.

    ``metric`` defaults to cosine to match BRAS; pass "euclidean" for the classic
    silhouette_batch.
    """
    from sklearn.metrics import silhouette_samples

    out: dict = {}
    codes, _ = pd.factorize(pd.Series(batch))
    if len(set(codes.tolist())) > 1:
        s = silhouette_samples(emb, codes, metric=metric)
        out["asw_batch_global"] = float(np.abs(s).mean())
        out["silhouette_batch_global"] = float(1.0 - np.abs(s).mean())

    per_label: dict[str, float] = {}
    for lab in sorted(set(label.tolist())):
        m = label == lab
        if m.sum() < 10:
            continue
        lb = codes[m]
        if len(set(lb.tolist())) < 2:
            per_label[str(lab)] = float("nan")  # single-batch label
            continue
        s = silhouette_samples(emb[m], lb, metric=metric)
        per_label[str(lab)] = float(1.0 - np.abs(s).mean())
    out["silhouette_batch_per_label"] = per_label
    vals = [v for v in per_label.values() if np.isfinite(v)]
    out["silhouette_batch_label_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["n_labels_total"] = len(per_label)
    out["n_labels_with_both_batches"] = len(vals)
    return out


def principal_component_regression(
    X: np.ndarray, covariate: np.ndarray, n_components: int = 50
) -> float:
    """Variance-weighted R^2 of ``covariate`` on the PCs of ``X`` (scIB's PCR).

    Returns a value in [0, 1]; 1.0 means the covariate explains all the variance
    captured by the PCs.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    n_components = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(np.nan_to_num(X.astype(np.float64)))
    var = pca.explained_variance_

    onehot = pd.get_dummies(pd.Series(covariate)).to_numpy(dtype=np.float64)
    r2 = np.empty(n_components)
    for i in range(n_components):
        y = pcs[:, i]
        pred = LinearRegression().fit(onehot, y).predict(onehot)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2[i] = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return float((r2 * var).sum() / var.sum())


# ---------------------------------------------------------------------------
# Subset construction — mirrors run_scib_benchmark's "bulk_vs_pb" benchmark
# ---------------------------------------------------------------------------

def build_benchmark_subset(
    adata: ad.AnnData, group_column: str, n_max: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_indices, batch_labels, group_labels) for benchmark 1."""
    rng = np.random.default_rng(seed)
    mod = adata.obs.get(
        _MODALITY_COL, pd.Series("", index=adata.obs_names, dtype=str)
    ).astype(str)

    all_bulk = mod.isin(BULK_MODALITIES).values
    all_pb = mod.isin(PB_MODALITIES).values

    def _sample(mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask)[0]
        if len(idx) > n_max:
            idx = rng.choice(idx, size=n_max, replace=False)
        return np.sort(idx)

    bulk_idx, pb_idx = _sample(all_bulk), _sample(all_pb)
    idx = np.concatenate([bulk_idx, pb_idx])
    batch = np.array(["bulk"] * len(bulk_idx) + ["pseudobulk"] * len(pb_idx))
    groups = (
        adata.obs[group_column].astype(str).to_numpy()[idx]
        if group_column in adata.obs.columns
        else np.full(len(idx), "unknown")
    )
    return idx, batch, groups


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def modality_census(adata: ad.AnnData) -> None:
    print("\n=== Modality census ===")
    if _MODALITY_COL not in adata.obs.columns:
        print(f"  '{_MODALITY_COL}' not in obs — nothing to report.")
        return
    print(adata.obs[_MODALITY_COL].astype(str).value_counts().to_string())

    mod = adata.obs[_MODALITY_COL].astype(str)
    pb, synth = (mod == MOD_PB).values, (mod == MOD_SYNTH_PB).values
    if pb.any() and synth.any():
        # build_eval_adata --precomputed-pb now relabels rows in place, so these two
        # groups should be disjoint. They used to be a relabelled *copy*, meaning the
        # same cells were embedded twice under different panels and the UMAP and scIB
        # each read a different copy. Kept as a regression check on stale eval files.
        base = adata.obs_names[pb].str.replace(r"-\d+$", "", regex=True)
        dup = adata.obs_names[synth].str.replace(r"-\d+$", "", regex=True)
        overlap = len(set(base) & set(dup))
        print(
            f"\n  '{MOD_PB}' rows: {int(pb.sum())}, '{MOD_SYNTH_PB}' rows: "
            f"{int(synth.sum())}, obs_name overlap after de-suffixing: {overlap}"
        )
        if overlap:
            print(
                "  [warn] the same cells appear under both labels, so they were "
                "embedded twice under independently fitted panels: the UMAP reads the "
                f"'{MOD_PB}' copy and scIB the '{MOD_SYNTH_PB}' one. This eval.h5ad "
                "predates the in-place relabelling fix -- rebuild it."
            )


def report(
    adata: ad.AnnData,
    group_column: str,
    n_max: int,
    seed: int,
    ks: tuple[int, ...],
) -> None:
    modality_census(adata)

    idx, batch, groups = build_benchmark_subset(adata, group_column, n_max, seed)
    if len(idx) == 0 or len(set(batch.tolist())) < 2:
        print("\nCould not build a two-batch bulk-vs-pseudobulk subset — stopping.")
        return

    print("\n=== Benchmarked subset (mirrors scib_bulk_vs_pb) ===")
    print(f"  {len(idx)} cells: " + ", ".join(
        f"{b}={int((batch == b).sum())}" for b in sorted(set(batch.tolist()))
    ))

    print(f"\n=== label x batch contingency ('{group_column}') ===")
    tab = pd.crosstab(pd.Series(groups, name=group_column), pd.Series(batch, name="batch"))
    print(tab.to_string())
    both = int(((tab > 0).sum(axis=1) == tab.shape[1]).sum())
    print(
        f"\n  labels total: {tab.shape[0]} | labels containing both batches: {both}"
    )
    if both < tab.shape[0]:
        print(
            "  -> label-conditioned metrics (BRAS, kBET) can only see batch mixing "
            f"inside those {both} label(s); the rest contribute no mixing signal."
        )

    emb_keys = sorted(k for k in adata.obsm if k.startswith("X_cf_"))
    if not emb_keys:
        print("\nNo 'X_cf_*' keys in obsm — stopping.")
        return

    pre = None
    if "X_pca" in adata.obsm:
        pre = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[idx]
        pcr_pre = principal_component_regression(pre, batch)
        print(f"\n=== PCR baseline ===\n  pcr(X_pca) = {pcr_pre:.4f}")
    else:
        pcr_pre = None
        print("\n=== PCR baseline ===\n  X_pca absent — PCR comparison was skipped.")

    rows: list[dict] = []
    for key in emb_keys:
        emb = np.asarray(adata.obsm[key], dtype=np.float32)[idx]
        print(f"\n=== {key} ===")

        geo = geometry_summary(emb, batch)
        for k, v in geo.items():
            print(f"  {k:34s} {v:.4f}")

        # Euclidean = what iLISI and a scanpy UMAP see; cosine = what BRAS sees.
        mix = knn_mixing(emb, batch, ks=ks, metric="euclidean")
        mix_cos = knn_mixing(emb, batch, ks=ks, metric="cosine")
        for k in ks:
            if f"same_batch_frac_k{k}" not in mix:
                continue
            print(
                f"  k={k:<3d} same-batch nbrs  euclid {mix[f'same_batch_frac_k{k}']:.3f} "
                f"| cosine {mix_cos[f'same_batch_frac_k{k}']:.3f}   "
                f"(perfect mixing -> {mix[f'expected_same_batch_k{k}']:.3f})   "
                f"iLISI-like euclid {mix[f'ilisi_like_k{k}']:.4f} "
                f"| cosine {mix_cos[f'ilisi_like_k{k}']:.4f}   "
                f"single-batch nbhds {mix[f'pure_nbhd_frac_k{k}']:.3f}"
            )

        sil_cos = batch_silhouette(emb, batch, groups, metric="cosine")
        sil_euc = batch_silhouette(emb, batch, groups, metric="euclidean")
        print(
            f"  silhouette_batch (cosine, ~BRAS)  global "
            f"{sil_cos.get('silhouette_batch_global', float('nan')):.4f} | "
            f"label-mean {sil_cos['silhouette_batch_label_mean']:.4f} "
            f"({sil_cos['n_labels_with_both_batches']}/{sil_cos['n_labels_total']} labels usable)"
        )
        print(
            f"  silhouette_batch (euclidean)      global "
            f"{sil_euc.get('silhouette_batch_global', float('nan')):.4f} | "
            f"label-mean {sil_euc['silhouette_batch_label_mean']:.4f}"
        )
        if sil_cos["n_labels_with_both_batches"] < 2:
            print(
                "  [warn] fewer than 2 labels hold both batches -> BRAS and kBET are "
                "near-meaningless for this comparison; rank on iLISI instead."
            )

        row = {"embedding": key, **geo,
               **{f"{k}_euclid": v for k, v in mix.items()},
               **{f"{k}_cosine": v for k, v in mix_cos.items()},
               "silhouette_batch_cosine_global": sil_cos.get("silhouette_batch_global"),
               "silhouette_batch_cosine_label_mean": sil_cos["silhouette_batch_label_mean"],
               "silhouette_batch_euclid_global": sil_euc.get("silhouette_batch_global"),
               "silhouette_batch_euclid_label_mean": sil_euc["silhouette_batch_label_mean"],
               "n_labels_with_both_batches": sil_cos["n_labels_with_both_batches"]}

        if pcr_pre is not None:
            pcr_post = principal_component_regression(emb, batch)
            scaled = max(0.0, (pcr_pre - pcr_post) / pcr_pre) if pcr_pre > 0 else 0.0
            print(
                f"  pcr(embedding) {pcr_post:.4f} vs pcr(X_pca) {pcr_pre:.4f} "
                f"-> pcr_comparison {scaled:.4f}"
                + ("   [clipped at 0: embedding is more batch-driven than the baseline]"
                   if pcr_post >= pcr_pre else "")
            )
            row.update(pcr_pre=pcr_pre, pcr_post=pcr_post, pcr_comparison=scaled)

        rows.append(row)

    df = pd.DataFrame(rows).set_index("embedding")
    print("\n=== Summary (higher iLISI-like / silhouette_batch = better mixing) ===")
    k = ks[-1]
    cols = [c for c in (
        f"ilisi_like_k{k}_euclid", f"pure_nbhd_frac_k{k}_euclid",
        f"same_batch_frac_k{k}_euclid", f"ilisi_like_k{k}_cosine",
        "silhouette_batch_cosine_label_mean", "silhouette_batch_euclid_global",
        "centroid_gap_over_within", "spread_ratio",
        "pcr_post", "pcr_comparison",
    ) if c in df.columns]
    print(df[cols].to_string())
    print(
        f"""
How to read this (k={k}):
  * iLISI-like is the mixing measure: 1.0 = every neighbourhood is balanced,
    0.0 = no neighbourhood contains both batches.
  * A LOW iLISI with pure_nbhd_frac near 1 but same_batch_frac near the
    perfect-mixing value means neighbourhoods are internally pure yet split
    between the batches -- the *mean* same-batch fraction averages the two out and
    looks mixed. Do not read same_batch_frac on its own.
  * That pattern together with centroid_gap_over_within << 1 and spread_ratio << 1
    is the concentric case: one modality is a dense clump sitting inside the
    other. A UMAP shows the clouds on top of each other while every k-NN of a
    clump member is another clump member, so the picture and iLISI are BOTH right
    -- they answer different questions. Fix it by comparing spreads, not positions.
  * centroid_gap_over_within >~ 1 instead means the clouds are genuinely displaced.
  * euclid vs cosine columns diverging explains a UMAP-vs-BRAS disagreement, since
    BRAS scores on cosine while the UMAP graph and iLISI use Euclidean distance.
  * n_labels_with_both_batches < 2 invalidates BRAS and kBET (they skip
    single-batch labels), leaving iLISI as the only usable batch metric."""
    )
    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-adata", type=Path, required=True,
                   help="eval.h5ad produced by build_eval_adata.py.")
    p.add_argument("--group-column", type=str, default="tissue_general",
                   help="label_key used by the scIB run (default: tissue_general).")
    p.add_argument("--n-max", type=int, default=500,
                   help="Cells per batch, matching run_scib_benchmark (default: 500).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", type=int, nargs="*", default=[1, 15, 50, 90],
                   help="Neighbourhood sizes to probe (scIB builds a 90-NN graph).")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to write the summary table.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    path = args.eval_adata.expanduser().resolve()
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    print(f"Loading {path} ...")
    adata = ad.read_h5ad(path)
    print(f"  {adata.n_obs} cells, obsm: {list(adata.obsm.keys())}")
    canonicalize_modality_column(adata)

    df = report(adata, args.group_column, args.n_max, args.seed, tuple(args.ks))
    if args.out_csv is not None and df is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv)
        print(f"\nSummary -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
