"""Add one or more obs columns to h5ad file(s), writing to a separate location.

Two modes, selected automatically from ``--input``:
  * directory  -> every ``{prefix}*.h5ad`` in it is processed and written under
    ``--output`` (a directory) with the same file names.
  * single file -> that file is processed and written to ``--output`` (a file, or a
    directory in which case the original name is kept).

Each field is given as ``NAME=VALUE`` and repeated with ``--field``. A constant
value is broadcast to every row. Two special cases infer the value instead of
broadcasting it:

  * ``--field tissue_general`` / ``--field tissue`` (value omitted or empty), or
    any field name in ``TISSUE_FIELDS`` -> the value is inferred per row with
    ``walk_tissue_names`` over the string obs columns (see ``--tissue-meta-fields``).
    Rows that stay "unknown" after keyword matching are then filled from a
    UMAP/PCA embedding: each unknown cell is assigned the nearest per-tissue
    centroid, provided it is within a threshold of the typical intra-tissue
    spread (``--tissue-fill-threshold``). Disable with ``--no-tissue-fill``.

Usage:
    # Broadcast a constant to every file in a directory
    python add_fields.py --input /in/h5ads --output /out/h5ads --field modality=bulk

    # Infer tissue_general (CellxGene labels only) + centroid fill on one file
    python add_fields.py --input bulk.h5ad --output out/bulk.h5ad \
        --field tissue_general --cellxgene-only
"""

import re
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
import scipy.sparse as sp

from utils import walk_tissue_names

# Field names that trigger tissue inference instead of constant broadcast.
TISSUE_FIELDS = {"tissue", "tissue_general"}


def read_anndata(file_path: Path) -> sc.AnnData:
    # Strip a stray log1p base entry that can make some CellxGene h5ads unreadable.
    try:
        with h5py.File(file_path, "r+") as f:
            if "uns" in f and "log1p" in f["uns"] and "base" in f["uns"]["log1p"]:
                del f["uns"]["log1p"]["base"]
    except (OSError, KeyError):
        pass
    return sc.read_h5ad(file_path)


def _normalize_text(s) -> str:
    s = str(s).lower()
    s = re.sub(r"[_\-/]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tissue_of_row(values, cellxgene_only: bool, precise: bool) -> str:
    """Infer a tissue label from a row's metadata field values."""
    norm_items = [_normalize_text(v) for v in values]

    def check(name: str) -> bool:
        words = _normalize_text(name).split()
        if not words:
            return False
        for item in norm_items:
            if len(words) == 1:
                if words[0] in item:
                    return True
            elif all(w in item for w in words):
                return True
        return False

    return walk_tissue_names(
        lambda name: check(name), cellxgene_only=cellxgene_only, precise=precise
    )


def infer_tissue_column(
    adata: sc.AnnData,
    meta_fields: list[str] | None,
    cellxgene_only: bool,
    precise: bool,
) -> pd.Series:
    """Keyword-match a tissue label for every cell from its string obs columns."""
    obs = adata.obs
    if not meta_fields:
        meta_fields = [
            c
            for c in obs.columns
            if obs[c].dtype == object or str(obs[c].dtype) == "category"
        ]
    meta_fields = [c for c in meta_fields if c in obs.columns]

    if not meta_fields:
        print("  No usable metadata columns for tissue inference; all 'unknown'")
        return pd.Series(["unknown"] * adata.n_obs, index=obs.index)

    print(f"  Inferring tissue from columns: {meta_fields}")
    sub = obs[meta_fields].astype(str)
    # Many cells share identical metadata; infer once per unique combination.
    cache: dict[tuple, str] = {}
    labels = []
    for row in map(tuple, sub.values):
        label = cache.get(row)
        if label is None:
            label = _tissue_of_row(row, cellxgene_only, precise)
            cache[row] = label
        labels.append(label)
    return pd.Series(labels, index=obs.index)


def compute_embedding(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    normalize: bool = True,
    do_umap: bool = False,
) -> bool:
    """Compute a PCA embedding and stash ``X_pca`` on ``adata.obsm``.

    Kept memory-light for large datasets: highly variable genes are selected and
    the matrix is subset to them *before* any dense operation, PCA runs directly
    on the sparse HVG matrix (no ``scale`` densification), and the UMAP/Leiden
    graph is only built when ``do_umap`` is set. Returns True on success.
    """
    if adata.n_obs < 3 or adata.n_vars < 2:
        print("    [embed] too few cells/genes for an embedding; skipping")
        return False

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.X.indices = adata.X.indices.astype(np.int64, copy=False)
        adata.X.indptr = adata.X.indptr.astype(np.int64, copy=False)
    try:
        print(f"    [embed] copying matrix ({adata.n_obs} x {adata.n_vars})...", flush=True)
        work = adata.copy()
        # work = adata
        # Select HVGs and subset in the same pass, so nothing downstream ever
        # operates at full gene width.
        n_top = min(n_top_genes, work.n_vars)
        if normalize:
            print("    [embed] Normalizing before embedding with flavor=seurat!", flush=True)
            # sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
            sc.pp.highly_variable_genes(work, n_top_genes=n_top, subset=True, flavor="seurat")
        else:
            print("    [embed] seurat_v3 HVGs on raw counts, then log1p", flush=True)
            sc.pp.highly_variable_genes(work, n_top_genes=n_top, flavor="seurat_v3", subset=True)
            sc.pp.log1p(work)
        print(f"    [embed] subset to {work.n_vars} HVGs", flush=True)
        # PCA directly on the sparse HVG matrix. zero_center=False uses truncated
        # SVD, which keeps the input sparse instead of allocating a dense copy.
        n_comps = min(n_pcs, work.n_vars - 1, work.n_obs - 1)
        print(f"    [embed] PCA ({n_comps} comps, sparse/truncated-SVD)...", flush=True)
        sc.tl.pca(work, n_comps=n_comps, zero_center=False)
        adata.obsm["X_pca"] = work.obsm["X_pca"]
        print("    [embed] X_pca ready", flush=True)
        # UMAP + clustering: opt-in (heavy on large data, unused by the centroid
        # fill), and non-fatal if the optional deps are missing.
        if do_umap:
            try:
                print("    [embed] neighbors + UMAP + Leiden...", flush=True)
                sc.pp.neighbors(work)
                sc.tl.umap(work)
                adata.obsm["X_umap"] = work.obsm["X_umap"]
                sc.tl.leiden(work)
                adata.obs["leiden"] = work.obs["leiden"].values
                print("    [embed] X_umap + leiden ready", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [embed] UMAP/Leiden step skipped ({e})", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [embed] embedding failed, skipping centroid fill ({e})", flush=True)
        return False


def fill_unknown_by_centroid(
    adata: sc.AnnData,
    col: str,
    rep: str = "X_pca",
    threshold: float = 2.0,
) -> int:
    """Assign each "unknown" cell to the nearest per-tissue centroid.

    A cell is only relabelled if its distance to the nearest centroid is within
    ``threshold`` times the median distance of labelled cells to their own
    tissue centroid. Returns the number of cells relabelled.
    """
    labels = adata.obs[col].astype(str).to_numpy().copy()
    unknown = labels == "unknown"
    known = ~unknown
    if not unknown.any() or known.sum() < 2:
        return 0
    if rep not in adata.obsm:
        return 0

    X = np.asarray(adata.obsm[rep])
    tissues = list(pd.unique(labels[known]))
    if not tissues:
        return 0
    centroids = {t: X[known & (labels == t)].mean(axis=0) for t in tissues}

    # Typical spread: median distance of labelled cells to their own centroid.
    known_idx = np.where(known)[0]
    own_dist = np.array(
        [np.linalg.norm(X[i] - centroids[labels[i]]) for i in known_idx]
    )
    spread = float(np.median(own_dist)) if own_dist.size else 0.0
    cutoff = threshold * spread if spread > 0 else np.inf

    C = np.stack([centroids[t] for t in tissues])
    unknown_idx = np.where(unknown)[0]
    dists = cdist(X[unknown_idx], C)
    nearest = dists.argmin(axis=1)
    nearest_dist = dists.min(axis=1)

    n_filled = 0
    for j, i in enumerate(unknown_idx):
        if nearest_dist[j] <= cutoff:
            labels[i] = tissues[nearest[j]]
            n_filled += 1

    adata.obs[col] = labels
    return n_filled


def add_fields_to_adata(
    adata: sc.AnnData,
    fields: list[tuple[str, str | None]],
    *,
    meta_fields: list[str] | None,
    cellxgene_only: bool,
    precise: bool,
    tissue_fill: bool,
    fill_threshold: float,
    fill_rep: str,
    fill_normalize: bool,
) -> None:
    for name, value in fields:
        is_tissue = name in TISSUE_FIELDS or (value is None or value == "")
        if is_tissue and (value is None or value == ""):
            tissue = infer_tissue_column(adata, meta_fields, cellxgene_only, precise)
            adata.obs[name] = tissue.to_numpy()
            counts = pd.Series(adata.obs[name]).value_counts()
            n_unknown = int(counts.get("unknown", 0))
            print(
                f"  {name}: {len(counts)} labels, {n_unknown} unknown after keyword match"
            )

            if tissue_fill and n_unknown > 0:
                if "X_pca" not in adata.obsm or fill_rep not in adata.obsm:
                    compute_embedding(adata, normalize=fill_normalize)
                n_filled = fill_unknown_by_centroid(
                    adata, name, rep=fill_rep, threshold=fill_threshold
                )
                remaining = int(pd.Series(adata.obs[name]).value_counts().get("unknown", 0))
                print(f"  {name}: filled {n_filled} by centroid, {remaining} still unknown")
        else:
            adata.obs[name] = value
            print(f"  {name} = '{value}' ({adata.n_obs} rows)")


def _resolve_targets(input_path: Path, output_path: Path, prefix: str):
    """Yield (in_file, out_file) pairs for the requested input/output."""
    if input_path.is_dir():
        files = sorted(input_path.glob(f"{prefix}*.h5ad"))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{prefix}*.h5ad' in {input_path}"
            )
        output_path.mkdir(parents=True, exist_ok=True)
        for f in files:
            yield f, output_path / f.name
    elif input_path.is_file():
        if output_path.suffix == ".h5ad":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            yield input_path, output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            yield input_path, output_path / input_path.name
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_fields(raw_fields: list[str]) -> list[tuple[str, str | None]]:
    parsed = []
    for item in raw_fields:
        if "=" in item:
            name, value = item.split("=", 1)
            parsed.append((name.strip(), value))
        else:
            parsed.append((item.strip(), None))
    return parsed


def main(args):
    fields = parse_fields(args.field)
    targets = list(_resolve_targets(args.input, args.output, args.prefix))
    print(f"Processing {len(targets)} file(s); fields: {fields}")

    for i, (in_file, out_file) in enumerate(targets):
        adata = read_anndata(in_file)
        if args.skip_existing and all(
            name in adata.obs.columns for name, _ in fields
        ):
            print(f"  [{i + 1}/{len(targets)}] {in_file.name}: all fields present, skipping")
            continue
        print(f"  [{i + 1}/{len(targets)}] {in_file.name} ({adata.n_obs} rows)")
        add_fields_to_adata(
            adata,
            fields,
            meta_fields=args.tissue_meta_fields,
            cellxgene_only=args.cellxgene_only,
            precise=args.tissue_precise,
            tissue_fill=args.tissue_fill,
            fill_threshold=args.tissue_fill_threshold,
            fill_rep=args.tissue_fill_rep,
            fill_normalize=args.tissue_fill_lognorm,
        )
        adata.write_h5ad(out_file)
        print(f"      -> wrote {out_file}")

    print("Done.")


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input h5ad file or a directory of h5ad files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (single-file input) or directory (directory input)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="When input is a directory, only process files matching '{prefix}*.h5ad'",
    )
    parser.add_argument(
        "--field",
        action="append",
        required=True,
        metavar="NAME=VALUE",
        help=(
            "Column to add as NAME=VALUE (repeatable). Omit '=VALUE' (or leave it "
            "empty) for a tissue field to infer the value per row instead."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a file when all requested field columns already exist in obs",
    )

    # --- Tissue inference ---
    parser.add_argument(
        "--tissue-meta-fields",
        type=str,
        nargs="+",
        default=None,
        help="obs columns to search when inferring tissue (default: all string columns)",
    )
    parser.add_argument(
        "--cellxgene-only",
        action="store_true",
        help="Only assign tissue labels that exist as CellxGene tissue_general values",
    )
    parser.add_argument(
        "--tissue-precise",
        action="store_true",
        help="Return the matching subtissue instead of its parent category",
    )
    parser.add_argument(
        "--no-tissue-fill",
        dest="tissue_fill",
        action="store_false",
        help="Skip the UMAP/centroid fill of unknown tissues",
    )
    parser.add_argument(
        "--tissue-fill-threshold",
        type=float,
        default=2.0,
        help=(
            "Max distance to nearest tissue centroid, as a multiple of the median "
            "intra-tissue spread, for an unknown cell to be relabelled (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--tissue-fill-rep",
        type=str,
        default="X_pca",
        help="obsm key used for centroid distances (default: X_pca)",
    )
    parser.add_argument(
        "--tissue-fill-lognorm",
        action="store_true",
        help="Whether to apply normalize_total+log1p before embedding",
    )
    parser.set_defaults(tissue_fill=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
