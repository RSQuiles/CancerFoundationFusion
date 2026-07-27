"""UMAP of the PCA space used to infer tissue labels in ``data_preprocess/add_fields.py``.

``add_fields.py`` infers ``tissue_general`` in two stages: keyword matching over the
string obs columns, then a nearest-centroid fill of the remaining ``"unknown"`` cells
computed in ``obsm["X_pca"]``. That ``X_pca`` is persisted into the output h5ad, so this
script can render exactly the space the fill operated in, coloured by the final label —
without ever touching the expression matrix.

Usage:
    python plot_tissue_umap.py \
        --adata /path/to/pretraining_bulk_RAW.h5ad \
        --color tissue_general \
        --sample-size 75_000 \
        --save-h5ad

If the file has no ``X_pca`` (the fill was skipped, or nothing was unknown), pass
``--recompute-pca`` to rebuild it here with the same recipe as ``add_fields.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / SLURM

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from umap_render import compute_umap, save_umap_plot

from utils import subsample_adata


def _compute_pca(
    adata: AnnData,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    normalize: bool = False,
) -> None:
    """Rebuild ``X_pca`` in place with the recipe from ``add_fields.compute_embedding``.

    Kept as a local copy rather than an import: ``add_fields.py`` does a flat
    ``from utils import walk_tissue_names``, which from this directory would resolve to
    ``evaluate/plot/utils.py``. Keep this in sync with ``data_preprocess/add_fields.py``.
    """
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.X.indices = adata.X.indices.astype(np.int64, copy=False)
        adata.X.indptr = adata.X.indptr.astype(np.int64, copy=False)

    work = adata.copy()
    n_top = min(n_top_genes, work.n_vars)
    if normalize:
        print("  [pca] log1p, then seurat HVGs", flush=True)
        sc.pp.log1p(work)
        sc.pp.highly_variable_genes(
            work, n_top_genes=n_top, subset=True, flavor="seurat"
        )
    else:
        print("  [pca] seurat_v3 HVGs on raw counts, then log1p", flush=True)
        sc.pp.highly_variable_genes(
            work, n_top_genes=n_top, flavor="seurat_v3", subset=True
        )
        sc.pp.log1p(work)

    n_comps = min(n_pcs, work.n_vars - 1, work.n_obs - 1)
    print(
        f"  [pca] {work.n_vars} HVGs -> PCA ({n_comps} comps, sparse/truncated-SVD)",
        flush=True,
    )
    sc.tl.pca(work, n_comps=n_comps, zero_center=False)
    adata.obsm["X_pca"] = work.obsm["X_pca"]


def load_embedding_adata(args) -> AnnData:
    """Return a lightweight AnnData carrying only obs + X_pca (+ X_umap when present).

    The expression matrix is left on disk unless ``--recompute-pca`` forces a rebuild.
    """
    print(f"Reading {args.adata} (backed)", flush=True)
    adata = sc.read_h5ad(args.adata, backed="r")

    if "X_pca" not in adata.obsm:
        if not args.recompute_pca:
            raise SystemExit(
                f"{args.adata} has no obsm['X_pca'] (available: {list(adata.obsm)}).\n"
                "It is written by data_preprocess/add_fields.py only when the centroid "
                "fill runs. Re-run add_fields.py, or pass --recompute-pca to rebuild it here."
            )
        print(
            "No X_pca found; recomputing in memory (this loads the matrix)", flush=True
        )
        del adata
        adata = sc.read_h5ad(args.adata)
        _compute_pca(adata, normalize=args.pca_lognorm)

    obsm = {"X_pca": np.asarray(adata.obsm["X_pca"], dtype=np.float32)}
    if "X_umap" in adata.obsm:
        obsm["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
        print("Reusing the X_umap already stored in the file", flush=True)

    # X is dropped on purpose: neighbors/UMAP/plotting only read obs + obsm.
    return AnnData(X=None, obs=adata.obs.copy(), obsm=obsm, shape=(adata.n_obs, 0))


def main(args) -> None:
    adata = load_embedding_adata(args)
    print(
        f"Loaded {adata.n_obs} observations, X_pca {adata.obsm['X_pca'].shape}",
        flush=True,
    )

    for key in args.color:
        if key in adata.obs.columns:
            counts = pd.Series(adata.obs[key].astype(str)).value_counts()
            print(
                f"\n{key} ({len(counts)} labels):\n{counts.to_string()}\n", flush=True
            )
        else:
            print(f"WARNING: '{key}' not in obs; panel will be uncoloured", flush=True)

    if args.sample_size and adata.n_obs > args.sample_size:
        print(f"Subsampling {args.sample_size} of {adata.n_obs}", flush=True)
        adata = subsample_adata(adata, args.sample_size, seed=args.seed)

    if "X_umap" not in adata.obsm:
        print(
            f"Computing UMAP on X_pca (n_neighbors={args.neighbors}, "
            f"min_dist={args.min_dist})",
            flush=True,
        )
        compute_umap(
            adata,
            use_rep="X_pca",
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
        )

    out_dir = Path(args.out_dir)
    prefix = args.out_prefix or Path(args.adata).stem
    out_png = save_umap_plot(
        adata,
        out_dir / f"{prefix}.tissue_umap.png",
        color=args.color,
        title=f"{prefix} — tissue inference space (X_pca)",
        dpi=args.dpi,
        skip_unknown=args.skip_unknown,
    )
    print(f"Wrote {out_png}", flush=True)

    if args.save_h5ad:
        out_h5ad = out_dir / f"{prefix}.tissue_umap.h5ad"
        adata.write_h5ad(out_h5ad)
        print(
            f"Wrote {out_h5ad} (re-run with --adata {out_h5ad} to skip the UMAP)",
            flush=True,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="h5ad written by add_fields.py (carries obs['tissue_general'] and obsm['X_pca'])",
    )
    parser.add_argument(
        "--color",
        type=str,
        nargs="*",
        default=["tissue_general"],
        help="obs column(s) to colour by; one panel each (default: tissue_general)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./umap", help="Output directory"
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: the input file stem)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=75_000,
        help="Subsample to this many cells before the UMAP (0 disables)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for subsampling and UMAP"
    )
    parser.add_argument(
        "--neighbors", type=int, default=15, help="sc.pp.neighbors n_neighbors"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.5, help="sc.tl.umap min_dist"
    )
    parser.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Grey out 'unknown'/'nan' categories instead of giving them a colour",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--save-h5ad",
        action="store_true",
        help="Also write obs + X_pca + X_umap, so re-colouring skips the UMAP",
    )
    parser.add_argument(
        "--recompute-pca",
        action="store_true",
        help="Rebuild X_pca if the file has none (loads the expression matrix)",
    )
    parser.add_argument(
        "--pca-lognorm",
        action="store_true",
        help="With --recompute-pca, log1p before HVG selection (mirrors --tissue-fill-lognorm)",
    )
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
