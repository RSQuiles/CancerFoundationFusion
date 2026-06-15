"""
UMAP of paired SC / pseudobulk / bulk profiles, coloured by cell line.

For every cell line that has both SC and bulk data the script:
  1. Embeds all individual SC cells.
  2. Computes one pseudobulk (mean of all SC cells) and embeds it.
  3. Embeds the paired bulk profile(s).

The joint embedding is projected to UMAP and plotted with three marker styles:
  SC cells      →  x  (small, semi-transparent)
  Bulk          →  o  (circle, filled, per cell line)
  SC mean / PB  →  ▲  (triangle, filled, per cell line)
All markers share the same per-cell-line colour.

Usage
-----
python evaluate/plot/plot_paired_umap.py \\
    --ckpt /path/to/model.ckpt \\
    --input-h5ad /path/to/paired.h5ad \\
    --out-dir ./umap_outputs \\
    --cell-line-col cell_line \\
    --domain-col domain \\
    --sc-label SC \\
    --bulk-label bulk \\
    --normalized          # pass if input is already CP10K+log1p
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.model.model import CancerFoundation
from data_preprocess.pseudobulk_paired_generation import generate_pseudobulk_chunks


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_dense(X) -> np.ndarray:
    return np.asarray(X.todense(), dtype=np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)


def _load_model(ckpt_path: str | Path, device: str | None = None) -> CancerFoundation:
    ckpt_path = Path(ckpt_path)
    model = CancerFoundation.load_from_checkpoint(str(ckpt_path), strict=False)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def _embed(model: CancerFoundation, adata: ad.AnnData, batch_size: int, normalized: bool) -> np.ndarray:
    """Return (N, D) embedding array for all cells in adata."""
    result = model.embed(adata, batch_size=batch_size, normalized=normalized)
    emb_df = result[0] if isinstance(result, tuple) else result
    return emb_df.to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _cell_line_palette(cell_lines: list[str]) -> dict[str, tuple]:
    """Return a colour mapping for cell lines, cycling through tab20 then tab20b."""
    n = len(cell_lines)
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors: dict[str, tuple] = {}
    for i, cl in enumerate(cell_lines):
        cmap = cmaps[(i // 20) % len(cmaps)]
        colors[cl] = cmap(i % 20)
    return colors


def plot_paired_umap(
    combined: ad.AnnData,
    paired_cls: list[str],
    cell_line_col: str,
    title: str | None = None,
    max_legend_cols: int = 4,
    show_sc: bool = True,
) -> plt.Figure:
    """Render the three-modality UMAP coloured by cell line."""
    umap_coords = combined.obsm["X_umap"]
    modality = combined.obs["modality"].astype(str).values
    cell_line = combined.obs[cell_line_col].astype(str).values

    sc_mask = modality == "sc"
    pb_mask = modality == "pseudobulk"
    bulk_mask = modality == "bulk"

    palette = _cell_line_palette(paired_cls)
    colors = np.array([palette[cl] for cl in cell_line], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    # SC cells — small x (only when show_sc is True)
    if show_sc and sc_mask.any():
        sc_colors = colors[sc_mask].copy()
        sc_colors[:, 3] = 0.35
        ax.scatter(
            umap_coords[sc_mask, 0], umap_coords[sc_mask, 1],
            c=sc_colors,
            s=8, marker="x", linewidths=0.6,
            rasterized=True, zorder=1, label="_nolegend_",
        )

    # Pseudobulk — upward triangle
    if pb_mask.any():
        ax.scatter(
            umap_coords[pb_mask, 0], umap_coords[pb_mask, 1],
            c=colors[pb_mask],
            s=80, marker="^", linewidths=0.5,
            edgecolors="black", alpha=0.9,
            rasterized=True, zorder=3, label="_nolegend_",
        )

    # Bulk — circle
    if bulk_mask.any():
        ax.scatter(
            umap_coords[bulk_mask, 0], umap_coords[bulk_mask, 1],
            c=colors[bulk_mask],
            s=80, marker="o", linewidths=0.5,
            edgecolors="black", alpha=0.9,
            rasterized=True, zorder=3, label="_nolegend_",
        )

    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    # ---- Cell-line colour legend (patch per cell line) ----
    n = len(paired_cls)
    leg_fontsize = 4 if n > 40 else 5 if n > 20 else 7
    ncol = min(max_legend_cols, max(1, n // 15))
    cl_handles = [
        mlines.Line2D([], [], marker="s", color=palette[cl], markersize=5,
                      linestyle="None", label=cl)
        for cl in paired_cls
    ]
    cl_legend = ax.legend(
        handles=cl_handles,
        title="cell line",
        fontsize=leg_fontsize,
        title_fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        ncol=ncol,
        borderpad=0.4,
        labelspacing=0.2,
        handlelength=0.8,
    )
    ax.add_artist(cl_legend)

    # ---- Modality marker legend ----
    marker_handles = []
    if show_sc:
        marker_handles.append(
            mlines.Line2D([], [], marker="x", color="grey", markersize=6, alpha=0.5,
                          linestyle="None", label="SC cell"),
        )
    marker_handles += [
        mlines.Line2D([], [], marker="o", color="grey", markersize=7, alpha=0.9,
                      markeredgecolor="black", markeredgewidth=0.4,
                      linestyle="None", label="bulk (per cell line)"),
        mlines.Line2D([], [], marker="^", color="grey", markersize=7, alpha=0.9,
                      markeredgecolor="black", markeredgewidth=0.4,
                      linestyle="None", label="SC mean (per cell line)"),
    ]
    ax.legend(
        handles=marker_handles,
        title="color = cell line",
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _build_combined(
    input_h5ad: str | Path,
    cell_line_col: str,
    n_cell_lines: int,
) -> tuple[ad.AnnData, list[str]]:
    """Load paired h5ad and return the joint SC + pseudobulk + bulk AnnData and paired cell lines."""
    print(f"Building combined SC + pseudobulk + bulk AnnData from {input_h5ad}")
    sc_adata, pb_adata, bulk_adata = generate_pseudobulk_chunks(
        input_h5ad=input_h5ad, return_adata=True, flush_every=n_cell_lines
    )
    combined = ad.concat(
        [a for a in [sc_adata, pb_adata, bulk_adata] if a is not None],
        axis=0,
        join="inner",
        merge="same",
    )
    combined.obs_names_make_unique()
    paired_cls = (
        sorted(set(sc_adata.obs[cell_line_col]) & set(bulk_adata.obs[cell_line_col]))
        if sc_adata is not None and bulk_adata is not None
        else []
    )
    print(f"  Combined: {combined.n_obs} total rows, {len(paired_cls)} paired cell lines")
    return combined, paired_cls


def _umap_and_plot_one(
    combined: ad.AnnData,
    paired_cls: list[str],
    cell_line_col: str,
    obsm_key: str,
    out_dir: Path,
    prefix: str,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    dpi: int,
    show_sc: bool = True,
) -> None:
    """Run UMAP on obsm_key, save the plot and persist UMAP coords under X_umap_{prefix}."""
    if show_sc:
        plot_data = combined
    else:
        plot_data = combined[combined.obs["modality"] != "sc"].copy()
        print(f"  --no-sc: dropped {(combined.obs['modality'] == 'sc').sum()} SC cells from UMAP")

    print("Computing UMAP...")
    sc.pp.neighbors(plot_data, use_rep=obsm_key, n_neighbors=min(n_neighbors, plot_data.n_obs - 1))
    sc.tl.umap(plot_data, min_dist=min_dist, random_state=seed)
    plot_data.obsm[f"X_umap_{prefix}"] = plot_data.obsm["X_umap"].copy()

    if not show_sc:
        # Write UMAP coords back into combined (SC rows stay NaN)
        umap_full = np.full((combined.n_obs, 2), np.nan, dtype=np.float32)
        idx = combined.obs_names.get_indexer(plot_data.obs_names)
        umap_full[idx] = plot_data.obsm["X_umap"]
        combined.obsm[f"X_umap_{prefix}"] = umap_full

    print("Plotting...")
    fig = plot_paired_umap(
        plot_data, paired_cls=paired_cls, cell_line_col=cell_line_col,
        title=prefix, show_sc=show_sc,
    )
    out_png = out_dir / f"{prefix}_paired_umap.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


def run(
    ckpt: str | Path | None,
    use_pca: bool,
    input_h5ad: str | Path,
    out_dir: str | Path,
    out_prefix: str,
    cell_line_col: str,
    domain_col: str,
    sc_label: str,
    bulk_label: str,
    is_log1p: bool,
    normalized: bool,
    n_cell_lines: int,
    batch_size: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    max_sc_per_line: int | None,
    device: str | None,
    dpi: int,
    show_sc: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined, paired_cls = _build_combined(input_h5ad, cell_line_col, n_cell_lines)

    print("Embedding...")
    if use_pca:
        from sklearn.decomposition import PCA
        n_components = min(50, combined.n_obs - 1, combined.n_vars - 1)
        pca = PCA(n_components=n_components, random_state=seed)
        X_dense = _to_dense(combined.X)
        combined.obsm["X_cf"] = pca.fit_transform(X_dense).astype(np.float32)
        print(f"  PCA: {n_components} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        if ckpt is None:
            raise ValueError("--ckpt is required when not using --pca or --ablation-dir.")
        print("Loading model...")
        model = _load_model(ckpt, device=device)
        combined.obsm["X_cf"] = _embed(model, combined, batch_size=batch_size, normalized=normalized)

    _umap_and_plot_one(combined, paired_cls, cell_line_col, "X_cf", out_dir, out_prefix, n_neighbors, min_dist, seed, dpi, show_sc=show_sc)

    out_h5ad = out_dir / f"{out_prefix}_paired_umap.h5ad"
    combined.write_h5ad(out_h5ad)
    print(f"Saved → {out_h5ad}")


def run_ablation(
    ablation_dir: str | Path,
    input_h5ad: str | Path,
    out_dir: str | Path,
    cell_line_col: str,
    n_cell_lines: int,
    batch_size: int,
    normalized: bool,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    device: str | None,
    dpi: int,
    show_sc: bool = True,
) -> None:
    """Embed with every checkpoint in ablation_dir, generate one UMAP plot per checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined, paired_cls = _build_combined(input_h5ad, cell_line_col, n_cell_lines)

    subdirs = sorted(d for d in Path(ablation_dir).iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {ablation_dir}")

    jobs: list[tuple[str, Path]] = []
    for subdir in subdirs:
        ckpts_in_dir = list(subdir.glob("*.ckpt"))
        if not ckpts_in_dir:
            print(f"  [skip] {subdir.name}: no .ckpt files found")
            continue
        latest = max(ckpts_in_dir, key=lambda p: p.stat().st_mtime)
        jobs.append((subdir.name, latest))
        print(f"  {subdir.name}: {latest.name}")

    if not jobs:
        raise FileNotFoundError(f"No subdirectories with .ckpt files found in {ablation_dir}")
    print(f"Found {len(jobs)} model(s)")

    for prefix, ckpt in jobs:
        obsm_key = f"X_cf_{prefix}"
        print(f"\n=== {prefix} ({ckpt.name}) ===")

        model = _load_model(ckpt, device=device)
        combined.obsm[obsm_key] = _embed(model, combined, batch_size=batch_size, normalized=normalized)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _umap_and_plot_one(combined, paired_cls, cell_line_col, obsm_key, out_dir, prefix, n_neighbors, min_dist, seed, dpi, show_sc=show_sc)

    out_h5ad = out_dir / "ablation_combined.h5ad"
    combined.write_h5ad(out_h5ad)
    print(f"\nSaved combined h5ad with all embeddings → {out_h5ad}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "UMAP of paired SC / pseudobulk / bulk profiles coloured by cell line."
        )
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ckpt", default=None, help="Path to a single CancerFoundation checkpoint.")
    mode.add_argument("--ablation-dir", default=None,
                      help="Directory of *.ckpt files; generates one UMAP plot per checkpoint.")
    mode.add_argument("--pca", action="store_true", help="Use PCA as embedder instead of a model.")
    p.add_argument("--input-h5ad", required=True, help="Paired h5ad with SC and bulk observations.")
    p.add_argument("--out-dir", default="./umap_outputs", help="Output directory.")
    p.add_argument("--out-prefix", default="paired", help="Prefix for output filenames (single-ckpt / PCA mode).")
    p.add_argument("--cell-line-col", default="cell_line", help="obs column with cell line identifier.")
    p.add_argument("--domain-col", default="domain", help="obs column distinguishing SC from bulk.")
    p.add_argument("--sc-label", default="SC", help="Value in domain_col for SC observations.")
    p.add_argument("--bulk-label", default="bulk", help="Value in domain_col for bulk observations.")
    p.add_argument("--is-log1p", action="store_true",
                   help="Input expression is log1p-transformed; expm1 before computing pseudobulk mean.")
    p.add_argument("--n-cell-lines", type=int, default=50, help="Number of cell lines to represent")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    p.add_argument("--neighbors", type=int, default=15, help="n_neighbors for UMAP.")
    p.add_argument("--min-dist", type=float, default=0.5, help="UMAP min_dist.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--max-sc-per-line", type=int, default=None,
                   help="Max SC cells to keep per cell line (randomly subsampled; default: all).")
    p.add_argument("--device", default=None, help="Torch device: cuda, cpu, cuda:0, … (auto).")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    p.add_argument("--no-sc", action="store_true",
                   help="Exclude SC cells from UMAP computation and plot (bulk + pseudobulk only).")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if not args.ckpt and not args.ablation_dir and not args.pca:
        _build_parser().error("one of --ckpt, --ablation-dir, or --pca is required.")

    if args.ablation_dir:
        run_ablation(
            ablation_dir=args.ablation_dir,
            input_h5ad=args.input_h5ad,
            out_dir=args.out_dir,
            cell_line_col=args.cell_line_col,
            n_cell_lines=args.n_cell_lines,
            batch_size=args.batch_size,
            normalized=True,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
            device=args.device,
            dpi=args.dpi,
            show_sc=not args.no_sc,
        )
    else:
        run(
            ckpt=args.ckpt,
            use_pca=args.pca,
            input_h5ad=args.input_h5ad,
            out_dir=args.out_dir,
            out_prefix=args.out_prefix,
            cell_line_col=args.cell_line_col,
            domain_col=args.domain_col,
            sc_label=args.sc_label,
            bulk_label=args.bulk_label,
            is_log1p=args.is_log1p,
            normalized=True,
            n_cell_lines=args.n_cell_lines,
            batch_size=args.batch_size,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
            max_sc_per_line=args.max_sc_per_line,
            device=args.device,
            dpi=args.dpi,
            show_sc=not args.no_sc,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
