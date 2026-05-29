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
# Data assembly
# --------------------------------------------------------------------------- #

def build_combined_adata(
    adata: ad.AnnData,
    cell_line_col: str,
    domain_col: str,
    sc_label: str,
    bulk_label: str,
    is_log1p: bool,
    max_sc_per_line: int | None,
    seed: int,
) -> tuple[ad.AnnData, list[str]]:
    """Build one AnnData containing SC + pseudobulk + bulk rows for each paired cell line.

    Returns the combined AnnData and the ordered list of paired cell line names.
    The AnnData has obs columns ``modality`` and ``cell_line_col``.
    """
    rng = np.random.default_rng(seed)

    domain_vals = adata.obs[domain_col].astype(str).values
    cell_line_vals = adata.obs[cell_line_col].astype(str).values

    sc_mask = domain_vals == sc_label
    bulk_mask = domain_vals == bulk_label

    sc_cls = set(cell_line_vals[sc_mask])
    bulk_cls = set(cell_line_vals[bulk_mask])
    paired_cls = sorted(sc_cls & bulk_cls)

    if not paired_cls:
        raise ValueError(
            f"No cell lines with both '{sc_label}' and '{bulk_label}' entries found. "
            f"Check --domain-col / --sc-label / --bulk-label."
        )
    print(f"Paired cell lines: {len(paired_cls)}")

    X_parts: list[np.ndarray] = []
    obs_rows: list[dict] = []

    for cl in paired_cls:
        cl_sc = (cell_line_vals == cl) & sc_mask
        cl_bulk = (cell_line_vals == cl) & bulk_mask

        X_sc = _to_dense(adata.X[cl_sc])
        X_bulk = _to_dense(adata.X[cl_bulk])

        # Optionally subsample SC cells for very large cell lines
        if max_sc_per_line is not None and X_sc.shape[0] > max_sc_per_line:
            idx = rng.choice(X_sc.shape[0], size=max_sc_per_line, replace=False)
            X_sc = X_sc[idx]

        # Pseudobulk: mean of all SC cells.
        # If the data is log1p-transformed, expm1 first so the mean is in linear space.
        if is_log1p:
            X_pb = np.expm1(X_sc).mean(axis=0, keepdims=True)
        else:
            X_pb = X_sc.mean(axis=0, keepdims=True)

        n_sc = X_sc.shape[0]
        n_bulk = X_bulk.shape[0]

        X_parts.extend([X_sc, X_pb, X_bulk])

        for _ in range(n_sc):
            obs_rows.append({"modality": "sc", cell_line_col: cl})
        obs_rows.append({"modality": "pseudobulk", cell_line_col: cl})
        for _ in range(n_bulk):
            obs_rows.append({"modality": "bulk", cell_line_col: cl})

    X_combined = np.vstack(X_parts).astype(np.float32)
    obs_df = pd.DataFrame(obs_rows)
    obs_df.index = obs_df.index.astype(str)

    combined = ad.AnnData(
        X=sp.csr_matrix(X_combined),
        obs=obs_df,
        var=adata.var.copy(),
    )
    return combined, paired_cls


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

    # Reduce alpha for SC scatter so larger markers remain visible
    sc_colors = colors[sc_mask].copy()
    sc_colors[:, 3] = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    # SC cells — small x
    if sc_mask.any():
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
    marker_handles = [
        mlines.Line2D([], [], marker="x", color="grey", markersize=6, alpha=0.5,
                      linestyle="None", label="SC cell"),
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

def run(
    ckpt: str | Path,
    input_h5ad: str | Path,
    out_dir: str | Path,
    out_prefix: str,
    cell_line_col: str,
    domain_col: str,
    sc_label: str,
    bulk_label: str,
    is_log1p: bool,
    normalized: bool,
    batch_size: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    max_sc_per_line: int | None,
    device: str | None,
    dpi: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading paired h5ad: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"  {adata.n_obs} obs × {adata.n_vars} vars")

    for col in [cell_line_col, domain_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not in obs. Available: {list(adata.obs.columns)}")

    print("Building combined SC + pseudobulk + bulk AnnData...")
    combined, paired_cls = build_combined_adata(
        adata,
        cell_line_col=cell_line_col,
        domain_col=domain_col,
        sc_label=sc_label,
        bulk_label=bulk_label,
        is_log1p=is_log1p,
        max_sc_per_line=max_sc_per_line,
        seed=seed,
    )
    print(f"  Combined: {combined.n_obs} total rows")

    print("Loading model...")
    model = _load_model(ckpt, device=device)

    print("Embedding...")
    combined.obsm["X_cf"] = _embed(model, combined, batch_size=batch_size, normalized=normalized)

    print("Computing UMAP...")
    n_cells = combined.n_obs
    sc.pp.neighbors(combined, use_rep="X_cf", n_neighbors=min(n_neighbors, n_cells - 1))
    sc.tl.umap(combined, min_dist=min_dist, random_state=seed)

    print("Plotting...")
    fig = plot_paired_umap(
        combined,
        paired_cls=paired_cls,
        cell_line_col=cell_line_col,
        title=out_prefix,
    )

    out_png = out_dir / f"{out_prefix}_paired_umap.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")

    out_h5ad = out_dir / f"{out_prefix}_paired_umap.h5ad"
    combined.write_h5ad(out_h5ad)
    print(f"Saved → {out_h5ad}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "UMAP of paired SC / pseudobulk / bulk profiles coloured by cell line."
        )
    )
    p.add_argument("--ckpt", required=True, help="Path to trained CancerFoundation checkpoint.")
    p.add_argument("--input-h5ad", required=True, help="Paired h5ad with SC and bulk observations.")
    p.add_argument("--out-dir", default="./umap_outputs", help="Output directory.")
    p.add_argument("--out-prefix", default="paired", help="Prefix for output filenames.")
    p.add_argument("--cell-line-col", default="cell_line", help="obs column with cell line identifier.")
    p.add_argument("--domain-col", default="domain", help="obs column distinguishing SC from bulk.")
    p.add_argument("--sc-label", default="SC", help="Value in domain_col for SC observations.")
    p.add_argument("--bulk-label", default="bulk", help="Value in domain_col for bulk observations.")
    p.add_argument("--is-log1p", action="store_true",
                   help="Input expression is log1p-transformed; expm1 before computing pseudobulk mean.")
    p.add_argument("--normalized", action="store_true",
                   help="Input is already CP10K+log1p; skip model-internal normalization.")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    p.add_argument("--neighbors", type=int, default=15, help="n_neighbors for UMAP.")
    p.add_argument("--min-dist", type=float, default=0.5, help="UMAP min_dist.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--max-sc-per-line", type=int, default=None,
                   help="Max SC cells to keep per cell line (randomly subsampled; default: all).")
    p.add_argument("--device", default=None, help="Torch device: cuda, cpu, cuda:0, … (auto).")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    run(
        ckpt=args.ckpt,
        input_h5ad=args.input_h5ad,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        cell_line_col=args.cell_line_col,
        domain_col=args.domain_col,
        sc_label=args.sc_label,
        bulk_label=args.bulk_label,
        is_log1p=args.is_log1p,
        normalized=args.normalized,
        batch_size=args.batch_size,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
        max_sc_per_line=args.max_sc_per_line,
        device=args.device,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
