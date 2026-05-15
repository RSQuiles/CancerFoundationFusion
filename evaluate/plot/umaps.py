"""UMAP evaluation utilities.

Single-model CLI usage:

    python umaps.py \
        --run-name experiment_name \
        --adata ./my_query.h5ad \
        --out-dir ./umap_outputs

Ablation CLI usage (generates one UMAP per model, saved inside each model dir):

    python umaps.py \
        --ablation-dir ./save/my_ablation \
        --adata ./my_query.h5ad \
        --color cancer_type modality

This script:
1) resolves a checkpoint from the run directory,
2) loads the `CancerFoundation` LightningModule,
3) embeds the provided AnnData via `CancerFoundation.embed(adata)`,
4) computes UMAP on the embeddings,
5) saves a UMAP plot and an annotated `.h5ad`.

When the AnnData contains a ``modality`` column, single-cell observations are
drawn as small transparent dots and bulk observations as larger stars, so both
populations are visually distinguishable while sharing the same colour scale.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch
from utils import sample_h5ad_subset_from_prefix, subsample_adata

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.model.model import CancerFoundation


_EPOCH_RE = re.compile(r"epoch_(\d+)")

# Modality aliases recognised as single-cell.
_SC_MODALITY_ALIASES: frozenset[str] = frozenset({
    "sc", "singlecell", "scrna", "scrnaseq",
})


# --------------------------------------------------------------------------- #
# Checkpoint / model helpers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_path: Path


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _epoch_num(p: Path) -> int:
    m = _EPOCH_RE.search(p.stem)
    return int(m.group(1)) if m else -1


def resolve_run_checkpoint(
    run_name: str,
    save_root: str | Path = "./save",
    ckpt: str | Path | None = None,
) -> RunPaths:
    """Resolve a run name to a checkpoint path.

    If `ckpt` is provided it is used directly; otherwise we look in
    `{save_root}/{run_name}` for `*.ckpt`.

    Heuristic:
    - Always prefer the newest-mtime (most recently modified) `*.ckpt`.
    """
    save_root = _as_path(save_root)
    run_dir = save_root / run_name
    if ckpt is not None:
        ckpt_path = _as_path(ckpt)
        if ckpt_path.is_dir():
            raise ValueError(f"--ckpt must be a file, got directory: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return RunPaths(run_dir=ckpt_path.parent, ckpt_path=ckpt_path)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpts = sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in: {run_dir}")

    # Always use the most recently modified checkpoint
    best = max(ckpts, key=lambda p: p.stat().st_mtime)
    return RunPaths(run_dir=run_dir, ckpt_path=best)


def _find_best_ckpt(model_dir: Path) -> Path | None:
    """Search model_dir and its checkpoints/ subdirectory for the best ckpt.
    
    Returns the most recently modified checkpoint.
    """
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "checkpoints/*.ckpt"):
        candidates.extend(model_dir.glob(pattern))
    if not candidates:
        return None
    # Return the most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_vocab_from_json(vocab_json: str | Path) -> dict:
    """Load a vocab mapping from a `vocab.json` file."""
    import json

    vocab_json = _as_path(vocab_json)
    with vocab_json.open("r") as f:
        return json.load(f)


def load_model_for_inference(
    ckpt_path: str | Path,
    vocab: dict | None = None,
    device: str | None = None,
) -> CancerFoundation:
    """Load a trained model checkpoint and move to the requested device."""
    ckpt_path = _as_path(ckpt_path)
    if vocab is None:
        model = CancerFoundation.load_from_checkpoint(str(ckpt_path))
    else:
        model = CancerFoundation.load_from_checkpoint(str(ckpt_path), vocab=vocab)

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Embedding / UMAP
# --------------------------------------------------------------------------- #

def embed_adata(
    model: CancerFoundation,
    adata: sc.AnnData,
    batch_size: int = 64,
    flavor: str = "seurat",
    obsm_key: str = "X_cf",
    normalized: bool = True,
) -> sc.AnnData:
    """Compute embeddings and store them in `adata.obsm[obsm_key]`."""
    emb_df: pd.DataFrame = model.embed(adata, flavor=flavor, batch_size=batch_size, normalized=normalized)
    adata.obsm[obsm_key] = emb_df.to_numpy(dtype=np.float32)
    return adata


def compute_umap(
    adata: sc.AnnData,
    use_rep: str = "X_cf",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 0,
) -> sc.AnnData:
    """Compute UMAP on `adata.obsm[use_rep]` and store in `adata.obsm['X_umap']`."""
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    return adata


# --------------------------------------------------------------------------- #
# Modality-aware UMAP plot
# --------------------------------------------------------------------------- #

def _is_sc_modality(val: str) -> bool:
    """Return True if the modality string refers to single-cell data."""
    normalised = val.lower().replace(" ", "").replace("-", "").replace("_", "")
    return normalised in _SC_MODALITY_ALIASES or normalised.startswith("sc")


def _assign_colors(
    col: pd.Series,
    skip_unknown: bool = False,
) -> tuple[np.ndarray, dict[str, tuple] | None, list[str] | None]:
    """
    Map a Series to RGBA colours.

    Returns
    -------
    rgba       : (N, 4) float32 array of RGBA colours.
    cat_colors : category → RGBA mapping, or None for continuous data.
    categories : sorted category list, or None for continuous data.
    """
    if (
        pd.api.types.is_numeric_dtype(col)
        and not pd.api.types.is_bool_dtype(col)
        and col.nunique() > 20
    ):
        vals = col.to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(vmin, vmax) if vmin != vmax else plt.Normalize(0, 1)
        rgba = plt.get_cmap("viridis")(norm(vals)).astype(np.float32)
        return rgba, None, None

    _SKIP = {"unknown", "nan", "none", "n/a", ""}
    all_categories = sorted(col.astype(str).unique())
    categories = (
        [c for c in all_categories if c.lower() not in _SKIP]
        if skip_unknown
        else all_categories
    )

    n = len(categories)
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    _NORMAL_ALPHA = 0.6
    invisible = (0.75, 0.75, 0.75, 0.05)
    cat_colors = {cat: (*cmap(i % cmap.N)[:3], _NORMAL_ALPHA) for i, cat in enumerate(categories)}

    # Skipped categories -> invisible
    rgba = np.array(
        [cat_colors.get(str(v), invisible) for v in col],
        dtype=np.float32,
    )
    return rgba, cat_colors, categories


def _plot_umap_modality_aware(
    adata: sc.AnnData,
    color_keys: list[str | None],
    title: str | None,
    skip_unknown: bool = False
) -> plt.Figure:
    """
    UMAP plot with modality-sensitive markers.

    Single-cell observations → small transparent dots (marker "o").
    Bulk observations        → larger stars with dark edge (marker "*").

    One panel is produced per entry in ``color_keys``.
    """
    umap_coords = adata.obsm["X_umap"]
    modality_vals = adata.obs["modality"].astype(str).replace("nan", "sc").to_numpy()
    sc_mask   = np.array([_is_sc_modality(v) for v in modality_vals])
    bulk_mask = ~sc_mask

    n_panels = max(len(color_keys), 1)
    # Extra width per panel for the legend
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7.5 * n_panels, 6.0),
        squeeze=False,
    )
    axes_flat = axes[0]

    for ax, color_key in zip(axes_flat, color_keys):
        if color_key is not None and color_key in adata.obs:
            # Force string to ensure categorical treatment
            col = adata.obs[color_key].astype(str)
            point_colors, cat_colors, categories = _assign_colors(col, skip_unknown=skip_unknown)
        else:
            fallback = np.array([0.35, 0.55, 0.80, 1.0], dtype=np.float32)
            point_colors = np.tile(fallback, (len(adata), 1))
            cat_colors, categories = None, None

        if sc_mask.any():
            ax.scatter(
                umap_coords[sc_mask, 0], umap_coords[sc_mask, 1],
                c=point_colors[sc_mask],
                s=4, marker="o", linewidths=0, rasterized=True,
            )
        if bulk_mask.any():
            ax.scatter(
                umap_coords[bulk_mask, 0], umap_coords[bulk_mask, 1],
                c=point_colors[bulk_mask],
                s=4, marker="D", linewidths=0, edgecolors="black", rasterized=True,
            )

        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(color_key or "", fontsize=10, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Place category legend inside the axes (upper right), pinned with add_artist
        if cat_colors and categories:
            color_handles = [
                mpatches.Patch(color=cat_colors[cat], label=cat)
                for cat in categories
            ]
            n_cats = len(categories)
            # For many categories use smaller font and multiple columns
            leg_fontsize = 3 if n_cats > 50 else 5 if n_cats > 20 else 6
            ncol = max(1, n_cats // 25)
            cat_legend = ax.legend(
                handles=color_handles,
                title=color_key,
                fontsize=leg_fontsize,
                title_fontsize=8,
                loc="lower right",
                frameon=True,
                framealpha=0.85,
                ncol=ncol,
                borderpad=0.5,
                labelspacing=0.3,
                handlelength=1.0,
            )
            ax.add_artist(cat_legend)  # pin so modality legend doesn't overwrite

    # Modality legend on first panel, lower left
    modality_handles = [
        mlines.Line2D(
            [], [], marker="o", color="grey", markersize=5, alpha=0.6,
            linestyle="None", label="Single-cell",
        ),
        mlines.Line2D(
            [], [], marker="D", color="grey", markersize=5, alpha=0.6,
            markeredgecolor="black", markeredgewidth=0.2,
            linestyle="None", label="Bulk",
        ),
    ]
    axes_flat[0].legend(
        handles=modality_handles,
        title="Modality", fontsize=8, title_fontsize=8,
        loc="lower left", frameon=True, framealpha=0.9,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Public save helper
# --------------------------------------------------------------------------- #

def save_umap_plot(
    adata: sc.AnnData,
    out_png: str | Path,
    color: str | Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 200,
    skip_unknown: bool = False,
) -> Path:
    """Save a UMAP plot to a PNG/PDF file.

    If ``adata.obs`` contains a ``modality`` column the plot uses
    modality-aware markers (dots for single-cell, stars for bulk).
    Otherwise the standard scanpy renderer is used.
    """
    out_png = _as_path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    has_modality = "modality" in adata.obs.columns

    if has_modality:
        if color is None:
            color_keys: list[str | None] = [None]
        elif isinstance(color, str):
            color_keys = [color]
        else:
            color_keys = list(color)

        fig = _plot_umap_modality_aware(adata, color_keys, title, skip_unknown=skip_unknown)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        sc.pl.umap(adata, color=color, title=title, show=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi)
        plt.close()

    return out_png


# --------------------------------------------------------------------------- #
# Modality-split UMAP helper
# --------------------------------------------------------------------------- #

def _plot_three_modality_umap(
    adata: sc.AnnData,
    title: str | None = None,
) -> plt.Figure:
    """UMAP coloured by modality only: single-cell, bulk, pseudobulk."""
    umap_coords = adata.obsm["X_umap"]
    modality_vals = adata.obs["modality"].astype(str).to_numpy()

    sc_mask   = modality_vals == "sc"
    bulk_mask = modality_vals == "bulk"
    pb_mask   = modality_vals == "pseudobulk"

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    if sc_mask.any():
        ax.scatter(
            umap_coords[sc_mask, 0], umap_coords[sc_mask, 1],
            color="#4393c3", s=4, marker="o", linewidths=0,
            alpha=0.4, rasterized=True, label="Single-cell",
        )
    if pb_mask.any():
        ax.scatter(
            umap_coords[pb_mask, 0], umap_coords[pb_mask, 1],
            color="#4dac26", s=30, marker="^", linewidths=0.5,
            edgecolors="black", alpha=0.85, rasterized=True, label="Pseudobulk",
        )
    if bulk_mask.any():
        ax.scatter(
            umap_coords[bulk_mask, 0], umap_coords[bulk_mask, 1],
            color="#d6604d", s=30, marker="D", linewidths=0.5,
            edgecolors="black", alpha=0.85, rasterized=True, label="Bulk",
        )

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        title="Modality", fontsize=8, title_fontsize=8,
        loc="lower left", frameon=True, framealpha=0.9,
    )
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _generate_pseudobulk_adata(
    sc_adata: sc.AnnData,
    group_column: str = "tissue_general",
    n_sc_per_pb: int = 10,
    agg_method: str = "mean",
    n_pb: int | None = None,
    seed: int = 0,
) -> sc.AnnData | None:
    """Aggregate SC expression within tissue groups to create pseudobulk profiles.

    For each pseudobulk, ``n_sc_per_pb`` cells are drawn from a single randomly
    chosen ``group_column`` group and their expression vectors are aggregated
    (element-wise mean or sum), mirroring ``BulkSCSampler``.

    Returns an AnnData with pseudobulk expression profiles sharing the same
    ``var`` as ``sc_adata``, or ``None`` if generation is not possible.
    """
    import scipy.sparse as sp

    if group_column not in sc_adata.obs.columns:
        print(f"  [warn] '{group_column}' not in obs — cannot generate pseudobulks")
        return None

    rng = np.random.default_rng(seed)

    group_vals    = sc_adata.obs[group_column].astype(str).to_numpy()
    unique_groups = np.unique(group_vals)
    group_to_idx: dict[str, np.ndarray] = {
        g: np.where(group_vals == g)[0] for g in unique_groups
    }
    valid_groups = [g for g, idx in group_to_idx.items() if len(idx) >= n_sc_per_pb]

    if not valid_groups:
        print(
            f"  [warn] pseudobulk UMAP skipped — no '{group_column}' group "
            f"has >= {n_sc_per_pb} SC cells"
        )
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
        pb_rows.append(expr.sum(axis=0) if agg_method == "sum" else expr.mean(axis=0))

    pb_X = np.array(pb_rows, dtype=np.float32)
    pb_obs = pd.DataFrame(
        {"modality": ["pseudobulk"] * n_pb, group_column: chosen_groups},
        index=[f"pb_{i}" for i in range(n_pb)],
    )
    return AnnData(X=pb_X, obs=pb_obs, var=sc_adata.var.copy())


def _save_pseudobulk_umap(
    adata: sc.AnnData,
    model: CancerFoundation,
    joint_out_png: Path,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    dpi: int = 200,
) -> None:
    """Compute and save a UMAP with SC, bulk, and pseudobulk observations.

    Pseudobulks are generated by aggregating (mean or sum) the raw expression
    of ``n_sc_per_pb`` single-cell observations from the same ``group_column``
    group, then embedding the resulting profile through the model — mirroring
    how real bulk samples are processed.  The UMAP is coloured by modality only.

    Output is named after ``joint_out_png`` with ``_pseudobulk`` inserted
    before the extension, e.g. ``umap.png`` → ``umap_pseudobulk.png``.
    """
    if "modality" not in adata.obs.columns:
        return
    if "X_cf" not in adata.obsm:
        print("  [warn] pseudobulk UMAP skipped — 'X_cf' not in obsm")
        return
    if group_column not in adata.obs.columns:
        print(f"  [warn] pseudobulk UMAP skipped — '{group_column}' not in obs")
        return

    modality_vals = adata.obs["modality"].astype(str).replace("nan", "sc").to_numpy()
    sc_mask   = np.array([_is_sc_modality(v) for v in modality_vals])
    bulk_mask = ~sc_mask

    sc_adata   = adata[sc_mask]
    bulk_adata = adata[bulk_mask]

    if sc_adata.n_obs == 0:
        print("  [warn] pseudobulk UMAP skipped — no SC cells found")
        return

    n_pb_target = bulk_adata.n_obs if bulk_adata.n_obs > 0 else None
    pb_adata = _generate_pseudobulk_adata(
        sc_adata,
        group_column=group_column,
        n_sc_per_pb=n_sc_per_pb,
        agg_method=agg_method,
        n_pb=n_pb_target,
        seed=seed,
    )
    if pb_adata is None:
        return

    try:
        embed_adata(model, pb_adata, batch_size=embed_batch_size,
                    flavor=flavor, obsm_key="X_cf")
    except Exception as exc:
        print(f"  [warn] pseudobulk embedding failed: {exc}")
        return

    sc_emb   = sc_adata.obsm["X_cf"]
    pb_emb   = pb_adata.obsm["X_cf"]
    bulk_emb = (
        bulk_adata.obsm["X_cf"]
        if bulk_adata.n_obs > 0
        else np.empty((0, sc_emb.shape[1]), dtype=np.float32)
    )

    combined_emb = np.vstack([sc_emb, pb_emb, bulk_emb])
    modality_col = (
        ["sc"]           * sc_adata.n_obs
        + ["pseudobulk"] * pb_adata.n_obs
        + ["bulk"]       * bulk_adata.n_obs
    )

    combined = AnnData(obs=pd.DataFrame({"modality": modality_col}))
    combined.obsm["X_cf"] = combined_emb

    n_cells = combined.n_obs
    try:
        compute_umap(
            combined,
            use_rep="X_cf",
            n_neighbors=min(n_neighbors, n_cells - 1),
            min_dist=min_dist,
            random_state=seed,
        )
    except Exception as exc:
        print(f"  [warn] pseudobulk UMAP computation failed: {exc}")
        return

    stem    = joint_out_png.stem
    suffix  = joint_out_png.suffix
    out_png = joint_out_png.parent / f"{stem}_pseudobulk{suffix}"

    try:
        fig = _plot_three_modality_umap(combined, title=f"{stem} (pseudobulk)")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {out_png}")
    except Exception as exc:
        print(f"  [warn] pseudobulk UMAP plot failed: {exc}")


def _save_modality_split_umaps(
    adata: sc.AnnData,
    joint_out_png: Path,
    color: list[str] | None,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    dpi: int = 200,
    skip_unknown: bool = False,
    model: CancerFoundation | None = None,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
    embed_batch_size: int = 64,
    flavor: str = "seurat",
) -> None:
    """Compute and save separate UMAP plots for SC-only and bulk-only subsets.

    Each subset re-runs ``sc.pp.neighbors`` + ``sc.tl.umap`` on its own cells
    (using the already-embedded ``X_cf`` vectors) so the UMAP layout reflects
    the internal structure of that modality rather than the shared joint space.

    When ``model`` is provided, also generates a pseudobulk UMAP (see
    ``_save_pseudobulk_umap``).

    Output files are named after ``joint_out_png`` with ``_sc`` / ``_bulk``
    inserted before the extension, e.g.:
        umap.png  →  umap_sc.png   umap_bulk.png   umap_pseudobulk.png
    """
    if "modality" not in adata.obs.columns:
        return

    modality_vals = adata.obs["modality"].astype(str).replace("nan", "sc").to_numpy()
    sc_mask   = np.array([_is_sc_modality(v) for v in modality_vals])
    bulk_mask = ~sc_mask

    stem = joint_out_png.stem
    suffix = joint_out_png.suffix
    out_dir = joint_out_png.parent

    for label, mask in [("sc", sc_mask), ("bulk", bulk_mask)]:
        n = int(mask.sum())
        if n == 0:
            continue

        subset = adata[mask].copy()
        out_png = out_dir / f"{stem}_{label}{suffix}"
        try:
            compute_umap(
                subset,
                use_rep="X_cf",
                n_neighbors=min(n_neighbors, n - 1),
                min_dist=min_dist,
                random_state=seed,
            )
            save_umap_plot(
                subset,
                out_png=out_png,
                color=color,
                title=f"{stem} ({label})",
                dpi=dpi,
                skip_unknown=skip_unknown,
            )
            print(f"  saved → {out_png}")
        except Exception as exc:
            print(f"  [warn] {label}-only UMAP failed: {exc}")

    if model is not None:
        _save_pseudobulk_umap(
            adata, model, joint_out_png,
            n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
            n_sc_per_pb=n_sc_per_pb, group_column=group_column,
            agg_method=agg_method, embed_batch_size=embed_batch_size,
            flavor=flavor, dpi=dpi,
        )


# --------------------------------------------------------------------------- #
# Ablation-level UMAP generation
# --------------------------------------------------------------------------- #

def run_ablation_umaps(
    ablation_dir: str | Path,
    adata: AnnData,
    color: list[str] | None = None,
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    seed: int = 0,
    device: str | None = None,
    vocab: dict | None = None,
    skip_unknown: bool = False,
    modality_split: bool = True,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
) -> None:
    """Generate and save a UMAP for every model inside an ablation directory.

    For each ``{model_dir}`` found under ``ablation_dir`` that contains a
    ``.ckpt`` file (directly or inside a ``checkpoints/`` subdirectory), the
    function:

    1. Loads the model from the best-epoch checkpoint.
    2. Embeds a copy of ``adata``.
    3. Computes UMAP coordinates.
    4. Saves the joint figure to ``{model_dir}/umap.png``.
    5. If ``adata.obs`` contains a ``modality`` column and ``modality_split``
       is True, also saves separate UMAPs for SC-only and bulk-only cells to
       ``{model_dir}/umap_sc.png`` and ``{model_dir}/umap_bulk.png``.

    Parameters
    ----------
    ablation_dir   : path to the top-level ablation experiment directory.
    adata          : AnnData to embed (a copy is made per model).
    color          : obs column(s) to colour by.
    modality_split : when True (default), generate per-modality UMAPs if the
                     ``modality`` column is present.
    """
    ablation_dir = _as_path(ablation_dir)

    model_dirs = sorted(d for d in ablation_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No subdirectories found in {ablation_dir}.")
        return

    for model_dir in model_dirs:
        ckpt_path = _find_best_ckpt(model_dir)
        if ckpt_path is None:
            print(f"[skip] {model_dir.name} — no checkpoint found")
            continue

        print(f"[{model_dir.name}] checkpoint: {ckpt_path.name}")

        try:
            model = load_model_for_inference(ckpt_path, vocab=vocab, device=device)
        except Exception as exc:
            print(f"  [error] could not load model: {exc}")
            continue

        adata_copy = adata.copy()
        try:
            embed_adata(model, adata_copy, batch_size=embed_batch_size,
                        flavor=flavor, obsm_key="X_cf")
            compute_umap(adata_copy, use_rep="X_cf", n_neighbors=n_neighbors,
                         min_dist=min_dist, random_state=seed)
            out_png = model_dir / "umap.png"
            save_umap_plot(adata_copy, out_png=out_png, color=color,
                           title=model_dir.name, skip_unknown=skip_unknown)
            print(f"  saved → {out_png}")

            if modality_split:
                _save_modality_split_umaps(
                    adata_copy, out_png, color,
                    n_neighbors, min_dist, seed,
                    skip_unknown=skip_unknown,
                    model=model,
                    n_sc_per_pb=n_sc_per_pb,
                    group_column=group_column,
                    agg_method=agg_method,
                    embed_batch_size=embed_batch_size,
                    flavor=flavor,
                )
        except Exception as exc:
            print(f"  [error] UMAP generation failed: {exc}")
        finally:
            del model, adata_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_color_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for v in values:
        if "," in v:
            out.extend([x for x in (s.strip() for s in v.split(",")) if x])
        else:
            out.append(v)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)

    # ---- common: must supply some adata source ----
    assert args.adata_dir is not None or args.adata is not None, (
        "A path to an AnnData (--adata) or a directory (--adata-dir) must be provided."
    )
    if args.adata_dir is not None:
        assert args.adata is None, (
            "Provide either --adata or --adata-dir, not both."
        )

    vocab = None
    if args.vocab_json is not None:
        vocab = load_vocab_from_json(args.vocab_json)

    color = _parse_color_list(args.color)

    # ---- load adata ----
    sample_size = args.sample_size
    if args.adata is not None:
        adata_path = _as_path(args.adata)
        if not adata_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {adata_path}")
        print(f"Loading {sample_size} cells from {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        adata = subsample_adata(adata, sample_size)
    else:
        adata_dir  = args.adata_dir
        adata_prefix = args.adata_prefix  # may be None → use all .h5ad files
        print(
            f"Loading {sample_size} cells from "
            f"{'all' if not adata_prefix else repr(adata_prefix)} "
            f".h5ad files in {adata_dir}..."
        )
        adata = sample_h5ad_subset_from_prefix(adata_prefix, adata_dir, sample_size)

    # ---- ablation mode ----
    if args.ablation_dir is not None:
        run_ablation_umaps(
            ablation_dir=args.ablation_dir,
            adata=adata,
            color=color,
            embed_batch_size=args.embed_batch_size,
            flavor=args.flavor,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
            device=args.device,
            vocab=vocab,
            skip_unknown=args.skip_unknown,
            modality_split=not args.no_modality_split,
            n_sc_per_pb=args.n_sc_per_pb,
            group_column=args.pb_group_column,
            agg_method=args.pb_agg_method,
        )
        return 0

    # ---- single-model mode ----
    assert args.run_name is not None, (
        "--run-name is required unless --ablation-dir is provided."
    )

    paths = resolve_run_checkpoint(
        run_name=args.run_name,
        save_root=args.save_root,
        ckpt=args.ckpt,
    )
    out_dir = _as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix   = args.out_prefix or args.run_name
    out_png  = out_dir / f"{prefix}.umap.png"
    out_h5ad = out_dir / f"{prefix}.umap.h5ad"

    try:
        print("Loading model...")
        model = load_model_for_inference(
            paths.ckpt_path, vocab=vocab, device=args.device
        )
    except TypeError as e:
        raise RuntimeError(
            "Failed to load model from checkpoint. If the checkpoint was saved "
            "without vocab in hparams, pass --vocab-json with the training vocab.json."
        ) from e

    print("Embedding...")
    print(adata)
    embed_adata(model, adata, batch_size=args.embed_batch_size,
                flavor=args.flavor, obsm_key="X_cf")

    print("Computing UMAP...")
    compute_umap(
        adata,
        use_rep="X_cf",
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )

    save_umap_plot(adata, out_png=out_png, color=color, title=prefix, skip_unknown=args.skip_unknown)

    if not args.no_modality_split:
        _save_modality_split_umaps(
            adata, out_png, color,
            args.neighbors, args.min_dist, args.seed,
            skip_unknown=args.skip_unknown,
            model=model,
            n_sc_per_pb=args.n_sc_per_pb,
            group_column=args.pb_group_column,
            agg_method=args.pb_agg_method,
            embed_batch_size=args.embed_batch_size,
            flavor=args.flavor,
        )

    adata.write_h5ad(out_h5ad)
    print(f"Checkpoint: {paths.ckpt_path}")
    print(f"Saved plot:  {out_png}")
    print(f"Saved h5ad:  {out_h5ad}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed AnnData and save UMAP plot.")

    # --- run identification (single-model mode) ---
    p.add_argument(
        "--run-name",
        default=None,
        help="Run name under --save-root (required for single-model mode).",
    )
    p.add_argument(
        "--save-root",
        default="../submits_biomed/save",
        help="Root folder containing run directories (default: ../submits_biomed/save).",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional explicit checkpoint path (overrides run-name resolution).",
    )

    # --- ablation mode ---
    p.add_argument(
        "--ablation-dir",
        default=None,
        help=(
            "Generate one UMAP per model found inside this ablation directory. "
            "Each UMAP is saved to {model_dir}/umap.png. "
            "Mutually exclusive with --run-name."
        ),
    )

    # --- data source ---
    p.add_argument("--adata", help="Path to input .h5ad.")
    p.add_argument(
        "--adata-dir",
        help="Directory with .h5ad files to virtually concatenate.",
    )
    p.add_argument(
        "--adata-prefix",
        type=str,
        default=None,
        help=(
            "Filename prefix for virtual concatenation (e.g. 'train'). "
            "If omitted, all .h5ad files in --adata-dir are used."
        ),
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help="Number of cells to sample from the input AnnData (default: 50000).",
    )

    # --- single-model output ---
    p.add_argument(
        "--out-dir",
        default="./umap",
        help="Output directory for plot and annotated h5ad (single-model mode).",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Optional prefix for output files (defaults to run-name).",
    )

    # --- misc ---
    p.add_argument(
        "--vocab-json",
        default=None,
        help="Optional vocab.json path (only needed if checkpoint lacks vocab).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: cuda, cpu, cuda:0, ... (default: auto).",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size inside model.embed (default: 64).",
    )
    p.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="n_neighbors for UMAP graph (default: 15).",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="min_dist for UMAP (default: 0.5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for UMAP (default: 0).",
    )
    p.add_argument(
        "--color",
        nargs="*",
        default=None,
        help="obs column(s) to colour by (space- or comma-separated).",
    )
    p.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        help="Flavor used for HVG selection (default: seurat).",
    )
    p.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Whether to skip the unknown / nan categories.",
    )
    p.add_argument(
        "--no-modality-split",
        action="store_true",
        help=(
            "Disable per-modality UMAPs. By default, when the AnnData contains "
            "a 'modality' column, separate SC-only and bulk-only UMAPs are "
            "computed and saved alongside the joint UMAP."
        ),
    )
    p.add_argument(
        "--n-sc-per-pb",
        type=int,
        default=10,
        help=(
            "Number of single-cell samples aggregated into each pseudobulk "
            "(default: 10)."
        ),
    )
    p.add_argument(
        "--pb-group-column",
        type=str,
        default="tissue_general",
        help=(
            "obs column used to group SC cells when sampling pseudobulks. "
            "Cells composing one pseudobulk are drawn exclusively from the "
            "same group (default: tissue_general)."
        ),
    )
    p.add_argument(
        "--pb-agg-method",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help=(
            "Aggregation method for combining SC expression into a pseudobulk "
            "profile before embedding: 'mean' (default) or 'sum'."
        ),
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
