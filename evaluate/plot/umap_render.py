"""Model-free UMAP computation and rendering.

Split out of ``umaps.py`` so scripts that already have an embedding (e.g.
``plot_tissue_umap.py``, which reads the ``X_pca`` written by
``data_preprocess/add_fields.py``) can render a UMAP without importing torch,
lightning or the ``cancerfoundation`` package. ``umaps.py`` re-imports everything
here, so its public API is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# Modality aliases recognised as single-cell.
_SC_MODALITY_ALIASES: frozenset[str] = frozenset({
    "sc", "singlecell", "scrna", "scrnaseq",
})

# Modality aliases recognised as pseudobulk (precomputed or generated).
_PB_MODALITY_ALIASES: frozenset[str] = frozenset({
    "pseudobulk", "synthpb", "pairedpb", "pseudo",
})

# Label values treated as "no value" when colouring. Module-level so callers that
# drop these rows outright (plot_tissue_umap.py --drop-unknown) and the colour
# mapper that greys them out (--skip-unknown) always mean the same rows.
UNKNOWN_LABELS: frozenset[str] = frozenset({"unknown", "nan", "none", "n/a", ""})


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


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


def _is_pb_modality(val: str) -> bool:
    """Return True if the modality string refers to pseudobulk data."""
    normalised = val.lower().replace(" ", "").replace("-", "").replace("_", "")
    return normalised in _PB_MODALITY_ALIASES or "pseudo" in normalised


def _canonical_modality(val: str) -> str:
    """Collapse a raw modality string to one of 'sc', 'pseudobulk', 'bulk'.

    Pseudobulk is checked first so a 'pseudobulk' label is never mistaken for bulk.
    """
    if _is_pb_modality(val):
        return "pseudobulk"
    if _is_sc_modality(val):
        return "sc"
    return "bulk"


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

    all_categories = sorted(col.astype(str).unique())
    categories = (
        [c for c in all_categories if c.lower() not in UNKNOWN_LABELS]
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
