"""UMAP computation and drawing, shared by ``plot_umap.py`` and ``plot_alignment.py``.

Ported from ``evaluate_old/plot/umap_render.py`` and ``evaluate_old/plot/umaps.py``,
with two changes: palettes cycle through tab20/tab20b/tab20c instead of repeating
colours past 20 categories, and modality is read from the canonical labels only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate.modalities import DISPLAY_NAMES, MODALITIES

# Label values treated as "no value" when colouring.
UNKNOWN_LABELS: frozenset[str] = frozenset({"unknown", "nan", "none", "n/a", ""})
GREY = (0.75, 0.75, 0.75, 0.15)
MAX_LEGEND_ENTRIES = 60  # beyond this colours repeat anyway

# Per-modality marker styling. Colour is used only when a panel is coloured by
# modality; other panels recolour per point and keep the marker shape.
MODALITY_STYLE: dict[str, dict] = {
    "sc":         dict(color="#4393c3", s=4,  marker="o", linewidths=0),
    "pseudobulk": dict(color="#4dac26", s=30, marker="^", linewidths=0.5, edgecolors="black"),
    # "bulk":       dict(color="#d6604d", s=30, marker="D", linewidths=0.5, edgecolors="black"),
    "bulk":       dict(color="#4393c3", s=4,  marker="o", linewidths=0),
}
_PLAIN_STYLE = dict(s=6, marker="o", linewidths=0)


def compute_umap(
    emb: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """2-D UMAP coordinates of ``emb`` (scanpy neighbors + umap)."""
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(emb))]),
                     obsm={"X_emb": np.asarray(emb, dtype=np.float32)})
    sc.pp.neighbors(tmp, use_rep="X_emb", n_neighbors=min(n_neighbors, max(2, len(emb) - 1)),
                    random_state=seed)
    sc.tl.umap(tmp, min_dist=min_dist, random_state=seed)
    return np.asarray(tmp.obsm["X_umap"])


def categorical_palette(categories: list[str]) -> dict[str, tuple]:
    """Distinct colours, never grey (grey means unknown/unpaired in these figures).

    tab10 for up to 9 categories, then tab20 -> tab20b -> tab20c, cycling past ~70.
    """
    if len(categories) <= 9:
        colors = [plt.get_cmap("tab10")(i) for i in range(10) if i != 7]
    else:
        colors = [plt.get_cmap("tab20")(i) for i in range(20) if i not in (14, 15)]
        colors += [plt.get_cmap(name)(i) for name in ("tab20b", "tab20c") for i in range(20)
                   if not (name == "tab20c" and i >= 16)]
    return {c: colors[i % len(colors)] for i, c in enumerate(categories)}


def assign_colors(
    col: pd.Series,
    skip_unknown: bool = False,
    alpha: float = 0.6,
) -> tuple[np.ndarray, dict[str, tuple] | None, list[str] | None]:
    """RGBA per value; categories (or a viridis scale for >20 numeric values).

    Returns the colours, the category -> colour mapping and the category order
    (both None for continuous data). ``skip_unknown`` greys out unknown labels.
    """
    if (
        pd.api.types.is_numeric_dtype(col)
        and not pd.api.types.is_bool_dtype(col)
        and col.nunique() > 20
    ):
        vals = col.to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(vmin, vmax) if vmin != vmax else plt.Normalize(0, 1)
        return plt.get_cmap("viridis")(norm(vals)).astype(np.float32), None, None

    values = col.astype(str)
    # Natural order, so pair ids read 1, 2, ..., 10 rather than 1, 10, 2.
    categories = sorted(values.unique(), key=lambda v: (0, int(v), "") if v.isdigit() else (1, 0, v.lower()))
    if skip_unknown:
        categories = [c for c in categories if c.lower() not in UNKNOWN_LABELS]
    palette = {c: (*rgb[:3], alpha) for c, rgb in categorical_palette(categories).items()}
    rgba = np.array([palette.get(v, GREY) for v in values], dtype=np.float32)
    return rgba, palette, categories


def _style_axes(ax, title: str, axis_labels: tuple[str, str] = ("UMAP 1", "UMAP 2")) -> None:
    ax.set_xlabel(axis_labels[0], fontsize=9)
    ax.set_ylabel(axis_labels[1], fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _scatter(ax, coords, modality, colors=None, order=None) -> None:
    """Draw points, with per-modality marker shapes when ``modality`` is given."""
    if modality is None:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=None if colors is not None else 0.5,
                   rasterized=True, **_PLAIN_STYLE)
        return
    for mod in order or MODALITIES:
        mask = modality == mod
        if not mask.any():
            continue
        style = dict(MODALITY_STYLE.get(mod, _PLAIN_STYLE))
        if colors is not None:
            style.pop("color", None)
            style["c"] = colors[mask]
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.5, rasterized=True, **style)


def _modality_handles(modality: np.ndarray, colored: bool) -> list:
    handles = []
    for mod in MODALITIES:
        if not (modality == mod).any():
            continue
        style = MODALITY_STYLE[mod]
        color = style["color"] if colored else "dimgrey"
        handles.append(mlines.Line2D(
            [], [], linestyle="None", markersize=6, marker=style["marker"], color=color,
            markeredgecolor="black" if mod != "sc" else color, label=DISPLAY_NAMES[mod],
        ))
    return handles


def _category_legend(ax, palette, categories, title, loc="center left") -> None:
    if len(categories) > MAX_LEGEND_ENTRIES:
        ax.text(1.01, 0.5, f"{len(categories)} {title} values\n(legend omitted)",
                transform=ax.transAxes, fontsize=7, va="center")
        return
    n = len(categories)
    legend = ax.legend(
        handles=[mpatches.Patch(color=palette[c], label=c) for c in categories],
        title=title, fontsize=5 if n > 20 else 7, title_fontsize=8, loc=loc,
        bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1 if n <= 25 else 2,
        handlelength=1.0, labelspacing=0.3,
    )
    ax.add_artist(legend)


def draw_umap_panels(
    coords: np.ndarray,
    obs: pd.DataFrame,
    color_keys: list[str],
    modality_col: str | None = None,
    skip_unknown: bool = False,
    title: str | None = None,
    segments: list[np.ndarray] | None = None,
    highlight: dict[str, np.ndarray] | None = None,
    labels: dict[str, str] | None = None,
    axis_labels: tuple[str, str] = ("UMAP 1", "UMAP 2"),
) -> plt.Figure:
    """One panel per colour key; marker shape encodes modality when ``modality_col`` is set.

    ``color_keys`` may include ``modality_col`` itself, which colours by the fixed
    modality palette. ``segments`` (each a (2, 2) array) are drawn faintly under the
    points of every panel. ``highlight`` maps a colour key to a boolean mask; rows
    outside it are drawn grey in that panel (e.g. unpaired rows on the pair panel).
    ``labels`` maps an obs column to the name shown in panel and legend titles
    (``{"tissue_general": "Tissue"}``); category values are drawn as they are.
    ``axis_labels`` names the two axes (``plot_pca`` passes ``"PC1 (12.3%)"`` and so on).
    """
    labels = labels or {}
    modality_label = labels.get(modality_col or "modality", "Modality")
    modality = (
        obs[modality_col].astype(str).to_numpy()
        if modality_col is not None and modality_col in obs.columns else None
    )
    keys = [k for k in color_keys if k in obs.columns] or [None]
    missing = sorted(set(color_keys) - set(obs.columns))
    if missing:
        print(f"  [warn] colour keys not in obs, skipped: {missing}")

    fig, axes = plt.subplots(1, len(keys), figsize=(6.5 * len(keys), 5.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        if segments:
            from matplotlib.collections import LineCollection

            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.3, alpha=0.25, zorder=0))
        if key is None:
            _scatter(ax, coords, modality)
            _style_axes(ax, "embedding", axis_labels)
        elif modality is not None and key == modality_col:
            _scatter(ax, coords, modality)
            _style_axes(ax, modality_label, axis_labels)
            ax.legend(handles=_modality_handles(modality, colored=True), title=modality_label,
                      fontsize=8, title_fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                      frameon=False)
        else:
            values = obs[key]
            mask = highlight.get(key) if highlight else None
            if mask is not None:
                values = values.astype(str).where(mask, "unknown")
            colors, palette, categories = assign_colors(
                values, skip_unknown=skip_unknown or mask is not None
            )
            _scatter(ax, coords, modality, colors=colors)
            _style_axes(ax, labels.get(key, key), axis_labels)
            if palette is not None:
                _category_legend(ax, palette, categories, labels.get(key, key))
            if modality is not None and len(np.unique(modality)) > 1:
                ax.legend(handles=_modality_handles(modality, colored=False), title=modality_label,
                          fontsize=7, title_fontsize=7, loc="lower left", frameon=True, framealpha=0.85)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    from matplotlib.legend import Legend

    path.parent.mkdir(parents=True, exist_ok=True)
    # Legends pinned with add_artist sit outside the axes and are not part of the
    # tight bbox by default; without this the last panel's legend is cut off.
    extra = [c for ax in fig.axes for c in ax.get_children() if isinstance(c, Legend)]
    if fig._suptitle is not None:
        extra.append(fig._suptitle)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  wrote {path}")


def save_coords(
    obs: pd.DataFrame,
    coords: np.ndarray,
    path: Path,
    columns: tuple[str, str] = ("umap_1", "umap_2"),
) -> Path:
    """Coordinates plus obs, so a figure can be restyled without refitting the UMAP."""
    table = obs.copy()
    table[columns[0]] = coords[:, 0]
    table[columns[1]] = coords[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(path)
    except (ImportError, ValueError):
        path = path.with_suffix(".csv.gz")
        table.to_csv(path)
    print(f"  wrote {path}")
    return path
