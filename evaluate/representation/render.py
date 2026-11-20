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

from evaluate.modalities import DISPLAY_NAMES, DRAW_ORDER

# Label values treated as "no value" when colouring.
UNKNOWN_LABELS: frozenset[str] = frozenset({"unknown", "nan", "none", "n/a", ""})
GREY = (0.75, 0.75, 0.75, 0.15)
LEGEND_ROWS_PER_COLUMN = 30  # a categorical legend wraps into a new column past this
PANEL_INCHES = 5.0            # width of one scatter panel
LEGEND_PAD_INCHES = 0.15      # gap between a panel and its legend column
Y_LABEL_INCHES = 0.4          # room for a panel's y-axis label
PANEL_HEIGHT_INCHES = 4.6     # height of the scatter axes

# Per-modality marker styling. Colour is used only when a panel is coloured by
# modality; other panels recolour per point and keep the marker shape.
MODALITY_COLORS: dict[str, str] = {
    "sc": "#2166ac",          # blue
    "bulk": "#d6604d",        # red
    "pseudobulk": "#4dac26",  # green
    "cell_mean": "black",
}
_CELL_MEAN = dict(color="black", s=70, marker="*", linewidths=0.5, edgecolors="black", alpha=1.0)
# Several modalities in one figure (plot_alignment): one shape per modality, sizes chosen
# so circle, square and triangle look about equally large.
JOINT_STYLE: dict[str, dict] = {
    "sc":         dict(color=MODALITY_COLORS["sc"], s=8, marker="o", linewidths=0),
    "bulk":       dict(color=MODALITY_COLORS["bulk"], s=7, marker="s", linewidths=0),
    "pseudobulk": dict(color=MODALITY_COLORS["pseudobulk"], s=12, marker="^", linewidths=0),
    "cell_mean":  _CELL_MEAN,
}
# One modality in the figure (plot_umap, plot_pca): circles; pseudobulks a little larger.
SINGLE_STYLE: dict[str, dict] = {
    "sc":         dict(color=MODALITY_COLORS["sc"], s=4, marker="o", linewidths=0),
    "bulk":       dict(color=MODALITY_COLORS["bulk"], s=6, marker="o", linewidths=0),
    "pseudobulk": dict(color=MODALITY_COLORS["pseudobulk"], s=10, marker="o", linewidths=0),
    "cell_mean":  _CELL_MEAN,
}
MODALITY_STYLE = JOINT_STYLE  # the name other modules import
_PLAIN_STYLE = dict(s=6, marker="o", linewidths=0)


def _style_for(modality: np.ndarray | None) -> dict[str, dict]:
    """``SINGLE_STYLE`` when the rows hold one data modality (cell means not counted)."""
    if modality is None:
        return JOINT_STYLE
    data = {m for m in np.unique(modality) if m != "cell_mean"}
    return SINGLE_STYLE if len(data) <= 1 else JOINT_STYLE


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

    tab10 for up to 9 categories, then tab20 -> tab20b -> tab20c (54 colours). Past
    that, the remaining categories get evenly spaced hues (``gist_ncar``, skipping its
    near-white and near-black ends), so no two categories of one legend share a colour.
    """
    if len(categories) <= 9:
        colors = [plt.get_cmap("tab10")(i) for i in range(10) if i != 7]
    else:
        colors = [plt.get_cmap("tab20")(i) for i in range(20) if i not in (14, 15)]
        colors += [plt.get_cmap(name)(i) for name in ("tab20b", "tab20c") for i in range(20)
                   if not (name == "tab20c" and i >= 16)]
    extra = len(categories) - len(colors)
    if extra > 0:
        cmap = plt.get_cmap("gist_ncar")
        colors += [cmap(x) for x in np.linspace(0.05, 0.9, extra)]
    return {c: colors[i] for i, c in enumerate(categories)}


def assign_colors(
    col: pd.Series,
    skip_unknown: bool = False,
    alpha: float = 0.6,
) -> tuple[np.ndarray, dict[str, tuple] | None, list[str] | None]:
    """RGBA per value; categories (or a viridis scale for >20 numeric values).

    Returns the colours, the category -> colour mapping and the category order
    (both None for continuous data). ``skip_unknown`` greys out unknown labels.
    """
    if _is_continuous(col):
        print(f"  [note] {col.name}: numeric with {col.nunique()} values, drawn as a continuous "
              f"scale (colour bar, no category legend)")
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
    styles = _style_for(modality)
    for mod in order or DRAW_ORDER:
        mask = modality == mod
        if not mask.any():
            continue
        style = dict(styles.get(mod, _PLAIN_STYLE))
        if colors is not None:
            style.pop("color", None)
            style["c"] = colors[mask]
        alpha = style.pop("alpha", 0.5)
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=alpha, rasterized=True, **style)


def _modality_handles(modality: np.ndarray, colored: bool) -> list:
    handles = []
    styles = _style_for(modality)
    for mod in DRAW_ORDER:
        if not (modality == mod).any():
            continue
        style = styles[mod]
        color = style["color"] if colored else "dimgrey"
        handles.append(mlines.Line2D(
            [], [], linestyle="None", markersize=9 if mod == "cell_mean" else 6,
            marker=style["marker"], color=color, markeredgecolor=color, label=DISPLAY_NAMES[mod],
        ))
    return handles


def legend_categories(values: pd.Series, categories: list[str], max_entries: int | None) -> tuple[list[str], int]:
    """The categories shown in the legend and how many are left out.

    Every category, unless ``max_entries`` caps it: then the most frequent ones (kept in
    their natural order) and the count of the rest, shown as one "+K more" line.
    """
    if not max_entries or len(categories) <= max_entries:
        return list(categories), 0
    counts = values.astype(str).value_counts()
    top = set(sorted(categories, key=lambda c: -int(counts.get(c, 0)))[:max_entries])
    return [c for c in categories if c in top], len(categories) - len(top)


def legend_columns(n_entries: int) -> int:
    return max(1, int(np.ceil(n_entries / LEGEND_ROWS_PER_COLUMN)))


def _category_legend(ax, palette, categories, title, n_more: int = 0) -> None:
    """The colour legend, always drawn: wraps into columns and shrinks its font with size.

    ``ax`` is the panel's legend column, so the layout reserves the legend's width.
    """
    handles = [mpatches.Patch(color=palette[c], label=c) for c in categories]
    if n_more:
        handles.append(mpatches.Patch(color=GREY[:3], label=f"+{n_more} more"))
    n = len(handles)
    legend = ax.legend(
        handles=handles, title=title, fontsize=7 if n <= 20 else 6 if n <= 60 else 5,
        title_fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=False,
        ncol=legend_columns(n), handlelength=1.0, labelspacing=0.3, columnspacing=0.8,
    )
    ax.add_artist(legend)


def _is_continuous(col: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
            and col.nunique() > 20)


COLORBAR_LABEL = "colorbar"  # label of a colour-bar axes, so callers can find it


def _colorbar(fig, lax, values: pd.Series, title: str) -> None:
    """A viridis colour bar in the panel's legend column."""
    import matplotlib.cm as cm

    vals = values.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = plt.Normalize(vmin, vmax) if vmin != vmax else plt.Normalize(0, 1)
    cax = lax.inset_axes([0.0, 0.1, 0.12, 0.8])
    cax.set_label(COLORBAR_LABEL)
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    bar.set_label(title, fontsize=8)
    bar.ax.tick_params(labelsize=7)


def _legend_column_inches(fig, lax, renderer) -> float:
    """Width the legend column needs: its legend or colour bar as drawn, else nothing."""
    from matplotlib.legend import Legend

    boxes = [c.get_window_extent(renderer) for c in lax.get_children() if isinstance(c, Legend)]
    boxes += [c.get_tightbbox(renderer) for c in lax.child_axes]
    if not boxes:
        return 0.05
    left = lax.get_window_extent(renderer).x0
    return max(b.x1 for b in boxes if b is not None) / fig.dpi - left / fig.dpi + LEGEND_PAD_INCHES


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
    max_legend_entries: int | None = None,
) -> plt.Figure:
    """One panel per colour key; marker shape encodes modality when ``modality_col`` is set.

    ``color_keys`` may include ``modality_col`` itself, which colours by the fixed
    modality palette. ``segments`` (each a (2, 2) array) are drawn faintly under the
    points of every panel. ``highlight`` maps a colour key to a boolean mask; rows
    outside it are drawn grey in that panel (e.g. unpaired rows on the pair panel).
    ``labels`` maps an obs column to the name shown in panel and legend titles
    (``{"tissue_general": "Tissue"}``); category values are drawn as they are.
    ``axis_labels`` names the two axes (``plot_pca`` passes ``"PC1 (12.3%)"`` and so on).
    A categorical panel always gets its legend, wrapped into columns, with the panel
    widened to fit; ``max_legend_entries`` caps it to the most frequent categories plus a
    "+K more" line. A numeric column with more than 20 values gets a colour bar instead.
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

    # Each panel is a scatter axes plus an axis-off legend column, sized after drawing to
    # the legend it holds: tight_layout does not count a legend placed outside its axes,
    # which is how a long (single-cell tissue) legend used to run into the next panel.
    fig = plt.figure(figsize=(len(keys) * (PANEL_INCHES + 2.0), 5.5))
    grid = fig.add_gridspec(1, 2 * len(keys), width_ratios=[PANEL_INCHES, 2.0] * len(keys), wspace=0.0)
    axes, legend_axes = [], []
    for i in range(len(keys)):
        axes.append(fig.add_subplot(grid[0, 2 * i]))
        lax = fig.add_subplot(grid[0, 2 * i + 1])
        lax.set_axis_off()
        legend_axes.append(lax)
    for ax, lax, key in zip(axes, legend_axes, keys):
        if segments:
            from matplotlib.collections import LineCollection

            ax.add_collection(LineCollection(segments, colors="black", linewidths=0.3, alpha=0.25, zorder=0))
        if key is None:
            _scatter(ax, coords, modality)
            _style_axes(ax, "embedding", axis_labels)
        elif modality is not None and key == modality_col:
            _scatter(ax, coords, modality)
            _style_axes(ax, modality_label, axis_labels)
            lax.legend(handles=_modality_handles(modality, colored=True), title=modality_label,
                       fontsize=8, title_fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.5),
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
                shown, n_more = legend_categories(values, categories, max_legend_entries)
                _category_legend(lax, palette, shown, labels.get(key, key), n_more)
            else:
                _colorbar(fig, lax, values, labels.get(key, key))
            if modality is not None and len(np.unique(modality)) > 1:
                ax.legend(handles=_modality_handles(modality, colored=False), title=modality_label,
                          fontsize=7, title_fontsize=7, loc="lower left", frameon=True, framealpha=0.85)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    # Lay the columns out in inches (no tight_layout, which would rescale them): each
    # legend column is as wide as what it holds, plus room for the next panel's y label.
    renderer = fig.canvas.get_renderer()
    legend_inches = [_legend_column_inches(fig, lax, renderer) for lax in legend_axes]
    widths = []
    for i, li in enumerate(legend_inches):
        widths += [PANEL_INCHES, li + (Y_LABEL_INCHES if i < len(keys) - 1 else 0.0)]
    left, right, bottom = Y_LABEL_INCHES, 0.1, 0.35
    top = 0.75 if title else 0.4
    width, height = left + sum(widths) + right, PANEL_HEIGHT_INCHES + top + bottom
    fig.set_size_inches(width, height)
    grid.set_width_ratios(widths)
    grid.update(left=left / width, right=1 - right / width, bottom=bottom / height,
                top=1 - top / height, wspace=0.0)
    if title:
        fig._suptitle.set_y(1 - 0.12 / height)
        fig._suptitle.set_va("top")
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    from matplotlib.legend import Legend

    path.parent.mkdir(parents=True, exist_ok=True)
    # Legends may reach past their axes and are not part of the tight bbox by default;
    # without this the last panel's legend could be cut off.
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
