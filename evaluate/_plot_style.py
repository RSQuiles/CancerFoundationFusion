"""Shared look of the comparison figures (GeneICL benchmark, reconstruction).

One palette (the dataviz reference: a one-hue blue ramp for magnitude, recessive ink),
the matplotlib setup, saving in several formats, and a metric table that shades the best
model of each column. Import matplotlib lazily through ``pyplot()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
MISSING = "#f0efec"
BEST_FILL = "#cde2fb"
# Categorical slots in fixed order (dataviz reference palette, light mode): models in a
# line or box plot take them by position, never cycled; baselines use MUTED.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": HAIRLINE, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return plt


def save(fig, stem: Path, formats: list[str], dpi: int) -> list[str]:
    """Write ``fig`` as ``<stem>.<fmt>`` for each format, close it, return the file names."""
    import matplotlib.pyplot as plt

    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path.name)
        print(f"  wrote {path}")
    plt.close(fig)
    return written


def is_best(values: np.ndarray, best: str) -> np.ndarray:
    """Mask of the best finite value(s), ties included; ``best`` is ``max`` or ``min``."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=bool)
    target = np.nanmax(values) if best == "max" else np.nanmin(values)
    return finite & np.isclose(values, target, rtol=0, atol=1e-12)


@dataclass(frozen=True)
class Col:
    """One table column: ``key`` in the frame, its header, the group drawn above it,
    which value is best (``max``, ``min`` or None for a plain count) and the format."""

    key: str
    header: str
    group: str
    best: str | None = "max"
    fmt: str = "{:.3f}"


def draw_metric_table(frame: pd.DataFrame, labels: list[str], columns: list[Col], title: str,
                      footnote: str, label_header: str = "Model"):
    """A table of ``frame`` (one row per label), the best value of each column shaded and
    bold. Columns absent from ``frame`` or all-NaN are left out; groups keep their order."""
    from matplotlib.patches import Rectangle

    plt = pyplot()
    cols = [c for c in columns if c.key in frame.columns and frame[c.key].notna().any()]
    groups = list(dict.fromkeys(c.group for c in cols))

    row_h, col_w = 0.34, 0.95
    name_w = max(1.4, 0.075 * max(len(x) for x in labels) + 0.3)
    # each column fits its bold header plus the direction arrow
    widths = [name_w] + [0.6 if c.best is None else max(col_w, 0.085 * (len(c.header) + 2) + 0.2) for c in cols]
    xs = np.concatenate([[0.0], np.cumsum(widths)])
    total_w = xs[-1]
    n = len(frame)
    header_rows = 2
    total_h = row_h * (n + header_rows) + 0.15

    fig = plt.figure(figsize=(total_w + 0.2, total_h + 0.75))
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.78])
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    # group header over its columns, with a hairline under it
    for group in groups:
        idx = [i for i, c in enumerate(cols) if c.group == group]
        x0, x1 = xs[1 + idx[0]], xs[1 + idx[-1] + 1]
        ax.text((x0 + x1) / 2, row_h * 0.5, group, ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.plot([x0 + 0.08, x1 - 0.08], [row_h * 0.95] * 2, color=MUTED, linewidth=0.8)
    # column headers
    ax.text(0.08, row_h * 1.5, label_header, ha="left", va="center", fontsize=8.5, color=INK_2, fontweight="bold")
    for i, c in enumerate(cols):
        arrow = " ↓" if c.best == "min" else (" ↑" if c.best == "max" else "")
        ax.text((xs[1 + i] + xs[2 + i]) / 2, row_h * 1.5, c.header + arrow, ha="center", va="center",
                fontsize=8, color=INK_2, fontweight="bold")
    ax.plot([0, total_w], [row_h * 2] * 2, color=MUTED, linewidth=1.0)

    best_masks = {c.key: is_best(frame[c.key].to_numpy(float), c.best) for c in cols if c.best}
    for r in range(n):
        y = row_h * (header_rows + r)
        ax.text(0.08, y + row_h / 2, labels[r], ha="left", va="center", fontsize=8.5, color=INK)
        for i, c in enumerate(cols):
            v = frame[c.key].iloc[r]
            x0, x1 = xs[1 + i], xs[2 + i]
            best = bool(c.best and best_masks[c.key][r])
            if best:
                ax.add_patch(Rectangle((x0 + 0.03, y + 0.03), x1 - x0 - 0.06, row_h - 0.06,
                                       facecolor=BEST_FILL, edgecolor="none"))
            if pd.isna(v):
                text, color = "n/a", MUTED
            elif c.best is None:
                text, color = f"{int(v)}", INK_2
            else:
                text, color = c.fmt.format(v), INK
            ax.text((x0 + x1) / 2, y + row_h / 2, text, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if best else "normal", family="DejaVu Sans Mono")
        ax.plot([0, total_w], [y + row_h] * 2, color=HAIRLINE, linewidth=0.6)
    # separator between groups
    for group in groups[1:]:
        i0 = next(i for i, c in enumerate(cols) if c.group == group)
        ax.plot([xs[1 + i0]] * 2, [row_h * 0.2, total_h - 0.15], color=HAIRLINE, linewidth=0.8)

    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK)
    fig.text(0.02, 0.04, footnote, fontsize=7, color=MUTED, ha="left")
    return fig
