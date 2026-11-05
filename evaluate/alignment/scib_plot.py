"""The classical scIB results table, drawn from a saved ``get_results`` CSV.

The figure is ``scib_metrics``' ``Benchmarker.plot_results_table`` (0.6.2) redrawn from
``scib.csv`` / ``scib_minmax.csv``: one row per model sorted by the total score, the
metrics as colored circles grouped by metric type, the aggregate scores as bars. Drawing
from the CSV instead of the ``Benchmarker`` lets the row names be chosen freely
(``--row-labels``) and the figure be redrawn without rerunning the benchmark
(``scib_benchmark.py --plot-only``). Needs ``plottable`` (a scib_metrics dependency),
not scib_metrics itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRIC_TYPE = "Metric Type"
AGGREGATE_SCORE = "Aggregate score"
# (results CSV, figure stem): raw values, as Benchmarker draws by default, and min-max
# scaled per metric across the rows (the aggregates recomputed from the scaled metrics).
TABLES = (("scib.csv", "scib_results"), ("scib_minmax.csv", "scib_results_minmax"))


def read_results(path: Path) -> pd.DataFrame:
    """A ``get_results`` CSV: one row per model plus the ``Metric Type`` row."""
    df = pd.read_csv(path, index_col=0)
    if METRIC_TYPE not in df.index:
        raise ValueError(f"{path} has no '{METRIC_TYPE}' row; is it a Benchmarker.get_results CSV?")
    df.index = df.index.astype(str)
    return df


def relabel(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Rename the model rows by ``labels`` (model -> display name); others keep their name."""
    rows = [r for r in df.index if r != METRIC_TYPE]
    unknown = sorted(set(labels) - set(rows))
    if unknown:
        print(f"  [warn] --row-labels for {unknown} match no row (rows: {rows})")
    names = [labels.get(r, r) for r in rows]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"Row labels must be unique; {duplicated} would name several rows")
    if METRIC_TYPE in names:
        raise ValueError(f"'{METRIC_TYPE}' cannot be a row label")
    return df.rename(index=dict(zip(rows, names)))


def plot_scib_table(df: pd.DataFrame):
    """The ``plot_results_table`` figure of a ``get_results`` frame; returns the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.font import contrasting_font_color
    from plottable.plots import bar

    types = df.loc[METRIC_TYPE]
    plot_df = df.drop(METRIC_TYPE, axis=0).astype(np.float64)
    score_cols = [c for c in df.columns if types[c] == AGGREGATE_SCORE]
    other_cols = [c for c in df.columns if types[c] != AGGREGATE_SCORE]
    sort_col = next((c for c in ("Total", "Batch correction", "Bio conservation") if c in score_cols), None)
    if sort_col:
        plot_df = plot_df.sort_values(by=sort_col, ascending=False)
    plot_df["Method"] = plot_df.index

    def cmap_fn(col_data):
        return normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)

    # scib_metrics' 1.5 fits short embedding keys; custom labels can be longer.
    method_width = max(1.5, 0.11 * max(len(m) for m in plot_df.index))
    column_definitions = [ColumnDefinition("Method", width=method_width, textprops={"ha": "left", "weight": "bold"})]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(col, title=col.replace(" ", "\n", 1), width=1,
                         textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
                         cmap=cmap_fn(plot_df[col]), group=types[col], formatter="{:.2f}")
        for col in other_cols
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(col, width=1, title=col.replace(" ", "\n", 1), plot_fn=bar,
                         plot_kw={"cmap": mpl.cm.YlGnBu, "plot_bg_bar": False, "annotate": True,
                                  "height": 0.9, "formatter": "{:.2f}"},
                         group=types[col], border="left" if i == 0 else None)
        for i, col in enumerate(score_cols)
    ]
    # Text stays editable in the svg (as in scib_metrics).
    with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25 + method_width - 1.5, 3 + 0.3 * len(plot_df)))
        tab = Table(
            plot_df, cell_kw={"linewidth": 0, "edgecolor": "k"}, column_definitions=column_definitions,
            ax=ax, row_dividers=True, footer_divider=True, textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"}, index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
        # autoset_fontcolors skips the bar cells' labels; recolor them against their bar.
        for col in score_cols:
            for cell in tab.get_column(col).cells:
                color = contrasting_font_color(mpl.cm.YlGnBu(float(cell.content)))
                for text in cell.axes_inset.texts:
                    text.set_color(color)
    return fig


def draw_scib_figures(out_dir: Path, labels: dict[str, str], formats: list[str], dpi: int = 300) -> list[str]:
    """Draw every results table found in ``out_dir``; returns the written file names."""
    import matplotlib.pyplot as plt

    written = []
    for csv, stem in TABLES:
        path = out_dir / csv
        if not path.exists():
            continue
        fig = plot_scib_table(relabel(read_results(path), labels))
        for fmt in formats:
            target = out_dir / f"{stem}.{fmt}"
            fig.savefig(target, facecolor=fig.axes[0].get_facecolor(), dpi=dpi, bbox_inches="tight")
            written.append(target.name)
            print(f"  wrote {target}")
        plt.close(fig)
    return written
