"""Self-check for the benchmark figure layout.

Two things are pinned here:

* **Per-task figures** place at most ``PER_TASK_MAX_COLS`` metrics per row and
  centre a row that ends up short — a lone metric lands in the middle, not hard
  left and not stretched across the page.
* **The combined grid is unchanged** by that: one row per task, as wide as the
  widest task, short rows left-aligned so columns line up across tasks.

Also checks that the rotated model names actually fit inside their axes, which is
the part that silently regresses whenever a font size or figure height changes.

Needs matplotlib and numpy; no cluster, GPU, checkpoints or metric files.

    python evaluate/check/check_benchmark_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.plot import plot_ablation_benchmark as pab  # noqa: E402
from evaluate.plot.experiment_selection import grouped_layout  # noqa: E402

FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label
          + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILED.append(label)


def rows_of(cells) -> dict[int, list[int]]:
    """grid row -> sorted half-column offsets of its cells."""
    out: dict[int, list[int]] = {}
    for cell in cells:
        out.setdefault(cell.row, []).append(cell.col)
    return {r: sorted(cols) for r, cols in out.items()}


def check_wrapping() -> None:
    print("\n-- per-task wrapping (max 2 per row) --")
    for n, expected in (
        (1, {0: [1]}),                       # centred in a 2-column grid
        (2, {0: [0, 2]}),
        (3, {0: [0, 2], 1: [1]}),            # lone second row centred
        (4, {0: [0, 2], 1: [0, 2]}),
        (5, {0: [0, 2], 1: [0, 2], 2: [1]}),
    ):
        metrics = {"t": [f"m{i}" for i in range(n)]}
        cells, n_rows, n_cols = pab._plan_cells(["t"], metrics, pab.PER_TASK_MAX_COLS)
        check(f"{n} metric(s) -> {len(expected)} row(s), {n_cols} col(s)",
              rows_of(cells) == expected and n_rows == len(expected) and n_cols == 2,
              f"{rows_of(cells)} rows={n_rows} cols={n_cols}")

    # Centring means the lone cell's midpoint equals the midpoint of a full row.
    cells, _, n_cols = pab._plan_cells(
        ["t"], {"t": ["a", "b", "c"]}, pab.PER_TASK_MAX_COLS
    )
    lone = next(c for c in cells if c.row == 1)
    full = [c for c in cells if c.row == 0]
    check("centred row shares the full row's midpoint",
          lone.col + 1 == (min(c.col for c in full) + max(c.col for c in full) + 2) / 2)
    check("and stays inside the grid", 0 <= lone.col and lone.col + 2 <= 2 * n_cols)


def check_combined_unchanged() -> None:
    print("\n-- combined grid unchanged --")
    metrics = {"a": ["m1", "m2", "m3", "m4"], "b": ["z1", "z2"]}
    cells, n_rows, n_cols = pab._plan_cells(["a", "b"], metrics, None)
    check("one row per task", n_rows == 2)
    check("as wide as the widest task", n_cols == 4)
    check("short row left-aligned, not centred",
          rows_of(cells) == {0: [0, 2, 4, 6], 1: [0, 2]}, str(rows_of(cells)))

    # Group names: bottom-most cell per column, so a column whose last task ends
    # early is still labelled.
    labels = pab._group_label_cells(cells, n_rows, wrapped=False)
    check("labels on the bottom-most cell of each column",
          labels == {(1, 0), (1, 2), (0, 4), (0, 6)}, str(sorted(labels)))


def check_group_labels_wrapped() -> None:
    print("\n-- group labels when wrapped --")
    for n, expected in (
        (3, {(1, 1)}),               # only the centred bottom cell
        (4, {(1, 0), (1, 2)}),
        (2, {(0, 0), (0, 2)}),
    ):
        cells, n_rows, _ = pab._plan_cells(
            ["t"], {"t": [f"m{i}" for i in range(n)]}, pab.PER_TASK_MAX_COLS
        )
        labels = pab._group_label_cells(cells, n_rows, wrapped=True)
        check(f"{n} metric(s): bottom row only", labels == expected, str(sorted(labels)))


def render(n_metrics: int, n_models: int = 19, figsize=(24, 8), max_cols=2):
    names = [f"experiment/name_{i}" for i in range(n_models)]
    metrics = [f"metric_{i}" for i in range(n_metrics)]
    results = {n: {"t": {m: 0.5 + 0.02 * i for m in metrics}} for i, n in enumerate(names)}
    groups = [("Group one", names[: n_models // 2]), ("Group two", names[n_models // 2:])]
    x, colors, spans = grouped_layout(names, groups)
    return pab._render_figure(
        tasks=["t"], results=results, task_metrics={"t": metrics},
        primary={"t": metrics[0]}, model_names=names,
        x_positions=np.asarray(x, float), colors=colors, spans=spans,
        named_groups=True, grouped=True, figsize=figsize, title="t",
        bar_names=True, output=None, max_cols=max_cols,
    )


def check_names_fit() -> None:
    """Every rotated model name must end up inside its axes.

    The headroom is computed from the axes' measured height; getting the formula
    subtly wrong clips only the tallest bar's label, which is easy to miss.
    """
    print("\n-- rotated names fit inside the axes --")
    for n_metrics in (1, 2, 4):
        fig = render(n_metrics)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        worst = 0.0
        for ax in fig.axes:
            ax_top = ax.get_window_extent().y1
            for text in ax.texts:
                if text.get_rotation() != 90:
                    continue          # value labels are horizontal
                overshoot = text.get_window_extent(renderer).y1 - ax_top
                worst = max(worst, overshoot)
        check(f"{n_metrics} metric(s): no name overflows (worst {worst:.1f}px)",
              worst <= 1.0, f"{worst:.1f}px above the axes")
        matplotlib.pyplot.close(fig)


def check_centred_on_canvas() -> None:
    print("\n-- centred subplot is centred on the figure --")
    fig = render(1)
    ax = fig.axes[0]
    pos = ax.get_position()
    centre = (pos.x0 + pos.x1) / 2
    check("lone axes centred", abs(centre - 0.5) < 0.02, f"centre={centre:.3f}")
    check("and not full width", pos.width < 0.75, f"width={pos.width:.3f}")

    # The footnote must not hold the saved crop out to the figure's right edge.
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    crop_centre = (bbox.x0 + bbox.x1) / 2
    check("saved crop stays roughly centred",
          abs(crop_centre - fig.get_figwidth() / 2) < 0.06 * fig.get_figwidth(),
          f"crop centre {crop_centre:.2f} of {fig.get_figwidth():.2f}in")
    matplotlib.pyplot.close(fig)


def check_font_sizes() -> None:
    print("\n-- label sizes --")
    check("bar names larger than the value labels", pab.BAR_NAME_FONTSIZE > 6)
    check("group names larger than the old 7.5", pab.GROUP_NAME_FONTSIZE > 7.5)
    check("legend larger than the old 9", pab.LEGEND_FONTSIZE > 9)

    fig = render(2)
    sizes = {
        round(t.get_fontsize(), 1)
        for ax in fig.axes for t in ax.texts if t.get_rotation() == 90
    }
    check("bar names drawn at BAR_NAME_FONTSIZE",
          sizes == {float(pab.BAR_NAME_FONTSIZE)}, str(sizes))
    legend = fig.legends[0]
    check("legend drawn at LEGEND_FONTSIZE",
          round(legend.get_texts()[0].get_fontsize(), 1) == float(pab.LEGEND_FONTSIZE))
    matplotlib.pyplot.close(fig)


def main() -> None:
    check_wrapping()
    check_combined_unchanged()
    check_group_labels_wrapped()
    check_names_fit()
    check_centred_on_canvas()
    check_font_sizes()

    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
