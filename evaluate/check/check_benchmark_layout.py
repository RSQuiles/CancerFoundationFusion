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
    for n, expected, expected_cols in (
        (1, {0: [0]}, 1),                       # single column, fills its figure
        (2, {0: [0, 2]}, 2),
        (3, {0: [0, 2], 1: [1]}, 2),            # lone second row centred
        (4, {0: [0, 2], 1: [0, 2]}, 2),
        (5, {0: [0, 2], 1: [0, 2], 2: [1]}, 2),
    ):
        metrics = {"t": [f"m{i}" for i in range(n)]}
        cells, n_rows, n_cols = pab._plan_cells(["t"], metrics, pab.PER_TASK_MAX_COLS)
        check(f"{n} metric(s) -> {len(expected)} row(s), {expected_cols} col(s)",
              rows_of(cells) == expected and n_rows == len(expected)
              and n_cols == expected_cols,
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
        (1, {(0, 0)}),
    ):
        cells, n_rows, _ = pab._plan_cells(
            ["t"], {"t": [f"m{i}" for i in range(n)]}, pab.PER_TASK_MAX_COLS
        )
        labels = pab._group_label_cells(cells, n_rows, wrapped=True)
        check(f"{n} metric(s): bottom row only", labels == expected, str(sorted(labels)))


def render(n_metrics: int, n_models: int = 19, figsize=None, max_cols=2, n_groups=5):
    """One per-task figure, auto-sized unless *figsize* is given.

    Several groups with names of realistic length, since a narrow group with a long
    name is exactly the case the group-label fitting has to handle.
    """
    names = [f"experiment/name_{i}" for i in range(n_models)]
    metrics = [f"metric_{i}" for i in range(n_metrics)]
    results = {n: {"t": {m: 0.5 + 0.02 * i for m in metrics}} for i, n in enumerate(names)}
    labels = ["Base comparison", "Paired alignment objectives", "Paired mix",
              "Big condition", "Baseline"]
    per = max(n_models // n_groups, 1)
    groups = [
        (labels[g % len(labels)], names[g * per: (g + 1) * per])
        for g in range(n_groups)
    ]
    groups = [(label, members) for label, members in groups if members]
    assigned = sum(len(m) for _, m in groups)
    if assigned < n_models:                     # remainder joins the last group
        groups[-1] = (groups[-1][0], groups[-1][1] + names[assigned:])
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
    """A short row sits in the middle, and the saved crop keeps it there."""
    print("\n-- centred subplot is centred on the figure --")
    fig = render(3)
    # Bottom row holds one subplot; it must be centred on the pair above it.
    by_row = sorted(fig.axes, key=lambda a: -a.get_position().y0)
    top = [a.get_position() for a in by_row[:2]]
    lone = by_row[2].get_position()
    check("lone bottom axes centred on the row above",
          abs((lone.x0 + lone.x1) / 2 - (min(p.x0 for p in top) + max(p.x1 for p in top)) / 2)
          < 0.01)
    check("and the same width as one of them",
          abs(lone.width - top[0].width) < 0.01,
          f"{lone.width:.3f} vs {top[0].width:.3f}")

    # The footnote must not hold the saved crop out to the figure's right edge.
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    crop_centre = (bbox.x0 + bbox.x1) / 2
    check("saved crop stays roughly centred",
          abs(crop_centre - fig.get_figwidth() / 2) < 0.06 * fig.get_figwidth(),
          f"crop centre {crop_centre:.2f} of {fig.get_figwidth():.2f}in")
    matplotlib.pyplot.close(fig)


def check_auto_size() -> None:
    """Figures must be sized from their content, not from a hardcoded default."""
    print("\n-- auto sizing --")
    narrow = pab._figure_width(2, 24, 1.2, bar_names=True)
    wide   = pab._figure_width(2, 48, 1.2, bar_names=True)
    check("width grows with the number of bars", wide > narrow,
          f"{narrow:.1f} vs {wide:.1f}")
    # Proportionally, once past the MIN_COL_INCHES floor that a handful of bars hits.
    check("twice the bars, twice the plotting width",
          abs((wide - 1.5) - 2 * (narrow - 1.5)) < 0.01,
          f"{wide - 1.5:.2f} vs 2 x {narrow - 1.5:.2f}")
    check("a handful of bars still gets a usable width",
          pab._figure_width(2, 3, 1.0, bar_names=True) >= 6.0)
    check("dropping bar names narrows it a lot",
          pab._figure_width(2, 48, 1.2, bar_names=False) < 0.6 * wide)
    check("fewer metric columns -> proportionally narrower",
          abs(pab._figure_width(4, 48, 1.2, True) - 1.5
              - 2 * (pab._figure_width(2, 48, 1.2, True) - 1.5)) < 0.01)
    check("height grows with the number of rows",
          pab._figure_height(2, 48) > pab._figure_height(1, 48))

    # The point of the exercise: a document-shaped figure, not a 4:1 ribbon.
    for n_metrics, limit in ((4, 2.0), (2, 3.0)):
        fig = render(n_metrics, n_models=48)
        aspect = fig.get_figwidth() / fig.get_figheight()
        check(f"{n_metrics} metrics, 48 models: aspect {aspect:.2f} <= {limit}",
              aspect <= limit,
              f"{fig.get_figwidth():.1f} x {fig.get_figheight():.1f}")
        matplotlib.pyplot.close(fig)


def check_labels_do_not_collide() -> None:
    """Adjacent bars' labels must stay clear at the auto width.

    This is what pins BAR_SLOT_INCHES and the value-label rotation: narrowing the
    figure any further merges the numbers of neighbouring bars into one string.
    """
    print("\n-- labels of adjacent bars --")
    selectors = {
        "values": lambda t: round(t.get_fontsize(), 1) == 6.0,
        "names":  lambda t: round(t.get_fontsize(), 1) == float(pab.BAR_NAME_FONTSIZE),
        # Group names are the italic ones, under the bottom row.
        "groups": lambda t: t.get_style() == "italic",
    }
    for n_models in (6, 20, 48):
        fig = render(4, n_models=n_models)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for what, keep in selectors.items():
            worst = 1.0
            # Per axes: labels in different subplots share x ranges and would
            # register as false overlaps.
            for ax in fig.axes:
                texts = sorted(
                    (t for t in ax.texts if keep(t)),
                    key=lambda t: t.get_window_extent(renderer).x0,
                )
                for a, b in zip(texts, texts[1:]):
                    gap = b.get_window_extent(renderer).x0 - a.get_window_extent(renderer).x1
                    worst = min(worst, gap)
            check(f"{n_models:>2} models: {what} clear of each other",
                  worst >= 0, f"{worst:.1f}px overlap")
        matplotlib.pyplot.close(fig)


def check_font_sizes() -> None:
    print("\n-- label sizes --")
    check("bar names larger than the value labels", pab.BAR_NAME_FONTSIZE > 6)
    check("group names larger than the old 7.5", pab.GROUP_NAME_FONTSIZE > 7.5)
    check("legend larger than the old 9", pab.LEGEND_FONTSIZE > 9)

    fig = render(2)
    # Select by size, not rotation: in a dense figure the value labels are turned
    # 90 degrees too, so rotation no longer distinguishes them.
    names = [t for ax in fig.axes for t in ax.texts
             if round(t.get_fontsize(), 1) == float(pab.BAR_NAME_FONTSIZE)]
    check("bar names drawn at BAR_NAME_FONTSIZE", bool(names))
    check("and rotated", {t.get_rotation() for t in names} == {90.0})
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
    check_auto_size()
    check_labels_do_not_collide()
    check_font_sizes()

    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
