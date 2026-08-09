"""
Ablation benchmark barplot.

Given an ablation experiment directory structured as:

    {ablation_dir}/
        {model_name}/
            metrics/
                results_{task_name}.json
                ...
        {model_name}/
            ...

For each task, every numeric metric gets its own subplot.  Within each
subplot all models are compared as bars.  The primary metric per task is
highlighted with a coloured background and a star in the title.

Layout:  rows = tasks,  columns = metrics within that task.

With ``--per-task`` each task is written to its own figure instead, named after
``--output`` with the task appended (``benchmark_deconv.png``). Colours, bar order
and group bands are computed once across all tasks and shared, so the separate
figures stay comparable with each other and with the combined one.

Every bar is labelled with its value and, unless ``--no-bar-names``, with the
model's display name — so a dense figure can be read without tracing colours back
to the legend.

Alternatively a YAML config selects individual runs — possibly from different
ablation directories — gives them display names, and optionally arranges them
into groups (see ``--config`` and :func:`load_config` for the schema).

Usage
-----
    python evaluate/plot/ablation_benchmark.py --ablation-dir path/to/ablation

    # One figure per task rather than a single grid:
    python evaluate/plot/ablation_benchmark.py --config comparison.yaml --per-task

    # Override which metric is "primary" for one or more tasks:
    python evaluate/plot/ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --primary canc_type_class=f1_weighted deconv=rmse

    # Save without showing:
    python evaluate/plot/ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --output benchmark.pdf --no-show

    # Compare hand-picked runs across ablations, grouped:
    python evaluate/plot/ablation_benchmark.py --config comparison.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluate.plot.experiment_selection import (  # noqa: E402
    GROUP_SHADES as _GROUP_SHADES,
    as_str_list as _as_metric_list,
    grouped_layout as _grouped_layout,
    is_model_dir,
    load_raw_config,
    parse_figsize,
    parse_groups,
)

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# Default primary metric per task (highlighted in the plot).
TASK_PRIMARY_METRIC: dict[str, str] = {
    "canc_type_class":  "accuracy",
    "deconv":           "mae",
    "survival":         "c_index",
    "proteome_pred":    "mean_pearson_r",
    "drug_sensitivity_v2": "mean_pearson_r",
}

# Metrics where lower is better — shown with a ↓ indicator.
LOWER_IS_BETTER: set[str] = {"mae", "mse", "rmse", "ibs", "d_calibration"}

# Informational / count metrics that are not meaningful to plot as bars.
SKIP_METRICS: set[str] = {
    "n_events",
    "n_folds_evaluated",
    "n_drugs_evaluated",
    "n_proteins_evaluated",
    "event_rate",
}

# Human-readable labels.
TASK_LABELS: dict[str, str] = {
    "canc_type_class":  "Cancer Type\nClassification",
    "deconv":           "Cell-Type\nDeconvolution",
    "survival":         "Survival\nPrediction",
    "proteome_pred":    "Proteome\nPrediction",
    "drug_sensitivity_v2": "Drug Sensitivity\nPrediction",
}

METRIC_LABELS: dict[str, str] = {
    "accuracy":          "Accuracy",
    "f1_weighted":       "F1 (weighted)",
    "precision_macro":   "Precision (macro)",
    "recall_macro":      "Recall (macro)",
    "mae":               "MAE",
    "mse":               "MSE",
    "rmse":              "RMSE",
    "c_index":           "C-index",
    "mean_pearson_r":    "Mean Pearson r",
    "median_pearson_r":  "Median Pearson r",
    "mean_auroc":        "Mean AUROC",
}

# Background colour for the primary-metric subplot.
_PRIMARY_BG   ="#FFFDE7"   # very light yellow
_PRIMARY_EDGE = "#F9A825"   # amber border


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def collect_model_metrics(model_dir: Path) -> dict[str, dict[str, float]]:
    """
    Read one model directory and return ``{task_name: {metric: value, ...}}``.

    A model directory is recognised by having a ``metrics/`` subfolder
    containing at least one ``results_*.json`` file.
    """
    per_task: dict[str, dict[str, float]] = {}

    for jf in sorted((model_dir / "metrics").glob("results_*.json")):
        task_name = jf.stem[len("results_"):]
        if task_name in TASK_LABELS.keys():
            try:
                with open(jf) as fh:
                    data = json.load(fh)
                    # Deal with drug_sensitivity_v2
                    if task_name == "drug_sensitivity_v2":
                        if isinstance(data, dict) and "aggregate" in data:
                            data = data["aggregate"]
                        # Separate Cmax classification and IC50 regression
                        data_cmax = {
                            metric.replace("cmax_classification_mean_", "") : value
                            for metric, value in data.items()
                            if "cmax_classification_mean_" in metric and value > 0
                        }
                        data_ic50 = {
                            metric.replace("ic50_regression_mean_", "") : value
                            for metric, value in data.items()
                            if "ic50_regression_mean_" in metric and value > 0
                        }
                        per_task["Drug Sensitivity Prediction (Cmax Classification)"] = data_cmax
                        per_task["Drug Sensitivity Prediction (IC50 Regression)"] = data_ic50
                    else :
                        per_task[task_name] = data
            except Exception as exc:
                print(f"[warning] Could not read {jf}: {exc}", file=sys.stderr)

    return per_task


def collect_metrics(ablation_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Walk ablation_dir and return:
        results[model_name][task_name] = {metric: value, ...}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_dir in sorted(ablation_dir.iterdir()):
        if not model_dir.is_dir() or not is_model_dir(model_dir):
            continue
        results[model_dir.name] = collect_model_metrics(model_dir)

    return results


# --------------------------------------------------------------------------- #
# Task-name resolution
# --------------------------------------------------------------------------- #

# collect_model_metrics() splits results_drug_sensitivity_v2.json into two tasks
# whose keys are display strings; give them short handles.
TASK_ALIASES: dict[str, str] = {
    "cmax":                     "Drug Sensitivity Prediction (Cmax Classification)",
    "drug_cmax":                "Drug Sensitivity Prediction (Cmax Classification)",
    "drug_sensitivity_v2_cmax": "Drug Sensitivity Prediction (Cmax Classification)",
    "ic50":                     "Drug Sensitivity Prediction (IC50 Regression)",
    "drug_ic50":                "Drug Sensitivity Prediction (IC50 Regression)",
    "drug_sensitivity_v2_ic50": "Drug Sensitivity Prediction (IC50 Regression)",
}


def _norm(name: str) -> str:
    """Lowercase and collapse non-alphanumerics, for forgiving name matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def resolve_task(user_key: str, available: list[str]) -> str:
    """
    Map a task name typed by the user onto a task key present in *available*.

    Resolution order: exact → alias → normalised → unique substring.
    Exits with a helpful message if the key is unknown or ambiguous.
    """
    if user_key in available:
        return user_key

    aliased = TASK_ALIASES.get(_norm(user_key))
    if aliased and aliased in available:
        return aliased

    norm_map: dict[str, list[str]] = {}
    for task in available:
        norm_map.setdefault(_norm(task), []).append(task)

    key = _norm(user_key)
    if key in norm_map and len(norm_map[key]) == 1:
        return norm_map[key][0]

    partial = [t for t in available if key in _norm(t)]
    if len(partial) == 1:
        return partial[0]

    if len(partial) > 1:
        print(
            f"ERROR: task '{user_key}' is ambiguous — matches: {partial}",
            file=sys.stderr,
        )
    else:
        print(
            f"ERROR: unknown task '{user_key}'. Available tasks:\n  "
            + "\n  ".join(available),
            file=sys.stderr,
        )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Config-driven selection
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkConfig:
    """A parsed comparison config (see :func:`load_config` for the schema)."""

    # (group_name | None, [(display_name, model_dir), ...]) in plot order.
    groups: list[tuple[str | None, list[tuple[str, Path]]]]
    title: str = "Ablation Benchmark"
    output: Path | None = None
    figsize: tuple[float, float] | None = None
    metrics: dict[str, list[str]] = field(default_factory=dict)
    primary: dict[str, str] = field(default_factory=dict)
    # One figure per task instead of one combined grid; see plot_benchmark().
    per_task: bool = False
    # Only used when per_task is set. None -> derived from the single-row layout,
    # keeping `figsize`'s width when one was given.
    per_task_figsize: tuple[float, float] | None = None
    # Print each model's display name above its bar, in addition to the legend.
    bar_names: bool = True

    @property
    def grouped(self) -> bool:
        """True if the config declared named groups (vs a flat experiment list)."""
        return any(name is not None for name, _ in self.groups)


def load_config(path: Path) -> BenchmarkConfig:
    """
    Parse a YAML (or JSON) comparison config.

    Schema
    ------
    ::

        title:   "Unified vs. big-condition"     # optional figure title
        output:  figures/comparison.png          # optional; else <config>.png
        figsize: [22, 14]                        # optional, inches

        per_task: true            # one figure per task instead of one grid.
                                  # Each is written next to `output` with the task
                                  # appended: comparison_canc_type_class.png, ...
        per_task_figsize: [30, 8] # optional; only read when per_task is set.
                                  # Omitted -> derived from the single-row layout,
                                  # reusing figsize's width if one was given.
        bar_names: false          # default true: print each model's display name
                                  # above its bar as well as in the legend.

        metrics:                                 # optional metric subsets
          canc_type_class: [accuracy, f1_weighted]
          ic50: [pearson_rho, r2]                # 'cmax'/'ic50' aliases work
        primary:                                 # optional highlighted metric
          deconv: rmse

        # Either a flat list of runs …
        experiments:
          - name: "Baseline"
            path: /abs/path/to/ablation_a/baseline      # a model dir
          - name: "PCA"
            dir:  /abs/path/to/ablation_a               # an ablation dir …
            model: pca_baseline                         # … plus a model in it

        # … or groups of runs, possibly from different ablations:
        groups:
          - name: "Big condition"
            dir: /abs/path/to/ablation_big_condition
            experiments:
              - {name: Baseline, model: baseline}
              - {name: "No DAT", model: no_dat}
          - name: "United data"
            dir: /abs/path/to/ablation_united_data
            all_models: true            # add every model dir found there
            exclude: [pca_baseline]     # optional, applies to all_models

    ``name`` defaults to the model directory's name.  Display names must be
    unique across the whole config, since they label the bars.
    """
    raw = load_raw_config(path)
    groups = parse_groups(raw, path, is_model_dir)
    figsize = parse_figsize(raw, path)

    # parse_figsize reads the 'figsize' key by name, so borrow it for the per-task
    # variant by presenting that value under the same key.
    per_task_figsize = (
        parse_figsize({"figsize": raw["per_task_figsize"]}, path)
        if raw.get("per_task_figsize")
        else None
    )

    metrics = {
        str(task): _as_metric_list(value, f"metrics['{task}']")
        for task, value in (raw.get("metrics") or {}).items()
    }
    primary = {
        str(task): str(value) for task, value in (raw.get("primary") or {}).items()
    }

    return BenchmarkConfig(
        groups=groups,
        title=str(raw.get("title") or "Ablation Benchmark"),
        output=Path(raw["output"]).expanduser() if raw.get("output") else None,
        figsize=figsize,
        metrics=metrics,
        primary=primary,
        per_task=bool(raw.get("per_task", False)),
        per_task_figsize=per_task_figsize,
        bar_names=bool(raw.get("bar_names", True)),
    )


def collect_from_config(
    config: BenchmarkConfig,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    list[tuple[str | None, list[str]]],
]:
    """
    Load the metrics for every experiment named in *config*.

    Returns ``(results, groups)`` where ``groups`` is the plot-order list of
    ``(group_name, [experiment_name, ...])``.  Experiments whose directory holds
    no readable metrics are dropped with a warning.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    groups: list[tuple[str | None, list[str]]] = []

    for group_name, members in config.groups:
        names: list[str] = []
        for name, model_dir in members:
            if not is_model_dir(model_dir):
                print(
                    f"[warning] no metrics/results_*.json under {model_dir} "
                    f"— skipping '{name}'.",
                    file=sys.stderr,
                )
                continue
            per_task = collect_model_metrics(model_dir)
            if not per_task:
                print(
                    f"[warning] no recognised task results for '{name}' "
                    f"({model_dir}) — skipping.",
                    file=sys.stderr,
                )
                continue
            results[name] = per_task
            names.append(name)

        if names:
            groups.append((group_name, names))

    return results, groups


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #

def _build_task_metrics(
    results: dict[str, dict[str, dict[str, float]]],
    primary_overrides: dict[str, str],
    metric_subsets: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """
    Derive the set of plottable metrics for each task.

    Parameters
    ----------
    metric_subsets : optional task → explicit list of metrics to plot.  Tasks
        listed here keep exactly those metrics, in the given order; tasks that
        are absent fall back to the default behaviour.  Explicitly requested
        metrics bypass SKIP_METRICS.

    Returns
    -------
    all_tasks : sorted list of task names found in results.
    task_metrics : task → sorted list of metrics to plot
                   (numeric, not in SKIP_METRICS, union across all models).
    primary : task → name of the primary (highlighted) metric.
    """
    all_tasks = sorted({t for m in results.values() for t in m})
    metric_subsets = metric_subsets or {}

    task_metrics: dict[str, list[str]] = {}
    for task in all_tasks:
        seen: set[str] = set()
        seen_all: set[str] = set()   # same, but including SKIP_METRICS
        for model_data in results.values():
            for k, v in model_data.get(task, {}).items():
                if isinstance(v, (int, float)):
                    seen_all.add(k)
                    if k not in SKIP_METRICS:
                        seen.add(k)

        # Explicit subset: honour the requested metrics and their order.
        if task in metric_subsets:
            requested = metric_subsets[task]
            for missing in (m for m in requested if m not in seen_all):
                print(
                    f"[warning] metric '{missing}' not found for task '{task}' — ignoring.",
                    file=sys.stderr,
                )
            task_metrics[task] = [m for m in requested if m in seen_all]
            continue

        # Ensure the primary metric is the first in the plot
        primary_metric = primary_overrides.get(task) or TASK_PRIMARY_METRIC.get(task)
        ordered = []
        if primary_metric and primary_metric in seen:
            ordered.append(primary_metric)
        ordered.extend(m for m in sorted(seen) if m != primary_metric)
        task_metrics[task] = ordered

    # A task whose explicit subset resolves to nothing is dropped entirely
    # (an empty subset is how a caller removes a task row from the figure).
    all_tasks = [
        t for t in all_tasks if task_metrics[t] or t not in metric_subsets
    ]

    primary: dict[str, str] = {}
    for task in all_tasks:
        override = primary_overrides.get(task)
        default  = TASK_PRIMARY_METRIC.get(task)
        metrics  = task_metrics.get(task, [])

        if override and override in metrics:
            primary[task] = override
        elif default and default in metrics:
            primary[task] = default
        elif metrics:
            primary[task] = metrics[0]

    return all_tasks, task_metrics, primary


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def task_output_path(output: Path, task: str) -> Path:
    """``comparison.png`` + ``deconv`` -> ``comparison_deconv.png``.

    ``_norm`` is reused so the display-string tasks that
    ``collect_model_metrics`` invents for drug sensitivity ("Drug Sensitivity
    Prediction (IC50 Regression)") also produce a clean filename.
    """
    return output.with_name(f"{output.stem}_{_norm(task)}{output.suffix}")


def _legend_inches(n_models: int, ncol: int) -> float:
    """Vertical space the shared legend needs, in inches.

    Computed rather than hardcoded because the old fixed 6% reserve only happened
    to fit the combined figure: a per-task figure is a third of the height, so the
    same fraction is a third of the space for exactly as many legend rows.
    """
    rows = math.ceil(n_models / max(ncol, 1))
    return 0.23 * rows + 0.45          # row height + frame/title padding


def _apply_name_headroom(
    fig,
    axes_list: list,
    model_names: list[str],
    fontsize: float,
    padding: float,
) -> None:
    """Extend each axes' y-range so the rotated model names fit above the bars.

    The names sit at a fixed offset in *points*, so how much data-space they need
    depends on the axes' physical height — a fraction that works for a tall
    combined grid clips badly on a short per-task figure. Measure instead: convert
    the label's height in points into a fraction of the axes, per axes.
    """
    # Rotated 90 degrees, the label's vertical extent is its rendered *length*.
    # 0.62 * fontsize is a good average advance for DejaVu Sans, matplotlib's
    # default; erring high only leaves a little extra whitespace.
    longest = max((len(n) for n in model_names), default=0)
    needed_pt = padding + 0.62 * fontsize * longest + 4.0

    fig.canvas.draw()
    for ax in axes_list:
        ax_pt = ax.get_window_extent().height * 72.0 / fig.dpi
        if ax_pt <= 0:
            continue
        lo, hi = ax.get_ylim()
        # Cap the expansion: a pathologically short axes would otherwise blow the
        # scale up so far the bars vanish.
        frac = min(needed_pt / ax_pt, 1.2)
        ax.set_ylim(lo, hi + (hi - lo) * frac)


def _render_figure(
    tasks: list[str],
    results: dict[str, dict[str, dict[str, float]]],
    task_metrics: dict[str, list[str]],
    primary: dict[str, str],
    model_names: list[str],
    x_positions: np.ndarray,
    colors: list,
    spans: list[tuple[float, float, str | None]],
    named_groups: bool,
    grouped: bool,
    figsize: tuple[float, float] | None,
    title: str,
    bar_names: bool,
    output: Path | None,
):
    """Draw one figure covering *tasks* (rows) x their metrics (columns).

    Returns the Figure; the caller decides whether to save, show or close it.
    """
    n_models = len(model_names)
    n_tasks  = len(tasks)
    n_cols   = max((len(task_metrics[t]) for t in tasks), default=1)

    # Widen the figure in proportion to the gaps inserted between groups
    # (width_scale is exactly 1.0 when there are none).
    width_scale = float(x_positions.max() - x_positions.min() + 1) / n_models

    ncol_legend = min(n_models, 6)
    legend_in   = _legend_inches(n_models, ncol_legend)

    col_w, row_h = 2.6, 3.2
    fig_w = figsize[0] if figsize else max(n_cols * col_w * width_scale + 1.5, 6.0)
    fig_h = figsize[1] if figsize else max(n_tasks * row_h + legend_in + 0.8, 4.0)

    fig, axes = plt.subplots(n_tasks, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    bar_width = 0.75

    # Bottom-most visible subplot per column — where group names are printed.
    # (Tasks have different metric counts, so the grid is ragged.)
    last_visible_row: dict[int, int] = {}
    for row, task in enumerate(tasks):
        for col in range(len(task_metrics[task])):
            last_visible_row[col] = row

    # Axes that ended up with bars, so the rotated-name headroom can be applied to
    # exactly those, after layout, when their real pixel height is known.
    axes_with_bars: list = []

    for row, task in enumerate(tasks):
        metrics_for_task = task_metrics[task]
        primary_metric   = primary.get(task)

        for col in range(n_cols):
            ax = axes[row][col]

            # Hide unused columns for tasks with fewer metrics.
            if col >= len(metrics_for_task):
                ax.set_visible(False)
                continue

            metric     = metrics_for_task[col]
            is_primary = metric == primary_metric

            # Highlight primary metric subplot.
            if is_primary:
                ax.set_facecolor(_PRIMARY_BG)
                for spine in ax.spines.values():
                    spine.set_edgecolor(_PRIMARY_EDGE)
                    spine.set_linewidth(1.8)

            any_bar = False
            for model_idx, model_name in enumerate(model_names):
                value = results.get(model_name, {}).get(task, {}).get(metric)
                if value is None or not isinstance(value, (int, float)):
                    continue

                bar = ax.bar(
                    x_positions[model_idx], float(value),
                    width=bar_width,
                    color=colors[model_idx],
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.bar_label(bar, fmt="%.3f", padding=2, fontsize=6)
                if bar_names:
                    # Above the value label (padding is in points, so this clears
                    # it regardless of the data scale), rotated to fit dense grids.
                    ax.bar_label(
                        bar, labels=[model_name], padding=13, fontsize=6,
                        rotation=90, color="#333333",
                    )
                any_bar = True

            # Subplot title: metric name + direction + star for primary.
            direction    = " ↓" if metric in LOWER_IS_BETTER else " ↑"
            star         = " ★" if is_primary else ""
            metric_label = METRIC_LABELS.get(metric, metric)
            ax.set_title(
                metric_label + direction + star,
                fontsize=9,
                fontweight="bold" if is_primary else "normal",
                color=_PRIMARY_EDGE if is_primary else "black",
                pad=4,
            )

            ax.set_xticks([])
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

            if any_bar:
                axes_with_bars.append(ax)

            # Grouped mode: shaded band behind each group + a fixed x-range so
            # every subplot lines up with the group labels under the figure.
            if spans:
                ax.set_xlim(spans[0][0] - 0.7, spans[-1][1] + 0.7)
                for span_idx, (span_lo, span_hi, _) in enumerate(spans):
                    ax.axvspan(
                        span_lo - 0.45, span_hi + 0.45,
                        color=_GROUP_SHADES[span_idx % len(_GROUP_SHADES)],
                        alpha=0.18, zorder=0,
                    )

            if not any_bar:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", fontsize=9,
                )

            # Group names, printed under the last visible subplot of each column.
            if named_groups and last_visible_row.get(col) == row:
                ax_x0, ax_x1 = ax.get_xlim()
                ax_width = ax_x1 - ax_x0
                for span_lo, span_hi, group_name in spans:
                    if not group_name:
                        continue
                    centre = ((span_lo + span_hi) / 2 - ax_x0) / ax_width
                    ax.text(
                        centre, -0.02, group_name,
                        transform=ax.transAxes,
                        ha="center", va="top",
                        fontsize=7.5, color="dimgrey", style="italic",
                    )

        # Task label as the y-axis label of the first (leftmost) subplot.
        axes[row][0].set_ylabel(
            TASK_LABELS.get(task, task),
            fontsize=10,
            fontweight="bold",
            labelpad=8,
        )

    # Shared legend below the figure.
    legend_handles = [
        mpatches.Patch(color=colors[i], label=name)
        for i, name in enumerate(model_names)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol_legend,
        frameon=True,
        fontsize=9,
        title="Experiment" if grouped else "Model",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.text(
        0.99, 0.005,
        "★ = primary metric",
        ha="right", va="bottom", fontsize=8,
        color=_PRIMARY_EDGE, style="italic",
    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    # Reserve exactly the legend's height rather than a fixed 6%, so the same code
    # works for a tall combined grid and a short single-task figure.
    plt.tight_layout(rect=[0, min(legend_in / fig_h, 0.5), 1, 1])

    # After layout, so each axes' real height is known: the names are drawn at a
    # point offset and are invisible to the autoscaler, so the y-range has to be
    # widened by hand. Measuring beats guessing here — the same label needs a much
    # larger fraction of a short axes than of a tall one.
    if bar_names and axes_with_bars:
        _apply_name_headroom(fig, axes_with_bars, model_names, fontsize=6, padding=13)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved to {output}")

    return fig


def plot_benchmark(
    results: dict[str, dict[str, dict[str, float]]],
    primary_overrides: dict[str, str],
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
    metric_subsets: dict[str, list[str]] | None = None,
    groups: list[tuple[str | None, list[str]]] | None = None,
    title: str = "Ablation Benchmark",
    per_task: bool = False,
    per_task_figsize: tuple[float, float] | None = None,
    bar_names: bool = True,
) -> list[Path]:
    """
    Generate the benchmark grid.

    Grid layout:  rows = tasks,  columns = metrics within each task.
    Within every subplot, one bar per model.
    The primary metric column is highlighted with a coloured background and ★.

    ``metric_subsets`` optionally restricts (and orders) the metrics plotted for
    individual tasks — see :func:`_build_task_metrics`.

    ``groups`` optionally arranges the bars into named groups: each group gets a
    base hue (lightened across its members), a gap and a shaded band, and its
    name is printed under the bottom row of subplots.

    ``per_task`` writes one figure per task instead of a single grid, each named
    after ``output`` with the task appended. Colours, ordering and group bands are
    computed once and shared, so the per-task figures stay comparable with each
    other and with the combined one.

    ``bar_names`` prints each model's display name above its bar as well as in the
    legend.

    Returns the list of paths written (empty when ``output`` is None).
    """
    if not results:
        print("No results found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    all_tasks, task_metrics, primary = _build_task_metrics(
        results, primary_overrides, metric_subsets
    )
    model_names = list(results.keys())
    n_models    = len(model_names)

    if not all_tasks:
        print("No tasks found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Colour palette — one fixed colour per model, shared across all subplots.
    # A single unnamed group carries no grouping information, so it is drawn
    # like the ungrouped case (distinct colour per bar) rather than as shades
    # of one hue.
    spans: list[tuple[float, float, str | None]] = []
    use_groups = bool(groups) and (
        len(groups) > 1 or any(name for name, _ in groups)
    )
    if use_groups:
        x_list, colors, spans = _grouped_layout(model_names, groups)
        x_positions = np.asarray(x_list, dtype=float)
        named_groups = any(name for _, _, name in spans)
    else:
        cmap   = plt.get_cmap("tab10") if n_models <= 10 else plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(n_models)]
        x_positions = np.arange(n_models, dtype=float)
        named_groups = False

    common = dict(
        results=results,
        task_metrics=task_metrics,
        primary=primary,
        model_names=model_names,
        x_positions=x_positions,
        colors=colors,
        spans=spans,
        named_groups=named_groups,
        grouped=bool(groups),
        bar_names=bar_names,
    )

    written: list[Path] = []

    if per_task:
        # Height is derived per figure unless the caller pinned it: a combined
        # figsize sized for N task rows is N times too tall for one row. The width
        # is kept, since it is set by the number of bars, which does not change.
        task_size = per_task_figsize or (
            (figsize[0], _single_row_height(n_models)) if figsize else None
        )
        for task in all_tasks:
            out = task_output_path(output, task) if output is not None else None
            fig = _render_figure(
                tasks=[task],
                figsize=task_size,
                title=f"{title} — {TASK_LABELS.get(task, task)}".replace("\n", " "),
                output=out,
                **common,
            )
            if out is not None:
                written.append(out)
            if not show:
                plt.close(fig)
    else:
        fig = _render_figure(
            tasks=all_tasks, figsize=figsize, title=title, output=output, **common
        )
        if output is not None:
            written.append(output)
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return written


def _single_row_height(n_models: int) -> float:
    """Figure height for a one-task figure: the row plus the legend it needs."""
    return 3.2 + _legend_inches(n_models, min(n_models, 6)) + 0.8


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation benchmark: one subplot per (task, metric), models as bars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ablation-dir", "-d",
        type=Path,
        help="Path to the ablation experiment directory (plots every model in it).",
    )
    source.add_argument(
        "--config", "-c",
        type=Path,
        help=(
            "YAML config selecting individual runs — possibly from different "
            "ablation directories — with display names and optional groups. "
            "See load_config() for the schema."
        ),
    )
    parser.add_argument(
        "--primary", "-p",
        nargs="*",
        default=[],
        metavar="TASK=METRIC",
        help=(
            "Override the highlighted primary metric for one or more tasks. "
            "Format: task_name=metric_name  (e.g. deconv=rmse canc_type_class=f1_weighted)."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save path for the figure (e.g. benchmark.pdf). Defaults to "
             "{ablation_dir}/benchmark.png, or for --config the config's "
             "'output:' key, else the config path with a .png suffix.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=None,
        help="Figure width and height in inches (auto if omitted).",
    )
    parser.add_argument(
        "--per-task",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write one figure per downstream task instead of a single combined "
            "grid. Each is saved next to --output with the task name appended "
            "(benchmark_deconv.png, ...). Overrides the config's 'per_task' key."
        ),
    )
    parser.add_argument(
        "--per-task-figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=None,
        help=(
            "Size of each per-task figure in inches. Omitted: the width comes from "
            "--figsize and the height is derived from the single-row layout."
        ),
    )
    parser.add_argument(
        "--bar-names",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Print each model's display name above its bar as well as in the "
            "legend (default: on). Overrides the config's 'bar_names' key."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    groups: list[tuple[str | None, list[str]]] | None = None
    metric_subsets: dict[str, list[str]] = {}
    primary_overrides: dict[str, str] = {}
    title = "Ablation Benchmark"
    figsize = tuple(args.figsize) if args.figsize else None
    # CLI wins over the config; None means "not specified on the CLI".
    per_task = bool(args.per_task) if args.per_task is not None else False
    per_task_figsize = tuple(args.per_task_figsize) if args.per_task_figsize else None
    bar_names = bool(args.bar_names) if args.bar_names is not None else True

    if args.config is not None:
        # ---- Config-driven: hand-picked runs, possibly across ablations ----
        config_path: Path = args.config.expanduser().resolve()
        if not config_path.is_file():
            print(f"ERROR: --config does not exist: {config_path}", file=sys.stderr)
            sys.exit(1)

        config = load_config(config_path)
        title  = config.title
        figsize = figsize or config.figsize
        per_task = args.per_task if args.per_task is not None else config.per_task
        per_task_figsize = per_task_figsize or config.per_task_figsize
        bar_names = args.bar_names if args.bar_names is not None else config.bar_names

        print(f"Loading experiments from {config_path} ...")
        results, groups = collect_from_config(config)
        if not results:
            print("No experiments with metrics found.", file=sys.stderr)
            sys.exit(1)

        available_tasks = sorted({t for m in results.values() for t in m})
        metric_subsets = {
            resolve_task(task, available_tasks): metrics
            for task, metrics in config.metrics.items()
        }
        primary_overrides = {
            resolve_task(task, available_tasks): metric
            for task, metric in config.primary.items()
        }

        output: Path | None = (
            args.output or config.output or config_path.with_suffix(".png")
        )

        for group_name, names in groups:
            label = group_name or "(ungrouped)"
            print(f"  {label}: {', '.join(names)}")
    else:
        # ---- Directory-driven: every model in one ablation dir ----
        ablation_dir: Path = args.ablation_dir.expanduser().resolve()
        if not ablation_dir.is_dir():
            print(
                f"ERROR: --ablation-dir does not exist: {ablation_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        output = args.output or (ablation_dir / "benchmark.png")

        print(f"Scanning {ablation_dir} ...")
        results = collect_metrics(ablation_dir)

        if not results:
            print(
                "No model metric directories found under the ablation dir.",
                file=sys.stderr,
            )
            sys.exit(1)

    for item in args.primary or []:
        if "=" not in item:
            print(f"WARNING: ignoring malformed --primary entry '{item}' (expected task=metric)")
            continue
        task, metric = item.split("=", 1)
        primary_overrides[task.strip()] = metric.strip()

    n_models = len(results)
    n_tasks  = len({t for m in results.values() for t in m})
    print(f"Found {n_models} model(s) and {n_tasks} task(s).")

    plot_benchmark(
        results=results,
        primary_overrides=primary_overrides,
        output=output,
        show=not args.no_show,
        figsize=figsize,
        metric_subsets=metric_subsets or None,
        groups=groups,
        title=title,
        per_task=per_task,
        per_task_figsize=per_task_figsize,
        bar_names=bar_names,
    )


if __name__ == "__main__":
    main()
