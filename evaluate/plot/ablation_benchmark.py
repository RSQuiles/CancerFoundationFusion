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

Usage
-----
    python evaluate/plot/ablation_benchmark.py --ablation-dir path/to/ablation

    # Override which metric is "primary" for one or more tasks:
    python evaluate/plot/ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --primary canc_type_class=f1_weighted deconv=rmse

    # Save without showing:
    python evaluate/plot/ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --output benchmark.pdf --no-show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# Default primary metric per task (highlighted in the plot).
TASK_PRIMARY_METRIC: dict[str, str] = {
    "canc_type_class":  "accuracy",
    "deconv":           "mae",
    "survival":         "c_index",
    "proteome_pred":    "mean_pearson_r",
    "drug_sensitivity": "mean_pearson_r",
}

# Metrics where lower is better — shown with a ↓ indicator.
LOWER_IS_BETTER: set[str] = {"mae", "mse", "rmse"}

# Informational / count metrics that are not meaningful to plot as bars.
SKIP_METRICS: set[str] = {
    "n_events",
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
    "drug_sensitivity": "Drug Sensitivity\nPrediction",
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
}

# Background colour for the primary-metric subplot.
_PRIMARY_BG   ="#FFFDE7"   # very light yellow
_PRIMARY_EDGE = "#F9A825"   # amber border


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def collect_metrics(ablation_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Walk ablation_dir and return:
        results[model_name][task_name] = {metric: value, ...}

    A model directory is recognised by having a ``metrics/`` subfolder
    containing at least one ``results_*.json`` file.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_dir in sorted(ablation_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        metrics_dir = model_dir / "metrics"
        if not metrics_dir.is_dir():
            continue

        json_files = sorted(metrics_dir.glob("results_*.json"))
        if not json_files:
            continue

        model_name = model_dir.name
        results[model_name] = {}

        for jf in json_files:
            task_name = jf.stem[len("results_"):]
            try:
                with open(jf) as fh:
                    results[model_name][task_name] = json.load(fh)
            except Exception as exc:
                print(f"[warning] Could not read {jf}: {exc}", file=sys.stderr)

    return results


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #

def _build_task_metrics(
    results: dict[str, dict[str, dict[str, float]]],
    primary_overrides: dict[str, str],
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """
    Derive the set of plottable metrics for each task.

    Returns
    -------
    all_tasks : sorted list of task names found in results.
    task_metrics : task → sorted list of metrics to plot
                   (numeric, not in SKIP_METRICS, union across all models).
    primary : task → name of the primary (highlighted) metric.
    """
    all_tasks = sorted({t for m in results.values() for t in m})

    task_metrics: dict[str, list[str]] = {}
    for task in all_tasks:
        seen: set[str] = set()
        for model_data in results.values():
            for k, v in model_data.get(task, {}).items():
                if k not in SKIP_METRICS and isinstance(v, (int, float)):
                    seen.add(k)
        task_metrics[task] = sorted(seen)

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

def plot_benchmark(
    results: dict[str, dict[str, dict[str, float]]],
    primary_overrides: dict[str, str],
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
) -> None:
    """
    Generate the benchmark grid.

    Grid layout:  rows = tasks,  columns = metrics within each task.
    Within every subplot, one bar per model.
    The primary metric column is highlighted with a coloured background and ★.
    """
    if not results:
        print("No results found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    all_tasks, task_metrics, primary = _build_task_metrics(results, primary_overrides)
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_tasks     = len(all_tasks)
    n_cols      = max((len(task_metrics[t]) for t in all_tasks), default=1)

    if n_tasks == 0:
        print("No tasks found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Colour palette — one fixed colour per model, shared across all subplots.
    cmap   = plt.get_cmap("tab10") if n_models <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_models)]

    col_w, row_h = 2.6, 3.2
    fig_w = figsize[0] if figsize else max(n_cols * col_w + 1.5, 6.0)
    fig_h = figsize[1] if figsize else max(n_tasks * row_h + 1.2, 4.0)

    fig, axes = plt.subplots(
        n_tasks, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    x_positions = np.arange(n_models, dtype=float)
    bar_width   = 0.75

    for row, task in enumerate(all_tasks):
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
                ax.bar_label(bar, fmt="%.3f", padding=3, fontsize=7)
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

            if not any_bar:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", fontsize=9,
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
        ncol=min(n_models, 6),
        frameon=True,
        fontsize=9,
        title="Model",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.text(
        0.99, 0.005,
        "★ = primary metric",
        ha="right", va="bottom", fontsize=8,
        color=_PRIMARY_EDGE, style="italic",
    )

    fig.suptitle("Ablation Benchmark", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved to {output}")

    if show:
        plt.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation benchmark: one subplot per (task, metric), models as bars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ablation-dir", "-d",
        required=True,
        type=Path,
        help="Path to the ablation experiment directory.",
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
        help="Save path for the figure (e.g. benchmark.pdf). "
             "Defaults to {ablation_dir}/benchmark.png.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ablation_dir: Path = args.ablation_dir.expanduser().resolve()
    if not ablation_dir.is_dir():
        print(f"ERROR: --ablation-dir does not exist: {ablation_dir}", file=sys.stderr)
        sys.exit(1)

    primary_overrides: dict[str, str] = {}
    for item in args.primary or []:
        if "=" not in item:
            print(f"WARNING: ignoring malformed --primary entry '{item}' (expected task=metric)")
            continue
        task, metric = item.split("=", 1)
        primary_overrides[task.strip()] = metric.strip()

    output: Path | None = args.output or (ablation_dir / "benchmark.png")

    print(f"Scanning {ablation_dir} ...")
    results = collect_metrics(ablation_dir)

    if not results:
        print("No model metric directories found under the ablation dir.", file=sys.stderr)
        sys.exit(1)

    n_models = len(results)
    n_tasks  = len({t for m in results.values() for t in m})
    print(f"Found {n_models} model(s) and {n_tasks} task(s).")
    for model, tasks in results.items():
        for task, metrics in tasks.items():
            print(f"  {model}/{task}: {metrics}")

    figsize = tuple(args.figsize) if args.figsize else None
    plot_benchmark(
        results=results,
        primary_overrides=primary_overrides,
        output=output,
        show=not args.no_show,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
