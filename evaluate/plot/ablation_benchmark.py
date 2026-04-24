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

Collects all available task metrics, selects a primary metric per task, and
generates a grouped barplot comparing every model across every task found.

Usage
-----
    python evaluate/plot/ablation_benchmark.py --ablation-dir path/to/ablation

    # Override which metric to display per task:
    python evaluate/plot/ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --metrics canc_type_class=f1_weighted deconv=rmse

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

# Primary metric selected for each task when not overridden on the CLI.
# For "lower is better" metrics the bar label will show "(↓ lower is better)".
TASK_PRIMARY_METRIC: dict[str, str] = {
    "canc_type_class": "accuracy",
    "deconv":          "mae",
    "survival":        "c_index",
    "proteome_pred":   "mean_pearson_r",
    "drug_sensitivity": "mean_pearson_r",
}

# Tasks where a lower value is better (used only for axis annotation).
LOWER_IS_BETTER: set[str] = {"deconv", "mae", "mse", "rmse"}

# Human-readable labels for known tasks and metrics.
TASK_LABELS: dict[str, str] = {
    "canc_type_class":  "Cancer Type\nClassification",
    "deconv":           "Cell-Type\nDeconvolution",
    "survival":         "Survival\nPrediction",
    "proteome_pred":    "Proteome\nPrediction",
    "drug_sensitivity": "Drug Sensitivity\nPrediction",
}

METRIC_LABELS: dict[str, str] = {
    "accuracy":        "Accuracy",
    "f1_weighted":     "F1 (weighted)",
    "mae":             "MAE",
    "mse":             "MSE",
    "rmse":            "RMSE",
    "c_index":         "C-index",
    "mean_pearson_r":  "Mean Pearson r",
    "median_pearson_r": "Median Pearson r",
}


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def collect_metrics(ablation_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Walk ablation_dir and return a nested dict:
        results[model_name][task_name] = {metric: value, ...}

    A model subdirectory is recognised by the presence of a ``metrics/``
    subfolder containing at least one ``results_*.json`` file.
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
            # filename: results_{task_name}.json
            task_name = jf.stem[len("results_"):]
            try:
                with open(jf) as fh:
                    task_metrics = json.load(fh)
                results[model_name][task_name] = task_metrics
            except Exception as exc:
                print(f"[warning] Could not read {jf}: {exc}", file=sys.stderr)

    return results


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def _pick_primary(task: str, metrics: dict[str, float], override: str | None) -> tuple[str, float] | None:
    """Return (metric_name, value) for the primary metric of a task."""
    metric_name = override or TASK_PRIMARY_METRIC.get(task)
    if metric_name is None:
        # Task not in defaults: fall back to the first numeric key.
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_name = k
                break

    if metric_name is None:
        return None

    value = metrics.get(metric_name)
    if value is None or not isinstance(value, (int, float)):
        return None

    return metric_name, float(value)


def plot_benchmark(
    results: dict[str, dict[str, dict[str, float]]],
    metric_overrides: dict[str, str],
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
) -> None:
    """
    Generate the benchmark barplot.

    Layout: one subplot per task, bars grouped by model.
    All models share the same color palette (consistent across subplots).
    """
    if not results:
        print("No results found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Collect all tasks that appear in at least one model's metrics.
    all_tasks: list[str] = []
    for model_metrics in results.values():
        for task in model_metrics:
            if task not in all_tasks:
                all_tasks.append(task)
    all_tasks.sort()

    model_names = list(results.keys())
    n_tasks  = len(all_tasks)
    n_models = len(model_names)

    if n_tasks == 0:
        print("No tasks found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Colour palette — one colour per model.
    cmap = plt.get_cmap("tab10") if n_models <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_models)]

    fig_w, fig_h = figsize or (max(5 * n_tasks, 8), 5)
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(fig_w, fig_h),
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]  # shape (n_tasks,)

    bar_width  = 0.7 / max(n_models, 1)
    group_offsets = np.arange(n_models) * bar_width - (n_models - 1) * bar_width / 2

    for ax_idx, task in enumerate(all_tasks):
        ax = axes[ax_idx]
        override = metric_overrides.get(task)

        plotted_metric_name: str | None = None
        any_bar = False

        for model_idx, model_name in enumerate(model_names):
            task_metrics = results.get(model_name, {}).get(task)
            if task_metrics is None:
                continue

            picked = _pick_primary(task, task_metrics, override)
            if picked is None:
                continue

            metric_name, value = picked
            plotted_metric_name = metric_name

            x = group_offsets[model_idx]
            bar = ax.bar(
                x, value,
                width=bar_width * 0.9,
                color=colors[model_idx],
                label=model_name,
                zorder=3,
            )
            ax.bar_label(bar, fmt="%.3f", padding=3, fontsize=8)
            any_bar = True

        # Axis decoration
        task_lower_is_better = (
            task in LOWER_IS_BETTER
            or (plotted_metric_name is not None and plotted_metric_name in LOWER_IS_BETTER)
        )
        direction_note = " (↓)" if task_lower_is_better else " (↑)"

        metric_label = METRIC_LABELS.get(plotted_metric_name or "", plotted_metric_name or "value")
        ax.set_title(TASK_LABELS.get(task, task), fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(metric_label + direction_note, fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        if not any_bar:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes,
                color="grey", fontsize=10,
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
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Ablation Benchmark", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

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
        description="Plot ablation benchmark barplot from downstream task metrics.",
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
        "--metrics", "-m",
        nargs="*",
        default=[],
        metavar="TASK=METRIC",
        help=(
            "Override the primary metric for one or more tasks. "
            "Format: task_name=metric_name, e.g. deconv=rmse canc_type_class=f1_weighted."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save the figure (e.g. benchmark.pdf or benchmark.png). "
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
        help="Figure width and height in inches. Auto-determined if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ablation_dir: Path = args.ablation_dir.expanduser().resolve()
    if not ablation_dir.is_dir():
        print(f"ERROR: --ablation-dir does not exist: {ablation_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse metric overrides: ["task=metric", ...] → {"task": "metric"}
    metric_overrides: dict[str, str] = {}
    for item in args.metrics or []:
        if "=" not in item:
            print(f"WARNING: ignoring malformed --metrics entry '{item}' (expected task=metric)")
            continue
        task, metric = item.split("=", 1)
        metric_overrides[task.strip()] = metric.strip()

    output: Path | None = args.output
    if output is None:
        output = ablation_dir / "benchmark.png"

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
        metric_overrides=metric_overrides,
        output=output,
        show=not args.no_show,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
