"""
Re-generate the ablation benchmark plot from metrics that already exist on disk.

Nothing is recomputed: this script only reads the JSON files that
``run_ablation_downstream.py`` wrote for every model,

    {ablation_dir}/{model_name}/metrics/results_{task}.json

and re-draws the figure produced by ``evaluate/plot/plot_ablation_benchmark.py``.
Its purpose is to make that figure presentable: by default every numeric metric
of every task is plotted, which is too wide to read.  With ``--metrics`` you
choose, per task, exactly which metrics to show and in which order.  Tasks you
do not mention keep the original default behaviour (all numeric metrics, the
task's primary metric first).

Usage
-----
    # What is available in this ablation directory?
    python evaluate/finetune/replot_ablation_benchmark.py -d path/to/ablation --list

    # Same figure as the original script:
    python evaluate/finetune/replot_ablation_benchmark.py -d path/to/ablation

    # Subset the metrics of two tasks, leave the others at their defaults:
    python evaluate/finetune/replot_ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --metrics canc_type_class=accuracy,f1_weighted deconv=mae,mse \\
        --output figures/benchmark_thesis.pdf --no-show

    # Reusable subsets from a file, plus a fixed model order:
    python evaluate/finetune/replot_ablation_benchmark.py \\
        --ablation-dir path/to/ablation \\
        --metrics-file evaluate/finetune/configs/benchmark_metrics.yaml \\
        --models baseline no_dat pca_baseline

Notes
-----
* An empty subset (``--metrics survival=``) drops that task from the figure.
* Metrics normally hidden as bookkeeping (``n_events``, ``n_folds_evaluated``, …)
  are plotted if you ask for them explicitly.
* ``results_drug_sensitivity_v2.json`` is split by the collector into two tasks;
  refer to them as ``cmax`` and ``ic50`` (or any unambiguous substring).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.plot.plot_ablation_benchmark import (  # noqa: E402
    METRIC_LABELS,
    SKIP_METRICS,
    TASK_PRIMARY_METRIC,
    collect_metrics,
    plot_benchmark,
)

# --------------------------------------------------------------------------- #
# Task-name resolution
# --------------------------------------------------------------------------- #

# collect_metrics() splits results_drug_sensitivity_v2.json into two tasks whose
# keys are display strings; give them short handles.
TASK_ALIASES: dict[str, str] = {
    "cmax":                 "Drug Sensitivity Prediction (Cmax Classification)",
    "drug_cmax":            "Drug Sensitivity Prediction (Cmax Classification)",
    "drug_sensitivity_v2_cmax": "Drug Sensitivity Prediction (Cmax Classification)",
    "ic50":                 "Drug Sensitivity Prediction (IC50 Regression)",
    "drug_ic50":            "Drug Sensitivity Prediction (IC50 Regression)",
    "drug_sensitivity_v2_ic50": "Drug Sensitivity Prediction (IC50 Regression)",
}


def _norm(name: str) -> str:
    """Lowercase and collapse non-alphanumerics, for forgiving name matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def resolve_task(user_key: str, available: list[str]) -> str:
    """
    Map a task name typed on the CLI onto a task key present in *available*.

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
# Metric subset parsing
# --------------------------------------------------------------------------- #

def _parse_metric_args(entries: list[str]) -> dict[str, list[str]]:
    """Parse ``TASK=m1,m2`` CLI entries into {task: [metrics]} (order kept)."""
    subsets: dict[str, list[str]] = {}
    for item in entries:
        if "=" not in item:
            print(
                f"WARNING: ignoring malformed --metrics entry '{item}' "
                "(expected task=metric1,metric2)",
                file=sys.stderr,
            )
            continue
        task, metrics = item.split("=", 1)
        subsets[task.strip()] = [m.strip() for m in metrics.split(",") if m.strip()]
    return subsets


def _load_metrics_file(path: Path) -> dict[str, list[str]]:
    """Load a YAML or JSON file mapping task → list of metrics."""
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        print(
            f"ERROR: {path} must contain a mapping of task → list of metrics.",
            file=sys.stderr,
        )
        sys.exit(1)

    subsets: dict[str, list[str]] = {}
    for task, metrics in data.items():
        if metrics is None:
            subsets[str(task)] = []
        elif isinstance(metrics, str):
            subsets[str(task)] = [m.strip() for m in metrics.split(",") if m.strip()]
        elif isinstance(metrics, (list, tuple)):
            subsets[str(task)] = [str(m) for m in metrics]
        else:
            print(
                f"ERROR: metrics for task '{task}' in {path} must be a list or "
                "a comma-separated string.",
                file=sys.stderr,
            )
            sys.exit(1)
    return subsets


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #

def select_models(
    results: dict[str, dict[str, dict[str, float]]],
    models: list[str] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Filter and re-order *results* by the requested model names."""
    if not models:
        return results

    selected: dict[str, dict[str, dict[str, float]]] = {}
    for name in models:
        if name in results:
            selected[name] = results[name]
        else:
            print(
                f"WARNING: model '{name}' has no metrics in the ablation dir "
                f"— skipping. Available: {sorted(results)}",
                file=sys.stderr,
            )

    if not selected:
        print("ERROR: none of the requested models have metrics.", file=sys.stderr)
        sys.exit(1)
    return selected


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #

def list_available(results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print every task with its available metrics, for building subsets."""
    print(f"Models ({len(results)}): {', '.join(results)}\n")

    tasks = sorted({t for m in results.values() for t in m})
    for task in tasks:
        metrics: set[str] = set()
        for model_data in results.values():
            metrics.update(
                k for k, v in model_data.get(task, {}).items()
                if isinstance(v, (int, float))
            )
        primary = TASK_PRIMARY_METRIC.get(task)
        print(f"{task}")
        for metric in sorted(metrics):
            flags = []
            if metric == primary:
                flags.append("primary")
            if metric in SKIP_METRICS:
                flags.append("hidden by default")
            suffix = f"   [{', '.join(flags)}]" if flags else ""
            label = METRIC_LABELS.get(metric, "")
            label = f"  ({label})" if label else ""
            print(f"    {metric}{label}{suffix}")
        print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-plot the ablation benchmark from existing results_*.json files, "
            "optionally showing only a subset of metrics per task."
        ),
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
        action="extend",          # so repeated -m flags accumulate
        default=None,
        metavar="TASK=M1,M2",
        help=(
            "Metrics to plot for a task, in the given order "
            "(e.g. canc_type_class=accuracy,f1_weighted). Repeatable. "
            "Tasks not listed keep their default metrics; an empty list "
            "(task=) drops the task from the figure."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "YAML or JSON file mapping task → list of metrics. "
            "Entries given with --metrics take precedence."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only plot these models, in this order (default: all, alphabetical).",
    )
    parser.add_argument(
        "--primary", "-p",
        nargs="*",
        action="extend",
        default=None,
        metavar="TASK=METRIC",
        help="Override the highlighted primary metric for one or more tasks.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "Save path for the figure. Defaults to "
            "{ablation_dir}/benchmark_replot.png (so the full benchmark.png "
            "written by the downstream run is never overwritten)."
        ),
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
        "--list",
        action="store_true",
        help="Print the tasks and metrics found in the ablation dir, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ablation_dir: Path = args.ablation_dir.expanduser().resolve()
    if not ablation_dir.is_dir():
        print(f"ERROR: --ablation-dir does not exist: {ablation_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {ablation_dir} ...")
    results = collect_metrics(ablation_dir)
    if not results:
        print(
            "No model metric directories found under the ablation dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = select_models(results, args.models)

    if args.list:
        list_available(results)
        return

    available_tasks = sorted({t for m in results.values() for t in m})

    # File first, CLI second, so CLI entries win.
    raw_subsets: dict[str, list[str]] = {}
    if args.metrics_file is not None:
        metrics_file = args.metrics_file.expanduser().resolve()
        if not metrics_file.is_file():
            print(f"ERROR: --metrics-file not found: {metrics_file}", file=sys.stderr)
            sys.exit(1)
        raw_subsets.update(_load_metrics_file(metrics_file))
    raw_subsets.update(_parse_metric_args(args.metrics or []))

    metric_subsets = {
        resolve_task(task, available_tasks): metrics
        for task, metrics in raw_subsets.items()
    }

    primary_overrides: dict[str, str] = {}
    for item in args.primary or []:
        if "=" not in item:
            print(
                f"WARNING: ignoring malformed --primary entry '{item}' "
                "(expected task=metric)",
                file=sys.stderr,
            )
            continue
        task, metric = item.split("=", 1)
        primary_overrides[resolve_task(task.strip(), available_tasks)] = metric.strip()

    n_tasks = len(available_tasks)
    print(f"Found {len(results)} model(s) and {n_tasks} task(s).")
    if metric_subsets:
        for task, metrics in metric_subsets.items():
            shown = ", ".join(metrics) if metrics else "(dropped)"
            print(f"  {task}: {shown}")

    output: Path = args.output or (ablation_dir / "benchmark_replot.png")

    plot_benchmark(
        results=results,
        primary_overrides=primary_overrides,
        output=output,
        show=not args.no_show,
        figsize=tuple(args.figsize) if args.figsize else None,
        metric_subsets=metric_subsets,
    )


if __name__ == "__main__":
    main()
