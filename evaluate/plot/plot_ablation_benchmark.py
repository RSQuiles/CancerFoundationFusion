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

Alternatively a YAML config selects individual runs — possibly from different
ablation directories — gives them display names, and optionally arranges them
into groups (see ``--config`` and :func:`load_config` for the schema).

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

    # Compare hand-picked runs across ablations, grouped:
    python evaluate/plot/ablation_benchmark.py --config comparison.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Grouped mode: one base hue per group, lightened across its members.
_GROUP_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
_GROUP_GAP    = 1.2         # bar-width units inserted between groups
_GROUP_SHADES = ("lightsteelblue", "lightyellow")   # alternating band colours


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def is_model_dir(path: Path) -> bool:
    """True if *path* looks like a model directory (has metrics/results_*.json)."""
    metrics_dir = path / "metrics"
    return metrics_dir.is_dir() and any(metrics_dir.glob("results_*.json"))


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

    @property
    def grouped(self) -> bool:
        """True if the config declared named groups (vs a flat experiment list)."""
        return any(name is not None for name, _ in self.groups)


def _as_metric_list(value, where: str) -> list[str]:
    """Coerce a config metrics entry to a list of metric names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [m.strip() for m in value.split(",") if m.strip()]
    if isinstance(value, (list, tuple)):
        return [str(m) for m in value]
    sys.exit(f"ERROR: {where} must be a list or a comma-separated string.")


def _resolve_experiment(entry, group_dir: Path | None, where: str) -> tuple[str, Path]:
    """Turn one experiment entry into a (display_name, model_dir) pair."""
    if isinstance(entry, str):
        entry = {"model": entry}
    if not isinstance(entry, dict):
        sys.exit(f"ERROR: {where} must be a mapping or a model name.")

    path = entry.get("path")
    model = entry.get("model") or entry.get("name")
    base = entry.get("dir", group_dir)

    if path is not None:
        model_dir = Path(path).expanduser()
    elif model is not None and base is not None:
        model_dir = Path(base).expanduser() / str(model)
    else:
        sys.exit(
            f"ERROR: {where} needs either 'path', or 'model' plus a 'dir' "
            "(on the entry or its group)."
        )

    name = str(entry.get("name") or model_dir.name)
    return name, model_dir.resolve()


def _discover_group_models(
    group_dir: Path,
    exclude: set[str],
    already: set[Path],
) -> list[tuple[str, Path]]:
    """All model dirs under *group_dir*, minus exclusions and ones already listed."""
    found: list[tuple[str, Path]] = []
    if not group_dir.is_dir():
        sys.exit(f"ERROR: group dir does not exist: {group_dir}")

    for child in sorted(group_dir.iterdir()):
        if not child.is_dir() or child.name in exclude:
            continue
        if not is_model_dir(child):
            continue
        if child.resolve() in already:
            continue
        found.append((child.name, child.resolve()))
    return found


def load_config(path: Path) -> BenchmarkConfig:
    """
    Parse a YAML (or JSON) comparison config.

    Schema
    ------
    ::

        title:   "Unified vs. big-condition"     # optional figure title
        output:  figures/comparison.png          # optional; else <config>.png
        figsize: [22, 14]                        # optional, inches

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
    text = path.read_text()
    if path.suffix.lower() in {".json"}:
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    if not isinstance(raw, dict):
        sys.exit(f"ERROR: {path} must contain a mapping at the top level.")

    if "groups" in raw and "experiments" in raw:
        sys.exit(
            f"ERROR: {path} defines both 'groups' and 'experiments' at the top "
            "level — use one or the other (put per-group runs under "
            "groups[].experiments)."
        )

    if "groups" in raw:
        raw_groups = raw["groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            sys.exit(f"ERROR: 'groups' in {path} must be a non-empty list.")
    elif "experiments" in raw:
        # A flat list is modelled as a single unnamed group.
        raw_groups = [{"name": None, "experiments": raw["experiments"]}]
    else:
        sys.exit(f"ERROR: {path} must define either 'experiments' or 'groups'.")

    groups: list[tuple[str | None, list[tuple[str, Path]]]] = []
    seen_names: dict[str, Path] = {}
    seen_dirs: set[Path] = set()

    for g_idx, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            sys.exit(f"ERROR: groups[{g_idx}] in {path} must be a mapping.")

        group_name = group.get("name")
        group_name = None if group_name is None else str(group_name)
        group_dir = group.get("dir")
        group_dir = Path(group_dir).expanduser() if group_dir else None
        where = f"groups[{g_idx}]" + (f" ('{group_name}')" if group_name else "")

        members: list[tuple[str, Path]] = []
        for e_idx, entry in enumerate(group.get("experiments") or []):
            members.append(
                _resolve_experiment(entry, group_dir, f"{where}.experiments[{e_idx}]")
            )

        if group.get("all_models"):
            if group_dir is None:
                sys.exit(f"ERROR: {where} sets 'all_models' but has no 'dir'.")
            exclude = {str(x) for x in (group.get("exclude") or [])}
            listed = {d for _, d in members}
            members.extend(_discover_group_models(group_dir, exclude, listed))

        if not members:
            sys.exit(f"ERROR: {where} selects no experiments.")

        for name, model_dir in members:
            if name in seen_names:
                sys.exit(
                    f"ERROR: duplicate experiment name '{name}' "
                    f"({seen_names[name]} and {model_dir}). Give one an "
                    "explicit, unique 'name'."
                )
            seen_names[name] = model_dir
            seen_dirs.add(model_dir)

        groups.append((group_name, members))

    figsize = raw.get("figsize")
    if figsize is not None:
        if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            sys.exit(f"ERROR: 'figsize' in {path} must be [width, height].")
        figsize = (float(figsize[0]), float(figsize[1]))

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

def _group_color(group_idx: int, member_idx: int, n_members: int) -> tuple:
    """Base hue from the group; successive members within it are lightened."""
    base = mcolors.to_rgba(_GROUP_PALETTE[group_idx % len(_GROUP_PALETTE)])
    lighten = 0.45 * (member_idx / max(n_members - 1, 1)) if n_members > 1 else 0.0
    return (
        base[0] + (1 - base[0]) * lighten,
        base[1] + (1 - base[1]) * lighten,
        base[2] + (1 - base[2]) * lighten,
        1.0,
    )


def _grouped_layout(
    model_names: list[str],
    groups: list[tuple[str | None, list[str]]],
) -> tuple[list[float], list[tuple], list[tuple[float, float, str | None]]]:
    """
    Lay bars out group by group, separated by ``_GROUP_GAP``.

    Returns (x per model, colour per model, [(x_lo, x_hi, group_name), ...]),
    all in the same order as *model_names*.
    """
    index  = {name: i for i, name in enumerate(model_names)}
    xs     = [0.0] * len(model_names)
    colors = [(0.0, 0.0, 0.0, 1.0)] * len(model_names)
    spans: list[tuple[float, float, str | None]] = []

    cursor = 0.0
    for g_idx, (group_name, members) in enumerate(groups):
        span_lo = cursor
        for m_idx, name in enumerate(members):
            i = index[name]
            xs[i] = cursor
            colors[i] = _group_color(g_idx, m_idx, len(members))
            cursor += 1.0
        spans.append((span_lo, cursor - 1.0, group_name))
        cursor += _GROUP_GAP

    return xs, colors, spans


def plot_benchmark(
    results: dict[str, dict[str, dict[str, float]]],
    primary_overrides: dict[str, str],
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
    metric_subsets: dict[str, list[str]] | None = None,
    groups: list[tuple[str | None, list[str]]] | None = None,
    title: str = "Ablation Benchmark",
) -> None:
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
    """
    if not results:
        print("No results found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    all_tasks, task_metrics, primary = _build_task_metrics(
        results, primary_overrides, metric_subsets
    )
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_tasks     = len(all_tasks)
    n_cols      = max((len(task_metrics[t]) for t in all_tasks), default=1)

    if n_tasks == 0:
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

    # Widen the figure in proportion to the gaps inserted between groups
    # (width_scale is exactly 1.0 when there are none).
    width_scale = float(x_positions.max() - x_positions.min() + 1) / n_models

    col_w, row_h = 2.6, 3.2
    fig_w = figsize[0] if figsize else max(n_cols * col_w * width_scale + 1.5, 6.0)
    fig_h = figsize[1] if figsize else max(n_tasks * row_h + 1.2, 4.0)

    fig, axes = plt.subplots(
        n_tasks, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    bar_width = 0.75

    # Bottom-most visible subplot per column — where group names are printed.
    # (Tasks have different metric counts, so the grid is ragged.)
    last_visible_row: dict[int, int] = {}
    for row, task in enumerate(all_tasks):
        for col in range(len(task_metrics[task])):
            last_visible_row[col] = row

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
        ncol=min(n_models, 6),
        frameon=True,
        fontsize=9,
        title="Experiment" if groups else "Model",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    groups: list[tuple[str | None, list[str]]] | None = None
    metric_subsets: dict[str, list[str]] = {}
    primary_overrides: dict[str, str] = {}
    title = "Ablation Benchmark"
    figsize = tuple(args.figsize) if args.figsize else None

    if args.config is not None:
        # ---- Config-driven: hand-picked runs, possibly across ablations ----
        config_path: Path = args.config.expanduser().resolve()
        if not config_path.is_file():
            print(f"ERROR: --config does not exist: {config_path}", file=sys.stderr)
            sys.exit(1)

        config = load_config(config_path)
        title  = config.title
        figsize = figsize or config.figsize

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
    )


if __name__ == "__main__":
    main()
