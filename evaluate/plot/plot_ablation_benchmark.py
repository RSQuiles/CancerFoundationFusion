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

Anything missing from ``metrics/`` — a whole task file, or individual metrics
within one — is filled in from a sibling ``metrics_old/`` when that directory
exists.  ``metrics/`` always wins where both have a value.  Every substitution is
reported on stderr and counted in a summary line, because a bar filled from
``metrics_old`` was produced by an earlier run and may not be comparable with its
neighbours.

Layout:  rows = tasks,  columns = metrics within that task.

With ``--per-task`` each task is written to its own figure instead, named after
``--output`` with the task appended (``benchmark_deconv.png``). Colours, bar order
and group bands are computed once across all tasks and shared, so the separate
figures stay comparable with each other and with the combined one. These figures
place at most two metrics per row (``PER_TASK_MAX_COLS``), wrapping onto further
rows and centring a row that ends up short.

Figures are sized from their content unless a size is given: width from how many
bars a subplot must hold, height from the number of rows and the legend. The floor
on width is the rotated model name above each bar, so ``--no-bar-names`` (they stay
in the legend) is the lever when a dense figure is still too wide for a document.

Every bar is labelled with its value and, unless ``--no-bar-names``, with a short
``{group}.{member}`` handle (1.1, 1.2, 2.3, …) whose meaning the legend spells out
as ``1.2: base/contrastive`` — so a dense figure can be read without tracing
colours back to the legend. ``--no-bar-aliases`` puts the full display names on the
bars instead.

Text size is set by ``--font-scale`` / ``--font-size role=size`` (config keys
``font_scale`` / ``font_sizes``); see :class:`FontSizes` for the roles.

Every figure is written with a ``{stem}.csv`` beside it holding exactly the numbers
it plots — one row per model, one column per metric — so ``--per-task`` yields one
CSV per task. ``--no-csv`` turns that off.

A metric with a known ceiling gets an axis that stops there — accuracy runs to 1,
not to just above the best model — so bar heights mean the same thing in every
figure. ``--y-max metric=value`` (config ``y_max``) overrides it; see
:data:`METRIC_UPPER_BOUND` and :func:`metric_upper_bound`.

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

    # Bigger text, but keep the numbers small; cap an unbounded metric:
    python evaluate/plot/ablation_benchmark.py --config comparison.yaml \\
        --font-scale 1.4 --font-size value=6 --y-max d_calibration=40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field, fields
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
    # Superseded here by is_benchmark_model_dir, which also accepts a model whose
    # results live in metrics_old/. Kept as a re-export: it was part of this
    # module's surface before that, and callers still import it from here.
    is_model_dir,  # noqa: F401
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

# Metrics with a known ceiling: the y-axis stops there rather than just above the
# best run. Autoscaling makes 0.42 accuracy look like a full bar, hides how much
# headroom is left, and stops two figures of the same metric being comparable by
# eye. Errors (mae/rmse/mse) and unbounded statistics (d_calibration) are absent on
# purpose — they have no ceiling to draw.
METRIC_UPPER_BOUND: dict[str, float] = {
    "accuracy": 1.0, "balanced_accuracy": 1.0,
    "f1": 1.0, "f1_weighted": 1.0, "f1_macro": 1.0, "f1_micro": 1.0,
    "precision": 1.0, "precision_macro": 1.0, "precision_weighted": 1.0,
    "recall": 1.0, "recall_macro": 1.0, "recall_weighted": 1.0,
    "auroc": 1.0, "mean_auroc": 1.0, "auprc": 1.0, "average_precision": 1.0,
    "c_index": 1.0, "antolini_concordance": 1.0,
    "ibs": 1.0,               # integrated Brier score
    "event_rate": 1.0,
    "r2": 1.0, "mean_r2": 1.0,
}

# Families whose members are all bounded by 1, matched as substrings so that
# task-specific variants (mean_pearson_r_present, ic50_spearman_rho, ...) are
# covered without listing every one.
_BOUNDED_BY_1_SUBSTRINGS: tuple[str, ...] = (
    "pearson", "spearman", "kendall", "concordance", "accuracy", "auroc", "auprc",
    "_auc", "auc_", "f1", "precision", "recall", "c_index", "jaccard", "dice",
)

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
    "mean_pearson_r_present": "Mean Pearson r (Present)",
    "rmse_present": "RMSE (present)",
    "mean_sample_pearson_r": "Mean Pearon r (sample)",
    "antolini_concordance": "Antolini Concordance",
    "ibs": "IBS",
    "d_calibration": "D-calibration",
    "spearman_rho": "Spearman ρ",
    "r2": "R2",
    "auroc": "AUROC",
    "auprc": "APRC",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced Accuracy"
}

# Background colour for the primary-metric subplot.
_PRIMARY_BG   ="#FFFDE7"   # very light yellow
_PRIMARY_EDGE = "#F9A825"   # amber border

# Metrics per row in a per-task figure. The combined grid keeps one row per task
# however many metrics it has; a single-task figure has the width to spare, so it
# wraps instead of shrinking every subplot. A row that ends up short is centred.
PER_TASK_MAX_COLS = 2

# The three labels that identify a bar: its name above it, its group under the
# bottom row, and the legend. Sized together, since they are read together.
# These are the defaults of FontSizes below; every one can be overridden per figure.
BAR_NAME_FONTSIZE   = 9
GROUP_NAME_FONTSIZE = 11
LEGEND_FONTSIZE     = 11

# Columns in the shared legend. The figure is saved with bbox_inches="tight", so a
# legend wider than the axes stretches the saved image beyond them — with long
# "1.2: base/contrastive" entries, six columns is what makes a figure too wide for a
# document. Lowering it trades width for legend rows; `legend_ncol` overrides it.
LEGEND_MAX_COLS = 6


@dataclass(frozen=True)
class FontSizes:
    """Point size of every piece of text in the figure.

    Configs set these through ``font_scale`` (multiply everything) and/or
    ``font_sizes`` (pin individual roles) — see :func:`resolve_font_sizes`. They are
    resolved once per figure and passed down, rather than read from the module
    constants at each call site, so one figure can be typeset large for a poster
    and another small for a paper in the same run.
    """

    bar_name:     float = BAR_NAME_FONTSIZE     # rotated name above each bar
    value:        float = 6.0                   # the number above each bar
    group_name:   float = GROUP_NAME_FONTSIZE   # group label under the bottom row
    legend:       float = LEGEND_FONTSIZE       # legend entries and its title
    metric_title: float = 9.0                   # per-subplot metric name
    task_label:   float = 10.0                  # task name on the y-axis
    tick:         float = 10.0                  # y-axis tick labels (matplotlib default)
    suptitle:     float = 13.0                  # figure title
    footnote:     float = 8.0                   # "★ = primary metric"
    no_data:      float = 9.0                   # placeholder in an empty subplot


# Role -> what a caller may name in `font_sizes`. Kept explicit so a typo is an
# error rather than a silently ignored key.
FONT_ROLES: tuple[str, ...] = tuple(f.name for f in fields(FontSizes))


def resolve_font_sizes(
    scale: float = 1.0,
    overrides: dict[str, float] | None = None,
) -> FontSizes:
    """Build a :class:`FontSizes` from a global scale and per-role overrides.

    ``scale`` multiplies the defaults; ``overrides`` then pins individual roles to
    an absolute size, so ``font_scale: 1.5`` with ``font_sizes: {value: 6}`` means
    "half again bigger, except keep the numbers small".
    """
    if scale <= 0:
        raise ValueError(f"font_scale must be positive, got {scale}")

    unknown = sorted(set(overrides or {}) - set(FONT_ROLES))
    if unknown:
        raise KeyError(
            f"Unknown font_sizes key(s): {unknown}. Valid roles: {list(FONT_ROLES)}"
        )

    defaults = FontSizes()
    resolved = {
        role: float(getattr(defaults, role)) * scale for role in FONT_ROLES
    }
    for role, size in (overrides or {}).items():
        if float(size) <= 0:
            raise ValueError(f"font_sizes['{role}'] must be positive, got {size}")
        resolved[role] = float(size)
    return FontSizes(**resolved)

# Auto-sizing, in inches. Width is per bar rather than per subplot: a subplot
# holding 48 bars needs a different width from one holding 4, and the old flat
# 2.6in per column is why every real config had to hardcode `figsize`.
#
# What sets the floor is the rotated model name above each bar — turned 90° its
# footprint is the font's line height, so bars any narrower than that and the names
# touch. Without them the bars can be packed much tighter, which is the single
# biggest lever on how wide (and so how document-shaped) a dense figure comes out.
# A rotated 9pt label measures 0.130in across (check_benchmark_layout asserts
# adjacent names stay clear), so 0.17 leaves ~0.04in of air — near the floor. The
# slot tracks the bar-name font size, otherwise raising it would silently start
# overlapping; _bar_slot_inches() reproduces 0.17 exactly at the 9pt default.
BAR_SLOT_INCHES       = 0.17   # bar names on, at BAR_NAME_FONTSIZE
BAR_SLOT_INCHES_PLAIN = 0.09   # bar names off
MIN_COL_INCHES        = 2.6    # keeps a 2-bar subplot from collapsing
ROW_INCHES            = 4.6    # tall rows, so a wrapped figure reads closer to square

# Width of a horizontal "0.408" value label. Bars narrower than this get their
# value labels rotated, otherwise adjacent numbers merge into one string.
_VALUE_LABEL_INCHES = 0.32


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

# Sibling of metrics/ consulted for anything metrics/ does not provide. Named
# rather than configurable: it is a convention for parking a previous run's results
# (`mv metrics metrics_old`) so a partially recomputed sweep still plots in full.
FALLBACK_METRICS_DIRNAME = "metrics_old"

# (model, task, (metric, ...)) for every value taken from the fallback directory.
# A whole task read from there is recorded with keys=None.
_FALLBACK_EVENTS: list[tuple[str, str, tuple[str, ...] | None]] = []


def fallback_events() -> list[tuple[str, str, tuple[str, ...] | None]]:
    """Substitutions made from ``metrics_old`` so far, for a caller's summary."""
    return list(_FALLBACK_EVENTS)


def _read_metrics_dir(metrics_dir: Path) -> dict[str, dict[str, float]]:
    """Read one ``metrics/``-shaped directory into ``{task: {metric: value}}``."""
    per_task: dict[str, dict[str, float]] = {}

    for jf in sorted(metrics_dir.glob("results_*.json")):
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


def collect_model_metrics(
    model_dir: Path,
    fallback_dirname: str | None = FALLBACK_METRICS_DIRNAME,
) -> dict[str, dict[str, float]]:
    """
    Read one model directory and return ``{task_name: {metric: value, ...}}``.

    A model directory is recognised by having a ``metrics/`` subfolder
    containing at least one ``results_*.json`` file — or, with a fallback
    configured, such a file under ``{fallback_dirname}/``.

    Whatever ``metrics/`` does not provide is filled in from
    ``{fallback_dirname}/`` (default ``metrics_old``): a task with no results file
    is taken from there wholesale, and a task that has one keeps every value it
    holds while missing metrics are borrowed. ``metrics/`` therefore always wins on
    a key both provide. Pass ``fallback_dirname=None`` to read ``metrics/`` alone.

    Every substitution is reported on stderr and recorded in
    :func:`fallback_events`. Mixing two runs in one figure is useful when only part
    of a sweep has been recomputed, but it is not something to discover afterwards.
    """
    per_task = _read_metrics_dir(model_dir / "metrics")

    if not fallback_dirname:
        return per_task

    old_dir = model_dir / fallback_dirname
    if not old_dir.is_dir():
        return per_task

    for task, old_values in _read_metrics_dir(old_dir).items():
        if task not in per_task:
            per_task[task] = dict(old_values)
            _FALLBACK_EVENTS.append((model_dir.name, task, None))
            print(
                f"[{fallback_dirname}] {model_dir.name}: task '{task}' taken from "
                f"{fallback_dirname}/ (absent from metrics/)",
                file=sys.stderr,
            )
            continue

        borrowed = tuple(k for k in old_values if k not in per_task[task])
        if not borrowed:
            continue
        for key in borrowed:
            per_task[task][key] = old_values[key]
        _FALLBACK_EVENTS.append((model_dir.name, task, borrowed))
        print(
            f"[{fallback_dirname}] {model_dir.name}: {task} <- "
            f"{len(borrowed)} metric(s) from {fallback_dirname}/ "
            f"({', '.join(sorted(borrowed))})",
            file=sys.stderr,
        )

    return per_task


def _has_results(directory: Path) -> bool:
    """True if *directory* holds at least one ``results_*.json``."""
    return directory.is_dir() and any(directory.glob("results_*.json"))


def is_benchmark_model_dir(path: Path) -> bool:
    """``is_model_dir``, but a model with only ``metrics_old/`` still counts.

    Kept separate from the shared :func:`is_model_dir`, which looks exclusively at
    ``metrics/`` and is also used by ``plot_unified_metrics_table`` with different
    filename patterns.
    """
    return _has_results(path / "metrics") or (
        bool(FALLBACK_METRICS_DIRNAME)
        and _has_results(path / FALLBACK_METRICS_DIRNAME)
    )


def collect_metrics(ablation_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Walk ablation_dir and return:
        results[model_name][task_name] = {metric: value, ...}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_dir in sorted(ablation_dir.iterdir()):
        if not model_dir.is_dir() or not is_benchmark_model_dir(model_dir):
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
    # Print a label above each bar, in addition to the legend.
    bar_names: bool = True
    # Print the task's name on the y-axis of its first subplot. Redundant with
    # `per_task`, where the task is already in the figure title.
    task_labels: bool = True
    # Columns in the shared legend; None keeps LEGEND_MAX_COLS. Fewer columns means
    # more legend rows and a narrower figure.
    legend_ncol: int | None = None
    # That label is a short {group}.{member} handle; the legend carries alias: model.
    bar_aliases: bool = True
    # Multiplies every label size; font_sizes then pins individual roles.
    font_scale: float = 1.0
    font_sizes: dict[str, float] = field(default_factory=dict)
    # metric -> y-axis ceiling; 0 or null forces autoscaling for that metric.
    y_max: dict[str, float] = field(default_factory=dict)

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
        bar_names: false          # default true: label each bar as well as the
                                  # legend.
        bar_aliases: false        # default true: that label is a short
                                  # {group}.{member} handle (1.1, 1.2, 2.3, ...)
                                  # and the legend reads "1.2: base/contrastive".
                                  # false prints full display names on the bars.

        font_scale: 1.4           # multiply every label size (default 1.0)
        font_sizes:               # …then pin individual roles, in points
          bar_name: 11            # roles: bar_name, value, group_name, legend,
          value: 7                #        metric_title, task_label, tick,
          legend: 13              #        suptitle, footnote, no_data
                                  # Raising bar_name widens the figure, so that
                                  # neighbouring names stay clear of each other.

        y_max:                    # y-axis ceiling per metric. Metrics bounded by 1
          d_calibration: 20       # (accuracy, correlations, C-index, ...) already
          rmse_present: 0         # stop at 1; 0/null forces autoscaling instead.

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
    # is_benchmark_model_dir, not is_model_dir: a run whose results have been parked
    # in metrics_old/ must still be discovered by 'all_models'.
    groups = parse_groups(raw, path, is_benchmark_model_dir)
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

    font_sizes = {
        str(role): float(size)
        for role, size in (raw.get("font_sizes") or {}).items()
    }
    # Validate here, at load time, so a typo names the config rather than surfacing
    # deep inside the renderer.
    resolve_font_sizes(float(raw.get("font_scale", 1.0) or 1.0), font_sizes)

    y_max = {
        str(metric): (0.0 if value is None else float(value))
        for metric, value in (raw.get("y_max") or {}).items()
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
        task_labels=bool(raw.get("task_labels", True)),
        legend_ncol=_legend_ncol_from_config(raw.get("legend_ncol")),
        bar_aliases=bool(raw.get("bar_aliases", True)),
        font_scale=float(raw.get("font_scale", 1.0) or 1.0),
        font_sizes=font_sizes,
        y_max=y_max,
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
            if not is_benchmark_model_dir(model_dir):
                print(
                    f"[warning] no results_*.json under {model_dir}/metrics "
                    f"or {model_dir}/{FALLBACK_METRICS_DIRNAME} "
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


def metric_upper_bound(
    metric: str, overrides: dict[str, float] | None = None
) -> float | None:
    """Ceiling for a metric's y-axis, or None to let matplotlib autoscale.

    Resolution: caller override -> exact name -> known bounded family. A
    non-positive override forces autoscaling for that one metric.
    """
    if overrides and metric in overrides:
        value = overrides[metric]
        return float(value) if value is not None and float(value) > 0 else None
    if metric in METRIC_UPPER_BOUND:
        return METRIC_UPPER_BOUND[metric]
    lowered = metric.lower()
    if any(token in lowered for token in _BOUNDED_BY_1_SUBSTRINGS):
        return 1.0
    return None


def _apply_metric_ylim(ax, values: list[float], bound: float | None) -> None:
    """Pin the y-axis top to the metric's ceiling, when it has one and it fits.

    Bars start at 0 unless something is negative (a correlation can be). If a value
    somehow exceeds the ceiling the top is left autoscaled rather than clipping the
    bar out of sight.
    """
    if not values:
        return
    lo, hi = ax.get_ylim()
    bottom = min(0.0, min(values), lo)
    top = bound if (bound is not None and max(values) <= bound) else hi
    ax.set_ylim(bottom, top)


def build_aliases(
    model_names: list[str],
    groups: list[tuple[str | None, list[str]]] | None,
) -> dict[str, str]:
    """Short handle per model: ``{group}.{member}``, both 1-based.

    Group 1's runs are 1.1, 1.2, …; group 2's third run is 2.3. Ungrouped runs are
    numbered straight through (1, 2, 3, …).

    A full display name printed above every bar is what forces a dense figure to be
    wide and tall — 48 of them, rotated, each as long as "palign/contrastive". A
    three-character handle costs one legend lookup and buys back the space, and the
    number itself carries the grouping, which a colour shade does not.
    """
    aliases: dict[str, str] = {}
    grouped = bool(groups) and (len(groups) > 1 or any(name for name, _ in groups))
    n_groups = 0
    if grouped:
        for group_idx, (_, members) in enumerate(groups, start=1):
            for member_idx, name in enumerate(members, start=1):
                aliases[name] = f"{group_idx}.{member_idx}"
            n_groups = group_idx

    # The ungrouped case, and any model the groups did not cover — the latter goes
    # into a trailing implicit group so its handle cannot read like a group index.
    leftover = 1
    for name in model_names:
        if name in aliases:
            continue
        aliases[name] = f"{n_groups + 1}.{leftover}" if grouped else str(leftover)
        leftover += 1
    return aliases


def csv_output_path(output: Path) -> Path:
    """The CSV that accompanies a figure: same directory, same stem, ``.csv``."""
    return output.with_suffix(".csv")


def write_benchmark_csv(
    output: Path,
    tasks: list[str],
    task_metrics: dict[str, list[str]],
    results: dict[str, dict[str, dict[str, float]]],
    model_names: list[str],
    aliases: dict[str, str] | None,
    groups: list[tuple[str | None, list[str]]] | None,
) -> Path:
    """Write the numbers behind a figure next to it, as ``{figure stem}.csv``.

    One row per model in plot order, one column per plotted metric — exactly the
    values the bars encode, so the figure can be re-read, re-plotted elsewhere or
    pasted into a table without going back to the per-model JSONs.

    With more than one task in the figure the columns are prefixed ``task::metric``,
    since two tasks can have a metric of the same name. A per-task figure gets bare
    metric names.
    """
    group_of: dict[str, str] = {}
    for group_name, members in groups or []:
        for name in members:
            group_of[name] = group_name or ""

    prefixed = len(tasks) > 1
    columns: list[tuple[str, str, str]] = [          # (header, task, metric)
        (f"{task}::{metric}" if prefixed else metric, task, metric)
        for task in tasks
        for metric in task_metrics.get(task, [])
    ]

    path = csv_output_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "alias", "model"] + [c[0] for c in columns])
        for name in model_names:
            row = [group_of.get(name, ""), (aliases or {}).get(name, ""), name]
            for _, task, metric in columns:
                value = results.get(name, {}).get(task, {}).get(metric)
                row.append(
                    value if isinstance(value, (int, float))
                    and not isinstance(value, bool) else ""
                )
            writer.writerow(row)

    print(f"Saved to {path}")
    return path


def _legend_inches(n_models: int, ncol: int, fontsize: float = LEGEND_FONTSIZE) -> float:
    """Vertical space the shared legend needs, in inches.

    Computed rather than hardcoded because the old fixed 6% reserve only happened
    to fit the combined figure: a per-task figure is a third of the height, so the
    same fraction is a third of the space for exactly as many legend rows. Scales
    with the legend's font size, which is what sets the row height.
    """
    rows = math.ceil(n_models / max(ncol, 1))
    row_in = 0.23 * (fontsize / 9.0)
    return row_in * rows + 0.45        # rows + frame/title padding


def _legend_ncol_from_config(value: object) -> int | None:
    """Validate a config's ``legend_ncol``; None/0 means "use the default cap"."""
    if value is None:
        return None
    try:
        ncol = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"legend_ncol must be a positive integer, got {value!r}")
    if ncol < 0:
        raise ValueError(f"legend_ncol must be a positive integer, got {value!r}")
    return ncol or None


def _legend_columns(n_models: int, legend_ncol: int | None = None) -> int:
    """Columns the shared legend is drawn in.

    ``legend_ncol`` caps them explicitly; without it the default cap applies. Never
    more columns than entries, so a 3-model figure does not reserve 6 columns' width.
    """
    return max(1, min(n_models, legend_ncol or LEGEND_MAX_COLS))


def _bar_slot_inches(bar_name_fontsize: float) -> float:
    """Width one bar needs so its rotated name clears the neighbouring one.

    Rotated 90°, the name's horizontal footprint is the font's line height, so the
    slot has to track the font size. Returns BAR_SLOT_INCHES at the default 9pt.
    """
    return 0.130 / 9.0 * bar_name_fontsize + 0.04


@dataclass(frozen=True)
class _Cell:
    """One subplot: a (task, metric) placed on the grid.

    ``col`` is in half-columns. Every cell spans two of them, so a row holding
    fewer cells than the grid is wide can be centred on an odd offset — which is
    what puts a lone metric in the middle instead of hard left.
    """

    row:    int
    col:    int
    task:   str
    metric: str


def _plan_cells(
    tasks: list[str],
    task_metrics: dict[str, list[str]],
    max_cols: int | None,
) -> tuple[list[_Cell], int, int]:
    """Lay the (task, metric) subplots out on a grid.

    ``max_cols`` caps the metrics per row; a task with more wraps onto further
    rows and a short row is centred. ``None`` keeps the original behaviour — one
    row per task, as wide as the widest task, short rows left-aligned — so the
    combined grid is unchanged.

    Returns ``(cells, n_grid_rows, n_cols)``.
    """
    widest = max((len(task_metrics[t]) for t in tasks), default=1) or 1
    n_cols = min(max_cols, widest) if max_cols else widest

    cells: list[_Cell] = []
    row = 0
    for task in tasks:
        metrics = task_metrics.get(task) or []
        if not metrics:
            row += 1          # keep an empty row so the task still gets its label
            continue
        for start in range(0, len(metrics), n_cols):
            chunk = metrics[start:start + n_cols]
            # Centre only when wrapping. In the combined grid a task with fewer
            # metrics than its neighbours has always been left-aligned, and its
            # columns line up with theirs; centring would break that.
            offset = (2 * n_cols - 2 * len(chunk)) // 2 if max_cols else 0
            for i, metric in enumerate(chunk):
                cells.append(_Cell(row, offset + 2 * i, task, metric))
            row += 1

    return cells, max(row, 1), n_cols


def _fit_group_labels(
    spans: list[tuple[float, float, str | None]],
    axes_width_in: float,
    fontsize: float,
) -> tuple[float, dict[str, str]]:
    """Wrap group names, and shrink them, to the width each group actually has.

    A group is only as wide as its bars: "Paired mix" spans two of them. At a
    readable font size the names of adjacent narrow groups run into each other,
    and the narrower the figure the worse it gets — so wrap them onto several
    lines, and scale down (never below 3/4) when even the longest single word
    does not fit, since that word is what wrapping cannot help with.

    Returns ``(fontsize, {group_name: wrapped_text})``.
    """
    named = [(lo, hi, name) for lo, hi, name in spans if name]
    if not named:
        return fontsize, {}

    # Matches the xlim the subplots are given: spans plus 0.7 padding each side.
    total = spans[-1][1] - spans[0][0] + 1.4
    if total <= 0 or axes_width_in <= 0:
        return fontsize, {}

    available = {
        name: (hi - lo + 1) / total * axes_width_in for lo, hi, name in named
    }

    def char_inches(size: float) -> float:
        return 0.62 * size / 72.0

    longest_word = {
        name: len(max(name.split(), key=len, default="")) for name in available
    }
    need = max(
        longest_word[name] * char_inches(fontsize) / width
        for name, width in available.items()
        if width > 0
    )
    size = fontsize * min(1.0, max(0.75, 1.0 / need)) if need > 1 else fontsize

    wrapped = {}
    for name, width in available.items():
        # Floor the wrap width: breaking a name into two-character shreds is worse
        # than letting the narrowest group's label overhang a little.
        max_chars = max(int(width / char_inches(size)), 8)
        # break_long_words=False: overhanging slightly beats "Big cond / ition".
        wrapped[name] = "\n".join(
            textwrap.wrap(name, max_chars, break_long_words=False)
        ) or name
    return size, wrapped


def _group_label_cells(
    cells: list[_Cell], n_grid_rows: int, wrapped: bool
) -> set[tuple[int, int]]:
    """Which cells print the group names underneath.

    Wrapped: the bottom grid row only. Anything higher would collide with the
    titles of the row beneath it.

    Unwrapped: the bottom-most cell of each column, as before — the combined grid
    is ragged, so a column whose last task has that metric may end above the
    figure's final row.
    """
    if wrapped:
        return {(c.row, c.col) for c in cells if c.row == n_grid_rows - 1}

    last_row: dict[int, int] = {}
    for cell in cells:
        last_row[cell.col] = max(last_row.get(cell.col, -1), cell.row)
    return {(row, col) for col, row in last_row.items()}


def _name_band_points(
    model_names: list[str], fontsize: float, padding: float
) -> float:
    """Vertical space the rotated model names occupy above a bar, in points.

    Rotated 90°, a label's extent is its rendered *length*; 0.62 * fontsize is a
    good average advance for DejaVu Sans, matplotlib's default, and erring high
    only costs a little whitespace.
    """
    longest = max((len(n) for n in model_names), default=0)
    return padding + 0.62 * fontsize * longest + 4.0


def _fit_titles_over_names(
    fig,
    capped: list[tuple[object, dict, float]],
    needed_pt: float,
    base_pad: float = 4.0,
) -> None:
    """Push each capped subplot's title above the names that overflow its frame.

    An axis pinned to its metric's ceiling cannot be stretched to fit the rotated
    names, so whatever does not fit under the ceiling is drawn above the frame
    (bar labels are annotations and are not clipped). Only the *overflow* needs
    reserving, so measure how much room is already there between the tallest bar
    and the ceiling — padding by the full band leaves the title floating.
    """
    fig.canvas.draw()
    for ax, title_kwargs, top_value in capped:
        height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
        lo, hi = ax.get_ylim()
        inside_pt = (hi - top_value) / (hi - lo) * height_pt if hi > lo else 0.0
        ax.set_title(**title_kwargs, pad=base_pad + max(0.0, needed_pt - inside_pt))


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

    Idempotent: the pre-expansion limits are remembered per axes and every call
    recomputes from those, so this can be re-run after a second layout pass
    without the expansions compounding.
    """
    needed_pt = _name_band_points(model_names, fontsize, padding)

    fig.canvas.draw()
    for ax in axes_list:
        ax_pt = ax.get_window_extent().height * 72.0 / fig.dpi
        if ax_pt <= 0:
            continue
        lo, hi = getattr(ax, "_bar_name_base_ylim", None) or ax.get_ylim()
        ax._bar_name_base_ylim = (lo, hi)
        # Growing the range by `frac` leaves only frac/(1+frac) of the axes above a
        # bar that reaches the old top — the axes' physical height does not change,
        # so the new headroom is a share of it, not an addition to it. Solving
        # frac/(1+frac) * ax_pt = needed_pt gives the denominator below; using
        # needed_pt/ax_pt directly (as this did) clips the tallest bar's label.
        # The floor on the denominator caps the expansion: a pathologically short
        # axes would otherwise blow the scale up until the bars vanish.
        frac = min(needed_pt / max(ax_pt - needed_pt, 0.25 * ax_pt), 3.0)
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
    task_labels: bool,
    output: Path | None,
    legend_ncol: int | None = None,
    max_cols: int | None = None,
    fonts: FontSizes | None = None,
    y_max: dict[str, float] | None = None,
    aliases: dict[str, str] | None = None,
):
    """Draw one figure covering *tasks* x their metrics.

    Without ``max_cols`` the layout is one row per task, as wide as the widest
    task. With it, a task's metrics wrap onto further rows of at most that many
    subplots and a short row is centred — see :func:`_plan_cells`.

    ``fonts`` sizes every label; ``y_max`` overrides the per-metric y-axis ceiling
    (see :func:`metric_upper_bound`). ``aliases`` maps a model to the short handle
    printed above its bars, with the legend carrying ``alias: model``.

    Returns the Figure; the caller decides whether to save, show or close it.
    """
    fonts = fonts or FontSizes()
    n_models = len(model_names)
    # What is written above each bar, and what the legend says it means. The handle
    # is parenthesised on the bar so a bare "2.3" beside a value cannot be misread
    # as part of the number; the legend keeps it unbracketed as the key.
    alias_of = aliases or {}
    bar_labels = [
        f"({alias_of[name]})" if name in alias_of else name for name in model_names
    ]
    legend_labels = [
        f"{alias_of[name]}: {name}" if name in alias_of else name
        for name in model_names
    ]

    cells, n_grid_rows, n_cols = _plan_cells(tasks, task_metrics, max_cols)
    label_cells = _group_label_cells(cells, n_grid_rows, wrapped=bool(max_cols))

    # Widen the figure in proportion to the gaps inserted between groups
    # (width_scale is exactly 1.0 when there are none).
    width_scale = float(x_positions.max() - x_positions.min() + 1) / n_models

    ncol_legend = _legend_columns(n_models, legend_ncol)
    legend_in   = _legend_inches(n_models, ncol_legend, fonts.legend)

    fig_w = figsize[0] if figsize else _figure_width(
        n_cols, n_models, width_scale, bar_names, fonts
    )
    fig_h = figsize[1] if figsize else _figure_height(
        n_grid_rows, n_models, fonts, legend_ncol
    )

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Two grid columns per subplot, so a centred row can sit on an odd offset.
    gridspec = fig.add_gridspec(n_grid_rows, 2 * n_cols)

    bar_width = 0.75

    # How much width each bar actually gets, so the value labels can be turned
    # before they run into each other. "0.408" is ~0.25in wide at 6pt; below that
    # the numbers of adjacent bars merge into one unreadable string.
    span = float(x_positions.max() - x_positions.min() + 1)
    pitch_in = (fig_w / max(n_cols, 1)) / max(span, 1.0)
    dense = pitch_in < _VALUE_LABEL_INCHES * (fonts.value / 6.0)
    group_fontsize, group_labels = _fit_group_labels(
        spans, 0.85 * fig_w / max(n_cols, 1), fonts.group_name
    )
    value_rotation = 90 if dense else 0
    # Rotated, a value label stands ~5 characters tall, so the model name above it
    # has to start that much higher.
    name_padding = (
        (0.62 * fonts.value * 5 + 6.0) if dense else (fonts.value + 7.0)
    )

    # Axes that ended up with bars, so the rotated-name headroom can be applied to
    # exactly those, after layout, when their real pixel height is known.
    axes_with_bars: list = []
    # (axes, title kwargs, tallest bar) for axes pinned to a metric ceiling: their
    # titles are repositioned instead, since their limits must not move.
    capped_axes: list[tuple[object, dict, float]] = []
    # First subplot of each task carries the task label.
    labelled_tasks: set[str] = set()
    first_axes = None

    for cell in cells:
        task, metric = cell.task, cell.metric
        ax = fig.add_subplot(gridspec[cell.row, cell.col:cell.col + 2])
        if first_axes is None:
            first_axes = ax

        is_primary = metric == primary.get(task)

        # Highlight primary metric subplot.
        if is_primary:
            ax.set_facecolor(_PRIMARY_BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_PRIMARY_EDGE)
                spine.set_linewidth(1.8)

        any_bar = False
        plotted: list[float] = []
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
            plotted.append(float(value))
            ax.bar_label(
                bar, fmt="%.3f", padding=2, fontsize=fonts.value,
                rotation=value_rotation,
            )
            if bar_names:
                # Above the value label (padding is in points, so this clears
                # it regardless of the data scale), rotated to fit dense grids.
                ax.bar_label(
                    bar, labels=[bar_labels[model_idx]], padding=name_padding,
                    fontsize=fonts.bar_name, rotation=90, color="#333333",
                )
            any_bar = True

        # Stop the axis at the metric's ceiling (1.0 for a rate or a correlation)
        # instead of just above the best run, so bar heights mean the same thing in
        # every figure.
        bound = metric_upper_bound(metric, y_max)
        capped = bool(plotted) and bound is not None and max(plotted) <= bound
        _apply_metric_ylim(ax, plotted, bound)

        # Subplot title: metric name + direction + star for primary.
        direction    = " ↓" if metric in LOWER_IS_BETTER else " ↑"
        star         = " ★" if is_primary else ""
        metric_label = METRIC_LABELS.get(metric, metric)
        title_kwargs = dict(
            label=metric_label + direction + star,
            fontsize=fonts.metric_title,
            fontweight="bold" if is_primary else "normal",
            color=_PRIMARY_EDGE if is_primary else "black",
        )
        ax.set_title(**title_kwargs, pad=4)

        # A capped axis cannot also be stretched to fit the rotated names — that is
        # what the y-range used to be inflated for. Names are annotations, so they
        # render above the frame instead, and the title is pushed clear of them
        # once the layout is known.
        if bar_names and capped and any_bar:
            capped_axes.append((ax, title_kwargs, max(plotted)))

        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # Only uncapped axes get the headroom treatment; a capped one must keep the
        # limit it was just given.
        if any_bar and not capped:
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
                color="grey", fontsize=fonts.no_data,
            )

        # Group names, printed under the bottom-most subplots.
        if named_groups and (cell.row, cell.col) in label_cells:
            ax_x0, ax_x1 = ax.get_xlim()
            ax_width = ax_x1 - ax_x0
            for span_lo, span_hi, group_name in spans:
                if not group_name:
                    continue
                centre = ((span_lo + span_hi) / 2 - ax_x0) / ax_width
                ax.text(
                    centre, -0.02, group_labels.get(group_name, group_name),
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=group_fontsize, color="dimgrey", style="italic",
                )

        # Task label as the y-axis label of the task's first subplot. When a task
        # wraps, only the first row is labelled — repeating it down the left edge
        # (or, worse, on a centred lone subplot) reads as a new task.
        if task_labels and task not in labelled_tasks:
            ax.set_ylabel(
                TASK_LABELS.get(task, task),
                fontsize=fonts.task_label,
                fontweight="bold",
                labelpad=8,
            )
            labelled_tasks.add(task)

        ax.tick_params(axis="y", labelsize=fonts.tick)

    # Shared legend below the figure.
    legend_handles = [
        # "1.2: base/contrastive" — the handle from the bar, then what it is.
        mpatches.Patch(color=colors[i], label=legend_labels[i])
        for i in range(n_models)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol_legend,
        frameon=True,
        fontsize=fonts.legend,
        title="Experiment" if grouped else "Model",
        title_fontsize=fonts.legend,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(title, fontsize=fonts.suptitle, fontweight="bold", y=1.01)
    # Reserve exactly the legend's height rather than a fixed 6%, so the same code
    # works for a tall combined grid and a short single-task figure.
    layout_rect = [0, min(legend_in / fig_h, 0.5), 1, 1]
    plt.tight_layout(rect=layout_rect)

    # After layout, so each axes' real height is known: the names are drawn at a
    # point offset and are invisible to the autoscaler, so the y-range has to be
    # widened by hand. Measuring beats guessing here — the same label needs a much
    # larger fraction of a short axes than of a tall one.
    if bar_names and (axes_with_bars or capped_axes):
        name_band = _name_band_points(bar_labels, fonts.bar_name, name_padding)

        def fit_names() -> None:
            # Uncapped axes make room by growing their y-range; capped ones keep
            # their limit and move the title instead.
            _apply_name_headroom(
                fig, axes_with_bars, bar_labels,
                fontsize=fonts.bar_name, padding=name_padding,
            )
            _fit_titles_over_names(fig, capped_axes, name_band)

        fit_names()
        # That first tight_layout sized the axes around names that then moved, so
        # lay out again and re-fit against the new axes heights — otherwise the
        # taller the names, the emptier the figure. Both fitters work from a
        # remembered base, so the two passes do not compound.
        plt.tight_layout(rect=layout_rect)
        fit_names()

    # Right-aligned to the rightmost subplot, not to the figure edge. Saving uses
    # bbox_inches="tight", so a footnote pinned at x=0.99 would hold the crop out
    # to the figure's right edge while the empty left margin was trimmed away —
    # which makes a centred lone subplot come out visibly off-centre.
    fig.text(
        max((ax.get_position().x1 for ax in fig.axes), default=0.99), 0.005,
        "★ = primary metric",
        ha="right", va="bottom", fontsize=8,
        color=_PRIMARY_EDGE, style="italic",
    )

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
    task_labels: bool = True,
    legend_ncol: int | None = None,
    font_scale: float = 1.0,
    font_sizes: dict[str, float] | None = None,
    y_max: dict[str, float] | None = None,
    bar_aliases: bool = True,
    write_csv: bool = True,
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
    after ``output`` with the task appended, and lays its metrics out at most
    ``PER_TASK_MAX_COLS`` per row with a short row centred. Colours, ordering and
    group bands are computed once and shared, so the per-task figures stay
    comparable with each other and with the combined one.

    ``bar_names`` labels each bar as well as the legend. ``bar_aliases`` makes that
    label a short ``{group}.{member}`` handle (1.1, 1.2, 2.3, …) and puts
    ``alias: model`` in the legend — see :func:`build_aliases`. Turn it off to print
    full display names on the bars, as before.

    ``legend_ncol`` caps the shared legend's columns (default
    :data:`LEGEND_MAX_COLS`). The figure is saved tight, so a legend wider than the
    axes widens the image: fewer columns stack the entries into more rows and give
    back that width, and the reserved height grows to match.

    ``font_scale`` multiplies every label's size and ``font_sizes`` pins individual
    roles (see :class:`FontSizes`). Raising the bar-name size widens the figure, so
    that names of neighbouring bars keep clear of each other.

    ``y_max`` overrides the per-metric y-axis ceiling; by default a metric bounded
    by 1 (accuracy, a correlation, C-index, ...) gets an axis that stops at 1
    instead of just above the best run — see :func:`metric_upper_bound`.

    ``write_csv`` also writes the plotted numbers beside each figure as
    ``{stem}.csv`` — one per figure, so ``per_task`` yields one CSV per task.

    Returns the list of figure paths written (empty when ``output`` is None); the
    CSVs sit next to them.
    """
    fonts = resolve_font_sizes(font_scale, font_sizes)

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

    # Built once from the shared ordering, so a run keeps the same handle in the
    # combined grid, in every per-task figure, and in the CSVs beside them.
    alias_map = build_aliases(model_names, groups) if bar_aliases else None

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
        task_labels=task_labels,
        legend_ncol=legend_ncol,
        fonts=fonts,
        y_max=y_max,
        aliases=alias_map,
    )

    written: list[Path] = []

    if per_task:
        for task in all_tasks:
            # Auto-sized unless the caller pinned per_task_figsize. `figsize`
            # deliberately does NOT carry over, in either dimension: it describes
            # the combined grid, which is as many task rows tall and as many metric
            # columns wide as the widest task. Reusing its width was what made
            # per-task figures come out at 4 columns' width for 2 columns of
            # content — the auto width below is derived from the bars themselves.
            out = task_output_path(output, task) if output is not None else None
            fig = _render_figure(
                tasks=[task],
                figsize=per_task_figsize,
                title=f"{title} — {TASK_LABELS.get(task, task)}".replace("\n", " "),
                output=out,
                max_cols=PER_TASK_MAX_COLS,
                **common,
            )
            if out is not None:
                written.append(out)
                if write_csv:
                    write_benchmark_csv(out, [task], task_metrics, results,
                                        model_names, alias_map, groups)
            if not show:
                plt.close(fig)
    else:
        fig = _render_figure(
            tasks=all_tasks, figsize=figsize, title=title, output=output, **common
        )
        if output is not None:
            written.append(output)
            if write_csv:
                write_benchmark_csv(output, all_tasks, task_metrics, results,
                                    model_names, alias_map, groups)
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return written


def _figure_height(
    n_grid_rows: int,
    n_models: int,
    fonts: FontSizes | None = None,
    legend_ncol: int | None = None,
) -> float:
    """Figure height in inches: the subplot rows plus the legend they need.

    ``legend_ncol`` must match what the legend is actually drawn with, or fewer
    columns would stack rows into height that was never reserved for them.
    """
    fonts = fonts or FontSizes()
    legend_in = _legend_inches(
        n_models, _legend_columns(n_models, legend_ncol), fonts.legend
    )
    return max(n_grid_rows * ROW_INCHES + legend_in + 0.8, 4.0)


def _figure_width(
    n_cols: int,
    n_models: int,
    width_scale: float,
    bar_names: bool,
    fonts: FontSizes | None = None,
) -> float:
    """Figure width in inches, from how many bars each subplot has to hold.

    ``width_scale`` folds in the gaps inserted between groups (1.0 when there are
    none). The per-bar slot tracks the bar-name font size, so raising it widens the
    figure rather than making the names overlap. Turning bar names off shrinks this
    a lot — the lever to reach for when a dense figure is too wide for a document.
    """
    fonts = fonts or FontSizes()
    slot = _bar_slot_inches(fonts.bar_name) if bar_names else BAR_SLOT_INCHES_PLAIN
    col_in = max(MIN_COL_INCHES, n_models * slot)
    return max(n_cols * col_in * width_scale + 1.5, 6.0)


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
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Columns in the shared legend (default: at most {LEGEND_MAX_COLS}). "
            "Fewer columns stack the entries into more rows, which narrows the "
            "figure. Overrides the config's 'legend_ncol' key."
        ),
    )
    parser.add_argument(
        "--task-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Print the task's name on the y-axis of its first subplot "
            "(default: on). Redundant with --per-task, where the task is already "
            "in the figure title. Overrides the config's 'task_labels' key."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help=(
            "Do not write the plotted numbers beside each figure. By default every "
            "figure gets a {stem}.csv in the same directory — one per task with "
            "--per-task."
        ),
    )
    parser.add_argument(
        "--bar-aliases",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Label bars with a short {group}.{member} handle (1.1, 1.2, 2.3, ...) "
            "and put 'alias: model' in the legend (default: on). "
            "--no-bar-aliases prints full display names on the bars instead. "
            "Overrides the config's 'bar_aliases' key."
        ),
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=None,
        metavar="FACTOR",
        help=(
            "Multiply every label's font size (1.0 = the defaults). Raising the "
            "bar-name size also widens the figure so names stay clear. Overrides "
            "the config's 'font_scale' key."
        ),
    )
    parser.add_argument(
        "--font-size",
        nargs="*",
        default=[],
        metavar="ROLE=SIZE",
        help=(
            "Pin individual label sizes in points, e.g. "
            "--font-size bar_name=11 legend=13. Applied on top of --font-scale. "
            f"Roles: {', '.join(FONT_ROLES)}."
        ),
    )
    parser.add_argument(
        "--y-max",
        nargs="*",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Y-axis ceiling per metric, e.g. --y-max d_calibration=20. Metrics "
            "bounded by 1 (accuracy, correlations, C-index, ...) already stop at 1; "
            "pass 0 to autoscale one instead. Merged over the config's 'y_max'."
        ),
    )
    return parser.parse_args()


def _parse_pairs(items: list[str], what: str) -> dict[str, float]:
    """Parse ``NAME=NUMBER`` CLI pairs, reporting anything malformed."""
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            print(
                f"WARNING: ignoring malformed --{what} entry '{item}' "
                f"(expected name=number)",
                file=sys.stderr,
            )
            continue
        name, value = item.split("=", 1)
        try:
            out[name.strip()] = float(value)
        except ValueError:
            print(
                f"WARNING: ignoring --{what} entry '{item}' (not a number)",
                file=sys.stderr,
            )
    return out


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
    task_labels = bool(args.task_labels) if args.task_labels is not None else True
    legend_ncol = _legend_ncol_from_config(args.legend_ncol)
    bar_aliases = bool(args.bar_aliases) if args.bar_aliases is not None else True
    font_scale = args.font_scale if args.font_scale is not None else 1.0
    font_sizes = _parse_pairs(args.font_size, "font-size")
    y_max = _parse_pairs(args.y_max, "y-max")

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
        task_labels = (
            args.task_labels if args.task_labels is not None else config.task_labels
        )
        legend_ncol = legend_ncol or config.legend_ncol
        bar_aliases = (
            args.bar_aliases if args.bar_aliases is not None else config.bar_aliases
        )
        font_scale = (
            args.font_scale if args.font_scale is not None else config.font_scale
        )
        # CLI pairs win per role/metric; the config supplies the rest.
        font_sizes = {**config.font_sizes, **font_sizes}
        y_max = {**config.y_max, **y_max}

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

    events = fallback_events()
    if events:
        n_keys  = sum(len(keys) for _, _, keys in events if keys)
        n_whole = sum(1 for _, _, keys in events if keys is None)
        parts = ([f"{n_keys} metric(s)"] if n_keys else []) + \
                ([f"{n_whole} whole task(s)"] if n_whole else [])
        print(
            f"NOTE: {' and '.join(parts)} across "
            f"{len({m for m, _, _ in events})} model(s) came from "
            f"{FALLBACK_METRICS_DIRNAME}/ rather than metrics/ — see the "
            f"[{FALLBACK_METRICS_DIRNAME}] lines above. Those bars are from an "
            f"earlier run."
        )

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
        task_labels=task_labels,
        legend_ncol=legend_ncol,
        bar_aliases=bar_aliases,
        font_scale=font_scale,
        font_sizes=font_sizes or None,
        y_max=y_max or None,
        write_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
