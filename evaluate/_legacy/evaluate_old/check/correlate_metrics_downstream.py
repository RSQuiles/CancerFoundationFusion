"""
Which internal metrics actually predict downstream performance?

Correlates each ``unified_metrics.json`` metric against one downstream task result —
by default cancer-type classification accuracy — across the selected runs, and draws
the answer as an annotated table.

    internal metric   x = {model_dir}/metrics/unified_metrics.json[key]
    target            y = {model_dir}/metrics/results_{task}.json[metric]

Spearman rather than Pearson: the internal metrics are on wildly different scales
(participation ratios in the hundreds, cosines in [-1, 1]) and only their *ordering*
of the runs is meaningful here.

Nothing is recomputed — both files must already exist. Run selection uses the same
``groups`` / ``experiments`` grammar as ``evaluate/plot/example_comparison_config.yaml``,
and the target is read through ``plot_ablation_benchmark.collect_model_metrics``, so
the ``metrics_old/`` fallback applies exactly as it does in the benchmark figure.

Usage
-----
    python evaluate/check/correlate_metrics_downstream.py --config cfg.yaml --no-show
    python evaluate/check/correlate_metrics_downstream.py --config cfg.yaml --list
    python evaluate/check/correlate_metrics_downstream.py --config cfg.yaml \\
        --scope within_group        # guard against between-experiment confounding

Reading the table
-----------------
Unlike ``plot_unified_metrics_table.py``, the colour here is an **absolute** scale:
rho runs from -1 (red) through 0 (white) to +1 (blue), so a pale row is a weak
correlation whatever the other rows do.

Two things to check before believing a row:

* **n**, printed per row. It varies: ``paired_*`` metrics are absent on datasets with
  no paired inputs, so those rows are correlated over far fewer runs than the rest.
* **q**, the Benjamini-Hochberg FDR-corrected p-value. Twenty-odd metrics are tested
  at once, so an uncorrected p < 0.05 is expected roughly once per figure by chance.

And note what a correlation over runs drawn from several ablation directories can
mean: those runs differ in training data and gene panel as well as in the metric, so
a pooled correlation may be carried entirely by between-experiment offsets rather
than by any within-experiment relationship. ``scope: within_group`` computes rho
inside each group and combines, which removes that.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate.plot.experiment_selection import (  # noqa: E402
    as_str_list,
    is_model_dir,
    load_raw_config,
    parse_figsize,
    parse_groups,
)
from evaluate.plot.plot_ablation_benchmark import (  # noqa: E402
    METRIC_LABELS as TARGET_METRIC_LABELS,
    TASK_LABELS,
    TASK_PRIMARY_METRIC,
    collect_model_metrics,
)
from evaluate.plot.plot_unified_metrics_table import (  # noqa: E402
    DOWN,
    METRIC_BY_KEY,
    NON_METRIC_KEYS,
    UP,
    _text_color,
    collect_unified_metrics,
    fmt_value,
)

# What is being predicted, when the config does not say.
DEFAULT_TARGET_TASK = "canc_type_class"

# Spearman over fewer runs than this is noise; those rows are drawn grey and their
# q-value is not computed. 5 is the smallest n at which a perfect monotone ranking is
# significant at the 5% level two-sided (p = 0.0167).
DEFAULT_MIN_RUNS = 5

SCOPES = ("pooled", "within_group")
SORTS = ("abs_rho", "rho", "config")

_NA_COLOR = (0.93, 0.93, 0.93, 1.0)
_PLAIN_COLOR = (1.0, 1.0, 1.0, 1.0)
_WEAK_COLOR = (0.90, 0.90, 0.90, 1.0)     # n below min_runs: no colour claim made


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

@dataclass
class Correlation:
    """One metric's relationship to the target, across the runs that have both."""

    key: str
    label: str
    direction: str
    rho: float = float("nan")
    p: float = float("nan")
    q: float = float("nan")          # Benjamini-Hochberg across the metrics tested
    n: int = 0                       # runs contributing
    n_groups: int = 0                # groups contributing (within_group scope only)
    note: str = ""                   # why it is nan, when it is

    @property
    def usable(self) -> bool:
        return math.isfinite(self.rho)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and its two-sided p-value; (nan, nan) when undefined.

    Undefined covers the two cases that actually occur here: fewer than three paired
    observations, and a constant column — every run scoring the same value has no
    ordering to correlate, which is not the same as "no relationship".
    """
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")

    from scipy.stats import spearmanr

    res = spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    return (rho, p) if math.isfinite(rho) else (float("nan"), float("nan"))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """FDR-adjusted p-values, NaN preserved in place.

    Applied because a figure tests every metric at once: with 20 columns, an
    uncorrected p < 0.05 turns up about once per figure with no real effect present.
    """
    idx = [i for i, p in enumerate(pvals) if math.isfinite(p)]
    out = [float("nan")] * len(pvals)
    if not idx:
        return out

    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    # Walk from the largest p down, carrying the running minimum, so the adjusted
    # values stay monotone in p.
    for rank, i in enumerate(reversed(order), start=1):
        adj = pvals[i] * m / (m - rank + 1)
        prev = min(prev, adj)
        out[i] = min(1.0, prev)
    return out


def combine_pvalues(pvals: list[float]) -> float:
    """Fisher's method over per-group p-values; NaN if none are usable."""
    usable = [p for p in pvals if math.isfinite(p) and p > 0.0]
    if not usable:
        return float("nan")
    if len(usable) == 1:
        return usable[0]

    from scipy.stats import chi2

    stat = -2.0 * float(np.sum(np.log(usable)))
    return float(chi2.sf(stat, 2 * len(usable)))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class CorrConfig:
    """A parsed correlation config (see :func:`load_config`)."""

    groups: list[tuple[str | None, list[tuple[str, Path]]]]
    target_task: str = DEFAULT_TARGET_TASK
    target_metric: str | None = None       # None -> TASK_PRIMARY_METRIC[task]
    metrics: list[str] = field(default_factory=list)   # [] -> everything present
    scope: str = "pooled"
    min_runs: int = DEFAULT_MIN_RUNS
    sort: str = "abs_rho"
    title: str | None = None
    output: Path | None = None
    figsize: tuple[float, float] | None = None
    cmap: str = "RdBu_r"
    dpi: int = 150

    @property
    def resolved_target_metric(self) -> str:
        if self.target_metric:
            return self.target_metric
        return TASK_PRIMARY_METRIC.get(self.target_task, "accuracy")


def is_correlation_model_dir(path: Path) -> bool:
    """A run is usable here only if it has the internal metrics to correlate."""
    return is_model_dir(path, patterns=("unified_metrics.json",))


def load_config(path: Path) -> CorrConfig:
    """
    Parse a YAML (or JSON) config.

    The ``groups:`` / ``experiments:`` half is shared with the other comparison
    scripts — see :func:`evaluate.plot.experiment_selection.parse_groups`. The rest::

        target:
          task:   canc_type_class    # results_{task}.json
          metric: accuracy           # omitted -> the task's primary metric

        metrics:                     # omitted -> every internal metric present
          - contrastive_energy_distance
          - geometry_pr_pooled

        scope:     pooled            # pooled | within_group
        min_runs:  5                 # rows with fewer paired runs are not scored
        sort:      abs_rho           # abs_rho | rho | config

        title:   "..."
        output:  figures/corr.png    # optional; else <config>.png
        cmap:    RdBu_r
        figsize: [9, 12]
        dpi:     150
    """
    raw = load_raw_config(path)
    groups = parse_groups(raw, path, is_correlation_model_dir)

    target_raw = raw.get("target") or {}
    if not isinstance(target_raw, dict):
        sys.exit(f"ERROR: 'target' in {path} must be a mapping (or omitted).")
    task = str(target_raw.get("task") or DEFAULT_TARGET_TASK)
    if task not in TASK_LABELS:
        print(
            f"[warning] target task '{task}' is not one of "
            f"{', '.join(sorted(TASK_LABELS))} — it will only work if "
            f"results_{task}.json exists.",
            file=sys.stderr,
        )

    scope = str(raw.get("scope") or "pooled")
    if scope not in SCOPES:
        sys.exit(f"ERROR: 'scope' in {path} must be one of {', '.join(SCOPES)}.")

    sort = str(raw.get("sort") or "abs_rho")
    if sort not in SORTS:
        sys.exit(f"ERROR: 'sort' in {path} must be one of {', '.join(SORTS)}.")

    min_runs = int(raw.get("min_runs") or DEFAULT_MIN_RUNS)
    if min_runs < 3:
        sys.exit(f"ERROR: 'min_runs' in {path} must be at least 3.")

    return CorrConfig(
        groups=groups,
        target_task=task,
        target_metric=(str(target_raw["metric"]) if target_raw.get("metric") else None),
        metrics=as_str_list(raw.get("metrics"), "metrics"),
        scope=scope,
        min_runs=min_runs,
        sort=sort,
        title=str(raw["title"]) if raw.get("title") else None,
        output=Path(raw["output"]).expanduser() if raw.get("output") else None,
        figsize=parse_figsize(raw, path),
        cmap=str(raw.get("cmap") or "RdBu_r"),
        dpi=int(raw.get("dpi") or 150),
    )


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def collect(
    config: CorrConfig,
) -> tuple[dict[str, dict], dict[str, float], list[tuple[str | None, list[str]]]]:
    """Read every selected run.

    Returns ``(internal, target, groups)``: the internal metric dict per run, the
    target value per run, and the plot-order grouping. A run missing either side is
    dropped with a warning — correlating over a run that has no target value would
    silently change which runs each row describes.
    """
    internal: dict[str, dict] = {}
    target: dict[str, float] = {}
    groups: list[tuple[str | None, list[str]]] = []

    metric_name = config.resolved_target_metric

    for group_name, members in config.groups:
        kept: list[str] = []
        for name, model_dir in members:
            data = collect_unified_metrics(model_dir)
            if not data:
                print(f"[skip] {name}: no unified_metrics.json", file=sys.stderr)
                continue

            per_task = collect_model_metrics(model_dir)
            task_results = per_task.get(config.target_task)
            if not task_results:
                print(
                    f"[skip] {name}: no results_{config.target_task}.json",
                    file=sys.stderr,
                )
                continue
            value = task_results.get(metric_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                print(
                    f"[skip] {name}: results_{config.target_task}.json has no "
                    f"numeric '{metric_name}'",
                    file=sys.stderr,
                )
                continue

            internal[name] = data
            target[name] = float(value)
            kept.append(name)

        if kept:
            groups.append((group_name, kept))

    return internal, target, groups


def available_metrics(internal: dict[str, dict]) -> list[str]:
    """Every numeric internal metric present in at least one run, in catalogue order."""
    present: set[str] = set()
    for data in internal.values():
        for key, value in data.items():
            if key in NON_METRIC_KEYS or key.startswith("_"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            present.add(key)

    catalogued = [m.key for m in METRIC_BY_KEY.values() if m.key in present]
    extra = sorted(present - set(catalogued))
    return catalogued + extra


def paired_values(
    internal: dict[str, dict],
    target: dict[str, float],
    key: str,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """The (metric, target) pairs over *names* that have a finite value for both."""
    xs: list[float] = []
    ys: list[float] = []
    for name in names:
        v = internal.get(name, {}).get(key)
        t = target.get(name)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if t is None or not math.isfinite(float(v)) or not math.isfinite(t):
            continue
        xs.append(float(v))
        ys.append(float(t))
    return np.asarray(xs), np.asarray(ys)


def correlate(
    internal: dict[str, dict],
    target: dict[str, float],
    groups: list[tuple[str | None, list[str]]],
    keys: list[str],
    config: CorrConfig,
) -> list[Correlation]:
    """One :class:`Correlation` per key, in *keys* order, with BH-adjusted q-values."""
    all_names = [name for _, names in groups for name in names]
    rows: list[Correlation] = []

    for key in keys:
        spec = METRIC_BY_KEY.get(key)
        row = Correlation(
            key=key,
            label=(spec.label.replace("\n", " ") if spec else key),
            direction=(spec.direction if spec else UP),
        )

        if config.scope == "pooled":
            x, y = paired_values(internal, target, key, all_names)
            row.n = len(x)
            if row.n < config.min_runs:
                row.note = f"only {row.n} run(s)"
            else:
                row.rho, row.p = spearman(x, y)
                if not row.usable:
                    row.note = "constant across runs"
        else:
            rhos: list[float] = []
            ps: list[float] = []
            total = 0
            for _, names in groups:
                x, y = paired_values(internal, target, key, names)
                if len(x) < config.min_runs:
                    continue
                rho, p = spearman(x, y)
                if math.isfinite(rho):
                    rhos.append(rho)
                    ps.append(p)
                    total += len(x)
            row.n = total
            row.n_groups = len(rhos)
            if not rhos:
                row.note = f"no group with >= {config.min_runs} usable runs"
            else:
                row.rho = float(np.mean(rhos))
                row.p = combine_pvalues(ps)

        rows.append(row)

    qs = benjamini_hochberg([r.p for r in rows])
    for row, q in zip(rows, qs):
        row.q = q
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _arrow(direction: str) -> str:
    return {UP: " ↑", DOWN: " ↓", "none": ""}[direction]


def sort_rows(rows: list[Correlation], how: str) -> list[Correlation]:
    """Order the table. Unusable rows always sink to the bottom."""
    if how == "config":
        return rows

    def key(r: Correlation):
        if not r.usable:
            return (1, 0.0)
        return (0, -(abs(r.rho) if how == "abs_rho" else r.rho))

    return sorted(rows, key=key)


def write_csv(output: Path, rows: list[Correlation], config: CorrConfig) -> Path:
    """The numbers behind the figure, beside it, as ``{figure stem}.csv``."""
    path = output.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["metric", "label", "direction", "spearman_rho", "p_value",
             "q_value_bh", "n_runs", "n_groups", "scope", "target", "note"]
        )
        target = f"{config.target_task}:{config.resolved_target_metric}"
        for r in rows:
            writer.writerow([
                r.key, r.label, r.direction,
                "" if not math.isfinite(r.rho) else r.rho,
                "" if not math.isfinite(r.p) else r.p,
                "" if not math.isfinite(r.q) else r.q,
                r.n, r.n_groups if config.scope == "within_group" else "",
                config.scope, target, r.note,
            ])
    print(f"Saved to {path}")
    return path


def plot_table(
    rows: list[Correlation],
    config: CorrConfig,
    n_runs: int,
    n_groups: int,
    output: Path | None,
    show: bool,
) -> None:
    """Draw the correlation table."""
    within = config.scope == "within_group"
    col_labels = ["Spearman ρ", "p", "q (BH)", "n runs"]
    if within:
        col_labels.append("groups")

    n_rows, n_cols = len(rows), len(col_labels)
    cmap = plt.get_cmap(config.cmap)

    rgba = np.zeros((n_rows, n_cols, 4))
    texts: list[list[str]] = []
    for i, r in enumerate(rows):
        if r.usable:
            # Absolute scale: -1 -> 0 -> +1 maps onto the full colormap, so a pale
            # cell means a weak correlation regardless of the other rows. This is the
            # opposite convention to plot_unified_metrics_table, where colour is a
            # within-column rank — rho is comparable across figures and ranks are not.
            rgba[i, 0] = cmap((r.rho + 1.0) / 2.0)
        else:
            rgba[i, 0] = _NA_COLOR if r.n else _NA_COLOR
        if r.usable and r.n < config.min_runs:
            rgba[i, 0] = _WEAK_COLOR
        rgba[i, 1:] = _PLAIN_COLOR

        star = ""
        if math.isfinite(r.q):
            star = "**" if r.q < 0.01 else ("*" if r.q < 0.05 else "")
        texts.append([
            (fmt_value(r.rho) + star) if r.usable else "n/a",
            _fmt_p(r.p),
            _fmt_p(r.q),
            str(r.n),
            *([str(r.n_groups)] if within else []),
        ])

    row_labels = [r.label + _arrow(r.direction) for r in rows]

    label_w = max((len(lab) for lab in row_labels), default=20)
    fig_w = config.figsize[0] if config.figsize else min(
        max(0.085 * label_w + 1.1 * n_cols + 1.6, 7.5), 16.0
    )
    fig_h = config.figsize[1] if config.figsize else max(0.40 * n_rows + 2.6, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j, i, texts[i][j], ha="center", va="center", fontsize=8,
                color=_text_color(tuple(rgba[i, j])),
                fontweight="bold" if j == 0 and texts[i][0].endswith("*") else "normal",
            )

    target_label = TARGET_METRIC_LABELS.get(
        config.resolved_target_metric, config.resolved_target_metric
    )
    task_label = TASK_LABELS.get(config.target_task, config.target_task).replace("\n", " ")
    title = config.title or f"Internal metrics vs {task_label} {target_label.lower()}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.04, aspect=45, shrink=0.6)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["ρ = -1", "0", "ρ = +1"])
    cbar.ax.tick_params(labelsize=7.5, length=0)
    cbar.set_label("absolute scale — pale means weakly correlated",
                   fontsize=7.5, color="dimgrey")
    cbar.outline.set_visible(False)

    scope_note = (
        f"ρ averaged within {n_groups} group(s), p combined (Fisher)"
        if within else f"pooled over {n_runs} run(s)"
    )
    notes = [
        scope_note,
        "* q < 0.05, ** q < 0.01 (BH across the rows shown)",
        "a ↓ metric should correlate negatively if it predicts the target",
        "n varies by row: a metric absent from a run drops that run",
    ]
    fig.text(0.01, 0.005, "  ·  ".join(notes), ha="left", va="bottom",
             fontsize=7, color="dimgrey", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=config.dpi)
        print(f"Saved to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spearman-correlate the internal unified-FM metrics against a downstream "
            "task result, as an annotated table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="YAML config selecting the runs. See load_config().")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Figure path. Defaults to the config's 'output:', "
                             "else the config path with a .png suffix.")
    parser.add_argument("--task", type=str, default=None,
                        help="Override the config's target.task.")
    parser.add_argument("--target-metric", type=str, default=None,
                        help="Override the config's target.metric.")
    parser.add_argument("--scope", choices=SCOPES, default=None,
                        help="Override the config's 'scope:'.")
    parser.add_argument("--sort", choices=SORTS, default=None,
                        help="Override the config's 'sort:'.")
    parser.add_argument("--min-runs", type=int, default=None,
                        help="Override the config's 'min_runs:'.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"),
                        default=None, help="Figure size in inches.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open an interactive window.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Do not write the numbers beside the figure.")
    parser.add_argument("--list", action="store_true",
                        help="List the correlatable metrics and the target values, "
                             "then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path: Path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"ERROR: --config does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    if args.task:
        config.target_task = args.task
    if args.target_metric:
        config.target_metric = args.target_metric
    if args.scope:
        config.scope = args.scope
    if args.sort:
        config.sort = args.sort
    if args.min_runs:
        config.min_runs = args.min_runs
    if args.figsize:
        config.figsize = (args.figsize[0], args.figsize[1])

    target_metric = config.resolved_target_metric
    print(
        f"Loading runs from {config_path} ...\n"
        f"  target: results_{config.target_task}.json['{target_metric}']"
    )

    internal, target, groups = collect(config)
    if not internal:
        print(
            "No run has both unified_metrics.json and "
            f"results_{config.target_task}.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    for group_name, names in groups:
        print(f"  {group_name or '(ungrouped)'}: {', '.join(names)}")

    keys = config.metrics or available_metrics(internal)
    if config.metrics:
        present = set(available_metrics(internal))
        missing = [k for k in keys if k not in present]
        for key in missing:
            print(f"[warning] metric '{key}' not found in any run — ignoring.",
                  file=sys.stderr)
        keys = [k for k in keys if k in present]

    if args.list:
        print(f"\n{len(internal)} run(s) with both files:\n")
        width = max((len(n) for n in internal), default=10)
        for name in internal:
            print(f"  {name:<{width}}  {target_metric} = {target[name]:.4f}")
        print(f"\n{len(keys)} correlatable metric(s):\n")
        for key in keys:
            spec = METRIC_BY_KEY.get(key)
            n = len(paired_values(internal, target, key,
                                  [n for _, ns in groups for n in ns])[0])
            fam = spec.family if spec else "Other"
            print(f"  {key:<44} n={n:<4} {fam}")
        return

    if not keys:
        print("No correlatable metrics in the selected runs.", file=sys.stderr)
        sys.exit(1)

    rows = sort_rows(correlate(internal, target, groups, keys, config), config.sort)
    n_runs = len(internal)
    scored = sum(1 for r in rows if r.usable)
    print(f"{n_runs} run(s), {len(rows)} metric(s), {scored} scored, "
          f"scope '{config.scope}'.")

    output: Path | None = args.output or config.output or config_path.with_suffix(".png")
    if output is not None and not args.no_csv:
        write_csv(output, rows, config)

    plot_table(rows, config, n_runs, len(groups), output, show=not args.no_show)


if __name__ == "__main__":
    main()
