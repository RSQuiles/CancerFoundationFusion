"""
Cross-experiment comparison of the *internal* (unified-FM) metrics.

Where ``plot_ablation_benchmark.py`` compares runs on the downstream task results
(``metrics/results_<task>.json``), this compares them on the metrics computed by
``evaluate/check/unified_metrics.py`` — reconstruction, paired alignment, aggregation
consistency and contrastive/distributional — read from

    {model_dir}/metrics/unified_metrics.json

and, optionally, the scIB batch-integration tables written alongside them

    {ablation_dir}/_scib_metrics/scib_{tag}.csv

Nothing is recomputed.  A YAML config selects individual runs — possibly from
different ablation directories — gives them display names and arranges them into
groups, using exactly the same grammar as ``example_comparison_config.yaml``.

Alongside the figure a ``{stem}.csv`` is written holding the same numbers — one row
per run, one column per plotted metric.  ``--no-csv`` turns that off.

The default figure is an **annotated heatmap**: one row per run, one column per
metric, the raw value printed in each cell and the cell coloured by a direction-aware
per-column normalisation (green = best in that column).  ``style: rank_table`` and
``style: bars`` render the same selection differently.

Usage
-----
    # What metrics do the selected runs actually have?
    python evaluate/plot/plot_unified_metrics_table.py --config cfg.yaml --list

    # The default annotated heatmap
    python evaluate/plot/plot_unified_metrics_table.py --config cfg.yaml --no-show

    # Same config, other styles
    python evaluate/plot/plot_unified_metrics_table.py --config cfg.yaml \\
        --style rank_table --output ranks.png --no-show
    python evaluate/plot/plot_unified_metrics_table.py --config cfg.yaml \\
        --style bars --output bars.png --no-show

Reading the colours
-------------------
Colour is **relative within a column**, never absolute: the best run in a column is
always fully green and the worst fully red, however small the gap between them.  With
only two runs selected the two extremes are therefore guaranteed.  Read the printed
numbers for magnitude and the colours for ordering only.
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

from evaluate.plot.experiment_selection import (  # noqa: E402
    GROUP_SHADES,
    as_str_list,
    group_color,
    grouped_layout,
    is_model_dir,
    load_raw_config,
    parse_figsize,
    parse_groups,
)

# --------------------------------------------------------------------------- #
# Metric catalogue
# --------------------------------------------------------------------------- #

UP, DOWN, NONE = "up", "down", "none"


@dataclass(frozen=True)
class MetricSpec:
    """One column of the table."""

    key: str          # key in unified_metrics.json, or "scib:<tag>:<metric>"
    label: str        # column header (an ↑/↓ arrow is appended at draw time)
    direction: str    # UP | DOWN | NONE
    family: str       # for the family brackets under the grid


# The curated catalogue.  Keys and directions mirror the label tables already kept by
# hand in unified_metrics._plot_metrics and compare_experiments._METRIC_META; the
# labels here are shorter because they head a column rather than a subplot.
#
# NONE marks metrics that genuinely have no better/worse direction: the two within-
# modality cosines are spread diagnostics, and paired_random_baseline_cosine is the
# reference level that paired_cosine_sim_mean should be read against.  Colouring them
# would invent a preference the metric does not have, so they are drawn grey.
METRIC_META: list[MetricSpec] = [
    # ── Reconstruction ─────────────────────────────────────────────────────
    MetricSpec("recon_pearson_r",                 "Pearson R",        UP,   "Reconstruction"),
    MetricSpec("recon_mae_bins",                  "MAE (bins)",       DOWN, "Reconstruction"),
    # ── Paired alignment ───────────────────────────────────────────────────
    MetricSpec("paired_cosine_sim_mean",          "Cosine sim",       UP,   "Paired alignment"),
    MetricSpec("paired_rank_mean",                "Rank (cos)",       DOWN, "Paired alignment"),
    MetricSpec("paired_l2_mean",                  "L2 dist",          DOWN, "Paired alignment"),
    MetricSpec("paired_rank_l2_mean",             "Rank (L2)",        DOWN, "Paired alignment"),
    MetricSpec("paired_random_baseline_cosine",   "Random cos\nbaseline", NONE, "Paired alignment"),
    # ── Aggregation consistency ────────────────────────────────────────────
    MetricSpec("agg_paired_cosine_pb_to_mean_sc", "Paired cos",       UP,   "Aggregation"),
    MetricSpec("agg_synth_cosine_pb_to_mean_sc",  "Synth cos",        UP,   "Aggregation"),
    MetricSpec("agg_paired_l2_pb_to_mean_sc",     "Paired L2",        DOWN, "Aggregation"),
    MetricSpec("agg_synth_l2_pb_to_mean_sc",      "Synth L2",         DOWN, "Aggregation"),
    # ── Contrastive ────────────────────────────────────────────────────────
    MetricSpec("contrastive_cross_cosine_mean",   "Cross cos",        UP,   "Contrastive"),
    MetricSpec("contrastive_within_bulk_cosine",  "Within-bulk\ncos", NONE, "Contrastive"),
    MetricSpec("contrastive_within_pb_cosine",    "Within-PB\ncos",   NONE, "Contrastive"),
    MetricSpec("contrastive_cross_l2_mean",       "Cross L2",         DOWN, "Contrastive"),
    # Spread, not distance: a near-zero spread means the embedding collapsed.
    MetricSpec("contrastive_within_bulk_l2",      "Within-bulk\nL2",  UP,   "Contrastive"),
    MetricSpec("contrastive_within_pb_l2",        "Within-PB\nL2",    UP,   "Contrastive"),
    MetricSpec("contrastive_wasserstein",         "Wasserstein",      DOWN, "Contrastive"),
    # Computed by unified_metrics.py but plotted by nothing else; off by default.
    MetricSpec("contrastive_mmd",                 "MMD",              DOWN, "Contrastive"),
]

METRIC_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in METRIC_META}

# The default column set: the same 17 metrics the existing bar grids show, in family
# order, so the default figure and unified_metrics.png agree on what is being compared.
DEFAULT_METRICS: list[str] = [m.key for m in METRIC_META if m.key != "contrastive_mmd"]

# Bookkeeping / provenance keys — never plottable, hidden from --list.
NON_METRIC_KEYS: set[str] = {
    "model", "checkpoint", "panel_hash", "shared_panel", "panel_strategy",
    "recon_mae_per_bin", "skipped_families", "modalities",
    "contrastive_bulk_source", "contrastive_pb_source",
}

# --------------------------------------------------------------------------- #
# scIB
# --------------------------------------------------------------------------- #

SCIB_TAGS: dict[str, str] = {
    "bulk_vs_pb":                "bulk vs PB",
    "paired_vs_nonpaired":       "paired vs non-paired",
    "paired_bulk_vs_paired_pb":  "paired bulk vs PB",
    "nonpaired_bulk_vs_synth_pb": "bulk vs synth PB",
}

# Every scIB score in the table is higher-is-better.
SCIB_METRICS: tuple[str, ...] = (
    "Isolated labels", "Silhouette label", "cLISI",
    "BRAS", "iLISI", "PCR comparison",
    "Batch correction", "Bio conservation", "Total",
)

# Only iLISI is a global, label-free mixing measure; see the caveat block in
# unified_metrics.run_scib_benchmark's docstring.
DEFAULT_SCIB_METRICS: tuple[str, ...] = ("iLISI",)

_SCIB_CACHE: dict[tuple[str, str], dict[str, dict[str, float]]] = {}


def _read_scib_table(ablation_dir: Path, tag: str) -> dict[str, dict[str, float]]:
    """``{model_name: {scib_metric: value}}`` for one ablation dir and tag.

    The CSV is indexed by obsm key (``X_cf_<model>``) and carries an extra
    ``Metric Type`` row that is not a model; both are handled here so callers can
    look models up by their directory name.  Returns ``{}`` when the file is absent.
    """
    cache_key = (str(ablation_dir), tag)
    if cache_key in _SCIB_CACHE:
        return _SCIB_CACHE[cache_key]

    csv_path = ablation_dir / "_scib_metrics" / f"scib_{tag}.csv"
    table: dict[str, dict[str, float]] = {}
    if csv_path.is_file():
        import pandas as pd

        try:
            df = pd.read_csv(csv_path, index_col=0)
            # The trailing 'Metric Type' row is a string label, which makes pandas
            # type every column as object — so drop the row first, then coerce, or
            # a dtype filter would throw the whole table away.
            df = df.drop(index="Metric Type", errors="ignore")
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            for idx, row in df.iterrows():
                name = str(idx)
                if name.startswith("X_cf_"):
                    name = name[len("X_cf_"):]
                table[name] = {
                    str(c): float(v) for c, v in row.items() if pd.notna(v)
                }
        except Exception as exc:
            print(f"[warning] could not read {csv_path}: {exc}", file=sys.stderr)
    else:
        print(f"[warning] no scIB table at {csv_path} - skipping.", file=sys.stderr)

    _SCIB_CACHE[cache_key] = table
    return table


def scib_spec(tag: str, metric: str) -> MetricSpec:
    """The MetricSpec for one scIB column."""
    short = SCIB_TAGS.get(tag, tag)
    return MetricSpec(
        key=f"scib:{tag}:{metric}",
        label=f"{metric}\n({short})",
        direction=UP,
        family=f"scIB · {short}",
    )


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def is_unified_model_dir(path: Path) -> bool:
    """True if *path* has the unified metrics JSON this script reads."""
    return is_model_dir(path, patterns=("unified_metrics.json",))


def collect_unified_metrics(model_dir: Path) -> dict:
    """Read ``{model_dir}/metrics/unified_metrics.json`` into a flat dict."""
    jf = model_dir / "metrics" / "unified_metrics.json"
    try:
        with open(jf) as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[warning] could not read {jf}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[warning] {jf} is not a JSON object - skipping.", file=sys.stderr)
        return {}
    # Nested entries have no place in a flat table; they stay in the JSON.
    return {k: v for k, v in data.items() if not isinstance(v, (dict, list))}


def collect_scib_metrics(
    model_dir: Path,
    tags: list[str],
    metrics: list[str],
) -> dict[str, float]:
    """scIB values for one model, keyed ``scib:<tag>:<metric>``.

    The ablation directory is the model directory's parent, and the scIB tables are
    indexed by the model *directory* name — never the config's display name.
    """
    out: dict[str, float] = {}
    for tag in tags:
        table = _read_scib_table(model_dir.parent, tag)
        row = table.get(model_dir.name)
        if row is None:
            if table:
                print(
                    f"[warning] '{model_dir.name}' absent from scib_{tag}.csv in "
                    f"{model_dir.parent.name} - leaving those cells empty.",
                    file=sys.stderr,
                )
            continue
        for metric in metrics:
            if metric in row:
                out[f"scib:{tag}:{metric}"] = row[metric]
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

STYLES = ("heatmap", "rank_table", "bars")
NORMALIZERS = ("minmax", "zscore", "rank")


@dataclass
class TableConfig:
    """A parsed unified-metrics comparison config (see :func:`load_config`)."""

    groups: list[tuple[str | None, list[tuple[str, Path]]]]
    title: str = "Unified metrics"
    output: Path | None = None
    figsize: tuple[float, float] | None = None
    style: str = "heatmap"
    normalize: str = "minmax"
    cmap: str = "RdYlGn"
    show_families: bool = True
    transpose: bool = False
    annotate: bool = True
    dpi: int = 150
    metrics: list[str] = field(default_factory=list)          # [] -> DEFAULT_METRICS
    scib_tags: list[str] = field(default_factory=list)        # [] -> no scIB columns
    scib_metrics: list[str] = field(default_factory=list)

    @property
    def grouped(self) -> bool:
        return any(name is not None for name, _ in self.groups)


def load_config(path: Path) -> TableConfig:
    """
    Parse a YAML (or JSON) config.

    The ``groups:`` / ``experiments:`` half is identical to
    ``plot_ablation_benchmark.py``'s config — see
    :func:`evaluate.plot.experiment_selection.parse_groups`.  The rest::

        title:   "Internal metrics"
        output:  figures/unified_table.png   # optional; else <config>.png
        figsize: [18, 7]                     # optional, inches
        dpi:     150

        style:     heatmap     # heatmap | rank_table | bars
        normalize: minmax      # minmax | zscore | rank   (colour only)
        cmap:      RdYlGn
        show_families: true    # family brackets under the grid
        transpose: false       # true -> metrics as rows, runs as columns
        annotate:  true        # print the value in each cell

        metrics:               # optional; omitted -> the curated default set
          - recon_pearson_r
          - contrastive_wasserstein

        scib:                  # optional; omitted -> no scIB columns
          tags:    [bulk_vs_pb]
          metrics: [iLISI]
    """
    raw = load_raw_config(path)
    groups = parse_groups(raw, path, is_unified_model_dir)

    style = str(raw.get("style") or "heatmap")
    if style not in STYLES:
        sys.exit(f"ERROR: 'style' in {path} must be one of {', '.join(STYLES)}.")

    normalize = str(raw.get("normalize") or "minmax")
    if normalize not in NORMALIZERS:
        sys.exit(
            f"ERROR: 'normalize' in {path} must be one of {', '.join(NORMALIZERS)}."
        )

    scib_raw = raw.get("scib") or {}
    if scib_raw is True:                     # `scib: true` -> sensible defaults
        scib_raw = {}
    if not isinstance(scib_raw, dict):
        sys.exit(f"ERROR: 'scib' in {path} must be a mapping (or omitted).")
    scib_tags = as_str_list(scib_raw.get("tags"), "scib.tags") if scib_raw else []
    if scib_raw and not scib_tags:
        scib_tags = ["bulk_vs_pb"]
    for tag in scib_tags:
        if tag not in SCIB_TAGS:
            sys.exit(
                f"ERROR: unknown scib tag '{tag}' in {path}. "
                f"Known tags: {', '.join(SCIB_TAGS)}."
            )
    scib_metrics = as_str_list(scib_raw.get("metrics"), "scib.metrics")
    if scib_tags and not scib_metrics:
        scib_metrics = list(DEFAULT_SCIB_METRICS)
    for metric in scib_metrics:
        if metric not in SCIB_METRICS:
            print(
                f"[warning] '{metric}' is not a known scIB column "
                f"({', '.join(SCIB_METRICS)}) - it will only appear if present.",
                file=sys.stderr,
            )

    return TableConfig(
        groups=groups,
        title=str(raw.get("title") or "Unified metrics"),
        output=Path(raw["output"]).expanduser() if raw.get("output") else None,
        figsize=parse_figsize(raw, path),
        style=style,
        normalize=normalize,
        cmap=str(raw.get("cmap") or "RdYlGn"),
        show_families=bool(raw.get("show_families", True)),
        transpose=bool(raw.get("transpose", False)),
        annotate=bool(raw.get("annotate", True)),
        dpi=int(raw.get("dpi") or 150),
        metrics=as_str_list(raw.get("metrics"), "metrics"),
        scib_tags=scib_tags,
        scib_metrics=scib_metrics,
    )


def collect_from_config(
    config: TableConfig,
) -> tuple[dict[str, dict], list[tuple[str | None, list[str]]]]:
    """
    Load metrics for every run named in *config*.

    Returns ``(results, groups)`` where ``results[display_name]`` is the flat metric
    dict and ``groups`` is the plot-order ``(group_name, [display_name, ...])``.
    Runs with no readable metrics are dropped with a warning.
    """
    results: dict[str, dict] = {}
    groups: list[tuple[str | None, list[str]]] = []

    for group_name, members in config.groups:
        names: list[str] = []
        for name, model_dir in members:
            if not is_unified_model_dir(model_dir):
                print(
                    f"[warning] no metrics/unified_metrics.json under {model_dir} "
                    f"- skipping '{name}'.",
                    file=sys.stderr,
                )
                continue
            data = collect_unified_metrics(model_dir)
            if not data:
                print(f"[warning] no metrics for '{name}' - skipping.", file=sys.stderr)
                continue
            if config.scib_tags:
                data.update(
                    collect_scib_metrics(model_dir, config.scib_tags, config.scib_metrics)
                )
            data["_model_dir"] = str(model_dir)
            results[name] = data
            names.append(name)

        if names:
            groups.append((group_name, names))

    return results, groups


def resolve_metrics(
    results: dict[str, dict],
    config: TableConfig,
) -> list[MetricSpec]:
    """Which columns to draw, in order.

    An explicit ``metrics:`` list is honoured verbatim (order included); otherwise the
    curated default set is used, minus metrics no selected run has.  scIB columns are
    appended after the internal ones either way.
    """
    present = {k for data in results.values() for k, v in data.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}

    if config.metrics:
        specs: list[MetricSpec] = []
        for key in config.metrics:
            if key not in present:
                print(
                    f"[warning] metric '{key}' not found in any selected run "
                    "- ignoring.",
                    file=sys.stderr,
                )
                continue
            specs.append(
                METRIC_BY_KEY.get(key)
                or MetricSpec(key, key, UP, "Other")
            )
    else:
        specs = [METRIC_BY_KEY[k] for k in DEFAULT_METRICS if k in present]

    # scIB columns, in the config's tag/metric order.
    if config.scib_tags:
        chosen = {s.key for s in specs}
        for tag in config.scib_tags:
            for metric in config.scib_metrics:
                key = f"scib:{tag}:{metric}"
                if key in present and key not in chosen:
                    specs.append(scib_spec(tag, metric))

    return specs


def csv_output_path(output: Path) -> Path:
    """The CSV that accompanies a figure: same directory, same stem, ``.csv``."""
    return output.with_suffix(".csv")


def write_table_csv(
    output: Path,
    results: dict[str, dict],
    specs: list[MetricSpec],
    groups: list[tuple[str | None, list[str]]],
) -> Path:
    """Write the numbers behind the figure next to it, as ``{figure stem}.csv``.

    One row per run in plot order, one column per plotted metric, keyed by the
    metric's ``key`` rather than its column header — the header is shortened for
    the figure, and scIB columns keep their ``scib:<tag>:<metric>`` form so they
    are unambiguous. Values are the raw numbers, not the within-column colour
    scores, and the rank table's derived "Mean rank" column is not included.
    """
    group_of: dict[str, str] = {}
    for group_name, members in groups:
        for name in members:
            group_of[name] = group_name or ""

    path = csv_output_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "run"] + [s.key for s in specs])
        for _, members in groups:
            for name in members:
                row: list = [group_of.get(name, ""), name]
                for spec in specs:
                    value = results.get(name, {}).get(spec.key)
                    row.append(
                        value if isinstance(value, (int, float))
                        and not isinstance(value, bool) else ""
                    )
                writer.writerow(row)

    print(f"Saved to {path}")
    return path


def warn_panel_mismatch(results: dict[str, dict]) -> str | None:
    """Warn when the runs were not all evaluated under the same gene panel.

    ``unified_metrics.py`` keys its metric cache on ``panel_hash`` precisely because
    metrics computed under different panels are not comparable.  Comparing across
    panels stays possible — it is flagged, not blocked.
    """
    by_hash: dict[str, list[str]] = {}
    for name, data in results.items():
        h = data.get("panel_hash")
        if h:
            by_hash.setdefault(str(h), []).append(name)

    if len(by_hash) <= 1:
        return None

    print(
        "[warning] the selected runs do not share a gene panel - their metrics are "
        "not strictly comparable:",
        file=sys.stderr,
    )
    for h, names in by_hash.items():
        # Abbreviate only what is too long to read; truncating short hashes would
        # make distinct panels look identical in the message.
        short = h if len(h) <= 20 else h[:16] + "..."
        print(f"[warning]   panel {short}: {', '.join(names)}", file=sys.stderr)
    return f"⚠ {len(by_hash)} different gene panels among these runs — see stderr"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def dense_ranks(values: np.ndarray, direction: str) -> np.ndarray:
    """Average ranks with 1 = best; NaN where the value is missing."""
    ranks = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return ranks

    x = values[finite].astype(float)
    if direction == DOWN:
        x = -x

    order = np.argsort(-x, kind="stable")
    sorted_x = x[order]
    out = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    ranks[finite] = out
    return ranks


def score_column(values: np.ndarray, direction: str, mode: str) -> np.ndarray:
    """Map one column onto ``[0, 1]`` for the colour fill; NaN = draw grey.

    The score is only ever used for colour — the printed text is the raw value.
    Degenerate columns (a single run, or every run equal) score 0.5 throughout,
    because there is no ordering to show.
    """
    scores = np.full(len(values), np.nan)
    if direction == NONE:
        return scores

    finite = np.isfinite(values)
    if not finite.any():
        return scores

    x = values[finite].astype(float)
    if direction == DOWN:
        x = -x

    if len(x) == 1 or np.allclose(x, x[0]):
        scores[finite] = 0.5
        return scores

    if mode == "minmax":
        s = (x - x.min()) / (x.max() - x.min())
    elif mode == "zscore":
        z = (x - x.mean()) / x.std()
        s = 1.0 / (1.0 + np.exp(-z))
    elif mode == "rank":
        r = dense_ranks(values, direction)[finite]
        s = (len(x) - r) / (len(x) - 1)
    else:                                    # unreachable; load_config validates
        raise ValueError(f"unknown normalisation '{mode}'")

    scores[finite] = np.clip(s, 0.0, 1.0)
    return scores


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def fmt_value(v: float | None) -> str:
    """Adaptive number formatting - enough digits to separate runs, no more."""
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return "n/a"
    v = float(v)
    a = abs(v)
    if a == 0.0:
        return "0"
    if a >= 1e4 or a < 1e-3:
        return f"{v:.1e}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def arrow(spec: MetricSpec) -> str:
    return {UP: " ↑", DOWN: " ↓", NONE: ""}[spec.direction]


def _text_color(rgba: tuple) -> str:
    """Black or white, whichever stays legible on *rgba*."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Contiguous runs of equal labels as ``(lo, hi, label)`` index spans."""
    spans: list[tuple[int, int, str]] = []
    for i, lab in enumerate(labels):
        if spans and spans[-1][2] == lab:
            lo, _, name = spans[-1]
            spans[-1] = (lo, i, name)
        else:
            spans.append((i, i, lab))
    return spans


# --------------------------------------------------------------------------- #
# Grid rendering
# --------------------------------------------------------------------------- #

_NA_COLOR = (0.93, 0.93, 0.93, 1.0)        # missing value
_NEUTRAL_COLOR = (0.82, 0.82, 0.82, 1.0)   # metric with no better/worse direction
_ROW_LABEL_FONT = 9


def _draw_grid(
    ax,
    rgba: np.ndarray,
    texts: list[list[str]],
    row_labels: list[str],
    col_labels: list[str],
    row_bands: list[tuple[int, int, str, tuple]],
    col_bands: list[tuple[int, int, str]],
    annotate: bool,
) -> None:
    """Draw one annotated grid.

    ``rgba`` is (n_rows, n_cols, 4).  ``row_bands`` are drawn as a coloured bracket in
    the left margin (used for experiment groups), ``col_bands`` as a bracket under the
    grid (used for metric families).  Which of the two carries groups vs families
    depends on ``transpose``; the caller decides.
    """
    n_rows, n_cols = len(row_labels), len(col_labels)

    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(n_cols))
    # Column headers are rotated, where a line break reads as a gap rather than as
    # wrapping — so they are always flattened to one line.  Row labels keep theirs.
    ax.set_xticklabels(
        [lab.replace("\n", " ") for lab in col_labels],
        rotation=45, ha="left", fontsize=8,
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=_ROW_LABEL_FONT)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White gridlines between cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    # The minor ticks would otherwise let the axes grow, so the separator lines
    # below would run past the last row of cells.
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, texts[i][j],
                    ha="center", va="center", fontsize=7.5,
                    color=_text_color(tuple(rgba[i, j])),
                )

    # Bracket in the left margin, beyond the row labels.  Its offset has to be in
    # inches rather than data units: one data unit is one column wide, so on a
    # 20-column grid a data-unit offset would put the bracket on top of the labels.
    label_in = _ROW_LABEL_FONT * 0.60 / 72.0 * max(
        (max(len(part) for part in lab.split("\n")) for lab in row_labels), default=4
    )
    band_tr = blended_transform_factory(ax.transAxes, ax.transData) + ScaledTranslation(
        -(label_in + 0.18), 0, ax.figure.dpi_scale_trans
    )

    for lo, hi, name, color in row_bands:
        if not name:
            continue
        ax.plot(
            [0, 0], [lo - 0.4, hi + 0.4],
            color=color, linewidth=4, solid_capstyle="butt",
            transform=band_tr, clip_on=False, zorder=5,
        )
        ax.text(
            0, (lo + hi) / 2, name + "  ",
            rotation=90, ha="right", va="center", transform=band_tr,
            fontsize=8, color="dimgrey", style="italic", clip_on=False,
        )

    # Separator lines between groups.
    for lo, _, name, _ in row_bands[1:]:
        if name:
            ax.axhline(lo - 0.5, color="dimgrey", linewidth=1.2, zorder=4)

    # Family bracket under the grid.
    for lo, hi, name in col_bands:
        y = n_rows - 0.35
        ax.plot(
            [lo - 0.4, hi + 0.4], [y, y],
            color="dimgrey", linewidth=1.2,
            clip_on=False, zorder=5,
        )
        ax.text(
            (lo + hi) / 2, y + 0.12, name,
            ha="center", va="top",
            fontsize=7.5, color="dimgrey", style="italic", clip_on=False,
        )

    # Separator lines between families.
    for lo, _, _ in col_bands[1:]:
        ax.axvline(lo - 0.5, color="dimgrey", linewidth=1.2, zorder=4)


def plot_table(
    results: dict[str, dict],
    specs: list[MetricSpec],
    groups: list[tuple[str | None, list[str]]],
    config: TableConfig,
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
    footnote: str | None = None,
) -> None:
    """The annotated heatmap (``style: heatmap``) and rank table (``rank_table``)."""
    run_names = [name for _, names in groups for name in names]
    n_runs, n_metrics = len(run_names), len(specs)
    rank_style = config.style == "rank_table"

    # values[run, metric]
    values = np.full((n_runs, n_metrics), np.nan)
    for i, name in enumerate(run_names):
        for j, spec in enumerate(specs):
            v = results.get(name, {}).get(spec.key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[i, j] = float(v)

    cmap = plt.get_cmap(config.cmap)
    mode = "rank" if rank_style else config.normalize

    scores = np.full((n_runs, n_metrics), np.nan)
    ranks = np.full((n_runs, n_metrics), np.nan)
    for j, spec in enumerate(specs):
        scores[:, j] = score_column(values[:, j], spec.direction, mode)
        if rank_style:
            ranks[:, j] = dense_ranks(values[:, j], spec.direction) \
                if spec.direction != NONE else np.nan

    col_labels = [s.label + arrow(s) for s in specs]
    families = [s.family for s in specs]

    # The rank table gets one extra, non-metric column: the mean rank across all
    # directional metrics, which is the summary a reader of a rank table wants.
    if rank_style:
        with np.errstate(invalid="ignore"):
            mean_rank = np.nanmean(ranks, axis=1)
        mean_score = score_column(mean_rank, DOWN, "minmax")
        values = np.column_stack([values, mean_rank])
        scores = np.column_stack([scores, mean_score])
        ranks = np.column_stack([ranks, np.full(n_runs, np.nan)])
        col_labels.append("Mean rank ↓")
        families.append("Summary")
        specs = specs + [MetricSpec("_mean_rank", "Mean rank", DOWN, "Summary")]
        n_metrics += 1

    # Cell colours and cell text, in logical (run, metric) orientation.
    rgba = np.zeros((n_runs, n_metrics, 4))
    texts = [["" for _ in range(n_metrics)] for _ in range(n_runs)]
    for i in range(n_runs):
        for j in range(n_metrics):
            if not math.isfinite(values[i, j]):
                rgba[i, j] = _NA_COLOR
                texts[i][j] = "n/a"
                continue
            s = scores[i, j]
            rgba[i, j] = _NEUTRAL_COLOR if not math.isfinite(s) else cmap(s)
            label = fmt_value(values[i, j])
            if rank_style and math.isfinite(ranks[i, j]):
                r = ranks[i, j]
                label += f"  ({r:g})"
            texts[i][j] = label

    # Group bands, in run order.
    run_bands: list[tuple[int, int, str, tuple]] = []
    cursor = 0
    for g_idx, (group_name, names) in enumerate(groups):
        if names:
            run_bands.append(
                (cursor, cursor + len(names) - 1, group_name or "",
                 group_color(g_idx, 0, 1))
            )
        cursor += len(names)

    family_bands = _spans(families) if config.show_families else []

    # Transposition happens here, once: everything above is (run, metric).
    if config.transpose:
        grid = np.transpose(rgba, (1, 0, 2))
        cell_text = [[texts[i][j] for i in range(n_runs)] for j in range(n_metrics)]
        row_labels, col_labels_final = col_labels, run_names
        row_bands = [(lo, hi, name, (0.4, 0.4, 0.4, 1.0)) for lo, hi, name in family_bands]
        col_bands: list[tuple[int, int, str]] = [
            (lo, hi, name) for lo, hi, name, _ in run_bands if name
        ]
    else:
        grid = rgba
        cell_text = texts
        row_labels, col_labels_final = run_names, col_labels
        row_bands = run_bands
        col_bands = [(lo, hi, name) for lo, hi, name in family_bands]

    n_grid_rows, n_grid_cols = len(row_labels), len(col_labels_final)

    # The rotated column headers need vertical room in proportion to their longest
    # label, which is why the height is not simply a multiple of the row count:
    # 20 short run names (transposed) need far less headroom than 20 metric names.
    longest = max((len(lab.replace("\n", " ")) for lab in col_labels_final), default=8)
    header_in = min(0.075 * longest + 0.8, 3.4)
    row_h = 0.5 if config.transpose else 0.46
    cell_w = 1.35 if rank_style else 1.05

    fig_w = figsize[0] if figsize else max(n_grid_cols * cell_w + 3.5, 7.0)
    fig_h = figsize[1] if figsize else max(n_grid_rows * row_h + header_in + 1.7, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_grid(
        ax, grid, cell_text, row_labels, col_labels_final,
        row_bands, col_bands, config.annotate,
    )

    # Colour legend: the scale is a within-column score, not a metric value, so the
    # bar is labelled by meaning rather than by number.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.05, aspect=45, shrink=0.55)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["worst", "mid", "best"])
    cbar.ax.tick_params(labelsize=7.5, length=0)
    cbar.set_label(
        f"numbers = absolute metric values   ·   colour = {mode} within each "
        "metric, direction-aware (ranks the runs shown; not an absolute scale)",
        fontsize=7.5, color="dimgrey",
    )
    cbar.outline.set_visible(False)

    fig.suptitle(config.title, fontsize=13, fontweight="bold")

    notes = ["grey = no better/worse direction", "n/a = metric not computed"]
    fig.text(0.01, 0.005, "  ·  ".join(notes), ha="left", va="bottom",
             fontsize=7, color="dimgrey", style="italic")
    if footnote:
        fig.text(0.99, 0.005, footnote, ha="right", va="bottom",
                 fontsize=7.5, color="#C44E52", style="italic")

    _finish(fig, output, show, config.dpi)


# --------------------------------------------------------------------------- #
# Bar style
# --------------------------------------------------------------------------- #

def plot_bars(
    results: dict[str, dict],
    specs: list[MetricSpec],
    groups: list[tuple[str | None, list[str]]],
    config: TableConfig,
    output: Path | None,
    show: bool,
    figsize: tuple[float, float] | None,
    ncols: int = 5,
    footnote: str | None = None,
) -> None:
    """One subplot per metric, one bar per run, grouped as the config says."""
    import matplotlib.patches as mpatches

    run_names = [name for _, names in groups for name in names]
    x_list, colors, spans = grouped_layout(run_names, groups)
    x_positions = np.asarray(x_list, dtype=float)

    n_metrics = len(specs)
    nrows = math.ceil(n_metrics / ncols)
    fig_w = figsize[0] if figsize else 3.2 * ncols
    fig_h = figsize[1] if figsize else 4.2 * nrows + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, spec in enumerate(specs):
        ax = axes[idx // ncols][idx % ncols]
        ys = [
            float(results.get(name, {}).get(spec.key, float("nan")))
            if isinstance(results.get(name, {}).get(spec.key), (int, float))
            else float("nan")
            for name in run_names
        ]

        bars = ax.bar(x_positions, ys, color=colors, width=0.75, zorder=3,
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=6.5)

        for span_idx, (span_lo, span_hi, _) in enumerate(spans):
            ax.axvspan(
                span_lo - 0.45, span_hi + 0.45,
                color=GROUP_SHADES[span_idx % len(GROUP_SHADES)],
                alpha=0.18, zorder=0,
            )

        ax.set_title(spec.label.replace("\n", " ") + arrow(spec),
                     fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlim(x_positions.min() - 0.7, x_positions.max() + 0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        finite = [y for y in ys if math.isfinite(y)]
        if finite:
            # Anchor at zero so bar length is proportional to the absolute value.
            # Auto-scaling would start the axis just below the smallest bar and turn
            # a 1% difference into a 10x-looking one.  min(0, ...) keeps negative
            # cosines visible instead of clipping them to nothing.
            ax.set_ylim(min(0.0, min(finite) * 1.15), max(0.0, max(finite) * 1.15))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=9)

    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color=colors[i], label=name)
        for i, name in enumerate(run_names)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(run_names), 6),
               fontsize=8, frameon=True, title="Experiment", title_fontsize=8,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(config.title, fontsize=13, fontweight="bold")
    if footnote:
        fig.text(0.99, 0.005, footnote, ha="right", va="bottom",
                 fontsize=7.5, color="#C44E52", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    _finish(fig, output, show, config.dpi, tight=False)


def _finish(fig, output: Path | None, show: bool, dpi: int, tight: bool = True) -> None:
    if tight:
        fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=dpi)
        print(f"Saved to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# --list
# --------------------------------------------------------------------------- #

def print_inventory(results: dict[str, dict], selected: list[MetricSpec]) -> None:
    """Every numeric metric found across the selected runs, by family."""
    chosen = {s.key for s in selected}
    present: dict[str, set[str]] = {}
    for data in results.values():
        for k, v in data.items():
            if k in NON_METRIC_KEYS or k.startswith("_"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            spec = METRIC_BY_KEY.get(k)
            family = spec.family if spec else ("scIB" if k.startswith("scib:") else "Other")
            present.setdefault(family, set()).add(k)

    # Plain ASCII: this goes to a terminal, which is not always UTF-8 capable.
    # The ↑/↓ arrows are for the figure.
    words = {UP: "higher is better", DOWN: "lower is better", NONE: "no direction"}

    print(f"\n{len(results)} run(s): {', '.join(results)}\n")
    for family in sorted(present):
        print(f"{family}")
        for key in sorted(present[family]):
            mark = "*" if key in chosen else " "
            spec = METRIC_BY_KEY.get(key)
            direction = words[spec.direction] if spec else "higher is better"
            print(f"  {mark} {key:<42} {direction}")
        print()
    print("* = in the current selection (change it with the config's 'metrics:' key)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare runs on the internal unified-FM metrics as an annotated "
            "heatmap, rank table or bar grid."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", type=Path, required=True,
        help="YAML config selecting the runs. See load_config() for the schema.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Save path for the figure. Defaults to the config's 'output:' key, "
             "else the config path with a .png suffix.",
    )
    parser.add_argument(
        "--style", choices=STYLES, default=None,
        help="Override the config's 'style:' key.",
    )
    parser.add_argument(
        "--normalize", choices=NORMALIZERS, default=None,
        help="Override the config's 'normalize:' key (colour scale only).",
    )
    parser.add_argument(
        "--transpose", action="store_true",
        help="Draw metrics as rows and runs as columns.",
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, metavar=("W", "H"), default=None,
        help="Figure width and height in inches (auto if omitted).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open an interactive plot window.",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help=(
            "Do not write the plotted numbers beside the figure. By default a "
            "{stem}.csv is written in the same directory."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List every metric available in the selected runs, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path: Path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"ERROR: --config does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    if args.style:
        config.style = args.style
    if args.normalize:
        config.normalize = args.normalize
    if args.transpose:
        config.transpose = True

    print(f"Loading runs from {config_path} ...")
    results, groups = collect_from_config(config)
    if not results:
        print("No runs with unified metrics found.", file=sys.stderr)
        sys.exit(1)

    for group_name, names in groups:
        print(f"  {group_name or '(ungrouped)'}: {', '.join(names)}")

    specs = resolve_metrics(results, config)

    if args.list:
        print_inventory(results, specs)
        return

    if not specs:
        print("No plottable metrics in the selected runs.", file=sys.stderr)
        sys.exit(1)

    footnote = warn_panel_mismatch(results)
    figsize = tuple(args.figsize) if args.figsize else config.figsize
    output: Path | None = args.output or config.output or config_path.with_suffix(".png")

    print(f"{len(results)} run(s), {len(specs)} metric(s), style '{config.style}'.")

    if output is not None and not args.no_csv:
        write_table_csv(output, results, specs, groups)

    if config.style == "bars":
        plot_bars(results, specs, groups, config, output,
                  show=not args.no_show, figsize=figsize, footnote=footnote)
    else:
        plot_table(results, specs, groups, config, output,
                   show=not args.no_show, figsize=figsize, footnote=footnote)


if __name__ == "__main__":
    main()
