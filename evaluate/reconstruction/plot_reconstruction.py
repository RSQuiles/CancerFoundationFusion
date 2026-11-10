"""Figures comparing models' reconstruction of one mask plan.

    python -m evaluate.reconstruction.plot_reconstruction --plan bulk:leiden_s0:test:r25 --name cf_vs_bf \\
        [--models unified_v1 unified_v2 bulkformer] [--plots mae_by_bin metrics_table] [--baselines]

Reads the ``run_reconstruction`` results of the selected models (``--models``; default
every model with results on the plan). ``--name`` (required) names the comparison, so one
plan can hold several comparisons of different models. Models are grouped by target space, bins models
also by ``n_bins`` (``bins51``, ``ln_tpm1``, ...): native-space numbers only compare
within a group, each on the masked entries every model of the group scored. The group
``all`` holds every selected model on the entries they all scored, for the space-free
plots.

Plots (``--plots``; default all), each drawn once per group it applies to:

- ``mae_by_bin`` (bins groups): mean MAE by true bin, one line per model, the entry
  count per bin below;
- ``mae_by_target`` (other spaces): mean MAE for zero targets, then by decile of the
  non-zero targets (edges shared by the group's models);
- ``metrics_table`` (every space group): global native and space-free metrics, the best
  model of each column shaded; the columns per space are in ``TABLE_COLUMNS``;
- ``space_free_table`` (``all``): per-sample Pearson / Spearman, global Spearman and
  zero AUROC, which compare across spaces;
- ``per_sample_spearman`` (``all``): per-sample Spearman distribution per model;
- ``rank_mae_by_target`` (``all``): the space-free analog of ``mae_by_bin``. Within
  each sample, targets and predictions become percentile ranks over the shared masked
  genes; the plot is the mean absolute rank error by target-rank decile.

**Tied bins.** CancerFoundation bins a row at quantile edges, and a value on repeated
edges gets a random bin among several equally correct ones. Runs that recorded those
ranges (``target_lo`` / ``target_hi`` in ``entries.parquet``) are scored tie-aware: the
error is the distance to the nearest equally correct bin, and a prediction inside the
range is exact. Older runs fall back to the raw bin error, which the figure notes. The
correlation and rank plots of those runs use the range midpoint as the target
(``metrics.correlation_target``), so tied values share a rank instead of a random order;
the figures and the manifest (``tie_midpoint_models``) name the models this applies to.

**Baselines** (``--baselines``): one ``gene_mean`` and one ``sample_mean`` row per space
group, labeled ``baseline: ...``, not one per model. Every model of a group is scored on
the same entries of the same plan, so their stored baselines are normally identical;
they are taken from the group's first model, and if another model's differ (e.g. CF
models with different vocabularies bin a sample differently) the figures and the
manifest (``baseline_source``) say so. The cross-space group has no baselines (a
baseline lives in one space).

Add a plot by writing a ``compute`` (the plotted numbers, saved as ``<plot>.csv``) and
a ``draw`` function and one ``PlotSpec`` in ``PLOTS``; give a new space its table
columns in ``TABLE_COLUMNS`` (otherwise ``"*"`` applies).

Writes ``results/tasks/reconstruction/_plots/<plan>/<name>/<group>/``: one ``.png`` /
``.pdf`` and ``.csv`` per plot, and ``manifest.json`` (``"models"``, groups, plots) in
``<name>/``. Rerunning a name overwrites its figures; a group no longer plotted keeps its
old files, so use a new name for a different set of models.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_with_config  # noqa: E402
from evaluate._plot_style import (  # noqa: E402
    HAIRLINE,
    INK,
    INK_2,
    MUTED,
    SEQ_RAMP,
    SERIES,
    Col,
    draw_metric_table,
    pyplot,
    save,
)
from evaluate.reconstruction import compare as C  # noqa: E402
from evaluate.reconstruction import metrics as M  # noqa: E402
from evaluate.reconstruction.mask_plan import load_plan, resolve_plan_path  # noqa: E402
from evaluate.reconstruction.run_reconstruction import RESULTS_TASK, plan_slug  # noqa: E402

CROSS_GROUP = "all"
BASELINES = ("gene_mean", "sample_mean")
# Row label of each baseline: one row per space group, shared by its models.
BASELINE_LABELS = {"gene_mean": "baseline: gene mean", "sample_mean": "baseline: sample mean"}

# Columns of metrics_table per space ("*": any space without its own list). The bins keys
# are the tie-aware metrics; runs without tie ranges fall back to TIE_FALLBACK.
_SPACE_FREE_COLUMNS = [
    Col("pearson_mean", "Mean", "Pearson (per sample)"),
    Col("pearson_median", "Median", "Pearson (per sample)"),
    Col("spearman_mean", "Mean", "Spearman"),
    Col("spearman_nonzero_mean", "Non-zero", "Spearman"),
    Col("spearman_global", "Global", "Spearman"),
]
TABLE_COLUMNS: dict[str, list[Col]] = {
    "bins": [
        Col("mae_tie", "All", "MAE (bins)", "min"),
        Col("mae_zero_target", "Zero", "MAE (bins)", "min"),
        Col("mae_nonzero_target_tie", "Non-zero", "MAE (bins)", "min"),
        Col("mse_tie", "MSE", "MAE (bins)", "min"),
        Col("acc_exact_tie", "Exact", "Accuracy"),
        Col("acc_within1_tie", "±1 bin", "Accuracy"),
        *_SPACE_FREE_COLUMNS,
    ],
    "*": [
        Col("mae", "All", "MAE", "min"),
        Col("mae_zero_target", "Zero", "MAE", "min"),
        Col("mae_nonzero_target", "Non-zero", "MAE", "min"),
        Col("mse", "MSE", "MAE", "min"),
        *_SPACE_FREE_COLUMNS,
        Col("auroc_zero", "AUROC", "Zeros"),
    ],
}
SPACE_FREE_TABLE = [*_SPACE_FREE_COLUMNS, Col("auroc_zero", "AUROC", "Zeros")]
TIE_FALLBACK = {"mae_tie": "mae", "mse_tie": "mse", "mae_nonzero_target_tie": "mae_nonzero_target",
                "acc_exact_tie": "acc_exact", "acc_within1_tie": "acc_within1"}


@dataclass
class Group:
    """Models compared together, each restricted to the entries all of them scored."""

    name: str
    space: str | None                     # None for the cross-space group
    models: list[str]
    frames: dict[str, pd.DataFrame]
    n_shared: int
    tie_aware: bool = False
    notes: list[str] = field(default_factory=list)
    # Models whose correlation / rank target is the tie midpoint (runs with tie ranges).
    midpoint_models: list[str] = field(default_factory=list)
    # Model whose stored baselines the group shows (space groups with --baselines).
    baseline_source: str | None = None


@dataclass(frozen=True)
class PlotSpec:
    name: str
    scope: str                                     # "space": per space group; "all": cross-space
    applies: Callable[[str | None], bool]          # space -> whether the plot is drawn for it
    compute: Callable[[Group, argparse.Namespace], pd.DataFrame]
    draw: Callable[[pd.DataFrame, Group, argparse.Namespace], object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figures comparing models' reconstruction of one mask plan.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--plan", default=None, help="MODALITY:SPLIT:PART:NAME or a .npz path")
    parser.add_argument("--name", default=None,
                        help="Name of this comparison (required): _plots/<plan>/<name>/")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Models to plot (default: every model with results on the plan)")
    parser.add_argument("--plots", nargs="*", default=[], help="Plots to draw (default: all; see --list)")
    parser.add_argument("--baselines", action="store_true",
                        help="Also show the gene-mean and sample-mean baselines (one row each per space group)")
    parser.add_argument("--bin-width", type=int, default=1, help="mae_by_bin: true bins merged per point")
    parser.add_argument("--n-quantiles", type=int, default=10,
                        help="mae_by_target / rank_mae_by_target: quantile groups of the target")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure file formats")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--list", action="store_true", help="Print the available models and plots, then exit")
    return parser


# ---- selection and groups -----------------------------------------------------

def select_models(requested: list[str], available: list[str], plan_id: str) -> list[str]:
    """The models to plot, in order (default: every model with results on the plan)."""
    if not available:
        sys.exit(f"Error: no model has reconstruction results on {plan_id}; run run_reconstruction first")
    if not requested:
        return list(available)
    models = list(dict.fromkeys(requested))
    unknown = [m for m in models if m not in available]
    if unknown:
        sys.exit(f"Error: no results on {plan_id} for {unknown}; available: {available}")
    return models


def space_group(card: dict) -> str:
    """``bins<n_bins>`` for bins models (different n_bins are different scales), else the space."""
    space = card["space"]
    n_bins = (card.get("provenance") or {}).get("n_bins")
    return f"bins{int(n_bins)}" if space == "bins" and n_bins is not None else str(space)


def baseline_source(frames: dict[str, pd.DataFrame]) -> tuple[str, list[str]]:
    """``(model whose baselines are shown, models whose baselines differ from it)``.

    ``frames`` are a group's models on their shared entries, so on the same plan their
    stored baselines agree unless the models' targets differ.
    """
    models = list(frames)
    ref = frames[models[0]].set_index(["sample", "gene"])[list(BASELINES)]
    differ = []
    for model in models[1:]:
        other = frames[model].set_index(["sample", "gene"])[list(BASELINES)].reindex(ref.index)
        if not np.allclose(ref.to_numpy(float), other.to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True):
            differ.append(model)
    return models[0], differ


def build_groups(frames: dict[str, pd.DataFrame], cards: dict[str, dict], baselines: bool = False) -> list[Group]:
    """One group per space (models in input order), plus the cross-space group."""
    by_group: dict[str, list[str]] = {}
    for model in frames:
        by_group.setdefault(space_group(cards[model]), []).append(model)
    groups = []
    for name, models in by_group.items():
        keys, subs = C.shared_entries({m: frames[m] for m in models})
        group = Group(name, cards[models[0]]["space"], models, subs, len(keys))
        if group.space == "bins":
            group.tie_aware = all(M.has_ties(f) for f in subs.values())
            if not group.tie_aware:
                stale = [m for m, f in subs.items() if not M.has_ties(f)]
                group.notes.append(f"raw bin error: no tie ranges for {stale} (rerun run_reconstruction)")
        if baselines and len(keys):
            group.baseline_source, differ = baseline_source(subs)
            if differ:
                note = (f"baselines from {group.baseline_source}; those of {differ} differ "
                        f"(different targets, e.g. another vocabulary)")
                print(f"  [warn] {name}: {note}")
                group.notes.append(note)
        groups.append(group)
    if len(frames) > 1:
        keys, subs = C.shared_entries(frames)
        groups.append(Group(CROSS_GROUP, None, list(frames), subs, len(keys)))
    for group in groups:
        group.midpoint_models = [m for m, f in group.frames.items() if M.has_ties(f)]
    return groups


def predictor_rows(group: Group, baselines: bool) -> list[tuple[str, str, pd.DataFrame]]:
    """``(label, column, entries)`` per row: every model, then the group's baselines if asked.

    The baselines are one row each per space group (from ``group.baseline_source``), not
    one per model. A baseline lives in one space, so the cross-space group has none.
    """
    rows = [(model, "pred", frame) for model, frame in group.frames.items()]
    if baselines and group.space is not None and group.baseline_source:
        frame = group.frames[group.baseline_source]
        for col in BASELINES:
            rows.append((BASELINE_LABELS[col], col, frame[frame[col].notna()]))
    return rows


def _error(frame: pd.DataFrame, col: str, tie_aware: bool) -> np.ndarray:
    if tie_aware:
        return M.tie_error(frame, col)
    return np.abs(frame[col].to_numpy(float) - frame["target"].to_numpy(float))


# ---- compute ------------------------------------------------------------------

def compute_mae_by_bin(group: Group, args) -> pd.DataFrame:
    width = max(1, int(args.bin_width))
    out = []
    for label, col, frame in predictor_rows(group, args.baselines):
        key = (frame["target"].to_numpy(float) // width * width).astype(int)
        g = pd.DataFrame({"bin": key, "mae": _error(frame, col, group.tie_aware)}).groupby("bin")["mae"]
        out.append(pd.DataFrame({"row": label, "bin": g.mean().index, "mae": g.mean().to_numpy(),
                                 "n": g.size().to_numpy()}))
    return pd.concat(out, ignore_index=True)


def quantile_edges(values: np.ndarray, n: int) -> np.ndarray:
    """Interior quantile edges of ``values`` (duplicates dropped)."""
    if len(values) == 0:
        return np.array([])
    return np.unique(np.quantile(values, np.linspace(0, 1, n + 1)[1:-1]))


def compute_mae_by_target(group: Group, args) -> pd.DataFrame:
    # Shared edges: every model of the group is binned the same way.
    targets = np.concatenate([f["target"].to_numpy(float) for f in group.frames.values()])
    edges = quantile_edges(targets[targets != 0], args.n_quantiles)
    out = []
    for label, col, frame in predictor_rows(group, args.baselines):
        t = frame["target"].to_numpy(float)
        key = np.where(t == 0, 0, np.searchsorted(edges, t, side="right") + 1)
        g = pd.DataFrame({"group": key, "mae": _error(frame, col, False)}).groupby("group")["mae"]
        out.append(pd.DataFrame({"row": label, "group": g.mean().index, "mae": g.mean().to_numpy(),
                                 "n": g.size().to_numpy()}))
    table = pd.concat(out, ignore_index=True)
    nonzero = targets[targets != 0]
    lower = np.r_[nonzero.min(), edges] if len(nonzero) else np.array([])
    upper = np.r_[edges, nonzero.max()] if len(nonzero) else np.array([])
    table["label"] = ["zero" if k == 0 else f"Q{k}" for k in table["group"]]
    table["lower"] = [0.0 if k == 0 else float(lower[k - 1]) for k in table["group"]]
    table["upper"] = [0.0 if k == 0 else float(upper[k - 1]) for k in table["group"]]
    return table


def _metric_rows(group: Group, args, columns: list[Col]) -> pd.DataFrame:
    rows = []
    for label, col, frame in predictor_rows(group, args.baselines):
        values = {**M.space_free(frame, col)}
        if group.space is not None:
            values.update(M.native(frame, group.space, col))
        rows.append({"row": label, **values})
    table = pd.DataFrame(rows)
    # The drawn columns first, then every other metric (the csv keeps them all).
    first = ["row", "n_entries", *[c.key for c in columns if c.key in table.columns]]
    return table[list(dict.fromkeys([*first, *table.columns]))]


def compute_metrics_table(group: Group, args) -> pd.DataFrame:
    return _metric_rows(group, args, drawn_columns(group))


def drawn_columns(group: Group) -> list[Col]:
    """``metrics_table`` columns of ``group``; raw bin metrics when a run has no tie ranges."""
    cols = table_columns(group.space)
    if group.space == "bins" and not group.tie_aware:
        cols = [Col(TIE_FALLBACK.get(c.key, c.key), c.header, c.group, c.best, c.fmt) for c in cols]
    return cols


def compute_space_free_table(group: Group, args) -> pd.DataFrame:
    return _metric_rows(group, args, SPACE_FREE_TABLE)


def compute_per_sample_spearman(group: Group, args) -> pd.DataFrame:
    out = []
    for label, col, frame in predictor_rows(group, args.baselines):
        ps = M.per_sample(frame, col)[["sample", "n", "spearman"]]
        ps.insert(0, "row", label)
        out.append(ps)
    return pd.concat(out, ignore_index=True)


def compute_rank_mae_by_target(group: Group, args) -> pd.DataFrame:
    n_q = max(1, int(args.n_quantiles))
    out = []
    for label, col, frame in predictor_rows(group, args.baselines):
        # Ranks of the tie midpoint when the run has tie ranges (metrics.correlation_target):
        # genes the binning tied share a rank instead of a random order.
        ranked = frame.assign(_rank_target=M.correlation_target(frame))
        by = ranked.groupby("sample")
        n = by["_rank_target"].transform("size").to_numpy(float)
        scale = np.where(n > 1, n - 1, np.nan)
        # Percentile ranks within the sample, 0 = lowest, 1 = highest; ties get the average rank.
        rt = (by["_rank_target"].rank(method="average").to_numpy(float) - 1) / scale
        rp = (by[col].rank(method="average").to_numpy(float) - 1) / scale
        ok = np.isfinite(rt) & np.isfinite(rp)
        decile = np.minimum((rt[ok] * n_q).astype(int), n_q - 1)
        g = pd.DataFrame({"quantile": decile, "mae": np.abs(rp[ok] - rt[ok])}).groupby("quantile")["mae"]
        out.append(pd.DataFrame({"row": label, "quantile": g.mean().index, "mae_rank": g.mean().to_numpy(),
                                 "n": g.size().to_numpy()}))
    return pd.concat(out, ignore_index=True)


def table_columns(space: str | None) -> list[Col]:
    return TABLE_COLUMNS.get(space or "*", TABLE_COLUMNS["*"])


# ---- draw ---------------------------------------------------------------------

def _colors(rows: list[str], group: Group) -> dict[str, tuple[str, str]]:
    """``{row: (color, linestyle)}``: models take the categorical slots in order, solid;
    the shared baselines are grey, dashed (gene mean) or dotted (sample mean)."""
    slot = dict(zip(group.models, SERIES))
    style = {BASELINE_LABELS["gene_mean"]: (MUTED, "--"), BASELINE_LABELS["sample_mean"]: (MUTED, ":")}
    return {row: style.get(row, (slot.get(row, MUTED), "-")) for row in rows}


def _title(group: Group, what: str, plan_id: str) -> str:
    where = f"{group.name}" if group.space else "all spaces"
    return f"Reconstruction {plan_id} ({where}): {what}"


def _footnote(group: Group, extra: str = "") -> str:
    parts = [f"{group.n_shared} masked entries scored by every model shown."]
    if extra:
        parts.append(extra)
    parts.extend(n[0].upper() + n[1:] + "." for n in group.notes)
    return " ".join(parts)


def _midpoint_note(group: Group) -> str:
    """Footnote for plots built on correlations or ranks: whose target is the tie midpoint."""
    if not group.midpoint_models:
        return ""
    return (f"Correlations and ranks of {', '.join(group.midpoint_models)} use the midpoint of each "
            f"tied bin range as the target, so genes the binning tied share a rank.")


def _add_footnote(fig, text: str) -> None:
    """Lay the figure out, then put ``text`` (wrapped) under everything else."""
    import textwrap

    fig.tight_layout()
    width = int(fig.get_figwidth() * 17)  # characters per line at 7 pt
    fig.text(0.01, 0.0, "\n".join(textwrap.wrap(text, width)), fontsize=7, color=MUTED, ha="left", va="top")


def _line_panel(data: pd.DataFrame, group: Group, x: str, y: str, xlabel: str, ylabel: str, title: str,
                footnote: str, xticks: tuple[list, list] | None = None):
    """Lines per row over ``x``, the entry count of the first model below as a strip."""
    plt = pyplot()
    rows = list(dict.fromkeys(data["row"]))
    style = _colors(rows, group)
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    for row in rows:
        d = data[data["row"] == row].sort_values(x)
        color, ls = style[row]
        ax.plot(d[x], d[y], color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.4,
                marker="o" if ls == "-" else None, markersize=3, label=row)
    first = data[data["row"] == rows[0]].sort_values(x)
    ax_n.bar(first[x], first["n"], width=0.8 * (np.min(np.diff(first[x])) if len(first) > 1 else 1),
             color=SEQ_RAMP[1], edgecolor="none")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax_n.set_ylabel("Entries", fontsize=8)
    ax_n.set_xlabel(xlabel)
    ax_n.tick_params(axis="y", labelsize=7)
    for a in (ax, ax_n):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    if xticks:
        ax_n.set_xticks(*xticks)
    if len(rows) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    _add_footnote(fig, footnote)
    return fig


def draw_mae_by_bin(data: pd.DataFrame, group: Group, args):
    width = max(1, int(args.bin_width))
    # Without tie ranges the group's note already says the error is raw.
    how = ("Tie-aware: distance to the nearest bin the binning could have given the value. "
           if group.tie_aware else "")
    xlabel = "True bin" if width == 1 else f"True bin (groups of {width})"
    return _line_panel(data, group, "bin", "mae", xlabel, "Mean absolute error (bins)",
                       _title(group, "MAE by true bin", args.plan_id),
                       _footnote(group, how + "Counts: entries per bin (first model)."))


def draw_mae_by_target(data: pd.DataFrame, group: Group, args):
    labels = data.drop_duplicates("group").sort_values("group")
    return _line_panel(data, group, "group", "mae", "True value: zero, then quantile of the non-zero targets",
                       f"Mean absolute error ({group.space})", _title(group, "MAE by target", args.plan_id),
                       _footnote(group, "Quantile edges shared by the group's models; see the csv for them."),
                       xticks=(list(labels["group"]), list(labels["label"])))


def draw_rank_mae_by_target(data: pd.DataFrame, group: Group, args):
    n_q = max(1, int(args.n_quantiles))
    ticks = list(range(n_q))
    return _line_panel(data, group, "quantile", "mae_rank", "Target rank within the sample (quantile, low to high)",
                       "Mean absolute rank error", _title(group, "rank error by target", args.plan_id),
                       _footnote(group, "Ranks are percentiles over each sample's shared masked genes; "
                                        "tied targets share their average rank. " + _midpoint_note(group)),
                       xticks=(ticks, [f"Q{q + 1}" for q in ticks]))


def _draw_table(data: pd.DataFrame, group: Group, args, columns: list[Col], what: str, extra: str):
    return draw_metric_table(data, list(data["row"]), columns, _title(group, what, args.plan_id),
                             "Shaded, bold: best per column (↑ higher is better, ↓ lower). " + _footnote(group, extra))


def draw_metrics_table(data: pd.DataFrame, group: Group, args):
    extra = ""
    if group.space == "bins":
        extra = ("MAE and accuracy count any bin a tied value could have been given as correct. "
                 if group.tie_aware else "")
    extra += _midpoint_note(group)
    return _draw_table(data, group, args, drawn_columns(group), "global metrics", extra)


def draw_space_free_table(data: pd.DataFrame, group: Group, args):
    return _draw_table(data, group, args, SPACE_FREE_TABLE, "space-free metrics",
                       "Scale-invariant, so comparable across target spaces. " + _midpoint_note(group))


def draw_per_sample_spearman(data: pd.DataFrame, group: Group, args):
    plt = pyplot()
    values = {r: data.loc[data["row"] == r, "spearman"].dropna().to_numpy() for r in dict.fromkeys(data["row"])}
    rows = [r for r, v in values.items() if len(v)]  # a constant predictor has no correlation
    values = [values[r] for r in rows]
    fig, ax = plt.subplots(figsize=(0.9 * len(rows) + 2.2, 3.8))
    box = ax.boxplot(values, showfliers=False, patch_artist=True, widths=0.55,
                     medianprops={"color": INK, "linewidth": 1.4})
    for patch in box["boxes"]:
        patch.set(facecolor=SEQ_RAMP[0], edgecolor=INK_2, linewidth=0.8)
    for part in ("whiskers", "caps"):
        for line in box[part]:
            line.set(color=INK_2, linewidth=0.8)
    ax.set_xticks(range(1, len(rows) + 1), rows, rotation=30, ha="right")
    ax.set_ylabel("Per-sample Spearman (masked genes)")
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(_title(group, "per-sample Spearman", args.plan_id), fontsize=10, color=INK, loc="left")
    _add_footnote(fig, _footnote(group, "Boxes: quartiles; whiskers: 1.5 IQR; outliers hidden. "
                                        + _midpoint_note(group)))
    return fig


def _not_bins(space: str | None) -> bool:
    return space is not None and space != "bins"


PLOTS: list[PlotSpec] = [
    PlotSpec("mae_by_bin", "space", lambda s: s == "bins", compute_mae_by_bin, draw_mae_by_bin),
    PlotSpec("mae_by_target", "space", _not_bins, compute_mae_by_target, draw_mae_by_target),
    PlotSpec("metrics_table", "space", lambda s: s is not None, compute_metrics_table, draw_metrics_table),
    PlotSpec("space_free_table", "all", lambda s: True, compute_space_free_table, draw_space_free_table),
    PlotSpec("per_sample_spearman", "all", lambda s: True, compute_per_sample_spearman, draw_per_sample_spearman),
    PlotSpec("rank_mae_by_target", "all", lambda s: True, compute_rank_mae_by_target, draw_rank_mae_by_target),
]
PLOT_NAMES = [p.name for p in PLOTS]
# One categorical slot per model, never cycled: more models than slots are not drawn as lines.
LINE_PLOTS = {"mae_by_bin", "mae_by_target", "rank_mae_by_target"}


def select_plots(requested: list[str]) -> list[PlotSpec]:
    if not requested:
        return list(PLOTS)
    unknown = [p for p in requested if p not in PLOT_NAMES]
    if unknown:
        sys.exit(f"Error: unknown plot(s) {unknown}; choose from {PLOT_NAMES}")
    return [p for p in PLOTS if p.name in requested]


# ---- main ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> Path | None:
    from experiments.layout import ASSESSMENT_MANIFEST, ExperimentLayout

    args = parse_with_config(build_parser(), "reconstruction_plot", argv)
    if not args.plan:
        sys.exit("Error: give --plan")
    if not args.name and not args.list:
        sys.exit("Error: give --name (the comparison's folder under _plots/<plan>/)")
    layout = ExperimentLayout.from_root(args.root)
    plan = load_plan(resolve_plan_path(args.plan, args.root))
    slug = plan_slug(plan)
    args.plan_id = plan.plan_id
    available = C.available_models(layout, slug)
    if args.list:
        print(f"models with results on {plan.plan_id}: {available or 'none'}")
        print(f"plots: {PLOT_NAMES}")
        return None
    try:
        out_dir = layout.task_plot_dir(RESULTS_TASK, slug, args.name)
    except ValueError as exc:
        sys.exit(f"Error: --name: {exc}")
    models = select_models(args.models, available, plan.plan_id)
    plots = select_plots(args.plots)

    frames, cards = C.load_runs(layout, models, slug)
    groups = build_groups(frames, cards, args.baselines)
    if (out_dir / ASSESSMENT_MANIFEST).exists():
        print(f"  [note] comparison '{args.name}' exists; its figures are overwritten")
    print(f"plotting {models} on {plan.plan_id} -> {out_dir}")

    written: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for group in groups:
        for spec in plots:
            if (spec.scope == "all") != (group.space is None) or not spec.applies(group.space):
                continue
            key = f"{group.name}/{spec.name}"
            if group.n_shared == 0:
                skipped[key] = "no entry scored by every model"
                print(f"  [skip] {key}: no entry scored by every model")
                continue
            if spec.name in LINE_PLOTS and len(group.models) > len(SERIES):
                skipped[key] = f"more than {len(SERIES)} models in one line plot"
                print(f"  [skip] {key}: {skipped[key]}; pass fewer --models")
                continue
            group_dir = out_dir / group.name
            group_dir.mkdir(parents=True, exist_ok=True)
            data = spec.compute(group, args)
            data.to_csv(group_dir / f"{spec.name}.csv", index=False)
            written[key] = [f"{spec.name}.csv", *save(spec.draw(data, group, args), group_dir / spec.name,
                                                       args.formats, args.dpi)]
    if len(models) == 1:
        skipped[f"{CROSS_GROUP}/*"] = "one model: no cross-space comparison"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ASSESSMENT_MANIFEST).write_text(json.dumps({
        "analysis": "reconstruction_plot", "plan": plan.plan_id, "name": args.name, "models": models,
        "groups": {g.name: {"space": g.space, "models": g.models, "n_shared_entries": g.n_shared,
                            "tie_aware": g.tie_aware, "tie_midpoint_models": g.midpoint_models,
                            "baseline_source": g.baseline_source, "notes": g.notes} for g in groups},
        "baselines": bool(args.baselines), "plots": written, "skipped": skipped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
