"""Figures comparing models on a GeneICL task suite.

    python -m evaluate.geneicl.plot_benchmark --name main [--models unified_v1 pca] \\
        [--plots summary_table heatmap_auroc] [--suite benchmark] [--reference scfoundation en] [--config cfg.yaml]

Reads the ``run_probe`` results of the selected models (``--models``; default every model
with results on the suite). ``--name`` (required) names the comparison, so one suite can
hold several comparisons of different models. The script computes the per-task and summary
tables with ``compare.py``'s own functions, so the numbers are those of ``compare`` run on
the same models. ``--reference`` adds GeneICL's own result files as ``ref:<tag>`` columns, as in
``compare``.

Plots (``--plots``; default all):

- ``heatmap_<metric>``: task x model heatmap of one metric, the best value of each task in
  bold, plus a mean row. Metrics: auroc, auprc, balanced_accuracy, mcc, f1_macro
  (classification), spearman, pearson, r2, r2_pooled_oof (regression). ``heatmaps``
  selects all of them.
- ``summary_table``: the ``compare`` summary table, the best value of each metric shaded
  and in bold (lowest for the mean ranks).

Writes ``results/tasks/geneicl_<suite>/_plots/<name>/``, next to the per-model
``run_probe`` results: one ``.png`` / ``.pdf`` per plot, the plotted numbers as
``per_task.csv`` and ``summary.csv``, and ``manifest.json`` (``"models"``, references,
plots). Rerunning a name overwrites its figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_with_config  # noqa: E402
from evaluate._plot_style import (  # noqa: E402
    INK, INK_2, MISSING, MUTED, SEQ_RAMP, SURFACE, Col, draw_metric_table, is_best, pyplot, save,
)
from evaluate.geneicl import compare as C  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402
from evaluate.geneicl.run_probe import task_name  # noqa: E402


# metric -> (task type, display name); every one is higher-is-better.
HEATMAP_METRICS: dict[str, tuple[str, str]] = {
    "auroc": ("classification", "AUROC"),
    "auprc": ("classification", "AUPRC"),
    "balanced_accuracy": ("classification", "Balanced accuracy"),
    "mcc": ("classification", "MCC"),
    "f1_macro": ("classification", "Macro F1"),
    "spearman": ("regression", "Spearman"),
    "pearson": ("regression", "Pearson"),
    "r2": ("regression", "R² (mean of folds)"),
    "r2_pooled_oof": ("regression", "R² (pooled out-of-fold)"),
}
HEATMAP_PLOTS = [f"heatmap_{m}" for m in HEATMAP_METRICS]
SUMMARY_PLOT = "summary_table"
PLOTS = [*HEATMAP_PLOTS, SUMMARY_PLOT]
PLOT_GROUPS = {"heatmaps": HEATMAP_PLOTS, "all": PLOTS}

# Summary-table columns, grouped by task type; rank columns are lower-is-better.
SUMMARY_COLUMNS = [
    Col("n_classification_tasks", "Tasks", "classification", None),
    Col("cla_auroc", "AUROC", "classification"),
    Col("cla_auprc", "AUPRC", "classification"),
    Col("cla_balanced_accuracy", "Bal. acc.", "classification"),
    Col("cla_mcc", "MCC", "classification"),
    Col("cla_f1_macro", "Macro F1", "classification"),
    Col("cla_rank_auroc", "Rank AUROC", "classification", "min", "{:.2f}"),
    Col("n_regression_tasks", "Tasks", "regression", None),
    Col("reg_spearman", "Spearman", "regression"),
    Col("reg_pearson", "Pearson", "regression"),
    Col("reg_r2", "R²", "regression"),
    Col("reg_r2_pooled_oof", "R² pooled", "regression"),
    Col("reg_rank_spearman", "Rank Spearman", "regression", "min", "{:.2f}"),
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figures comparing models on a GeneICL task suite.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--geneicl-repo", type=Path, default=None,
                        help=f"GeneICL checkout, for --reference (default: ${S.GENEICL_ENV_VAR}, else ../GeneICL)")
    parser.add_argument("--geneicl-results", type=Path, default=None,
                        help="Directory of GeneICL's <tag>_<task_type>.csv (default: <geneicl>/eval/benchmark/results)")
    parser.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
    parser.add_argument("--name", default=None,
                        help="Name of this comparison (required): tasks/geneicl_<suite>/_plots/<name>/")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Models to plot (default: every model with run_probe results on the suite)")
    parser.add_argument("--reference", nargs="*", default=[],
                        help="GeneICL result tags shown alongside (e.g. scfoundation en)")
    parser.add_argument("--plots", nargs="*", default=[],
                        help=f"Plots to draw (default: all): {', '.join(PLOTS)}; or the groups "
                             f"{', '.join(PLOT_GROUPS)}")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure file formats")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--list", action="store_true", help="Print the available models and plots, then exit")
    return parser


def select_models(requested: list[str], available: list[str]) -> list[str]:
    """The models to plot, in order (default: every model with results on the suite)."""
    if not available:
        sys.exit("Error: no model has GeneICL results; run evaluate.geneicl.run_probe first")
    if not requested:
        return list(available)
    models = list(dict.fromkeys(requested))
    unknown = [m for m in models if m not in available]
    if unknown:
        sys.exit(f"Error: no GeneICL results for {unknown}; available: {available}")
    return models


def select_plots(requested: list[str]) -> list[str]:
    if not requested:
        return list(PLOTS)
    out: list[str] = []
    for name in requested:
        expanded = PLOT_GROUPS.get(name, [name])
        for p in expanded:
            if p not in PLOTS:
                sys.exit(f"Error: unknown plot {name!r}; choose from {PLOTS} or {list(PLOT_GROUPS)}")
            out.append(p)
    return list(dict.fromkeys(out))


# ---- drawing ----------------------------------------------------------------

def _label(row: str) -> str:
    return row[len(C.REFERENCE_PREFIX):] + " (GeneICL)" if row.startswith(C.REFERENCE_PREFIX) else row


def heatmap_matrix(table: pd.DataFrame, metric: str, rows: list[str]) -> pd.DataFrame:
    """Tasks (sorted by dataset, then task) x rows of one metric; empty when absent."""
    tt = HEATMAP_METRICS[metric][0]
    sub = table[table["task_type"] == tt] if len(table) else table
    if sub.empty or metric not in sub.columns or sub[metric].isna().all():
        return pd.DataFrame()
    mat = sub.pivot_table(index=["dataset", "task_id"], columns="row", values=metric, aggfunc="first")
    mat = mat.sort_index().reindex(columns=rows)
    mat.index = mat.index.get_level_values("task_id")
    return mat


def plot_heatmap(mat: pd.DataFrame, metric: str, suite: str):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    plt = pyplot()
    tt, display = HEATMAP_METRICS[metric]
    values = mat.to_numpy(float)
    means = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0) \
        if np.isfinite(values).any() else np.full(values.shape[1], np.nan)
    n_tasks, n_cols = values.shape

    finite = values[np.isfinite(values)]
    vmin, vmax = (np.percentile(finite, 2), np.percentile(finite, 98)) if finite.size else (0.0, 1.0)
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    def norm_value(v: float) -> float:
        return float((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))

    col_chars = max(len(_label(c)) for c in mat.columns)
    width = 0.75 * n_cols + 1.6
    height = 0.26 * (n_tasks + 1.6) + 0.012 * col_chars * 9 + 0.9
    fig, ax = plt.subplots(figsize=(width, height))

    gap = 0.4  # blank band between the tasks and the mean row
    rows_y = list(range(n_tasks)) + [n_tasks + gap]
    all_values = np.vstack([values, means[None, :]])
    for i, y in enumerate(rows_y):
        best = is_best(all_values[i], "max")
        for j in range(n_cols):
            v = all_values[i, j]
            if np.isfinite(v):
                frac = float(norm_value(v))
                face = cmap(frac)
                text_color = "#ffffff" if frac > 0.55 else INK
                text = f"{v + 0.0:.2f}".replace("-0.00", "0.00")
            else:
                face, text_color, text = MISSING, MUTED, "n/a"
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=face, edgecolor=SURFACE, linewidth=1.5))
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center", fontsize=7, color=text_color,
                    fontweight="bold" if best[j] else "normal")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(rows_y[-1] + 1, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, [_label(c) for c in mat.columns], rotation=35, ha="left",
                  rotation_mode="anchor")
    ax.set_yticks(np.array(rows_y) + 0.5, [*mat.index, f"Mean ({n_tasks} tasks)"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0, labelsize=8)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, aspect=30,
                        extend="both" if finite.size and (finite.min() < vmin or finite.max() > vmax) else "neither")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=0, colors=INK_2)
    cbar.set_label(display, fontsize=8, color=INK_2)
    ax.set_title(f"GeneICL {suite}: {display} per {tt} task", fontsize=10, color=INK, pad=10, loc="left")
    how = "All folds' predictions pooled" if metric == "r2_pooled_oof" else "Mean over folds"
    fig.text(0.0, 0.0, f"Bold: best per task. {how}; colors clipped to the 2nd-98th percentile.",
             fontsize=7, color=MUTED, ha="left", va="top", transform=ax.transAxes)
    return fig


def plot_summary_table(summary: pd.DataFrame, suite: str):
    present = [tt for tt in C.TASK_TYPES
               if f"n_{tt}_tasks" in summary.columns and (summary[f"n_{tt}_tasks"] > 0).any()]
    cols = [Col(c.key, c.header, c.group.capitalize(), c.best, c.fmt) for c in SUMMARY_COLUMNS
            if c.group in present]
    return draw_metric_table(
        summary, [_label(r) for r in summary["row"]], cols,
        f"GeneICL {suite}: mean over the tasks every row scored",
        "Shaded, bold: best per column (↑ higher is better, ↓ lower). Rank: mean per-task rank, 1 = best.")


# ---- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> Path | None:
    from experiments.layout import ASSESSMENT_MANIFEST, ExperimentLayout

    args = parse_with_config(build_parser(), "geneicl_plot", argv)
    if not args.name and not args.list:
        sys.exit("Error: give --name (the comparison's folder under tasks/geneicl_<suite>/_plots/)")
    layout = ExperimentLayout.from_root(args.root)
    available = C.available_models(layout, args.suite)
    if args.list:
        print(f"models with GeneICL {args.suite} results: {available or 'none'}")
        print(f"plots: {PLOTS} (groups: {list(PLOT_GROUPS)})")
        return None
    try:
        out_dir = layout.task_plot_dir(task_name(args.suite), args.name)
    except ValueError as exc:
        sys.exit(f"Error: --name: {exc}")
    models = select_models(args.models, available)
    plots = select_plots(args.plots)

    runs, references = C.load_runs(layout, args.suite, models, args.reference,
                                   args.geneicl_repo, args.geneicl_results)
    table, dropped = C.per_task_table(runs)
    rows = list(runs)
    summary = C.summarize(table, rows)

    if (out_dir / ASSESSMENT_MANIFEST).exists():
        print(f"  [note] comparison '{args.name}' exists; its figures are overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "per_task.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"plotting {rows} -> {out_dir}")

    written: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for plot in plots:
        if plot == SUMMARY_PLOT:
            if table.empty:
                skipped[plot] = "no shared tasks"
                continue
            fig = plot_summary_table(summary, args.suite)
        else:
            metric = plot[len("heatmap_"):]
            mat = heatmap_matrix(table, metric, rows)
            if mat.empty:
                skipped[plot] = f"no {metric} values"
                print(f"  [skip] {plot}: no {metric} values on the shared tasks")
                continue
            fig = plot_heatmap(mat, metric, args.suite)
        written[plot] = save(fig, out_dir / plot, args.formats, args.dpi)

    (out_dir / ASSESSMENT_MANIFEST).write_text(json.dumps({
        "analysis": "geneicl_plot", "suite": args.suite, "name": args.name,
        "models": models, "references": references,
        "plots": written, "skipped": skipped, "n_tasks_dropped": dropped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
