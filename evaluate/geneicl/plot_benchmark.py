"""Figures comparing models on a GeneICL task suite.

    python -m evaluate.geneicl.plot_benchmark [--models unified_v1 pca] [--plots summary_table heatmap_auroc] \\
        [--suite benchmark] [--reference scfoundation en] [--config cfg.yaml]

Reads the ``run_probe`` results of the selected models (``--models``; default every model
with results on the suite) and computes the per-task and summary tables with
``compare.py``'s own functions, so the numbers are those of ``compare`` run on the same
models. ``--reference`` adds GeneICL's own result files as ``ref:<tag>`` columns, as in
``compare``.

Plots (``--plots``; default all):

- ``heatmap_<metric>``: task x model heatmap of one metric, the best value of each task in
  bold, plus a mean row. Metrics: auroc, auprc, balanced_accuracy, mcc, f1_macro
  (classification), spearman, pearson, r2, r2_pooled_oof (regression). ``heatmaps``
  selects all of them.
- ``summary_table``: the ``compare`` summary table, the best value of each metric shaded
  and in bold (lowest for the mean ranks).

Writes ``benchmarks/<selection>/geneicl_<suite>/`` under the experiments root, where
``<selection>`` is ``all_models`` when every available model is plotted and the plotted
models joined by ``_`` otherwise: one ``.png`` / ``.pdf`` per plot, the plotted numbers as
``per_task.csv`` and ``summary.csv``, and ``manifest.json`` (``"models"``, references,
plots).
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
from evaluate.geneicl import compare as C  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402
from evaluate.geneicl.run_probe import task_name  # noqa: E402

ALL_MODELS = "all_models"

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

# Summary-table columns: (summary column, header, task type, best = "max" | "min" | None).
SUMMARY_COLUMNS = [
    ("n_classification_tasks", "Tasks", "classification", None),
    ("cla_auroc", "AUROC", "classification", "max"),
    ("cla_auprc", "AUPRC", "classification", "max"),
    ("cla_balanced_accuracy", "Bal. acc.", "classification", "max"),
    ("cla_mcc", "MCC", "classification", "max"),
    ("cla_f1_macro", "Macro F1", "classification", "max"),
    ("cla_rank_auroc", "Rank AUROC", "classification", "min"),
    ("n_regression_tasks", "Tasks", "regression", None),
    ("reg_spearman", "Spearman", "regression", "max"),
    ("reg_pearson", "Pearson", "regression", "max"),
    ("reg_r2", "R²", "regression", "max"),
    ("reg_r2_pooled_oof", "R² pooled", "regression", "max"),
    ("reg_rank_spearman", "Rank Spearman", "regression", "min"),
]

# Reference palette (dataviz skill): one-hue blue ramp for magnitude, recessive ink.
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
MISSING = "#f0efec"
BEST_FILL = "#cde2fb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figures comparing models on a GeneICL task suite.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--geneicl-repo", type=Path, default=None,
                        help=f"GeneICL checkout, for --reference (default: ${S.GENEICL_ENV_VAR}, else ../GeneICL)")
    parser.add_argument("--geneicl-results", type=Path, default=None,
                        help="Directory of GeneICL's <tag>_<task_type>.csv (default: <geneicl>/eval/benchmark/results)")
    parser.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
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


def select_models(requested: list[str], available: list[str]) -> tuple[list[str], str]:
    """The models to plot, in order, and the selection directory name."""
    if not available:
        sys.exit("Error: no model has GeneICL results; run evaluate.geneicl.run_probe first")
    if not requested:
        return list(available), ALL_MODELS
    models = list(dict.fromkeys(requested))
    unknown = [m for m in models if m not in available]
    if unknown:
        sys.exit(f"Error: no GeneICL results for {unknown}; available: {available}")
    if set(models) == set(available):
        return models, ALL_MODELS
    return models, "_".join(models)


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

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": HAIRLINE, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return plt


def _save(fig, stem: Path, formats: list[str], dpi: int) -> list[str]:
    import matplotlib.pyplot as plt

    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path.name)
        print(f"  wrote {path}")
    plt.close(fig)
    return written


def _label(row: str) -> str:
    return row[len(C.REFERENCE_PREFIX):] + " (GeneICL)" if row.startswith(C.REFERENCE_PREFIX) else row


def _is_best(values: np.ndarray, best: str) -> np.ndarray:
    """Mask of the best finite value(s), ties included."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=bool)
    target = np.nanmax(values) if best == "max" else np.nanmin(values)
    return finite & np.isclose(values, target, rtol=0, atol=1e-12)


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

    plt = _pyplot()
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
        best = _is_best(all_values[i], "max")
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
    from matplotlib.patches import Rectangle

    plt = _pyplot()
    cols = [c for c in SUMMARY_COLUMNS if c[0] in summary.columns and summary[c[0]].notna().any()]
    present_types = [tt for tt in C.TASK_TYPES
                     if f"n_{tt}_tasks" in summary.columns and (summary[f"n_{tt}_tasks"] > 0).any()]
    cols = [c for c in cols if c[2] in present_types]
    labels = [_label(r) for r in summary["row"]]

    row_h, col_w = 0.34, 0.95
    name_w = max(1.4, 0.075 * max(len(x) for x in labels) + 0.3)
    # each column fits its bold header plus the direction arrow
    widths = [name_w] + [0.6 if c[3] is None else max(col_w, 0.085 * (len(c[1]) + 2) + 0.2) for c in cols]
    xs = np.concatenate([[0.0], np.cumsum(widths)])
    total_w = xs[-1]
    n = len(summary)
    header_rows = 2
    total_h = row_h * (n + header_rows) + 0.15

    fig = plt.figure(figsize=(total_w + 0.2, total_h + 0.75))
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.78])
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    # group header (task type) over its columns, with a hairline under it
    for tt in present_types:
        idx = [i for i, c in enumerate(cols) if c[2] == tt]
        x0, x1 = xs[1 + idx[0]], xs[1 + idx[-1] + 1]
        ax.text((x0 + x1) / 2, row_h * 0.5, tt.capitalize(), ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.plot([x0 + 0.08, x1 - 0.08], [row_h * 0.95] * 2, color=MUTED, linewidth=0.8)
    # column headers
    ax.text(0.08, row_h * 1.5, "Model", ha="left", va="center", fontsize=8.5, color=INK_2, fontweight="bold")
    for i, (_, header, _, best) in enumerate(cols):
        arrow = " ↓" if best == "min" else (" ↑" if best == "max" else "")
        ax.text((xs[1 + i] + xs[2 + i]) / 2, row_h * 1.5, header + arrow, ha="center", va="center",
                fontsize=8, color=INK_2, fontweight="bold")
    ax.plot([0, total_w], [row_h * 2] * 2, color=MUTED, linewidth=1.0)

    best_masks = {c[0]: _is_best(summary[c[0]].to_numpy(float), c[3]) for c in cols if c[3]}
    for r in range(n):
        y = row_h * (header_rows + r)
        ax.text(0.08, y + row_h / 2, labels[r], ha="left", va="center", fontsize=8.5, color=INK)
        for i, (col, _, _, best) in enumerate(cols):
            v = summary[col].iloc[r]
            x0, x1 = xs[1 + i], xs[2 + i]
            is_best = bool(best and best_masks[col][r])
            if is_best:
                ax.add_patch(Rectangle((x0 + 0.03, y + 0.03), x1 - x0 - 0.06, row_h - 0.06,
                                       facecolor=BEST_FILL, edgecolor="none"))
            if pd.isna(v):
                text, color = "n/a", MUTED
            elif best is None:
                text, color = f"{int(v)}", INK_2
            elif col.startswith(("cla_rank", "reg_rank")):
                text, color = f"{v:.2f}", INK
            else:
                text, color = f"{v:.3f}", INK
            ax.text((x0 + x1) / 2, y + row_h / 2, text, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if is_best else "normal", family="DejaVu Sans Mono")
        ax.plot([0, total_w], [y + row_h] * 2, color=HAIRLINE, linewidth=0.6)
    # separator between the two task types
    for tt in present_types[1:]:
        i0 = next(i for i, c in enumerate(cols) if c[2] == tt)
        ax.plot([xs[1 + i0]] * 2, [row_h * 0.2, total_h - 0.15], color=HAIRLINE, linewidth=0.8)

    fig.suptitle(f"GeneICL {suite}: mean over the tasks every row scored", x=0.02, ha="left",
                 fontsize=10, color=INK)
    fig.text(0.02, 0.04, "Shaded, bold: best per column (↑ higher is better, ↓ lower). "
             "Rank: mean per-task rank, 1 = best.", fontsize=7, color=MUTED, ha="left")
    return fig


# ---- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> Path | None:
    from experiments.layout import ASSESSMENT_MANIFEST, ExperimentLayout

    args = parse_with_config(build_parser(), "geneicl_plot", argv)
    layout = ExperimentLayout.from_root(args.root)
    available = C.available_models(layout, args.suite)
    if args.list:
        print(f"models with GeneICL {args.suite} results: {available or 'none'}")
        print(f"plots: {PLOTS} (groups: {list(PLOT_GROUPS)})")
        return None
    models, selection = select_models(args.models, available)
    plots = select_plots(args.plots)

    runs, references = C.load_runs(layout, args.suite, models, args.reference,
                                   args.geneicl_repo, args.geneicl_results)
    table, dropped = C.per_task_table(runs)
    rows = list(runs)
    summary = C.summarize(table, rows)

    out_dir = layout.benchmark_plot_dir(selection, task_name(args.suite))
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
        written[plot] = _save(fig, out_dir / plot, args.formats, args.dpi)

    (out_dir / ASSESSMENT_MANIFEST).write_text(json.dumps({
        "analysis": "geneicl_plot", "suite": args.suite, "selection": selection,
        "models": models, "references": references,
        "plots": written, "skipped": skipped, "n_tasks_dropped": dropped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
