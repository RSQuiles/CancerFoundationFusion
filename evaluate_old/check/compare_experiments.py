"""Compare unified_metrics.csv files across multiple ablation experiments.

Each CSV contains one row per model (index column = model name) and one
column per metric.  Models may share the same name across experiments; the
plot groups all models from the same experiment together, labelling each
group with the experiment name.

Usage
-----
    python compare_experiments.py \\
        --csvs /path/to/exp1/unified_metrics.csv /path/to/exp2/unified_metrics.csv \\
        --names "Experiment 1" "Experiment 2" \\
        --out comparison.png \\
        [--ncols 5]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric catalogue — same order as unified_metrics.py
# ---------------------------------------------------------------------------
_METRIC_META: list[tuple[str, str]] = [
    ("recon_pearson_r",                 "Reconstruction\nPearson R ↑"),
    ("recon_mae_bins",                  "Reconstruction\nMAE (bins, edge-aware) ↓"),
    ("paired_cosine_sim_mean",          "Paired Cosine Sim ↑"),
    ("paired_rank_mean",                "Paired Rank\n(cosine) ↓"),
    ("paired_l2_mean",                  "Paired L2 Dist ↓"),
    ("paired_rank_l2_mean",             "Paired Rank\n(L2) ↓"),
    ("agg_paired_cosine_pb_to_mean_sc", "Agg Consistency\n(paired, cosine) ↑"),
    ("agg_synth_cosine_pb_to_mean_sc",  "Agg Consistency\n(synth, cosine) ↑"),
    ("agg_paired_l2_pb_to_mean_sc",     "Agg Consistency\n(paired, L2) ↓"),
    ("agg_synth_l2_pb_to_mean_sc",      "Agg Consistency\n(synth, L2) ↓"),
    ("contrastive_cross_cosine_mean",   "Cross Cosine Sim ↑"),
    ("contrastive_within_bulk_cosine",  "Within-Bulk\nCosine Sim"),
    ("contrastive_within_pb_cosine",    "Within-PB\nCosine Sim"),
    ("contrastive_cross_l2_mean",       "Cross L2 Dist ↓"),
    ("contrastive_within_bulk_l2",      "Within-Bulk\nL2 Spread ↑"),
    ("contrastive_within_pb_l2",        "Within-PB\nL2 Spread ↑"),
    ("contrastive_within_bulk_over_cross_l2", "Within-Bulk / Cross\nL2 ratio (→1) ↑"),
    ("contrastive_within_pb_over_cross_l2",   "Within-PB / Cross\nL2 ratio (→1) ↑"),
    ("contrastive_energy_distance",     "Energy Distance\n(normalised, 0=same) ↓"),
    ("contrastive_wasserstein",         "Wasserstein ↓"),
    ("geometry_pr_frac_pooled",         "Participation Ratio\n(frac. of isotropic) ↑"),
    ("geometry_pr_pooled",              "Participation Ratio\n(effective dims) ↑"),
    ("geometry_modality_var_frac",      "Variance along the\nmodality axis ↓"),
]

# Metrics that share a common Y-axis range for direct comparability
_SHARED_YLIM_GROUPS: list[frozenset] = [
    frozenset({"paired_rank_mean", "paired_rank_median", "paired_rank_l2_mean", "paired_rank_l2_median"}),
]

# Gap (in bar-width units) inserted between experiment groups
_GROUP_GAP = 1.5

# One base colour per experiment; bars within an experiment are lightened progressively
_EXP_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csvs(
    csv_paths: list[Path],
    exp_names: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    result = []
    for path, name in zip(csv_paths, exp_names):
        df = pd.read_csv(path, index_col=0)
        # Keep only numeric columns; nested-dict columns (recon_mae_per_bin) are dropped
        df = df.select_dtypes(include="number")
        result.append((name, df))
    return result


def _exp_color(exp_idx: int, model_idx: int, n_models: int) -> tuple:
    """Base hue from experiment; successive models within the group are lightened."""
    base = matplotlib.colors.to_rgba(_EXP_PALETTE[exp_idx % len(_EXP_PALETTE)])
    lighten = 0.45 * (model_idx / max(n_models - 1, 1)) if n_models > 1 else 0.0
    r = base[0] + (1 - base[0]) * lighten
    g = base[1] + (1 - base[1]) * lighten
    b = base[2] + (1 - base[2]) * lighten
    return (r, g, b, 1.0)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_comparison(
    experiments: list[tuple[str, pd.DataFrame]],
    out_path: Path,
    ncols: int = 5,
) -> None:
    # Identify metrics present in at least one experiment
    all_cols: set[str] = set()
    for _, df in experiments:
        all_cols.update(df.columns)
    available = [(col, lbl) for col, lbl in _METRIC_META if col in all_cols]

    if not available:
        print("No plottable metrics found across the provided CSVs.", file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # Pre-compute flat bar layout
    # Each entry: (x_position, exp_name, model_name, color)
    # ------------------------------------------------------------------
    entries: list[tuple[float, str, str, tuple]] = []
    exp_spans: list[tuple[float, float, str]] = []   # (x_lo, x_hi, exp_name)

    cursor = 0.0
    for exp_idx, (exp_name, df) in enumerate(experiments):
        models = df.index.tolist()
        span_lo = cursor
        for m_idx, model_name in enumerate(models):
            color = _exp_color(exp_idx, m_idx, len(models))
            entries.append((cursor, exp_name, model_name, color))
            cursor += 1.0
        exp_spans.append((span_lo, cursor - 1.0, exp_name))
        cursor += _GROUP_GAP

    x_positions = [e[0] for e in entries]
    bar_colors   = [e[3] for e in entries]
    bar_labels   = [e[2] for e in entries]   # model name per bar

    # ------------------------------------------------------------------
    # Build shared y-lim mapping
    # ------------------------------------------------------------------
    col_to_ylim_group: dict[str, int] = {}
    for gid, fset in enumerate(_SHARED_YLIM_GROUPS):
        for col in fset:
            col_to_ylim_group[col] = gid
    ylim_group_ranges: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Draw subplots
    # ------------------------------------------------------------------
    n_metrics = len(available)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for metric_idx, (col, lbl) in enumerate(available):
        row_i, col_i = divmod(metric_idx, ncols)
        ax = axes[row_i][col_i]

        # Value per bar entry
        ys: list[float] = []
        for _, exp_name, model_name, _ in entries:
            df = dict(experiments)[exp_name]
            if col in df.columns and model_name in df.index:
                v = df.loc[model_name, col]
                ys.append(float(v) if not pd.isna(v) else float("nan"))
            else:
                ys.append(float("nan"))

        ax.bar(x_positions, ys, color=bar_colors, width=0.75, zorder=3)

        # Background shading per experiment group (alternating)
        for span_idx, (span_lo, span_hi, _) in enumerate(exp_spans):
            ax.axvspan(
                span_lo - 0.45, span_hi + 0.45,
                color="lightsteelblue" if span_idx % 2 == 0 else "lightyellow",
                alpha=0.18, zorder=0,
            )

        # Vertical separator lines between groups
        for span_lo, span_hi, _ in exp_spans[:-1]:
            sep_x = span_hi + _GROUP_GAP / 2
            ax.axvline(sep_x, color="grey", linewidth=0.7, linestyle="--", zorder=2)

        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlim(x_positions[0] - 0.7, x_positions[-1] + 0.7)

        finite_ys = [y for y in ys if not np.isnan(y)]
        if finite_ys:
            ax.set_ylim(bottom=0, top=max(finite_ys) * 1.15)

        # Experiment name labels below tick labels
        # Drawn in axes-fraction coordinates so they sit just under the x-axis
        ax_x0, ax_x1 = ax.get_xlim()
        ax_width = ax_x1 - ax_x0
        for span_lo, span_hi, exp_name in exp_spans:
            centre_data = (span_lo + span_hi) / 2
            centre_frac = (centre_data - ax_x0) / ax_width
            ax.text(
                centre_frac, -0.28,
                exp_name,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=6, color="dimgrey", style="italic",
            )

        # Shared y-lim accumulation
        gid = col_to_ylim_group.get(col)
        if gid is not None and finite_ys:
            ylim_group_ranges.setdefault(gid, []).extend(finite_ys)

    # Apply shared y-limits
    for metric_idx, (col, _) in enumerate(available):
        gid = col_to_ylim_group.get(col)
        if gid is None or gid not in ylim_group_ranges:
            continue
        row_i, col_i = divmod(metric_idx, ncols)
        ax = axes[row_i][col_i]
        ymax = max(ylim_group_ranges[gid])
        ax.set_ylim(0, ymax * 1.15)

    # Hide unused subplots
    for idx in range(n_metrics, nrows * ncols):
        axes[divmod(idx, ncols)[0]][divmod(idx, ncols)[1]].set_visible(False)

    # Legend: one patch per experiment
    legend_handles = [
        mpatches.Patch(color=_exp_color(i, 0, 1), label=name)
        for i, (name, _) in enumerate(experiments)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(experiments), 6),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle("Experiment comparison", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csvs", nargs="+", required=True, metavar="PATH",
        help="unified_metrics.csv files, one per experiment",
    )
    p.add_argument(
        "--names", nargs="+", required=True, metavar="NAME",
        help="Experiment name for each CSV (same order as --csvs)",
    )
    p.add_argument(
        "--out", default="experiment_comparison.png",
        help="Output PNG path (default: experiment_comparison.png)",
    )
    p.add_argument(
        "--ncols", type=int, default=5,
        help="Subplot columns (default: 5)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_paths = [Path(p) for p in args.csvs]

    if len(csv_paths) != len(args.names):
        sys.exit(
            f"--csvs and --names must have the same length "
            f"({len(csv_paths)} vs {len(args.names)})"
        )
    for p in csv_paths:
        if not p.exists():
            sys.exit(f"CSV not found: {p}")

    experiments = _load_csvs(csv_paths, args.names)
    plot_comparison(experiments, Path(args.out), ncols=args.ncols)


if __name__ == "__main__":
    main()
