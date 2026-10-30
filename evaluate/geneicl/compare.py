"""Compare models on a GeneICL task suite, optionally next to GeneICL's own result files.

    python -m evaluate.geneicl.compare --models unified_v1 pca [--suite benchmark] [--name main] \\
        [--reference scfoundation bulkformer en geneicl_trm_s0]

Reads ``results/tasks/geneicl_<suite>/<model>/probe_*.csv`` (``run_probe.py``) per model and,
for every ``--reference`` tag, ``<tag>_{classification,regression}.csv`` from GeneICL's
results directory (``--geneicl-results``, default ``<geneicl>/eval/benchmark/results``);
a missing reference is reported and skipped. Rows are compared on the tasks every row
scored. Per task the fold metrics are averaged, then the tasks:

- classification: AUROC, AUPRC, balanced accuracy, MCC, macro F1;
- regression: Spearman, Pearson, mean-of-folds R2 and pooled out-of-fold R2 (all folds'
  predictions of a task pooled, the GeneICL paper's protocol; needs ``*.oof.csv``);
- the mean rank over tasks of AUROC / Spearman (1 = best).

Writes ``results/assessment/geneicl_<suite>[_<name>]/``: ``summary.csv`` (one row per
model or reference), ``per_task.csv`` and ``manifest.json`` (``"models"`` = the models of
this repository; the references are listed apart, since they are GeneICL's files).
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
from evaluate.geneicl import suite as S  # noqa: E402
from evaluate.geneicl.run_probe import OUT_STEM, task_name  # noqa: E402

TASK_TYPES = ("classification", "regression")
METRICS = {
    "classification": ["auroc", "auprc", "balanced_accuracy", "mcc", "f1_macro"],
    "regression": ["spearman", "pearson", "r2"],
}
RANK_METRIC = {"classification": "auroc", "regression": "spearman"}
REFERENCE_PREFIX = "ref:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models on a GeneICL task suite.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--geneicl-repo", type=Path, default=None,
                        help=f"GeneICL checkout, for --reference (default: ${S.GENEICL_ENV_VAR}, else ../GeneICL)")
    parser.add_argument("--geneicl-results", type=Path, default=None,
                        help="Directory of GeneICL's <tag>_<task_type>.csv (default: <geneicl>/eval/benchmark/results)")
    parser.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
    parser.add_argument("--models", nargs="*", default=[], help="Models scored by run_probe")
    parser.add_argument("--reference", nargs="*", default=[],
                        help="GeneICL result tags shown alongside (e.g. scfoundation en geneicl_trm_s0)")
    parser.add_argument("--name", default=None, help="Suffix of the assessment directory")
    return parser


def read_run(prefix: Path) -> dict[str, pd.DataFrame | None]:
    """``<prefix>_<task_type>.csv`` and ``<prefix>_regression.oof.csv``; None when absent."""
    out: dict[str, pd.DataFrame | None] = {}
    for tt in TASK_TYPES:
        path = prefix.parent / f"{prefix.name}_{tt}.csv"
        out[tt] = pd.read_csv(path) if path.exists() else None
    oof = prefix.parent / f"{prefix.name}_regression.oof.csv"
    out["oof"] = pd.read_csv(oof) if oof.exists() else None
    return out


def pooled_r2(oof: pd.DataFrame | None) -> dict[str, float]:
    """Per task: R2 of all folds' out-of-fold predictions pooled."""
    if oof is None or oof.empty:
        return {}
    out = {}
    for task, g in oof.groupby("task_id"):
        y, p = g["y_true"].to_numpy(float), g["pred"].to_numpy(float)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[str(task)] = 1.0 - float(((y - p) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return out


def per_task_table(runs: dict[str, dict]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fold-averaged metrics on the tasks every row scored; ``{task_type: n dropped}``."""
    frames, dropped = [], {}
    for tt in TASK_TYPES:
        present = {label: run[tt] for label, run in runs.items() if run[tt] is not None and len(run[tt])}
        if len(present) < len(runs):
            absent = sorted(set(runs) - set(present))
            print(f"  [{tt}] no results for {absent}; {tt} is left out of the comparison")
            continue
        shared = set.intersection(*(set(df["task_id"]) for df in present.values()))
        everywhere = set.union(*(set(df["task_id"]) for df in present.values()))
        dropped[tt] = len(everywhere - shared)
        if dropped[tt]:
            print(f"  [{tt}] {dropped[tt]} tasks not scored by every row are left out")
        for label, df in present.items():
            cols = [c for c in METRICS[tt] if c in df.columns]
            sub = df[df["task_id"].isin(shared)]
            agg = sub.groupby(["task_id", "dataset"], as_index=False)[cols].mean()
            agg["n_folds"] = sub.groupby(["task_id", "dataset"])["fold"].nunique().to_numpy()
            if tt == "regression":
                agg["r2_pooled_oof"] = agg["task_id"].map(pooled_r2(runs[label]["oof"]))
            agg.insert(0, "task_type", tt)
            agg.insert(0, "row", label)
            frames.append(agg)
    if not frames:
        return pd.DataFrame(), dropped
    table = pd.concat(frames, ignore_index=True)
    for tt, metric in RANK_METRIC.items():
        mask = table["task_type"] == tt
        if mask.any():
            table.loc[mask, f"rank_{metric}"] = (
                table[mask].groupby("task_id")[metric].rank(ascending=False, method="average"))
    return table, dropped


def summarize(table: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        row = {"row": label}
        for tt in TASK_TYPES:
            sub = table[(table["row"] == label) & (table["task_type"] == tt)] if len(table) else table
            row[f"n_{tt}_tasks"] = len(sub)
            cols = METRICS[tt] + (["r2_pooled_oof"] if tt == "regression" else []) + [f"rank_{RANK_METRIC[tt]}"]
            for c in cols:
                row[f"{tt[:3]}_{c}"] = float(sub[c].mean()) if c in sub.columns and len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    from experiments.layout import ExperimentLayout

    args = parse_with_config(build_parser(), "geneicl_compare", argv)
    if not (args.models or args.reference):
        sys.exit("Error: give --models and/or --reference")
    layout = ExperimentLayout.from_root(args.root)

    runs: dict[str, dict] = {}
    for model in args.models:
        prefix = layout.task_dir(task_name(args.suite), model) / OUT_STEM
        run = read_run(prefix)
        if all(run[tt] is None for tt in TASK_TYPES):
            sys.exit(f"Error: no {prefix}_*.csv; run evaluate.geneicl.run_probe for '{model}' first")
        runs[model] = run
    references = []
    if args.reference:
        results = args.geneicl_results or S.resolve_geneicl(args.geneicl_repo) / "eval" / "benchmark" / "results"
        for tag in args.reference:
            run = read_run(results / tag)
            if all(run[tt] is None for tt in TASK_TYPES):
                print(f"  [warn] no GeneICL results for '{tag}' in {results}; skipped")
                continue
            runs[REFERENCE_PREFIX + tag] = run
            references.append(tag)
    if not runs:
        sys.exit("Error: nothing to compare")

    table, dropped = per_task_table(runs)
    summary = summarize(table, list(runs))
    out_dir = layout.assessment_dir(task_name(args.suite) + (f"_{args.name}" if args.name else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "per_task.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "analysis": "geneicl", "suite": args.suite, "models": list(args.models), "references": references,
        "n_tasks": {tt: int((table["task_type"] == tt).sum() // max(len(runs), 1)) if len(table) else 0
                    for tt in TASK_TYPES},
        "n_tasks_dropped": dropped,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    show = ["row", "n_classification_tasks", "cla_auroc", "cla_rank_auroc",
            "n_regression_tasks", "reg_spearman", "reg_r2_pooled_oof", "reg_rank_spearman"]
    with pd.option_context("display.width", 160, "display.max_columns", 12, "display.precision", 3):
        print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    print(f"wrote {out_dir}")
    return summary


if __name__ == "__main__":
    main()
