#!/usr/bin/env python3
"""
Evaluate CFF survival functions with SurvBoard metrics.

Reads the per-fold survival-function CSVs written by SurvBoardTask and computes
Antolini concordance, Integrated Brier Score (IBS), and D-calibration — the
same metrics as SurvBoard's evaluate_metrics.py.

Run this script from any conda environment that has pycox, sksurv, and
survival_evaluation installed (e.g. the SurvBoard environment).

Writes two outputs
------------------
1.  {ablation_dir}/{model_name}/metrics/results_survival.json
    Mean across all folds and cancer types — consumed by ablation_benchmark.py.

2.  {ablation_dir}/{model_name}/metrics/results_survival_detailed.csv
    Per-(cohort, cancer, fold) breakdown for detailed analysis.

Usage
-----
    python evaluate/finetune/tasks/evaluate_survboard_metrics.py \\
        --data-dir     /path/to/survboard/parquets \\
        --splits-dir   /path/to/survboard/splits \\
        --results-dir  /path/to/cff/survival/results \\
        --ablation-dir /path/to/ablation_results \\
        --cohorts      TCGA \\
        --cancer-types BRCA LUAD PAAD \\
        [--model-name  cff] \\
        [--ibs-grid-len 100]

Expected file layout produced by SurvBoardTask
----------------------------------------------
    {results_dir}/{COHORT}/{CANCER}/cff/split_0.csv
    {results_dir}/{COHORT}/{CANCER}/cff/split_1.csv
    ...

Each CSV: rows = test patients, columns = float time-grid values + metadata
    (model_type, modality, project, cancer, split).

Expected split CSV layout (SurvBoard format)
--------------------------------------------
    {splits_dir}/{COHORT}/{CANCER}_train_splits.csv
    {splits_dir}/{COHORT}/{CANCER}_test_splits.csv
    Rows = folds, columns = NaN-padded integer patient indices into the parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pycox.evaluation.eval_surv import EvalSurv
from scipy.stats import chi2
from sksurv.nonparametric import kaplan_meier_estimator
from survival_evaluation.utility import to_array

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# EvalSurvDCalib copy from survboard/scripts/python/evaluate_metrics.py
# --------------------------------------------------------------------------- #

def _d_calibration(event_indicators, predictions, bins: int = 5) -> dict:
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions      = to_array(predictions)

    bin_index              = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
    censored_bin_indexes   = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]

    censored_predictions            = predictions[~event_indicators]
    censored_contribution           = 1 - (censored_bin_indexes / bins) * (1 / censored_predictions)
    censored_following_contribution = 1 / (bins * censored_predictions)

    contribution_pattern  = np.tril(np.ones([bins, bins]), k=-1).astype(bool)
    following_contribs    = np.matmul(censored_following_contribution, contribution_pattern[censored_bin_indexes])
    single_contribs       = np.matmul(censored_contribution, np.eye(bins)[censored_bin_indexes])
    uncensored_contribs   = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    bin_count             = single_contribs + following_contribs + uncensored_contribs

    chi2_stat = np.sum(np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins))
    return dict(
        p_value          = 1 - chi2.cdf(chi2_stat, bins - 1),
        test_statistic   = chi2_stat,
        bin_proportions  = bin_count / len(predictions),
    )


class EvalSurvDCalib(EvalSurv):
    def __init__(self, surv, durations, events, censor_surv=None, censor_durations=None, steps="post"):
        super().__init__(surv, durations, events, censor_surv, censor_durations, steps)

    def d_calibration_(self, bins: int = 5, p_value: bool = False) -> float:
        indices = self.idx_at_times(self.durations)
        d_calib = _d_calibration(
            self.events,
            np.array([self.surv.iloc[indices[ix], ix] for ix in range(self.events.shape[0])]),
            bins=bins,
        )
        return d_calib["p_value"] if p_value else d_calib["test_statistic"]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def _load_survival_labels(data_dir: Path, cohort: str, cancer: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (OS_days, OS_event) arrays from the SurvBoard parquet."""
    path = data_dir / f"{cohort}_{cancer}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df        = pd.read_parquet(path)
    event_col = "OS" if "OS" in df.columns else "OS_event"
    return df["OS_days"].to_numpy(dtype=np.float64), df[event_col].to_numpy(dtype=np.float64)


def _load_fold_indices(
    splits_dir: Path, cohort: str, cancer: str, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_ix, test_ix) for one fold from SurvBoard split CSVs."""
    train_path = splits_dir / cohort / f"{cancer}_train_splits.csv"
    test_path  = splits_dir / cohort / f"{cancer}_test_splits.csv"
    train_ix   = pd.read_csv(train_path).iloc[fold, :].dropna().astype(int).to_numpy()
    test_ix    = pd.read_csv(test_path).iloc[fold, :].dropna().astype(int).to_numpy()
    return train_ix, test_ix


def _count_folds(splits_dir: Path, cohort: str, cancer: str) -> int:
    """Return the number of rows in the split CSV (= number of folds)."""
    path = splits_dir / cohort / f"{cancer}_test_splits.csv"
    return len(pd.read_csv(path))


def _load_survival_csv(
    results_dir: Path, cohort: str, cancer: str, model_name: str, fold: int
) -> pd.DataFrame | None:
    """
    Load a survival-function CSV and return it as a (time_grid × patients) DataFrame.
    Returns None if the file does not exist.
    """
    path = results_dir / cohort / cancer / model_name / f"split_{fold}.csv"
    if not path.exists():
        return None

    df        = pd.read_csv(path)
    meta_cols = {"model_type", "modality", "project", "cancer", "split"}
    time_cols = [c for c in df.columns if c not in meta_cols]

    surv_df         = df[time_cols].copy()
    surv_df.columns = surv_df.columns.astype(float)
    surv_df         = surv_df.T.sort_index()   # (n_times × n_patients)
    return surv_df


# --------------------------------------------------------------------------- #
# Per-fold metric computation
# --------------------------------------------------------------------------- #

def _compute_fold_metrics(
    surv_df:     pd.DataFrame,
    time:        np.ndarray,
    event:       np.ndarray,
    train_ix:    np.ndarray,
    test_ix:     np.ndarray,
    ibs_grid_len: int,
) -> dict[str, float]:
    """
    Compute EvalSurvDCalib metrics for one fold.

    Falls back to the training-set Kaplan-Meier curve if any survival
    predictions contain NaN — matching evaluate_metrics.py behaviour.
    """
    if np.any(np.isnan(surv_df.values)):
        warnings.warn("NaN in survival predictions; replacing with KM baseline.")
        x, y    = kaplan_meier_estimator(event[train_ix].astype(bool), time[train_ix])
        n_test  = len(test_ix)
        surv_df = pd.DataFrame(np.tile(y, (n_test, 1)), columns=x).T.sort_index()

    ev = EvalSurvDCalib(
        surv           = surv_df,
        durations      = time[test_ix],
        events         = event[test_ix],
        censor_surv    = "km",
        steps          = "post",
    )

    time_grid = np.linspace(time[test_ix].min(), time[test_ix].max(), ibs_grid_len)

    return {
        "antolini_concordance": float(ev.concordance_td()),
        "ibs":                  float(ev.integrated_brier_score(time_grid)),
        "d_calibration":        float(ev.d_calibration_()),
        "n_events":             int(event[test_ix].astype(bool).sum()),
        "event_rate":           float(event[test_ix].mean()),
    }


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def evaluate_all(
    data_dir:     Path,
    splits_dir:   Path,
    results_dir:  Path,
    ablation_dir: Path,
    cohorts:      list[str],
    cancer_types: list[str],
    model_name:   str,
    ibs_grid_len: int,
) -> None:
    rows: list[dict] = []

    for cohort in cohorts:
        for cancer in cancer_types:
            parquet_path = data_dir / f"{cohort}_{cancer}.parquet"
            if not parquet_path.exists():
                log.warning(f"Parquet not found, skipping: {parquet_path}")
                continue

            split_csv = splits_dir / cohort / f"{cancer}_test_splits.csv"
            if not split_csv.exists():
                log.warning(f"Split CSV not found, skipping: {split_csv}")
                continue

            time, event = _load_survival_labels(data_dir, cohort, cancer)
            n_folds     = _count_folds(splits_dir, cohort, cancer)

            n_found = 0
            for fold in range(n_folds):
                surv_df = _load_survival_csv(results_dir, cohort, cancer, model_name, fold)
                if surv_df is None:
                    log.debug(f"  {cohort}/{cancer} fold {fold}: CSV not found, skipping.")
                    continue

                try:
                    train_ix, test_ix = _load_fold_indices(splits_dir, cohort, cancer, fold)
                    metrics = _compute_fold_metrics(
                        surv_df, time, event, train_ix, test_ix, ibs_grid_len
                    )
                except Exception as exc:
                    log.warning(f"  {cohort}/{cancer} fold {fold}: error — {exc}")
                    continue

                rows.append({
                    "cohort":                cohort,
                    "cancer":                cancer,
                    "fold":                  fold,
                    "antolini_concordance":  metrics["antolini_concordance"],
                    "ibs":                   metrics["ibs"],
                    "d_calibration":         metrics["d_calibration"],
                    "n_events":              metrics["n_events"],
                    "event_rate":            metrics["event_rate"],
                })
                n_found += 1
                log.info(
                    f"  {cohort}/{cancer} fold {fold}: "
                    f"concordance={metrics['antolini_concordance']:.4f}, "
                    f"ibs={metrics['ibs']:.4f}, "
                    f"d_cal={metrics['d_calibration']:.4f}"
                )

            if n_found:
                log.info(f"{cohort}/{cancer}: {n_found}/{n_folds} folds evaluated.")
            else:
                log.warning(f"{cohort}/{cancer}: no survival CSVs found under "
                            f"{results_dir / cohort / cancer / model_name}/")

    if not rows:
        log.error("No metrics computed — check --results-dir and --model-name.")
        sys.exit(1)

    detail_df = pd.DataFrame(rows)

    # ---- Aggregated metrics (mean across all folds and cancer types) --------
    mean_keys = ["antolini_concordance", "ibs", "d_calibration", "event_rate"]
    aggregated: dict[str, float | int] = {
        k: float(detail_df[k].mean()) for k in mean_keys
    }
    aggregated["n_events"]     = int(detail_df["n_events"].sum())
    aggregated["n_folds_evaluated"] = len(detail_df)

    # ---- Save outputs -------------------------------------------------------
    out_dir = ablation_dir / model_name / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results_survival.json"
    with open(json_path, "w") as fh:
        json.dump(aggregated, fh, indent=2)
    log.info(f"Saved aggregated metrics → {json_path}")

    csv_path = out_dir / "results_survival_detailed.csv"
    detail_df.to_csv(csv_path, index=False)
    log.info(f"Saved detailed metrics  → {csv_path}")

    # ---- Summary to stdout --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Model: {model_name}   Folds evaluated: {aggregated['n_folds_evaluated']}")
    print(f"{'='*60}")
    print(f"  Antolini concordance : {aggregated['antolini_concordance']:.4f}")
    print(f"  IBS                  : {aggregated['ibs']:.4f}")
    print(f"  D-calibration stat   : {aggregated['d_calibration']:.4f}")
    print(f"  Total events         : {aggregated['n_events']}")
    print(f"  Mean event rate      : {aggregated['event_rate']:.3f}")
    print(f"{'='*60}\n")

    # Per-cancer summary
    per_cancer = (
        detail_df
        .groupby(["cohort", "cancer"])[mean_keys]
        .mean()
        .round(4)
    )
    print("Per-cancer means:")
    print(per_cancer.to_string())
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate CFF survival functions with SurvBoard metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", required=True, type=Path,
        help="Directory containing {COHORT}_{CANCER}.parquet files.",
    )
    p.add_argument(
        "--splits-dir", required=True, type=Path,
        help="Directory containing SurvBoard split CSVs ({COHORT}/{CANCER}_*_splits.csv).",
    )
    p.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory where SurvBoardTask wrote survival CSVs "
             "({cohort}/{cancer}/{model_name}/split_*.csv).",
    )
    p.add_argument(
        "--ablation-dir", required=True, type=Path,
        help="Root ablation directory. Metrics are written to "
             "{ablation_dir}/{model_name}/metrics/.",
    )
    p.add_argument(
        "--cohorts", nargs="+", required=True,
        help="Cohort names to evaluate (e.g. TCGA METABRIC).",
    )
    p.add_argument(
        "--cancer-types", nargs="+", required=True,
        help="Cancer type codes to evaluate (e.g. BRCA LUAD PAAD).",
    )
    p.add_argument(
        "--model-name", default="cff",
        help="Sub-directory name inside results-dir and output name in ablation-dir "
             "(default: cff).",
    )
    p.add_argument(
        "--ibs-grid-len", type=int, default=100,
        help="Number of time points for IBS integration (default: 100).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    for name, path in [
        ("--data-dir",    args.data_dir),
        ("--splits-dir",  args.splits_dir),
        ("--results-dir", args.results_dir),
    ]:
        if not path.exists():
            log.error(f"{name} does not exist: {path}")
            sys.exit(1)

    log.info(f"Evaluating model '{args.model_name}' across {len(args.cohorts)} cohort(s) "
             f"and {len(args.cancer_types)} cancer type(s).")

    evaluate_all(
        data_dir     = args.data_dir.resolve(),
        splits_dir   = args.splits_dir.resolve(),
        results_dir  = args.results_dir.resolve(),
        ablation_dir = args.ablation_dir.resolve(),
        cohorts      = args.cohorts,
        cancer_types = args.cancer_types,
        model_name   = args.model_name,
        ibs_grid_len = args.ibs_grid_len,
    )


if __name__ == "__main__":
    main()
