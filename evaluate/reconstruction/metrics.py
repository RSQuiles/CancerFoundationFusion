"""Reconstruction metrics on masked entries (numpy, pandas and scipy only).

Two families:

- **native-space** metrics (``mse``, ``mae`` and, for bins, exact / within-one-bin
  accuracy) depend on the model's target scale, so they only compare models that share
  one;
- **space-free** metrics (per-sample Pearson / Spearman over the masked genes, the
  zero-vs-non-zero AUROC, the global Spearman) are invariant to a monotone change of
  scale and are what ``compare.py`` uses across models.

Every metric is also computed for two baselines on the same entries, so a number has a
reference: ``gene_mean`` (each gene's mean target over the plan's samples where it was
NOT masked) and ``sample_mean`` (each sample's mean target over its context genes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTORS = ("pred", "gene_mean", "sample_mean")


def baselines(targets: np.ndarray, plan) -> tuple[np.ndarray, np.ndarray]:
    """``(gene_mean per gene, sample_mean per sample)`` from the unmasked targets."""
    visible = targets.copy()
    for i in range(plan.n_samples):
        visible[i, plan.masked(i)] = np.nan
    with np.errstate(invalid="ignore"), _quiet():
        gene_mean = np.nanmean(visible, axis=0)
        sample_mean = np.array([np.nanmean(targets[i, plan.context(i)]) if len(plan.context(i)) else np.nan
                                for i in range(plan.n_samples)])
    return gene_mean, sample_mean


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def entries_frame(result: dict, plan, obs: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per scored masked entry: sample, gene, target, pred, baselines (+ obs columns)."""
    targets = result["targets"]
    gene_mean, sample_mean = baselines(targets, plan)
    row, gene = result["row"].astype(int), result["gene"].astype(int)
    frame = pd.DataFrame({
        "row": row, "sample": plan.obs_names[row], "gene": plan.genes[gene],
        "target": targets[row, gene], "pred": result["pred"],
        "gene_mean": gene_mean[gene], "sample_mean": sample_mean[row],
    })
    for col in group_cols or []:
        if col in obs.columns:
            frame[col] = obs[col].astype(str).to_numpy()[row]
    return frame[np.isfinite(frame["target"].to_numpy())].reset_index(drop=True)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    return _corr(rankdata(x), rankdata(y))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC of ``scores`` for ``labels`` (True = positive); ties count half."""
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_sample(frame: pd.DataFrame, col: str = "pred") -> pd.DataFrame:
    """Per-sample Pearson / Spearman of ``col`` against the target (all and non-zero targets)."""
    out = []
    for sample, g in frame.groupby("sample", sort=False):
        t, p = g["target"].to_numpy(), g[col].to_numpy()
        nz = t != 0
        out.append({"sample": sample, "n": len(g), "pearson": _corr(t, p), "spearman": _spearman(t, p),
                    "spearman_nonzero": _spearman(t[nz], p[nz]) if nz.sum() >= 3 else np.nan})
    return pd.DataFrame(out)


def _summary(values: pd.Series, prefix: str) -> dict:
    v = values.dropna()
    return {f"{prefix}_mean": float(v.mean()) if len(v) else np.nan,
            f"{prefix}_median": float(v.median()) if len(v) else np.nan,
            f"{prefix}_std": float(v.std()) if len(v) > 1 else np.nan}


def space_free(frame: pd.DataFrame, col: str = "pred") -> dict:
    """Scale-invariant metrics of predictor ``col``."""
    ps = per_sample(frame, col)
    t, p = frame["target"].to_numpy(), frame[col].to_numpy()
    zero = t == 0
    out = {"n_entries": int(len(frame)), "n_samples": int(frame["sample"].nunique()),
           "spearman_global": _spearman(t, p), "zero_frac": float(zero.mean()) if len(t) else np.nan,
           # A lower prediction should mean "more likely zero", hence the minus sign.
           "auroc_zero": auroc(-p, zero)}
    for metric in ("pearson", "spearman", "spearman_nonzero"):
        out.update(_summary(ps[metric], metric))
    return out


def native(frame: pd.DataFrame, space: str, col: str = "pred") -> dict:
    """Metrics in the model's own target scale."""
    t, p = frame["target"].to_numpy(dtype=float), frame[col].to_numpy(dtype=float)
    err = p - t
    zero = t == 0
    out = {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
           "mae_zero_target": float(np.mean(np.abs(err[zero]))) if zero.any() else np.nan,
           "mae_nonzero_target": float(np.mean(np.abs(err[~zero]))) if (~zero).any() else np.nan}
    if space == "bins":
        rounded = np.rint(p)
        out["acc_exact"] = float(np.mean(rounded == t))
        out["acc_within1"] = float(np.mean(np.abs(rounded - t) <= 1))
    return out


def mae_by_target(frame: pd.DataFrame, space: str, col: str = "pred", n_groups: int = 10) -> pd.DataFrame:
    """MAE grouped by the target: by true bin (bins) or by target decile (other spaces)."""
    t = frame["target"]
    if space == "bins":
        width = max(1, int(np.ceil((t.max() + 1) / n_groups)))
        key = (t // width * width).astype(int).rename("target_group")
    else:
        key = pd.qcut(t, q=n_groups, duplicates="drop").astype(str).rename("target_group")
    err = (frame[col] - t).abs()
    return err.groupby(key).agg(["mean", "count"]).rename(columns={"mean": "mae", "count": "n"}).reset_index()


def evaluate(frame: pd.DataFrame, space: str) -> dict:
    """``{predictor: {metric: value}}`` for the model and both baselines.

    A baseline is undefined for a gene masked in every sample (no unmasked value to
    average); those entries are left out of that baseline only, as ``n_entries`` shows.
    """
    out = {}
    for col in PREDICTORS:
        sub = frame[np.isfinite(frame[col].to_numpy(dtype=float))]
        out[col] = {**native(sub, space, col), **space_free(sub, col)} if len(sub) else {"n_entries": 0}
    return out
