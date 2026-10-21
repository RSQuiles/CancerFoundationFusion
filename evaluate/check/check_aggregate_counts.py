"""Verify _aggregate_sc no longer saturates on raw counts (the cdd-arm bug).

Reimplements the two code paths over the same inputs rather than constructing a
full BulkSCCollator (which needs the training stack), and asserts on the numbers
the two produce.
"""
import numpy as np

FLOAT32_MAX = float(np.finfo(np.float32).max)
rng = np.random.default_rng(0)

N_CELLS, N_GENES = 10, 200
# Raw counts, as written by data_preprocess (nothing normalizes them).
counts = rng.poisson(8, size=(N_CELLS, N_GENES)).astype(np.float64)
# A few high-expression genes, which is where the saturation bites: expm1(89) is
# already past float32 max.
counts[:, :5] += rng.integers(80, 400, size=(N_CELLS, 5))


def aggregate(exprs_per_cell, sc_counts: bool, input_data: str = "counts"):
    """Mirror of BulkSCCollator._aggregate_sc's value handling."""
    parts = []
    for exprs in exprs_per_cell:
        e = np.asarray(exprs, dtype=np.float64)
        if not sc_counts:
            e = np.clip(np.expm1(e), 0, FLOAT32_MAX).astype(np.float32)
        else:
            e = e.astype(np.float32)
        parts.append(e)
    expr_sum = np.sum(np.stack(parts), axis=0).astype(np.float64)
    if input_data != "counts":
        total = expr_sum.sum()
        if total > 0:
            expr_sum = expr_sum / total * 1e6
        expr_sum = np.log1p(expr_sum)
    return expr_sum


cells = [counts[i] for i in range(N_CELLS)]

old = aggregate(cells, sc_counts=False)   # the bug: expm1 on counts
new = aggregate(cells, sc_counts=True)    # fixed: sum counts directly
truth = counts.sum(axis=0)                # what a pseudobulk of counts should be

print(f"input counts:     min={counts.min():.0f} max={counts.max():.0f}")
print(f"OLD (expm1):      max={old.max():.3e}  saturated genes="
      f"{int((old >= FLOAT32_MAX).sum())}/{N_GENES}")
print(f"NEW (sum counts): max={new.max():.0f}   saturated genes="
      f"{int((new >= FLOAT32_MAX).sum())}/{N_GENES}")
print(f"expected (truth): max={truth.max():.0f}")

assert (old >= FLOAT32_MAX).any(), "fixture should trigger saturation on the old path"
assert not (new >= FLOAT32_MAX).any(), "fixed path must not saturate"
assert np.allclose(new, truth), "fixed path must equal the true count sum"

# The damage is not a rescaling: the gene *ranking* is destroyed, which is what
# makes the CDD source centroids wrong rather than merely scaled.
from scipy.stats import spearmanr
r_old = spearmanr(old, truth).statistic
r_new = spearmanr(new, truth).statistic
print(f"\nSpearman vs true count-sum profile:  OLD={r_old:.4f}   NEW={r_new:.4f}")
n_tied_old = N_GENES - len(np.unique(old))
print(f"genes collapsed to identical values by saturation (OLD): {n_tied_old}/{N_GENES}")
assert r_new > 0.999, r_new
assert r_old < r_new, "old path should rank genes worse than the fixed one"

print("\nAGGREGATION CHECKS PASSED")
