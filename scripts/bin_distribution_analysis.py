# %% [markdown]
# # How quantile binning spreads single-cell vs bulk values over the bins
#
# Training (and the reconstruction targets) bin each row with
# `cancerfoundation.data.preprocess.binning`: zeros are bin 0, and the NON-ZERO values of
# the row are cut at `n_bins - 1` quantile edges (bins 1..n_bins-1). When many non-zero
# values are equal (single-cell counts of 1 and 2), many edges are equal, and
# `_digitize` gives such a value a **uniformly random bin** between the two
# `np.digitize` sides. So a count of 1 can be any of bins 1..k, redrawn every time the
# row is binned, and the model regresses on that noise.
#
# This notebook measures that, for the rows of the reconstruction plans:
# 1. raw non-zero value distribution per row (share of 1s, 2s, ...);
# 2. how many of the bins one raw value occupies (tie width), and how many distinct
#    bins a row really has;
# 3. the irreducible error random tie-breaking adds to the MSE / MAE targets;
# 4. bin -> raw-value composition (what "bin 10" means in sc vs bulk);
# 5. the model's predictions on masked entries, by raw value (from `entries.parquet`);
# 6. the same occupancy under a few alternative binning schemes.
#
# Run in the container (or any env with numpy, pandas, anndata, matplotlib, pyarrow),
# from the repository root (or set `REPO`).

# %% Settings
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:  # plain python: print the tables instead
    display = print

REPO = Path("/cluster/work/boeva/rquiles/CancerFoundationFusion")
ROOT = Path("/cluster/work/boeva/rquiles/unifiedFM_experiments")
sys.path.insert(0, str(REPO))

from experiments.layout import ExperimentLayout  # noqa: E402
from evaluate.reconstruction.mask_plan import load_plan  # noqa: E402

layout = ExperimentLayout(ROOT)

N_BINS = 51                      # model n_bins (bins 0..50); check the checkpoint's hparams
PLANS = {                        # label -> plan spec (MODALITY, SPLIT, PART, NAME)
    "sc": ("sc", "leiden_s0", "test", "r25"),
    "bulk": ("bulk", "leiden_s0", "test", "r25"),
}
MAX_ROWS = 1000                  # rows per plan analysed (the plan has 2000)
MODELS = ["bulk_then_sc", "bulk_then_sc75_bulk25", "bulk_pretrain"]   # for section 5
# Vocabulary the model bins over: the training store's vocab.json, else the package default.
VOCAB_PATH = layout.pipeline_ready("sc", "leiden_s0") / "vocab.json"
if not VOCAB_PATH.exists():
    VOCAB_PATH = REPO / "cancerfoundation" / "assets" / "vocab.json"
SEED = 0
rng = np.random.default_rng(SEED)

vocab = set(json.loads(VOCAB_PATH.read_text()))
print(f"vocab: {VOCAB_PATH} ({len(vocab)} tokens)")

# %% Binning, exactly as cancerfoundation.data.preprocess (re-implemented: no torch import)
def quantile_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    return np.quantile(values, np.linspace(0, 1, n_bins - 1))


def bin_row(row: np.ndarray, n_bins: int = N_BINS, rng=rng):
    """(bins, lo, hi) for one row: the random bin binning() draws, and the range it draws from."""
    out = np.zeros(row.shape, dtype=np.int64)
    lo, hi = out.copy(), out.copy()
    if row.size == 0 or row.max() == 0:
        return out, lo, hi
    ids = row.nonzero()[0] if row.min() <= 0 else np.arange(row.size)
    v = row[ids]
    edges = quantile_edges(v, n_bins)
    left = np.digitize(v, edges)
    right = np.digitize(v, edges, right=True)
    out[ids] = np.ceil(rng.random(len(v)) * (right - left) + left).astype(np.int64)
    lo[ids] = np.where(right < left, right + 1, left)
    hi[ids] = left
    return out, lo, hi


# %% Load the plan rows (vocabulary genes only, as the model bins them)
def load_rows(label: str):
    mod, split, part, name = PLANS[label]
    import anndata as ad

    plan = load_plan(layout.mask_plan(mod, split, part, name))
    names = plan.obs_names[:MAX_ROWS]
    backed = ad.read_h5ad(layout.raw_h5ad(mod), backed="r")
    try:
        cols = np.flatnonzero(np.isin(backed.var_names, list(vocab)))
        a = backed[backed.obs_names.isin(names)].to_memory()
    finally:
        backed.file.close()
    a = a[[n for n in names if n in a.obs_names]]
    X = a.X[:, cols]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    print(f"[{label}] {X.shape[0]} rows x {X.shape[1]} vocabulary genes (of {a.n_vars} in the file)")
    OBS[label] = a.obs.copy()
    return plan, a.obs_names.to_numpy(), a.var_names[cols].to_numpy(), X.astype(np.float64)


OBS: dict[str, pd.DataFrame] = {}
data = {label: load_rows(label) for label in PLANS}

# %% 0. Value scale per row: raw counts, log1p of counts, or something else
# Quantile binning ranks values within a row, so it does not care; BulkFormer, the PCA
# baseline's CPM and any scheme on absolute values do.
SOURCE_COLUMNS = ["dataset_id", "dataset", "assay", "tissue_general"]   # first ones present are shown


def classify_row(r: np.ndarray) -> str:
    nz = r[r > 0]
    if nz.size == 0:
        return "empty"
    if np.allclose(nz, np.round(nz)):
        return "counts"
    e = np.expm1(nz)
    if np.allclose(e, np.round(e), rtol=1e-4, atol=1e-3):
        return "log1p_counts"
    return "other"


def as_counts(r: np.ndarray) -> np.ndarray | None:
    """The row as raw counts when it is counts or log1p(counts), else None."""
    kind = classify_row(r)
    if kind == "counts":
        return r
    if kind == "log1p_counts":
        return np.round(np.expm1(r))
    return None


scale = {}
for label, (_, _, _, X) in data.items():
    kinds = pd.Series([classify_row(r) for r in X], index=OBS[label].index, name="scale")
    scale[label] = kinds
    print(f"\n[{label}] row scale: " + ", ".join(f"{k} {v:.1%}" for k, v in kinds.value_counts(normalize=True).items()))
    for col in [c for c in SOURCE_COLUMNS if c in OBS[label].columns][:2]:
        tab = pd.crosstab(OBS[label][col].astype(str), kinds)
        display(tab[tab.drop(columns="counts", errors="ignore").sum(axis=1) > 0].head(30))

# %% 1. Raw non-zero values per row
def row_stats(X: np.ndarray) -> pd.DataFrame:
    recs = []
    for r in X:
        nz = r[r > 0]
        if nz.size == 0:
            continue
        edges = quantile_edges(nz)
        integral = np.allclose(nz, np.round(nz))
        recs.append({
            "nnz": nz.size, "frac_nonzero": nz.size / r.size, "integral": integral,
            "share_eq1": np.mean(np.isclose(nz, 1)), "share_eq2": np.mean(np.isclose(nz, 2)),
            "share_le3": np.mean(nz <= 3), "share_le5": np.mean(nz <= 5),
            "max": nz.max(), "median_nz": np.median(nz),
            "n_distinct_values": np.unique(nz).size,
            "n_distinct_edges": np.unique(edges).size,   # real resolution of the 50 non-zero bins
        })
    return pd.DataFrame(recs)


stats = {label: row_stats(X) for label, (_, _, _, X) in data.items()}
summary = pd.concat({k: v.describe(percentiles=[0.1, 0.5, 0.9]).T for k, v in stats.items()}, axis=0)
display(summary[["mean", "10%", "50%", "90%"]].round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for label, s in stats.items():
    axes[0].hist(s["share_eq1"], bins=40, alpha=0.6, label=label)
    axes[1].hist(s["n_distinct_edges"], bins=np.arange(0, N_BINS + 1), alpha=0.6, label=label)
    axes[2].hist(np.log10(s["nnz"]), bins=40, alpha=0.6, label=label)
axes[0].set(title="share of non-zero values equal to 1", xlabel="share", ylabel="rows")
axes[1].set(title=f"distinct quantile edges per row (of {N_BINS - 1})", xlabel="distinct edges")
axes[2].set(title="non-zero genes per row", xlabel="log10(nnz)")
for ax in axes:
    ax.legend()
plt.tight_layout()

# %% 2. Tie width: how many bins one raw value can land in
def entry_table(X: np.ndarray, max_rows: int = 300) -> pd.DataFrame:
    """One row per non-zero entry of the first rows: raw value, drawn bin, [lo, hi]."""
    parts = []
    for i, r in enumerate(X[:max_rows]):
        b, lo, hi = bin_row(r)
        nz = r > 0
        parts.append(pd.DataFrame({"row": i, "value": r[nz], "bin": b[nz], "lo": lo[nz], "hi": hi[nz]}))
    t = pd.concat(parts, ignore_index=True)
    t["width"] = t["hi"] - t["lo"] + 1
    return t


entries = {label: entry_table(X) for label, (_, _, _, X) in data.items()}


def value_bucket(v: pd.Series) -> "pd.Series":
    """Raw values 1..10 as themselves, larger ones in coarse buckets (counts) or deciles."""
    if np.allclose(v, np.round(v)):
        return pd.cut(v, [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, np.inf],
                      labels=["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51-100", ">100"])
    return pd.qcut(v, 10, duplicates="drop").astype(str)


for label, t in entries.items():
    g = t.groupby(value_bucket(t["value"]), observed=True).agg(
        share=("value", lambda s: len(s) / len(t)), lo=("lo", "mean"), hi=("hi", "mean"),
        width=("width", "mean"), bins_drawn=("bin", lambda s: s.nunique()))
    print(f"\n[{label}] raw value -> bin range (mean over entries)")
    display(g.round(2))

fig, ax = plt.subplots(figsize=(7, 4))
for label, t in entries.items():
    ax.hist(t["width"], bins=np.arange(1, N_BINS + 1), alpha=0.6, label=label, density=True)
ax.set(title="tie width per non-zero entry (bins a value is randomly spread over)",
       xlabel="width (bins)", ylabel="density", yscale="log")
ax.legend()

# %% 3. Irreducible target noise from random tie-breaking
# A value spread uniformly over w bins: the best constant prediction (the midpoint) still
# has MSE (w^2 - 1) / 12 and MAE ~ w / 4 against the drawn bin.
for label, t in entries.items():
    w = t["width"].to_numpy(float)
    mse_floor = np.mean((w ** 2 - 1) / 12)
    mae_floor = np.mean([np.mean(np.abs(np.arange(k) - (k - 1) / 2)) for k in w.astype(int)])
    tied = np.mean(w > 1)
    print(f"[{label}] non-zero entries in a tied range: {tied:.1%}; "
          f"irreducible MSE {mse_floor:.2f}, MAE {mae_floor:.2f} (bins), mean width {w.mean():.1f}")

# %% 4. Bin -> raw value composition
fig, axes = plt.subplots(1, len(entries), figsize=(7 * len(entries), 4), sharey=False)
for ax, (label, t) in zip(np.atleast_1d(axes), entries.items()):
    comp = t.assign(bucket=value_bucket(t["value"])).pivot_table(
        index="bin", columns="bucket", values="value", aggfunc="size", fill_value=0, observed=False)
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp.plot.bar(stacked=True, ax=ax, width=1.0, colormap="viridis", legend=(label == list(entries)[-1]))
    ax.set(title=f"{label}: raw values inside each bin", xlabel="bin", ylabel="share")
    ax.set_xticks(range(0, len(comp), 5))
    ax.set_xticklabels(comp.index[::5])
if len(entries) > 1:
    np.atleast_1d(axes)[-1].legend(title="raw value", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()

# Where the bins of value 1 end, per row (bins 1..k all mean "count 1")
for label, t in entries.items():
    ones = t[np.isclose(t["value"], 1)]
    if len(ones):
        k = ones.groupby("row")["hi"].max()
        print(f"[{label}] count 1 occupies bins 1..k, k per row: median {k.median():.0f}, "
              f"10-90% {k.quantile(.1):.0f}-{k.quantile(.9):.0f}")

# %% 5. Model predictions on masked entries, by raw value (entries.parquet)
def entries_with_raw(model: str, label: str) -> pd.DataFrame | None:
    plan, obs_names, genes, X = data[label]
    slug = plan.plan_id.replace(":", "_").replace("/", "_")
    path = layout.task_dir("reconstruction", model) / slug / "entries.parquet"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    e = pd.read_parquet(path)
    row_of = {n: i for i, n in enumerate(obs_names)}
    col_of = {g: j for j, g in enumerate(genes)}
    keep = e["sample"].isin(row_of) & e["gene"].isin(col_of)
    e = e[keep].copy()
    e["raw"] = X[e["sample"].map(row_of).to_numpy(), e["gene"].map(col_of).to_numpy()]
    return e


for label in PLANS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    band = None   # the tied range is the same for every model (same targets)
    for model in MODELS:
        e = entries_with_raw(model, label)
        if e is None or e.empty:
            continue
        nz = e[e["raw"] > 0]
        lo_col = "target_lo" if "target_lo" in nz else "target"   # runs before tie ranges lack them
        hi_col = "target_hi" if "target_hi" in nz else "target"
        g = nz.groupby(value_bucket(nz["raw"]), observed=True).agg(
            pred=("pred", "mean"), lo=(lo_col, "mean"), hi=(hi_col, "mean"))
        axes[0].plot(g.index.astype(str), g["pred"], marker="o", label=f"{model} pred")
        band = g if band is None else band
        ones = nz[np.isclose(nz["raw"], 1)]
        if len(ones):
            axes[1].hist(ones["pred"], bins=np.arange(0, N_BINS + 1), alpha=0.5, label=model, density=True)
    if band is not None:
        axes[0].fill_between(band.index.astype(str), band["lo"], band["hi"], color="grey", alpha=0.2,
                             label="mean tied range [lo, hi]")
    axes[0].set(title=f"{label}: mean prediction by raw value (masked non-zero entries)",
                xlabel="raw value", ylabel="bin")
    axes[1].set(title=f"{label}: predictions for raw count 1", xlabel="predicted bin", ylabel="density")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()

# %% 6. Alternative binnings: occupancy and ties (ideas, not what training does)
def scheme_bins(row: np.ndarray, scheme: str, global_edges=None) -> np.ndarray:
    nz = row > 0
    out = np.zeros(row.shape, dtype=np.int64)
    if not nz.any():
        return out
    v = row[nz]
    if scheme == "quantile (current)":
        out[:] = bin_row(row)[0]
        return out
    if scheme == "quantile over distinct values":
        # each distinct value gets one quantile position, so ties no longer eat bins
        u = np.unique(v)
        edges = np.quantile(u, np.linspace(0, 1, N_BINS - 1))
        out[nz] = np.clip(np.searchsorted(np.unique(edges), v, side="right"), 1, N_BINS - 1)
        return out
    if scheme == "log1p equal-width per row":
        lv = np.log1p(v)
        edges = np.linspace(lv.min(), lv.max() + 1e-9, N_BINS)
        out[nz] = np.clip(np.digitize(lv, edges), 1, N_BINS - 1)
        return out
    if scheme == "log1p(CP10K) global edges":
        lv = np.log1p(v / row.sum() * 1e4)
        out[nz] = np.clip(np.digitize(lv, global_edges), 1, N_BINS - 1)
        return out
    raise ValueError(scheme)


SCHEMES = ["quantile (current)", "quantile over distinct values",
           "log1p equal-width per row", "log1p(CP10K) global edges"]
for label, (_, _, _, X) in data.items():
    rows = X[:300]
    pooled = np.concatenate([np.log1p(r[r > 0] / r.sum() * 1e4) for r in rows if r.sum() > 0])
    global_edges = np.linspace(pooled.min(), pooled.max() + 1e-9, N_BINS)
    fig, axes = plt.subplots(1, len(SCHEMES), figsize=(5 * len(SCHEMES), 3.5), sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        bins = np.concatenate([scheme_bins(r, scheme, global_edges)[r > 0] for r in rows])
        vals = np.concatenate([r[r > 0] for r in rows])
        occ = np.bincount(bins, minlength=N_BINS)[1:]
        ax.bar(np.arange(1, N_BINS), occ / occ.sum(), width=1.0)
        ones = bins[np.isclose(vals, 1)]
        spread = f"count 1 -> {np.unique(ones).size} bins" if ones.size else ""
        ax.set(title=f"{label}: {scheme}\n{spread}", xlabel="bin")
    axes[0].set_ylabel("share of non-zero entries")
    plt.tight_layout()

# %% 7. Proposed schemes: deterministic mid-rank, and a hybrid with exact low counts
def midrank_quantile(v: np.ndarray) -> np.ndarray:
    """Within-row percentile in (0, 1) of each value; tied values share their mean rank."""
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts                 # ranks before each distinct value
    mean_rank = start + (counts + 1) / 2.0             # 1-based average rank of the tie group
    return (mean_rank[inverse] - 0.5) / v.size


def midrank_bins(row: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """PROPOSAL. Zeros -> 0; non-zeros -> 1 + floor(mid-rank percentile * (n_bins - 1)).

    Deterministic (no random tie-breaking), monotone, and scale-free like the current
    quantile binning; on rows without ties it gives the same equal-occupancy bins.
    """
    out = np.zeros(row.shape, dtype=np.int64)
    nz = row > 0
    if nz.any():
        q = midrank_quantile(row[nz])
        out[nz] = np.clip(1 + np.floor(q * (n_bins - 1)).astype(np.int64), 1, n_bins - 1)
    return out


def hybrid_bins(row: np.ndarray, k: int = 5, n_bins: int = N_BINS) -> np.ndarray:
    """COMPARISON. Counts 1..k -> bins 1..k exactly; mid-rank over counts > k -> bins k+1..n_bins-1.

    Needs absolute values: log1p(counts) rows are converted back, other rows fall back to
    mid-rank, which is exactly the weakness of absolute schemes on mixed-scale data.
    """
    c = as_counts(row)
    if c is None:
        return midrank_bins(row, n_bins)
    out = np.zeros(row.shape, dtype=np.int64)
    low = (c >= 1) & (c <= k)
    out[low] = c[low].astype(np.int64)
    high = c > k
    if high.any():
        q = midrank_quantile(c[high])
        out[high] = np.clip(k + 1 + np.floor(q * (n_bins - 1 - k)).astype(np.int64), k + 1, n_bins - 1)
    return out


def current_bins(row: np.ndarray, seed: int) -> np.ndarray:
    return bin_row(row, rng=np.random.default_rng(seed))[0]


PROPOSED = {
    "quantile (current)": lambda r, s=0: current_bins(r, s),
    "mid-rank (proposal)": lambda r, s=0: midrank_bins(r),
    "hybrid k=5 (comparison)": lambda r, s=0: hybrid_bins(r),
}
N_ROWS, REDRAWS = 300, 5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman")) if x.size > 2 else np.nan


rows_summary, meaning = [], {}
for label, (_, _, _, X) in data.items():
    rows = [r for r in X[:N_ROWS] if (r > 0).any()]
    for scheme, fn in PROPOSED.items():
        used, rho, noise_sd, noise_mse, one_bins, pct_by_bin = [], [], [], [], [], []
        for i, r in enumerate(rows):
            nz = r > 0
            draws = np.stack([fn(r, SEED + 1000 * d + i)[nz] for d in range(REDRAWS)]).astype(float)
            b = draws[0]
            used.append(np.unique(b).size)
            rho.append(spearman(r[nz], b))
            noise_sd.append(draws.std(axis=0).mean())
            noise_mse.append(np.mean((draws[0] - draws[1]) ** 2) / 2)   # per-draw variance estimate
            c = as_counts(r)
            if c is not None:
                one_bins.append(np.unique(draws[:, np.isclose(c[nz], 1)]))
            pct_by_bin.append(pd.DataFrame({"bin": b, "pct": midrank_quantile(r[nz])}))
        ones = np.unique(np.concatenate(one_bins)) if one_bins else np.array([])
        rows_summary.append({
            "modality": label, "scheme": scheme,
            "bins used per row (median)": np.median(used),
            "count 1 lands in # bins": ones.size,
            "count 1 bins": f"{ones.min():.0f}-{ones.max():.0f}" if ones.size else "-",
            "redraw noise sd (bins)": np.mean(noise_sd),
            "redraw noise MSE (bins^2)": np.mean(noise_mse),
            "Spearman raw vs bin": np.nanmean(rho),
        })
        meaning[(label, scheme)] = pd.concat(pct_by_bin).groupby("bin")["pct"].mean()

summary7 = pd.DataFrame(rows_summary).set_index(["scheme", "modality"]).sort_index()
display(summary7.round(3))

# Bin occupancy per scheme (share of non-zero entries in each bin)
fig, axes = plt.subplots(len(data), len(PROPOSED), figsize=(5 * len(PROPOSED), 3.2 * len(data)),
                         sharey="row", squeeze=False)
for i, (label, (_, _, _, X)) in enumerate(data.items()):
    rows = [r for r in X[:N_ROWS] if (r > 0).any()]
    for j, (scheme, fn) in enumerate(PROPOSED.items()):
        b = np.concatenate([fn(r)[r > 0] for r in rows])
        occ = np.bincount(b, minlength=N_BINS)[1:]
        axes[i, j].bar(np.arange(1, N_BINS), occ / occ.sum(), width=1.0)
        axes[i, j].set(title=f"{label}: {scheme}", xlabel="bin")
    axes[i, 0].set_ylabel("share of non-zero entries")
plt.tight_layout()

# What a bin means: mean within-row percentile of the values in it, sc vs bulk.
# Overlapping sc and bulk lines = the bin index means the same thing in both modalities.
fig, axes = plt.subplots(1, len(PROPOSED), figsize=(5 * len(PROPOSED), 3.5), sharey=True)
for ax, scheme in zip(axes, PROPOSED):
    for label, style in zip(data, ["-", "--", ":"]):
        m = meaning[(label, scheme)]
        ax.plot(m.index, m.values, style, label=label)
    ax.plot([1, N_BINS - 1], [0, 1], color="grey", lw=0.8, alpha=0.6, label="ideal")
    ax.set(title=scheme, xlabel="bin", ylabel="mean within-row percentile")
    ax.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 8. Analysis and proposal
#
# **What the data shows (leiden_s0 r25 plan rows, first run of this notebook)**
#
# | | single-cell | bulk |
# |---|---|---|
# | non-zero values equal to 1 (mean / median row) | 51% / 63% | 5% / 2% |
# | non-zero values <= 3 | 82% | 10% |
# | distinct quantile edges per row (of 50) | ~9 | ~46 |
# | bins a count of 1 (or log1p(1)) is spread over | ~29-35 | ~2 (lowest decile only) |
#
# 1. **Most single-cell bins are random relabelings of the same value.** With half the
#    non-zeros equal to 1, about 25-35 of the 50 quantile edges equal 1, and `_digitize`
#    draws a count of 1 uniformly from that whole range, anew every time the row is
#    binned. The row really has ~9 distinguishable levels; the other ~40 bin indices
#    carry no information about the value.
# 2. **This hurts training twice.** The *targets* are noisy: a value spread over w bins
#    has an irreducible MSE of (w^2 - 1)/12, about 75 for w = 30, so the MSE loss on sc
#    rows is dominated by noise the model cannot fit, and the best it can do for low
#    counts is predict the middle of the range (section 5). The *inputs* are noisy too:
#    the context values the model conditions on change between epochs for the same cell.
# 3. **It explains the mae_by_bin shape.** Bins 1-25 look good because the tie-aware MAE
#    accepts any bin of the range, and the model predicts its middle. The error then rises
#    towards bins 35-45: those few bins hold every count from ~5 up to hundreds
#    (section 4), so real expression differences are compressed into a handful of bins
#    and the regression pulls them towards the middle. bulk_pretrain, which never saw
#    this distribution, is the worst there.
# 4. **The same bin index means different things in sc and bulk.** Bin 25 is "a count of
#    1" in a typical sc row and "~400 counts, the 50th percentile" in bulk. A model
#    warm-started on bulk and continued on sc has to relearn what the indices mean.
# 5. **Separate data issue: mixed scales in the sc h5ad.** About 17% of sc non-zero
#    entries are 0.693 = log1p(1) (and 1.609, 2.303 = log1p(4), log1p(9)): some sources
#    are stored as log1p, not counts (section 0 names them). Quantile and mid-rank binning
#    rank values within a row, so CancerFoundation is unaffected, but BulkFormer (it
#    refused the file), the PCA baseline's CPM and any absolute-value scheme are not.
#
# **Proposal: deterministic mid-rank binning.** Keep the per-row, scale-free quantile
# idea, but give tied values one bin, the bin of their *average* rank:
# `bin = 1 + floor(((mean_rank - 0.5) / n_nonzero) * (n_bins - 1))`; zeros stay 0.
# - No random tie-breaking: zero target noise and stable inputs across epochs
#   (section 7: redraw noise 0, a count of 1 lands in one bin per row).
# - The bin keeps one meaning, "percentile of the value among the row's non-zeros", in
#   both modalities and on log1p or count rows alike (section 7, meaning plot).
# - Bulk is essentially unchanged (few ties), so bulk-pretrained weights stay meaningful.
# - Drop-in: same `n_bins`, same value encoder and losses; only `binning` changes.
# - Cost: sc rows still use only as many levels as they have distinct values (a count of
#   1 sits at one middle bin), but the unused indices are now simply unused, not noise.
#
# The hybrid (counts 1..5 in bins 1..5 exactly) gives low counts an absolute meaning, but
# it needs real counts, which part of the sc data is not, and it gives sc and bulk
# different bin meanings again; keep it as a comparison, not a recommendation.
#
# **To adopt it** (not done here): a `binning_method` option (`quantile` by default, so
# old checkpoints are unchanged; `midrank`) in `cancerfoundation.data.preprocess.binning`,
# stored as a model hparam and used by the collator, `CancerFoundation.embed` and the
# reconstruction targets; then continue or retrain the sc models. Reconstruction targets
# become deterministic, so the tie-aware metrics coincide with the plain ones.
# Separately, convert the log1p sources of the sc file back to counts before BulkFormer
# and PCA.

# %% [markdown]
# ## 9. How many bins does the data support?
#
# A bin count is **enough** when binning throws away no information the data actually
# carries, and **too many** when the extra bins can only be filled with noise or stay
# empty. Three criteria, measured per modality with mid-rank binning (deterministic, so
# every number below is a property of the data and not of a random draw):
#
# 1. **Resolution ceiling (distinct values per row).** Mid-rank gives each distinct
#    non-zero value at most one bin. A row with d distinct values can fill at most d bins;
#    `n_bins - 1 > d` leaves bins empty in that row.
# 2. **Information retained (entropy).** For deterministic binning, `H(bin) <= H(value)`,
#    and `H(bin) / H(value)` is the share of the row's value (rank) information the bins
#    keep. It rises with n_bins and saturates at 1; the knee, where it reaches ~95% for
#    most rows, is the smallest n_bins that loses almost nothing.
# 3. **Quantize below the noise floor.** Counts are Poisson-like. After the Anscombe
#    transform `A(x) = 2 sqrt(x + 3/8)` their sampling noise has variance ~1 at every
#    level. If every value in a bin is replaced by the bin's mean `A`, the quantization
#    MSE should be well below that noise (here: <= 0.1). Finer bins then only resolve
#    sampling noise, which the model cannot learn from. Relatedly,
#    `A(max) - A(1) + 1` is the number of levels one noise SD apart that a row can
#    really distinguish. (Count and log1p(count) rows only; other rows are skipped.)
#
# sc and bulk share one `n_bins`. With mid-rank, bins a sparse sc row cannot fill are
# simply unused (with random tie-breaking they were filled with noise), so the shared
# value should come from the modality that needs the most resolution, as long as the
# vocabulary of bins stays small enough for the value encoder.

# %% 9. Data-driven n_bins: compute the three criteria over a grid
N_BINS_GRID = [5, 9, 11, 17, 21, 33, 51, 65, 101, 129]
ENTROPY_TARGET, QUANT_TARGET, ROW_SHARE = 0.95, 0.10, 0.90


def entropy_bits(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def anscombe(c: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(c + 3.0 / 8.0)


def quantization_ratio(a: np.ndarray, bins: np.ndarray) -> float:
    """MSE of replacing each value by its bin's mean, in Anscombe units (noise var ~ 1)."""
    sums = np.bincount(bins, weights=a)
    sizes = np.bincount(bins)
    means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
    return float(np.mean((a - means[bins]) ** 2))


grid_rows, per_row = [], []
for label, (_, _, _, X) in data.items():
    for i, r in enumerate(X[:N_ROWS]):
        nz = r > 0
        if not nz.any():
            continue
        v = r[nz]
        h_value = entropy_bits(v)
        c = as_counts(r)
        a = anscombe(c[nz]) if c is not None else None
        per_row.append({
            "modality": label, "row": i, "distinct_values": np.unique(v).size,
            "poisson_levels": (anscombe(c[nz].max()) - anscombe(1.0) + 1) if c is not None else np.nan,
        })
        for n in N_BINS_GRID:
            b = midrank_bins(r, n)[nz]
            grid_rows.append({
                "modality": label, "row": i, "n_bins": n,
                "bins_used": np.unique(b).size,
                "entropy_ratio": entropy_bits(b) / h_value if h_value > 0 else 1.0,
                "quant_ratio": quantization_ratio(a, b) if a is not None else np.nan,
            })

grid = pd.DataFrame(grid_rows)
rows9 = pd.DataFrame(per_row)

table9 = grid.groupby(["modality", "n_bins"]).agg(
    bins_used_median=("bins_used", "median"),
    entropy_ratio_mean=("entropy_ratio", "mean"),
    share_entropy_ok=("entropy_ratio", lambda s: np.mean(s >= ENTROPY_TARGET)),
    quant_ratio_median=("quant_ratio", "median"),
    share_quant_ok=("quant_ratio", lambda s: np.mean(s.dropna() <= QUANT_TARGET) if s.notna().any() else np.nan),
)
table9["utilization"] = table9["bins_used_median"] / (table9.index.get_level_values("n_bins") - 1)
display(table9.round(3))
display(rows9.groupby("modality")[["distinct_values", "poisson_levels"]]
        .describe(percentiles=[0.1, 0.5, 0.9]).round(1))

# %% 9. Plots: the knees, and the per-row resolution against the current n_bins
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
for label in data:
    t = table9.loc[label]
    axes[0].plot(t.index, t["entropy_ratio_mean"], marker="o", label=label)
    axes[1].plot(t.index, t["share_entropy_ok"], marker="o", label=label)
    if t["quant_ratio_median"].notna().any():
        axes[2].plot(t.index, t["quant_ratio_median"], marker="o", label=label)
axes[0].axhline(ENTROPY_TARGET, color="grey", ls="--", lw=0.8)
axes[1].axhline(ROW_SHARE, color="grey", ls="--", lw=0.8)
axes[2].axhline(QUANT_TARGET, color="grey", ls="--", lw=0.8)
axes[0].set(title="information kept: H(bin) / H(value)", xlabel="n_bins", xscale="log")
axes[1].set(title=f"rows keeping >= {ENTROPY_TARGET:.0%} of it", xlabel="n_bins", xscale="log")
axes[2].set(title="quantization MSE / Poisson noise (Anscombe)", xlabel="n_bins", xscale="log", yscale="log")
for label in data:
    s = rows9.loc[rows9["modality"] == label, "distinct_values"]
    axes[3].hist(np.log10(s), bins=40, alpha=0.6, label=f"{label}: distinct values")
    p = rows9.loc[rows9["modality"] == label, "poisson_levels"].dropna()
    if len(p):
        axes[3].hist(np.log10(p), bins=40, alpha=0.4, histtype="step", lw=2, label=f"{label}: Poisson levels")
axes[3].axvline(np.log10(N_BINS - 1), color="black", ls=":", label=f"current: {N_BINS - 1} non-zero bins")
axes[3].set(title="per-row resolution", xlabel="log10(levels)", ylabel="rows")
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()

# %% 9. Recommendation: smallest grid n_bins meeting each criterion for ROW_SHARE of rows
def smallest(t: pd.DataFrame, col: str) -> int | None:
    ok = t.index[t[col] >= ROW_SHARE]
    return int(ok.min()) if len(ok) else None


needs = {}
for label in data:
    t = table9.loc[label]
    d90 = rows9.loc[rows9["modality"] == label, "distinct_values"].quantile(ROW_SHARE)
    p90 = rows9.loc[rows9["modality"] == label, "poisson_levels"].quantile(ROW_SHARE)
    n_ent, n_quant = smallest(t, "share_entropy_ok"), smallest(t, "share_quant_ok")
    needs[label] = [n for n in (n_ent, n_quant) if n is not None]
    print(f"[{label}] {ROW_SHARE:.0%} of rows have <= {d90:.0f} distinct values "
          f"and <= {p90:.0f} Poisson-distinguishable levels")
    print(f"    entropy >= {ENTROPY_TARGET:.0%}: n_bins >= {n_ent or f'> {N_BINS_GRID[-1]}'}; "
          f"quantization <= {QUANT_TARGET}: n_bins >= {n_quant or f'> {N_BINS_GRID[-1]} (or no count rows)'}")

flat = [n for ns in needs.values() for n in ns]
if flat:
    print(f"\nShared n_bins meeting every criterion found on this grid: {max(flat)} "
          f"(current: {N_BINS}). Per modality: "
          + ", ".join(f"{k} {max(v) if v else '-'}" for k, v in needs.items()))
print("A criterion not met within the grid means that modality could use more bins than "
      f"{N_BINS_GRID[-1]}; weigh that against the value encoder's capacity.")

# %% [markdown]
# **How to read section 9**
# - `bins_used_median` / `utilization`: how many bins a typical row fills. Low utilization
#   in sc is expected with mid-rank (few distinct values) and costs nothing; it is a
#   problem only with random tie-breaking, which fills the empty bins with noise.
# - The entropy knee is where more bins stop preserving more of the rows' information.
#   sc should saturate early (a row has ~9-60 distinct values); bulk late (thousands).
# - The quantization criterion is the noise-aware one: below it, two values sharing a
#   bin differ by less than their own Poisson noise, so splitting them further only
#   teaches the model sampling noise. It needs count (or log1p count) rows. Poisson is
#   the smallest possible noise: real counts are overdispersed (negative binomial), most
#   of all in deep bulk libraries, so for bulk this criterion is very strict (it can ask
#   for hundreds of bins). Read it as an upper bound there and lean on the entropy knee.
# - The suggested shared n_bins is the largest of the per-modality requirements. If bulk
#   needs more than sc, the shared value comes from bulk, and sc rows simply leave the
#   extra bins empty. If the value is close to the current 51, keep 51: changing n_bins
#   changes the decoder/encoder output range and breaks warm starts from existing
#   checkpoints, which is a cost the criteria do not count.
