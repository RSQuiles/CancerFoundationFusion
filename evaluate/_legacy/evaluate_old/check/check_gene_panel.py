"""Self-contained checks for select_shared_panel + looks_like_counts.

Run directly; needs only numpy/pandas/scipy/anndata/scanpy -- no checkpoint, no
torch-lightning, no bionemo:

    python evaluate/check/check_gene_panel.py

Exits non-zero on failure. The substantive check is the consensus-vs-pooled
contrast: on a fixture where 50 genes are constant *within* each modality but
offset hugely *between* them, a pooled fit selects all 50 (half its panel), while
consensus selects ~0. That is why consensus is the default -- a pooled panel is
built out of exactly the genes that separate the modalities being compared.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd


def _load(mod_name: str, rel_path: str):
    """Load a module by file path, bypassing cancerfoundation/__init__.py.

    The package __init__ pulls in bionemo/pytorch_lightning/transformers, none of
    which this check needs — that gene_panel.py can be loaded without them is
    part of what is being verified.
    """
    for pkg in ("cancerfoundation", "cancerfoundation.data"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(ROOT / pkg.replace(".", "/"))]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pre = _load("cancerfoundation.data.preprocess", "cancerfoundation/data/preprocess.py")
_gp = _load("cancerfoundation.data.gene_panel", "cancerfoundation/data/gene_panel.py")
looks_like_counts = _pre.looks_like_counts
select_shared_panel = _gp.select_shared_panel
_rank_desc = _gp._rank_desc

rng = np.random.default_rng(0)
N_GENES = 400
N_TOP = 100

# ---- looks_like_counts ---------------------------------------------------
counts = rng.poisson(5, size=(200, N_GENES)).astype(np.float32)
counts[0, 0] = 500  # ensure max > 20
log1p = np.log1p(counts)
cp10k_log1p = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)

assert looks_like_counts(counts), "raw counts not detected"
assert not looks_like_counts(log1p), "log1p misdetected as counts"
assert not looks_like_counts(cp10k_log1p), "CP10K+log1p misdetected as counts"
assert not looks_like_counts(np.zeros((10, 10))), "all-zero should not be counts"
import scipy.sparse as sp
assert looks_like_counts(sp.csr_matrix(counts)), "sparse counts not detected"
print("looks_like_counts: OK")

# ---- _rank_desc ---------------------------------------------------------
r = _rank_desc(np.array([10.0, 30.0, 20.0]))
assert list(r) == [2.0, 0.0, 1.0], r
print("_rank_desc: OK")

# ---- build a two-group AnnData -----------------------------------------
# Group A ("bulk") is variable in genes 0-49; group B ("pseudobulk") in genes 50-99.
# Genes 100-149 are variable in BOTH. Genes 300+ are near-constant.
# A pooled fit is additionally drawn to genes 350-399, which are constant *within*
# each group but differ hugely *between* them -- exactly the modality-discriminating
# genes a shared panel must avoid.
def make_group(n, hot, offset_block=None, offset=0.0):
    X = rng.normal(2.0, 0.05, size=(n, N_GENES))
    X[:, hot] += rng.normal(0, 3.0, size=(n, len(hot)))
    X[:, 100:150] += rng.normal(0, 3.0, size=(n, 50))
    if offset_block is not None:
        X[:, offset_block] += offset
    return np.abs(X).astype(np.float32)

A = make_group(150, np.arange(0, 50), offset_block=np.arange(350, 400), offset=0.0)
B = make_group(150, np.arange(50, 100), offset_block=np.arange(350, 400), offset=40.0)

genes = [f"g{i}" for i in range(N_GENES)]
adata = ad.AnnData(
    X=np.vstack([A, B]),
    obs=pd.DataFrame(
        {"mod": ["bulk"] * len(A) + ["pseudobulk"] * len(B)},
        index=[f"c{i}" for i in range(len(A) + len(B))],
    ),
    var=pd.DataFrame(index=genes),
)
groups = adata.obs["mod"].to_numpy()

# ---- consensus ---------------------------------------------------------
print("\n--- consensus ---")
panel = select_shared_panel(adata, groups, N_TOP, strategy="consensus")
assert len(panel) == N_TOP, len(panel)
assert panel == sorted(panel), "panel must be sorted"
assert len(set(panel)) == len(panel), "panel has duplicates"

idx = np.array([int(g[1:]) for g in panel])
both = ((idx >= 100) & (idx < 150)).sum()
a_only = (idx < 50).sum()
b_only = ((idx >= 50) & (idx < 100)).sum()
discriminating = (idx >= 350).sum()
print(f"both-variable: {both}/50, A-only: {a_only}, B-only: {b_only}, "
      f"modality-discriminating: {discriminating}")
assert both == 50, f"consensus should keep all commonly-variable genes, got {both}"
# The fixture only has 150 genuinely variable genes for a 100-gene panel, so the
# tail is filled at random from the ~250 near-constant genes; a couple of the
# discriminating block landing there is noise, not selection pressure. What must
# hold is that they are not *sought out* (contrast with pooled, below).
assert discriminating <= 0.05 * N_TOP, (
    f"consensus should not be drawn to modality-discriminating genes, got {discriminating}"
)
assert abs(a_only - b_only) <= 6, f"consensus should be balanced: {a_only} vs {b_only}"

# symmetry: swapping the group labels must not change the panel
swapped = np.where(groups == "bulk", "pseudobulk", "bulk")
panel_swapped = select_shared_panel(adata, swapped, N_TOP, strategy="consensus",
                                   verbose=False)
assert panel == panel_swapped, "consensus must be invariant to group relabelling"
print("consensus: OK (symmetric, avoids discriminating genes)")

# ---- reference ---------------------------------------------------------
print("\n--- reference ---")
p_a = select_shared_panel(adata, groups, N_TOP, strategy="reference",
                          reference="bulk")
p_b = select_shared_panel(adata, groups, N_TOP, strategy="reference",
                          reference="pseudobulk")
assert p_a != p_b, "reference strategy should be asymmetric"
ia = np.array([int(g[1:]) for g in p_a])
print(f"reference=bulk -> A-only {(ia < 50).sum()}, B-only "
      f"{((ia >= 50) & (ia < 100)).sum()}, discriminating {(ia >= 350).sum()}")
assert (ia < 50).sum() > ((ia >= 50) & (ia < 100)).sum(), "should favour its reference"
try:
    select_shared_panel(adata, groups, N_TOP, strategy="reference", reference="nope")
    raise AssertionError("expected ValueError for unknown reference")
except ValueError:
    pass
print("reference: OK")

# ---- pooled (documented as biased) -------------------------------------
print("\n--- pooled ---")
p_pool = select_shared_panel(adata, groups, N_TOP, strategy="pooled")
ip = np.array([int(g[1:]) for g in p_pool])
disc_pool = (ip >= 350).sum()
print(f"pooled -> modality-discriminating: {disc_pool}/50")
assert disc_pool > discriminating, (
    f"pooled ({disc_pool}) should pick MORE modality-discriminating genes than "
    f"consensus ({discriminating}); if not, the fixture no longer demonstrates the bias"
)
print(f"pooled: OK (picks {disc_pool} discriminating genes vs consensus' "
      f"{discriminating} -- this is why consensus is the default)")

# ---- single group is a no-op-ish path ----------------------------------
print("\n--- single group ---")
one = select_shared_panel(adata, np.array(["bulk"] * adata.n_obs), N_TOP,
                          strategy="consensus", verbose=False)
assert len(one) == N_TOP
print("single group: OK")

# ---- tiny groups are skipped -------------------------------------------
print("\n--- min_cells skip ---")
g_tiny = groups.copy()
g_tiny[:5] = "runt"
p_tiny = select_shared_panel(adata, g_tiny, N_TOP, strategy="consensus")
assert len(p_tiny) == N_TOP
print("min_cells skip: OK")

# ---- unknown strategy --------------------------------------------------
try:
    select_shared_panel(adata, groups, N_TOP, strategy="bogus")
    raise AssertionError("expected ValueError")
except ValueError:
    print("unknown strategy rejected: OK")

print("\nALL PANEL CHECKS PASSED")
