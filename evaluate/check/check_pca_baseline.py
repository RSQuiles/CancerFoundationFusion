"""Self-checks for the PCA baseline and per-modality scale handling in build_eval_adata.

Run directly; needs only numpy/scipy/pandas/anndata/scanpy (no cluster, no GPU, no
checkpoints, no training stack):

    python evaluate/check/check_pca_baseline.py

Exits non-zero on failure.

The substantive checks are the two failures that actually happened on the cluster:

  1. Memory. The baseline used to densify the whole matrix and hold ~3 copies of it
     before cutting to HVGs, which at 300k cells x 28.7k genes is ~100 GB and got
     SIGKILLed. The peak-allocation assertion below is the regression guard.
  2. Mixed scales. pseudobulk_paired_generation.py log1p's its SC rows uncondition-
     ally while honouring use_counts for PB and bulk, so a *_counts dataset holds
     log1p and raw-count rows in one matrix. A single global looks_like_counts()
     verdict is then wrong for some groups, and drove expm1 onto raw counts.
"""

from __future__ import annotations

import importlib.util
import sys
import tracemalloc
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate.utils import (  # noqa: E402
    MODALITY_COL,
    detect_scale_by_modality,
    generate_pseudobulk_adata,
    looks_like_counts,
)

FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if not cond else ""))
    if not cond:
        FAILED.append(label)


def _load_build_module():
    """Load build_eval_adata without its CancerFoundation import.

    The module needs the training stack only for embedding; the two functions under
    test here are pure numpy/scanpy. Stub the heavy deps so this runs anywhere.
    """
    for name in ("bionemo", "bionemo.scdl", "bionemo.scdl.io",
                 "bionemo.scdl.io.single_cell_memmap_dataset", "tokenizers",
                 "transformers", "safetensors", "pytorch_lightning",
                 "pytorch_lightning.utilities", "pytorch_lightning.utilities.types"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["bionemo.scdl.io.single_cell_memmap_dataset"].SingleCellMemMapDataset = object
    sys.modules["pytorch_lightning"].LightningModule = type("LM", (object,), {})
    sys.modules["pytorch_lightning.utilities.types"].OptimizerLRSchedulerConfig = dict
    sys.modules["transformers"].get_scheduler = lambda *a, **k: None
    sys.modules["safetensors"].safe_open = lambda *a, **k: None
    for attr in ("Tokenizer", "models", "pre_tokenizers", "trainers"):
        setattr(sys.modules["tokenizers"], attr, object)

    spec = importlib.util.spec_from_file_location(
        "_build_eval_adata", ROOT / "evaluate" / "check" / "build_eval_adata.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_build_eval_adata"] = mod
    spec.loader.exec_module(mod)
    return mod


N_GENES = 600
rng = np.random.default_rng(0)


def _counts(n: int, lam: float, density: float = 0.1) -> np.ndarray:
    """Sparse-ish count block. Real scRNA-seq is ~90% zeros, and that matters here:
    with a dense-ish matrix, CSR is *larger* than the dense form and the memory
    comparison below would be meaningless."""
    X = rng.poisson(lam, (n, N_GENES)).astype(np.float32)
    X[rng.random((n, N_GENES)) > density] = 0.0
    X[:, 0] = rng.integers(200, 900, n)          # ensure max > 20 (counts signature)
    return X


def make_mixed(n_per_group: int = 200) -> ad.AnnData:
    """sc + bulk in raw counts, paired_sc in log1p — the real-world mix."""
    counts_a = _counts(n_per_group, 4)
    counts_b = _counts(n_per_group, 9)
    log_c = np.log1p(_counts(n_per_group, 4))

    X = sp.csr_matrix(np.vstack([counts_a, counts_b, log_c]))
    obs = pd.DataFrame(
        {
            MODALITY_COL: (["sc"] * n_per_group + ["bulk"] * n_per_group
                           + ["paired_sc"] * n_per_group),
            "tissue_general": rng.choice(["lung", "colon"], 3 * n_per_group),
        },
        index=[f"c{i}" for i in range(3 * n_per_group)],
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]))


def main() -> int:
    build = _load_build_module()
    adata = make_mixed()

    print("=== per-modality scale detection ===")
    scale = detect_scale_by_modality(adata, MODALITY_COL)
    check("sc detected as counts", scale.get("sc") is True, str(scale))
    check("bulk detected as counts", scale.get("bulk") is True, str(scale))
    check("paired_sc detected as log1p", scale.get("paired_sc") is False, str(scale))
    check("one verdict per modality", set(scale) == {"sc", "bulk", "paired_sc"})
    # The whole point: one global call cannot describe this matrix.
    check("a single global verdict is not right for every group",
          len(set(scale.values())) > 1
          and any(looks_like_counts(adata.X) != v for v in scale.values()),
          f"global={looks_like_counts(adata.X)} per-group={scale}")
    check("no modality column -> empty dict",
          detect_scale_by_modality(ad.AnnData(X=adata.X.copy()), MODALITY_COL) == {})

    print("\n=== _to_log1p_scale ===")
    scaled = build._to_log1p_scale(adata.copy(), scale)
    after = detect_scale_by_modality(scaled, MODALITY_COL)
    check("every group is log1p afterwards", not any(after.values()), str(after))
    dense_after = scaled.X.toarray() if sp.issparse(scaled.X) else np.asarray(scaled.X)
    check("values stay finite", bool(np.isfinite(dense_after).all()))
    check("log1p rows are untouched",
          np.allclose(dense_after[400:], adata.X.toarray()[400:], atol=1e-5))

    print("\n=== aggregation no longer saturates ===")
    # expm1 on raw counts is what saturated float32; after _to_log1p_scale every row
    # is log1p, so expm1 is valid for all of them.
    pb = generate_pseudobulk_adata(
        scaled, group_column="tissue_general", n_sc_per_pb=10, n_pb=25,
        seed=0, is_log1p=True, normalize=True,
    )
    check("pseudobulks generated", pb is not None and pb.n_obs == 25)
    pb_dense = pb.X.toarray() if sp.issparse(pb.X) else np.asarray(pb.X)
    check("pseudobulk values finite", bool(np.isfinite(pb_dense).all()))
    check("nowhere near float32 max",
          float(pb_dense.max()) < 1e6, f"max={pb_dense.max():.3e}")
    check("pseudobulk X is sparse (keeps the concat sparse)", sp.issparse(pb.X))
    check("var is preserved", list(pb.var.index) == list(scaled.var.index))

    # Contrast: the pre-fix path, expm1 over the raw-count rows.
    pb_bad = generate_pseudobulk_adata(
        adata, group_column="tissue_general", n_sc_per_pb=10, n_pb=25,
        seed=0, is_log1p=True, normalize=True,
    )
    bad_dense = pb_bad.X.toarray() if sp.issparse(pb_bad.X) else np.asarray(pb_bad.X)
    check("the unscaled path really does blow up (fixture is meaningful)",
          (not np.isfinite(bad_dense).all()) or float(bad_dense.max()) > 1e6,
          f"max={bad_dense.max():.3e}")

    print("\n=== PCA baseline ===")
    coords, recipe = build._pca_baseline(adata, 20, 100, scale, 0)
    check("one row of coordinates per cell", coords.shape[0] == adata.n_obs)
    check("n_components honoured", coords.shape[1] == 20, str(coords.shape))
    check("coords are float32 and finite",
          coords.dtype == np.float32 and bool(np.isfinite(coords).all()))
    check("recipe records the HVG count", recipe["n_hvg"] == 100, str(recipe.get("n_hvg")))
    check("recipe records per-modality scale",
          recipe["scale_by_modality"] == {k: bool(v) for k, v in scale.items()})
    check("recipe names which groups were normalised",
          "sc" in recipe["normalisation"] and "bulk" in recipe["normalisation"],
          recipe["normalisation"])
    check("explained variance reported", 0.0 <= recipe["explained_variance_ratio_sum"] <= 1.0)
    check("input adata was not mutated",
          looks_like_counts(adata.X[np.where(
              adata.obs[MODALITY_COL].to_numpy() == "sc")[0]]) is True)

    print("\n=== memory: reduce before densifying ===")
    big = make_mixed(n_per_group=700)          # 2100 x 600
    n_hvg = 50
    tracemalloc.start()
    tracemalloc.reset_peak()
    build._pca_baseline(big, 10, n_hvg, detect_scale_by_modality(big, MODALITY_COL), 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full_dense = big.n_obs * big.n_vars * 4
    hvg_dense = big.n_obs * n_hvg * 4
    print(f"    peak {peak/1e6:.1f} MB | full dense {full_dense/1e6:.1f} MB "
          f"| HVG block {hvg_dense/1e6:.1f} MB")
    # The old code held ~3 full dense copies; the new one should not even hold one.
    check("peak stays below a single full dense copy", peak < full_dense,
          f"peak={peak/1e6:.1f}MB full={full_dense/1e6:.1f}MB")
    check("peak is within a small multiple of the HVG block",
          peak < max(8 * hvg_dense, 20e6),
          f"peak={peak/1e6:.1f}MB hvg={hvg_dense/1e6:.1f}MB")

    print("\n=== degenerate inputs ===")
    # Genuinely all-log1p (not just relabelled), so no normalisation is needed.
    n = 60
    all_log = ad.AnnData(
        X=sp.csr_matrix(np.log1p(_counts(n, 4))),
        obs=pd.DataFrame({MODALITY_COL: ["paired_sc"] * n,
                          "tissue_general": ["lung"] * n},
                         index=[f"l{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]),
    )
    s2 = detect_scale_by_modality(all_log, MODALITY_COL)
    check("all-log1p fixture detected as log1p", s2 == {"paired_sc": False}, str(s2))
    _, rec2 = build._pca_baseline(all_log, 5, 40, s2, 0)
    check("all-log1p input skips normalisation",
          rec2["normalisation"] == "none", rec2["normalisation"])

    # A detection false negative must degrade, not raise: scanpy's seurat HVG expm1's
    # its input, so leftover count-scale rows overflow to inf and pd.cut raises.
    mislabelled = ad.AnnData(
        X=sp.csr_matrix(_counts(n, 6)),
        obs=pd.DataFrame({MODALITY_COL: ["paired_sc"] * n,
                          "tissue_general": ["lung"] * n},
                         index=[f"m{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]),
    )
    try:
        coords_m, rec_m = build._pca_baseline(
            mislabelled, 5, 40, {"paired_sc": False}, 0   # deliberately wrong verdict
        )
        check("counts mislabelled as log1p degrade instead of raising",
              coords_m.shape == (n, 5) and bool(np.isfinite(coords_m).all()))
        check("the fallback is recorded in the recipe",
              "fallback" in rec_m["normalisation"], rec_m["normalisation"])
    except Exception as exc:
        check("counts mislabelled as log1p degrade instead of raising", False,
              f"{type(exc).__name__}: {exc}")

    coords3, rec3 = build._pca_baseline(adata, 999, 100, scale, 0)
    check("n_components clamped to the data",
          coords3.shape[1] == min(adata.n_obs - 1, 100), str(coords3.shape))
    check("clamped value recorded", rec3["n_components"] == coords3.shape[1])

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("ALL PCA-BASELINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
