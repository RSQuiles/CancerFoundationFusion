"""Self-check for the ``binning_method`` option (quantile | midrank).

    python scripts/check_binning_method.py

Part 1 needs only numpy: ``preprocess.binning`` / ``binning_with_edges`` with both
methods, and that ``quantile`` still draws exactly what the previous implementation drew
(compared with ``git show HEAD:cancerfoundation/data/preprocess.py`` when git has it).
Part 2 needs torch: ``AnnDataCollator`` with both methods on the same batch and seeds.
CLS, mask and pad values, gene order and masked positions must be identical; only the
binned values may differ, and mid-rank values must not change between calls.

The package ``__init__`` imports scanpy and the model, so the modules are loaded as
plain files under stub packages; nothing here needs a GPU, a checkpoint or data.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stub_packages() -> None:
    """``cancerfoundation`` and ``cancerfoundation.data`` without running their __init__."""
    for name, path in (("cancerfoundation", ROOT / "cancerfoundation"),
                       ("cancerfoundation.data", ROOT / "cancerfoundation" / "data")):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg


def rows(rng):
    sc = np.where(rng.random(2000) < 0.15, rng.negative_binomial(1, 0.5, 2000) + 1, 0).astype(np.float32)
    cont = rng.gamma(2.0, 1.0, size=500).astype(np.float32)               # no ties, no zeros
    return sc, cont


def check_numpy() -> None:
    print("\n-- preprocess.binning --")
    stub_packages()
    pp = load_file("cancerfoundation.data.preprocess", ROOT / "cancerfoundation" / "data" / "preprocess.py")
    rng = np.random.default_rng(0)
    sc, cont = rows(rng)
    n_bins = 51

    a = pp.binning(sc.copy(), n_bins, method="midrank")
    np.random.seed(123)
    b = pp.binning(sc.copy(), n_bins, method="midrank")
    nz = sc > 0
    check("midrank is deterministic", np.array_equal(a, b))
    check("zeros stay bin 0", (a[~nz] == 0).all())
    check("non-zeros in 1..n_bins-1", a[nz].min() >= 1 and a.max() <= n_bins - 1)
    check("tied values share one bin", all(np.unique(a[sc == v]).size == 1 for v in np.unique(sc[nz])))
    order = np.argsort(sc[nz], kind="stable")
    check("monotone in the value", (np.diff(a[nz][order]) >= 0).all())
    check("input dtype kept (like quantile)", a.dtype == sc.dtype)

    np.random.seed(0)
    q = pp.binning(cont.copy(), n_bins)
    m = pp.binning(cont.copy(), n_bins, method="midrank")
    check("no ties: midrank ~ quantile (+-1 bin)", np.abs(q - m).max() <= 1, f"max diff {np.abs(q - m).max()}")
    check("all-positive row is binned too", m.min() >= 1)

    bw, edges = pp.binning_with_edges(sc.copy(), n_bins, method="midrank")
    _, edges_q = pp.binning_with_edges(sc.copy(), n_bins)
    check("binning_with_edges(midrank) == binning(midrank)", np.array_equal(bw, a.astype(np.int64)))
    check("edges are the quantile edges either way", np.array_equal(edges, edges_q))
    check("all-zero row", (pp.binning(np.zeros(5, np.float32), n_bins, method="midrank") == 0).all())
    try:
        pp.binning(sc.copy(), n_bins, method="nope")
        check("unknown method refused", False)
    except ValueError:
        check("unknown method refused", True)

    # The default must reproduce the previous implementation draw for draw.
    try:
        old_src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:cancerfoundation/data/preprocess.py"],
                                 capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        old_src = None
    if old_src is None or "BINNING_METHODS" in old_src:
        print("  [skip] previous preprocess.py not available from git HEAD")
    else:
        old = types.ModuleType("_old_preprocess")
        exec(compile(old_src, "old_preprocess.py", "exec"), old.__dict__)
        same = True
        for row in (sc, cont):
            np.random.seed(7)
            x = old.binning(row.copy(), n_bins)
            np.random.seed(7)
            y = pp.binning(row.copy(), n_bins)
            same &= np.array_equal(x, y)
        check("quantile (default) identical to the previous binning", same)


def check_collator() -> None:
    print("\n-- AnnDataCollator --")
    try:
        import torch
    except ImportError:
        print("  [skip] torch not installed")
        return
    stub_packages()
    load_file("cancerfoundation.data.preprocess", ROOT / "cancerfoundation" / "data" / "preprocess.py")
    dc = load_file("cancerfoundation.data.data_collator", ROOT / "cancerfoundation" / "data" / "data_collator.py")

    rng = np.random.default_rng(1)
    CLS, PAD_TOKEN, PAD_VALUE, MASK_VALUE, n_bins = 0, 1, -2, -1, 51

    def examples():
        out = []
        for _ in range(4):
            n = 600
            expr = np.where(rng.random(n) < 0.3, rng.negative_binomial(1, 0.5, n) + 1, 0).astype(np.float32)
            genes = np.r_[CLS, rng.choice(np.arange(3, 5000), n, replace=False)]
            out.append({"genes": torch.from_numpy(genes), "expressions": torch.from_numpy(np.r_[PAD_VALUE, expr])})
        return out

    base = examples()

    def run(style: str, method: str, np_seed: int) -> dict:
        col = dc.AnnDataCollator(
            normalise_bins=False, condition_token=False, pad_token_id=PAD_TOKEN, pad_value=PAD_VALUE,
            mask_value=MASK_VALUE, max_length=301, data_style=style, n_bins=n_bins, mask_ratio=0.25,
            binning_method=method,
        )
        torch.manual_seed(0)
        np.random.seed(np_seed)
        batch = [{k: v.clone() for k, v in e.items()} for e in base]
        return col(batch)

    for style in ("pcpt", "both"):
        q, m, m2 = run(style, "quantile", 0), run(style, "midrank", 0), run(style, "midrank", 99)
        gene_keys = [k for k in q if k.endswith("gene")]
        check(f"[{style}] same genes in the same order", all(torch.equal(q[k], m[k]) for k in gene_keys))
        expr_key = "pcpt_expr" if style == "both" else "expr"
        check(f"[{style}] CLS value unchanged (pad_value)",
              bool((q[expr_key][:, 0] == PAD_VALUE).all() and (m[expr_key][:, 0] == PAD_VALUE).all()))
        if style == "pcpt":
            check("[pcpt] masked positions identical",
                  torch.equal(q["masked_expr"] == MASK_VALUE, m["masked_expr"] == MASK_VALUE))
            check("[pcpt] pad positions identical",
                  torch.equal(q["expr"] == PAD_VALUE, m["expr"] == PAD_VALUE))
            values = m["expr"][m["expr"] >= 0]
        else:
            check("[both] pad positions identical",
                  torch.equal(q["pcpt_expr"] == PAD_VALUE, m["pcpt_expr"] == PAD_VALUE)
                  and torch.equal(q["gen_expr_target"] == PAD_VALUE, m["gen_expr_target"] == PAD_VALUE))
            values = torch.cat([m["pcpt_expr"][m["pcpt_expr"] >= 0], m["gen_expr_target"][m["gen_expr_target"] >= 0]])
        check(f"[{style}] binned values stay in 0..n_bins-1 (no clash with mask/pad)",
              bool(values.min() >= 0 and values.max() <= n_bins - 1))
        same = all(torch.equal(m[k], m2[k]) for k in m if isinstance(m[k], torch.Tensor))
        check(f"[{style}] midrank batch independent of np.random", same)
        differ = any(not torch.equal(q[k], m[k]) for k in q if k.endswith("expr") or k.endswith("target"))
        check(f"[{style}] the methods do bin differently", differ)
    try:
        dc.AnnDataCollator(normalise_bins=False, condition_token=False, pad_token_id=PAD_TOKEN,
                           max_length=10, n_bins=n_bins, binning_method="nope")
        check("collator refuses an unknown method", False)
    except ValueError:
        check("collator refuses an unknown method", True)


if __name__ == "__main__":
    check_numpy()
    check_collator()
    print(f"\n{'All checks passed.' if not FAILURES else f'{len(FAILURES)} check(s) FAILED: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)
