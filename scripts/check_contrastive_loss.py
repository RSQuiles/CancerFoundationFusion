"""Self-check for the tissue-aware contrastive loss and the general class option names.

    python scripts/check_contrastive_loss.py

Part 1 needs only the standard library: the renamed CLI flags (``--class-aware``,
``--class-column``, ...) still accept their old ``--cdd-*`` spellings, and old config
keys are translated (and refused together with the new key).
It also checks ``utils.sampling.class_aware_allocation``, the class-aware batch layout.
Part 2 needs torch: ``TransformerModule.modality_contrastive_loss`` against a
hand-computed value, its tissue sensitivity, exclusions, symmetry, bf16 gradients and
monitoring stats; then ``BulkSCSampler.sample_class_aware_batch`` on fake pools (shared,
bulk-only, pseudobulk-only and excluded tissues).

The package ``__init__`` imports scanpy and the model, so ``module.py`` is loaded as a
plain file under stub packages (its ``layers``/``utils`` imports stubbed); nothing here
needs a GPU, a checkpoint or data.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def check_cli() -> None:
    print("CLI and config names")
    from utils_config import build_parser, _filter_known_config_keys

    parser = build_parser()
    old = parser.parse_args(["--cdd-class-aware", "--cdd-bulk-class-frac", "0.3",
                             "--cdd-class-column", "tissue", "--cdd-exclude-labels", "a", "b",
                             "--cdd-min-class-count", "4"])
    new = parser.parse_args(["--class-aware", "--class-aware-bulk-frac", "0.3",
                             "--class-column", "tissue", "--class-exclude-labels", "a", "b",
                             "--class-min-count", "4"])
    keys = ("class_aware", "class_aware_bulk_frac", "class_column", "class_exclude_labels", "class_min_count")
    check("old flags map onto the new dests", all(getattr(old, k) == getattr(new, k) for k in keys),
          str({k: (getattr(old, k), getattr(new, k)) for k in keys}))
    defaults = parser.parse_args([])
    check("defaults", (defaults.class_aware, defaults.class_column, defaults.class_min_count,
                       defaults.class_exclude_labels, defaults.class_aware_n_classes,
                       defaults.contrastive_temperature, defaults.class_aware_pb_frac)
          == (False, "tissue_general", 2, ["unknown"], 8, 0.1, 0.6))
    translated = _filter_known_config_keys(parser, {"cdd_class_aware": True, "cdd_class_column": "x"})
    check("old config keys translated", translated == {"class_aware": True, "class_column": "x"}, str(translated))
    try:
        _filter_known_config_keys(parser, {"cdd_class_aware": True, "class_aware": False})
        check("old + new config key refused", False)
    except ValueError:
        check("old + new config key refused", True)


def check_allocation() -> None:
    print("class-aware batch layout")
    from utils.sampling import class_aware_allocation

    # batch 64, bulk/pb ratio 0.35: 22 + 22 rows, 13 matched bulk. Round-robin over 8
    # tissues would leave 3 of them with one bulk row; K drops to 6 (>= 2 rows each).
    a = class_aware_allocation(22, 22, 30, 8, 2, pb_frac=1.0, bulk_frac=0.6)
    check("K lowered so every tissue has >= class_min_count",
          (a["k_requested"], a["k"], a["bulk_per_tissue"], a["pb_per_tissue"])
          == (8, 6, (2, 3), (3, 4)), str(a))
    a = class_aware_allocation(22, 22, 30, 4, 2, pb_frac=1.0, bulk_frac=0.6)
    check("K kept when the slots suffice", a["k"] == 4 and a["n_pb_free"] == 0, str(a))
    a = class_aware_allocation(22, 22, 30, 8, 2, pb_frac=0.75, bulk_frac=0.6)
    check("pb_frac splits matched / free pseudobulk",
          (a["n_pb_matched"], a["n_pb_free"], a["n_bulk_matched"], a["n_bulk_free"], a["k"])
          == (16, 6, 13, 9, 6), str(a))
    a = class_aware_allocation(22, 22, 3, 8, 2, pb_frac=1.0)
    check("K capped by the shared tissues", a["k"] == 3, str(a))
    a = class_aware_allocation(22, 2, 30, 8, 2, pb_frac=1.0, bulk_frac=0.0)
    check("bulk_frac 0 does not limit K", a["k"] == 8 and a["n_bulk_matched"] == 0, str(a))


def load_module_py():
    """``cancerfoundation.model.module`` without the package __init__ or its heavy imports."""
    for name, path in (("cancerfoundation", ROOT / "cancerfoundation"),
                       ("cancerfoundation.model", ROOT / "cancerfoundation" / "model")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
    layers = types.ModuleType("cancerfoundation.model.layers")
    for cls in ("RefactoredCFGenerator", "QuickCFGenerator", "CFLayer", "CFGenerator"):
        setattr(layers, cls, type(cls, (), {}))
    sys.modules["cancerfoundation.model.layers"] = layers
    sys.modules["cancerfoundation.model.utils"] = types.ModuleType("cancerfoundation.model.utils")
    spec = importlib.util.spec_from_file_location(
        "cancerfoundation.model.module", ROOT / "cancerfoundation" / "model" / "module.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cancerfoundation.model.module"] = module
    spec.loader.exec_module(module)
    return module


def check_loss() -> None:
    print("contrastive loss")
    try:
        import torch
    except ImportError:
        print("  [skip] torch not importable")
        return
    loss_fn = load_module_py().TransformerModule.modality_contrastive_loss

    def loss(emb, mod, lab, **kw):
        return float(loss_fn(None, emb, torch.tensor(mod), torch.tensor(lab), **kw))

    t = 0.1
    e = torch.eye(4)
    # bulk rows 0,1 (tissues 0,1), pseudobulk rows 2,3 at the same points (tissues 0,1)
    emb = torch.stack([e[0], e[1], e[0], e[1]])
    mod, lab = [0, 0, 2, 2], [0, 1, 0, 1]
    expected = -(1 / t - math.log(math.exp(1 / t) + 1))
    got = loss(emb, mod, lab, temperature=t)
    check("hand-computed value", abs(got - expected) < 1e-5, f"{got} vs {expected}")

    swapped = loss(emb, mod, [0, 1, 1, 0], temperature=t)
    check("matching tissues give a lower loss than swapped ones", got < swapped, f"{got} vs {swapped}")

    # rows that must not change anything: a single cell, an excluded label, a -1 label
    emb_x = torch.cat([emb, torch.stack([e[2], e[3], e[2]])])
    got_x = loss(emb_x, mod + [1, 0, 2], lab + [0, 7, -1], temperature=t, exclude_class_codes=(7,))
    check("sc, excluded and -1 rows have no effect", abs(got_x - got) < 1e-5, f"{got_x} vs {got}")

    check("no shared tissue -> 0", loss(emb, mod, [0, 0, 1, 1], temperature=t) == 0.0)
    check("min_class_count: classes below it are not anchors",
          loss(emb, mod, lab, temperature=t, min_class_count=2) == 0.0)

    gen = torch.Generator().manual_seed(0)
    emb_r = torch.randn(12, 16, generator=gen)
    mod_r = [0] * 6 + [2] * 6
    lab_r = [0, 0, 1, 1, 2, 3, 0, 1, 1, 2, 2, 4]
    a = loss(emb_r, mod_r, lab_r, temperature=t)
    b = loss(emb_r, [2 if m == 0 else 0 for m in mod_r], lab_r, temperature=t)
    check("symmetric in bulk <-> pseudobulk", abs(a - b) < 1e-5, f"{a} vs {b}")

    x = torch.randn(12, 3, 16, generator=gen).to(torch.bfloat16).requires_grad_(True)
    value = loss_fn(None, x, torch.tensor(mod_r), torch.tensor(lab_r), temperature=t)
    value.backward()
    check("bf16 (B, L, D) input: finite value and gradient on the CLS only",
          bool(torch.isfinite(value)) and bool(torch.isfinite(x.grad.float()).all())
          and float(x.grad[:, 1:].abs().sum()) == 0.0 and float(x.grad[:, 0].abs().sum()) > 0)


def load_sampler_cls():
    """``BulkSCSampler`` without the package __init__, SCDL or the store reader."""
    for name, path in (("cancerfoundation", ROOT / "cancerfoundation"),
                       ("cancerfoundation.data", ROOT / "cancerfoundation" / "data")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules.setdefault(name, pkg)
    ds = types.ModuleType("cancerfoundation.data.dataset")
    ds.DatasetDir = type("DatasetDir", (), {})
    ds.open_store_memmap = lambda *a, **k: None
    sys.modules["cancerfoundation.data.dataset"] = ds
    du = types.ModuleType("cancerfoundation.data.utils")
    du.concat_categorical_codes = lambda *a, **k: None
    sys.modules["cancerfoundation.data.utils"] = du
    spec = importlib.util.spec_from_file_location(
        "cancerfoundation.data.bulk_sc_data", ROOT / "cancerfoundation" / "data" / "bulk_sc_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cancerfoundation.data.bulk_sc_data"] = module
    spec.loader.exec_module(module)
    return module.BulkSCSampler


def check_sampler() -> None:
    print("class-aware sampler")
    try:
        import numpy as np
        import torch  # noqa: F401  (bulk_sc_data imports it)
    except ImportError:
        print("  [skip] torch not importable")
        return
    import contextlib
    import io

    Sampler = load_sampler_cls()
    # tissues: 0, 1, 2 shared; 3 bulk-only; 4 pseudobulk-only; 9 excluded ("unknown")
    bulk_t = [0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [9] * 40
    pb_t = [0] * 30 + [1] * 30 + [2] * 30 + [4] * 30 + [9] * 30
    n_sc = 100
    sc = np.arange(n_sc)
    pb = np.arange(n_sc, n_sc + len(pb_t))
    bulk = np.arange(pb[-1] + 1, pb[-1] + 1 + len(bulk_t))
    tissue = {**dict(zip(pb.tolist(), pb_t)), **dict(zip(bulk.tolist(), bulk_t))}

    def groups(idx, labels):
        out: dict = {}
        for i, t in zip(idx, labels):
            out.setdefault(t, []).append(int(i))
        return {t: np.array(v) for t, v in out.items()}

    class FakeDataset(types.SimpleNamespace):
        def __len__(self):
            return n_sc + len(pb_t) + len(bulk_t)

    dataset = FakeDataset(
        bulk_indices=bulk, sc_indices=sc, pb_indices=pb,
        sc_group_to_indices=None, bulk_group_to_indices=groups(bulk, bulk_t),
        pb_group_to_indices=groups(pb, pb_t), sc_pair_to_indices={}, sc_pb_to_indices={},
        labels=None,
    )

    def make(pb_frac, n_classes=8):
        with contextlib.redirect_stdout(io.StringIO()):
            return Sampler(dataset, batch_size=64, bulk_ratio=0.35, pb_ratio=0.35,
                           precomputed_pb=True, class_aware=True,
                           class_aware_exclude_group_codes=[9], class_min_count=2,
                           class_aware_n_classes=n_classes, class_aware_bulk_frac=0.6,
                           class_aware_pb_frac=pb_frac, seed=0, epoch_size=200)

    for pb_frac in (1.0, 0.75):
        s = make(pb_frac)
        plan = s.class_aware_plan
        ok_layout = ok_count = True
        pb_tissues: set = set()
        for batch in s:
            batch = [int(i) for i in batch]
            sc_part = batch[:s.n_sc]
            pb_part = batch[s.n_sc:s.n_sc + s.n_pb]
            bulk_part = batch[s.n_sc + s.n_pb:]
            ok_layout &= (len(batch) == s.raw_batch_size and all(i < n_sc for i in sc_part)
                          and all(i in tissue and i < bulk[0] for i in pb_part)
                          and all(i >= bulk[0] for i in bulk_part))
            m_pb = [tissue[i] for i in pb_part[:plan["n_pb_matched"]]]
            m_bulk = [tissue[i] for i in bulk_part[:plan["n_bulk_matched"]]]
            for t in set(m_pb):
                ok_count &= m_pb.count(t) >= 2 and m_bulk.count(t) >= 2
            pb_tissues |= {tissue[i] for i in pb_part}
        check(f"pb_frac {pb_frac}: batch layout [sc, pb, bulk] and sizes", bool(ok_layout))
        check(f"pb_frac {pb_frac}: every chosen tissue >= 2 rows in both domains", bool(ok_count))
        if pb_frac == 1.0:
            check("pb_frac 1.0: only shared tissues drawn as pseudobulk", pb_tissues <= {0, 1, 2},
                  str(pb_tissues))
        else:
            check("pb_frac < 1: pseudobulk-only and excluded tissues are drawn too",
                  {4, 9} <= pb_tissues, str(pb_tissues))
    s = make(1.0, n_classes=2)
    check("K kept when the slots suffice (sampler)", s.class_aware_plan["k"] == 2)


def check_stats() -> None:
    print("monitoring stats")
    try:
        import torch
    except ImportError:
        print("  [skip] torch not importable")
        return
    mod = load_module_py().TransformerModule
    e = torch.eye(4)
    emb = torch.stack([e[0], e[1], e[2], e[0], e[1]])
    stats: dict = {}
    mod.modality_contrastive_loss(None, emb, torch.tensor([0, 0, 0, 2, 2]),
                                  torch.tensor([0, 1, 3, 0, 1]), stats=stats)
    check("contrastive anchor shares (2 of 3 bulk, 2 of 2 pseudobulk)",
          abs(float(stats["class_anchor_frac_bulk"]) - 2 / 3) < 1e-6
          and float(stats["class_anchor_frac_pb"]) == 1.0, str(stats))
    stats = {}
    fake = types.SimpleNamespace(_pairwise_sq_dists=mod._pairwise_sq_dists)
    mod._cdd_loss(fake, emb[:3], torch.tensor([0, 0, 1]), emb[3:], torch.tensor([0, 0]),
                  min_class_count=2, stats=stats)
    check("cdd_n_classes counts the classes used", float(stats["cdd_n_classes"]) == 1.0, str(stats))


def main() -> None:
    check_cli()
    check_allocation()
    check_loss()
    check_stats()
    check_sampler()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    print("all checks passed")


if __name__ == "__main__":
    main()
