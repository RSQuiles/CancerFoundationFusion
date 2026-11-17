"""Self-check for the tissue-aware contrastive loss and the general class option names.

    python scripts/check_contrastive_loss.py

Part 1 needs only the standard library: the renamed CLI flags (``--class-aware``,
``--class-column``, ...) still accept their old ``--cdd-*`` spellings, and old config
keys are translated (and refused together with the new key).
Part 2 needs torch: ``TransformerModule.modality_contrastive_loss`` against a
hand-computed value, its tissue sensitivity, exclusions, symmetry and bf16 gradients.

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
                       defaults.contrastive_temperature)
          == (False, "tissue_general", 2, ["unknown"], 8, 0.1))
    translated = _filter_known_config_keys(parser, {"cdd_class_aware": True, "cdd_class_column": "x"})
    check("old config keys translated", translated == {"class_aware": True, "class_column": "x"}, str(translated))
    try:
        _filter_known_config_keys(parser, {"cdd_class_aware": True, "class_aware": False})
        check("old + new config key refused", False)
    except ValueError:
        check("old + new config key refused", True)


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


def main() -> None:
    check_cli()
    check_loss()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    print("all checks passed")


if __name__ == "__main__":
    main()
