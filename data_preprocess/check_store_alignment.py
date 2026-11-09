"""Check that a training store matches its source h5ad and the file evaluation reads.

Read-only diagnostic for a store built by ``build_pipeline_ready`` (or
``split_train_test.py --build-store``). Four checks, each PASS or FAIL:

0. **Sources.** The store is built from the split's input (``split.json`` -> source path);
   reconstruction reads the mask plan's source (``layout.raw_h5ad`` unless ``--input`` was
   given). If they are different files, their genes and the values of shared rows are
   compared.
1. **Store vs training source, row by row.** Token ids map back (through ``vocab.json``)
   to the source's in-vocabulary genes, and the stored values equal the source values.
2. **Evaluation-side gene ids.** The reconstructor maps ``var_names`` through the vocabulary;
   the builder mapped ``--gene-column`` (default: also ``var_names``). Any column whose two
   ids differ, and any mask-plan gene affected, is reported.
3. **Data profile, train rows vs test rows** (informative, flags large shifts): library
   size, non-zero genes, value scale, tied-bin width and how much a random 720-gene
   context carries.

    python data_preprocess/check_store_alignment.py --modality bulk --split leiden_s0
    python data_preprocess/check_store_alignment.py --store <dir> --source a.h5ad \\
        --eval-source b.h5ad --test-names test.txt --out-dir out/

Needs bionemo SCDL (the container). Writes ``report.json`` to
``results/assessment/store_check_<modality>_<split>/`` (or ``--out-dir``) and exits 1 on
any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "data_preprocess"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from h5ad_io import SourceReader  # noqa: E402

CLS_TOKEN, PAD_TOKEN = "<cls>", "<pad>"
SPECIAL = {CLS_TOKEN, PAD_TOKEN}
SHIFT_RATIO = 2.0  # train/test median ratio above which the profile flags a shift


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class Report:
    def __init__(self) -> None:
        self.checks: dict[str, dict] = {}

    def add(self, name: str, ok: bool | None, **details) -> None:
        status = "PASS" if ok else ("INFO" if ok is None else "FAIL")
        self.checks[name] = {"status": status, **details}
        print(f"\n[{status}] {name}")
        for key, value in details.items():
            if key == "examples" and value:
                print(f"    {key}:")
                for ex in value[:5]:
                    print(f"      {ex}")
            elif not isinstance(value, (dict, list)) or len(str(value)) < 160:
                print(f"    {key}: {value}")

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks.values())


def dense_rows(reader: SourceReader, positions: np.ndarray) -> np.ndarray:
    """Rows at ``positions`` (any order) as a dense float32 array, in that order."""
    positions = np.asarray(positions, dtype=np.int64)
    order = np.argsort(positions)
    block = reader.rows(positions[order]).toarray().astype(np.float32)
    out = np.empty_like(block)
    out[order] = block
    return out


def builder_genes(reader: SourceReader, vocab: dict, gene_column: str | None):
    """``(in_vocab mask, names)`` exactly as ``build_pipeline_ready`` selects them."""
    names = reader.gene_names(gene_column)
    in_vocab = np.asarray(names.isin(list(vocab))) & ~np.asarray(names.isin(list(SPECIAL)))
    return in_vocab, names


def quantiles(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {f"q{int(p * 100):02d}": float(x) for p, x in zip([0.05, 0.25, 0.5, 0.75, 0.95], q)}


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_sources(report: Report, train_source: Path, eval_source: Path, test_names: list[str],
                  n_rows: int, rng: np.random.Generator) -> None:
    same = train_source.resolve() == eval_source.resolve()
    details = {"train_source": str(train_source), "eval_source": str(eval_source), "same_file": same}
    if same:
        report.add("0. store source vs evaluation source", True, **details)
        return
    with SourceReader(train_source) as tr, SourceReader(eval_source) as ev:
        tg, eg = tr.gene_names(), ev.gene_names()
        common = tg.intersection(eg)
        details.update(train_genes=len(tg), eval_genes=len(eg), common_genes=len(common),
                       same_gene_order=bool(len(tg) == len(eg) and (tg == eg).all()))
        names = [n for n in test_names if n in tr.obs.index and n in ev.obs.index]
        details["test_rows_in_both"] = f"{len(names)} of {len(test_names)}"
        if not names or not len(common):
            report.add("0. store source vs evaluation source", False,
                       note="the files share no test rows or no genes", **details)
            return
        pick = rng.choice(len(names), size=min(n_rows, len(names)), replace=False)
        names = [names[i] for i in pick]
        a = dense_rows(tr, tr.obs.index.get_indexer(names))[:, tg.get_indexer(common)]
        b = dense_rows(ev, ev.obs.index.get_indexer(names))[:, eg.get_indexer(common)]
        diff = np.abs(a - b)
        rows_differ = int((diff.max(axis=1) > 0).sum())
        # Rank agreement tells "same data, other scale" apart from "different data".
        from scipy.stats import spearmanr

        rhos = [spearmanr(a[i], b[i])[0] for i in range(len(names)) if a[i].std() and b[i].std()]
        details.update(rows_compared=len(names), rows_with_different_values=rows_differ,
                       max_abs_diff=float(diff.max()),
                       median_rank_corr=float(np.median(rhos)) if rhos else None,
                       examples=[f"{names[i]}: max |diff| {diff[i].max():.4g}"
                                 for i in np.flatnonzero(diff.max(axis=1) > 0)[:5]])
    report.add("0. store source vs evaluation source", rows_differ == 0 and details["same_gene_order"],
               note=("evaluation reads a different file than the one the store was built from; "
                     "the values it bins must equal the training values"), **details)


def check_store_rows(report: Report, ds, vocab: dict, train_source: Path, gene_column: str | None,
                     n_rows: int, rng: np.random.Generator) -> None:
    inv = {v: k for k, v in vocab.items()}
    with SourceReader(train_source) as reader:
        in_vocab, names = builder_genes(reader, vocab, gene_column)
        exp_ids = np.array([vocab[g] for g in names[in_vocab]], dtype=np.int64)
        dup_ids = int(len(exp_ids) - len(np.unique(exp_ids)))
        idx = np.sort(rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False))
        obs_names = ds.obs.index.astype(str)[idx]
        pos = reader.obs.index.astype(str).get_indexer(obs_names)
        missing = [str(n) for n, p in zip(obs_names, pos) if p < 0]
        ok_rows = np.flatnonzero(pos >= 0)
        src = dense_rows(reader, pos[ok_rows])[:, in_vocab] if len(ok_rows) else np.zeros((0, 0))

    bad_genes, bad_values, order_differs, max_diff, examples = 0, 0, 0, 0.0, []
    for k, j in enumerate(ok_rows):
        item = ds[int(idx[j])]
        ids = item["genes"].numpy()[1:].astype(np.int64)
        vals = item["expressions"].numpy()[1:].astype(np.float32)
        if len(ids) != len(exp_ids) or not np.array_equal(np.sort(ids), np.sort(exp_ids)):
            bad_genes += 1
            extra = sorted(set(ids) - set(exp_ids))[:3]
            examples.append(f"{obs_names[j]}: {len(ids)} stored genes vs {len(exp_ids)} expected; "
                            f"unexpected tokens {[inv.get(t, t) for t in extra]}")
            continue
        if not np.array_equal(ids, exp_ids):
            order_differs += 1
        # Compare (id, value) pairs as multisets, so duplicate gene names are handled.
        s_ord, e_ord = np.lexsort((vals, ids)), np.lexsort((src[k], exp_ids))
        d = np.abs(vals[s_ord] - src[k][e_ord])
        max_diff = max(max_diff, float(d.max()) if d.size else 0.0)
        if d.size and d.max() > 0:
            bad_values += 1
            worst = int(np.argmax(d))
            examples.append(f"{obs_names[j]}: gene {inv.get(int(ids[s_ord][worst]))} store "
                            f"{vals[s_ord][worst]:.6g} vs source {src[k][e_ord][worst]:.6g}")
    report.add(
        "1. store rows vs training source", bad_genes == 0 and bad_values == 0 and not missing,
        rows_checked=int(len(ok_rows)), rows_missing_in_source=len(missing),
        rows_with_wrong_genes=bad_genes, rows_with_wrong_values=bad_values,
        rows_in_another_gene_order=order_differs, max_abs_value_diff=max_diff,
        duplicate_gene_ids_in_source=dup_ids, store_vocab_genes=len(vocab) - 2,
        source_genes_in_vocab=int(in_vocab.sum()),
        examples=examples + [f"missing in source: {m}" for m in missing[:3]],
    )


def check_eval_ids(report: Report, vocab: dict, eval_source: Path, gene_column: str | None,
                   plan_path: Path | None) -> None:
    with SourceReader(eval_source) as reader:
        var_names = reader.gene_names(None)
        build_names = reader.gene_names(gene_column)
    eval_ids = np.array([vocab.get(g, -1) if g not in SPECIAL else -1 for g in var_names])
    build_ids = np.array([vocab.get(g, -1) if g not in SPECIAL else -1 for g in build_names])
    differ = np.flatnonzero(eval_ids != build_ids)
    dup_var = int(var_names.duplicated().sum())
    eval_in = eval_ids >= 0
    dup_tokens = int(eval_in.sum() - len(np.unique(eval_ids[eval_in])))
    details = {
        "eval_source": str(eval_source), "gene_column": gene_column,
        "columns": len(var_names), "columns_in_vocab_eval_side": int(eval_in.sum()),
        "columns_whose_token_differs": int(len(differ)), "duplicate_var_names": dup_var,
        "columns_sharing_a_token_eval_side": dup_tokens,
        "examples": [f"column {i}: var_name {var_names[i]} -> {eval_ids[i]}, "
                     f"build name {build_names[i]} -> {build_ids[i]}" for i in differ[:5]],
    }
    ok = len(differ) == 0 and dup_var == 0
    if plan_path is not None and plan_path.exists():
        from evaluate.reconstruction.mask_plan import load_plan

        plan = load_plan(plan_path)
        col = var_names.get_indexer(plan.genes)
        affected = int(((col < 0) | (eval_ids[np.maximum(col, 0)] != build_ids[np.maximum(col, 0)])).sum())
        plan_src = plan.meta.get("source", {}).get("path")
        details.update(plan=str(plan_path), plan_source=plan_src, plan_genes=len(plan.genes),
                       plan_genes_with_other_token=affected)
        if plan_src and Path(plan_src).resolve() != eval_source.resolve():
            details["note"] = "the plan was built on another file than --eval-source"
        ok = ok and affected == 0
    report.add("2. evaluation-side gene ids", ok, **details)


def profile_rows(X: np.ndarray, n_bins: int, ctx_len: int, rng: np.random.Generator) -> dict:
    from cancerfoundation.data.preprocess import binning, value_scale
    from evaluate.reconstruction.reconstructors import bin_tie_range

    lib = X.sum(axis=1)
    nnz = (X > 0).sum(axis=1)
    tie_width, ctx_zero, ctx_low = [], [], []
    for row in X:
        lo, hi = bin_tie_range(row, n_bins)
        nz = row > 0
        tie_width.append(float((hi - lo)[nz].mean()) if nz.any() else np.nan)
        bins = np.asarray(binning(row.copy(), n_bins))
        ctx = bins[rng.choice(len(row), size=min(ctx_len, len(row)), replace=False)]
        ctx_zero.append(float((ctx == 0).mean()))
        ctx_low.append(float(((ctx >= 1) & (ctx <= 5)).mean()))
    return {
        "rows": int(len(X)), "genes": int(X.shape[1]), "value_scale": value_scale(X),
        "library_size": quantiles(lib), "nonzero_genes": quantiles(nnz),
        "zero_fraction": quantiles(1 - nnz / max(X.shape[1], 1)),
        "mean_tie_width_bins": quantiles(tie_width),
        f"context{ctx_len}_share_zero": quantiles(ctx_zero),
        f"context{ctx_len}_share_bins_1_5": quantiles(ctx_low),
    }


def check_profile(report: Report, ds, vocab: dict, eval_source: Path, test_names: list[str],
                  n_rows: int, n_bins: int, ctx_len: int, rng: np.random.Generator) -> None:
    idx = np.sort(rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False))
    train_X = np.stack([ds[int(i)]["expressions"].numpy()[1:] for i in idx]).astype(np.float32)
    with SourceReader(eval_source) as reader:
        var_names = reader.gene_names(None)
        cols = np.flatnonzero(np.asarray(var_names.isin(list(vocab))) & ~np.asarray(var_names.isin(list(SPECIAL))))
        names = [n for n in test_names if n in reader.obs.index]
        names = [names[i] for i in rng.choice(len(names), size=min(n_rows, len(names)), replace=False)]
        test_X = dense_rows(reader, reader.obs.index.get_indexer(names))[:, cols] if names else None
    train = profile_rows(train_X, n_bins, ctx_len, rng)
    test = profile_rows(test_X, n_bins, ctx_len, rng) if test_X is not None else {}
    flags = []
    for key in ("library_size", "nonzero_genes"):
        a, b = train.get(key, {}).get("q50"), test.get(key, {}).get("q50")
        if a and b and max(a / b, b / a) > SHIFT_RATIO:
            flags.append(f"median {key}: train {a:.4g} vs test {b:.4g}")
    if test and train["value_scale"] != test["value_scale"]:
        flags.append(f"value scale: train {train['value_scale']} vs test {test['value_scale']}")
    # Informative only: a shift is a finding about the split, not a broken store.
    report.add("3. data profile (train store rows vs evaluation test rows)", None,
               shifts=flags or "none above the thresholds", train=train, test=test)
    for label, prof in (("train", train), ("test", test)):
        if prof:
            print(f"    {label}: value_scale={prof['value_scale']}, "
                  f"median library={prof['library_size'].get('q50', float('nan')):.4g}, "
                  f"median non-zero genes={prof['nonzero_genes'].get('q50', float('nan')):.0f} of {prof['genes']}, "
                  f"median tie width={prof['mean_tie_width_bins'].get('q50', float('nan')):.2f} bins, "
                  f"context zeros={prof[f'context{ctx_len}_share_zero'].get('q50', float('nan')):.2f}, "
                  f"context bins 1-5={prof[f'context{ctx_len}_share_bins_1_5'].get('q50', float('nan')):.2f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None, help="Experiments root")
    p.add_argument("--modality", default=None, help="Layout mode: bulk | sc | pseudobulk")
    p.add_argument("--split", default=None, help="Layout mode: split name")
    p.add_argument("--store", type=Path, default=None, help="Store directory (default: pipeline_ready/<split>)")
    p.add_argument("--source", type=Path, default=None, help="h5ad the store was built from (default: split.json source)")
    p.add_argument("--eval-source", type=Path, default=None,
                   help="h5ad reconstruction reads (default: the plan's source, else layout.raw_h5ad)")
    p.add_argument("--plan", default=None, help="Mask plan MODALITY:SPLIT:PART:NAME or .npz (default: <mod>:<split>:test:r25 if present)")
    p.add_argument("--test-names", type=Path, default=None, help="Test row names (default: the split's test.txt)")
    p.add_argument("--gene-column", default=None, help="var column the store was built with (default: var index)")
    p.add_argument("--n-rows", type=int, default=200, help="Rows compared in checks 0 and 1")
    p.add_argument("--profile-rows", type=int, default=1000, help="Rows per side in the profile")
    p.add_argument("--n-bins", type=int, default=51)
    p.add_argument("--context-len", type=int, default=720)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def resolve_inputs(args):
    layout = None
    if args.modality and args.split:
        from experiments.layout import ExperimentLayout, canonical_modality

        layout = ExperimentLayout.from_root(args.root)
        args.modality = canonical_modality(args.modality)
    store = args.store or (layout.pipeline_ready(args.modality, args.split) if layout else None)
    if store is None:
        sys.exit("Error: give --modality and --split, or --store")
    source = args.source
    if source is None and layout is not None:
        card = layout.split_card(args.modality, args.split)
        if card.exists():
            source = Path(json.loads(card.read_text(encoding="utf-8"))["source"]["path"])
            print(f"Training source from {card.name}: {source}")
    if source is None:
        sys.exit("Error: the store's source h5ad is unknown; pass --source")

    plan_path = None
    spec = args.plan or (f"{args.modality}:{args.split}:test:r25" if layout else None)
    if spec:
        from evaluate.reconstruction.mask_plan import resolve_plan_path

        plan_path = resolve_plan_path(spec, args.root)
        if not plan_path.exists():
            print(f"  [info] no mask plan at {plan_path}; plan checks skipped")
            plan_path = None
    eval_source = args.eval_source
    if eval_source is None and plan_path is not None:
        from evaluate.reconstruction.mask_plan import load_plan

        eval_source = Path(load_plan(plan_path).meta["source"]["path"])
    if eval_source is None:
        eval_source = layout.raw_h5ad(args.modality) if layout else source

    test_file = args.test_names or (layout.split_index_file(args.modality, args.split, "test") if layout else None)
    test_names = test_file.read_text(encoding="utf-8").split() if test_file and test_file.exists() else []
    out_dir = args.out_dir or (layout.assessment_dir(f"store_check_{args.modality}_{args.split}") if layout else None)
    if out_dir is None:
        sys.exit("Error: give --out-dir")
    return Path(store), Path(source), Path(eval_source), plan_path, test_names, Path(out_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store, source, eval_source, plan_path, test_names, out_dir = resolve_inputs(args)
    from cancerfoundation.data.dataset import SingleCellDataset

    print(f"Store:         {store}\nTrain source:  {source}\nEval source:   {eval_source}\n"
          f"Plan:          {plan_path}\nTest names:    {len(test_names)}")
    ds = SingleCellDataset(store, pad_value=-2.0)
    vocab = ds.vocab
    report = Report()
    rng = np.random.default_rng(args.seed)

    check_sources(report, source, eval_source, test_names, args.n_rows, rng)
    check_store_rows(report, ds, vocab, source, args.gene_column, args.n_rows, rng)
    check_eval_ids(report, vocab, eval_source, args.gene_column, plan_path)
    if test_names:
        check_profile(report, ds, vocab, eval_source, test_names, args.profile_rows, args.n_bins,
                      args.context_len, rng)
    else:
        print("\n[SKIP] 3. data profile: no test row names")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"store": str(store), "source": str(source), "eval_source": str(eval_source),
               "plan": str(plan_path) if plan_path else None, "gene_column": args.gene_column,
               "seed": args.seed, "checks": report.checks}
    out = out_dir / "report.json"
    out.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    print(f"\n{'FAIL' if report.failed else 'PASS'}: wrote {out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
