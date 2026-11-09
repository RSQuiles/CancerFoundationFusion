"""Check that a training store matches its source h5ad and the file evaluation reads.

Read-only diagnostic for a store built by ``build_pipeline_ready`` (or
``split_train_test.py --build-store``). Five checks, each PASS or FAIL (3 is informative):

0. **Sources.** The store is built from the split's input (``split.json`` -> source path);
   reconstruction reads the mask plan's source (``layout.raw_h5ad`` unless ``--input`` was
   given). If they are different files, their genes and the values of shared rows are
   compared.
1. **Store vs training source, row by row.** Token ids map back (through ``vocab.json``)
   to the source's in-vocabulary genes, and the stored values equal the source values.
1b. **Corruption mechanism.** How the stored entries relate to the ones SCDL was given:
   memmap/source inventory (dtypes, sizes, SCDL version), row pointers vs the source
   ``indptr``, per-row identity / integer-width-misread / offset tests on the raw memmap
   arrays, and a reproduction of the builder's conversion on ``--repro-rows`` rows with
   variants of the transient matrix (as read, canonical CSR, float32+int32, float32+int64).
   ``--only-diagnose`` runs this check alone and writes ``corruption.json``.
2. **Evaluation-side gene ids.** The reconstructor maps ``var_names`` through the vocabulary;
   the builder mapped ``--gene-column`` (default: also ``var_names``). Any column whose two
   ids differ, and any mask-plan gene affected, is reported.
3. **Data profile, train rows vs test rows** (informative, flags large shifts): library
   size, non-zero genes, value scale, tied-bin width and how much a random 720-gene
   context carries.

    python data_preprocess/check_store_alignment.py --modality bulk --split leiden_s0
    python data_preprocess/check_store_alignment.py --modality bulk --split leiden_s0 --only-diagnose
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
import pandas as pd

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
# 1b. corruption mechanism
# --------------------------------------------------------------------------- #

COL_WIDTHS = [("int64", "int32"), ("int64", "uint32"), ("int32", "int64"), ("int32", "uint16"),
              ("int64", "uint16"), ("int32", "int32"), ("int64", "int64")]
DATA_WIDTHS = [("float64", "float32"), ("float32", "float64"), ("int64", "float32"), ("int64", "int32"),
               ("int32", "float32"), ("float32", "int32"), ("float64", "float64"), ("float32", "float32")]


def _memmap_arrays(memmap) -> dict:
    """``{"data", "cols", "rows"}`` arrays of an SCDL memmap dataset, found by name then shape."""
    arrays = {k: v for k, v in vars(memmap).items() if isinstance(v, np.ndarray)}
    found = {}
    for role, names in (("data", ("data",)), ("cols", ("col_index", "col_ptr", "colptr", "indices")),
                        ("rows", ("row_index", "row_ptr", "rowptr", "indptr"))):
        for name in names:
            if name in arrays:
                found[role] = (name, arrays[name])
                break
    return found, {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()}


def _inventory(store: Path, memmap, reader: SourceReader) -> dict:
    inv: dict = {}
    try:
        from importlib.metadata import version

        inv["bionemo_scdl_version"] = version("bionemo-scdl")
    except Exception as exc:  # not installed as a distribution, or another name
        inv["bionemo_scdl_version"] = f"unknown ({exc.__class__.__name__})"
    mm_dir = store / "mem.map"
    files = {}
    for path in sorted(mm_dir.rglob("*")) if mm_dir.exists() else []:
        if path.is_file():
            rel = str(path.relative_to(mm_dir))
            files[rel] = path.stat().st_size
            if path.suffix == ".json" and path.stat().st_size < 20_000:
                try:
                    inv.setdefault("json", {})[rel] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
    inv["memmap_files"] = files
    _, inv["memmap_arrays"] = _memmap_arrays(memmap)
    for attr in ("dtypes", "data_dtype", "version", "mode"):
        if hasattr(memmap, attr):
            inv[f"memmap.{attr}"] = str(getattr(memmap, attr))
    X = reader._file["X"]
    import h5py

    if isinstance(X, h5py.Group):
        inv["source_X"] = {"encoding": str(X.attrs.get("encoding-type")), "shape": [int(s) for s in X.attrs.get("shape", [])],
                           **{k: str(X[k].dtype) for k in ("data", "indices", "indptr") if k in X}}
    else:
        inv["source_X"] = {"encoding": "dense", "dtype": str(X.dtype), "shape": list(X.shape)}
    return inv


class _Stream:
    """The entry stream SCDL was given: CSR rows of the store's source rows, in store order."""

    def __init__(self, reader: SourceReader, positions: np.ndarray, in_vocab: np.ndarray, ptr: np.ndarray | None):
        self.reader, self.positions, self.in_vocab, self.ptr = reader, positions, in_vocab, ptr

    def rows(self, r0: int, r1: int):
        pos = self.positions[r0:r1]
        order = np.argsort(pos)
        X = self.reader.rows(pos[order])[:, self.in_vocab].tocsr()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return X[inv]

    def entries(self, lo: int, hi: int):
        """``(cols, values)`` of global entries ``[lo, hi)`` (clipped)."""
        lo, hi = max(lo, 0), min(hi, int(self.ptr[-1]))
        if hi <= lo:
            return np.zeros(0, np.int64), np.zeros(0, np.float64)
        r0 = int(np.searchsorted(self.ptr, lo, side="right") - 1)
        r1 = int(np.searchsorted(self.ptr, hi, side="left"))
        X = self.rows(r0, r1)
        base = int(self.ptr[r0])
        return X.indices[lo - base:hi - base], X.data[lo - base:hi - base]


def _as_read(stream: _Stream, which: int, orig: str, read: str, g0: int, g1: int) -> np.ndarray:
    """Global entries ``[g0, g1)`` as they would look if an ``orig`` array were read as ``read``."""
    wo, wr = np.dtype(orig).itemsize, np.dtype(read).itemsize
    e_lo, e_hi = (g0 * wr) // wo, -(-(g1 * wr) // wo) + 1
    seg = stream.entries(e_lo, e_hi)[which]
    raw = np.asarray(seg).astype(orig).tobytes()
    view = np.frombuffer(raw[: len(raw) // wr * wr], dtype=read)  # a wider read can leave a partial tail
    skip = (g0 * wr - e_lo * wo) // wr
    out = np.full(g1 - g0, np.nan)
    got = view[skip:skip + (g1 - g0)].astype(np.float64)
    out[:len(got)] = got
    return out


def _match(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    return float(np.mean(a[:n] == b[:n])) if n else float("nan")


def _find_offset(stream: _Stream, values: np.ndarray, g0: int, window: int) -> int | None:
    """Offset ``d`` such that the stored values equal the original stream at ``g0 + d``."""
    probe = values[values != 0][:16]
    if len(probe) < 4:
        return None
    lo = max(g0 - window, 0)
    _, seg = stream.entries(lo, g0 + len(values) + window)
    seg = np.asarray(seg, dtype=np.float64)
    nz = np.flatnonzero(seg != 0)
    segnz = seg[nz]
    for c in np.flatnonzero(segnz == probe[0]):
        if c + len(probe) <= len(segnz) and np.array_equal(segnz[c:c + len(probe)], probe):
            first_stored = int(np.flatnonzero(values != 0)[0])
            return int(lo + nz[c] - (g0 + first_stored))
    return None


def _diagnose_rows(stream: _Stream, data, cols, rp, ds, rows: list[int]) -> list[dict]:
    out = []
    mean_nnz = float(stream.ptr[-1] / max(len(stream.ptr) - 1, 1))
    for r in rows:
        g0, g1 = int(rp[r]), int(rp[r + 1])
        e0, e1 = int(stream.ptr[r]), int(stream.ptr[r + 1])
        sc, sv = np.asarray(cols[g0:g1]), np.asarray(data[g0:g1])
        src = stream.rows(r, r + 1)
        res = {"row": r, "obs_name": str(ds.obs.index[r]), "stored_entries": g1 - g0, "expected_entries": e1 - e0,
               "stored_start": g0, "expected_start": e0,
               "stored_nonzero": int((sv != 0).sum()), "expected_nonzero": int((src.data != 0).sum()),
               "source_row_sorted_indices": bool(np.all(np.diff(src.indices) > 0)),
               "source_row_explicit_zeros": int((src.data == 0).sum()),
               # Global-position tests assume the stream has exactly the indptr's entries.
               "stream_consistent": bool(src.nnz == e1 - e0)}
        oc, ov = stream.entries(g0, g1)
        res["identity_same_global_position"] = {"cols": _match(sc, oc), "values": _match(sv, ov)}
        res["identity_own_row"] = {"cols": _match(sc, src.indices), "values": _match(sv, src.data)}
        res["cols_width_views"] = {f"{o}->{w}": _match(sc, _as_read(stream, 0, o, w, g0, g1)) for o, w in COL_WIDTHS}
        res["data_width_views"] = {f"{o}->{w}": _match(sv, _as_read(stream, 1, o, w, g0, g1)) for o, w in DATA_WIDTHS}
        res["value_offset_in_stream"] = _find_offset(stream, sv, g0, int(5 * mean_nnz))
        # Read path: the dense row SingleCellDataset returns vs the raw arrays placed by column.
        dense = ds[r]["expressions"].numpy()[1:].astype(np.float64)
        raw = np.zeros_like(dense)
        ok_cols = (sc >= 0) & (sc < len(raw))
        raw[sc[ok_cols].astype(np.int64)] = sv[ok_cols]
        res["dataset_row_equals_raw_arrays"] = bool(np.array_equal(dense, raw))
        out.append(res)
    return out


def _diagnosis_line(ptr_test: dict, rows: list[dict]) -> str:
    def mean(key, sub):
        vals = [r[key][sub] for r in rows if np.isfinite(r[key][sub])]
        return float(np.mean(vals)) if vals else float("nan")

    parts = []
    if not ptr_test.get("available"):
        parts.append("row pointers: not testable (expected nnz unknown)")
    elif ptr_test["row_pointers_equal"]:
        parts.append("row pointers: correct")
    else:
        parts.append(f"row pointers: WRONG (equal {ptr_test['fraction_equal']:.2%}, "
                     f"x2 {ptr_test['fraction_twice']:.2%}, /2 {ptr_test['fraction_half']:.2%})")
    if not all(r["dataset_row_equals_raw_arrays"] for r in rows):
        parts.append("read path: get_row_padded differs from the raw arrays")
    if mean("identity_same_global_position", "cols") > 0.99 and mean("identity_same_global_position", "values") > 0.99:
        parts.append("memmap arrays hold the original entries")
        if ptr_test.get("available") and not ptr_test["row_pointers_equal"]:
            parts.append("=> rows are cut at the wrong boundaries")
        return "; ".join(parts)
    for key, label in (("cols_width_views", "column indices"), ("data_width_views", "values")):
        scores = {k: np.nanmean([r[key][k] for r in rows]) for k in rows[0][key]}
        best = max(scores, key=lambda k: -1 if np.isnan(scores[k]) else scores[k])
        same = best.split("->")[0] == best.split("->")[1]
        if scores[best] > 0.9:
            parts.append(f"{label}: {'intact' if same else 'written as ' + best.replace('->', ', read as ')} "
                         f"({scores[best]:.2%} match)")
        else:
            parts.append(f"{label}: no width view explains them (best {best} {scores[best]:.2%})")
    offsets = {r["value_offset_in_stream"] for r in rows}
    if offsets - {None, 0}:
        parts.append(f"stored values found in the stream at offsets {sorted(o for o in offsets if o is not None)}")
    return "; ".join(parts)


def _repro(reader: SourceReader, positions: np.ndarray, in_vocab: np.ndarray, names, vocab: dict,
           scratch: Path) -> dict:
    """Run the builder's exact conversion on ``positions`` with variants of the transient matrix."""
    import shutil

    import anndata as ad
    from bionemo.scdl.io.single_cell_collection import SingleCellCollection
    from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

    from bulk_sc_data_preprocessing import GENE_ID, _load_into_collection

    order = np.argsort(positions)
    X0 = reader.rows(positions[order])[:, in_vocab].tocsr()
    gene_names = names[in_vocab]
    ids = np.array([vocab[g] for g in gene_names])
    expected = X0.toarray().astype(np.float64)

    def canonical(X):
        X = X.copy()
        X.sum_duplicates()
        X.sort_indices()
        return X

    def cast(X, data, idx, ptr):
        X = X.copy()
        X.data, X.indices, X.indptr = X.data.astype(data), X.indices.astype(idx), X.indptr.astype(ptr)
        return X

    variants = {
        "as_read": (lambda X: X, True),
        "as_read_keep_features": (lambda X: X, False),
        "canonical": (canonical, True),
        "f32_i32": (lambda X: cast(canonical(X), np.float32, np.int32, np.int64), True),
        "f32_i64": (lambda X: cast(canonical(X), np.float32, np.int64, np.int64), True),
    }
    results = {"rows": int(len(positions)), "source_dtypes": {
        "data": str(X0.data.dtype), "indices": str(X0.indices.dtype), "indptr": str(X0.indptr.dtype)}}
    obs_index = pd.Index([str(i) for i in range(len(positions))])
    for name, (transform, drop_features) in variants.items():
        work = scratch / name
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        X = transform(X0)
        tmp = work / "transient.h5ad"
        ad.AnnData(X=X, obs=pd.DataFrame(index=obs_index), var=pd.DataFrame({GENE_ID: ids}, index=gene_names)).write_h5ad(tmp)
        try:
            collection = SingleCellCollection(str(work / "tmp"))
            _load_into_collection(collection, tmp)
            mm_path = work / "mem.map"
            collection.flatten(mm_path, destroy_on_copy=True)
            if drop_features and (mm_path / "features/dataframe_00.parquet").is_file():
                (mm_path / "features/dataframe_00.parquet").unlink()  # as build_pipeline_ready does
            mm = SingleCellMemMapDataset(str(mm_path))
            lookup = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
            lookup[ids] = np.arange(len(ids))
            bad, nnz_ratio = 0, []
            for i in range(len(positions)):
                exp_row, feats = mm.get_row_padded(i, return_features=True, feature_vars=[GENE_ID])
                exp_row = np.asarray(exp_row, dtype=np.float64)
                feat_ids = np.asarray(feats[0]).astype(np.int64)
                ok = (feat_ids >= 0) & (feat_ids < len(lookup))
                col = np.where(ok, lookup[np.clip(feat_ids, 0, len(lookup) - 1)], -1)
                want = np.where(col >= 0, expected[i, np.maximum(col, 0)], 0.0)
                if not np.array_equal(exp_row, want):
                    bad += 1
                nnz_ratio.append((exp_row != 0).sum() / max((want != 0).sum(), 1))
            _, arrays = _memmap_arrays(mm)
            results[name] = {"status": "PASS" if bad == 0 else "FAIL", "rows_wrong": bad,
                             "median_stored_over_expected_nonzero": float(np.median(nnz_ratio)),
                             "transient_dtypes": {"data": str(X.data.dtype), "indices": str(X.indices.dtype),
                                                  "indptr": str(X.indptr.dtype)},
                             "memmap_arrays": arrays}
        except Exception as exc:
            results[name] = {"status": "ERROR", "error": f"{exc.__class__.__name__}: {exc}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        print(f"    repro {name:24s} {results[name].get('status')} "
              f"{results[name].get('rows_wrong', '')} {results[name].get('error', '')}")
    return results


def check_corruption(report: Report, ds, vocab: dict, source: Path, store: Path, gene_column: str | None,
                     n_rows: int, repro_rows: int, scratch: Path | None, rng: np.random.Generator) -> None:
    import h5py

    memmap = ds.memmap
    with SourceReader(source) as reader:
        inventory = _inventory(store, memmap, reader)
        print("\n  inventory:")
        print(json.dumps({k: v for k, v in inventory.items() if k != "json"}, indent=2, default=str))
        found, _ = _memmap_arrays(memmap)
        if set(found) != {"data", "cols", "rows"}:
            report.add("1b. corruption mechanism", False, note="could not identify the memmap arrays",
                       inventory=inventory)
            return
        data, cols, rp = found["data"][1], found["cols"][1], found["rows"][1]

        in_vocab, names = builder_genes(reader, vocab, gene_column)
        positions = reader.obs.index.astype(str).get_indexer(ds.obs.index.astype(str))
        if (positions < 0).any():
            report.add("1b. corruption mechanism", False, note="store rows missing from the source",
                       missing=int((positions < 0).sum()))
            return
        monotonic = bool(np.all(np.diff(positions) > 0))

        # Global pointer test from the source indptr (exact when every gene is in the vocabulary).
        X = reader._file["X"]
        ptr_test: dict = {"available": False}
        ptr = None
        if isinstance(X, h5py.Group) and str(X.attrs.get("encoding-type")) == "csr_matrix" and in_vocab.all():
            nnz = np.diff(X["indptr"][:])
            ptr = np.concatenate([[0], np.cumsum(nnz[positions])]).astype(np.int64)
            rp_arr = np.asarray(rp, dtype=np.int64)
            n = min(len(rp_arr), len(ptr))
            ptr_test = {"available": True, "stored_rows": int(len(rp_arr) - 1), "expected_rows": int(len(ptr) - 1),
                        "stored_total_entries": int(len(data)), "stored_last_pointer": int(rp_arr[-1]),
                        "expected_total_entries": int(ptr[-1]),
                        "fraction_equal": float(np.mean(rp_arr[:n] == ptr[:n])),
                        "fraction_twice": float(np.mean(rp_arr[1:n] == 2 * ptr[1:n])),
                        "fraction_half": float(np.mean(2 * rp_arr[1:n] == ptr[1:n])),
                        "first_mismatch_row": (int(np.flatnonzero(rp_arr[:n] != ptr[:n])[0])
                                               if (rp_arr[:n] != ptr[:n]).any() else None)}
            ptr_test["row_pointers_equal"] = ptr_test["fraction_equal"] == 1.0 and len(rp_arr) == len(ptr)
        else:
            ptr_test["reason"] = "source X is not CSR or not every gene is in the vocabulary"
        print("\n  row pointers:", json.dumps(ptr_test, default=str))

        rows_out: list[dict] = []
        if ptr is not None:
            stream = _Stream(reader, positions, in_vocab, ptr)
            picks = sorted({0, 1, *rng.choice(len(ds), size=max(n_rows - 2, 0), replace=False).tolist()})
            rows_out = _diagnose_rows(stream, data, cols, rp, ds, picks)
            for r in rows_out:
                print(f"\n  row {r['row']} ({r['obs_name']}): stored {r['stored_entries']} entries "
                      f"({r['stored_nonzero']} non-zero) vs expected {r['expected_entries']} ({r['expected_nonzero']})")
                print(f"    identity same position: {r['identity_same_global_position']}, own row: {r['identity_own_row']}")
                print(f"    best col view: {max(r['cols_width_views'].items(), key=lambda kv: np.nan_to_num(kv[1]))}, "
                      f"best data view: {max(r['data_width_views'].items(), key=lambda kv: np.nan_to_num(kv[1]))}")
                print(f"    value offset in stream: {r['value_offset_in_stream']}, "
                      f"dataset row == raw arrays: {r['dataset_row_equals_raw_arrays']}")
        repro = None
        if repro_rows > 0 and scratch is not None:
            print(f"\n  reproducing the conversion on {repro_rows} rows under {scratch}")
            take = positions[:repro_rows]
            repro = _repro(reader, take, in_vocab, names, vocab, scratch)

    line = _diagnosis_line(ptr_test, rows_out) if rows_out else "per-row analysis unavailable"
    rows_corrupt, broken = corruption_status(rows_out, ptr_test)
    repro_reproduced = bool(repro) and repro.get("as_read", {}).get("status") == "FAIL"
    if repro:
        fixes = [k for k in ("canonical", "f32_i32", "f32_i64", "as_read_keep_features")
                 if repro.get(k, {}).get("status") == "PASS"]
        if repro_reproduced:
            line += f"; repro: as_read FAILS, clean variants: {fixes or 'none'}"
        elif repro.get("as_read", {}).get("status") == "PASS":
            line += ("; store corrupt, but the repro did not reproduce it at this size (paging-dependent)"
                     if rows_corrupt else "; repro: as_read is clean")
    ok = None if not rows_out else rows_corrupt == 0
    report.add("1b. corruption mechanism", ok, diagnosis=line, rows_corrupt=rows_corrupt,
               broken_arrays=broken, repro_reproduced=repro_reproduced, rows_in_source_order=monotonic,
               inventory=inventory, row_pointers=ptr_test, rows=rows_out, repro=repro)


def corruption_status(rows_out: list[dict], ptr_test: dict) -> tuple[int, list[str]]:
    """``(corrupt rows, broken arrays)``: a row is corrupt when its columns, values, entry count
    or the dataset's read of it differ from the source."""
    broken = set()
    if ptr_test.get("available") and not ptr_test.get("row_pointers_equal"):
        broken.add("row_index")
    corrupt = 0
    for r in rows_out:
        bad = False
        if r["identity_own_row"]["cols"] < 1.0:
            broken.add("col_index")
            bad = True
        if r["identity_own_row"]["values"] < 1.0:
            broken.add("data")
            bad = True
        if r["stored_entries"] != r["expected_entries"]:
            broken.add("row_index")
            bad = True
        if not r["dataset_row_equals_raw_arrays"]:
            broken.add("read_path")
            bad = True
        corrupt += bad
    order = ["col_index", "data", "row_index", "read_path"]
    return corrupt, [a for a in order if a in broken]


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
    diag = p.add_argument_group("1b. corruption mechanism")
    diag.add_argument("--only-diagnose", action="store_true",
                      help="Run only check 1b (writes corruption.json instead of report.json)")
    diag.add_argument("--diag-rows", type=int, default=5, help="Rows analysed entry by entry (rows 0 and 1 always)")
    diag.add_argument("--repro-rows", type=int, default=2000,
                      help="Rows converted again through SCDL with variants of the transient matrix (0 = skip)")
    diag.add_argument("--scratch", type=Path, default=None,
                      help="Directory for the reproduction (default: <out-dir>/_repro, deleted afterwards)")
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

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch or out_dir / "_repro"
    if not args.only_diagnose:
        check_sources(report, source, eval_source, test_names, args.n_rows, rng)
        check_store_rows(report, ds, vocab, source, args.gene_column, args.n_rows, rng)
    check_corruption(report, ds, vocab, source, store, args.gene_column, args.diag_rows,
                     args.repro_rows, scratch, rng)
    if not args.only_diagnose:
        check_eval_ids(report, vocab, eval_source, args.gene_column, plan_path)
        if test_names:
            check_profile(report, ds, vocab, eval_source, test_names, args.profile_rows, args.n_bins,
                          args.context_len, rng)
        else:
            print("\n[SKIP] 3. data profile: no test row names")

    payload = {"store": str(store), "source": str(source), "eval_source": str(eval_source),
               "plan": str(plan_path) if plan_path else None, "gene_column": args.gene_column,
               "seed": args.seed, "checks": report.checks}
    out = out_dir / ("corruption.json" if args.only_diagnose else "report.json")
    out.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    print(f"\n{'FAIL' if report.failed else 'PASS'}: wrote {out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
