"""Check that a training store has what each loss needs, before a run finds out.

    python data_preprocess/check_store_losses.py --store <dir>
    python data_preprocess/check_store_losses.py --mix sc_bulk_pb_leiden_s0 [--root ROOT]
    python data_preprocess/check_store_losses.py --pipeline-ready bulk:leiden_s0 --losses reconstruction
    python data_preprocess/check_store_losses.py --mix NAME --losses contrastive cdd --pb-source precomputed \\
        --class-column tissue_general --class-min-count 1 --batch-size 64 --bulk-ratio 0.25 --pb-ratio 0.25

Reads only ``obs.parquet``, ``mapping.json``, ``vocab.json`` and ``store.json`` (pandas +
pyarrow; no torch, no SCDL), and mirrors the checks ``BulkSCDataset``, ``BulkSCSampler``
and the loss blocks make at train time:

- reconstruction: the store files, the vocabulary, obs rows vs the store's row count, and
  the ``--conditions`` columns.
- unified mode (every loss below): the modality column with ``bulk`` and ``sc`` rows.
- the pseudobulk source, per mode (``--pb-source``): on the fly from single cells (tissue
  pools of ``--pb-group-column`` with at least ``--n-sc-per-pseudobulk`` cells), or
  precomputed rows (``--pb-label``).
- mmd, dat: a pseudobulk source; dat also its ``--dat-columns``.
- contrastive, cdd: the class column, and classes (not excluded) present in bulk and in
  the source with at least ``--class-min-count`` rows; how many bulk rows that covers.
- class_aware: ``--pb-group-column``, the shared tissues, and the expected rows per tissue
  in a batch against ``--class-min-count``.
- aggregation: on the fly, cells per tissue pool; precomputed, ``--pb-id-column`` linking
  pseudobulk rows to the cells in the store.
- paired: precomputed pseudobulks, the ``--paired-column`` (category ``"0"`` = unpaired must
  be code 0), and pair ids shared by pseudobulk and bulk rows.

Per loss and source mode the summary says READY, READY (warnings) or NOT READY. The exit
code is 1 when a requested loss is NOT READY in every source mode checked. Counts are over
the whole store; training splits off a validation part at random, so pools shrink a bit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LOSSES = ("reconstruction", "mmd", "dat", "contrastive", "cdd", "class_aware", "aggregation", "paired")
# Losses whose batches need a pseudobulk source (checked per source mode).
NEEDS_SOURCE = {"mmd", "dat", "contrastive", "cdd", "class_aware", "aggregation"}
MODES = ("on-the-fly", "precomputed")
BULK_LABEL, SC_LABEL = "bulk", "sc"      # what BulkSCDataset looks up in mapping.json

OK, WARN, FAIL = "OK", "WARN", "FAIL"


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    """The checks of one topic (a loss, possibly in one source mode)."""

    title: str
    lines: list = field(default_factory=list)   # (status, message)

    def add(self, status: str, message: str) -> None:
        self.lines.append((status, message))

    def ok(self, message: str) -> None:
        self.add(OK, message)

    def warn(self, message: str) -> None:
        self.add(WARN, message)

    def fail(self, message: str) -> None:
        self.add(FAIL, message)

    def info(self, message: str) -> None:
        self.add("", message)

    def extend(self, other: "Report") -> None:
        self.lines.extend(other.lines)

    @property
    def status(self) -> str:
        found = {s for s, _ in self.lines}
        return FAIL if FAIL in found else WARN if WARN in found else OK

    def print(self) -> None:
        print(f"\n== {self.title}")
        for status, message in self.lines:
            tag = f"[{status}]" if status else "      "
            print(f"  {tag:<6} {message}")


VERDICT = {OK: "READY", WARN: "READY (warnings)", FAIL: "NOT READY"}


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #


class Store:
    """obs codes, the category mapping and the row pools the loaders build."""

    def __init__(self, path: Path, modality_column: str, pb_label: str):
        self.path = path
        self.obs = pd.read_parquet(path / "obs.parquet")
        self.mapping = json.loads((path / "mapping.json").read_text(encoding="utf-8"))
        self.modality_column = modality_column
        self.pb_label = pb_label
        self.bulk = self.sc = self.pb = np.empty(0, dtype=np.int64)
        mod_map = self.mapping.get(modality_column, {})
        if modality_column in self.obs.columns:
            codes = self.obs[modality_column].to_numpy()
            pick = lambda label: (np.flatnonzero(codes == mod_map[label])  # noqa: E731
                                  if label in mod_map else np.empty(0, dtype=np.int64))
            self.bulk, self.sc, self.pb = pick(BULK_LABEL), pick(SC_LABEL), pick(pb_label)

    def has(self, column: str | None) -> bool:
        return column is not None and column in self.obs.columns

    def codes(self, column: str) -> np.ndarray:
        return self.obs[column].to_numpy()

    def names(self, column: str) -> dict[int, str]:
        return {int(code): name for name, code in self.mapping.get(column, {}).items()}

    def code_of(self, column: str, name: str):
        return self.mapping.get(column, {}).get(name)

    def counts(self, column: str, rows: np.ndarray) -> dict[int, int]:
        values, n = np.unique(self.codes(column)[rows], return_counts=True)
        return {int(v): int(c) for v, c in zip(values, n)}

    def label_list(self, column: str, codes, limit: int = 8) -> str:
        names = self.names(column)
        shown = [names.get(int(c), str(c)) for c in list(codes)[:limit]]
        more = len(codes) - limit
        return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_store_files(path: Path) -> Report:
    rep = Report("store")
    missing = [n for n in ("vocab.json", "mapping.json", "obs.parquet") if not (path / n).is_file()]
    if not (path / "mem.map").exists():
        missing.append("mem.map/")
    if missing:
        rep.fail(f"missing in {path}: {', '.join(missing)}")
        return rep
    rep.ok(f"vocab.json, mapping.json, obs.parquet and mem.map/ present in {path}")
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    rep.info(f"vocabulary: {len(vocab)} tokens")
    for token in ("<cls>", "<pad>"):
        if token not in vocab:
            rep.fail(f"vocabulary has no {token} token (the loaders insert it)")
    meta_path = path / "store.json"
    import pyarrow.parquet as pq

    obs_rows = pq.ParquetFile(path / "obs.parquet").metadata.num_rows
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_rows = meta.get("n_rows")
        if n_rows is not None and n_rows != obs_rows:
            rep.fail(f"obs.parquet has {obs_rows} rows, store.json says {n_rows}")
        else:
            rep.ok(f"{obs_rows} rows (matches store.json)")
        status = (meta.get("verified") or {}).get("status")
        if status in (None, "SKIPPED"):
            rep.warn("store.json: the store was not verified against its sources "
                     "(data_preprocess/check_store_alignment.py checks it)")
        elif status != "PASS":
            rep.fail(f"store.json: verification status {status!r}")
        else:
            rep.ok(f"store.json: verified ({status})")
    else:
        rep.warn(f"no store.json; {obs_rows} rows (cannot compare with the store's own count)")
    return rep


def check_reconstruction(store: Store, conditions: list[str]) -> Report:
    rep = Report("reconstruction")
    rep.info(f"obs columns: {', '.join(store.obs.columns)}")
    for col in conditions:
        if not store.has(col):
            rep.fail(f"--conditions column '{col}' is not in obs")
        elif col not in store.mapping:
            rep.fail(f"--conditions column '{col}' has no entry in mapping.json")
        else:
            rep.ok(f"condition '{col}': {len(store.mapping[col])} categories")
    if not conditions:
        rep.ok("no --conditions to check")
    return rep


def check_unified(store: Store) -> Report:
    rep = Report("unified mode (--unified)")
    col = store.modality_column
    if not store.has(col):
        rep.fail(f"no modality column '{col}' in obs")
        return rep
    labels = store.mapping.get(col, {})
    rep.info(f"'{col}' categories: {labels}")
    for label, rows in ((BULK_LABEL, store.bulk), (SC_LABEL, store.sc)):
        if label not in labels:
            rep.fail(f"modality label '{label}' missing from mapping.json (BulkSCDataset requires it)")
        elif len(rows) == 0:
            rep.fail(f"no '{label}' rows")
        else:
            rep.ok(f"{len(rows)} '{label}' rows")
    if store.pb_label in labels:
        rep.ok(f"{len(store.pb)} precomputed '{store.pb_label}' rows")
    else:
        rep.info(f"no '{store.pb_label}' rows (precomputed pseudobulks unavailable)")
    return rep


def source_pools(store: Store, args, mode: str) -> tuple[Report, dict | None]:
    """The pseudobulk source of ``mode`` as ``{group code: n rows}`` (``{-1: n}`` ungrouped)."""
    rep = Report(f"pseudobulk source ({mode})")
    group = args.pb_group_column
    if mode == "precomputed":
        if len(store.pb) == 0:
            rep.fail(f"no precomputed pseudobulk rows (modality '{store.pb_label}'; --pb-label)")
            return rep, None
        rows = store.pb
        rep.ok(f"{len(rows)} precomputed pseudobulk rows")
    else:
        if len(store.sc) < args.n_sc_per_pseudobulk:
            rep.fail(f"{len(store.sc)} single cells, fewer than --n-sc-per-pseudobulk "
                     f"{args.n_sc_per_pseudobulk}")
            return rep, None
        rows = store.sc
        rep.ok(f"{len(rows)} single cells to aggregate, {args.n_sc_per_pseudobulk} per pseudobulk")
    if not group:
        rep.info("no --pb-group-column: pseudobulks are not tissue-pure")
        return rep, {-1: len(rows)}
    if not store.has(group):
        rep.fail(f"--pb-group-column '{group}' is not in obs")
        return rep, None
    pools = store.counts(group, rows)
    if mode == "on-the-fly":
        small = {g: n for g, n in pools.items() if n < args.n_sc_per_pseudobulk}
        if small:
            rep.warn(f"{len(small)} of {len(pools)} '{group}' pools have fewer cells than "
                     f"--n-sc-per-pseudobulk (drawn with replacement): "
                     f"{store.label_list(group, sorted(small))}")
    rep.ok(f"{len(pools)} '{group}' groups in the source")
    return rep, pools


def shared_classes(store: Store, args, pools: dict, rep: Report, column: str) -> list[int]:
    """Classes in bulk and in the source, not excluded, with >= class_min_count rows each."""
    exclude = {store.code_of(column, name) for name in args.class_exclude_labels} - {None}
    unknown_names = [n for n in args.class_exclude_labels if store.code_of(column, n) is None]
    if unknown_names:
        rep.info(f"--class-exclude-labels not in '{column}': {', '.join(unknown_names)}")
    bulk = store.counts(column, store.bulk)
    m = max(1, args.class_min_count)
    shared = sorted(c for c in pools if c not in exclude and c >= 0
                    and pools[c] >= m and bulk.get(c, 0) >= m)
    covered = sum(bulk[c] for c in shared)
    only_bulk = sorted(c for c in bulk if c not in pools and c not in exclude)
    excluded_bulk = sum(n for c, n in bulk.items() if c in exclude)
    if shared:
        rep.ok(f"{len(shared)} class(es) shared by bulk and source (>= {m} rows each): "
               f"{store.label_list(column, shared)}")
    else:
        rep.fail(f"no class is present in both bulk and source with >= {m} rows")
    rep.info(f"bulk rows in shared classes: {covered}/{len(store.bulk)} "
             f"({covered / max(1, len(store.bulk)):.0%}); excluded labels: {excluded_bulk}; "
             f"bulk-only classes: {len(only_bulk)}"
             + (f" ({store.label_list(column, only_bulk, 5)})" if only_bulk else ""))
    if shared and covered / max(1, len(store.bulk)) < 0.5:
        rep.warn("less than half of the bulk rows can get a class-conditional signal")
    return shared


def check_class_loss(name: str, store: Store, args, mode: str, pools) -> Report:
    rep = Report(f"{name} ({mode})")
    column = args.class_column
    if pools is None:
        rep.fail("no pseudobulk source in this mode (see above)")
        return rep
    if not store.has(column):
        rep.fail(f"--class-column '{column}' is not in obs")
        return rep
    rep.info(f"train with '{column}' in --conditions (label only: leave it out of --condition-encoders)")
    if args.pb_group_column != column:
        # the source pools are keyed by the group column; re-key them by the class column
        rows = store.pb if mode == "precomputed" else store.sc
        pools = store.counts(column, rows)
        if args.pb_group_column:
            rep.warn(f"--pb-group-column '{args.pb_group_column}' differs from --class-column "
                     f"'{column}': pseudobulks may mix classes, and --class-aware matches groups, not classes")
        else:
            rep.warn("without --pb-group-column, pseudobulks are not class-pure: their cells "
                     "may come from several classes")
    shared_classes(store, args, pools, rep, column)
    if name == "cdd" and "class_aware" not in args.losses:
        rep.info("CDD needs matched classes in each batch: use --class-aware")
    if name == "contrastive":
        rep.info("contrastive needs same-class partners in each batch: use --class-aware")
    return rep


def check_class_aware(store: Store, args, mode: str, pools) -> Report:
    rep = Report(f"class_aware sampling ({mode})")
    group = args.pb_group_column
    if not group:
        rep.fail("--class-aware requires --pb-group-column")
        return rep
    if pools is None:
        rep.fail("no pseudobulk source in this mode (see above)")
        return rep
    shared = shared_classes(store, args, pools, rep, group)
    if not shared:
        return rep
    # Rows per tissue in one class-aware batch, laid out as the sampler does.
    from utils.sampling import class_aware_allocation

    n_bulk = round(args.batch_size * args.bulk_ratio)
    n_pb = round(args.batch_size * args.pb_ratio)
    plan = class_aware_allocation(n_pb, n_bulk, len(shared), args.class_aware_n_classes,
                                  args.class_min_count, args.class_aware_pb_frac,
                                  args.class_aware_bulk_frac)
    k = plan["k"]
    lo_p, hi_p = plan["pb_per_tissue"]
    lo_b, hi_b = plan["bulk_per_tissue"]
    rep.info(f"batch {args.batch_size}: {n_bulk} bulk ({plan['n_bulk_matched']} matched, "
             f"{plan['n_bulk_free']} free), {n_pb} pseudobulk ({plan['n_pb_matched']} matched, "
             f"{plan['n_pb_free']} free) over K={k} tissues -> {lo_p}-{hi_p} pseudobulk and "
             f"{lo_b}-{hi_b} bulk per tissue")
    if k < plan["k_requested"]:
        rep.info(f"K lowered from {plan['k_requested']} to {k} so every tissue has "
                 f">= --class-min-count {args.class_min_count} rows in both domains")
    # Rows outside the shared classes train reconstruction only, through the free slots.
    n_src = sum(pools.values())
    src_out = n_src - sum(pools[c] for c in shared)
    if src_out:
        where = (f"drawn by the {plan['n_pb_free']} free pseudobulk slots per batch"
                 if plan["n_pb_free"] else
                 "never drawn as pseudobulk in a class-aware batch (--class-aware-pb-frac 1.0)")
        rep.info(f"source rows outside the shared classes (reconstruction only): {src_out}/{n_src} "
                 f"({src_out / max(1, n_src):.0%}); {where}")
        if not plan["n_pb_free"] and mode == "precomputed":
            rep.warn("these pseudobulks are drawn only by paired batches: set "
                     "--class-aware-pb-frac < 1 (e.g. 0.75) so they still train reconstruction")
    if n_pb == 0:
        rep.fail("--pb-ratio gives 0 pseudobulks per batch")
    elif plan["n_pb_matched"] and min(lo_p, lo_b if plan["n_bulk_matched"] else lo_p) < max(1, args.class_min_count):
        rep.warn(f"below --class-min-count {args.class_min_count}: some tissues will not count in "
                 f"cdd/contrastive (raise the matched fractions or lower --class-min-count)")
    else:
        rep.ok("enough rows per tissue for --class-min-count")
    return rep


def check_simple(name: str, store: Store, args, mode: str, pools) -> Report:
    rep = Report(f"{name} ({mode})")
    if pools is None:
        rep.fail("no pseudobulk source in this mode (see above)")
        return rep
    n_bulk = round(args.batch_size * args.bulk_ratio)
    n_pb = round(args.batch_size * args.pb_ratio)
    if n_pb == 0:
        rep.fail(f"--pb-ratio {args.pb_ratio} gives no pseudobulk in a batch of {args.batch_size}")
    else:
        rep.ok(f"{n_bulk} bulk and {n_pb} pseudobulk rows per batch of {args.batch_size}")
    if name == "dat":
        for col in args.dat_columns:
            if not store.has(col):
                rep.fail(f"--dat-columns '{col}' is not in obs")
                continue
            if col == store.modality_column:
                rep.ok("modality discriminator: bulk vs pseudobulk")
                continue
            n = len(store.counts(col, np.concatenate([store.bulk, store.pb, store.sc])))
            (rep.ok if n >= 2 else rep.fail)(f"'{col}': {n} categories (must also be in --conditions)")
    return rep


def check_aggregation(store: Store, args, mode: str, pools) -> Report:
    rep = Report(f"aggregation ({mode})")
    if pools is None:
        rep.fail("no pseudobulk source in this mode (see above)")
        return rep
    if mode == "on-the-fly":
        rep.ok("pseudobulks are built from cells in the batch, so their constituents are always known")
        return rep
    col = args.pb_id_column
    if not store.has(col):
        rep.fail(f"--pb-id-column '{col}' is not in obs: precomputed pseudobulks cannot be "
                 f"linked to their cells")
        return rep
    names = store.mapping.get(col, {})
    fill = names.get("0")
    others = [n for n in names if n != "0"]
    if fill is not None and others and all(n.lstrip("-").isdigit() for n in others):
        rep.warn(f"'{col}' holds plain integer ids, so category '0' is ambiguous (fill and a real id)")
    elif others and not any(n.startswith("pb:") for n in others):
        rep.info(f"'{col}' ids are not prefixed 'pb:' (build_mix --pb-id-column writes them that way)")
    ids = store.codes(col)
    sc_ids = ids[store.sc]
    sc_ids = sc_ids[sc_ids != fill] if fill is not None else sc_ids
    cells = pd.Series(sc_ids).value_counts()
    pb_ids = ids[store.pb]
    with_cells = np.isin(pb_ids, cells.index.to_numpy())
    n_with = int(with_cells.sum())
    msg = f"{n_with}/{len(store.pb)} precomputed pseudobulks have constituent cells in the store"
    if n_with == 0:
        rep.fail(msg + f" (check that '{col}' is set on both pseudobulk and cell rows)")
        return rep
    (rep.ok if n_with == len(store.pb) else rep.warn)(msg)
    linked = cells[cells.index.isin(pb_ids)]
    few = int((linked < args.n_sc_per_pseudobulk).sum())
    rep.info(f"cells per linked pseudobulk: median {int(linked.median())}, min {int(linked.min())}"
             + (f"; {few} with fewer than --n-sc-per-pseudobulk {args.n_sc_per_pseudobulk}" if few else ""))
    if args.pb_group_column and store.has(args.pb_group_column):
        groups = store.codes(args.pb_group_column)[store.pb]
        lost = set(np.unique(groups)) - set(np.unique(groups[with_cells]))
        if lost:
            rep.warn(f"{len(lost)} '{args.pb_group_column}' groups lose every pseudobulk with "
                     f"aggregation on: {store.label_list(args.pb_group_column, sorted(lost))}")
    return rep


def check_paired(store: Store, args) -> Report:
    rep = Report("paired")
    col = args.paired_column
    if len(store.pb) == 0:
        rep.fail(f"no precomputed pseudobulk rows ('{store.pb_label}'): paired batches pair them with bulk")
        return rep
    if not store.has(col):
        rep.fail(f"--paired-column '{col}' is not in obs")
        return rep
    zero = store.code_of(col, "0")
    if zero != 0:
        rep.fail(f"'{col}' category '0' (unpaired) has code {zero}, but the sampler treats code 0 "
                 f"as unpaired")
        return rep
    codes = store.codes(col)
    pb_ids, bulk_ids = codes[store.pb], codes[store.bulk]
    pb_set, bulk_set = set(pb_ids[pb_ids != 0].tolist()), set(bulk_ids[bulk_ids != 0].tolist())
    common = pb_set & bulk_set
    rep.info(f"pair ids: {len(pb_set)} on pseudobulk rows, {len(bulk_set)} on bulk rows")
    if not common:
        rep.fail("no pair id is shared by a pseudobulk and a bulk row")
        return rep
    rep.ok(f"{len(common)} pseudobulk-bulk pairs")
    for label, ids in (("pseudobulk", pb_ids), ("bulk", bulk_ids)):
        dup = pd.Series(ids[np.isin(ids, list(common))]).value_counts()
        if (dup > 1).any():
            rep.warn(f"{int((dup > 1).sum())} pair ids have several {label} rows (only the last is used)")
    sc_ids = codes[store.sc]
    sc_pairs = set(sc_ids[sc_ids != 0].tolist()) & common
    if sc_pairs:
        rep.ok(f"{len(sc_pairs)} pairs also have matched single cells (used in the paired SC term)")
    else:
        rep.info("no single cell carries a pair id: the paired loss uses bulk-pseudobulk only")
    if round(args.batch_size * args.bulk_ratio) != round(args.batch_size * args.pb_ratio):
        rep.fail(f"--paired-sampling needs --bulk-ratio == --pb-ratio (got {args.bulk_ratio}, {args.pb_ratio})")
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check that a training store supports the chosen losses.")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--store", type=Path, help="Store directory (obs.parquet, mapping.json, ...)")
    where.add_argument("--mix", help="Multi-modality store name: <root>/data/pipeline_ready/<mix>")
    where.add_argument("--pipeline-ready", metavar="MODALITY:SPLIT", help="Single-modality store")
    p.add_argument("--root", type=Path, default=None, help="Experiments root (with --mix / --pipeline-ready)")
    p.add_argument("--losses", nargs="+", choices=LOSSES, default=list(LOSSES),
                   help="Losses to check (default: all)")
    p.add_argument("--pb-source", choices=("both",) + MODES, default="both",
                   help="Pseudobulk source to check the losses under (default: both)")
    # the training options the checks depend on, with the training defaults
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--modality-column", default="modality")
    p.add_argument("--pb-label", default="pseudobulk")
    p.add_argument("--pb-group-column", default="tissue_general",
                   help="Tissue column for tissue-pure pseudobulks / class-aware sampling ('' = none)")
    p.add_argument("--pb-id-column", default="pseudobulk_id")
    p.add_argument("--paired-column", default="paired")
    p.add_argument("--class-column", default="tissue_general")
    p.add_argument("--class-exclude-labels", nargs="+", default=["unknown"])
    p.add_argument("--class-min-count", type=int, default=2)
    p.add_argument("--class-aware-n-classes", type=int, default=8)
    p.add_argument("--class-aware-bulk-frac", type=float, default=0.6)
    p.add_argument("--class-aware-pb-frac", type=float, default=0.6)
    p.add_argument("--dat-columns", nargs="+", default=["modality"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bulk-ratio", type=float, default=0.3)
    p.add_argument("--pb-ratio", type=float, default=0.3)
    p.add_argument("--n-sc-per-pseudobulk", type=int, default=10)
    return p


def resolve_store(args) -> Path:
    if args.store is not None:
        return args.store
    from experiments.layout import ExperimentLayout

    layout = ExperimentLayout.from_root(args.root)
    if args.mix:
        return layout.mix_dir(args.mix)
    modality, _, split = args.pipeline_ready.partition(":")
    if not split:
        sys.exit("Error: --pipeline-ready takes MODALITY:SPLIT")
    return layout.pipeline_ready(modality, split)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.pb_group_column = args.pb_group_column or None
    path = resolve_store(args)
    modes = list(MODES) if args.pb_source == "both" else [args.pb_source]
    print(f"Store: {path}")
    print(f"Losses: {', '.join(args.losses)} | pseudobulk source: {', '.join(modes)}")

    files = check_store_files(path)
    files.print()
    if files.status == FAIL:
        print("\nNOT READY: the store itself is incomplete")
        return 1
    store = Store(path, args.modality_column, args.pb_label)

    verdicts: dict[str, dict[str, str]] = {}
    if "reconstruction" in args.losses:
        rep = check_reconstruction(store, args.conditions)
        rep.print()
        verdicts["reconstruction"] = {"-": rep.status}

    aux = [loss for loss in args.losses if loss != "reconstruction"]
    if aux:
        unified = check_unified(store)
        unified.print()
        pools = {}
        if unified.status != FAIL:
            for mode in modes if NEEDS_SOURCE & set(aux) else []:
                rep, pools[mode] = source_pools(store, args, mode)
                rep.print()
        for loss in aux:
            verdicts[loss] = {}
            loss_modes = ["-"] if loss == "paired" else modes
            for mode in loss_modes:
                if unified.status == FAIL:
                    rep = Report(f"{loss} ({mode})")
                    rep.fail("unified mode is not possible (see above)")
                elif loss == "paired":
                    rep = check_paired(store, args)
                elif loss in ("contrastive", "cdd"):
                    rep = check_class_loss(loss, store, args, mode, pools.get(mode))
                elif loss == "class_aware":
                    rep = check_class_aware(store, args, mode, pools.get(mode))
                elif loss == "aggregation":
                    rep = check_aggregation(store, args, mode, pools.get(mode))
                else:
                    rep = check_simple(loss, store, args, mode, pools.get(mode))
                rep.print()
                verdicts[loss][mode] = rep.status

    print("\n== summary")
    width = max(len(loss) for loss in verdicts)
    failed = []
    for loss, by_mode in verdicts.items():
        cells = "  ".join(f"{mode}: {VERDICT[s]}" if mode != "-" else VERDICT[s] for mode, s in by_mode.items())
        print(f"  {loss:<{width}}  {cells}")
        if all(s == FAIL for s in by_mode.values()):
            failed.append(loss)
    if failed:
        print(f"\nNot ready in any source mode: {', '.join(failed)}")
        return 1
    print("\nEvery requested loss is ready in at least one source mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
