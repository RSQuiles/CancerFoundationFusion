"""Recover the pseudo-bulk -> source single-cell association.

`create_pseudo_bulk_data_RAW.py` never records which cells were summed into which
pseudo-bulk: the selection happens in-memory in `generate_pseudo_bulk_chunks*()` and is
discarded after the sum. It is however driven by a seeded RNG, so it can be replayed
exactly.

This script replays that RNG stream to rebuild the mapping, and can verify it by
re-summing the recovered cells and comparing against the pseudo-bulk matrix.

Layout it relies on (both mirror `create_pseudo_bulk_data_RAW.py`):

  * pool row order == row order of `sampled_source_cells.csv`
  * `source_cells_chunk_{k:05d}.h5ad` row i  ==  pool row k * DOWNLOAD_CHUNK_SIZE + i

Examples
--------
Build the map only (needs the two CSVs, no h5ad, no chunk files)::

    python data/reconstruct_pseudobulk_cell_map.py --pseudo-bulk-dir data/pseudo_bulk

Build and verify 20 random pseudo-bulks against the generated matrix::

    python data/reconstruct_pseudobulk_cell_map.py \
        --pseudo-bulk-dir data/pseudo_bulk \
        --pseudo-bulk-h5ad /path/to/pseudo_bulk_RAW.h5ad \
        --verify 20
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Defaults mirror create_pseudo_bulk_data_RAW.py. Changing any of these changes the RNG
# stream and therefore the recovered mapping.
RANDOM_SEED = 2021
WRITE_CHUNK_SIZE = 500
DOWNLOAD_CHUNK_SIZE = 5000
TISSUE_COLUMN = "tissue_general"
CHUNK_NAME_RE = re.compile(r"source_cells_chunk_(\d+)\.h5ad$")


# =========================
# Loading
# =========================

def normalize_context_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mirror `normalize_obs_chunk` for string columns (str, NaN -> 'unknown').

    Both sides of the pool join must be normalized identically or the join silently
    produces empty pools.
    """
    for col in columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.cat.rename_categories(
                series.cat.categories.astype(str)
            )
            if series.isna().any():
                if "unknown" not in series.cat.categories:
                    series = series.cat.add_categories(["unknown"])
                series = series.fillna("unknown")
        else:
            series = series.astype("string").fillna("unknown")
            series = series.replace("<NA>", "unknown").astype(str)
        df[col] = series
    return df


def load_plan(path: Path, tissue_column: str) -> list[dict]:
    """Load the sampling plan, preserving row order and cell-type key order.

    `cell_type_counts` was written with `json.dumps(..., sort_keys=True)`; `json.loads`
    reproduces that order via dict insertion order. The RNG consumes cell types in that
    order, so it must not be re-sorted.
    """
    plan_df = pd.read_csv(path)
    context_cols = ["sample_id", "dataset_id", "donor_id", tissue_column]
    plan_df = normalize_context_columns(plan_df, context_cols)

    plan_rows = []
    for row in plan_df.itertuples(index=False):
        plan_rows.append(
            {
                "sample_id": row.sample_id,
                "context": (
                    row.dataset_id,
                    row.donor_id,
                    getattr(row, tissue_column),
                ),
                "cell_type_counts": json.loads(row.cell_type_counts),
            }
        )
    return plan_rows


def load_source_meta(path: Path, tissue_column: str) -> pd.DataFrame:
    """Load `sampled_source_cells.csv`. Row order defines the pool row indices."""
    context_cols = ["dataset_id", "donor_id", tissue_column, "cell_type"]
    meta = pd.read_csv(
        path,
        dtype={
            "soma_joinid": np.int64,
            **{col: "category" for col in context_cols},
        },
    )
    return normalize_context_columns(meta, context_cols)


def build_pool_index(
    meta: pd.DataFrame, tissue_column: str
) -> dict[tuple, np.ndarray]:
    """Map (dataset_id, donor_id, tissue, cell_type) -> ascending pool row indices.

    Equivalent to `build_source_chunk_index()` but keyed on absolute pool rows, so it
    does not require every chunk file to be present on disk.
    """
    keys = ["dataset_id", "donor_id", tissue_column, "cell_type"]
    grouped = meta.groupby(keys, observed=True, sort=False).indices
    return {
        key: np.asarray(positions, dtype=np.int64)
        for key, positions in grouped.items()
    }


# =========================
# Replay
# =========================

def replay_map(
    plan_rows: list[dict],
    pool_index: dict[tuple, np.ndarray],
    *,
    seed: int,
    write_chunk_size: int,
    strict: bool,
):
    """Yield one DataFrame per plan chunk of recovered (sample, cell) associations.

    Reproduces the draw sequence of `generate_pseudo_bulk_chunks_from_cached_sources()`.
    The RNG is re-seeded per chunk of `write_chunk_size` plan rows and consumed
    sequentially within it, so every row of a chunk must be replayed in order -- you
    cannot skip to a single sample. Chunks are independent and could be parallelized.
    """
    n_chunks = int(np.ceil(len(plan_rows) / write_chunk_size))
    missing_pools: dict[tuple, int] = defaultdict(int)

    for chunk_id in range(n_chunks):
        start = chunk_id * write_chunk_size
        chunk_plan = plan_rows[start:start + write_chunk_size]
        rng = np.random.default_rng(seed + chunk_id)

        sample_ids: list[str] = []
        pool_rows: list[np.ndarray] = []
        copies: list[np.ndarray] = []

        for row in chunk_plan:
            context = row["context"]
            drawn: list[np.ndarray] = []

            for cell_type, count in row["cell_type_counts"].items():
                pool = pool_index.get((*context, cell_type))
                if pool is None or pool.size == 0:
                    # Cannot skip the draw: doing so would desynchronize the RNG for
                    # every later sample in this chunk.
                    missing_pools[(*context, cell_type)] += 1
                    if strict:
                        raise ValueError(
                            f"No source pool for context={context}, "
                            f"cell_type={cell_type!r} (sample {row['sample_id']}). "
                            "The plan and source metadata do not come from the same run."
                        )
                    continue
                count = int(count)
                chosen = rng.choice(
                    len(pool), size=count, replace=len(pool) < count
                )
                drawn.append(pool[np.asarray(chosen, dtype=np.int64)])

            if not drawn:
                continue

            # Pools are disjoint across cell types (a cell has exactly one cell_type),
            # so deduplicating on pool row alone is safe.
            unique_rows, n_copies = np.unique(
                np.concatenate(drawn), return_counts=True
            )
            sample_ids.append(row["sample_id"])
            pool_rows.append(unique_rows)
            copies.append(n_copies)

        if not sample_ids:
            continue

        lengths = np.fromiter((len(r) for r in pool_rows), dtype=np.int64)
        frame = pd.DataFrame(
            {
                "sample_id": np.repeat(np.asarray(sample_ids, dtype=object), lengths),
                "pool_row": np.concatenate(pool_rows),
                "n_copies": np.concatenate(copies).astype(np.int32),
            }
        )
        frame["plan_chunk_id"] = np.int32(chunk_id)
        yield frame

        del frame, pool_rows, copies
        gc.collect()

    if missing_pools:
        total = sum(missing_pools.values())
        print(
            f"WARNING: {len(missing_pools)} (context, cell_type) pools were absent from "
            f"the source metadata, affecting {total} draws. Affected samples are "
            "incomplete and their sums will NOT reproduce. Re-run with --strict to fail "
            "instead.",
            file=sys.stderr,
        )


def attach_cell_metadata(
    frame: pd.DataFrame, meta: pd.DataFrame, chunk_size: int
) -> pd.DataFrame:
    """Add soma_joinid, cell_type and the physical (chunk_id, row_idx) location."""
    pool_row = frame["pool_row"].to_numpy()
    frame["soma_joinid"] = meta["soma_joinid"].to_numpy()[pool_row]
    frame["cell_type"] = pd.Categorical(meta["cell_type"].to_numpy()[pool_row])
    frame["chunk_id"] = (pool_row // chunk_size).astype(np.int32)
    frame["row_idx"] = (pool_row % chunk_size).astype(np.int32)
    return frame[
        [
            "sample_id",
            "soma_joinid",
            "cell_type",
            "n_copies",
            "chunk_id",
            "row_idx",
            "pool_row",
            "plan_chunk_id",
        ]
    ]


# =========================
# Coverage
# =========================

def available_chunk_ids(chunk_dir: Path) -> set[int]:
    if not chunk_dir.is_dir():
        return set()
    ids = set()
    for path in chunk_dir.glob("source_cells_chunk_*.h5ad"):
        match = CHUNK_NAME_RE.search(path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


# =========================
# Verification
# =========================

def sum_cells_from_chunks(
    cell_rows: pd.DataFrame, chunk_dir: Path, n_genes: int
) -> np.ndarray:
    """Sum the recovered cells (weighted by n_copies) into one expression vector.

    Follows the read pattern of `aggregate_selected_cached_cells()`: one chunk file open
    at a time, rows pulled by fancy index.
    """
    import anndata as ad

    total = np.zeros(n_genes, dtype=np.float64)
    for chunk_id, group in cell_rows.groupby("chunk_id", sort=True):
        path = chunk_dir / f"source_cells_chunk_{int(chunk_id):05d}.h5ad"
        adata = ad.read_h5ad(path)
        rows = group["row_idx"].to_numpy(dtype=np.int64)
        weights = group["n_copies"].to_numpy(dtype=np.float64)
        selected = adata.X[rows]
        if sparse.issparse(selected):
            # Stay sparse: densifying 1000 x 68987 would cost ~550 MB per sample.
            weighted = selected.multiply(weights[:, None])
            total += np.asarray(weighted.sum(axis=0), dtype=np.float64).ravel()
            del weighted
        else:
            total += (
                np.asarray(selected, dtype=np.float64) * weights[:, None]
            ).sum(axis=0)
        del adata, selected
        gc.collect()
    return total


def verify(
    cell_map: pd.DataFrame,
    *,
    pseudo_bulk_h5ad: Path,
    chunk_dir: Path,
    n_samples: int,
    verify_seed: int,
    rtol: float,
    atol: float,
) -> bool:
    """Re-sum the recovered cells for random samples and compare to the pseudo-bulk."""
    import anndata as ad

    have_chunks = available_chunk_ids(chunk_dir)
    if not have_chunks:
        raise FileNotFoundError(f"No source chunks found in {chunk_dir}")

    # Only samples whose cells are all on disk can be re-summed.
    per_sample_chunks = cell_map.groupby("sample_id", observed=True)["chunk_id"].apply(
        lambda s: set(s.unique())
    )
    complete = per_sample_chunks[
        per_sample_chunks.apply(lambda s: s.issubset(have_chunks))
    ].index

    pb = ad.read_h5ad(pseudo_bulk_h5ad, backed="r")
    try:
        pb_names = pd.Index(pb.obs_names.astype(str))
        candidates = pb_names.intersection(pd.Index(complete))
        if len(candidates) == 0:
            raise ValueError(
                "No pseudo-bulk sample has all of its source cells available locally; "
                "nothing can be verified. Download more source chunks."
            )
        print(
            f"Verifiable samples: {len(candidates)} "
            f"(of {pb.n_obs} in the h5ad, {cell_map['sample_id'].nunique()} in the map)"
        )

        rng = np.random.default_rng(verify_seed)
        n_pick = min(n_samples, len(candidates))
        picked = rng.choice(np.asarray(candidates), size=n_pick, replace=False)

        # Align gene axes: the source chunks may carry more genes than the pseudo-bulk
        # if the latter was reindexed to a shorter gene list.
        first_chunk = chunk_dir / f"source_cells_chunk_{min(have_chunks):05d}.h5ad"
        src = ad.read_h5ad(first_chunk, backed="r")
        try:
            src_genes = pd.Index(src.var_names.astype(str))
        finally:
            src_file = getattr(src, "file", None)
            if src_file is not None:
                src_file.close()
        pb_genes = pd.Index(pb.var_names.astype(str))
        if src_genes.equals(pb_genes):
            gene_idx = None
        else:
            gene_idx = src_genes.get_indexer(pb_genes)
            n_missing = int((gene_idx < 0).sum())
            print(
                f"Gene axes differ (source {len(src_genes)}, pseudo-bulk "
                f"{len(pb_genes)}); aligned on names, {n_missing} pseudo-bulk genes "
                "absent from source (treated as 0)."
            )

        indexed = cell_map.set_index("sample_id", drop=False)
        all_ok = True
        print(
            f"\n{'sample_id':<22}{'cells':>7}{'uniq':>7}"
            f"{'max_abs_diff':>15}{'n_diff':>9}  result"
        )
        print("-" * 74)

        for sample_id in picked:
            rows = indexed.loc[[sample_id]]
            recon = sum_cells_from_chunks(rows, chunk_dir, len(src_genes))
            if gene_idx is not None:
                aligned = np.where(gene_idx >= 0, recon[gene_idx], 0.0)
            else:
                aligned = recon

            # Index X directly: row access on a backed AnnData view is unreliable.
            pb_row = pb.X[int(pb_names.get_loc(sample_id))]
            expected = np.asarray(
                pb_row.todense() if sparse.issparse(pb_row) else pb_row,
                dtype=np.float64,
            ).ravel()

            diff = np.abs(aligned - expected)
            ok = np.allclose(aligned, expected, rtol=rtol, atol=atol)
            all_ok &= bool(ok)
            print(
                f"{sample_id:<22}{int(rows['n_copies'].sum()):>7}{len(rows):>7}"
                f"{diff.max():>15.6g}{int((diff > atol).sum()):>9}"
                f"  {'PASS' if ok else 'FAIL'}"
            )
    finally:
        file_obj = getattr(pb, "file", None)
        if file_obj is not None:
            file_obj.close()

    print("-" * 74)
    print(
        "VERIFICATION PASSED: the recovered map reproduces the pseudo-bulks exactly."
        if all_ok
        else "VERIFICATION FAILED: the pseudo-bulks were not generated from this plan / "
        "source metadata / seed."
    )
    return all_ok


# =========================
# Entrypoint
# =========================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pseudo-bulk-dir", type=Path, default=Path("data/pseudo_bulk"),
                        help="Root holding the generator's CSV artifacts and source_cell_chunks/.")
    parser.add_argument("--plan-csv", type=Path, default=None,
                        help="Default: <pseudo-bulk-dir>/pseudo_bulk_sampling_plan.csv")
    parser.add_argument("--source-meta-csv", type=Path, default=None,
                        help="Default: <pseudo-bulk-dir>/sampled_source_cells.csv")
    parser.add_argument("--source-chunk-dir", type=Path, default=None,
                        help="Default: <pseudo-bulk-dir>/source_cell_chunks")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output map. Default: <pseudo-bulk-dir>/pseudo_bulk_cell_map.parquet")
    parser.add_argument("--pseudo-bulk-h5ad", type=Path, default=None,
                        help="Generated pseudo-bulk matrix; required for --verify.")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="Verify N randomly chosen pseudo-bulks by re-summing (0 = skip).")
    parser.add_argument("--verify-seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Must equal the generator's SCBFM_PSEUDO_RANDOM_SEED.")
    parser.add_argument("--write-chunk-size", type=int, default=WRITE_CHUNK_SIZE,
                        help="Must equal the generator's SCBFM_PSEUDO_WRITE_CHUNK_SIZE.")
    parser.add_argument("--download-chunk-size", type=int, default=DOWNLOAD_CHUNK_SIZE,
                        help="Must equal the generator's SCBFM_PSEUDO_DOWNLOAD_CHUNK_SIZE.")
    parser.add_argument("--tissue-column", type=str, default=TISSUE_COLUMN)
    parser.add_argument("--only-available", action="store_true",
                        help="Restrict the written map to samples whose cells are all on disk.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail instead of warning when a (context, cell_type) pool is missing.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.pseudo_bulk_dir
    plan_csv = args.plan_csv or root / "pseudo_bulk_sampling_plan.csv"
    meta_csv = args.source_meta_csv or root / "sampled_source_cells.csv"
    chunk_dir = args.source_chunk_dir or root / "source_cell_chunks"
    out_path = args.out or root / "pseudo_bulk_cell_map.parquet"

    for path in (plan_csv, meta_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    print(f"Plan:        {plan_csv}")
    print(f"Source meta: {meta_csv}")
    print(f"Chunks:      {chunk_dir}")
    print(f"Seed {args.seed}, write_chunk_size {args.write_chunk_size}, "
          f"download_chunk_size {args.download_chunk_size}")

    plan_rows = load_plan(plan_csv, args.tissue_column)
    print(f"Loaded {len(plan_rows)} plan rows")

    meta = load_source_meta(meta_csv, args.tissue_column)
    print(f"Loaded {len(meta)} source cells")

    pool_index = build_pool_index(meta, args.tissue_column)
    print(f"Built {len(pool_index)} (context, cell_type) pools")

    frames = []
    for frame in replay_map(
        plan_rows,
        pool_index,
        seed=args.seed,
        write_chunk_size=args.write_chunk_size,
        strict=args.strict,
    ):
        frames.append(attach_cell_metadata(frame, meta, args.download_chunk_size))
    if not frames:
        raise ValueError("Replay produced no associations.")
    cell_map = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    have_chunks = available_chunk_ids(chunk_dir)
    per_sample_chunks = cell_map.groupby("sample_id", observed=True)["chunk_id"].apply(
        lambda s: set(s.unique())
    )
    complete = set(
        per_sample_chunks[
            per_sample_chunks.apply(lambda s: s.issubset(have_chunks))
        ].index
    )
    cell_map["cells_available"] = cell_map["sample_id"].isin(complete)

    n_samples = cell_map["sample_id"].nunique()
    print(
        f"\nRecovered {len(cell_map)} unique (pseudo-bulk, cell) pairs across "
        f"{n_samples} pseudo-bulks\n"
        f"Distinct source cells used: {cell_map['soma_joinid'].nunique()}\n"
        f"Source chunks on disk: {len(have_chunks)}\n"
        f"Pseudo-bulks with all cells available: {len(complete)} / {n_samples}"
    )

    written = cell_map[cell_map["cells_available"]] if args.only_available else cell_map
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        written.to_parquet(out_path, index=False)
    else:
        written.to_csv(out_path, index=False)
    print(f"Wrote {len(written)} rows to {out_path}")

    summary = {
        "plan_csv": str(plan_csv),
        "source_meta_csv": str(meta_csv),
        "source_chunk_dir": str(chunk_dir),
        "seed": args.seed,
        "write_chunk_size": args.write_chunk_size,
        "download_chunk_size": args.download_chunk_size,
        "n_pairs": int(len(cell_map)),
        "n_pseudo_bulks": int(n_samples),
        "n_distinct_cells": int(cell_map["soma_joinid"].nunique()),
        "n_source_chunks_on_disk": len(have_chunks),
        "n_pseudo_bulks_fully_available": len(complete),
        "only_available": bool(args.only_available),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote {summary_path}")

    if args.verify > 0:
        if args.pseudo_bulk_h5ad is None:
            raise ValueError("--verify requires --pseudo-bulk-h5ad")
        ok = verify(
            cell_map,
            pseudo_bulk_h5ad=args.pseudo_bulk_h5ad,
            chunk_dir=chunk_dir,
            n_samples=args.verify,
            verify_seed=args.verify_seed,
            rtol=args.rtol,
            atol=args.atol,
        )
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
