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

Materialize a paired dataset keeping 50 cells per pseudo-bulk, tagged with the modality
labels the training dataset splits on::

    python data/reconstruct_pseudobulk_cell_map.py \
        --pseudo-bulk-dir data/pseudo_bulk \
        --pseudo-bulk-h5ad /path/to/pseudo_bulk_RAW.h5ad \
        --rebuild --output-dir data/paired \
        --max-cells-per-pb 50

Both rebuild outputs carry ``obs['modality']`` (``pseudobulk`` / ``sc``) so they can be
fed straight to ``bulk_sc_data_preprocessing.py``, which needs it to tell the row kinds
apart. Capping cells per pseudo-bulk breaks the "cells sum to their pseudo-bulk"
contract by design; the cap is recorded in the cell file's ``uns`` and
``--verify-rebuilt`` skips rather than reporting a false mismatch.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
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
# Rebuild: materialize a paired (pseudo-bulk, single-cell) dataset
# =========================

VLEN_STR = h5py.special_dtype(vlen=str)
SAMPLE_ID_RE = re.compile(r"(\d+)\s*$")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values],
        dtype=object,
    )


def pseudobulk_id_from_sample_id(sample_ids) -> np.ndarray:
    """`pseudo_bulk:000123` -> 123. Falls back to positional ids if unparseable."""
    ids = []
    for value in sample_ids:
        match = SAMPLE_ID_RE.search(str(value))
        ids.append(int(match.group(1)) if match else -1)
    ids = np.asarray(ids, dtype=np.int64)
    if (ids < 0).any():
        print(
            "WARNING: some sample_ids have no trailing integer; falling back to "
            "positional pseudobulk_id.",
            file=sys.stderr,
        )
        ids = np.arange(len(ids), dtype=np.int64)
    return ids


def write_elem(group: "h5py.Group", key: str, value) -> None:
    """anndata's own element writer: `anndata.io` on >=0.11, `.experimental` before.

    Used instead of hand-rolling the on-disk encoding for obs/var and the empty
    slots, so categorical/string columns follow whatever spec version the installed
    anndata reads back.
    """
    try:
        from anndata.io import write_elem as _write_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import write_elem as _write_elem
    _write_elem(group, key, value)


class ChunkReader:
    """Partial CSR row reads from the source chunks, with a small open-file cache.

    Reads only the rows requested (via indptr slicing) instead of loading whole
    chunk files, which matters because a rebuild touches thousands of chunks.
    """

    def __init__(self, chunk_dir: Path, max_open: int = 16):
        self.chunk_dir = chunk_dir
        self.max_open = max_open
        self._open: dict[int, tuple] = {}

    def _get(self, chunk_id: int):
        entry = self._open.get(chunk_id)
        if entry is not None:
            return entry
        if len(self._open) >= self.max_open:
            oldest = next(iter(self._open))
            self._open.pop(oldest)[0].close()
        path = self.chunk_dir / f"source_cells_chunk_{chunk_id:05d}.h5ad"
        handle = h5py.File(path, "r")
        entry = (handle, handle["X/indptr"][:])
        self._open[chunk_id] = entry
        return entry

    def n_vars(self) -> int:
        chunk_id = min(available_chunk_ids(self.chunk_dir))
        handle, _ = self._get(chunk_id)
        return int(handle["X"].attrs["shape"][1])

    def var_names(self) -> pd.Index:
        chunk_id = min(available_chunk_ids(self.chunk_dir))
        handle, _ = self._get(chunk_id)
        var = handle["var"]
        index_name = var.attrs.get("_index", "ensembl_id")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        return pd.Index(_decode(var[index_name][:]))

    def read_rows(self, chunk_id: int, rows: np.ndarray):
        """Return (csr block over the source gene axis, obs dict) for `rows`."""
        handle, indptr = self._get(chunk_id)
        data_ds, idx_ds = handle["X/data"], handle["X/indices"]
        n_vars = int(handle["X"].attrs["shape"][1])

        data_parts, idx_parts, counts = [], [], []
        for row in rows:
            start, end = int(indptr[row]), int(indptr[row + 1])
            counts.append(end - start)
            if end > start:
                data_parts.append(data_ds[start:end])
                idx_parts.append(idx_ds[start:end])
        block = sparse.csr_matrix(
            (
                np.concatenate(data_parts) if data_parts else np.zeros(0, np.float32),
                np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int32),
                np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            ),
            shape=(len(rows), n_vars),
        )
        return block, self._read_obs(handle, rows)

    @staticmethod
    def _read_obs(handle, rows: np.ndarray) -> dict:
        obs = handle["obs"]
        index_name = obs.attrs.get("_index", "cell_id")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        out: dict[str, np.ndarray] = {}
        for key in [index_name] + [k for k in obs.keys() if k != index_name]:
            node = obs[key]
            if isinstance(node, h5py.Group):  # categorical
                categories = _decode(node["categories"][:])
                codes = node["codes"][:][rows]
                out[key] = np.where(codes >= 0, categories[codes], "unknown")
            else:
                values = node[:][rows]
                out[key] = _decode(values) if values.dtype == object else values
        out["__index__"] = out.pop(index_name)
        return out

    def close(self):
        for handle, _ in self._open.values():
            handle.close()
        self._open.clear()


class StreamingCSRWriter:
    """Append CSR blocks to an .h5ad without holding the matrix in memory.

    Only X is written by hand -- a full rebuild is far too large to build in memory
    and hand to `write_h5ad`. obs/var and the empty slots go through anndata's own
    writer in `finalize()`.
    """

    def __init__(self, path: Path, n_vars: int, compression: str | None):
        # libver='earliest' matches write_h5ad_compat() so clusters on HDF5 < 1.10
        # can read the output.
        self.handle = h5py.File(path, "w", libver="earliest")
        self.handle.attrs["encoding-type"] = "anndata"
        self.handle.attrs["encoding-version"] = "0.1.0"
        self.n_vars = n_vars
        self.nnz = 0
        self.n_obs = 0
        # int64 numpy parts rather than a Python list of ints: a full rebuild passes
        # the 2^31 nonzeros addressable by scipy's default int32 indptr, and
        # `block.indptr + self.nnz` overflows there (NumPy 2 raises OverflowError).
        self._indptr_parts: list[np.ndarray] = []

        group = self.handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        kwargs = dict(chunks=(1 << 16,), maxshape=(None,), compression=compression)
        self.data = group.create_dataset("data", shape=(0,), dtype=np.float32, **kwargs)
        self.indices = group.create_dataset("indices", shape=(0,), dtype=np.int32, **kwargs)

    def append(self, block: sparse.csr_matrix) -> None:
        block = block.tocsr()
        block.sort_indices()
        n_new = block.data.shape[0]
        self.data.resize((self.nnz + n_new,))
        self.indices.resize((self.nnz + n_new,))
        if n_new:
            self.data[self.nnz:] = block.data.astype(np.float32, copy=False)
            self.indices[self.nnz:] = block.indices.astype(np.int32, copy=False)
        self._indptr_parts.append(block.indptr[1:].astype(np.int64) + self.nnz)
        self.nnz += n_new
        self.n_obs += block.shape[0]

    def finalize(
        self,
        obs: pd.DataFrame,
        var_names: pd.Index,
        uns: dict | None = None,
    ) -> None:
        group = self.handle["X"]
        group.attrs["shape"] = np.array([self.n_obs, self.n_vars], dtype="int64")
        # int64 indptr: a full rebuild can exceed the 2^31 nnz that int32 allows.
        group.create_dataset(
            "indptr",
            data=np.concatenate([np.zeros(1, dtype=np.int64), *self._indptr_parts]),
        )
        if self.nnz > np.iinfo(np.int32).max:
            print(
                f"NOTE: {self.nnz} nonzeros exceeds int32; the file needs an int64-aware "
                "reader and ~"
                f"{self.nnz * 12 / 1e9:.0f} GB of RAM to load unbacked. Consider "
                "--limit-samples, or dropping --expand-copies and using the n_copies "
                "weight instead."
            )

        obs = obs.rename_axis("cell_id")
        write_elem(self.handle, "obs", obs)
        write_elem(
            self.handle,
            "var",
            pd.DataFrame(index=pd.Index(var_names, dtype=str, name="ensembl_id")),
        )
        for name in ("layers", "obsm", "obsp", "varm", "varp"):
            write_elem(self.handle, name, {})
        # uns records how the file was built, so verify_rebuilt can tell a genuine
        # mismatch from an expected one after --max-cells-per-pb dropped rows.
        write_elem(self.handle, "uns", uns or {})
        self.handle.close()


def annotate_pseudo_bulk(
    src_path: Path,
    dst_path: Path,
    complete: set[str],
    modality_label: str = "pseudobulk",
) -> tuple[np.ndarray, pd.Index]:
    """Copy the pseudo-bulk h5ad verbatim, adding pseudobulk_id, cells_available, modality.

    Copying rather than rebuilding preserves uns (the cell_type_proportion_* maps the
    deconv runner relies on) and every existing obs column.

    ``modality`` is what ``BulkSCDataset`` splits rows on downstream (``--pb-label``),
    so it has to be on the file before ``bulk_sc_data_preprocessing.py`` sees it.
    """
    with h5py.File(src_path, "r") as src:
        obs = src["obs"]
        index_name = obs.attrs.get("_index", "sample_id")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        sample_ids = _decode(obs[index_name][:])
        column_order = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in obs.attrs.get("column-order", [])
        ]
        var = src["var"]
        var_index = var.attrs.get("_index", "ensembl_id")
        if isinstance(var_index, bytes):
            var_index = var_index.decode()
        var_names = pd.Index(_decode(var[var_index][:]))

        with h5py.File(dst_path, "w", libver="earliest") as dst:
            for key in src.keys():
                src.copy(key, dst)
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            pb_ids = pseudobulk_id_from_sample_id(sample_ids)
            available = np.isin(sample_ids.astype(str), np.asarray(sorted(complete)))

            dst_obs = dst["obs"]
            added = ("pseudobulk_id", "cells_available", "modality")
            for name, values in (
                ("pseudobulk_id", pb_ids),
                ("cells_available", available),
            ):
                if name in dst_obs:
                    del dst_obs[name]
                ds = dst_obs.create_dataset(name, data=values)
                ds.attrs["encoding-type"] = "array"
                ds.attrs["encoding-version"] = "0.2.0"

            # A single-valued string column: written through anndata's own writer as a
            # Categorical rather than a raw dataset, so it round-trips whatever encoding
            # the installed anndata reads back.
            if "modality" in dst_obs:
                del dst_obs["modality"]
            write_elem(
                dst_obs,
                "modality",
                pd.Categorical([modality_label] * len(sample_ids)),
            )

            new_order = [c for c in column_order if c not in added]
            new_order += list(added)
            dst_obs.attrs.create(
                "column-order", np.array(new_order, dtype=object), dtype=VLEN_STR
            )

    return sample_ids, var_names


def rebuild(
    cell_map: pd.DataFrame,
    *,
    pseudo_bulk_h5ad: Path,
    chunk_dir: Path,
    output_dir: Path,
    gene_axis: str,
    expand_copies: bool,
    limit_samples: int,
    compression: str | None,
    max_open_chunks: int,
    max_cells_per_pb: int = 0,
    subsample_seed: int = 0,
    pb_modality: str = "pseudobulk",
    sc_modality: str = "sc",
) -> tuple[Path, Path]:
    """Write a paired pseudo-bulk + matched single-cell dataset joined on pseudobulk_id.

    ``max_cells_per_pb`` caps how many distinct source cells are stored per pseudo-bulk
    (0 = all). Capping keeps the file to a workable size when only a handful of cells
    per pseudo-bulk are ever needed — as in aggregation-consistency training, which
    samples n_sc_per_pseudobulk of them per batch. It does mean the stored cells no
    longer sum to their pseudo-bulk, so the cap is recorded in the cell file's ``uns``
    and :func:`verify_rebuilt` reports "not verifiable" rather than a false mismatch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pb_out = output_dir / "pseudo_bulk_matched.h5ad"
    sc_out = output_dir / "source_cells_matched.h5ad"

    have_chunks = available_chunk_ids(chunk_dir)
    if not have_chunks:
        raise FileNotFoundError(f"No source chunks found in {chunk_dir}")

    if "cells_available" in cell_map.columns:
        complete = set(cell_map.loc[cell_map["cells_available"], "sample_id"].unique())
    else:
        per_sample_chunks = cell_map.groupby("sample_id", observed=True)["chunk_id"].apply(
            lambda s: set(s.unique())
        )
        complete = set(
            per_sample_chunks[per_sample_chunks.apply(lambda s: s.issubset(have_chunks))].index
        )

    print(f"Annotating pseudo-bulk -> {pb_out}")
    sample_ids, pb_var_names = annotate_pseudo_bulk(
        pseudo_bulk_h5ad, pb_out, complete, modality_label=pb_modality
    )

    # Emit cells only for pseudo-bulks that exist in the h5ad AND have all cells on disk.
    usable = [s for s in sample_ids.astype(str) if s in complete]
    usable = sorted(usable, key=lambda s: int(SAMPLE_ID_RE.search(s).group(1))
                    if SAMPLE_ID_RE.search(s) else 0)
    if limit_samples > 0:
        usable = usable[:limit_samples]
    if not usable:
        raise ValueError(
            "No pseudo-bulk in the h5ad has all of its source cells available locally."
        )
    print(
        f"Pseudo-bulks: {len(sample_ids)} in h5ad, {len(complete)} with cells on disk, "
        f"{len(usable)} being written"
    )

    reader = ChunkReader(chunk_dir, max_open=max_open_chunks)
    try:
        src_var_names = reader.var_names()
        if gene_axis == "pseudobulk":
            target_var = pb_var_names
            gene_idx = src_var_names.get_indexer(target_var)
            present_dst = np.where(gene_idx >= 0)[0]
            present_src = gene_idx[present_dst]
            n_missing = len(target_var) - len(present_dst)
            if n_missing:
                print(
                    f"{n_missing} pseudo-bulk genes absent from the source chunks; "
                    "zero-filled in the cell matrix."
                )
        else:
            target_var = src_var_names
            present_dst = present_src = None

        # Restrict before grouping: the full map is ~14M rows / 20k groups.
        subset = cell_map[cell_map["sample_id"].isin(set(usable))]
        by_sample = {k: v for k, v in subset.groupby("sample_id", observed=True)}
        writer = StreamingCSRWriter(sc_out, len(target_var), compression)
        obs_frames: list[pd.DataFrame] = []

        subsample_rng = np.random.default_rng(subsample_seed)
        n_capped = 0
        for position, sample_id in enumerate(usable):
            rows = by_sample[sample_id].sort_values(["chunk_id", "row_idx"])
            if max_cells_per_pb > 0 and len(rows) > max_cells_per_pb:
                # Uniform over the distinct cells, not weighted by n_copies: the
                # aggregation-consistency sampler draws uniformly from whatever is
                # stored, so a weighted pick here would bias it twice. Kept in
                # (chunk_id, row_idx) order so the reads below stay sequential.
                keep = subsample_rng.choice(
                    len(rows), size=max_cells_per_pb, replace=False
                )
                rows = rows.iloc[np.sort(keep)]
                n_capped += 1
            blocks, obs_parts = [], []
            for chunk_id, group in rows.groupby("chunk_id", sort=True):
                row_idx = group["row_idx"].to_numpy(dtype=np.int64)
                block, obs_dict = reader.read_rows(int(chunk_id), row_idx)
                copies = group["n_copies"].to_numpy(dtype=np.int64)
                if expand_copies:
                    repeat = np.repeat(np.arange(len(row_idx)), copies)
                    block = block[repeat]
                    obs_dict = {k: np.asarray(v)[repeat] for k, v in obs_dict.items()}
                    copies = np.ones(block.shape[0], dtype=np.int64)
                blocks.append(block)
                part = pd.DataFrame({k: v for k, v in obs_dict.items() if k != "__index__"})
                part["cell_id"] = obs_dict["__index__"]
                part["n_copies"] = copies.astype(np.int32)
                obs_parts.append(part)

            block = sparse.vstack(blocks, format="csr")
            if present_dst is not None:
                sub = block.tocsc()[:, present_src].tocoo()
                block = sparse.csr_matrix(
                    (sub.data, (sub.row, present_dst[sub.col])),
                    shape=(block.shape[0], len(target_var)),
                    dtype=np.float32,
                )
            writer.append(block)

            obs_part = pd.concat(obs_parts, ignore_index=True)
            obs_part["sample_id"] = sample_id
            obs_part["pseudobulk_id"] = np.int64(
                pseudobulk_id_from_sample_id([sample_id])[0]
            )
            obs_frames.append(obs_part)

            if (position + 1) % 250 == 0 or position + 1 == len(usable):
                print(
                    f"  {position + 1}/{len(usable)} pseudo-bulks | "
                    f"{writer.n_obs} cells | {writer.nnz} nonzeros"
                )

        obs = pd.concat(obs_frames, ignore_index=True)
        del obs_frames
        gc.collect()

        # Unique per (pseudo-bulk, cell); a cell reused by several pseudo-bulks appears once each.
        obs.index = pd.Index(
            obs["sample_id"].astype(str) + "|" + obs["cell_id"].astype(str),
            name="cell_id",
        )
        obs = obs.drop(columns=["cell_id"])
        if obs.index.duplicated().any():
            obs.index = pd.Index(
                [f"{name}#{i}" for i, name in enumerate(obs.index)], name="cell_id"
            )
        # What BulkSCDataset splits rows on downstream (--sc-label).
        obs["modality"] = sc_modality
        for column in (
            "dataset_id", "donor_id", "tissue_general", "cell_type", "sample_id",
            "modality",
        ):
            if column in obs.columns:
                obs[column] = pd.Categorical(obs[column])
        tail = ["sample_id", "pseudobulk_id", "n_copies", "modality"]
        ordered = [c for c in obs.columns if c not in tail]
        obs = obs[ordered + tail]

        writer.finalize(
            obs,
            target_var,
            uns={
                # Self-describing: verify_rebuilt reads this to decide whether the
                # stored cells are supposed to reproduce their pseudo-bulk.
                "max_cells_per_pseudobulk": int(max_cells_per_pb),
                "subsample_seed": int(subsample_seed),
                "expand_copies": bool(expand_copies),
                "sc_modality": str(sc_modality),
                "pb_modality": str(pb_modality),
            },
        )
        print(
            f"Wrote {sc_out}: {writer.n_obs} cells x {len(target_var)} genes, "
            f"{writer.nnz} nonzeros ({sc_out.stat().st_size / 1e9:.2f} GB)"
        )
        if max_cells_per_pb > 0:
            print(
                f"Capped at {max_cells_per_pb} cells/pseudo-bulk: {n_capped} of "
                f"{len(usable)} pseudo-bulks were subsampled. Their stored cells no "
                f"longer sum to the pseudo-bulk, so --verify-rebuilt cannot check them."
            )
    finally:
        reader.close()

    return pb_out, sc_out


def verify_rebuilt(
    pb_path: Path, sc_path: Path, n_samples: int, seed: int, rtol: float, atol: float
) -> bool:
    """Re-sum cells from the WRITTEN cell file and compare to the written pseudo-bulk.

    Skipped when the file was written with a per-pseudo-bulk cell cap: only a subset of
    each pseudo-bulk's cells is stored, so the sums are *expected* to differ and a
    comparison would report a mismatch that is not one.
    """
    with h5py.File(sc_path, "r") as sc, h5py.File(pb_path, "r") as pb:
        cap = 0
        if "uns/max_cells_per_pseudobulk" in sc:
            cap = int(sc["uns/max_cells_per_pseudobulk"][()])
        if cap > 0:
            print(
                f"\nSKIPPED: the cell file stores at most {cap} cells per pseudo-bulk "
                f"(--max-cells-per-pb), so its cells are not meant to sum to the "
                f"pseudo-bulk. Nothing was verified. Rebuild without the cap to check "
                f"the mapping itself."
            )
            return True

        sc_ids = sc["obs/pseudobulk_id"][:]
        copies = sc["obs/n_copies"][:].astype(np.float64)
        sc_indptr, sc_data, sc_indices = sc["X/indptr"][:], sc["X/data"], sc["X/indices"]
        n_vars = int(sc["X"].attrs["shape"][1])

        pb_ids = pb["obs/pseudobulk_id"][:]
        pb_indptr, pb_data, pb_indices = pb["X/indptr"][:], pb["X/data"][:], pb["X/indices"][:]

        rng = np.random.default_rng(seed)
        targets = np.unique(sc_ids)
        picked = rng.choice(targets, size=min(n_samples, len(targets)), replace=False)

        print(f"\n{'pseudobulk_id':<16}{'cells':>8}{'max_abs_diff':>15}{'n_diff':>9}  result")
        print("-" * 60)
        all_ok = True
        for pb_id in picked:
            recon = np.zeros(n_vars, dtype=np.float64)
            for row in np.where(sc_ids == pb_id)[0]:
                start, end = int(sc_indptr[row]), int(sc_indptr[row + 1])
                if end > start:
                    recon[sc_indices[start:end]] += sc_data[start:end] * copies[row]
            pb_row = int(np.where(pb_ids == pb_id)[0][0])
            expected = np.zeros(n_vars, dtype=np.float64)
            start, end = int(pb_indptr[pb_row]), int(pb_indptr[pb_row + 1])
            expected[pb_indices[start:end]] = pb_data[start:end]

            diff = np.abs(recon - expected)
            ok = np.allclose(recon, expected, rtol=rtol, atol=atol)
            all_ok &= bool(ok)
            print(
                f"{int(pb_id):<16}{int(copies[sc_ids == pb_id].sum()):>8}"
                f"{diff.max():>15.6g}{int((diff > atol).sum()):>9}  {'PASS' if ok else 'FAIL'}"
            )
        print("-" * 60)
        print(
            "REBUILD VERIFIED: summing each pseudobulk_id's cells reproduces the pseudo-bulk."
            if all_ok
            else "REBUILD MISMATCH: the written files are not consistent."
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

    rebuild_group = parser.add_argument_group("rebuild")
    rebuild_group.add_argument("--rebuild", action="store_true",
                               help="Materialize a paired dataset: an annotated pseudo-bulk h5ad "
                                    "plus a matched single-cell h5ad, joined on pseudobulk_id. "
                                    "Requires --pseudo-bulk-h5ad and --output-dir.")
    rebuild_group.add_argument("--output-dir", type=Path, default=None,
                               help="Destination for pseudo_bulk_matched.h5ad and "
                                    "source_cells_matched.h5ad.")
    rebuild_group.add_argument("--gene-axis", choices=("pseudobulk", "source"), default="pseudobulk",
                               help="Gene axis for the cell matrix. 'pseudobulk' (default) makes both "
                                    "files share a var axis; 'source' keeps all source genes.")
    rebuild_group.add_argument("--expand-copies", action="store_true",
                               help="Emit one row per drawn copy so a plain sum reproduces the "
                                    "pseudo-bulk. Default emits unique cells with an n_copies weight.")
    rebuild_group.add_argument("--limit-samples", type=int, default=0, metavar="N",
                               help="Only write the first N pseudo-bulks (0 = all). Useful for a trial run.")
    rebuild_group.add_argument("--max-cells-per-pb", type=int, default=0, metavar="N",
                               help="Store at most N distinct source cells per pseudo-bulk "
                                    "(0 = all). Keeps the cell file small when only a few "
                                    "cells per pseudo-bulk are ever used, as in "
                                    "aggregation-consistency training. The stored cells then "
                                    "no longer sum to their pseudo-bulk, so --verify-rebuilt "
                                    "skips the comparison. With --expand-copies the cap "
                                    "applies to distinct cells, so the rows written per "
                                    "pseudo-bulk are the sum of their n_copies.")
    rebuild_group.add_argument("--subsample-seed", type=int, default=0,
                               help="Seed for the --max-cells-per-pb draw (default: 0). "
                                    "Independent of --seed, which must match the generator.")
    rebuild_group.add_argument("--pb-modality-label", type=str, default="pseudobulk",
                               help="Value written to obs['modality'] on the pseudo-bulk file. "
                                    "Must match --pb-label at training time (default: pseudobulk).")
    rebuild_group.add_argument("--sc-modality-label", type=str, default="sc",
                               help="Value written to obs['modality'] on the cell file. "
                                    "Must match --sc-label at training time (default: sc).")
    rebuild_group.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    rebuild_group.add_argument("--max-open-chunks", type=int, default=16)
    rebuild_group.add_argument("--verify-rebuilt", type=int, default=0, metavar="N",
                               help="After rebuilding, re-sum N pseudobulk_ids from the written "
                                    "files and compare.")
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

    exit_code = 0

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
        exit_code |= 0 if ok else 1

    if args.rebuild:
        if args.pseudo_bulk_h5ad is None:
            raise ValueError("--rebuild requires --pseudo-bulk-h5ad")
        if args.output_dir is None:
            raise ValueError("--rebuild requires --output-dir")
        print()
        pb_out, sc_out = rebuild(
            cell_map,
            pseudo_bulk_h5ad=args.pseudo_bulk_h5ad,
            chunk_dir=chunk_dir,
            output_dir=args.output_dir,
            gene_axis=args.gene_axis,
            expand_copies=args.expand_copies,
            limit_samples=args.limit_samples,
            compression=None if args.compression == "none" else args.compression,
            max_open_chunks=args.max_open_chunks,
            max_cells_per_pb=args.max_cells_per_pb,
            subsample_seed=args.subsample_seed,
            pb_modality=args.pb_modality_label,
            sc_modality=args.sc_modality_label,
        )
        if args.verify_rebuilt > 0:
            ok = verify_rebuilt(
                pb_out, sc_out, args.verify_rebuilt, args.verify_seed, args.rtol, args.atol
            )
            exit_code |= 0 if ok else 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
