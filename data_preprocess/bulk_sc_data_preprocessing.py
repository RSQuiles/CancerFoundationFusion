"""Convert h5ad files into a training store for ``pretrain.py``.

A store is the directory ``pretrain.py --train-path`` reads (see ``DatasetDir``)::

    vocab.json     gene -> token id (<cls>, <pad>, then the genes)
    mapping.json   obs column -> {category: code}
    obs.parquet    the selected obs columns, as category codes, one row per stored row
    mem.map/       the expression matrix (bionemo SCDL memmap)

Expression values are stored exactly as they are in the source ``X``: nothing is
normalized, log-transformed or filtered besides dropping genes outside the vocabulary.

    python bulk_sc_data_preprocessing.py --h5ad-path data/sc.h5ad data/bulk.h5ad \\
        --data-path out/pipeline_ready --obs-columns tissue_general assay paired \\
        --pb-id-column pseudobulk_id

The source files are only ever read. SCDL builds a memmap from an h5ad *file*, and the
training loaders read each gene's token id from a ``_cf_gene_id`` var column and expect
every stored column to be in the vocabulary. So for each source the selected rows and
in-vocabulary genes, plus that column, are written to ONE transient h5ad inside the
store, converted, and deleted before the next source is read.

Store contract the loaders rely on (``cancerfoundation/data/bulk_sc_data.py``):

- ``modality`` is always stored: taken from the source's obs, else from the source's
  label (``Source.modality``), else ``--default-modality``.
- an obs column a source does not have is filled with ``0``, and category ``"0"``
  means "none": an unpaired row for ``paired``, a row with no pseudobulk for the
  ``--pb-id-column``. The pseudobulk id is stored as ``"pb:<id>"`` so that a genuine
  id 0 cannot collide with that fill.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cancerfoundation.data.dataset import DatasetDir  # noqa: E402
from h5ad_io import SourceReader, expand_h5ad_paths  # noqa: E402

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"
MODALITY_COL = "modality"
PB_ID_PREFIX = "pb:"
TRANSIENT_DIR = "_transient"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    """One h5ad to put in the store; ``rows`` (obs names) selects a subset of it."""

    path: Path
    rows: list[str] | None = None
    modality: str | None = None


# --------------------------------------------------------------------------- #
# Vocabulary and obs encoding
# --------------------------------------------------------------------------- #


def build_vocab(gene_sets: list[pd.Index], mode: str = "intersection") -> dict[str, int]:
    """``<cls>``, ``<pad>``, then the genes shared by (or present in any of) the sources, sorted."""
    sets = [set(g) for g in gene_sets]
    genes = set.intersection(*sets) if mode == "intersection" else set.union(*sets)
    if not genes:
        raise ValueError(f"The sources share no genes ({mode}); check --gene-column")
    return {gene: i for i, gene in enumerate([CLS_TOKEN, PAD_TOKEN] + sorted(genes))}


def convert_columns_to_categorical_with_mapping(df):
    df_copy = df.copy()

    category_mappings = {}

    df_categorical = pd.DataFrame(index=df.index)

    for column in df.columns:
        col = df_copy[column]
        # Ensure its a 1D series
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        col = col.astype(str).astype("category") # convert to str first to deal with mixed types
        category_mappings[column] = dict(
            zip(
                col.cat.categories,
                range(len(col.cat.categories)),
            )
        )

        df_categorical[column] = col.cat.codes

    return df_categorical, category_mappings


def _store_obs(obs: pd.DataFrame, columns: list[str], modality: str,
               pb_id_column: str | None) -> pd.DataFrame:
    """The store's obs for one source; the file's own ``modality`` column wins over ``modality``."""
    out = obs.copy()
    if MODALITY_COL not in out.columns:
        out[MODALITY_COL] = modality
    if pb_id_column and pb_id_column in out.columns:
        # Every column becomes a category code and missing values are filled with 0,
        # so a bare id 0 would read as "no pseudobulk" and bind every unrelated cell.
        out[pb_id_column] = PB_ID_PREFIX + out[pb_id_column].astype(str)
    # reindex: a column this source lacks (e.g. "paired" in a non-paired file) is 0.
    return out.reindex(columns=columns, fill_value=0)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


# SCDL 0.1.2 keeps column indices in a uint32 file and, on its large-file path, copies int64
# indices into it byte for byte: every row then reads the wrong genes (see
# check_store_alignment.py, check 1b). scipy only keeps int32 indices below 2**31 entries,
# so each source is written as several transient files of at most this many entries, each
# cast to int32 explicitly. Smaller chunks also bound the memory of one conversion.
MAX_TRANSIENT_NNZ = 500_000_000
INT32_MAX = int(np.iinfo(np.int32).max)
FLOAT32_EXACT = 2 ** 24  # integers above this are not exactly representable in float32
# Read rows back after every build and compare them with the source. The offline self-check
# (check_preprocessing.py) fakes SCDL and turns this off; nothing else should.
VERIFY_STORES = True


def _row_chunks(reader: SourceReader, positions: np.ndarray, n_cols: int, max_nnz: int) -> list[np.ndarray]:
    """Consecutive pieces of ``positions`` with at most ``max_nnz`` stored entries each.

    Counts come from the source ``indptr`` (all genes, so an upper bound on the
    in-vocabulary entries). Without one (dense or CSC source) every entry is assumed
    stored. A single row above the limit is a chunk of its own.
    """
    nnz = reader.row_nnz(positions)
    if nnz is None:
        nnz = np.full(len(positions), n_cols, dtype=np.int64)
    chunks, start, total = [], 0, 0
    for i, n in enumerate(nnz):
        if i > start and total + int(n) > max_nnz:
            chunks.append(positions[start:i])
            start, total = i, 0
        total += int(n)
    chunks.append(positions[start:])
    return chunks


def _transient_matrix(X):
    """What SCDL may convert safely: canonical CSR, float32 values, int32 indices and indptr."""
    import warnings

    import scipy.sparse as sp

    X = sp.csr_matrix(X)
    X.sum_duplicates()
    X.sort_indices()
    if X.nnz > INT32_MAX or X.shape[1] > INT32_MAX:
        raise ValueError(f"A transient chunk has {X.nnz} entries and {X.shape[1]} columns, too many for "
                         "int32 indices; lower max_transient_nnz")
    if X.nnz and float(np.abs(X.data).max()) > FLOAT32_EXACT:
        warnings.warn(f"Values above {FLOAT32_EXACT} are not exact in float32 (max {np.abs(X.data).max():.6g})")
    X = sp.csr_matrix((X.data.astype(np.float32), X.indices.astype(np.int32), X.indptr.astype(np.int32)),
                      shape=X.shape)
    if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
        raise TypeError(f"transient indices must be int32, got {X.indices.dtype}/{X.indptr.dtype}")
    return X


def verify_store(data_path: Path, row_map: list[tuple], readers: list[SourceReader], n_rows: int = 50,
                 seed: int = 0) -> dict:
    """Read rows back through the training loader and compare them exactly with the source.

    ``row_map`` has one ``(first_store_row, source_index, positions, in_vocab, token_ids)``
    per transient file, in store order. Checks ``n_rows`` random rows plus the first and
    last row of every transient file (where a merge offset would show). Raises on any
    difference, so a corrupt store is never left behind as if it were valid.
    """
    from cancerfoundation.data.dataset import SingleCellDataset

    ds = SingleCellDataset(data_path)
    n = len(ds)
    rng = np.random.default_rng(seed)
    rows = set(rng.choice(n, size=min(n_rows, n), replace=False).tolist())
    for start, _, positions, _, _ in row_map:
        rows |= {start, start + len(positions) - 1}
    starts = np.array([m[0] for m in row_map])
    bad = []
    for r in sorted(rows):
        start, src, positions, in_vocab, token_ids = row_map[int(np.searchsorted(starts, r, side="right") - 1)]
        expected = readers[src].rows(np.array([positions[r - start]]))[:, in_vocab].toarray().ravel().astype(np.float32)
        item = ds[int(r)]
        ids, values = item["genes"].numpy()[1:], item["expressions"].numpy()[1:]
        if not np.array_equal(ids, token_ids):
            bad.append(f"store row {r} ({ds.obs.index[r]}): gene ids differ from the source columns")
        elif not np.array_equal(values, expected):
            bad.append(f"store row {r} ({ds.obs.index[r]}): {int((values != expected).sum())} of "
                       f"{len(values)} values differ (stored non-zeros {int((values != 0).sum())}, "
                       f"source {int((expected != 0).sum())})")
    if bad:
        raise RuntimeError(f"Store verification FAILED for {len(bad)} of {len(rows)} rows of {data_path}:\n  "
                           + "\n  ".join(bad[:10]))
    print(f"Store verified: {len(rows)} rows identical to their source")
    return {"rows": len(rows), "status": "PASS"}


def _load_into_collection(collection, h5ad_path: Path) -> None:
    """``SingleCellCollection.load_h5ad``, reading ``X`` on SCDL versions that default to ``raw.X``."""
    kwargs = {}
    if "use_X_not_raw" in inspect.signature(collection.load_h5ad).parameters:
        kwargs["use_X_not_raw"] = True
    collection.load_h5ad(str(h5ad_path), **kwargs)


def build_pipeline_ready(
    sources: list[Source],
    data_path: Path,
    obs_columns: list[str] | None,
    vocab_path: Path | None = None,
    vocab_mode: str = "intersection",
    gene_column: str | None = None, # Defaults to var.index if None
    pb_id_column: str | None = None,
    default_modality: str = "sc",
    max_transient_nnz: int = MAX_TRANSIENT_NNZ,
) -> dict:
    """Build the training store at ``data_path`` from ``sources``; returns a summary.

    ``vocab_path`` reuses an existing vocabulary; otherwise it is built from the
    sources' genes (``vocab_mode``: ``intersection`` or ``union``). Genes outside the
    vocabulary are left out of the store.

    Each source is converted in consecutive chunks of at most ``max_transient_nnz``
    stored entries, each written with int32 indices (see ``MAX_TRANSIENT_NNZ``), and the
    finished store is read back and compared with the sources (``verify_store``).
    """
    import anndata as ad
    from bionemo.scdl.io.single_cell_collection import SingleCellCollection

    data_path = Path(data_path)
    data_dir = DatasetDir(data_path)
    data_dir.mkdir()
    columns = list(dict.fromkeys((obs_columns or []) + [MODALITY_COL]))
    if pb_id_column and pb_id_column not in columns:
        columns.append(pb_id_column)

    readers = [SourceReader(s.path) for s in sources]
    try:
        if vocab_path is not None:
            print(f"Reading gene vocabulary {vocab_path}")
            vocab = json.loads(Path(vocab_path).read_text())
        else:
            print(f"Building the gene vocabulary ({vocab_mode} of {len(sources)} source(s))")
            vocab = build_vocab([r.gene_names(gene_column) for r in readers], vocab_mode)
        with open(data_dir.vocab_path, "w") as f:
            json.dump(vocab, f)

        transient = data_path / TRANSIENT_DIR
        transient.mkdir(parents=True, exist_ok=True)
        collection = SingleCellCollection(str(data_dir.data_dir / "tmp"))
        obs_parts: dict[str, pd.DataFrame] = {}
        summary = []
        chunk_info: dict[str, tuple] = {}
        for i, (source, reader) in enumerate(zip(sources, readers)):
            obs = reader.obs
            if source.rows is not None:
                keep = obs.index.isin(source.rows)
                missing = len(set(source.rows)) - int(keep.sum())
                if missing:
                    print(f"  [warn] {source.path.name}: {missing} requested rows are not in the file")
            else:
                keep = np.ones(len(obs), dtype=bool)
            positions = np.flatnonzero(keep)
            genes = reader.gene_names(gene_column)
            in_vocab = np.asarray(genes.isin(list(vocab))) & ~np.asarray(genes.isin([CLS_TOKEN, PAD_TOKEN]))
            print(f"[{i + 1}/{len(sources)}] {source.path.name}: {len(positions)} rows, "
                  f"{int(in_vocab.sum())}/{len(genes)} genes in the vocabulary")
            if not len(positions):
                print("  no rows selected; skipped")
                continue

            token_ids = np.array([vocab[g] for g in genes[in_vocab]])
            var = pd.DataFrame({GENE_ID: token_ids}, index=genes[in_vocab])
            modality = source.modality if source.modality is not None else default_modality
            chunks = _row_chunks(reader, positions, len(genes), max_transient_nnz)
            print(f"  {len(chunks)} transient file(s) of at most {max_transient_nnz:,} entries, int32 indices")
            for k, chunk in enumerate(chunks):
                label = f"{i:03d}_{source.path.stem}_c{k:03d}"
                X = _transient_matrix(reader.rows(chunk)[:, in_vocab])
                tmp_path = transient / f"{label}.h5ad"
                ad.AnnData(X=X, obs=pd.DataFrame(index=obs.index[chunk].astype(str)), var=var).write_h5ad(tmp_path)
                del X
                _load_into_collection(collection, tmp_path)
                tmp_path.unlink()
                obs_parts[label] = _store_obs(obs.iloc[chunk], columns, modality, pb_id_column)
                chunk_info[label] = (i, chunk, in_vocab, token_ids)
            summary.append({"source": str(source.path), "rows": int(len(positions)),
                            "genes_kept": int(in_vocab.sum()), "genes_total": int(len(genes)),
                            "transient_files": len(chunks)})
        shutil.rmtree(transient, ignore_errors=True)

        if not obs_parts:
            raise ValueError("No rows selected from any source")
        # flatten() concatenates the memmaps in insertion order; collect obs the same way.
        order = [Path(name).name for name in collection.fname_to_mmap.keys()]
        obs = pd.concat([obs_parts[label] for label in order])
        obs, mapping = convert_columns_to_categorical_with_mapping(obs)

        print("Exporting files...")
        obs.to_parquet(data_dir.obs_path)
        with data_dir.mapping_path.open("w") as f:
            json.dump(mapping, f, indent=4)

        collection.flatten(data_dir.memmap_path, destroy_on_copy=True)
        # Remove duplicate metadata file that causes issues
        if os.path.isfile(data_dir.memmap_path / "features/dataframe_00.parquet"):
            os.remove(data_dir.memmap_path / "features/dataframe_00.parquet")
        shutil.rmtree(data_dir.data_dir / "tmp", ignore_errors=True)

        verified = {"status": "SKIPPED"}
        if VERIFY_STORES:
            row_map, next_row = [], 0
            for label in order:
                src, chunk, in_vocab, token_ids = chunk_info[label]
                row_map.append((next_row, src, chunk, in_vocab, token_ids))
                next_row += len(chunk)
            verified = verify_store(data_path, row_map, readers)
    finally:
        for reader in readers:
            reader.close()

    print(f"Store written to {data_path} ({len(obs)} rows, {len(vocab) - 2} genes in the vocabulary)")
    return {"sources": summary, "n_rows": int(len(obs)), "n_vocab_genes": len(vocab) - 2,
            "obs_columns": columns, "max_transient_nnz": int(max_transient_nnz),
            "transient_dtypes": {"data": "float32", "indices": "int32", "indptr": "int32"},
            "verified": verified}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(args):
    data_path = Path(args.data_path)
    if data_path.exists() and any(data_path.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{data_path} is not empty; pass --overwrite to replace it")
        shutil.rmtree(data_path)
    rows = args.rows.read_text(encoding="utf-8").split() if args.rows else None
    sources = [Source(path=p, rows=rows) for p in expand_h5ad_paths(args.h5ad_path)]
    summary = build_pipeline_ready(
        sources,
        data_path,
        obs_columns=args.obs_columns,
        vocab_path=args.vocab_path,
        vocab_mode=args.vocab_mode,
        gene_column=args.gene_column,
        pb_id_column=args.pb_id_column,
        default_modality=args.default_modality,
        max_transient_nnz=args.max_transient_nnz,
    )
    (data_path / "store.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Conversion completed successfully.")


def get_args():
    parser = ArgumentParser(description="Convert h5ad files into a training store.")
    parser.add_argument("--h5ad-path", type=Path, nargs="+", required=True,
                        help="h5ad files and/or directories of h5ad files")
    parser.add_argument("--data-path", type=Path, required=True, help="Output store directory")
    parser.add_argument("--obs-columns", type=str, nargs="+", default=None,
                        help="obs columns kept in obs.parquet (modality is always kept)")
    parser.add_argument("--rows", type=Path, default=None,
                        help="Text file of obs names to keep (one per line); default: every row; can be any <part>.txt written by split_train_test.py")
    parser.add_argument("--vocab-path", type=Path, default=None, help="Path to an existing vocab.json")
    parser.add_argument("--vocab-mode", choices=["intersection", "union"], default="intersection",
                        help="Genes in the new vocabulary when --vocab-path is not given")
    parser.add_argument("--gene-column", default=None,
                        help="var column holding the gene names (default: the var index)")
    parser.add_argument("--pb-id-column", type=str, default=None,
                        help="obs column linking a precomputed pseudobulk to its cells (e.g. pseudobulk_id)")
    parser.add_argument("--default-modality", default="sc",
                        help="Modality written for files whose obs has no 'modality' column")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty --data-path")
    parser.add_argument("--max-transient-nnz", type=int, default=MAX_TRANSIENT_NNZ,
                        help="Stored entries per transient file handed to SCDL (kept below 2**31 for int32 indices)")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
