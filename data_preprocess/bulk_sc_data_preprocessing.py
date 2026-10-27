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
) -> dict:
    """Build the training store at ``data_path`` from ``sources``; returns a summary.

    ``vocab_path`` reuses an existing vocabulary; otherwise it is built from the
    sources' genes (``vocab_mode``: ``intersection`` or ``union``). Genes outside the
    vocabulary are left out of the store.
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
        for i, (source, reader) in enumerate(zip(sources, readers)):
            label = f"{i:03d}_{source.path.stem}"
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

            X = reader.rows(positions)[:, in_vocab]
            var = pd.DataFrame({GENE_ID: [vocab[g] for g in genes[in_vocab]]}, index=genes[in_vocab])
            tmp_path = transient / f"{label}.h5ad"
            ad.AnnData(X=X, obs=pd.DataFrame(index=obs.index[positions].astype(str)), var=var).write_h5ad(tmp_path)
            del X
            _load_into_collection(collection, tmp_path)
            tmp_path.unlink()

            modality = source.modality if source.modality is not None else default_modality
            obs_parts[label] = _store_obs(obs.iloc[positions], columns, modality, pb_id_column)
            summary.append({"source": str(source.path), "rows": int(len(positions)),
                            "genes_kept": int(in_vocab.sum()), "genes_total": int(len(genes))})
        shutil.rmtree(transient, ignore_errors=True)
    finally:
        for reader in readers:
            reader.close()

    if not obs_parts:
        raise ValueError("No rows selected from any source")
    # flatten() concatenates the memmaps in insertion order; collect obs the same way.
    obs = pd.concat([obs_parts[Path(name).name] for name in collection.fname_to_mmap.keys()])
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
    print(f"Store written to {data_path} ({len(obs)} rows, {len(vocab) - 2} genes in the vocabulary)")
    return {"sources": summary, "n_rows": int(len(obs)), "n_vocab_genes": len(vocab) - 2,
            "obs_columns": columns}


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
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
