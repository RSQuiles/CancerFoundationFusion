# BULK DATA
## Imports
import sys

sys.path.insert(0, "./")

from argparse import ArgumentParser
import h5py
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from bionemo.scdl.io.single_cell_collection import SingleCellCollection
from cancerfoundation.data.dataset import DatasetDir
from typing import Optional

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def h5_to_h5ad(
    bulk_dir: Path | str,
    obs_columns: list[
        str
    ],  # Metadata columns to include in obs.parquet (must exist in metadata.csv)
    chunk_size: int = 10_000,
    expr_key: str = "expression",  # Key in the h5 file where the expression matrix is stored
    existing_vocab_path: Optional[
        Path
    ] = None,  #  path to an existing vocab.json to reuse (set to None to generate from gene_list.txt)
):
    # --- INPUT: bulk data files (in the same directory) ---
    BULK_DIR = Path(bulk_dir)
    EXPRESSION_H5 = BULK_DIR / "expression.h5"
    GENE_LIST_TXT = BULK_DIR / "gene_list.txt"
    METADATA_CSV = BULK_DIR / "metadata.csv"

    # --- OUTPUT: corresponding h5ad files ---
    OUTPUT_DIR = BULK_DIR / "h5ads"

    # Checks
    assert EXPRESSION_H5.is_file(), f"Expression file not found: {EXPRESSION_H5}"
    assert GENE_LIST_TXT.is_file(), f"Gene list file not found: {GENE_LIST_TXT}"
    assert METADATA_CSV.is_file(), f"Metadata file not found: {METADATA_CSV}"
    if existing_vocab_path is not None:
        assert (
            existing_vocab_path.is_file()
        ), f"Existing vocab file not found: {existing_vocab_path}"
    metadata_columns = pd.read_csv(METADATA_CSV).columns
    for col_name in obs_columns:
        assert (
            col_name in metadata_columns
        ), f"{col_name} not found in metadata columns: {metadata_columns}"

    # Load gene list
    with open(GENE_LIST_TXT) as f:
        gene_names = [line.strip() for line in f if line.strip()]

    print(f"Genes: {len(gene_names)}")

    # Load metadata
    metadata = pd.read_csv(METADATA_CSV)
    print(f"Samples in metadata: {len(metadata)}")
    print(f"Columns: {list(metadata.columns)}")

    # Inspect expression matrix shape without loading it all
    with h5py.File(EXPRESSION_H5, "r") as f:
        assert (
            expr_key in list(f.keys())
        ), f"Expression key '{expr_key}' not found in h5 file. Available keys: {list(f.keys())}"
        print("Datasets in HDF5:", list(f.keys()))
        for key in f.keys():
            ds = f[key]
            if hasattr(ds, "shape"):
                print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

        # Determine the expression dataset key
        n_samples, n_genes_h5 = f[expr_key].shape
        print(f"\nExpression matrix: {n_samples} samples × {n_genes_h5} genes")
        assert (
            n_genes_h5 == len(gene_names)
        ), f"Gene count mismatch: h5 has {n_genes_h5}, gene_list.txt has {len(gene_names)}"

        # Convert expression file to chunked h5ad files
        h5ad_dir = OUTPUT_DIR
        h5ad_dir.mkdir(parents=True, exist_ok=True)

        # Map gene names to vocab IDs; drop genes not in vocab
        if existing_vocab_path is not None:
            with open(existing_vocab_path) as v:
                vocab = json.load(v)
            gene_ids = np.array([vocab.get(g, -1) for g in gene_names])
            valid_mask = gene_ids >= 0
            valid_gene_names = [g for g, v in zip(gene_names, valid_mask) if v]
            valid_gene_ids = gene_ids[valid_mask]
            valid_col_indices = np.where(valid_mask)[0]

            print(f"Genes in vocab: {len(valid_gene_names)}/{len(gene_names)}")

        # Use sample_id as index if available
        if "sample_id" in metadata.columns:
            metadata = metadata.set_index("sample_id", drop=False)

        expr_ds = f[expr_key]
        n_total = expr_ds.shape[0]
        n_chunks = (n_total + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_total)

            # Read chunk and select valid genes
            X_dense = (
                expr_ds[start:end, :][:, valid_col_indices]
                if existing_vocab_path is not None
                else expr_ds[start:end, :]
            )
            X_sparse = sp.csr_matrix(X_dense.astype(np.float32))

            obs_chunk = metadata.iloc[start:end][obs_columns].copy()
            obs_chunk.index = obs_chunk.index.astype(str)

            var = pd.DataFrame({GENE_ID: valid_gene_ids}, index=valid_gene_names)
            var[GENE_ID] = var[GENE_ID].astype(int)

            adata = ad.AnnData(X=X_sparse, obs=obs_chunk, var=var)

            fname = f"bulk_chunk_{chunk_idx:04d}.h5ad"
            adata.write_h5ad(h5ad_dir / fname)
            print(
                f"  [{chunk_idx+1}/{n_chunks}] wrote {fname}  ({end - start} samples)"
            )

    print(f"\nCreated {n_chunks} h5ad files in {h5ad_dir}")


def _generate_vocab_from_h5ads(
    h5ads: Path, cls_token: str, pad_token: str
) -> dict[str, int]:
    genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        var_names = sc.read_h5ad(path, backed="r").var_names
        genes.update(var_names)
    vocab = {gene: i for i, gene in enumerate([cls_token, pad_token] + list(genes))}
    return vocab


def _save_vocab_to_dir(vocab: dict[str, int], data_dir: DatasetDir) -> None:
    with open(data_dir.vocab_path, "w") as f:
        json.dump(vocab, f)


def _add_gene_id_to_h5ads(h5ads: Path, vocab: dict[str, int], data_path: Path) -> None:
    (data_path / "h5ads").mkdir()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        adata = sc.read_h5ad(path)
        adata.var[GENE_ID] = adata.var_names.map(vocab)
        adata = adata[:, ~adata.var[GENE_ID].isna()].copy()
        adata.var[GENE_ID] = adata.var[GENE_ID].astype(int)
        adata.write_h5ad(data_path / "h5ads" / path.name)


def convert_columns_to_categorical_with_mapping(df):
    df_copy = df.copy()

    category_mappings = {}

    df_categorical = pd.DataFrame(index=df.index)

    for column in df.columns:
        df_copy[column] = df_copy[column].astype("category")

        category_mappings[column] = dict(
            zip(
                df_copy[column].cat.categories,
                range(len(df_copy[column].cat.categories)),
            )
        )

        df_categorical[column] = df_copy[column].cat.codes

    return df_categorical, category_mappings


def main(args):
    # Checks
    if not args.from_bulk:
        assert (
            args.h5ad_path is not None
        ), "h5ad_path is required when --from_bulk is not set"
        h5ad_path = args.h5ad_path

    # Bulk preprocessing
    if args.from_bulk:
        assert (
            args.bulk_path is not None
        ), "bulk_path is required when --from_bulk is set"
        h5_to_h5ad(
            bulk_dir=args.bulk_path,
            obs_columns=args.obs_columns,
            expr_key=args.bulk_expr_key,
            existing_vocab_path=args.vocab_path,
        )
        h5ad_path = args.bulk_path / "h5ads"

    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabulary
    if args.vocab_path is None:
        vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    else:
        vocab = json.load(args.vocab_path.open())
    _save_vocab_to_dir(vocab, data_path)
    # Add gene IDs to h5ad files
    _add_gene_id_to_h5ads(h5ad_path, vocab, args.data_path)

    # Create and process memmap files
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(args.data_path / "h5ads", max_workers=12)

    # Collect observations
    obs_list = []
    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = sc.read_h5ad(h5ad_path / (fname.name + ".h5ad"), backed="r")

        # Extract tissue name from the filename (e.g., '..._kidney' -> 'kidney')
        # tissue_name = fname.name.split("_")[1]

        # Create the 'tissue' column and assign the extracted name to all cells
        # adata.obs["tissue"] = tissue_name

        obs_list.append(adata.obs[args.obs_columns].copy())

    obs = pd.concat(obs_list)

    obs, mapping = convert_columns_to_categorical_with_mapping(obs)

    obs.to_parquet(data_path.obs_path)

    with data_path.mapping_path.open("w") as f:
        json.dump(mapping, f, indent=4)

    # Flatten and create memmap dataset
    memmaps.flatten(data_path.memmap_path, destroy_on_copy=True)

    print("Conversion completed successfully.")

    # Remove duplicate metadata file that causes issues
    if os.path.isfile(data_path.memmap_path / "features/dataframe_00.parquet"):
        os.remove(data_path.memmap_path / "features/dataframe_00.parquet")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--from-bulk",
        action="store_true",
        help="Whether to run from the bulk data preprocessing pipeline",
    )

    parser.add_argument(
        "--h5ad-path",
        type=Path,
        required=False,
        help="Directory containing input h5ad files",
    )
    parser.add_argument(
        "--bulk-path",
        type=Path,
        required=False,
        help="Directory containing bulk data files (expression.h5, gene_list.txt, metadata.csv)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of samples per chunk when converting h5 to h5ad",
    )
    parser.add_argument(
        "--bulk-expr-key",
        type=str,
        default="expression",
        help="Name of expression dataset in bulk .h5 file",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Directory to save processed data (h5ads, vocab, memmaps)",
    )
    parser.add_argument(
        "--obs-columns",
        type=str,
        nargs="+",
        default=["sample_id", "tissue"],
        help="Metadata columns to include in obs.parquet (must exist in metadata.csv)",
    )
    parser.add_argument(
        "--vocab-path", type=Path, required=False, help="Path to existing vocab file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
