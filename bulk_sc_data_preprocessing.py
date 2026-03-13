# BULK DATA
## Imports
import sys
sys.path.insert(0, "./")

import h5py
import json
import os
import shutil
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

from cancerfoundation.data.dataset import DatasetDir

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"

# Inputs and general settings
from pathlib import Path

# --- INPUT: bulk data files (in the same directory) ---
# BULK_DIR = Path("/cluster/work/boeva/bulkFM/data/processed/archs4")
BULK_DIR = Path("tutorials/archs4_test")
EXPRESSION_H5 = BULK_DIR / "expression.h5"
GENE_LIST_TXT = BULK_DIR / "gene_list.txt"
METADATA_CSV = BULK_DIR / "metadata.csv"

# --- OUTPUT: pipeline-ready processed data ---
OUTPUT_DIR = BULK_DIR / "h5ads"

# Optional: path to an existing vocab.json to reuse (set to None to generate from gene_list.txt)
EXISTING_VOCAB_PATH = None

# Metadata columns to include in obs.parquet (must exist in metadata.csv)
OBS_COLUMNS = ["tissue", "series_id", "sample_id", "total_counts"]

# How many samples per h5ad chunk (controls memory during conversion)
CHUNK_SIZE = 500

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
    print("Datasets in HDF5:", list(f.keys()))
    # Adapt the dataset key if needed (common keys: "expression", "data/expression", "X")
    for key in f.keys():
        ds = f[key]
        if hasattr(ds, "shape"):
            print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

    # Determine the expression dataset key
    expr_key = "expression"  # adjust if your file uses a different key
    n_samples, n_genes_h5 = f[expr_key].shape
    print(f"\nExpression matrix: {n_samples} samples × {n_genes_h5} genes")
    assert n_genes_h5 == len(gene_names), (
        f"Gene count mismatch: h5 has {n_genes_h5}, gene_list.txt has {len(gene_names)}"
    )

# Convert expression file to chunked h5ad files
h5ad_dir = OUTPUT_DIR
h5ad_dir.mkdir(parents=True, exist_ok=True)

# Map gene names to vocab IDs; drop genes not in vocab
gene_ids = np.array([vocab.get(g, -1) for g in gene_names])
valid_mask = gene_ids >= 0
valid_gene_names = [g for g, v in zip(gene_names, valid_mask) if v]
valid_gene_ids = gene_ids[valid_mask]
valid_col_indices = np.where(valid_mask)[0]

print(f"Genes in vocab: {len(valid_gene_names)}/{len(gene_names)}")

# Use sample_id as index if available, else generate one
if "sample_id" in metadata.columns:
    metadata = metadata.set_index("sample_id", drop=False)

with h5py.File(EXPRESSION_H5, "r") as f:
    expr_ds = f[expr_key]
    n_total = expr_ds.shape[0]
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_total)

        # Read chunk and select valid genes
        X_dense = expr_ds[start:end, :][:, valid_col_indices]
        X_sparse = sp.csr_matrix(X_dense.astype(np.float32))

        obs_chunk = metadata.iloc[start:end][OBS_COLUMNS].copy()
        obs_chunk.index = obs_chunk.index.astype(str)

        var = pd.DataFrame({GENE_ID: valid_gene_ids}, index=valid_gene_names)
        var[GENE_ID] = var[GENE_ID].astype(int)

        adata = ad.AnnData(X=X_sparse, obs=obs_chunk, var=var)

        fname = f"bulk_chunk_{chunk_idx:04d}.h5ad"
        adata.write_h5ad(h5ad_dir / fname)
        print(f"  [{chunk_idx+1}/{n_chunks}] wrote {fname}  ({end - start} samples)")

print(f"\nCreated {n_chunks} h5ad files in {h5ad_dir}")

# H5AD PREPROCESING
from pathlib import Path
import sys

sys.path.insert(0, "./")

from cancerfoundation.data.dataset import DatasetDir
from argparse import ArgumentParser
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

import scanpy as sc
import pandas as pd
import json
import os

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--h5ad-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--vocab-path", type=Path, required=False)
    return parser.parse_args()


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
    if args is None:
        args = get_args()

    h5ad_path = args["h5ad_path"]
    columns = ["sample_id", "total_counts"]

    data_path = DatasetDir(args["data_path"])
    data_path.mkdir()
    # Generate and save vocabulary
    if "vocab_path" not in args:
        vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    else:
        vocab = json.load(args["vocab_path"].open())
    _save_vocab_to_dir(vocab, data_path)
    # Add gene IDs to h5ad files
    _add_gene_id_to_h5ads(h5ad_path, vocab, args["data_path"])

    # Create and process memmap files
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(args["data_path"] / "h5ads", max_workers=12)

    # Collect observations
    obs_list = []
    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = sc.read_h5ad(h5ad_path / (fname.name + ".h5ad"), backed="r")

        # Extract tissue name from the filename (e.g., '..._kidney' -> 'kidney')
        tissue_name = fname.name.split("_")[1]

        # Create the 'tissue' column and assign the extracted name to all cells
        # adata.obs["tissue"] = tissue_name

        obs_list.append(adata.obs[columns])

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

# Run Main
args = {
    "h5ad_path":Path("./tutorials/archs4_test/h5ads"),
    "data_path":Path("./tutorials/archs4_test/pipeline")
}

main(args)