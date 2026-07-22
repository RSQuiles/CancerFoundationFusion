import re
import sys
from utils import walk_tissue_names

sys.path.insert(0, "../")

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
from collections import Counter, defaultdict
import tqdm

from pseudobulk_paired_generation import generate_pseudobulk_chunks

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"

def read_anndata(file_path):
    # Make anndata readable
    with h5py.File(file_path, "r+") as f:
        if "uns" in f and "log1p" in f["uns"] and "base" in f["uns"]["log1p"]:
            del f["uns"]["log1p"]["base"]
            # print("Deleted /uns/log1p/base")
    
    adata = sc.read_h5ad(file_path)
    return adata


def get_normalized_metadata_bulk(
    bulk_metadata_path: Path, 
    meta_fields=["characteristics", "extract_protocol", "source_name", "title"],
    tissue_map_path: str | None = "/cluster/work/boeva/rquiles/data/sample_id_to_tissue.json"
    ):
    """
    Generate a metadata pd.Dataframe with a normalized tissue column from other specified fields.

    Args:
        metadata (str): The pd.Dataframe containing the metadata.
        meta_fields (list, optional): The list of metadata fields to search within.
            Defaults to ["geo_accession", "series_id", "characteristics_ch1", "extract_protocol_ch1", "source_name_ch1", "title"].

    Returns:
        pd.DataFrame: DataFrame containing the updated metadata.
    """
    print("Normalizing bulk tissue names...")

    bulk_obs = pd.read_csv(bulk_metadata_path)
    # print(bulk_obs.columns)

    tissue_map: dict[str, str] = {}
    if tissue_map_path is not None and Path(tissue_map_path).is_file():
        with open(tissue_map_path) as f:
            tissue_map = json.load(f)
        print(f"Loaded {len(tissue_map)} cached tissue labels")
    
    def check_name(name, index, meta, meta_fields):
        def normalize(s):
            s = str(s).lower()
            s = re.sub(r"[_\-/]", " ", s)
            s = re.sub(r"[^a-z0-9 ]", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        norm_name = normalize(name)
        words = norm_name.split()

        for col, item in meta.loc[index, meta_fields].items():
            try:
                norm_item = normalize(str(item))
                # Match if ANY word from the name appears in the text
                # For single-word names: direct substring match
                # For multi-word names: match if ALL words appear (in any order)
                if len(words) == 1:
                    if words[0] in norm_item:
                        return True
                else:
                    if all(w in norm_item for w in words):
                        return True
            except Exception:
                # print(f"Error checking index {index}, col {col}")
                continue
        return False

    # Index as sample_id, resolved once
    index_ids = bulk_obs["sample_id"].astype(str)
    tissues = index_ids.map(tissue_map)

    # Only the cache MISSES need inference
    missing_mask = tissues.isna()
    n_missing = int(missing_mask.sum())
    n_cached = len(tissues) - n_missing
    print(f"Tissues from cache: {n_cached}, to infer: {n_missing}")

    if n_missing > 0:
        missing_positions = np.where(missing_mask)[0]
        inferred = {}
        for i in tqdm.tqdm(missing_positions, miniters=1000, maxinterval=float("inf")):
            inferred[i] = walk_tissue_names(check_name, i, bulk_obs, meta_fields)
        # Fill the misses by position
        tissues = tissues.copy()
        tissues.iloc[missing_positions] = [inferred[i] for i in missing_positions]

    bulk_obs["tissue"] = tissues.values

    # Include tissues in bulk_obs
    bulk_obs["tissue"] = tissues

    return bulk_obs


def h5_to_h5ad(
    bulk_dir: Path | str,
    obs_columns: list[str],  # Metadata columns to include in obs.parquet (must exist in metadata.csv)
    normalize_tissues: bool = False,  # Whether to normalize bulk tissue names
    chunk_size: int = 10_000,
    expr_key: str = "expression",  # Key in the h5 file where the expression matrix is stored
    h5ad_dir: Optional[Path] = None,  # Directory to save the generated h5ad files (defaults to bulk_dir/h5ads)
    existing_vocab_path: Optional[Path] = None,  #  path to an existing vocab.json to reuse (set to None to generate from gene_list.txt)
    use_counts: bool = False # Wether to normalize the data
):
    """
    Transform original .h5 file with counts in h5ad chunks.
    Applies CPM and log1p normalization if use_counts == False
    """
    # --- INPUT: bulk data files (in the same directory) ---
    BULK_DIR = Path(bulk_dir)
    EXPRESSION_H5 = BULK_DIR / "expression.h5"
    GENE_LIST_TXT = BULK_DIR / "gene_list.txt"
    METADATA_CSV = BULK_DIR / "metadata.csv"

    # --- OUTPUT: corresponding h5ad files ---
    h5ad_dir = BULK_DIR / "h5ads" if h5ad_dir is None else h5ad_dir

    # Checks
    assert EXPRESSION_H5.is_file(), f"Expression file not found: {EXPRESSION_H5}"
    assert GENE_LIST_TXT.is_file(), f"Gene list file not found: {GENE_LIST_TXT}"
    assert METADATA_CSV.is_file(), f"Metadata file not found: {METADATA_CSV}"
    if existing_vocab_path is not None:
        assert (
            existing_vocab_path.is_file()
        ), f"Existing vocab file not found: {existing_vocab_path}"

    # Load gene list
    with open(GENE_LIST_TXT) as f:
        gene_names = [line.strip() for line in f if line.strip()]

    print(f"Genes: {len(gene_names)}")

    # Load metadata
    # Normalize bulk tissue names if requested
    if normalize_tissues:
        metadata = get_normalized_metadata_bulk(METADATA_CSV)
    else:
        metadata = pd.read_csv(METADATA_CSV)

    # Ad-hoc column renaming to match single-cell OBS columns
    metadata = metadata.rename(columns={
        "tissue": "tissue_general",
        "instrument": "assay"
    })
    metadata_columns = list(metadata.columns)
    
    available_obs_columns = []
    for col_name in obs_columns:
        if col_name not in metadata_columns:
            print(f"WARNING (bulk preprocessing): {col_name} not found in metadata columns: {metadata_columns}")
        else:
            available_obs_columns.append(col_name)
    obs_columns = available_obs_columns
    
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

            if not use_counts:
                # Library-size normalize (CP10K or CPM), then log1p
                library_size = X_dense.sum(axis=1, keepdims=True)
                library_size[library_size == 0] = 1.0
                X_dense = X_dense / library_size * 1e6
                X_dense = np.log1p(X_dense)

            X_sparse = sp.csr_matrix(X_dense.astype(np.float32))

            obs_chunk = metadata.iloc[start:end][obs_columns].copy()
            # Define modality
            obs_chunk["modality"] = "bulk"
            obs_chunk.index = obs_chunk.index.astype(str)

            var = (
                pd.DataFrame({GENE_ID: valid_gene_ids}, index=valid_gene_names)
                if existing_vocab_path is not None
                else pd.DataFrame(index=gene_names)
                # else pd.DataFrame({GENE_ID: range(len(gene_names))}, index=gene_names)
            )
            # Only cast GENE_ID if it exists
            if GENE_ID in var.columns:
                var[GENE_ID] = var[GENE_ID].astype(int)

            adata = ad.AnnData(X=X_sparse, obs=obs_chunk, var=var)

            fname = f"bulk_chunk_{chunk_idx:04d}.h5ad"
            adata.write_h5ad(h5ad_dir / fname)
            print(
                f"  [{chunk_idx+1}/{n_chunks}] wrote {fname}  ({end - start} samples)"
            )

    print(f"\nCreated {n_chunks} h5ad files in {h5ad_dir}")

def _set_h5ad_index(h5ads: Path, var_field: str = "feature_id") -> None:
    """
    Set the index of an h5ad file to a given field in adata.var
    """
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        adata = read_anndata(path)
        if var_field in adata.var.columns:
            adata.var_names = adata.var[var_field].astype(str)
            adata.write_h5ad(path)
            print(f"Updated {path.name} index to {var_field}")
        else:
            print(f"Skipped {path.name}")

def _generate_vocab_from_h5ads(
    h5ads: Path, cls_token: str, pad_token: str, intersect: bool=True
) -> dict[str, int]:
    if intersect:
        genes = None
    else:
        genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        if intersect:
            genes = var_names if genes is None else genes & var_names
        else:
            var_names = read_anndata(path).var_names
            genes.update(var_names)
    vocab = {gene: i for i, gene in enumerate([cls_token, pad_token] + list(genes))}
    return vocab


def _save_vocab_to_dir(vocab: dict[str, int], data_dir: DatasetDir) -> None:
    with open(data_dir.vocab_path, "w") as f:
        json.dump(vocab, f)


def _add_gene_id_to_h5ads(h5ads: Path, vocab: dict[str, int], data_path: Path) -> None:
    (data_path / "h5ads").mkdir(parents=True, exist_ok=True)
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        adata = read_anndata(path)
        adata.var[GENE_ID] = adata.var_names.map(vocab)
        n_before = adata.n_vars
        adata = adata[:, ~adata.var[GENE_ID].isna()].copy()
        n_after = adata.n_vars
        print(f"  {path.name}: {n_before} → {n_after} genes after vocab filter, {adata.n_obs} cells")
        adata.var[GENE_ID] = adata.var[GENE_ID].astype(int)
        adata.write_h5ad(data_path / "h5ads" / path.name)


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


def main(args):
    # Checks
    assert not (
        args.bulk_only and args.mixed
    ), "Cannot set both --bulk-only and --mixed"

    # Resolve h5ad_path: the staging directory where all h5ad files accumulate
    # before being converted to memory-mapped format.
    if args.bulk_only or args.mixed:
        # h5ad_path is set below after the bulk preprocessing block
        h5ad_path = None
    elif args.h5ad_path is not None:
        h5ad_path = args.h5ad_path
    elif args.paired_h5ad is not None:
        # Paired-only mode: no CellxGene h5ads, no ARCHS4 bulk.
        # Default the staging dir to a sibling of data_path.
        h5ad_path = args.data_path / "source_h5ads"
        h5ad_path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(
            "--h5ad-path is required when --bulk-only and --mixed are both unset "
            "and no --paired-h5ad is provided."
        )

    # Bulk preprocessing
    if args.bulk_only or args.mixed:
        assert (
            args.bulk_path is not None
        ), "--bulk_path is required when --bulk-only or --mixed is set"

        h5ad_path = args.bulk_path / "h5ads" if args.h5ad_path is None else args.h5ad_path
        h5_to_h5ad(
            bulk_dir=args.bulk_path,
            obs_columns=args.obs_columns,
            normalize_tissues=args.normalize_bulk_tissues,
            chunk_size=args.chunk_size,
            expr_key=args.bulk_expr_key,
            h5ad_dir=h5ad_path,
            existing_vocab_path=args.vocab_path,
            use_counts=args.use_counts
        )

    # Paired SC + bulk dataset preprocessing (generates pseudobulk chunks)
    if args.paired_h5ad is not None:
        print("Generating pseudobulk chunks from paired dataset...")
        generate_pseudobulk_chunks(
            input_h5ad=args.paired_h5ad,
            output_dir=h5ad_path,
            n_pseudobulk=args.n_pseudobulk,
            n_cells_per_pb=args.n_cells_per_pb,
            is_log1p=args.paired_is_log1p,
            extra_obs_columns=args.paired_extra_obs_columns,
            cell_line_col=args.paired_cell_line_col,
            domain_col=args.paired_domain_col,
            sc_label=args.paired_sc_label,
            bulk_label=args.paired_bulk_label,
            include_sc=args.include_paired_sc,
            include_bulk=args.include_paired_bulk,
            existing_vocab_path=args.vocab_path,
            min_cells=args.paired_min_cells,
            chunk_prefix="paired",
            seed=args.seed,
            use_counts=args.use_counts
        )

    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabulary
    if args.vocab_path is None:
        if args.update_index is not None:
            print("Setting h5ad index to gene names...")
            _set_h5ad_index(h5ad_path, var_field=args.update_index)
        
        print("Generating gene_vocabulary...")
        vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    else:
        print("Reading gene vocabulary")
        vocab = json.load(args.vocab_path.open())
    _save_vocab_to_dir(vocab, data_path)
    # Add gene IDs to h5ad files
    print("Adding gene ids...")
    _add_gene_id_to_h5ads(h5ad_path, vocab, args.data_path)

    # Create and process memmap files
    print("Creating memmap files...")
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(args.data_path / "h5ads", max_workers=12)

    # Collect observations
    print("Collecting observations...")
    obs_list = []
    obs_columns = (args.obs_columns or []) + ["modality"]  # Ensure modality is included
    # "paired" column links pseudobulk↔bulk rows; add it when paired data is present
    # so that BulkSCDataset can use it for paired sampling.
    if args.paired_h5ad is not None and "paired" not in obs_columns:
        obs_columns = obs_columns + ["paired"]

    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = read_anndata(h5ad_path / (fname.name + ".h5ad"))
        # Add modality column if not present
        if "modality" not in adata.obs.columns:
            adata.obs["modality"] = "sc"

        # Use reindex so that columns absent in this h5ad (e.g. "paired" for
        # CellxGene files, or "tissue_general" for paired-dataset files) are
        # filled with 0 rather than raising a KeyError.
        obs_list.append(adata.obs.reindex(columns=obs_columns, fill_value=0))

    obs = pd.concat(obs_list)

    obs, mapping = convert_columns_to_categorical_with_mapping(obs)

    print("Exporting files...")
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
        "--bulk-only",
        action="store_true",
        help="Whether to run from the bulk data preprocessing pipeline only",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Whether to prepare data for mixed training (with both bulk and sc samples)",
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
        "--normalize-bulk-tissues",
        action="store_true",
        help="Whether to normalize bulk tissue names",
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
        default=None,
        help="Metadata columns to include in obs.parquet (must exist in metadata.csv)",
    )
    parser.add_argument(
        "--update-index",
        type=str,
        default=None,
        help="var column to use to replace the AnnDatas' index"
    )
    parser.add_argument(
        "--vocab-path", type=Path, required=False, help="Path to existing vocab file"
    )

    # --- Paired SC+bulk dataset ---
    parser.add_argument(
        "--paired-h5ad",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to a paired h5ad file containing single-cell and bulk profiles "
            "identified by obs[cell_line_col] and obs[domain_col]. Pseudobulk "
            "samples are generated from the SC profiles and written alongside the "
            "bulk samples into h5ad_path."
        ),
    )
    parser.add_argument(
        "--n-pseudobulk",
        type=int,
        default=1,
        help="Pseudobulk samples to generate per cell_line (default: 10)",
    )
    parser.add_argument(
        "--n-cells-per-pb",
        type=int,
        default=None,
        help=(
            "Number of SC cells to aggregate per pseudobulk. "
            "Defaults to None (use all available cells)."
        ),
    )
    parser.add_argument(
        "--paired-is-log1p",
        action="store_true",
        help="Counts in the paired h5ad are log1p-transformed; expm1 before summing",
    )
    parser.add_argument(
        "--paired-cell-line-col",
        type=str,
        default="cell_line",
        help="obs column with the sample / pair identifier (default: cell_line)",
    )
    parser.add_argument(
        "--paired-domain-col",
        type=str,
        default="domain",
        help="obs column distinguishing SC from bulk profiles (default: domain)",
    )
    parser.add_argument(
        "--paired-sc-label",
        type=str,
        default="SC",
        help="Value in domain_col that identifies SC profiles (default: SC)",
    )
    parser.add_argument(
        "--paired-bulk-label",
        type=str,
        default="bulk",
        help="Value in domain_col that identifies bulk profiles (default: bulk)",
    )
    parser.add_argument(
        "--include-paired-sc",
        action="store_true",
        default=True,
        help="Include individual SC profiles from the paired dataset (default: True)",
    )
    parser.add_argument(
        "--no-paired-sc",
        dest="include_paired_sc",
        action="store_false",
        help="Exclude individual SC profiles from the paired dataset",
    )
    parser.add_argument(
        "--include-paired-bulk",
        action="store_true",
        default=True,
        help="Include bulk profiles from the paired dataset (default: True)",
    )
    parser.add_argument(
        "--no-paired-bulk",
        dest="include_paired_bulk",
        action="store_false",
        help="Exclude bulk profiles from the paired dataset",
    )
    parser.add_argument(
        "--paired-extra-obs-columns",
        type=str,
        nargs="+",
        default=None,
        help="Extra obs columns from the paired h5ad to carry into the output h5ads",
    )
    parser.add_argument(
        "--paired-min-cells",
        type=int,
        default=1,
        help="Minimum SC cells per cell_line to attempt pseudobulk generation (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pseudobulk sampling (default: 42)",
    )
    parser.add_argument(
        "--use-counts",
        action="store_true",
        help="Skip normalization and use counts"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Note: by default, it only runs the single-cell data preprocessing
    args = get_args()
    main(args)
