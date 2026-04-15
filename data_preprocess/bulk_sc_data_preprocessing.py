# BULK DATA
## Imports
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

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


def normalize_bulk_tissues(bulk_metadata_path: Path, sc_path: Path) -> pd.DataFrame:
    print("Normalizing bulk tissue names...")

    # Retrieve at most 5 files from the sc_path directory
    sc_files = list(sc_path.glob("*.h5ad"))[:5]
    bulk_obs = pd.read_csv(bulk_metadata_path)
    adata_list = [read_anndata(f) for f in sc_files]

    # Match bulk tissue names to sc categories, and cluster unmatched one
    tissue_match = build_tissue_match(adata_list, bulk_obs)

    # Apply the mapping to the bulk metadata
    bulk_obs["tissue"] = (
        bulk_obs["tissue"].astype(str).map(tissue_match).fillna("unknown")
    )

    return bulk_obs


def build_tissue_match(
    adata, bulk_obs, min_cluster_size=5, n_clusters=30, max_features=5000
):
    def norm(s):
        s = str(s).lower().strip().replace('"', "")
        s = re.sub(r"[_/,+()-]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    # Create readable labels for each auto-cluster
    def cluster_name(items):
        stop = {
            "l",
            "r",
            "left",
            "right",
            "post",
            "anterior",
            "medial",
            "lateral",
            "normal",
            "responsive",
            "refractory",
            "disease",
            "treat",
            "treatment",
            "untreated",
            "treated",
            "sample",
            "hrs",
            "the",
            "and",
        }
        c = Counter()
        for x in items:
            c.update(
                t
                for t in re.findall(r"[a-z0-9]+", norm(x))
                if t not in stop and len(t) > 1
            )
        return "auto_" + ("_".join(t for t, _ in c.most_common(3)) if c else "misc")

    if isinstance(adata, list):
        sc_tissues = set()
        for ad in adata:
            sc_tissues.update(str(x) for x in ad.obs.tissue_general.dropna().unique())
        sc_tissues = sorted(sc_tissues)
    else:
        sc_tissues = [str(x) for x in adata.obs.tissue_general.dropna().unique()]
    sc_tissues_norm = {sc_t: norm(sc_t) for sc_t in sc_tissues}

    tissue_match = {sc_t: [] for sc_t in sc_tissues}
    no_match = []

    # First pass: direct matching to known tissues
    for t in bulk_obs.tissue.astype(str):
        t_norm = norm(t)
        matched = False
        for sc_t, sc_norm in sc_tissues_norm.items():
            if sc_norm in t_norm or t_norm in sc_norm:
                tissue_match[sc_t].append(t)
                matched = True
                break
        if not matched:
            no_match.append(t)

    # Second pass: cluster unmatched strings
    if no_match:
        unique_no_match = sorted(set(no_match))
        unique_no_match_norm = [norm(x) for x in unique_no_match]

        X = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
        ).fit_transform(unique_no_match_norm)

        n_clusters_eff = max(1, min(n_clusters, len(unique_no_match)))
        labels = MiniBatchKMeans(
            n_clusters=n_clusters_eff,
            random_state=0,
            batch_size=1024,
            n_init="auto",
        ).fit_predict(X)

        norm_to_cluster = {
            norm_text: lab for norm_text, lab in zip(unique_no_match_norm, labels)
        }

        clusters = defaultdict(list)
        for x in no_match:
            clusters[norm_to_cluster[norm(x)]].append(x)

        misc = []
        for items in clusters.values():
            if len(items) < min_cluster_size:
                misc.extend(items)
                continue
            name = cluster_name(items)
            base = name
            i = 2
            while name in tissue_match:
                name = f"{base}_{i}"
                i += 1
            tissue_match[name] = items

        if misc:
            tissue_match["auto_misc"] = misc

    tissue_match = {k: v for k, v in tissue_match.items() if len(v) > 0}
    assert len(bulk_obs) == sum(len(v) for v in tissue_match.values())

    # Transform into sets and invert the mapping
    tissue_match = {k: set(v) for k, v in tissue_match.items()}
    tissue_invert_match = {t: k for k, v in tissue_match.items() for t in v}

    return tissue_invert_match

def get_normalized_metadata_bulk(bulk_metadata_path: Path, meta_fields=["characteristics", "extract_protocol", "source_name", "title"]):
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
    
    def check_name(name, index, meta, meta_fields):
        for col, item in meta.loc[index, meta_fields].items():
            try:
                if re.search(name, item, re.IGNORECASE):
                    return True
            except Exception as e:
                print(f"Error checking index {index}")
        return False

    tissues = []
    for i in tqdm.tqdm(range(bulk_obs.shape[0])):
        tissues.append(walk_tissue_names(check_name, i, bulk_obs, meta_fields))

    # Include tissues in bulk_obs
    bulk_obs["tissue"] = tissues

    return bulk_obs


def h5_to_h5ad(
    bulk_dir: Path | str,
    obs_columns: list[
        str
    ],  # Metadata columns to include in obs.parquet (must exist in metadata.csv)
    normalize_tissues: bool = False,  # Whether to normalize bulk tissue names
    chunk_size: int = 10_000,
    expr_key: str = "expression",  # Key in the h5 file where the expression matrix is stored
    h5ad_dir: Optional[
        Path
    ] = None,  # Directory to save the generated h5ad files (defaults to bulk_dir/h5ads)
    existing_vocab_path: Optional[
        Path
    ] = None,  #  path to an existing vocab.json to reuse (set to None to generate from gene_list.txt)
):
    """
    Transform original .h5 file with counts in h5ad chunks.
    Applies CP10K and log1p normalization
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
        #metadata = normalize_bulk_tissues(METADATA_CSV, h5ad_dir)
    else:
        metadata = pd.read_csv(METADATA_CSV)

    # Ad-hoc column renaming to match single-cell OBS columns
    metadata = metadata.rename(columns={
        "tissue": "tissue_general",
        "instrument": "assay"
    })
    metadata_columns = list(metadata.columns)
        
    for col_name in obs_columns:
        assert (
            col_name in metadata_columns
        ), f"{col_name} not found in metadata columns: {metadata_columns}"
    
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

            # Store raw counts
            X_counts = X_dense.copy()

            # Library-size normalize (CP10K) to target_sum=1e4, then log1p
            library_size = X_dense.sum(axis=1, keepdims=True)
            library_size[library_size == 0] = 1.0
            X_dense = X_dense / library_size * 1e4
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
    h5ads: Path, cls_token: str, pad_token: str
) -> dict[str, int]:
    genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
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
    assert not (
        args.bulk_only and args.mixed
    ), "Cannot set both --bulk-only and --mixed"
    if not args.bulk_only:
        assert (
            args.h5ad_path is not None
        ), "h5ad_path is required when --bulk-only is not set"
        h5ad_path = args.h5ad_path

    # Bulk preprocessing
    if args.bulk_only or args.mixed:
        assert (
            args.bulk_path is not None
        ), "--bulk_path is required when --bulk-only or --mixed is set"

        h5ad_path = (
            args.bulk_path / "h5ads" if args.h5ad_path is None else args.h5ad_path
        )
        h5_to_h5ad(
            bulk_dir=args.bulk_path,
            obs_columns=args.obs_columns,
            normalize_tissues=args.normalize_bulk_tissues,
            chunk_size=args.chunk_size,
            expr_key=args.bulk_expr_key,
            h5ad_dir=h5ad_path,
            existing_vocab_path=args.vocab_path,
        )

    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabularyº
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
    obs_columns = args.obs_columns + ["modality"]  # Ensure modality is included
    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = read_anndata(h5ad_path / (fname.name + ".h5ad"))
        # Add modality column if not present
        if "modality" not in adata.obs.columns:
            adata.obs["modality"] = "sc"

        # Extract tissue name from the filename (e.g., '..._kidney' -> 'kidney')
        # tissue_name = fname.name.split("_")[1]

        # Create the 'tissue' column and assign the extracted name to all cells
        # adata.obs["tissue"] = tissue_name

        obs_list.append(adata.obs[obs_columns].copy())

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
    return parser.parse_args()


if __name__ == "__main__":
    # Note: by default, it only runs the single-cell data preprocessing
    args = get_args()
    main(args)
