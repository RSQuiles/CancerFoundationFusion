"""Add a `modality` column to all h5ad files in a directory matching a prefix.

Useful for backfilling the `modality` obs column on already-generated h5ad files
(e.g. subsampled CellxGene chunks) so downstream preprocessing/plotting can read
it directly from the file.

Usage:
    python add_modality.py --dir /path/to/h5ads --prefix subsampled --modality sc
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import scanpy as sc


def read_anndata(file_path: Path) -> sc.AnnData:
    # Strip a stray log1p base entry that can make some CellxGene h5ads unreadable.
    with h5py.File(file_path, "r+") as f:
        if "uns" in f and "log1p" in f["uns"] and "base" in f["uns"]["log1p"]:
            del f["uns"]["log1p"]["base"]
    return sc.read_h5ad(file_path)


def add_modality(directory: Path, prefix: str, modality: str, overwrite: bool) -> None:
    files = sorted(directory.glob(f"{prefix}*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No files matching '{prefix}*.h5ad' in {directory}")
    print(f"Found {len(files)} file(s) matching '{prefix}*.h5ad'")

    for i, path in enumerate(files):
        adata = read_anndata(path)
        if "modality" in adata.obs.columns and not overwrite:
            print(
                f"  [{i + 1}/{len(files)}] {path.name}: 'modality' already present, "
                f"skipping (use --overwrite to force)"
            )
            continue

        adata.obs["modality"] = modality
        adata.write_h5ad(path)
        print(f"  [{i + 1}/{len(files)}] {path.name}: set modality='{modality}' ({adata.n_obs} cells)")

    print("Done.")


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing the h5ad files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Only process files whose name starts with this prefix",
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="Value to assign to the 'modality' obs column (e.g. sc, bulk, pseudobulk)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing 'modality' column instead of skipping the file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    add_modality(args.dir, args.prefix, args.modality, args.overwrite)
