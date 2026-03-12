"""Generate dummy h5ad files for testing the data processing pipeline.

Creates synthetic single-cell RNA-seq h5ad files with the structure expected
by data_processing.ipynb / scripts/h5ads_to_sc.py:
  - .X:         sparse count matrix (cells x genes)
  - .var_names: gene symbols
  - .obs:       metadata with columns: sample, cancer_type, technology

Filenames follow the convention {prefix}_{tissue}.h5ad so that the processing
pipeline can extract the tissue name from the filename.

Usage:
    python scripts/generate_dummy_h5ads.py              # defaults
    python scripts/generate_dummy_h5ads.py --out-dir ./DATA/test/raw_data/train \
        --n-cells 100 --n-genes 200 --n-files 4 --seed 42
"""

from argparse import ArgumentParser
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_h5ad(
    n_cells: int,
    gene_names: list[str],
    tissue: str,
    technology: str,
    rng: np.random.Generator,
) -> ad.AnnData:
    """Create one dummy AnnData with a sparse count matrix."""
    n_genes = len(gene_names)

    # ~30 % non-zero, integer counts in [1, 50)
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=rng)
    X.data = rng.integers(1, 50, size=X.data.shape).astype(np.float32)

    obs = pd.DataFrame(
        {
            "sample": [f"sample_{i % 3}" for i in range(n_cells)],
            "cancer_type": rng.choice(["TypeA", "TypeB"], size=n_cells).tolist(),
            "technology": [technology] * n_cells,
        },
        index=[f"cell_{tissue}_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=X, obs=obs, var=var)


def generate(
    out_dir: Path,
    n_cells: int = 50,
    n_genes: int = 100,
    n_files: int = 2,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    tissues = ["brain", "lung", "kidney", "liver"]
    technologies = ["10X", "SmartSeq2"]

    for i in range(n_files):
        tissue = tissues[i % len(tissues)]
        tech = technologies[i % len(technologies)]
        adata = make_h5ad(n_cells, gene_names, tissue, tech, rng)

        # Filename convention: {prefix}_{tissue}.h5ad
        fname = f"dummy_{tissue}.h5ad"
        path = out_dir / fname
        adata.write_h5ad(path)
        print(f"  wrote {path}  ({n_cells} cells, {n_genes} genes, tech={tech})")

    print(f"\nGenerated {n_files} h5ad files in {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate dummy h5ad files for testing")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./DATA/test/raw_data/train"),
        help="Directory to write h5ad files to",
    )
    parser.add_argument("--n-cells", type=int, default=50, help="Cells per file")
    parser.add_argument("--n-genes", type=int, default=100, help="Number of genes")
    parser.add_argument("--n-files", type=int, default=2, help="Number of h5ad files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate(args.out_dir, args.n_cells, args.n_genes, args.n_files, args.seed)
