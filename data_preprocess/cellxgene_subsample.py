import gc
import scanpy as sc
import numpy as np
import anndata as ad
from pathlib import Path
import h5py

def read_anndata(file_path):
    # Make anndata readable
    with h5py.File(file_path, "r+") as f:
        if "uns" in f and "log1p" in f["uns"] and "base" in f["uns"]["log1p"]:
            del f["uns"]["log1p"]["base"]
            # print("Deleted /uns/log1p/base")
    
    adata = sc.read_h5ad(file_path)
    return adata

SAMPLE_COLS = [
    "dataset_id",
    "donor_id",
    "tissue_ontology_term_id",
    "disease_ontology_term_id",
    "assay_ontology_term_id",
    "suspension_type",
    "development_stage_ontology_term_id",
]


def assign_sample_id(adata: sc.AnnData, sample_cols: list[str] = SAMPLE_COLS) -> sc.AnnData:
    available_cols = [c for c in sample_cols if c in adata.obs.columns]
    adata.obs["sample_id"] = (
        adata.obs[available_cols]
        .astype(str)
        .agg("__".join, axis=1)
    )
    return adata


def subsample_by_sample(
    adata: sc.AnnData,
    fraction: float = 0.5,
    min_cells: int = 10,
    seed: int = 42,
) -> sc.AnnData:
    assert "sample_id" in adata.obs.columns
    rng = np.random.default_rng(seed)
    keep_indices = []
    for sample_id, group in adata.obs.groupby("sample_id"):
        n = len(group)
        if n <= min_cells:
            keep_indices.extend(group.index.tolist())
        else:
            n_keep = max(min_cells, int(np.ceil(n * fraction)))
            chosen = rng.choice(group.index, size=n_keep, replace=False)
            keep_indices.extend(chosen.tolist())
    return adata[keep_indices].copy()


def process_partitions(
    input_dir: str | Path,
    output_dir: str | Path,
    prefix: str = "partition",
    fraction: float = 0.5,
    min_cells: int = 10,
    max_output_cells: int = 200_000,
    seed: int = 42,
):
    """
    For each partition_n.h5ad in input_dir:
      1. Load the file
      2. Assign biological sample IDs
      3. Subsample to `fraction` of cells per sample
      4. Accumulate subsampled cells into output chunks of max `max_output_cells`
      5. Save each output chunk to output_dir as subsampled_0.h5ad, subsampled_1.h5ad, ...

    Args:
        input_dir:        directory containing partition_n.h5ad files
        output_dir:       directory to save subsampled output files
        prefix:           filename prefix to match (default: "partition")
        fraction:         fraction of cells to keep per biological sample
        min_cells:        minimum cells to keep per sample regardless of fraction
        max_output_cells: maximum cells per output file
        seed:             random seed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and sort all matching files
    files = sorted(input_dir.glob(f"{prefix}*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No files matching '{prefix}*.h5ad' in {input_dir}")
    print(f"Found {len(files)} partition files to process")

    buffer: list[sc.AnnData] = []
    buffer_size: int = 0
    output_idx: int = 0
    total_input: int = 0
    total_output: int = 0

    def flush_buffer():
        """Concatenate buffer and save to disk, then free memory."""
        nonlocal buffer, buffer_size, output_idx, total_output

        if not buffer:
            return

        print(f"  Writing output chunk {output_idx} ({buffer_size:,} cells)...")
        combined = ad.concat(buffer, join="outer", merge="same", index_unique=None)
        combined.obs_names_make_unique()

        out_path = output_dir / f"subsampled_{output_idx}.h5ad"
        combined.write_h5ad(out_path)
        print(f"  Saved → {out_path}")

        total_output += buffer_size
        output_idx += 1

        # Free memory
        del combined
        for a in buffer:
            del a
        buffer = []
        buffer_size = 0
        gc.collect()

    for file_idx, file_path in enumerate(files):
        print(f"[{file_idx + 1}/{len(files)}] Processing {file_path.name}...")

        # Load
        adata = read_anndata(file_path)
        n_input = len(adata)
        total_input += n_input

        # Assign sample IDs and subsample
        adata = assign_sample_id(adata, SAMPLE_COLS)
        adata = subsample_by_sample(adata, fraction=fraction, min_cells=min_cells, seed=seed)
        n_output = len(adata)
        print(f"  {n_input:,} → {n_output:,} cells ({n_output/n_input*100:.1f}% retained)")

        # Split this file's output if it would overflow the buffer
        start = 0
        while start < n_output:
            space_left = max_output_cells - buffer_size
            chunk = adata[start: start + space_left].copy()
            chunk_size = len(chunk)

            buffer.append(chunk)
            buffer_size += chunk_size
            start += chunk_size

            # Flush when buffer is full
            if buffer_size >= max_output_cells:
                flush_buffer()

        # Free the partition from memory
        del adata
        gc.collect()

    # Flush any remaining cells
    flush_buffer()

    print(f"\nDone.")
    print(f"  Total input cells:  {total_input:,}")
    print(f"  Total output cells: {total_output:,}")
    print(f"  Output files:       {output_idx}")
    print(f"  Overall retention:  {total_output/total_input*100:.1f}%")


# --- Usage ---
process_partitions(
    input_dir="/cluster/work/boeva/rquiles/data/cellxgene_full",
    output_dir="/cluster/work/boeva/rquiles/data/cellxgene_bulk",
    prefix="partition",
    fraction=0.5,
    min_cells=10,
    max_output_cells=200_000,
    seed=42,
)