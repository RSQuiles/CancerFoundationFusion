"""
Generate pseudobulk samples from a paired single-cell / bulk h5ad dataset.

Input h5ad format:
  obs[cell_line_col] : sample identifier (pair key)
  obs[domain_col]    : domain label ("SC" or "bulk" by default, configurable)

For each cell_line that has both SC and bulk profiles:
  - n_pseudobulk pseudobulk samples are built by randomly drawing
    n_cells_per_pb SC profiles (or all of them if n_cells_per_pb is None)
    and summing counts. If the data is log1p-transformed, counts are
    exponentiated before summing (expm1).
  - The pseudobulk rows and their matching bulk row share the same integer
    "paired" obs value so that BulkSCSampler can align them in paired batches.

All output samples are CP10K + log1p normalized before writing.

Output h5ad files are written to output_dir with obs columns:
  "modality"        : "sc" | "pseudobulk" | "bulk"
  cell_line_col     : original cell_line identifier
  "paired"          : 0 (unpaired) or integer pair ID (links PB ↔ bulk)
  extra_obs_columns : any additional columns requested
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


GENE_ID = "_cf_gene_id"


def cpm_to_cp10k_log1p(X: np.ndarray) -> np.ndarray:
    """CP10K-normalize each row then apply log1p. Input must be a dense 2-D array."""
    return np.log1p(X / 100)


def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def generate_pseudobulk_chunks(
    input_h5ad: Path | str,
    output_dir: Path | str,
    n_pseudobulk: int = 10,
    n_cells_per_pb: Optional[int] = None,
    is_log1p: bool = False,
    extra_obs_columns: Optional[list[str]] = None,
    cell_line_col: str = "cell_line",
    domain_col: str = "domain",
    sc_label: str = "SC",
    bulk_label: str = "bulk",
    include_sc: bool = True,
    include_bulk: bool = True,
    existing_vocab_path: Optional[Path | str] = None,
    min_cells: int = 1,
    chunk_prefix: str = "paired",
    seed: int = 42,
    flush_every: int = 50,
) -> None:
    """
    Read a paired h5ad and write pseudobulk + bulk (+ optionally SC) h5ad files
    to output_dir for inclusion in the main preprocessing pipeline.
    The data is considered to be CPM by default. It is translated to CP10K and
    log1p to match the other model inputs.

    Parameters
    ----------
    input_h5ad : Path
        Input h5ad with obs[cell_line_col] and obs[domain_col].
    output_dir : Path
        Directory where output h5ad files are written.
    n_pseudobulk : int
        Number of pseudobulk samples to generate per cell_line (default 10).
    n_cells_per_pb : int | None
        Number of SC cells to aggregate per pseudobulk. None means all cells
        in the cell_line are used (with replacement when n_pseudobulk > 1).
    is_log1p : bool
        If True, counts are log1p-transformed in the h5ad; expm1 is applied
        before summing so that pseudobulk aggregation operates on raw counts.
    extra_obs_columns : list[str] | None
        Additional obs columns from the input to carry through to output.
        Columns absent in the input are silently skipped.
    cell_line_col : str
        obs column containing the sample / pair identifier.
    domain_col : str
        obs column distinguishing SC from bulk profiles.
    sc_label : str
        Value in domain_col for single-cell profiles.
    bulk_label : str
        Value in domain_col for bulk profiles.
    include_sc : bool
        If True, individual SC profiles are written to output (modality="sc").
    include_bulk : bool
        If True, bulk profiles are written to output (modality="bulk").
    existing_vocab_path : Path | None
        Path to an existing vocab.json. Genes absent from the vocab are
        dropped and the _cf_gene_id var column is populated from the vocab.
    min_cells : int
        Minimum SC cells a cell_line must have to generate pseudobulk.
        Cell_lines below this threshold have their SC cells skipped for PB.
    chunk_prefix : str
        Filename prefix for output h5ad files.
    seed : int
        Random seed for reproducible pseudobulk sampling.
    """
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    extra_obs_columns = list(extra_obs_columns) if extra_obs_columns else []

    print(f"Reading {input_h5ad} ...")
    adata = ad.read_h5ad(input_h5ad)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    for col in [cell_line_col, domain_col]:
        assert col in adata.obs.columns, (
            f"Column '{col}' not found in obs. Available: {list(adata.obs.columns)}"
        )

    # --- Optional gene filtering against an existing vocab ---
    gene_ids: Optional[np.ndarray] = None
    if existing_vocab_path is not None:
        with open(existing_vocab_path) as f:
            vocab = json.load(f)
        gene_id_arr = np.array([vocab.get(g, -1) for g in adata.var_names])
        valid_mask = gene_id_arr >= 0
        adata = adata[:, valid_mask].copy()
        gene_ids = gene_id_arr[valid_mask].astype(int)
        print(f"  Genes matched in vocab: {int(valid_mask.sum())}/{len(valid_mask)}")

    n_genes = adata.n_vars

    # --- Split by domain ---
    domain_vals = adata.obs[domain_col].astype(str).values
    sc_mask_arr  = np.isin(domain_vals, [sc_label, ""]) # Assume unlabelled samples to be sc
    bulk_mask_arr = domain_vals == bulk_label

    print(f"  SC cells: {sc_mask_arr.sum()}  Bulk samples: {bulk_mask_arr.sum()}")

    cell_line_vals = adata.obs[cell_line_col].astype(str).values
    sc_cls   = set(cell_line_vals[sc_mask_arr])
    bulk_cls = set(cell_line_vals[bulk_mask_arr])
    paired_cls = sc_cls & bulk_cls
    all_cls    = sorted(sc_cls | bulk_cls)
    print(f"  Unique cell_lines: {len(all_cls)}  Paired (SC+bulk): {len(paired_cls)}")

    # Assign stable pair IDs: 0 = unpaired, k ≥ 1 = paired
    pair_id_map: dict[str, int] = {}
    counter = 1
    for cl in sorted(paired_cls):
        pair_id_map[cl] = counter
        counter += 1

    # --- Build var DataFrame ---
    var = pd.DataFrame(index=list(adata.var_names))
    if gene_ids is not None:
        var[GENE_ID] = gene_ids.astype(int)

    # Pre-extract obs arrays for fast row slicing
    # X_all = _to_dense(adata.X)  # (n_obs, n_genes)
    extra_present = [c for c in extra_obs_columns if c in adata.obs.columns]

    # --- Accumulate output rows per modality ---
    sc_X_parts:   list[np.ndarray]    = []
    sc_obs_parts:  list[pd.DataFrame] = []
    pb_X_parts:   list[np.ndarray]    = []
    pb_obs_parts:  list[pd.DataFrame] = []
    bulk_X_parts: list[np.ndarray]    = []
    bulk_obs_parts: list[pd.DataFrame] = []

    file_idx = [0]

    print("Working on cell lines:")
    for i, cl in enumerate(all_cls):
        print(f"- {cl}")
        cl_mask  = cell_line_vals == cl
        sc_sel   = cl_mask & sc_mask_arr
        bulk_sel = cl_mask & bulk_mask_arr
        pair_id  = pair_id_map.get(cl, 0)

        # ---- Single-cell profiles ----
        if include_sc and sc_sel.any():
            X_sc = _to_dense(adata.X[sc_sel])
            if is_log1p:
                X_sc = np.expm1(X_sc)
            sc_X_parts.append(cpm_to_cp10k_log1p(X_sc).astype(np.float32))
            n_sc_rows = int(sc_sel.sum())
            obs_dict: dict = {
                "modality":   ["sc"] * n_sc_rows,
                cell_line_col: [cl]  * n_sc_rows,
                "paired":      [pair_id]   * n_sc_rows,
            }
            for col in extra_present:
                obs_dict[col] = adata.obs[col].values[sc_sel].tolist()
            sc_obs_parts.append(pd.DataFrame(obs_dict))

        # ---- Pseudobulk profiles ----
        n_sc = int(sc_sel.sum())
        if n_sc >= min_cells:
            X_sc_cnt = _to_dense(adata.X[sc_sel])
            if is_log1p:
                X_sc_cnt = np.expm1(X_sc_cnt)

            n_draw = n_cells_per_pb if n_cells_per_pb is not None else n_sc
            pb_X = np.zeros((n_pseudobulk, n_genes), dtype=np.float64)
            for pb_i in range(n_pseudobulk):
                idxs = rng.choice(n_sc, size=n_draw, replace=(n_draw > n_sc))
                pb_X[pb_i] = X_sc_cnt[idxs].sum(axis=0)

            pb_X_parts.append(cpm_to_cp10k_log1p(pb_X).astype(np.float32))
            obs_dict = {
                "modality":    ["pseudobulk"] * n_pseudobulk,
                cell_line_col: [cl]           * n_pseudobulk,
                "paired":      [pair_id]      * n_pseudobulk,
            }
            # Use first SC obs value as representative metadata for pseudobulks
            first_sc_idx = int(np.where(sc_sel)[0][0])
            for col in extra_present:
                obs_dict[col] = [adata.obs[col].values[first_sc_idx]] * n_pseudobulk
            pb_obs_parts.append(pd.DataFrame(obs_dict))
        elif n_sc > 0:
            print(f"  [skip PB] {cl}: {n_sc} SC cells < min_cells={min_cells}")

        # ---- Bulk profiles ----
        if include_bulk and bulk_sel.any():
            X_bulk = _to_dense(adata.X[bulk_sel])
            if is_log1p:
                X_bulk = np.expm1(X_bulk)
            bulk_X_parts.append(cpm_to_cp10k_log1p(X_bulk).astype(np.float32))
            n_bulk_rows = int(bulk_sel.sum())
            obs_dict = {
                "modality":    ["bulk"]   * n_bulk_rows,
                cell_line_col: [cl]       * n_bulk_rows,
                "paired":      [pair_id]  * n_bulk_rows,
            }
            for col in extra_present:
                obs_dict[col] = adata.obs[col].values[bulk_sel].tolist()
            bulk_obs_parts.append(pd.DataFrame(obs_dict))

        # --- Write h5ad output files ---
        # Flush periodically to avoid accumulating too much in memory

        def _flush(X_parts, obs_parts, label: str) -> None:
            if not X_parts:
                return
            X_cat = np.concatenate(X_parts, axis=0)
            obs_cat = pd.concat(obs_parts, ignore_index=True)
            obs_cat.index = obs_cat.index.astype(str)
            adata_out = ad.AnnData(X=sp.csr_matrix(X_cat), obs=obs_cat, var=var.copy())
            fname = f"{chunk_prefix}_{label}_chunk_{file_idx[0]:04d}.h5ad"
            adata_out.write_h5ad(output_dir / fname)
            print(f"  [{label}] wrote {fname} ({X_cat.shape[0]} rows)")
            file_idx[0] += 1

        is_last = (i + 1) == len(all_cls)
        if (i + 1) % flush_every == 0 or is_last:
            print(f"  Flushing after {i + 1}/{len(all_cls)} cell lines...")
            _flush(sc_X_parts,   sc_obs_parts, "sc")
            _flush(pb_X_parts,   pb_obs_parts, "pb")
            _flush(bulk_X_parts, bulk_obs_parts, "bulk")

            # Ensure clearing of information
            sc_X_parts = []
            sc_obs_parts = []
            pb_X_parts = []
            pb_obs_parts = []
            bulk_X_parts = []
            bulk_obs_parts = []

    print(f"Done: {file_idx[0]} file(s) written to {output_dir}")


def _get_args():
    parser = ArgumentParser(
        description="Generate pseudobulk h5ads from a paired SC+bulk h5ad file."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True,
                        help="Paired h5ad with obs[cell_line_col] and obs[domain_col]")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where output h5ad files are written")
    parser.add_argument("--n-pseudobulk", type=int, default=10,
                        help="Pseudobulk samples per cell_line (default: 10)")
    parser.add_argument("--n-cells-per-pb", type=int, default=500,
                        help="SC cells per pseudobulk (default: all cells)")
    parser.add_argument("--is-log1p", action="store_true",
                        help="Counts are log1p-transformed; expm1 before summing")
    parser.add_argument("--cell-line-col", type=str, default="cell_line",
                        help="obs column with pair identifier (default: cell_line)")
    parser.add_argument("--domain-col", type=str, default="domain",
                        help="obs column with domain label (default: domain)")
    parser.add_argument("--sc-label", type=str, default="SC",
                        help="SC value in domain_col (default: SC)")
    parser.add_argument("--bulk-label", type=str, default="bulk",
                        help="Bulk value in domain_col (default: bulk)")
    parser.add_argument("--no-sc", action="store_true",
                        help="Do not write individual SC profiles to output")
    parser.add_argument("--no-bulk", action="store_true",
                        help="Do not write bulk profiles to output")
    parser.add_argument("--extra-obs-columns", type=str, nargs="+", default=None,
                        help="Extra obs columns to carry to output h5ads")
    parser.add_argument("--vocab-path", type=Path, default=None,
                        help="Existing vocab.json to filter genes and populate _cf_gene_id")
    parser.add_argument("--min-cells", type=int, default=1,
                        help="Minimum SC cells per cell_line for pseudobulk (default: 1)")
    parser.add_argument("--chunk-prefix", type=str, default="paired",
                        help="Output filename prefix (default: paired)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    generate_pseudobulk_chunks(
        input_h5ad=args.input_h5ad,
        output_dir=args.output_dir,
        n_pseudobulk=args.n_pseudobulk,
        n_cells_per_pb=args.n_cells_per_pb,
        is_log1p=args.is_log1p,
        extra_obs_columns=args.extra_obs_columns,
        cell_line_col=args.cell_line_col,
        domain_col=args.domain_col,
        sc_label=args.sc_label,
        bulk_label=args.bulk_label,
        include_sc=not args.no_sc,
        include_bulk=not args.no_bulk,
        existing_vocab_path=args.vocab_path,
        min_cells=args.min_cells,
        chunk_prefix=args.chunk_prefix,
        seed=args.seed,
    )
