# scbFM Pretraining Data Pipeline

This document summarizes how the three pretraining data modalities — **single-cell**,
**pseudobulk**, and **real bulk** — are generated, and what fields to expect in the
resulting `.h5ad` files.

All three producer scripts emit **raw counts**. Tokenization (gene reindexing,
min-gene filtering, scGPT-style quantile binning) is a separate downstream step
(`preprocess_raw_h5ad.py`), described at the end.

---

## Common foundations

Every pipeline shares the same spine:

| Aspect | Value |
|---|---|
| **Gene axis** | `data/gene_list.txt` (ordered Ensembl IDs). All outputs reindex/align to this exact list, so `var_names` is identical across every file. |
| **`var`** | Empty `DataFrame` whose index is named `ensembl_id`. |
| **Quality filter** | `MIN_GENES = 200` — a sample is dropped unless ≥200 genes are non-zero. |
| **`X`** | Raw counts, sparse CSR, `float32`. No normalization/binning at this stage. |
| **Sampling weighting** | `√(availability)` weighting to avoid large datasets/contexts dominating; reservoir sampling for a fixed memory footprint. |

---

## 1. Single-cell

**Script:** `create_pretrain_data_sc_RAW.py`
**Output:** `pretraining_sc_RAW.h5ad`
**Source:** CELLxGENE Census (`2025-11-08`), human, `is_primary_data == True` cells only.

### Pipeline
1. **Count** cells per `(dataset_id, tissue_general, cell_type)` group.
2. **Allocate quotas** toward `TARGET_TOTAL_CELLS` (default 642,406): split across datasets
   by √-weighting, then across tissue×cell_type groups within each dataset.
3. **Reservoir-sample** cell IDs (`soma_joinid`) to fill each group quota → `sampling_plan_RAW.csv`.
4. **Download** expression in chunks of 5,000 cells, reorder genes to `gene_list`,
   re-apply the ≥200-gene filter per batch, truncate to the exact target.
5. **Merge** chunks pairwise into the final file.
6. **Overdraw logic:** samples 1.10× the target and doubles the factor across up to 4
   attempts, since the min-gene filter drops cells after download.

### `obs` fields
`soma_joinid`, `dataset_id`, `cell_type`, `tissue_general`, `donor_id`,
`dataset` (= `"CELLxGENE_Census"`).
Index (`cell_id`) = `cellxgene:<soma_joinid>`.

---

## 2. Pseudobulk

**Script:** `create_pseudo_bulk_data_RAW.py`
**Output:** `pseudo_bulk/pseudo_bulk_RAW.h5ad`
**Source:** CELLxGENE Census, with single cells **aggregated (summed) into synthetic bulk samples**.

### Pipeline
1. **Define contexts** = `(dataset_id, donor_id, tissue_general)`. Count cell types per
   context; keep only *eligible* contexts: ≥2 cell types (each with ≥20 cells) and
   ≥100 total cells → `eligible_contexts.csv`.
2. **Simulate a plan** for `TARGET_PSEUDO_BULKS` (default 20,000) samples of
   `CELLS_PER_PSEUDO_BULK` (default 1,000) cells each:
   - Pick a context (√-weighted).
   - With prob 0.5 (`SPARSE_SAMPLE_PROB`), drop a random subset of cell types (realistic
     sparse mixtures).
   - Draw random cell-type proportions → `rng.multinomial` gives per-cell-type counts.
   - Saved to `pseudo_bulk_sampling_plan.csv`.
3. **Compute source-cell pool quotas** per (context, cell_type) and **reservoir-sample**
   real cells to fill them (`source_cell_pool_quotas.csv`, `sampled_source_cells.csv`).
4. **Download** those source cells once, aligned to `gene_list`
   (`sampled_source_cells_aligned.h5ad`).
5. **Generate** each pseudobulk: per planned sample, randomly select the required cells
   per cell type and **sum their raw counts** into one row. Apply ≥200-gene filter, merge.

### `obs` fields
`dataset` (= `"CELLxGENE_Census_pseudobulk"`), `dataset_id`, `donor_id`, `tissue_general`,
`total_cells`, `n_cell_types`, and **one `prop__<cell_type>` column per cell type**
(`float32` proportion, `0.0` where absent).
Index (`sample_id`) = `pseudo_bulk:000123`.

### `uns` fields (unique to this file)
- `cell_type_proportion_cell_types` — sorted list of all cell types
- `cell_type_proportion_obs_columns` — matching `prop__*` column names
- `cell_type_proportion_columns` — dict mapping cell_type → column name

> Written with HDF5 `libver="earliest"` for old-cluster HDF5 (<1.10) compatibility.

---

## 3. Real bulk

**Script:** `create_pretrain_data_bulk_RAW.py`
**Outputs:** `pretraining_bulk_RAW.h5ad`, `preadapt_bulk_RAW.h5ad`
**Source:** GTEx + ARCHS4 (pre-aligned to `gene_list` by upstream scripts; the loader
validates gene order and errors if not).

### Pipeline
1. **Validate** both inputs are already gene-aligned.
2. Apply the ≥200-gene filter to both.
3. **ARCHS4-specific filters:**
   - Drop single-cell-like samples (`singlecellprobability ≥ 0.5` from the metadata HDF5).
   - **Leakage removal:** scan all obs metadata text for GTEx donor IDs (`GTEX-xxxx`) and
     for downstream-benchmark identifiers (TCGA, DepMap/CCLE, GDSC, DiSignAtlas, LINCS/L1000);
     matched rows are excluded so they cannot contaminate pretraining. Hits logged to
     `archs4_gtex_donor_hits_RAW.csv` / `archs4_downstream_hits_RAW.csv`.
4. **Split** surviving samples 90/10 (`PRETRAIN_FRACTION=0.9`, seed 42) into
   pretraining vs. "preadapt".
5. **Stream** rows into resizable CSR datasets (memory-bounded, `lzf` compression).

### `obs` fields
The original GTEx/ARCHS4 obs columns (heterogeneous — outer-joined, so source-specific
columns are `NaN` for the other source), plus:
- `dataset` (= `"GTEx"` or `"ARCHS4"`)
- `bulk_source` (same as `dataset`)
- `bulk_dataset` (= `"pretraining_bulk_RAW"` or `"preadapt_bulk_RAW"`)

Original sample IDs are kept as the index.

> **No tissue harmonization.** Unlike the single-cell/pseudobulk pipelines, there is no
> `tissue_general` field. GTEx rows carry GTEx's native tissue columns (e.g. `SMTS`/`SMTSD`)
> if the upstream `gtex.h5ad` preserved them; ARCHS4 rows carry unharmonized free-text
> metadata. The only columns guaranteed present for every bulk row are `dataset`,
> `bulk_source`, `bulk_dataset`.

---

## Tokenization (downstream, shared)

**Script:** `preprocess_raw_h5ad.py` → `src/preprocess.py::preprocess_raw_h5ad`

The `*_RAW.h5ad` files hold raw counts. Before pretraining, each is passed through:
1. **Reindex** to `gene_list` (missing genes zero-filled; `.missing_genes.json` sidecar written).
2. Re-apply the **min-genes** filter.
3. **scGPT-style quantile binning:** token `0` reserved for true zeros; nonzero values
   quantile-binned into `1..bin_num` (default 10). `X` becomes **`uint8` tokens**.

Adds `uns["scbfm_preprocess"]` recording: `input_path`, `original_shape`, `output_shape`,
`gene_list_path`, `missing_genes`, `min_genes`, `bin_num`,
`binning_strategy = "scgpt_nonzero_quantile"`, `zero_token_reserved = True`,
`output_dtype = "uint8"`.

---

## Quick comparison

| | Single-cell | Pseudobulk | Bulk |
|---|---|---|---|
| **Source** | CELLxGENE Census | CELLxGENE (summed cells) | GTEx + ARCHS4 |
| **Output** | `pretraining_sc_RAW.h5ad` | `pseudo_bulk_RAW.h5ad` | `pretraining_bulk_RAW.h5ad`, `preadapt_bulk_RAW.h5ad` |
| **`X`** | raw counts, CSR f32 | summed raw counts, CSR f32 | raw counts, CSR f32 |
| **Key `obs`** | dataset_id, cell_type, tissue_general, donor_id | dataset_id, donor_id, tissue_general, total_cells, n_cell_types, `prop__*` | original GTEx/ARCHS4 cols + dataset, bulk_source, bulk_dataset |
| **Tissue field** | `tissue_general` (CELLxGENE ontology) | `tissue_general` (context key) | none (native GTEx/ARCHS4 columns only) |
| **`uns`** | — | cell-type proportion maps | — |
| **Leakage filtering** | `is_primary_data` only | `is_primary_data` only | explicit TCGA/DepMap/GDSC/LINCS + GTEx-donor removal |
| **`var`** | gene_list (`ensembl_id`) | same | same |