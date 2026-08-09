# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

CancerFoundation is a PyTorch Lightning-based Transformer foundation model for single-cell RNA-seq gene expression prediction, developed as part of the **master thesis of Rafael Quiles** in the **Boeva Lab, D-INFK, ETH Zürich**. It is a fork of an earlier codebase developed by **Alexander Theus**. Large parts of the architecture are based on [scGPT](https://github.com/bowang-lab/scGPT).

## Commands

### Training
```bash
python pretrain.py \
    --gpus 1 \
    --save-dir ./save/experiment_name \
    --train-path ./DATA/brain/processed_data/train \
    --epochs 15 \
    --batch-size 16
```

See `debug.sh` for a complete example with all common parameters. Key parameters:
- `--training-tasks`: "pcpt" (masked prediction), "gen" (generation), or "both"
- `--do-mvc`: Enable Masked Value prediction for Cell embeddings
- `--input-emb-style`: "mine" or "theirs" (different value encoding strategies)
- `--precision`: "32", "16-mixed", or "bf16-mixed"
- `--conditions`: Metadata column(s) to condition on (e.g., `technology`)
- `--gen-method`: "theirs", "mine", "orig", or "quick" (generative training strategy)
- `--compile`: Enable `torch.compile` for the model
- `--unified`: Enable Unified FM mode (adds bulk data, contrastive, and aggregation losses)

### Post-training Analysis (UMAPs + unified metrics + benchmark plot)
```bash
# Inspect the exact commands without running anything
python evaluate/run_analysis.py --config evaluate/example_analysis_config.yaml --dry-run

# Submit (one SLURM job per experiment, or one job for all — set slurm.mode)
python evaluate/run_analysis.py --config evaluate/example_analysis_config.yaml

# Subset by experiment and/or step
python evaluate/run_analysis.py --config <cfg> --only monitor_align --step umap benchmark
```
One config drives N ablation directories. Steps: `build`, `metrics`, `scib`,
`diagnose`, `umap`, `paired_umap`, `downstream`, `benchmark`. **They do not all run in
the same environment** — `build`/`metrics`/`umap`/`downstream` need the bionemo
container, `scib` needs the conda env with `scib_metrics` (the container lacks it), and
`diagnose`/`benchmark` run anywhere. The orchestrator picks per step; do not run
`--scib` without having run the plain `metrics` pass in the container first, or
`recon_*` will have no live fallback.

`paired_umap` and `downstream` are opt-in and absent from the default step list.
`downstream` runs `evaluate/finetune/run_ablation_downstream.py`, producing the
`{model}/metrics/results_{task}.json` that `benchmark` plots — `benchmark` itself is
plot-only and computes nothing. Enabling `downstream` requires `downstream.tasks`.

### Downstream normalization (`--normalize`)
`run_ablation_downstream.py --normalize / --no-normalize` forces one normalization
policy across every task and model, overriding each task config's `normalize:` key;
`run_analysis.py` exposes it as `downstream.normalize`. Resolution is
CLI → task config `normalize:` → **off**. See `evaluate/finetune/normalization.py`.

- `--normalize`: CP10K+log1p is applied **once, centrally**, then every embedder is
  called as a pass-through via `policy.embed_kwargs()`. If a matrix does not look like
  raw counts (`looks_like_counts`) it is reported and skipped, never log1p'd twice.
- `--no-normalize`: nothing is applied at any point, for any task.

The decision is logged in a banner, per `[model/task]` line, under the summary table,
and recorded in each `results_{task}.json` under `"normalization"`.

**Mind the polarity.** Three nearby names mean different things:
`normalize` (do it) vs `embed(normalized=...)` (data is *already* normalized, so skip)
vs the analysis config's experiment-level `normalized` (same "already" sense, for
`build`/`metrics`). Never call an embedder with a bare `normalized=`/`log1p_only=` in
the downstream path — use `**policy.embed_kwargs()`, which pins both, since
`PCAEmbedder.embed(normalized=True, log1p_only=True)` still applies log1p.

### Cross-experiment comparison plots
Two config-driven scripts share one run-selection grammar (`groups`/`experiments` with
display names, `all_models`, `exclude` — see `evaluate/plot/experiment_selection.py`).
Both read metrics already on disk and recompute nothing.

```bash
# Downstream task results ({model}/metrics/results_<task>.json) as a bar grid
python evaluate/plot/plot_ablation_benchmark.py --config evaluate/plot/example_comparison_config.yaml

# Internal unified-FM metrics ({model}/metrics/unified_metrics.json, plus optional
# scIB columns from {ablation}/_scib_metrics/scib_<tag>.csv) as an annotated heatmap
python evaluate/plot/plot_unified_metrics_table.py --config evaluate/plot/example_unified_metrics_config.yaml --no-show
python evaluate/plot/plot_unified_metrics_table.py --config <cfg> --list          # what's available
python evaluate/plot/plot_unified_metrics_table.py --config <cfg> --style rank_table
```
Styles: `heatmap` (default), `rank_table`, `bars`. The **numbers printed in each cell
are the raw absolute values** straight from `unified_metrics.json`; only the *fill
colour* is a direction-aware, within-column normalisation (`normalize: minmax | zscore
| rank`). The best run in a column is always fully green, so read the numbers for
magnitude and the colours for ordering. `bars` shows the same absolute values with a
zero-anchored axis, so bar length is proportional to the value. Metrics with no
better/worse direction are drawn grey rather than ranked. The script warns and adds a
figure footnote when the selected runs disagree on `panel_hash`, since metrics computed
under different gene panels are not comparable.

`python evaluate/check/check_unified_table.py` self-checks both scripts offline (no
cluster, GPU or checkpoints needed).

**`metrics_old` fallback.** `plot_ablation_benchmark.py` fills anything missing from
`{model}/metrics/` — a whole `results_{task}.json`, or individual metrics inside one —
from a sibling `{model}/metrics_old/`, so a partially recomputed sweep still plots in
full (`mv metrics metrics_old` before re-running). `metrics/` always wins where both
have a value; a model with only `metrics_old/` is still picked up, including by
`all_models: true`. Every substitution prints a `[metrics_old]` line plus a summary
count — those bars come from an earlier run and may not be comparable with their
neighbours. `evaluate/check/check_benchmark_fallback.py` self-checks it.

**Per-task layout.** `per_task` figures place at most `PER_TASK_MAX_COLS` (2) metrics
per row, wrapping onto further rows and centring a short row. The combined grid is
unaffected: one row per task, short rows left-aligned so columns line up across tasks.
Bar names, group names and the legend are sized by `BAR_NAME_FONTSIZE` /
`GROUP_NAME_FONTSIZE` / `LEGEND_FONTSIZE`; the y-range is then widened to fit the
rotated names, measured per axes.

**Figure size is derived from the content** — `BAR_SLOT_INCHES` per bar for width,
`ROW_INCHES` per subplot row plus the legend for height — so `figsize` should not
normally be set. `figsize` describes the combined grid only and is **not** inherited
by per-task figures (`per_task_figsize` overrides those); inheriting it is what used
to make them come out at four columns' width for two columns of content. The width
floor is the rotated model name above each bar, so **`bar_names: false` is the lever**
when a dense figure is still too wide for a document — 48 runs give ~22x13in with
names, ~12x12in without. Below ~0.32in per bar the value labels rotate too, otherwise
adjacent numbers merge. `evaluate/check/check_benchmark_layout.py` self-checks all of
this, including that no two labels of adjacent bars overlap.

### Downstream Tasks
```bash
python evaluate/finetune/run_downstream_task.py \
    --config evaluate/finetune/cancer_annot_config_normalized.yaml
```

Available tasks: `cancer_annot` (cancer type classification), `deconv` (cell type deconvolution).
Config YAML files: `evaluate/finetune/cancer_annot_config_normalized.yaml`, `evaluate/finetune/deconv_config_normalized.yaml`.

### Survival: on-disk layout and the metrics pass
The `survival` task writes survival functions to
`{survboard_results_dir}/{COHORT}/{CANCER}/{ablation}_{model}/split_{fold}.csv`. The
`{ablation}_` prefix is **required for correctness**: model names (`unified`, `dat`,
`bulk_baseline`, …) repeat across ablation directories that share one
`survboard_results_dir`, so the older bare-`{model}` layout had several models writing
to, reading from and `_clear_survival_csvs`-ing the same files. `evaluate/finetune/
survival_layout.py` is the only place that name is composed; the writer
(`tasks/survboard_task.py`) and both readers go through it. **There is no fallback to
the old layout** — pre-cutover CSVs cannot be attributed to an ablation and are ignored.

The task itself writes only `c_index`. Antolini concordance, IBS and D-calibration
come from a separate CPU pass that must run before the `benchmark` plot:

```bash
# One ablation dir, from the survival task config (needs the SurvBoard env: pycox,
# sksurv, survival_evaluation)
python evaluate/finetune/tasks/evaluate_survboard_metrics.py --config <survival cfg> --ablation

# Many ablation dirs from one config; --dry-run needs none of that env and reports
# resolved directory names, collisions and orphaned pre-cutover CSVs
python evaluate/finetune/scripts/run_ablation_survboard_metrics.py \
    --config evaluate/finetune/scripts/ablation_survboard_metrics.yaml [--dry-run]
```

`evaluate/check/check_survival_layout.py` and `evaluate/check/check_survboard_sweep.py`
self-check both offline.

### Ablation Studies
```bash
python ablate/ablate.py --config ablate/example_ablation_config.json [--dry-run]
```

Generates training runs from a base config plus per-ablation overrides, optionally submitting to SLURM.

### Linting
```bash
ruff check --fix    # Lint with auto-fix
ruff format         # Format code
```

Pre-commit hooks run automatically on commit. Install with `pre-commit install`.

### Development Environment
Uses Docker devcontainer with NVIDIA CUDA support. Launch via:
- VSCode: "Reopen in Container" prompt
- CLI: `devcontainer up --workspace-folder . && devcontainer exec --workspace-folder . bash`

## Architecture

```
cancerfoundation/
├── model/
│   ├── model.py              # CancerFoundation LightningModule (training wrapper)
│   ├── module.py             # TransformerModule (core transformer architecture)
│   ├── layers.py             # Custom attention layers, CFGenerator variants
│   ├── grad_reverse.py       # Gradient reversal layer (for DAT)
│   ├── perturbation_model.py # Gene perturbation prediction variant
│   └── utils.py              # Pretrained weight loading, gene mapping
├── data/
│   ├── data_module.py        # SingleCellDataModule (Lightning DataModule)
│   ├── dataset.py            # SingleCellDataset (memory-mapped h5ad loading)
│   ├── data_collator.py      # AnnDataCollator (masking, padding, binning)
│   ├── bulk_sc_data.py       # Bulk and single-cell paired dataset handling
│   ├── bulk_sc_collator.py   # Collator for bulk/SC paired data
│   ├── data_sampler.py       # Balanced sampling across metadata categories
│   ├── gene_panel.py         # Gene-panel selection for embedding (shared + per-modality)
│   └── preprocess.py         # Binning + looks_like_counts (counts vs log1p detection)
├── assets/
│   └── vocab.json            # Default gene vocabulary
├── loss.py                   # MSE, ordinal cross-entropy, ZINB losses
├── gene_tokenizer.py         # GeneVocab tokenizer for gene names
└── utils.py                  # Pretrained weight loading, gene mapping

evaluate/
├── run_analysis.py           # Config-driven orchestrator: UMAPs + metrics + benchmark over N experiments
├── analysis_config.py        # Config schema/loader for run_analysis.py
├── example_analysis_config.yaml
├── check_analysis_plan.py    # Self-checks for the orchestrator (no cluster needed)
├── check/
│   ├── build_eval_adata.py   # Builds eval.h5ad: per-model embeddings + PCA baseline
│   ├── unified_metrics.py    # Unified-FM metrics + scIB batch-integration benchmark
│   ├── diagnose_scib.py      # Explains scIB numbers when they disagree with the UMAPs
│   ├── compare_experiments.py# Cross-experiment bar charts from unified_metrics.csv
│   └── check_*.py            # Standalone self-checks (no checkpoint needed)
├── finetune/
│   ├── normalization.py           # NormalizationPolicy: the ONE place input normalization is decided
│   ├── survival_layout.py         # The ONE place survival-CSV directory names are composed
│   ├── scripts/
│   │   ├── run_ablation_survboard_metrics.py  # SurvBoard metrics over N ablation dirs
│   │   └── ablation_survboard_metrics.yaml    # its config (the downstream-sweep dirs)
│   ├── downstream_task.py         # DownstreamTask abstract base class + TaskRegistry
│   ├── base_downstream_runner.py  # BaseDownstreamRunner (shared training loop, DDP, checkpointing)
│   ├── downstream_tasks_impl.py   # CancTypeClassTask, DeconvTask implementations
│   ├── run_downstream_task.py     # Unified CLI entry point for all downstream tasks
│   ├── task_template.py           # Template for implementing new tasks
│   └── utils.py                   # Downstream task utilities
└── plot/
    ├── experiment_selection.py     # Shared YAML run-selection (groups, display names, palette)
    ├── plot_ablation_benchmark.py  # Downstream-task bar grid (results_*.json)
    ├── plot_unified_metrics_table.py # Internal-metrics table (unified_metrics.json + scIB)
    ├── example_comparison_config.yaml
    ├── example_unified_metrics_config.yaml
    ├── umaps.py
    └── utils.py

ablate/
├── ablate.py          # Main ablation runner
├── config.py          # Ablation config dataclasses
├── runtime.py         # Runtime execution
└── slurm_worker.py    # SLURM job submission

data_preprocess/
├── data_processing.ipynb         # Interactive h5ad → memory-mapped conversion
├── bulk_preprocessing.ipynb      # Bulk RNA-seq preprocessing
├── bulk_sc_data_preprocessing.py # Paired bulk/SC preprocessing script
└── protein_embeddings.py         # ESM3/RNABert embedding generation
```

Top-level scripts and config:
- `pretrain.py` — main training entry point
- `utils.py` — `get_args()` thin wrapper that reads from `utils_config.py`
- `utils_config.py` — full argument parser with ~80 hyperparameters (canonical config definition)
- `scripts/h5ads_to_sc.py` — CLI batch conversion of h5ad files to memory-mapped format
- `bionemo_clariden.toml` / `bionemo_bristen.toml` — Enroot/Pyxis container configs for CSCS clusters

### Data Flow
1. Raw h5ad → SingleCellMemMapDataset (memory-mapped)
2. → SingleCellDataset (loads vocab, mappings, metadata)
3. → AnnDataCollator (masking, binning, padding)
4. → SingleCellDataModule (train/val splitting)
5. → CancerFoundation model

### Key Model Components
- **TransformerModule** (`module.py`): Gene encoder + value encoder → TransformerEncoder → decoder
- **CancerFoundation** (`model.py`): Lightning wrapper handling training loop, loss, optimization
  - `embed(adata)` method: produces cell embeddings directly from an AnnData object (handles gene intersection, HVG selection, binning, batched inference)
  - `embed(..., modality_col=...)` on multi-modality data uses ONE shared gene panel across modalities by default (`shared_panel=True`, consensus rank aggregation). Per-modality panels make the model's input distribution differ *with* modality, which is indistinguishable from a batch effect — never compare bulk to pseudobulk embeddings fitted on different panels. Pass `shared_panel=False` only to reproduce that older behaviour.
- Optional features: MVC decoder, DAT (Domain Adversarial Training), explicit zero probability modeling, contrastive loss (pseudobulk vs bulk), aggregation consistency loss, denoising, ESM/RNABert gene embeddings

### Loss Functions (`cancerfoundation/loss.py`)
- `mse`: Mean Squared Error
- `ordinal_cross_entropy`: Ordinal cross-entropy for binned expression
- `corn`: CORN ordinal loss
- `zinb`: Zero-Inflated Negative Binomial (for sparse expression)

### Downstream Task Framework (`evaluate/finetune/`)
Plugin-based architecture: `DownstreamTask` defines what (data, head, loss, metrics); `BaseDownstreamRunner` implements how (training loop, DDP, checkpointing). Add a new task by subclassing `DownstreamTask` — see `task_template.py` for the required interface.

### Ablation Framework (`ablate/`)
Config-driven: a JSON file specifies a base pretraining config plus a list of ablations (each as a dict of overrides). `ablate.py` generates one run per ablation and optionally submits to SLURM via `slurm_worker.py`. Use `--dry-run` to validate configs without launching jobs.

### Data Format
Processed data structure:
```
DATA/{tissue}/processed_data/train/
├── vocab.json      # Gene vocabulary
├── mapping.json    # Category mappings for metadata columns
├── obs.parquet     # Cell metadata (categorical-encoded)
└── mem.map/        # Memory-mapped expressions
```

## Entry Points

- **Training**: `pretrain.py` — main training script
- **Data Processing**: `data_preprocess/data_processing.ipynb` (interactive) or `scripts/h5ads_to_sc.py` (CLI batch)
- **Embedding**: `embed.ipynb` — generate cell embeddings from a trained model (or use `CancerFoundation.embed(adata)` directly)
- **Downstream Evaluation**: `evaluate/finetune/run_downstream_task.py` — cancer annotation, deconvolution
- **Ablation Studies**: `ablate/ablate.py` — run systematic feature ablations locally or on SLURM
- **HPC submission**:
  - `submits_biomed/` — SLURM job scripts for LeoMed (Singularity + multi-GPU)
  - `submits_cscs/` — SLURM job scripts for CSCS Alps (Enroot/Pyxis + multi-GPU)
- **Tutorials**: `tutorials/` — notebooks adapted from scGPT

## Configuration

All hyperparameters defined in `utils_config.py:get_args()`. `utils.py` at the top level is a thin wrapper. W&B integration configured via `.devcontainer/devcontainer.env` with `WANDB_API_KEY`.

**New `CancerFoundation.__init__` arguments must have a default**, chosen to reproduce the previous behaviour. `save_hyperparameters()` writes the signature into every checkpoint and Lightning replays it on load, so a new *required* argument breaks every checkpoint saved before it.

### Loading old checkpoints
Use `CancerFoundation.load_for_inference(path)` (see `cancerfoundation/checkpoint.py`), not `load_from_checkpoint`, anywhere a model is loaded to produce embeddings or reconstructions. It fills in hyperparameters the checkpoint predates, strips the `torch.compile` `_orig_mod.` key prefix, and drops the training-only DAT discriminators — whose `modality` head changed from 3 to 2 classes in July 2026. It is not for resuming training. `python scripts/inspect_checkpoint.py <ckpt>` reports hyperparameter and weight-shape drift for a checkpoint that will not load.
