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

### Downstream Tasks
```bash
python evaluate/finetune/run_downstream_task.py \
    --config evaluate/finetune/cancer_annot_config_normalized.yaml
```

Available tasks: `cancer_annot` (cancer type classification), `deconv` (cell type deconvolution).
Config YAML files: `evaluate/finetune/cancer_annot_config_normalized.yaml`, `evaluate/finetune/deconv_config_normalized.yaml`.

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
│   └── preprocess.py         # Binning and normalization utilities
├── assets/
│   └── vocab.json            # Default gene vocabulary
├── loss.py                   # MSE, ordinal cross-entropy, ZINB losses
├── gene_tokenizer.py         # GeneVocab tokenizer for gene names
└── utils.py                  # Pretrained weight loading, gene mapping

evaluate/
├── finetune/
│   ├── downstream_task.py         # DownstreamTask abstract base class + TaskRegistry
│   ├── base_downstream_runner.py  # BaseDownstreamRunner (shared training loop, DDP, checkpointing)
│   ├── downstream_tasks_impl.py   # CancTypeClassTask, DeconvTask implementations
│   ├── run_downstream_task.py     # Unified CLI entry point for all downstream tasks
│   ├── task_template.py           # Template for implementing new tasks
│   └── utils.py                   # Downstream task utilities
└── plot/
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
