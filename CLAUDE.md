# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CancerFoundation is a PyTorch Lightning-based Transformer foundation model for single-cell RNA-seq gene expression prediction. It supports masked language modeling (MLM) and generative pretraining tasks on single-cell data. Large parts of the codebase are based on [scGPT](https://github.com/bowang-lab/scGPT).

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
│   └── perturbation_model.py # Gene perturbation prediction variant
├── data/
│   ├── data_module.py        # SingleCellDataModule (Lightning DataModule)
│   ├── dataset.py            # SingleCellDataset (memory-mapped h5ad loading)
│   ├── data_collator.py      # AnnDataCollator (masking, padding, binning)
│   ├── data_sampler.py       # Balanced sampling across metadata categories
│   └── preprocess.py         # Binning and normalization utilities
├── assets/
│   └── vocab.json            # Default gene vocabulary
├── loss.py                   # MSE and ordinal cross-entropy losses
├── gene_tokenizer.py         # GeneVocab tokenizer for gene names
└── utils.py                  # Pretrained weight loading, gene mapping
```

Top-level scripts and config:
- `pretrain.py` — main training entry point
- `utils.py` — argument parsing (`get_args()`) and hyperparameter definitions (separate from `cancerfoundation/utils.py`)
- `scripts/h5ads_to_sc.py` — CLI batch conversion of h5ad files to memory-mapped format (uses BioNeMo `SingleCellCollection`)
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
- Optional features: MVC decoder, DAT (Domain Adversarial Training), explicit zero probability modeling

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
- **Data Processing**: `data_processing.ipynb` (interactive) or `scripts/h5ads_to_sc.py` (CLI batch) — prepare h5ad files into memory-mapped format
- **Embedding**: `embed.ipynb` — generate cell embeddings from a trained model (or use `CancerFoundation.embed(adata)` directly)
- **HPC submission**:
  - `submits_biomed/` — SLURM job scripts for LeoMed (Singularity + multi-GPU)
  - `submits_cscs/` — SLURM job scripts for CSCS Alps (Enroot/Pyxis + multi-GPU)
- **Tutorials**: `tutorials/` — notebooks adapted from scGPT (see [scGPT repo](https://github.com/bowang-lab/scGPT) for more)

## Configuration

All hyperparameters defined in top-level `utils.py:get_args()`. W&B integration configured via `.devcontainer/devcontainer.env` with `WANDB_API_KEY`.
