# CancerFoundation

A Transformer foundation model for single-cell RNA-seq gene expression prediction, built with PyTorch Lightning. Supports masked language modeling (MLM) and generative pretraining on single-cell data across multiple cancer types and tissues.

Large parts of this codebase are based on [scGPT](https://github.com/bowang-lab/scGPT). If you want to implement new functionality, check whether it already exists in scGPT first -- it can often be adapted with minor modifications.

## Installation

### Prerequisites

1. **Docker**
2. **NVIDIA GPU Drivers**
3. **NVIDIA Container Toolkit**

   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
4. **Devcontainer**

   **VSCode (Recommended)**:

   https://code.visualstudio.com/docs/devcontainers/containers

   **CLI**:

   https://code.visualstudio.com/docs/devcontainers/devcontainer-cli
5. **Weights and Biases (optional)**

All dependencies are provided by the devcontainer (NVIDIA BioNeMo 2.7 base image). There is no separate `requirements.txt`; the project must be run inside the devcontainer.

### Step-by-Step Guide
1. **Clone the repository**:
   ```bash
   git clone https://github.com/BoevaLab/CancerFoundation.git
   cd CancerFoundation
   ```

2. **Launch the dev container**:

   **VSCode**:
      Open the cloned repo folder in VSCode. A notification will open prompting to "Reopen in Container". Click it. VSCode will build the image and start the container.

   **CLI**:
      In the command line, build the devcontainer and then open a terminal inside it as follows:
      ```bash
      devcontainer up --workspace-folder .
      devcontainer exec --workspace-folder . bash
      ```

## Data Preparation

Training data must be preprocessed into a memory-mapped format. Starting from raw `.h5ad` files, use `data_processing.ipynb` to produce the expected directory structure:

```
DATA/{tissue}/processed_data/train/
├── vocab.json      # Gene vocabulary
├── mapping.json    # Category mappings for metadata columns
├── obs.parquet     # Cell metadata (categorical-encoded)
└── mem.map/        # Memory-mapped expression data
```

## Training

### Local / Workstation

See `debug.sh` for a complete example with all common parameters. A minimal run:

```bash
python pretrain.py \
    --gpus 1 \
    --save-dir ./save/experiment_name \
    --train-path ./DATA/brain/processed_data/train \
    --epochs 15 \
    --batch-size 16
```

Key parameters:
- `--training-tasks`: `"pcpt"` (masked prediction), `"gen"` (generation), or `"both"`
- `--do-mvc`: Enable Masked Value prediction for Cell embeddings
- `--input-emb-style`: `"mine"` or `"theirs"` (different value encoding strategies)
- `--precision`: `"32"`, `"16-mixed"`, or `"bf16-mixed"`
- `--conditions`: Metadata column(s) to condition on (e.g., `technology`)
- `--gen-method`: `"theirs"`, `"mine"`, `"orig"`, or `"quick"` (generative training strategy)
- `--compile`: Enable `torch.compile` for the model

Optionally, run the training script inside a tmux session for long runs.

### HPC (SLURM)

SLURM job submission scripts are provided for two clusters:
- `submits_biomed/` — LeoMed (Singularity-based)
- `submits_cscs/` — CSCS Alps (Enroot/Pyxis-based, configured via `bionemo_clariden.toml` / `bionemo_bristen.toml`)

Both use the BioNeMo container image with multi-GPU support.

## Embedding

Use `embed.ipynb` to generate cell embeddings from a trained model checkpoint.

Alternatively, the `CancerFoundation.embed(adata)` method can produce embeddings directly from an AnnData object — it handles gene intersection, HVG selection, binning, and batched inference internally.

## Data Processing

- `data_processing.ipynb` — interactive notebook for converting raw `.h5ad` files to the memory-mapped format
- `scripts/h5ads_to_sc.py` — CLI script for batch-converting a directory of `.h5ad` files (uses BioNeMo's `SingleCellCollection`)

## Tutorials

The `tutorials/` directory contains notebooks adapted from scGPT. For additional tutorials (e.g., perturbation prediction, GRN inference), refer to the [scGPT repository](https://github.com/bowang-lab/scGPT).

## Logging with Weights & Biases

If you want to use Weights & Biases to track your training run, put the following inside `.devcontainer/devcontainer.env`:
```
WANDB_API_KEY={YOUR_API_KEY}
```
You can find this key [here](https://wandb.ai/authorize).
