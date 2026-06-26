#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --job-name=paired_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G

# --partition=gpu
# --gres=gpu:rtx4090:1


set -euo pipefail

# Default: use singularity
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts"

SCRIPT_ARGS=(
    --input-h5ad /cluster/work/boeva/bulkFM/data/processed/paired_samples.h5ad
    --ablation-dir $ABLATION_DIR
    --n-cell-lines 10
    --out-dir $ABLATION_DIR/paired_umaps
    --out-prefix pca)

# --no-sc

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u plot_paired_umap.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u plot_paired_umap.py "${SCRIPT_ARGS[@]}"
fi
