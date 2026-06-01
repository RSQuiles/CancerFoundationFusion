#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=paired_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G


set -euo pipefail

# Default: use singularity
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

SCRIPT_ARGS=(
    --ckpt /cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison/unified_baseline/step_step=900000_epoch_epoch=01.ckpt
    --input-h5ad /cluster/work/boeva/bulkFM/data/processed/paired_samples.h5ad
    --out-dir ./umap_outputs
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u umaps.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u plot_paired_umap.py "${SCRIPT_ARGS[@]}"
fi