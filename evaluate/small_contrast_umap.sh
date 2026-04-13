#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=plot_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G


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
    --run-name small_contrast_8725461
    --flavor seurat_v3
    --color tissue_general assay
    --out-dir ./umap_plots
    --out-prefix small_contrast
    --adata-dir /cluster/work/boeva/rquiles/data/small_partition
    --adata-prefix partition_1
    --sample-size 10_000
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u umaps.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundation/evaluate \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u umaps.py "${SCRIPT_ARGS[@]}"
fi