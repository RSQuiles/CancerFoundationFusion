#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=plot_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --partition=gpu

# --gres=gpu:rtx4090:1
# --partition=gpu

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
    --run-name paired_counts_dat
    --ckpt "/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts/contrastive/step_step=340000_epoch_epoch=00.ckpt"
    --adata-dir /cluster/work/boeva/rquiles/data/paired_dataset_counts/pipeline_ready/h5ads
    --color tissue_general assay
    --sample-size 10_000
    --plot-pb-only
    --skip-unknown
)

# --ablation-dir /cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u umaps.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u umaps.py "${SCRIPT_ARGS[@]}"
fi