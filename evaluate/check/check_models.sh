#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=check_models
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts"

SCRIPT_ARGS=(
    --eval-adata $ABLATION_DIR/eval.h5ad
    --ablation-dir $ABLATION_DIR
)

echo "Running with singularity"
singularity run \
    --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
    --bind /cluster \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python -u unified_metrics.py "${SCRIPT_ARGS[@]}"