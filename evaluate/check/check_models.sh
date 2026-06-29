#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=check_models
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

USE_LOCAL=0
# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_corn"

SCRIPT_ARGS=(
    --plot-csv $ABLATION_DIR/unified_metrics.csv
    --batch-size 64
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally"
    python -u unified_metrics.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u unified_metrics.py "${SCRIPT_ARGS[@]}"
fi