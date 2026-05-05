#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=benchmark
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G

set -euo pipefail

# Default: use singularity
USE_LOCAL=1

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done


SCRIPT_ARGS=(
    --ablation-dir /cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison
    --tasks canc_type_class proteome_pred drug_sensitivity survival
    --config-dir /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs
    --pca-baseline
    --skip-existing
    --plot
)


if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u run_ablation_downstream.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u run_ablation_downstream.py "${SCRIPT_ARGS[@]}"
fi