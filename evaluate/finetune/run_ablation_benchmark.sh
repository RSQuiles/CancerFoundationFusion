#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=benchmark
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=1
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

# Must match survival_pred_config.yaml
ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison"

SCRIPT_ARGS=(
    --ablation-dir $ABLATION_DIR
    --tasks survival canc_type_class
    --config-dir /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs
    --pca-baseline
)

echo "=== GPU status ==="
nvidia-smi

# ---- Run downstream tasks -------------------------------- #
echo "=== Running Downstream tasks ==="
if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u run_ablation_downstream.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u run_ablation_downstream.py "${SCRIPT_ARGS[@]}"
fi

# ---- SurvBoard metric evaluation -------------------------------- #
source $surv
CONFIG="/cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs/survival_pred_config.yaml"

echo "=== SurvBoard metric evaluation ==="
python -u ./tasks/evaluate_survboard_metrics.py \
     --config "$CONFIG" \
     --ablation
echo "Completed."
echo "Survboard metrics saved"

# ---- Plot -------------------------------- #
echo "=== Plotting metrics ==="
python ./plot/ablation_benchmark.py \
    --ablation-dir $ABLATION_DIR