#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --job-name=survival
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_outputs"

# ---- Configuration — edit these paths ------------------------------------ #
# Activate virtual enviroment
source $surv
CONFIG="/cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs/survival_pred_config.yaml"

# ---- SurvBoard metric evaluation -------------------------------- #
echo "=== Phase 2: SurvBoard metric evaluation ==="
python -u "${SCRIPT_DIR}/../tasks/evaluate_survboard_metrics.py" \
    --config     "$CONFIG" \
    --ablation
echo "Completed."
echo "Metrics written to: ${ABLATION_DIR}/${MODEL_NAME}/metrics/results_survival.json"