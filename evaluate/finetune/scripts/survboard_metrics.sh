#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=survival_metrics
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

mkdir -p ./slurm_outputs

# ---- Configuration — edit these paths ------------------------------------ #
# Activate virtual enviroment
source $surv
CONFIG="/cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs/survival_pred_config.yaml"

# ---- SurvBoard metric evaluation -------------------------------- #
echo "=== Phase 2: SurvBoard metric evaluation ==="
python -u ./../tasks/evaluate_survboard_metrics.py \
    --config "$CONFIG" \
    --ablation
echo "Completed."
echo "Metrics written to: ${ABLATION_DIR}/${MODEL_NAME}/metrics/results_survival.json"