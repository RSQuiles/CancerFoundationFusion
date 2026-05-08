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

SURVBOARD_CONDA_ENV="survboard"

# Must match survival_pred_config.yaml
SURVBOARD_DATA_DIR="/cluster/work/boeva/bulkFM/data/processed/survboard"
SPLITS_DIR="${SURVBOARD_DATA_DIR}/splits"
RESULTS_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/survival/results"
ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison"

COHORTS=(TCGA)
CANCER_TYPES=(BRCA BLCA COAD GBM KIRC LAML LGG LIHC LUAD LUSC OV PAAD SARC SKCM STAD UCEC)
MODEL_NAME="cff"

# Activate conda env
conda activate survboard

# ---- SurvBoard metric evaluation -------------------------------- #
echo "=== Phase 2: SurvBoard metric evaluation ==="
python -u "${SCRIPT_DIR}/../tasks/evaluate_survboard_metrics.py" \
    --data-dir     "$SURVBOARD_DATA_DIR" \
    --splits-dir   "$SPLITS_DIR" \
    --results-dir  "$RESULTS_DIR" \
    --ablation-dir "$ABLATION_DIR" \
    --cohorts      "${COHORTS[@]}" \
    --cancer-types "${CANCER_TYPES[@]}" \
    --model-name   "$MODEL_NAME"
echo "Completed."
echo "Metrics written to: ${ABLATION_DIR}/${MODEL_NAME}/metrics/results_survival.json"