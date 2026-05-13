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
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

# Must match survival_pred_config.yaml
SURVBOARD_DATA_DIR="/cluster/work/boeva/bulkFM/data/processed/survboard"
SPLITS_DIR="${SURVBOARD_DATA_DIR}/splits"
RESULTS_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/survival/results"
ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_base_comparison"

COHORTS=(TCGA)
CANCER_TYPES=(BRCA BLCA COAD GBM KIRC LAML LGG LIHC LUAD LUSC OV PAAD SARC SKCM STAD UCEC)
MODEL_NAME="cff"

SCRIPT_ARGS=(
    --ablation-dir $ABLATION_DIR
    --tasks canc_type_class survival
    --config-dir /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs
    --pca-baseline
    --skip-existing
    --plot
)

# ---- Run downstream tasks -------------------------------- #
echo "=== Running Downstream tasks ==="
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

# ---- SurvBoard metric evaluation -------------------------------- #
conda activate survboard

echo "=== SurvBoard metric evaluation ==="
python -u ./tasks/evaluate_survboard_metrics.py \
    --data-dir     "$SURVBOARD_DATA_DIR" \
    --splits-dir   "$SPLITS_DIR" \
    --results-dir  "$RESULTS_DIR" \
    --ablation-dir "$ABLATION_DIR" \
    --cohorts      "${COHORTS[@]}" \
    --cancer-types "${CANCER_TYPES[@]}" \
    --model-name   "$MODEL_NAME"
echo "Completed."
echo "Metrics written to: ${ABLATION_DIR}/${MODEL_NAME}/metrics/results_survival.json"

# ---- Plot -------------------------------- #
echo "=== Plotting metrics ==="
python ./plot/ablation_benchmark.py \
    --ablation-dir $ABLATION_DIR