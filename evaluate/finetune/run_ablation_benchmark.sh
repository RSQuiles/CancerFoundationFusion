#!/bin/bash -l
#SBATCH --time=16:00:00
#SBATCH --job-name=benchmark
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

# Include with preceding #SBATCH if GPU wanted
# --partition=gpu
# --gres=gpu:rtx4090:1

set -euo pipefail

# Default: use singularity
USE_LOCAL=0
# Set to 1 to run SurvBoard metric evaluation after downstream tasks
RUN_SURV=0
# Set to 1 to plot the benchmark results
PLOT=1

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
    if [[ "$arg" == "--surv" ]]; then
        RUN_SURV=1
    fi
    if [[ "$arg" == "--plot" ]]; then
        PLOT=1
    fi
done

# Must match survival_pred_config.yaml
ABLATION_DIR="/cluster/work/boeva/rquiles/outputs_rquiles/save_CFF/ablation_paired_counts"

SCRIPT_ARGS=(
    --ablation-dir $ABLATION_DIR
    --tasks canc_type_class deconv survival drug_sensitivity_v2
    --pca-baseline
    --config-dir /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs
)

# echo "=== GPU status ==="
# nvidia-smi

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

if [[ "$RUN_SURV" -eq 1 ]]; then
    CONFIG="/cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/finetune/configs/survival_pred_config.yaml"
    echo "=== SurvBoard metric evaluation ==="
    python -u ./tasks/evaluate_survboard_metrics.py \
        --config "$CONFIG" \
        --ablation
    echo "Completed."
    echo "Survboard metrics saved"
else
    echo "=== Skipping SurvBoard metric evaluation ==="
fi

# ---- Plot -------------------------------- #
if [[ "$PLOT" -eq 1 ]]; then
    echo "=== Plotting metrics ==="
    python ../plot/plot_ablation_benchmark.py \
        --ablation-dir $ABLATION_DIR
else
    echo "=== Skipping Benchmark Plotting ==="
fi