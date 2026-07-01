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

SCIB_ONLY=0
# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--scib" ]]; then
        SCIB_ONLY=1
    fi
done

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts"

SCRIPT_ARGS=(
    --eval-adata $ABLATION_DIR/eval.h5ad
    --ablation-dir $ABLATION_DIR
    --batch-size 64
)

# ── Step 1: main metrics (no scIB) ───────────────────────────────────────────
# Runs inside the singularity container (or locally) which does not need scib.
echo "=== Step 1: computing main metrics ==="
if [[ "$USE_LOCAL" -eq 1 && "$SCIB_ONLY" -eq 0 ]]; then
    python -u unified_metrics.py "${SCRIPT_ARGS[@]}"
elif [[ "$SCIB_ONLY" -eq 0 ]]; then
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u unified_metrics.py "${SCRIPT_ARGS[@]}"
fi

# ── Step 2: scIB batch integration metrics ───────────────────────────────────
# Uses the conda python that has scib + scanpy. --skip-existing avoids
# recomputing metrics 1-4; --scib appends scIB keys to each cached JSON
# and writes batch_integration.png.
echo "=== Step 2: computing scIB batch integration metrics ==="
source ~/.bashrc
conda activate bulkFM
python -u unified_metrics.py "${SCRIPT_ARGS[@]}" --scib --skip-existing