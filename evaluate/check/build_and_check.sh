#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=build_check
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G
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

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_monitor_align"

BUILD_ARGS=(
    --adata-dir /cluster/work/boeva/rquiles/data/mini_eheiss_RAW/pipeline_ready/h5ads
    --sample-size 5000
    --out $ABLATION_DIR/eval.h5ad
    --ablation-dir $ABLATION_DIR
    --precomputed-pb
)

echo "=== Step 0: Building evaluation AnnData ==="
singularity run \
    --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
    --bind /cluster \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python -u build_eval_adata.py "${BUILD_ARGS[@]}"


CHECK_ARGS=(
    --eval-adata $ABLATION_DIR/eval.h5ad
    --ablation-dir $ABLATION_DIR
    --batch-size 64
    --skip-existing
)

# ── Step 1: main metrics (no scIB) ───────────────────────────────────────────
# Runs inside the singularity container (or locally) which does not need scib.
echo "=== Step 1: computing main metrics ==="
if [[ "$USE_LOCAL" -eq 1 && "$SCIB_ONLY" -eq 0 ]]; then
    python -u unified_metrics.py "${CHECK_ARGS[@]}"
elif [[ "$SCIB_ONLY" -eq 0 ]]; then
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u unified_metrics.py "${CHECK_ARGS[@]}"
fi

# ── Step 2: scIB batch integration metrics ───────────────────────────────────
# Uses the conda python that has scib + scanpy. --skip-existing avoids
# recomputing metrics 1-4; --scib appends scIB keys to each cached JSON
# and writes batch_integration.png.
#
# Note --skip-existing is now panel-aware: a cached unified_metrics.json whose
# panel_hash differs from this eval.h5ad's is ignored and recomputed, so rebuilding
# with a different gene panel can no longer serve stale numbers.
echo "=== Step 2: computing scIB batch integration metrics ==="
source ~/.bashrc
conda activate bulkFM
python -u unified_metrics.py "${CHECK_ARGS[@]}" --scib --skip-existing

# ── Step 3: diagnose the scIB numbers ────────────────────────────────────────
# Reproduces the exact bulk-vs-pseudobulk subset scIB scores and reports the
# quantities those metrics are built from: the label x batch contingency (BRAS and
# kBET skip labels holding a single batch), Euclidean *and* cosine kNN mixing (iLISI
# and UMAPs use Euclidean, BRAS uses cosine), cloud geometry, and unscaled PCR.
# Use it whenever the table and the UMAPs disagree.
echo "=== Step 3: diagnosing the scIB numbers ==="
python -u diagnose_scib.py \
    --eval-adata $ABLATION_DIR/eval.h5ad \
    --out-csv $ABLATION_DIR/_scib_metrics/diagnosis.csv
