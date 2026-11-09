#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=train_eval_check
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# Train/eval consistency check in the bionemo container (GPU). Arguments go to
# check_train_eval.py, e.g.:  sbatch check_train_eval.sh --model bulk_only --split leiden_s0
# Pass --local to run without singularity.

set -euo pipefail

USE_LOCAL=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.reconstruction.check_train_eval "${EXTRA_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run --nv \
        --pwd "$PROJECT_ROOT" \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u -m evaluate.reconstruction.check_train_eval "${EXTRA_ARGS[@]}"
fi
