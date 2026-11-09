#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=store_check
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G

# Store integrity check in the bionemo container (needs SCDL, no GPU). Arguments go to
# check_store_alignment.py, e.g.:  sbatch check_store_alignment.sh --modality bulk --split leiden_s0
# Add --gene-column <col> if the store was built with one. Pass --local to run without singularity.

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
    python -u data_preprocess/check_store_alignment.py "${EXTRA_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd "$PROJECT_ROOT" \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u data_preprocess/check_store_alignment.py "${EXTRA_ARGS[@]}"
fi
