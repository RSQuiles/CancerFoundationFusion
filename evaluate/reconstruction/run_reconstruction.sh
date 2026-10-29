#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=reconstruction
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# CancerFoundation reconstruction in the bionemo container (GPU). Arguments go to
# run_reconstruction.py, e.g.:  sbatch run_reconstruction.sh --model unified_v1
# BulkFormer needs its own conda env instead (see embed/embed_any.py):
#   conda activate bulkformer-latest && python -m evaluate.reconstruction.run_reconstruction \
#       --config <cfg> --reconstructor bulkformer --name bulkformer --option repo=...
# mask_plan.py and compare.py need only numpy/pandas/anndata and run anywhere.
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
CONFIG="$PROJECT_ROOT/evaluate/reconstruction/example_reconstruction_config.yaml"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.reconstruction.run_reconstruction "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run --nv \
        --pwd "$PROJECT_ROOT" \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u -m evaluate.reconstruction.run_reconstruction "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
