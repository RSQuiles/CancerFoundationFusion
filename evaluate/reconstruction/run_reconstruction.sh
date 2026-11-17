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
CONDA_ENV=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) USE_LOCAL=1; shift ;;
        --conda)
            [[ $# -ge 2 ]] || { echo "Error: --conda needs an environment name" >&2; exit 1; }
            CONDA_ENV="$2"; shift 2 ;;
        --conda=*) CONDA_ENV="${1#--conda=}"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"
CONFIG="$PROJECT_ROOT/evaluate/reconstruction/configs/bulk_only_reconstruction.yaml"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.reconstruction.run_reconstruction "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
elif [[ -n "$CONDA_ENV" ]]; then
    # conda's activation scripts reference unset variables, which `set -u` would turn into errors
    set +u
    source ~/.bashrc
    conda activate "$CONDA_ENV"
    set -u
    echo "Activated conda env: $CONDA_ENV"
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
