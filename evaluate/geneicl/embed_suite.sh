#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=geneicl_embed
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclude=gpu-biomed-05,gpu-biomed-25

# Embed the GeneICL task suite with a CancerFoundation model in the bionemo container
# (GPU). Arguments go to embed_suite.py, e.g.:  sbatch embed_suite.sh --model unified_v1
# Then score the embeddings in the GeneICL eval env with run_probe.sh.
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
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/geneicl/configs/bulkformer_config.yaml}"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.geneicl.embed_suite "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"

elif [[ -n "$CONDA_ENV" ]]; then
        # conda's activation scripts reference unset variables, which `set -u` would turn into errors
        set +u
        source ~/.bashrc
        conda activate "$CONDA_ENV"
        set -u
        echo "Activated conda env: $CONDA_ENV; running with $(command -v python)"
        cd "$PROJECT_ROOT"
        python -u -m evaluate.geneicl.embed_suite "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
        
else
    echo "Running with singularity"
    singularity run --nv \
        --pwd "$PROJECT_ROOT" \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u -m evaluate.geneicl.embed_suite "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
