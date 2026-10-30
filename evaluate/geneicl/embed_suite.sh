#!/bin/bash -l
#SBATCH --time=8:00:00
#SBATCH --job-name=geneicl_embed
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# Embed the GeneICL task suite with a CancerFoundation model in the bionemo container
# (GPU). Arguments go to embed_suite.py, e.g.:  sbatch embed_suite.sh --model unified_v1
# Then score the embeddings in the GeneICL eval env with run_probe.sh.
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
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/geneicl/example_geneicl_config.yaml}"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
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
