#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --job-name=plot_alignment
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# CPU only: reads embeddings/<model>/ (embed/embed_splits.py) and fits UMAPs.
# Pass --local to run without singularity; any other argument goes to the script.

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
CONFIG="$PROJECT_ROOT/evaluate/representation/configs/example_umap_config.yaml"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.representation.plot_alignment "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run         --pwd "$PROJECT_ROOT"         --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles         /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif         python -u -m evaluate.representation.plot_alignment "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
