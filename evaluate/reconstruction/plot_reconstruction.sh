#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=reconstruction_plot
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G

# Draw the reconstruction comparison figures (MAE by bin / by target, metric tables,
# space-free comparisons) from the run_reconstruction results into
# results/tasks/reconstruction/_plots/{all_models,<m1>_<m2>_...}/<plan>/. No GPU needed.
# Arguments go to plot_reconstruction.py, e.g.:
#   sbatch plot_reconstruction.sh --plan bulk:leiden_s0:test:r25 [--models unified_v1 bulkformer]
# Pass --local to run without singularity (any env with pandas, pyarrow, scipy, matplotlib).

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
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/reconstruction/example_reconstruction_config.yaml}"

SCRIPT_ARGS=(
    --config "$CONFIG"
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    cd "$PROJECT_ROOT"
    python -u -m evaluate.reconstruction.plot_reconstruction "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd "$PROJECT_ROOT" \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u -m evaluate.reconstruction.plot_reconstruction "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
