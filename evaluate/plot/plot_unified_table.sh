#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_unified_table
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# Comparison table of the internal (unified-FM) metrics across runs.
# Nothing is recomputed — only {model}/metrics/unified_metrics.json and, when the
# config asks for it, {ablation}/_scib_metrics/scib_<tag>.csv are read.
#
#   ./plot_unified_table.sh [CONFIG] [STYLE] [OUTPUT]
#   ./plot_unified_table.sh my_cfg.yaml rank_table ranks.png
#   ./plot_unified_table.sh --local          # bare python instead of singularity
#
# No GPU or CancerFoundation import is needed, so this runs in either environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_LOCAL=0
POSITIONAL=()
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    else
        POSITIONAL+=("$arg")
    fi
done

CONFIG="${POSITIONAL[0]:-$SCRIPT_DIR/config_non_paired_metrics.yaml}"
STYLE="${POSITIONAL[1]:-heatmap}"

SCRIPT_ARGS=(
    --config "$CONFIG"
    --style "$STYLE"
    --no-show
)
if [[ -n "${POSITIONAL[2]:-}" ]]; then
    SCRIPT_ARGS+=(--output "${POSITIONAL[2]}")
fi

if [[ "$USE_LOCAL" -eq 1 ]]; then
    source ~/.bashrc
    conda activate bulkFM
    python -u "$SCRIPT_DIR/plot_unified_metrics_table.py" "${SCRIPT_ARGS[@]}"
else
    source $surv
    python -u "$SCRIPT_DIR/plot_unified_metrics_table.py" "${SCRIPT_ARGS[@]}"
fi
