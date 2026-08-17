#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_scib_joint
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# One scIB benchmark table over runs from several ablation directories, drawn by
# scib_metrics' own Benchmarker.plot_results_table.
#
# Reads {ablation}/_scib_metrics/scib_<tag>.csv only — nothing is recomputed, so the
# 'scib' step must already have run for every directory the config names
# (evaluate/run_analysis.py --step scib).
#
#   ./plot_scib_joint.sh [CONFIG] [OUTPUT]
#   ./plot_scib_joint.sh my_cfg.yaml figures/scib_joint.svg
#   ./plot_scib_joint.sh my_cfg.yaml --dry-run     # merged CSV only, no plotting
#   ./plot_scib_joint.sh my_cfg.yaml --list        # what the source tables contain
#
# NOTE the environment. This plots with scib_metrics, which the bionemo container
# does NOT have — the same split that makes 'scib' the one analysis step needing the
# conda env. There is no container branch here on purpose: the container cannot run
# this script at all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
OUTPUT=""
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --*)
            PASSTHROUGH+=("$arg")
            ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$arg"
            elif [[ -z "$OUTPUT" ]]; then
                OUTPUT="$arg"
            else
                echo "Unexpected argument: $arg" >&2
                exit 1
            fi
            ;;
    esac
done

CONFIG="${CONFIG:-$SCRIPT_DIR/example_scib_joint_config.yaml}"

SCRIPT_ARGS=(--config "$CONFIG")
if [[ -n "$OUTPUT" ]]; then
    SCRIPT_ARGS+=(--output "$OUTPUT")
fi
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
    SCRIPT_ARGS+=("${PASSTHROUGH[@]}")
fi

source ~/.bashrc
conda activate bulkFM

python -u "$SCRIPT_DIR/plot_scib_joint.py" "${SCRIPT_ARGS[@]}"
