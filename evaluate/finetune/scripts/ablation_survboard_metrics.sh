#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=survboard_metrics_sweep
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

# SurvBoard metrics for every model of every ablation directory in the config.
# CPU only — this reads the survival-function CSVs the 'survival' downstream task
# already wrote, and computes no embeddings.
#
#   sbatch ablation_survboard_metrics.sh
#   sbatch ablation_survboard_metrics.sh --only big_condition
#
# Extra arguments are forwarded to the script, so --only / --models /
# --skip-existing / --dry-run all work.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_outputs"

# The SurvBoard environment (pycox, sksurv, survival_evaluation), as in
# survboard_metrics.sh.
source $surv

CONFIG="${SCRIPT_DIR}/ablation_survboard_fill.yaml"

echo "=== SurvBoard metric sweep ==="
python -u "${SCRIPT_DIR}/run_ablation_survboard_metrics.py" \
    --config "$CONFIG" \
    "$@"
