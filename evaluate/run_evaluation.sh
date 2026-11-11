#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=evaluation
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclude=gpu-biomed-05,gpu-biomed-25

# Every evaluation step of one global config, sequentially, in ONE allocation (hence the
# GPU: embedding and CancerFoundation reconstruction need it). The coordinator runs on
# the host, in a light env with pyyaml (CONDA_ENV); each step then starts in the
# environment the config gives it (bionemo container, a conda env, ...), never nested.
#   sbatch evaluate/run_evaluation.sh                                   # default config
#   CONFIG=my_eval.yaml sbatch evaluate/run_evaluation.sh --only representation
#   bash evaluate/run_evaluation.sh --dry-run                           # anywhere, no submission
# Every argument goes to evaluate/run_evaluation.py (--only, --skip, --from, --list,
# --dry-run, --no-preflight, --continue-on-error, --run-dir).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cluster/work/boeva/rquiles/CancerFoundationFusion}"
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/configs/example_evaluation.yaml}"
CONDA_ENV="${CONDA_ENV:-bulkFM}"

# Environment setup runs WITHOUT -e and -u: /etc/profile.d scripts read unset
# variables (fatal under -u) and a .bashrc that returns early for non-interactive
# shells is non-zero (fatal under -e, silently). Both are restored right after.
set +eu
# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
CONDA_STATUS=$?
set -eu

if [[ "$CONDA_STATUS" -ne 0 ]]; then
    echo "ERROR: 'conda activate $CONDA_ENV' failed (exit $CONDA_STATUS); the coordinator needs pyyaml." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
python -u -m evaluate.run_evaluation --config "$CONFIG" "$@"
