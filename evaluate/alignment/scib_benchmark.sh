#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=scib_benchmark
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# scIB benchmark over embeddings/<model>/ (embed/embed_splits.py). scib_metrics lives
# in the bulkFM conda env, not in the bionemo container, so this script has no
# container branch. Arguments go to the script, e.g. --dry-run or --name x.

set -euo pipefail

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/alignment/example_scib_config.yaml}"
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
    echo "ERROR: 'conda activate $CONDA_ENV' failed (exit $CONDA_STATUS); scib_metrics lives only there." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
python -u -m evaluate.alignment.scib_benchmark --config "$CONFIG" "$@"
