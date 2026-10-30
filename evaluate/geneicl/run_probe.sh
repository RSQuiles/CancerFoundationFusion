#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=geneicl_probe
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

# Score GeneICL-suite embeddings (embed_suite.sh) with GeneICL's own probe. benchmark.py
# needs the GeneICL eval env (GeneICL's eval/environment.yaml: hydra, psutil, pinned
# scikit-learn), not the bionemo container, so this script has no container branch.
# run_probe.py itself is torch-free and starts benchmark.py with that env's python.
# Arguments go to the script, e.g.:  sbatch run_probe.sh --model unified_v1 [--resume]

set -euo pipefail

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/geneicl/example_geneicl_config.yaml}"
CONDA_ENV="${CONDA_ENV:-geneicl-benchmark}"

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
    echo "ERROR: 'conda activate $CONDA_ENV' failed (exit $CONDA_STATUS); benchmark.py needs GeneICL's eval env." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
python -u -m evaluate.geneicl.run_probe --config "$CONFIG" --geneicl-python "$(command -v python)" "$@"
