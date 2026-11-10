#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --job-name=geneicl_plot
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

# Draw the GeneICL comparison figures (task x model heatmaps, summary table) from the
# run_probe results into results/tasks/geneicl_<suite>/_plots/<name>/ (--name, or name:
# in the config's geneicl_plot block, is required). No GPU and no earlier compare.py run
# needed. Arguments go to plot_benchmark.py, e.g.:
#   sbatch plot_benchmark.sh --name main [--models unified_v1 pca] [--plots summary_table heatmaps]
# Pass --local to run without singularity (any env with pandas, matplotlib, pyyaml).

set -euo pipefail

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"
CONFIG="${CONFIG:-$PROJECT_ROOT/evaluate/geneicl/example_geneicl_config.yaml}"
CONDA_ENV="${CONDA_ENV:-geneicl}"
CONDA_STATUS=$?
set -eu

if [[ "$CONDA_STATUS" -ne 0 ]]; then
    echo "ERROR: 'conda activate $CONDA_ENV' failed (exit $CONDA_STATUS)." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
python -u -m evaluate.geneicl.plot_benchmark --config "$CONFIG"