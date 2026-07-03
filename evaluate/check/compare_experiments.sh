#!/bin/bash -l
#SBATCH --time=0:30:00
#SBATCH --job-name=compare_experiments
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

BASE="/cluster/work/boeva/rquiles/outputs/save_CFF"

# ── Configure experiments here ────────────────────────────────────────────────
# Add one --csvs path and one --names entry per experiment (same order).

SCRIPT_ARGS=(
    --csvs
        $BASE/ablation_paired_counts/unified_metrics.csv
        $BASE/ablation_paired_corn/unified_metrics.csv
        $BASE/ablation_paired_mix/unified_metrics.csv
    --names
        "Paired counts"
        "Paired CORN"
        "Paired mix"
    --out $BASE/experiment_comparison.png
    --ncols 5
)

# ── Run (no GPU / singularity needed — pure numpy + matplotlib) ───────────────
USE_LOCAL=0
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

if [[ "$USE_LOCAL" -eq 1 ]]; then
    source ~/.bashrc
    conda activate bulkFM
    python -u compare_experiments.py "${SCRIPT_ARGS[@]}"
else
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
        --bind /cluster \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u compare_experiments.py "${SCRIPT_ARGS[@]}"
fi
