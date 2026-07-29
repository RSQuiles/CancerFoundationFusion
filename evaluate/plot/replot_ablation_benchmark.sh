#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --job-name=replot_ablation
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Re-generate the ablation benchmark figure from metrics already on disk.
# Nothing is recomputed — only {model}/metrics/results_*.json are read.
#
#   ./replot_ablation_benchmark.sh [ABLATION_DIR] [OUTPUT]
#
# Metric selection:
#   survival, canc_type_class, deconv  → left at their defaults
#   drug sensitivity (IC50 regression) → mae, mse, pearson_rho, r2, rmse, spearman_rho
#   drug sensitivity (Cmax classifier) → accuracy, auprc, auroc, balanced_accuracy,
#                                        f1, mcc, precision, recall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ABLATION_DIR="${1:-/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_united_data}"
OUTPUT="${2:-$ABLATION_DIR/benchmark_replot.png}"

source $surv

python -u "$SCRIPT_DIR/replot_ablation_benchmark.py" \
    --ablation-dir "$ABLATION_DIR" \
    --metrics \
        ic50=mae,mse,pearson_rho,r2,rmse,spearman_rho \
        cmax=accuracy,auprc,auroc,balanced_accuracy,f1,mcc,precision,recall \
    --output "$OUTPUT" \
    --no-show

echo "Benchmark plot saved to $OUTPUT"
