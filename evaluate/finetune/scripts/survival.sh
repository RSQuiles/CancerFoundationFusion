#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --job-name=survival
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/slurm_outputs"
CONFIG="${SCRIPT_DIR}/../configs/survival_pred_config.yaml"

# ---- Phase 1: CFF Cox training + survival CSV export --------------------- #
echo "=== Phase 1: CFF Cox training (fold_index: all) ==="
CUDA_LAUNCH_BLOCKING=1 \
python -u "${SCRIPT_DIR}/../run_downstream_task.py" \
    --config "$CONFIG"
echo "Completed."
