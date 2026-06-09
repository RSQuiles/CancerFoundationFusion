#!/bin/bash -l
#SBATCH --job-name=test_run
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

set -e

# Default, do not use Singularity
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

SCRIPT_ARGS=(
    --config ./config_test.json
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    CUDA_LAUNCH_BLOCKING=1 python -u ../pretrain.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        CUDA_LAUNCH_BLOCKING=1 python -u ../pretrain.py "${SCRIPT_ARGS[@]}"
fi


if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
# mv "./outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"
