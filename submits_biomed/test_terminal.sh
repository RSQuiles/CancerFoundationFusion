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

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/cluster/work/boeva/rquiles/data/small_partition/pipeline_ready"
mkdir -p "$SAVE_DIR"

# Improve CUDA traceback
CUDA_LAUNCH_BLOCKING=1 python -u ../pretrain.py \
    --loss zinb \
    --config ./config_test.json \

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
# mv "./outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"