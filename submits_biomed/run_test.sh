#!/bin/bash -l
# SBATCH --job-name=dummy_testrun
# SBATCH --output=./outputs/%x_%j.out
# SBATCH --time=04:00:00
# SBATCH --partition=gpu
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:rtx4090:1
# SBATCH --cpus-per-task=3

set -e

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/cluster/work/boeva/rquiles/data/dummy_test"
mkdir -p "$SAVE_DIR"

python ../pretrain.py \
    --gpus 1 \
    --save-dir "$SAVE_DIR" \
    --max-seq-len 1200 \
    --batch-size 64 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 128 \
    --d-hi 256 \
    --epochs 2 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --val-check-interval 0.5 \
    --trunc-by-sample \
    --loss mse \
    --conditions technology \
    --balance-primary technology \
    --train-path "$TRAIN_DIR" \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp' \
    --seed 0 \
    --precision "bf16-mixed" \
    --do-mvc \
    --log-interval 50 \
    --training-tasks "both" \
    --where-condition "end" \
    --gen-method "theirs" \
    --compile \
    --num-workers 3
    #  --wandb "brain" \
    #  --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
# mv "./outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"
