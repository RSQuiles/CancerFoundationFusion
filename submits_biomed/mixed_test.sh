#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --job-name=mixed_testrun
#SBATCH --output=./outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --cpus-per-task=4

set -e

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/cluster/work/boeva/rquiles/data/mixed_test/pipeline_ready"
mkdir -p "$SAVE_DIR"

# Improve CUDA traceback
CUDA_LAUNCH_BLOCKING=1 python ../pretrain.py \
    --unified \
    --bulk-ratio 0.3 \
    --pb-ratio 0.3 \
    --n-sc-per-pseudobulk 10 \
    --gpus 1 \
    --save-dir "$SAVE_DIR" \
    --max-seq-len 600 \
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
    --conditions tissue \
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
    --num-workers 1 \
    # --compile \
    #  --wandb "brain" \
    #  --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
# mv "./outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"
