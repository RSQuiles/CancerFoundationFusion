#!/bin/bash -l
#SBATCH --job-name=train_medium_condtech_my_init_weights_4gpu_lrx2_warmupratio
#SBATCH --output=./%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --account=a132

set -x

ulimit -c 0

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/iopsstor/scratch/cscs/rvander/DATA/cancer_gpt/"

srun -ul --environment=./bionemo.toml bash -c "
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    python pretrain.py \
    --gpus 4 \
    --save-dir "$SAVE_DIR" \
    --max-seq-len 1200 \
    --batch-size 32 \
    --nlayers 6 \
    --nheads 8 \
    --embsize 256 \
    --d-hi 512 \
    --epochs 15 \
    --lr 0.0002 \
    --warmup-ratio-or-step 0.15 \
    --val-check-interval 1.0 \
    --trunc-by-sample \
    --loss mse \
    --balance-primary tissue \
    --balance-secondary technology \
    --train-path "$TRAIN_DIR" \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp' \
    --seed 0 \
    --wandb "brain" \
    --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    --precision "bf16-mixed" \
    --do-mvc \
    --log-interval 100 \
    --training-tasks "both" \
    --gen-method "orig" \
    --input-emb-style "theirs" \
    --compile \
    --conditions technology \
    --where-condition end
"

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
mv "./${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"