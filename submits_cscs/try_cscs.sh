#!/bin/bash
#SBATCH --account=a132
#SBATCH --job-name=test
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/slurm-%x-%j.log

set -x

ulimit -c 0 

srun -ul --environment=./test.toml bash -c "
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    python pretrain.py \
        --save-dir "/iopsstor/scratch/cscs/rvander/save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
        --max-seq-len 1200 \
        --batch-size 64 \
        --nlayers 6 \
        --nheads 8 \
        --embsize 128 \
        --d-hi 256 \
        --epochs 50 \
        --lr 0.0001 \
        --warmup-ratio-or-step 10000 \
        --val-check-interval 0.5 \
        --trunc-by-sample \
        --loss mse \
        --conditions "technology" \
        --balance-primary technology \
        --train-path "/iopsstor/scratch/cscs/rvander/DATA/brain/processed_data/train" \
        --zero-percentages 0.2 0.4 0.6 \
        --strategy='ddp' \
        --seed 0 \
        --wandb "brain" \
        --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
        --precision "bf16-mixed" \
        --do-mvc \
        --log-interval 50 \
        --training-tasks "both" \
        --where-condition "end" \
        --gen-method "theirs" \
        --compile
"