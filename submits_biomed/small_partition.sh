#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --job-name=small
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

set -e

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/cluster/work/boeva/rquiles/data/small_partition/pipeline_ready"
mkdir -p "$SAVE_DIR"

srun singularity run \
    --pwd /cluster/work/boeva/rquiles/CancerFoundation/submits_biomed \
    --bind /cluster/work/boeva/rquiles/CancerFoundation:/cluster/work/boeva/rquiles/CancerFoundation \
    --bind $TRAIN_DIR:$TRAIN_DIR \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    bash -c "
    export PATH=/usr/bin:/bin
    # Improve CUDA traceback
    CUDA_LAUNCH_BLOCKING=1 python -u ../pretrain.py \
        --unified \
        --gpus 2 \
        --num-workers 4 \
        --bulk-ratio 0.5 \
        --pb-ratio 0.0 \
        --n-sc-per-pseudobulk 10 \
        --save-dir "$SAVE_DIR" \
        --max-seq-len 1200 \
        --batch-size 32 \
        --nlayers 6 \
        --nheads 8 \
        --embsize 128 \
        --d-hi 256 \
        --epochs 10 \
        --lr 0.0001 \
        --warmup-ratio-or-step 10000 \
        --val-check-interval 0.5 \
        --trunc-by-sample \
        --loss mse \
        --conditions tissue_general assay \
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
        --balanced-sampler separate \
        --balanced-labels tissue_general assay \
        --wandb-entity "rquiles" \
        --wandb "small_partition" \
        --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
        --compile \
        "

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
# mv "./outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
echo "Job finished. Outputs and logs are in $SAVE_DIR"
