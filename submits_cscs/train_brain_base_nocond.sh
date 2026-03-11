#!/bin/bash -l
#SBATCH --job-name=train_brain_base_nocond
#SBATCH --output=./%x_%j.out
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --account=a132

set -e

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="/iopsstor/scratch/cscs/rvander/DATA/brain/processed_data/train"


podman load -i /capstor/scratch/cscs/rvander/images/bionemo-framework_nightly.tar
srun podman run \
    -e WANDB_API_KEY \
    --workdir /users/rvander/project_dir/my_prop/CancerFoundation \
    --volume /users/rvander/project_dir/my_prop/CancerFoundation:/users/rvander/project_dir/my_prop/CancerFoundation \
    --volume $TEMP_SAVE_DIR:$TEMP_SAVE_DIR \
    --volume $TRAIN_DIR:$TRAIN_DIR \
    --gpus $CUDA_VISIBLE_DEVICES \
    --rm \
    nvcr.io/nvidia/clara/bionemo-framework:nightly \
    python pretrain.py \
    --gpus 2 \
    --save-dir "$TEMP_SAVE_DIR" \
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
    --balance-primary technology \
    --train-path "$TRAIN_DIR" \
    --zero-percentages 0.2 0.4 0.6 \
    --strategy='ddp' \
    --seed 0 \
    --wandb "brain" \
    --wandb-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    --precision "bf16-mixed" \
    --do-mvc \
    --log-interval 50 \
    --training-tasks "both" \
    --gen-method "theirs" \
    --compile

SAVE_DIR="./save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"

if [ -d "./lightning_logs/version_${SLURM_JOB_ID}" ]; then
    mv "./lightning_logs/version_${SLURM_JOB_ID}" "$SAVE_DIR/lightning_log"
fi

cp "$TRAIN_DIR/vocab.json" "$SAVE_DIR/vocab.json"
cp "$0" "$SAVE_DIR/run_script.sh"
mv "/capstor/scratch/cscs/rvander/save/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$SAVE_DIR/slurm.out"
mv "$TEMP_SAVE_DIR"/* "$SAVE_DIR"/
rm -r "$TEMP_SAVE_DIR"
echo "Job finished. Outputs and logs are in $SAVE_DIR"