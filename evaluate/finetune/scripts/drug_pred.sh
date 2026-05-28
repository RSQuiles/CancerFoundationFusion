#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --job-name=drug_pred
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

# Improve CUDA traceback
CUDA_LAUNCH_BLOCKING=1 python -u ../run_downstream_task.py \
    --config ../configs/drug_sensitivity_v2_config.yaml \