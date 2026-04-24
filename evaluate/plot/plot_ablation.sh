#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_ablation
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python ablation_benchmark.py \
    --ablation-dir /cluster/work/boeva/rquiles/outputs/save_CFF/ablation_test
