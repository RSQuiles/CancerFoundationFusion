#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_ablation
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

source $surv

python plot_ablation_benchmark.py \
    --config comparison_non_precomputed_experiments.yaml
