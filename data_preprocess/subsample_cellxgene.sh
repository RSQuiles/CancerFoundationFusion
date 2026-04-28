#!/bin/bash -l
#SBATCH --job-name=subsample
#SBATCH --output=slurm_outputs/subsample_%j.out
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/.bashrc
conda activate bulkFM

python -u cellxgene_subsample.py