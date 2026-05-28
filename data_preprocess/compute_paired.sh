#!/bin/bash -l
#SBATCH --job-name=preprocess_bulk
#SBATCH --output=slurm_outputs/paired_pseudobulk_%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/.bashrc
conda activate bulkFM

python -u pseudobulk_paired_generation.py \
    --input-h5ad /cluster/work/boeva/bulkFM/data/processed/paired_samples.h5ad \
    --output-dir /cluster/work/boeva/rquiles/data/paired_dataset \
    --n-pseudobulk 50 \
    --extra-obs-columns dataset \
