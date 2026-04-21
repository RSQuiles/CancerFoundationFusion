#!/bin/bash -l
#SBATCH --job-name=esm3_embs
#SBATCH --output=slurm_outputs/esm3_embs_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/.bashrc
conda activate bulkFM

python -u protein_embeddings.py \
    --https \
    --save-path /cluster/work/boeva/rquiles/data/pretrained_homo_emb.parquet \