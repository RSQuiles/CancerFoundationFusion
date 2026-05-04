#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_outputs/preprocess_%j.out
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u bulk_sc_data_preprocessing.py \
    --h5ad-path /cluster/work/boeva/rquiles/data/cellxgene_bulk \
    --data-path /cluster/work/boeva/rquiles/data/cellxgene_bulk/pipeline_ready \
    --obs-columns tissue_general assay \
