#!/bin/bash -l
#SBATCH --job-name=preprocess_bulk
#SBATCH --output=slurm_outputs/preprocess_bulk_%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u bulk_sc_data_preprocessing.py \
    --bulk-only \
    --bulk-path /cluster/work/boeva/rquiles/bulkFM-data/data/processed/archs4 \
    --h5ad-path /cluster/work/boeva/rquiles/data/bulk_only \
	--data-path /cluster/work/boeva/rquiles/data/bulk_only/pipeline_ready \
	--obs-columns tissue_general assay \
