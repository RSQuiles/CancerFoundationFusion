#!/bin/bash -l
#SBATCH --job-name=preprocess_bulk
#SBATCH --output=outputs/preprocess_bulk_%j.out
#SBATCH --time=5:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u bulk_sc_data_preprocessing.py \
	--data-path /cluster/work/boeva/rquiles/data/bulk_only/pipeline_ready \
        --h5ad-path /cluster/work/boeva/rquiles/data/bulk_only \
	--obs-columns tissue_general assay \
