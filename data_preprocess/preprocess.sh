#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --output=outputs/preprocess_%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u bulk_sc_data_preprocessing.py \
	--h5ad-path /cluster/work/boeva/rquiles/data/small_partition \
        --data-path /cluster/work/boeva/rquiles/data/small_partition/pipeline_ready \
	--obs-columns tissue_general assay
