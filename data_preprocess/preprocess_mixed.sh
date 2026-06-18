#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_outputs/preprocess_mixed_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u bulk_sc_data_preprocessing.py \
    --mixed \
    --bulk-path /cluster/work/boeva/rquiles/bulkFM-data/data/processed/archs4 \
    --normalize-bulk-tissues \
    --use-counts \
    --paired-h5ad /cluster/work/boeva/bulkFM/data/processed/paired_samples.h5ad \
    --paired-extra-obs-columns dataset \
    --h5ad-path /cluster/work/boeva/rquiles/data/paired_dataset_counts \
    --data-path /cluster/work/boeva/rquiles/data/paired_dataset_counts/pipeline_ready \
    --obs-columns tissue_general assay paired \
