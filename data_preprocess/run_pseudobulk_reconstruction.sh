#!/bin/bash -l
#SBATCH --job-name=reconstruct
#SBATCH --output=slurm_outputs/reconstruct_%j.out
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

python -u reconstruct_pseudobulk_cell_map.py \
	--pseudo-bulk-dir /cluster/work/boeva/eheiss/datasets/pseudo_bulk \
	--pseudo-bulk-h5ad /cluster/work/boeva/rquiles/data/eheiss_RAW/pseudo_bulk_RAW.h5ad \
	--verify 1 \
	--rebuild \
	--max-cells-per-pb 30 \
	--output-dir /cluster/work/boeva/rquiles/data/eheiss_RAW_matched
