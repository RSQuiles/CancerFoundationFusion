#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=plot_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G


srun singularity run \
    --pwd /cluster/work/boeva/rquiles/CancerFoundation/evaluate \
    --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python -u umaps.py \
    --run-name bulk_only_8657635 \
    --flavor seurat_v3 \
    --color tissue_general assay \
    --out-dir ./umap_plots \
    --out-prefix bulk_chunk_0 \
    --adata /cluster/work/boeva/rquiles/data/bulk_only/bulk_chunk_0000.h5ad \
    # --adata /cluster/work/boeva/rquiles/data/bulk_only/combined_bulk_reduced.h5ad \
