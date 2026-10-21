#!/bin/bash -l
#SBATCH --job-name=add_fields
#SBATCH --output=slurm_outputs/add_fields_%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G

source ~/.bashrc
conda activate bulkFM

OUT_DIR="/cluster/work/boeva/rquiles/data/eheiss_RAW_tissue"

echo "=== Step 4: Inferring tissue_general ==="
python -u add_fields.py \
    --input /cluster/work/boeva/rquiles/eheiss_datasets/bulk/pretraining_bulk_RAW.h5ad \
    --no-tissue-fill \
    --tissue-fill-lognorm \
    --output $OUT_DIR \
    --field tissue_general \
    --cellxgene-only

:'
# --- Step 1: Add modality=sc to SC file ---
echo "=== Step 1: Adding modality=sc ==="
python -u add_fields.py \
    --input /cluster/work/boeva/rquiles/eheiss_datasets/sc/pretraining_sc_RAW.h5ad \
    --output $OUT_DIR \
    --field modality=sc

# --- Step 2: Add modality=pseudobulk to pseudobulk file ---
echo "=== Step 2: Adding modality=pseudobulk ==="
python -u add_fields.py \
    --input /cluster/work/boeva/rquiles/eheiss_datasets/pseudo_bulk/pseudo_bulk_RAW.h5ad \
    --output $OUT_DIR \
    --field modality=pseudobulk

# --- Step 3:Add modality=bulk for bulk file ---
echo "=== Step 3: Adding modality=bulk ==="
python -u add_fields.py \
    --input /cluster/work/boeva/rquiles/eheiss_datasets/bulk/pretraining_bulk_RAW.h5ad \
    --output $OUT_DIR \
    --field modality=bulk
'

echo "=== Done. Outputs in $OUT_DIR ==="