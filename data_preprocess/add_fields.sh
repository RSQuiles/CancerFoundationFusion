#!/bin/bash -l
#SBATCH --job-name=add_fields
#SBATCH --output=slurm_outputs/add_fields_%j.out
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G

source ~/.bashrc
conda activate bulkFM

OUT_DIR="/cluster/work/boeva/rquiles/data/eheiss_RAW"

# --- Step 3: Infer tissue_general for bulk file ---
echo "=== Step 3: Inferring tissue_general ==="
python -u add_fields.py \
    --input /cluster/work/boeva/rquiles/eheiss_datasets/bulk/pretraining_bulk_RAW.h5ad \
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
'

echo "=== Done. Outputs in $OUT_DIR ==="