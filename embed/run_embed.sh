#!/bin/bash -l

# Three entry points (see each script's docstring):
#   embed.py         one h5ad + a CancerFoundation checkpoint -> a directory
#   embed_splits.py  test rows and/or a train subset of each modality's split -> embeddings/<model>/
#   embed_any.py     one h5ad + any registered embedder (pca, bulkformer, ...) -> a directory
# Set UNIFIEDFM_ROOT to use another experiments root.
MODEL="unified_paired_tissue_s0"
SPLIT="tissue_s0"

python embed_splits.py \
    --model "$MODEL" \
    --split "$SPLIT" \
    --modalities sc bulk pseudobulk \
    --parts test train_subset \
    --train-subset 5000 \
    --stratify-by tissue_general

# PCA baseline on the same rows (written as embeddings/pca/):
# python embed_splits.py --embedder pca --split "$SPLIT" --modalities sc bulk pseudobulk --parts test train_subset

# One arbitrary file:
# python embed.py --checkpoint /path/model.ckpt --input data.h5ad --out-dir out/
# python embed_any.py --embedder pca --input data.h5ad --out-dir out/ --option n_pcs=50
