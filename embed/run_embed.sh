#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --job-name=embed
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# Three entry points (see each script's docstring):
#   embed.py         one h5ad + a CancerFoundation checkpoint -> a directory
#   embed_splits.py  test rows and/or a train subset of each modality's split -> embeddings/<model>/
#   embed_any.py     one h5ad + any registered embedder (pca, bulkformer, ...) -> a directory
# Set UNIFIEDFM_ROOT to use another experiments root.
#
#   bash run_embed.sh --bionemo            # inside the bionemo container (CancerFoundation models)
#   sbatch run_embed.sh --bionemo
#   bash run_embed.sh                      # with whatever python is active (e.g. a conda env)
# Any other argument is passed on to embed_splits.py (later flags win, e.g. --model other).
# BulkFormer needs its own conda env: run it WITHOUT --bionemo after `conda activate bulkformer-latest`.

set -euo pipefail

USE_BIONEMO=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--bionemo" ]]; then
        USE_BIONEMO=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

PROJECT_ROOT="/cluster/work/boeva/rquiles/CancerFoundationFusion"
IMAGE="/cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif"
# Everything the job reads or writes (code, experiments root, symlink targets) must be under a bind.
BINDS="/cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles"

MODEL="bulk_only"
SPLIT="leiden_s0"

SCRIPT_ARGS=(
    --model "$MODEL"
    --split "$SPLIT"
    --modalities bulk
    --parts test
    --test-subset 20000
    --stratify-by tissue_general
)

if [[ "$USE_BIONEMO" -eq 1 ]]; then
    echo "Running inside the bionemo container ($IMAGE)"
    singularity run --nv \
        --pwd "$PROJECT_ROOT" \
        --bind "$BINDS" \
        "$IMAGE" \
        python -u "$PROJECT_ROOT/embed/embed_splits.py" "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    echo "Running with $(command -v python)"
    python -u "$PROJECT_ROOT/embed/embed_splits.py" "${SCRIPT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

# PCA baseline on the same rows (written as embeddings/pca/):
# python embed_splits.py --embedder pca --split "$SPLIT" --modalities sc bulk pseudobulk --parts test train_subset

# One arbitrary file:
# python embed.py --checkpoint /path/model.ckpt --input data.h5ad --out-dir out/
# python embed_any.py --embedder pca --input data.h5ad --out-dir out/ --option n_pcs=50
