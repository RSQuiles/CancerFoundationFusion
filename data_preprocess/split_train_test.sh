#!/bin/bash -l
#SBATCH --job-name=split
#SBATCH --output=slurm_outputs/split_%j.out
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G

export TMPDIR=/cluster/work/boeva/rquiles/tmp
mkdir -p $TMPDIR

# Default, do not use Singularity
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

DATA_DIR=/cluster/work/boeva/rquiles/unifiedFM_experiments/data
SPLIT_NAME=leiden_s0

# Writes train.txt, test.txt, split.json (and clusters.csv with --cluster leiden) to --out-dir.
# The source h5ad is only read; expression values are never normalized.
SCRIPT_ARGS=(
    --input $DATA_DIR/bulk/BULK_pretraining.h5ad
    --out-dir $DATA_DIR/bulk/splits/$SPLIT_NAME
    --fractions train=0.8 test=0.2
    --cluster leiden
    --group-by paired:0
    --seed 0
    --overwrite
    # --build-store
    # --store-dir $DATA_DIR/bulk/pipeline_ready/$SPLIT_NAME
)

# --stratify-by tissue_general   stratifies by fields in obs DataFrame
# --group-by donor_id paired:0   groups that never straddle parts (paired:0 = 0 means "no group")
# --cluster leiden               also stratify on Leiden clusters (add --cluster-rep X_pca to reuse an embedding)
# --filter tissue_general=lung,breast   only split these rows
# --write-h5ad                   also write <part>.h5ad for every part (reads the full source again)
# --overwrite                    replaces the existing out-dir (it is deleted first, so check the path)
#
# Layout shorthand instead of --input/--out-dir (needs data/<modality>/<modality>.h5ad):
#   --modality single_cell --split-name $SPLIT_NAME --stratify-by tissue_general
#
# Child split that inherits a parent's parts through a link column:
#   --modality pseudobulk --split-name $SPLIT_NAME --inherit-from single_cell:$SPLIT_NAME --link-col pseudobulk_id
#
# Convert one part (default: train) into a training store
#   --build-store

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally"
    source ~/.bashrc
    conda activate bulkFM
    python -u split_train_test.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/data_preprocess \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u split_train_test.py "${SCRIPT_ARGS[@]}"
fi