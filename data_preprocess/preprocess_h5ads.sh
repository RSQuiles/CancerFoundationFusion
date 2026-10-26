#!/bin/bash -l
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_outputs/preprocess_%j.out
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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

# Every row of every file, values as stored (nothing is normalized). To build a store
# from one part of a split instead, use split_train_test.py --build-store or build_mix.py.
SCRIPT_ARGS=(
    --h5ad-path /cluster/work/boeva/rquiles/data/eheiss_RAW_paired
    --data-path /cluster/work/boeva/rquiles/data/eheiss_RAW_paired/pipeline_ready
    --obs-columns tissue_general paired
    --pb-id-column pseudobulk_id
)

# --rows train.txt           keep only these obs names
# --vocab-path vocab.json    reuse an existing vocabulary (else: intersection of the files)
# --gene-column feature_name var column holding gene names (instead of rewriting the index)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u bulk_sc_data_preprocessing.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    srun singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/data_preprocess \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif \
        python -u bulk_sc_data_preprocessing.py "${SCRIPT_ARGS[@]}"
fi
