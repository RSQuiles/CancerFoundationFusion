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

SCRIPT_ARGS=(
    --h5ad-path /cluster/work/boeva/rquiles/data/eheiss_RAW_paired
    --data-path /cluster/work/boeva/rquiles/data/eheiss_RAW_paired/pipeline_ready
    --obs-columns tissue_general
)

# --obs-columns tissue_general assay

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
