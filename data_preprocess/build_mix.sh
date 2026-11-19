#!/bin/bash -l
#SBATCH --job-name=build_mix
#SBATCH --output=slurm_outputs/build_mix_%j.out
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G

export TMPDIR=/cluster/work/boeva/rquiles/tmp
mkdir -p $TMPDIR

USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

SPLIT_NAME=leiden_s0

SCRIPT_ARGS=(
    --name sc_bulk_pb_$SPLIT_NAME
    --source single_cell:$SPLIT_NAME bulk:$SPLIT_NAME pseudobulk:$SPLIT_NAME
    --obs-columns tissue_general assay paired modality
    --pb-id-column pseudobulk_id
    --overwrite
)

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (host enviroment)"
    python -u build_mix.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with conda enviroment"
    source ~/.bashrc
    conda activate bulkFM
    python -u build_mix.py "${SCRIPT_ARGS[@]}"

fi