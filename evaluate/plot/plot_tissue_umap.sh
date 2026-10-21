#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=plot_tissue_umap
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G

# CPU only: this reads obs + X_pca from the h5ad and runs scanpy/matplotlib.
# The expression matrix is never loaded unless --recompute-pca is passed.

set -euo pipefail

# Default: use singularity
USE_LOCAL=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL=1
    fi
done

SCRIPT_ARGS=(
    --adata /cluster/work/boeva/rquiles/data/eheiss_RAW/pretraining_bulk_RAW.h5ad
    --color tissue_general
    --sample-size 75_000
    --out-dir ./umap
    --save-h5ad
)

# --skip-unknown          # leave off: seeing where 'unknown' cells sit is the point
# --recompute-pca         # only if add_fields.py ran without the centroid fill
# --color tissue_general modality

if [[ "$USE_LOCAL" -eq 1 ]]; then
    echo "Running locally (no singularity)"
    python -u plot_tissue_umap.py "${SCRIPT_ARGS[@]}"
else
    echo "Running with singularity"
    singularity run \
        --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/plot \
        --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
        /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
        python -u plot_tissue_umap.py "${SCRIPT_ARGS[@]}"
fi
