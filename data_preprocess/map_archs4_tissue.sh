#!/bin/bash -l
#SBATCH --job-name=archs4_tissue
#SBATCH --output=slurm_outputs/archs4_tissue_%j.out
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#
# ARCHS4 sample -> CellxGene tissue_general map, then relabel tissue_general in an h5ad.
#
#   cd data_preprocess && sbatch map_archs4_tissue.sh                 # build the map + apply it
#   cd data_preprocess && sbatch map_archs4_tissue.sh --map-only
#   cd data_preprocess && sbatch map_archs4_tissue.sh --apply-only    # reuse the saved map
#   cd data_preprocess && sbatch map_archs4_tissue.sh --dry-run       # apply: statistics only
#
# apply edits only obs/$COLUMN (X, var, uns and the other obs columns stay byte for byte).
# OUT empty = relabel $H5AD in place (via a temporary copy next to it, so the disk needs
# room for one more copy of the file); set OUT to write a relabelled copy instead.
# The conda env needs obonet, networkx, cellxgene-ontology-guide, h5py, pandas, anndata.

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-bulkFM}
ARCHS4_DIR=/cluster/work/boeva/rquiles/data/ARCHS4-data/data/processed/archs4
METADATA=$ARCHS4_DIR/metadata.csv
UBERON_OBO=/cluster/work/boeva/rquiles/ontologies/uberon-basic.obo
MAP_DIR=/cluster/work/boeva/rquiles/CancerFoundationFusion/data_preprocess/archs4_tissue_map
H5AD=/cluster/work/boeva/rquiles/unifiedFM_experiments/data/bulk/BULK_pretraining.h5ad
OUT=""                    # "" = --overwrite
COLUMN=tissue_general
ID_COLUMN=""              # "" = match on the obs index

RUN_MAP=1
RUN_APPLY=1
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --map-only)   RUN_APPLY=0 ;;
        --apply-only) RUN_MAP=0 ;;
        --dry-run)    DRY_RUN=1 ;;
        *) echo "unknown option: $arg (use --map-only, --apply-only, --dry-run)"; exit 1 ;;
    esac
done

set +u   # conda's activation scripts read unset variables
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u
mkdir -p slurm_outputs
echo "Environment: conda '$CONDA_ENV' ($(command -v python))"

if [[ "$RUN_MAP" -eq 1 ]]; then
    echo "=== build-map $(date '+%F %T') ==="
    python -u map_archs4_tissue.py build-map \
        --metadata "$METADATA" \
        --uberon-obo "$UBERON_OBO" \
        --out-dir "$MAP_DIR"
fi

if [[ "$RUN_APPLY" -eq 1 ]]; then
    echo "=== apply $(date '+%F %T') ==="
    APPLY_ARGS=(--h5ad "$H5AD" --map "$MAP_DIR" --column "$COLUMN")
    [[ -n "$ID_COLUMN" ]] && APPLY_ARGS+=(--id-column "$ID_COLUMN")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        APPLY_ARGS+=(--dry-run)
    elif [[ -n "$OUT" ]]; then
        APPLY_ARGS+=(--out "$OUT")
    else
        APPLY_ARGS+=(--overwrite)
    fi
    python -u map_archs4_tissue.py apply "${APPLY_ARGS[@]}"
fi

echo "=== done $(date '+%F %T') ==="
