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
#   cd data_preprocess && sbatch map_archs4_tissue.sh --curate-only [--dry-run]
#
# --curate (after apply) / --curate-only: curated tissue_general for the class-aware losses
# (single cell and pseudobulk: tissue "0" of the paired datasets is renamed "paired" first)
# and the evaluation. Bulk: the old column moves to tissue_noncurated, rows without a map
# label become "unknown". Single cell and pseudobulk only gain tissue_label_source. Edits
# obs in place, no copy; refuses a file that was curated already. Set NONCURATED_FROM to
# an original copy of the bulk h5ad to keep the labels "apply --overwrite" replaced.
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
# curate
SC_H5AD=/cluster/work/boeva/rquiles/unifiedFM_experiments/data/single_cell/single_cell.h5ad
PB_H5AD=/cluster/work/boeva/rquiles/unifiedFM_experiments/data/pseudobulk/pseudobulk.h5ad
NONCURATED_FROM=""        # original BULK_pretraining.h5ad from before apply, if one exists
KEYWORD_COLUMNS=""        # e.g. "source_name description": keep keyword-reproduced heuristic labels

RUN_MAP=1
RUN_APPLY=1
RUN_CURATE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --map-only)    RUN_APPLY=0 ;;
        --apply-only)  RUN_MAP=0 ;;
        --curate)      RUN_CURATE=1 ;;
        --curate-only) RUN_MAP=0; RUN_APPLY=0; RUN_CURATE=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        *) echo "unknown option: $arg (use --map-only, --apply-only, --curate, --curate-only, --dry-run)"; exit 1 ;;
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

if [[ "$RUN_CURATE" -eq 1 ]]; then
    MODE=(--in-place)
    [[ "$DRY_RUN" -eq 1 ]] && MODE=(--dry-run)
    echo "=== curate bulk $(date '+%F %T') ==="
    CURATE_ARGS=(--h5ad "$H5AD" --modality bulk --map "$MAP_DIR" --column "$COLUMN" --vocab-h5ad "$PB_H5AD")
    [[ -n "$ID_COLUMN" ]] && CURATE_ARGS+=(--id-column "$ID_COLUMN")
    [[ -n "$NONCURATED_FROM" ]] && CURATE_ARGS+=(--noncurated-from "$NONCURATED_FROM")
    # shellcheck disable=SC2206
    [[ -n "$KEYWORD_COLUMNS" ]] && CURATE_ARGS+=(--trust-heuristic --keyword-columns $KEYWORD_COLUMNS)
    python -u map_archs4_tissue.py curate "${CURATE_ARGS[@]}" "${MODE[@]}"
    for pair in "pseudobulk:$PB_H5AD" "single_cell:$SC_H5AD"; do
        # The paired datasets' cells (and pseudobulks) carry tissue "0" (the store fill for
        # "none"): name them "paired". Nothing happens when there is no "0".
        echo "=== rename tissue 0 -> paired in ${pair%%:*} $(date '+%F %T') ==="
        python -u map_archs4_tissue.py rename-category --h5ad "${pair#*:}" --column "$COLUMN"             --rename 0=paired --source-value paired "${MODE[@]}"
        echo "=== curate ${pair%%:*} $(date '+%F %T') ==="
        python -u map_archs4_tissue.py curate --h5ad "${pair#*:}" --modality "${pair%%:*}"             --column "$COLUMN" "${MODE[@]}"
    done
fi

echo "=== done $(date '+%F %T') ==="
