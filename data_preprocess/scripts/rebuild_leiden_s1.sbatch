#!/bin/bash -l
#SBATCH --job-name=rebuild_leiden_s1
#SBATCH --output=rebuild_leiden_s1_%j.out
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#
# Linked train/test splits + every training store for leiden_s1, in one allocation.
#
#   sbatch data_preprocess/scripts/rebuild_leiden_s1.sbatch                  # bionemo container
#   sbatch data_preprocess/scripts/rebuild_leiden_s1.sbatch --conda [ENV]    # conda env (default bulkFM)
#   CONDA_ENV=myenv sbatch data_preprocess/scripts/rebuild_leiden_s1.sbatch --conda
#
# The conda env needs what the container has: scanpy + igraph/leidenalg (Leiden),
# anndata, h5py, pandas/pyarrow, and bionemo-scdl with torch (the store builds).
#
# Single cells are the root of every link (pseudobulk_id: cells <-> Census pseudobulk;
# paired: cell-line cells <-> paired pseudobulk <-> paired bulk), so they are split
# first and the other modalities inherit:
#   1. single_cell  Leiden strata x linked flags, pseudobulk / pair groups kept whole
#   2. bulk         paired rows follow their cells; the rest split the same way
#   3. pseudobulk   follows its cells (pseudobulk_id), else its bulk partner (paired)
#   4. check_split_links.py   stops the job if a link key spans train and test
#   5. build_mix    joint store sc + bulk + pseudobulk (its vocabulary is reused below)
#   6. single-modality stores with the joint vocabulary, so all models share the genes
#   7. check_store_losses.py  report of what each loss finds in the joint store
#
# --stratify-links puts ~test-fraction of the pairs / pseudobulks WHOLE in test, so they
# are unseen during training. Values 0, -1, empty and NaN mean "no link" everywhere
# (utils.sampling.link_keys); Census pseudobulk 0 therefore loses its link (1 of ~20k).
# Every output is overwritten (--overwrite): check the paths below first.

set -euo pipefail

# ---- inputs --------------------------------------------------------------------------
ROOT=/cluster/work/boeva/rquiles/unifiedFM_experiments
DATA=$ROOT/data
SPLIT=leiden_s1
SEED=1
FRACTIONS=(train=0.9 test=0.1)
MIX=sc_bulk_pb_${SPLIT}
MAX_LEAK=0                          # link keys allowed to span parts before the job stops

SC_H5AD=$DATA/single_cell/single_cell.h5ad
BULK_H5AD=$DATA/bulk/BULK_pretraining.h5ad
PB_H5AD=$DATA/pseudobulk/pseudobulk.h5ad

STORE_COLUMNS=(tissue_general)                      # single-modality stores
MIX_COLUMNS=(tissue_general paired pseudobulk_id)   # joint store

# ---- environment ---------------------------------------------------------------------
PROJECT=/cluster/work/boeva/rquiles/CancerFoundationFusion
IMAGE=/cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif
export TMPDIR=/cluster/work/boeva/rquiles/tmp
mkdir -p "$TMPDIR"

# --conda [ENV]: run every step in a conda env instead of the container.
USE_CONDA=1
CONDA_ENV=${CONDA_ENV:-bulkFM}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda)
            USE_CONDA=1
            if [[ $# -gt 1 && "$2" != --* ]]; then CONDA_ENV=$2; shift; fi ;;
        *) echo "unknown option: $1 (use --conda [ENV])"; exit 1 ;;
    esac
    shift
done

if [[ "$USE_CONDA" -eq 1 ]]; then
    set +u   # conda's activation scripts read unset variables
    source ~/.bashrc
    conda activate "$CONDA_ENV"
    set -u
    echo "Environment: conda '$CONDA_ENV' ($(command -v python))"
    py() {   # python in the conda env, from data_preprocess/
        (cd "$PROJECT/data_preprocess" && python -u "$@")
    }
else
    echo "Environment: container $IMAGE"
    py() {   # python inside the bionemo container (scanpy, SCDL), from data_preprocess/
        singularity run \
            --pwd "$PROJECT/data_preprocess" \
            --bind /cluster/work/boeva/rquiles:/cluster/work/boeva/rquiles \
            "$IMAGE" python -u "$@"
    }
fi

STEP=0
banner() {
    STEP=$((STEP + 1))
    echo
    echo "================================================================================"
    echo "[$STEP] $*    ($(date '+%F %T'))"
    echo "================================================================================"
}

for f in "$SC_H5AD" "$BULK_H5AD" "$PB_H5AD"; do
    [[ -f "$f" ]] || { echo "missing input: $f"; exit 1; }
done

COMMON=(--root "$ROOT" --split-name "$SPLIT" --fractions "${FRACTIONS[@]}" --seed "$SEED"
        --cluster leiden --overwrite)

# ---- 1-3. splits ---------------------------------------------------------------------
banner "single_cell split (root of every link)"
py split_train_test.py "${COMMON[@]}" --modality single_cell --input "$SC_H5AD" \
    --out-dir "$DATA/single_cell/splits/$SPLIT" \
    --group-by pseudobulk_id:0,-1 paired:0 \
    --stratify-links pseudobulk_id:0,-1 paired:0

banner "bulk split (paired rows inherit from single_cell)"
py split_train_test.py "${COMMON[@]}" --modality bulk --input "$BULK_H5AD" \
    --out-dir "$DATA/bulk/splits/$SPLIT" \
    --group-by paired:0 --stratify-links paired:0 \
    --inherit-from "single_cell:$SPLIT" --link-col paired

banner "pseudobulk split (inherits from single_cell, then bulk)"
py split_train_test.py "${COMMON[@]}" --modality pseudobulk --input "$PB_H5AD" \
    --out-dir "$DATA/pseudobulk/splits/$SPLIT" \
    --group-by paired:0 pseudobulk_id:0,-1 --stratify-links paired:0 pseudobulk_id:0,-1 \
    --inherit-from "single_cell:$SPLIT" "bulk:$SPLIT" --link-col pseudobulk_id paired

# ---- 4. links ------------------------------------------------------------------------
banner "link check across the three splits"
py check_split_links.py --root "$ROOT" --split "$SPLIT" \
    --modalities single_cell bulk pseudobulk --links pseudobulk_id:0,-1 paired:0 \
    --max-leak "$MAX_LEAK"

# ---- 5. joint store ------------------------------------------------------------------
banner "joint store $MIX"
py build_mix.py --root "$ROOT" --name "$MIX" --overwrite \
    --source "single_cell:$SPLIT" "bulk:$SPLIT" "pseudobulk:$SPLIT" \
    --obs-columns "${MIX_COLUMNS[@]}" --pb-id-column pseudobulk_id
VOCAB=$DATA/pipeline_ready/$MIX/vocab.json

# ---- 6. single-modality stores (same vocabulary) ------------------------------------
build_store() {   # <modality dir> <modality label> <source h5ad> [extra args]
    local mod=$1 label=$2 src=$3; shift 3
    banner "$mod store (train part, joint vocabulary)"
    py bulk_sc_data_preprocessing.py --h5ad-path "$src" \
        --rows "$DATA/$mod/splits/$SPLIT/train.txt" \
        --data-path "$DATA/$mod/pipeline_ready/$SPLIT" \
        --vocab-path "$VOCAB" --default-modality "$label" \
        --obs-columns "${STORE_COLUMNS[@]}" --overwrite "$@"
}
build_store single_cell sc "$SC_H5AD"
build_store bulk bulk "$BULK_H5AD"
build_store pseudobulk pseudobulk "$PB_H5AD" --pb-id-column pseudobulk_id

# ---- 7. report -----------------------------------------------------------------------
banner "losses supported by $MIX (report only)"
py check_store_losses.py --mix "$MIX" --root "$ROOT" --pb-source precomputed \
    --conditions tissue_general --batch-size 64 --bulk-ratio 0.35 --pb-ratio 0.35 \
    || echo "[warn] check_store_losses reports a loss as not ready (see above)"

banner "store sizes (for the step budget in experiments/configs/leiden_s1_baselines.json)"
for store in "$DATA/single_cell/pipeline_ready/$SPLIT" "$DATA/bulk/pipeline_ready/$SPLIT" \
             "$DATA/pseudobulk/pipeline_ready/$SPLIT" "$DATA/pipeline_ready/$MIX"; do
    rows=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['n_rows'])" "$store/store.json" 2>/dev/null \
           || python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['n_rows'])" "$store/mix.json" 2>/dev/null \
           || echo "?")
    echo "  $store: $rows rows"
done

echo
echo "Done ($(date '+%F %T'))."
