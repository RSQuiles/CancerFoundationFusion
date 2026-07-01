#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --job-name=build_adata
#SBATCH --output=./slurm_outputs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

ABLATION_DIR="/cluster/work/boeva/rquiles/outputs/save_CFF/ablation_paired_counts"

SCRIPT_ARGS=(
    --adata-dir /cluster/work/boeva/rquiles/data/paired_dataset_counts/pipeline_ready/h5ads
    --sample-size 5000
    --out $ABLATION_DIR/eval.h5ad
    --ablation-dir $ABLATION_DIR
)

echo "Running with singularity"
singularity run \
    --pwd /cluster/work/boeva/rquiles/CancerFoundationFusion/evaluate/check \
    --bind /cluster \
    --nv /cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif \
    python -u build_eval_adata.py "${SCRIPT_ARGS[@]}"
