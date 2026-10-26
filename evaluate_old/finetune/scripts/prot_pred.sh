#!/bin/bash -l

# Improve CUDA traceback
CUDA_LAUNCH_BLOCKING=1 python -u ../run_downstream_task.py \
    --config ../configs/proteome_pred_config.yaml \