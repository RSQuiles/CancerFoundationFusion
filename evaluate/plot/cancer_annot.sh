#!/bin/bash -l

# Improve CUDA traceback
CUDA_LAUNCH_BLOCKING=1 python -u downstream_annotation.py \
    --config ./downstream_config.yaml \