#!/bin/bash -l

export work="directory/where/checkpoints/are/saved"
export data_dir="directory/where/data/is/stored"

python -u ablate.py \
	--config ./configs/paired_config_cscs.json \
