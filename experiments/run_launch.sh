#!/bin/bash -l

cd "$(dirname "$0")/.." || exit 1

python -u -m experiments.launch \
	--config ./experiments/configs/example_unifiedfm.json \
