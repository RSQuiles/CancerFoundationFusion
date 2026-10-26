#!/bin/bash -l

# Experiments root (data/, checkpoints/, embeddings/, results/); overrides the default.
export UNIFIEDFM_ROOT="directory/of/unifiedFM_experiments"

cd "$(dirname "$0")/.." || exit 1

python -u -m experiments.launch \
	--config ./experiments/configs/example_unifiedfm.json \
