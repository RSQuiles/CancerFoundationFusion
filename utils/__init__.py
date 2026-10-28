"""Project-wide helpers.

- ``utils.sampling``: obs filtering, stratified subsampling, group-aware train/test
  splits (numpy + pandas only).
- ``utils.training``: what used to be the top-level ``utils.py`` (``get_args``,
  ``save_resolved_config``, ``LossCSVLogger``, ``MyProgressBar``), plus
  ``KeepExistingCheckpoint``. It imports pytorch_lightning.
- ``utils.resume``: continuing a run in its checkpoint directory (checkpoint lookup,
  W&B run id, segment log). Torch-free at import time.

The training names are re-exported **lazily**, so ``from utils import get_args``
keeps working while ``from utils.sampling import ...`` never pulls in lightning or
torch.
"""

from __future__ import annotations

import importlib

_TRAINING_NAMES = {
    "get_args", "save_resolved_config", "LossCSVLogger", "MyProgressBar", "KeepExistingCheckpoint",
}

__all__ = sorted(_TRAINING_NAMES)


def __getattr__(name: str):
    if name in _TRAINING_NAMES:
        return getattr(importlib.import_module("utils.training"), name)
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
