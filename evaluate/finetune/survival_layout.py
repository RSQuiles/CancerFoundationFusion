"""Where survival-function CSVs live on disk — the one place that decides.

Layout::

    {survboard_results_dir}/{COHORT}/{CANCER}/{ablation}_{model}/split_{fold}.csv
    e.g. .../TCGA/BRCA/ablation_big_condition_unified/split_5.csv

The ``{ablation}_`` prefix is load-bearing. The directory used to be named after the
model alone, but model names repeat across ablation directories (``unified``, ``dat``
and ``bulk_baseline`` each occur in several), and every ablation shares one
``survboard_results_dir`` — so several models wrote to and read from the same CSVs,
and whichever survival run finished last was what all of them got scored on.
``_clear_survival_csvs`` deleted by that same shared name, so runs also wiped each
other's output.

Both sides of the contract import from here — ``tasks/survboard_task.py`` writes,
``tasks/evaluate_survboard_metrics.py`` and
``scripts/run_ablation_survboard_metrics.py`` read — so the two cannot drift. Keep
this module free of torch/pycox/anndata imports (same discipline as
``normalization.py``): it makes the naming testable offline, without a checkpoint or
the SurvBoard environment.

There is no fallback to the old bare-``{model}`` layout. CSVs written before this
change are exactly the ambiguous ones, so they are orphaned deliberately; re-run the
survival task rather than trying to read them.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SEPARATOR = "_"

# A checkpoint may sit directly in the model directory or one level down, since
# run_ablation_downstream._find_best_ckpt searches both.
_CKPT_SUBDIR = "checkpoints"


def model_name_from_checkpoint(checkpoint_path: str | Path) -> str:
    """Model directory name for a checkpoint.

    ``{ablation}/{model}/x.ckpt`` and ``{ablation}/{model}/checkpoints/x.ckpt`` both
    give ``{model}``. Without the second case the name comes out as ``"checkpoints"``.
    """
    parent = Path(checkpoint_path).parent
    if parent.name == _CKPT_SUBDIR:
        parent = parent.parent
    return parent.name


def ablation_name(
    checkpoint_path: str | Path | None = None,
    ablation_dir:    str | Path | None = None,
) -> str | None:
    """Name of the ablation a model belongs to, or None if it cannot be determined.

    The checkpoint wins over ``ablation_dir``: its grandparent is ground truth for
    which ablation a model came from, whereas ``ablation_dir`` is a config key that
    goes stale (``survival_pred_config.yaml`` ships one that disagrees with its own
    ``pretrained_model_path``). ``ablation_dir`` is what covers the PCA baseline,
    which has no checkpoint of its own.
    """
    if checkpoint_path:
        parent = Path(checkpoint_path).parent
        if parent.name == _CKPT_SUBDIR:
            parent = parent.parent
        name = parent.parent.name
        if name:
            return name
    if ablation_dir:
        name = Path(ablation_dir).name
        if name:
            return name
    return None


def storage_name(
    model_name:      str,
    checkpoint_path: str | Path | None = None,
    ablation_dir:    str | Path | None = None,
) -> str:
    """Directory name under ``{results_dir}/{cohort}/{cancer}/`` for one model.

    Falls back to the bare model name when the ablation cannot be determined — the
    old, collision-prone layout — and says so loudly, because such CSVs are the ones
    that silently get shared between ablations.
    """
    ablation = ablation_name(checkpoint_path, ablation_dir)
    if not ablation:
        log.warning(
            "Cannot determine the ablation for model '%s' (checkpoint_path=%r, "
            "ablation_dir=%r); falling back to the bare model name. These CSVs will "
            "be shared with any other ablation holding a model of the same name.",
            model_name, str(checkpoint_path) if checkpoint_path else None,
            str(ablation_dir) if ablation_dir else None,
        )
        return model_name
    return f"{ablation}{SEPARATOR}{model_name}"


def survival_csv_dir(
    results_dir:  str | Path,
    cohort:       str,
    cancer:       str,
    storage_name: str,
) -> Path:
    """Directory holding ``split_{fold}.csv`` for one model and one cancer block."""
    return Path(results_dir) / cohort / cancer / storage_name
