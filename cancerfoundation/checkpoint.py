"""Backward-compatible checkpoint loading for inference.

``LightningModule.load_from_checkpoint`` re-invokes ``CancerFoundation.__init__``
with the hyperparameters stored in the checkpoint and then loads the weights into
the result. That breaks whenever the *code* moves in a way the hyperparameters do
not describe:

* a new required argument is added to ``CancerFoundation.__init__`` — every
  checkpoint saved before it raises ``TypeError: missing required argument``;
* an architecture dimension is recomputed at construction time rather than read
  from an hparam — e.g. the DAT modality head went from 3 to 2 classes, so old
  checkpoints raise a size mismatch that ``strict=False`` cannot absorb (it
  tolerates missing/unexpected keys, never a shape change on a shared key);
* ``--compile`` was used, so the keys carry an ``_orig_mod.`` prefix.

``load_for_inference`` reconciles all three. It is for evaluation and embedding
only: it deliberately drops the DAT discriminators, which are training-only, and
disables ``torch.compile``. Do not use it to resume training.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

log = logging.getLogger(__name__)

# state_dict prefixes that only matter during training. Weights under these are
# dropped from the reconstructed model, so a shape change in them is a note in the
# log rather than a failure. Anything NOT listed here is on the inference forward
# path and a mismatch there is raised.
_TRAINING_ONLY_PREFIXES = ("grad_reverse_discriminators.",)

_COMPILE_PREFIX = "_orig_mod."


def read_checkpoint_hparams(
    path: Union[str, Path], map_location: str = "cpu"
) -> Dict[str, Any]:
    """Read a checkpoint's stored hyperparameters without building a model.

    Args:
        path: Path to a Lightning ``.ckpt``.
        map_location: Passed through to ``torch.load``.

    Returns:
        The ``hyper_parameters`` dict, or ``{}`` if the checkpoint has none.
    """
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    return dict(ckpt.get("hyper_parameters", {}))


def _cli_defaults() -> Dict[str, Any]:
    """Argparse defaults, keyed by dest, used to fill in hparams a checkpoint predates.

    When an argument is added to the parser its default is chosen to preserve the
    previous behaviour, which makes it the right value for a checkpoint saved
    before the argument existed. Returns ``{}`` if ``utils_config`` is not
    importable (it lives at the repo root, not inside the package).
    """
    try:
        from utils_config import build_parser
    except ImportError:
        log.warning(
            "utils_config is not importable — cannot fill missing hyperparameters "
            "from CLI defaults. Put the repository root on sys.path if an old "
            "checkpoint fails to construct."
        )
        return {}

    return {
        action.dest: action.default
        for action in build_parser()._actions
        if action.dest != "help"
    }


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the ``_orig_mod.`` segments that ``torch.compile`` inserts."""
    if not any(_COMPILE_PREFIX in key for key in state_dict):
        return state_dict
    log.info("Checkpoint was saved with torch.compile — stripping '%s' from keys.", _COMPILE_PREFIX)
    return {key.replace(_COMPILE_PREFIX, ""): value for key, value in state_dict.items()}


def _encoded_conditions_from_state_dict(
    state_dict: Dict[str, Any], conditions: Optional[Dict[str, int]]
) -> Optional[list]:
    """Recover which conditions actually had a ``ConditionEncoder`` in the checkpoint.

    ``where_condition="end"`` sizes the expression and MVC decoders as
    ``d_model * (len(encoded_conditions) + 1)``, so this count has to match what
    the checkpoint was built with or the decoder weights will not fit. Reading it
    back from the saved ``condition_encoders.*`` keys pins it exactly, for both
    old checkpoints (which predate the ``encoded_conditions`` argument) and new
    ones. Order follows ``conditions`` to match the construction order in
    ``TransformerModule.__init__``.
    """
    if not conditions:
        return None

    marker = "condition_encoders."
    found = set()
    for key in state_dict:
        idx = key.find(marker)
        if idx == -1:
            continue
        remainder = key[idx + len(marker):]
        name = remainder.split(".", 1)[0]
        if name:
            found.add(name)

    if not found:
        return None
    return [name for name in conditions if name in found]


def _reconcile_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Make a checkpoint's hparams callable against the current ``__init__``.

    Drops arguments the signature no longer accepts, and fills in arguments the
    checkpoint predates — from the signature default where there is one,
    otherwise from the CLI default. Every substitution is logged, because a
    wrong guess on a shape-neutral argument (``norm_scheme``, say) would
    otherwise be silent.
    """
    from cancerfoundation.model.model import CancerFoundation

    signature = inspect.signature(CancerFoundation.__init__)
    parameters = {
        name: param
        for name, param in signature.parameters.items()
        if name != "self"
    }

    dropped = sorted(set(hparams) - set(parameters))
    if dropped:
        log.warning(
            "Checkpoint has %d hyperparameter(s) the current CancerFoundation.__init__ "
            "no longer accepts; ignoring them: %s",
            len(dropped), dropped,
        )
    reconciled = {name: value for name, value in hparams.items() if name in parameters}

    required_missing = [
        name
        for name, param in parameters.items()
        if name not in reconciled and param.default is inspect.Parameter.empty
    ]
    if not required_missing:
        return reconciled

    cli_defaults = _cli_defaults()
    unresolved = []
    for name in required_missing:
        if name in cli_defaults:
            reconciled[name] = cli_defaults[name]
            log.warning(
                "Checkpoint predates required argument '%s'; filling it with the CLI "
                "default %r.",
                name, cli_defaults[name],
            )
        else:
            unresolved.append(name)

    if unresolved:
        raise TypeError(
            f"Cannot reconstruct the model: CancerFoundation.__init__ requires "
            f"{unresolved}, which this checkpoint does not store and which have no "
            f"default in the signature or in utils_config.build_parser(). Give these "
            f"arguments a default in CancerFoundation.__init__ (choosing the value "
            f"that reproduces the previous behaviour) so old checkpoints stay loadable."
        )
    return reconciled


def _is_training_only(key: str) -> bool:
    return any(prefix in key for prefix in _TRAINING_ONLY_PREFIXES)


def _load_weights(model, state_dict: Dict[str, Any]) -> None:
    """Load weights, tolerating training-only drift and raising on anything else."""
    reference = model.state_dict()

    mismatched = []
    filtered = {}
    for key, value in state_dict.items():
        expected = reference.get(key)
        if expected is not None and hasattr(value, "shape") and expected.shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(expected.shape)))
            continue
        filtered[key] = value

    fatal = [entry for entry in mismatched if not _is_training_only(entry[0])]
    if fatal:
        details = "\n".join(
            f"  {key}: checkpoint {have} vs model {want}" for key, have, want in fatal
        )
        raise RuntimeError(
            "Checkpoint is incompatible with the current model on the inference "
            f"forward path:\n{details}\n"
            "These weights are used to compute embeddings, so they cannot be skipped. "
            "Run scripts/inspect_checkpoint.py on this checkpoint to see the full "
            "hyperparameter and shape comparison."
        )
    for key, have, want in mismatched:
        log.info("  skipping training-only weight %s (checkpoint %s vs model %s)", key, have, want)

    incompatible = model.load_state_dict(filtered, strict=False)

    unexpected = [key for key in incompatible.unexpected_keys if not _is_training_only(key)]
    dropped_dat = len(incompatible.unexpected_keys) - len(unexpected)
    if dropped_dat:
        log.info("  dropped %d training-only weight(s) (DAT discriminators)", dropped_dat)
    if incompatible.missing_keys:
        log.info(
            "  %d key(s) missing from the checkpoint, left at their initial values: %s",
            len(incompatible.missing_keys), sorted(incompatible.missing_keys)[:10],
        )
    if unexpected:
        log.info(
            "  %d unexpected key(s) in the checkpoint, ignored: %s",
            len(unexpected), sorted(unexpected)[:10],
        )


def load_for_inference(
    path: Union[str, Path],
    map_location: str = "cpu",
    **overrides: Any,
):
    """Load a ``CancerFoundation`` checkpoint for evaluation or embedding.

    Reconstructs the model from the checkpoint's hyperparameters, repairing the
    drift described in the module docstring, then loads the weights. DAT
    discriminators are dropped and ``torch.compile`` is disabled — neither
    affects embeddings or the expression decoder — so the result must not be used
    to resume training.

    Args:
        path: Path to a Lightning ``.ckpt``.
        map_location: Passed through to ``torch.load``.
        **overrides: Constructor arguments that win over the stored
            hyperparameters, e.g. ``vocab=``.

    Returns:
        The reconstructed ``CancerFoundation``, in eval mode.

    Raises:
        TypeError: A required constructor argument cannot be recovered.
        RuntimeError: A weight on the inference forward path has the wrong shape.
    """
    from cancerfoundation.model.model import CancerFoundation

    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state_dict = _strip_compile_prefix(dict(ckpt.get("state_dict", {})))

    # Inference never needs the compiled graph, and skipping it keeps the module
    # tree free of the `_orig_mod` wrapper the keys were just stripped of.
    hparams["compile_model"] = False
    # DAT is training-only; not building the discriminators makes their weights
    # merely unexpected rather than a shape conflict.
    hparams["do_dat"] = False

    encoded = _encoded_conditions_from_state_dict(state_dict, hparams.get("conditions_nums"))
    if encoded is not None and encoded != hparams.get("encoded_conditions"):
        log.info("Recovered encoded_conditions from the checkpoint: %s", encoded)
        hparams["encoded_conditions"] = encoded

    hparams = _reconcile_hparams(hparams)
    hparams.update(overrides)

    model = CancerFoundation(**hparams)
    _load_weights(model, state_dict)
    model.eval()
    return model
