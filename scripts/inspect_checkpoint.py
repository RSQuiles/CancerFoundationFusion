"""Report why a CancerFoundation checkpoint does or does not load.

Prints the stored hyperparameters, how they differ from the current
``CancerFoundation.__init__`` signature, and every weight whose shape disagrees
with a freshly constructed model — without needing the checkpoint to be loadable
in the first place.

Usage:
    python scripts/inspect_checkpoint.py path/to/model.ckpt [--hparams]
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cancerfoundation.checkpoint import (  # noqa: E402
    _COMPILE_PREFIX,
    _encoded_conditions_from_state_dict,
    _reconcile_hparams,
    _strip_compile_prefix,
)
from cancerfoundation.model.model import CancerFoundation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--hparams", action="store_true", help="Print the full hyperparameter dict."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    raw_state = dict(ckpt.get("state_dict", {}))

    print(f"\n=== {args.checkpoint} ===")
    print(f"hyperparameters : {len(hparams)}")
    print(f"state_dict keys : {len(raw_state)}")
    compiled = any(_COMPILE_PREFIX in key for key in raw_state)
    print(f"torch.compile   : {'yes (keys carry ' + _COMPILE_PREFIX + ')' if compiled else 'no'}")

    if args.hparams:
        print("\n--- hyperparameters ---")
        for name in sorted(hparams):
            print(f"  {name} = {hparams[name]!r}")

    # --- signature drift -----------------------------------------------------
    signature = inspect.signature(CancerFoundation.__init__)
    parameters = {n: p for n, p in signature.parameters.items() if n != "self"}

    removed = sorted(set(hparams) - set(parameters))
    added = sorted(set(parameters) - set(hparams))
    required_added = [
        name for name in added if parameters[name].default is inspect.Parameter.empty
    ]

    print("\n--- signature drift ---")
    print(f"  in checkpoint but no longer accepted : {removed or 'none'}")
    print(f"  added since, with a default          : {sorted(set(added) - set(required_added)) or 'none'}")
    print(f"  added since, REQUIRED (breaks load)  : {required_added or 'none'}")

    # --- shape comparison ----------------------------------------------------
    # Needs a constructed model, which is exactly what a missing required
    # argument prevents — so reconcile the hyperparameters first.
    state_dict = _strip_compile_prefix(raw_state)
    probe = dict(hparams)
    probe["compile_model"] = False
    encoded = _encoded_conditions_from_state_dict(state_dict, probe.get("conditions_nums"))
    if encoded is not None:
        probe["encoded_conditions"] = encoded
        print(f"\n  encoded_conditions recovered from weights: {encoded}")

    print("\n--- weight comparison (do_dat kept as saved) ---")
    try:
        model = CancerFoundation(**_reconcile_hparams(probe))
    except Exception as exc:  # noqa: BLE001 - diagnostic tool, report and stop
        print(f"  could not construct a model to compare against: {type(exc).__name__}: {exc}")
        return 1

    reference = model.state_dict()
    mismatched = [
        (key, tuple(value.shape), tuple(reference[key].shape))
        for key, value in state_dict.items()
        if key in reference and hasattr(value, "shape")
        and reference[key].shape != value.shape
    ]
    missing = sorted(set(reference) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(reference))

    if mismatched:
        print(f"  {len(mismatched)} shape mismatch(es):")
        for key, have, want in mismatched:
            print(f"    {key}: checkpoint {have} vs model {want}")
    else:
        print("  no shape mismatches")
    print(f"  missing from checkpoint : {len(missing)}{' -> ' + str(missing[:10]) if missing else ''}")
    print(f"  unexpected in checkpoint: {len(unexpected)}{' -> ' + str(unexpected[:10]) if unexpected else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
