"""The ``recompute`` switch shared by every evaluation script (torch-free).

With ``recompute`` false a per-model result is skipped when its output is present at its
usual location; nothing else is compared, so results made before this switch existed count
as present. A stale result (retrained model, changed setting) is not detected: delete it or
run with ``--recompute``. Comparisons across models are always redone. The default is true,
so a script run on its own behaves as before.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def add_recompute_arg(parser: argparse.ArgumentParser, comparison: bool = False) -> None:
    help_text = ("Accepted for the global evaluation switch; a comparison is always redone."
                 if comparison else
                 "--no-recompute skips a model's result when its output is already present at "
                 "its usual location (no check that it is up to date); default: recompute.")
    parser.add_argument("--recompute", action=argparse.BooleanOptionalAction, default=True,
                        help=help_text)


def note_comparison(args) -> None:
    """For comparison scripts: say that ``recompute`` does not apply to them."""
    if getattr(args, "recompute", True) is False:
        print("[recompute] comparison: always redone")


def skip_existing(what: str, paths: Iterable[Path | str], recompute: bool) -> bool:
    """True (and a ``[skip]`` line) when ``recompute`` is false and every path exists."""
    if recompute:
        return False
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        print(f"[compute] {what}: missing {missing[0]}"
              + (f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""), flush=True)
        return False
    print(f"[skip] {what}: exists (recompute false)", flush=True)
    return True
