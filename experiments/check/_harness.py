"""Shared PASS/FAIL bookkeeping for the self-checks in this directory."""

from __future__ import annotations

import sys

FAILED: list[str] = []


def check(name: str, condition: bool) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        FAILED.append(name)


def raises(fn, exc: type[BaseException] = Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def finish() -> None:
    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")
