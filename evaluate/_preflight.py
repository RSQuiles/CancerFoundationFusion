"""Check a step's arguments without running it (used by ``evaluate.run_evaluation``).

    python -m evaluate._preflight <module> <step.yaml> [--pass-as config|cli] [-- EXTRA ARGV...]

Imports ``<module>``, builds its parser with ``build_parser()`` and parses the step's
arguments the way the script itself would: ``--config <step.yaml>`` through
``evaluate._cli.parse_with_config`` (``config``), or the YAML turned into flags
(``cli``). Unknown keys, wrong types and bad choices exit non-zero here, in the step's
own environment, before any step of the run has started. Checks a script makes after
parsing (e.g. "--model or --embedder") are not repeated; they still fail at run time.

A module without ``build_parser`` cannot be checked and is reported as skipped (exit 0).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SKIPPED_EXIT = 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("module")
    parser.add_argument("config", type=Path)
    parser.add_argument("--pass-as", choices=("config", "cli"), default="config")
    args = parser.parse_args(argv)

    module = importlib.import_module(args.module)
    build = getattr(module, "build_parser", None)
    if build is None:
        print(f"[preflight] {args.module} has no build_parser(); arguments not checked")
        return SKIPPED_EXIT

    script_parser = build()
    script_parser.prog = args.module
    if args.pass_as == "config":
        from evaluate._cli import parse_with_config

        # The step file is flat (no section blocks), so the section name is irrelevant.
        parse_with_config(script_parser, "_preflight", ["--config", str(args.config), *extra])
    else:
        import yaml

        from evaluate.run_evaluation import to_cli_args

        values = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        script_parser.parse_args([*to_cli_args(values), *extra])
    print(f"[preflight] {args.module}: arguments OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
