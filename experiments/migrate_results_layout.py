"""Move results from the old layout (``results/tasks/``, ``results/assessment/``) to the new one.

    python -m experiments.migrate_results_layout              # dry run: print every move
    python -m experiments.migrate_results_layout --apply      # move, rewrite paths in JSON

Old -> new (see ``experiments/layout.py``)::

    results/tasks/<task>/...                                -> results/<task>/...
    results/assessment/reconstruction_<plan>[_<name>]/      -> results/reconstruction/_compare/<plan>/<name|default>/
    results/assessment/geneicl_<suite>[_<name>]/            -> results/geneicl_<suite>/_compare/<name|default>/
    results/assessment/scib_<name>/                         -> results/scib_benchmark/<name>/
    results/assessment/store_check_<modality>_<split>/     -> results/_checks/store_<modality>_<split>/

The plan and suite of a comparison come from its ``manifest.json`` (``"plan"`` /
``"suite"``), since ``<plan>_<name>`` cannot be split reliably by name. A directory that
matches no rule, or a comparison without a usable manifest, is reported and left where it
is. Every move is planned and checked first: an existing destination stops the run before
anything moves. After the moves, path strings inside the JSON files under ``results/``
are rewritten to the new locations (key by key, as ``rename_model`` does); model names
are not touched. Any failure moves everything back. Empty ``tasks/`` and ``assessment/``
directories are removed at the end; a non-empty one is kept and its contents listed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.layout import DEFAULT_COMPARISON, ExperimentLayout  # noqa: E402
from experiments.rename_model import _path_rewriter, rewrite_json_value  # noqa: E402

MAX_JSON_BYTES = 50 * 1024 * 1024
_NO_NAME = "\0"  # rewrite_json_value only rewrites paths when no name equals this


def plan_slug(plan_id: str) -> str:
    """As ``evaluate.reconstruction.run_reconstruction.plan_slug``."""
    return plan_id.replace(":", "_").replace("/", "_")


def _manifest(path: Path) -> dict:
    try:
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def assessment_target(layout: ExperimentLayout, src: Path) -> tuple[Path | None, str]:
    """``(new location, reason)`` of one ``results/assessment/<name>/`` directory."""
    name = src.name
    if name.startswith("scib_"):
        return layout.scib_dir(name[len("scib_"):]), "scIB analysis"
    if name.startswith("store_check_"):
        return layout.check_dir("store_" + name[len("store_check_"):]), "store check"
    if name.startswith("reconstruction_"):
        plan = _manifest(src).get("plan")
        if not plan:
            return None, "reconstruction comparison without a manifest 'plan'"
        prefix = f"reconstruction_{plan_slug(plan)}"
        if not name.startswith(prefix):
            return None, f"name does not start with its manifest plan ({prefix})"
        label = name[len(prefix):].lstrip("_") or DEFAULT_COMPARISON
        return layout.task_compare_dir("reconstruction", plan_slug(plan), label), "reconstruction comparison"
    if name.startswith("geneicl_"):
        suite = _manifest(src).get("suite")
        if not suite:
            return None, "GeneICL comparison without a manifest 'suite'"
        prefix = f"geneicl_{suite}"
        if not name.startswith(prefix):
            return None, f"name does not start with its manifest suite ({prefix})"
        label = name[len(prefix):].lstrip("_") or DEFAULT_COMPARISON
        return layout.task_compare_dir(prefix, label), "GeneICL comparison"
    return None, "no rule for this name"


def plan_moves(layout: ExperimentLayout) -> tuple[list[tuple[Path, Path]], list[str]]:
    """``(moves, notes)``. Raises when a destination exists or two sources share one."""
    results = layout.results_dir
    moves: list[tuple[Path, Path]] = []
    notes: list[str] = []

    tasks = results / "tasks"
    if tasks.is_dir():
        for task in sorted(p for p in tasks.iterdir() if p.is_dir()):
            dest_task = layout.task_root(task.name)
            if not dest_task.exists():
                moves.append((task, dest_task))
                continue
            # The task directory exists already (new runs since the code change): merge.
            for child in sorted(task.iterdir()):
                moves.append((child, dest_task / child.name))
        stray = [p.name for p in tasks.iterdir() if not p.is_dir()]
        if stray:
            notes.append(f"files directly in results/tasks/ are left there: {stray}")

    assessment = results / "assessment"
    if assessment.is_dir():
        for src in sorted(p for p in assessment.iterdir() if p.is_dir()):
            dest, reason = assessment_target(layout, src)
            if dest is None:
                notes.append(f"left in place: assessment/{src.name} ({reason})")
            else:
                moves.append((src, dest))

    problems = [f"{dst} exists (for {src})" for src, dst in moves if dst.exists()]
    seen: dict[Path, Path] = {}
    for src, dst in moves:
        if dst in seen:
            problems.append(f"{seen[dst]} and {src} would both move to {dst}")
        seen[dst] = src
    if problems:
        raise RuntimeError("Nothing was moved; resolve these first:\n  " + "\n  ".join(problems))
    return moves, notes


def json_files(results: Path) -> list[Path]:
    return [p for p in sorted(results.rglob("*.json"))
            if p.is_file() and p.stat().st_size <= MAX_JSON_BYTES]


def apply(layout: ExperimentLayout, moves: list[tuple[Path, Path]]) -> int:
    """Move, then rewrite moved paths in the JSON files under results/. Returns files rewritten."""
    done: list[tuple[Path, Path]] = []
    backups: dict[Path, str] = {}
    try:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            done.append((src, dst))
        rewrite = _path_rewriter(moves)
        changed = 0
        for path in json_files(layout.results_dir):
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
            except ValueError:
                continue
            new = rewrite_json_value(data, _NO_NAME, _NO_NAME, rewrite)
            if new != data:
                backups[path] = text
                path.write_text(json.dumps(new, indent=2), encoding="utf-8")
                changed += 1
        return changed
    except Exception:
        for path, text in backups.items():
            path.write_text(text, encoding="utf-8")
        for src, dst in reversed(done):
            shutil.move(str(dst), str(src))
        raise


def remove_if_empty(path: Path) -> str:
    if not path.is_dir():
        return f"{path.name}/: not present"
    leftovers = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    if leftovers:
        return f"{path.name}/ kept, still holds {len(leftovers)} file(s), e.g. {leftovers[:5]}"
    shutil.rmtree(path)
    return f"{path.name}/ removed (empty)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move results from results/tasks/ and results/assessment/ "
                                                 "to one directory per task.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--apply", action="store_true", help="Move (default: print the plan only)")
    args = parser.parse_args(argv)

    layout = ExperimentLayout.from_root(args.root)
    results = layout.results_dir
    try:
        moves, notes = plan_moves(layout)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"{len(moves)} move(s) under {results}:")
    for src, dst in moves:
        print(f"  {src.relative_to(results).as_posix()}  ->  {dst.relative_to(results).as_posix()}")
    for note in notes:
        print(f"  [note] {note}")
    if not args.apply:
        print("dry run: nothing moved; rerun with --apply")
        return 0
    changed = apply(layout, moves)
    print(f"moved {len(moves)}; rewrote paths in {changed} JSON file(s)")
    for old in ("tasks", "assessment"):
        print("  " + remove_if_empty(results / old))
    return 0


if __name__ == "__main__":
    sys.exit(main())
