"""Give a model a better name: move every directory and rewrite every file that names it.

    python -m experiments.rename_model OLD NEW                 # dry run: print the plan
    python -m experiments.rename_model OLD NEW --apply         # do it
    python -m experiments.rename_model OLD NEW --apply --config experiments/configs/x.json

What changes:
  1. every directory in ``ExperimentLayout.model_owned_dirs`` (checkpoints, embeddings,
     representation, alignment and per-task results) is moved to the new name;
  2. JSON files inside those directories, plus ``ExperimentLayout.model_reference_files``
     (launch summaries, assessment manifests), are rewritten key by key: name fields
     equal to OLD, ``models`` lists and paths under a moved directory;
  3. launch configs passed with ``--config`` get the model's entry renamed, so a
     relaunch or resume targets the new directory;
  4. one line is appended to ``checkpoints/_renames.jsonl``.

``run.json`` keeps the history in ``previous_names``. Any failure rolls back what was
already done. Binary ``.ckpt`` files and the W&B run are left alone; see the notes
printed at the end.
"""

from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from experiments.layout import RUN_CARD, ExperimentLayout, validate_model_name

# String values under these keys are model names when they equal OLD exactly.
NAME_KEYS = {"name", "model", "model_name", "wandb_name"}
# Under these keys a list element (or dict key) equal to OLD is a model name.
NAME_COLLECTION_KEYS = {"models", "previous_names"}
MAX_JSON_BYTES = 50 * 1024 * 1024


class RenameError(RuntimeError):
    pass


@dataclass
class RenamePlan:
    old: str
    new: str
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    json_files: list[Path] = field(default_factory=list)  # paths *after* the moves
    launch_configs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# JSON rewriting
# --------------------------------------------------------------------------- #


def _path_rewriter(moves: list[tuple[Path, Path]]) -> Callable[[str], str]:
    """Rewrite a string that is (or starts with) a moved directory's path."""
    pairs = []
    for src, dst in moves:
        for s, d in {(str(src), str(dst)), (src.as_posix(), dst.as_posix())}:
            pairs.append((s, d))
    pairs.sort(key=lambda p: -len(p[0]))  # longest prefix first

    def rewrite(value: str) -> str:
        for s, d in pairs:
            if value == s or value.startswith(s + "/") or value.startswith(s + "\\"):
                return d + value[len(s):]
        return value

    return rewrite


def rewrite_json_value(obj: Any, old: str, new: str, rewrite_path: Callable[[str], str], key: str | None = None) -> Any:
    """Return ``obj`` with model-name fields and moved paths updated. Pure."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if key in NAME_COLLECTION_KEYS and k == old:
                k = new
            out[k] = rewrite_json_value(v, old, new, rewrite_path, key=k)
        return out
    if isinstance(obj, list):
        if key == "previous_names":
            return obj  # history: never rewrite the old names
        return [
            new if (key in NAME_COLLECTION_KEYS and item == old)
            else rewrite_json_value(item, old, new, rewrite_path, key=key)
            for item in obj
        ]
    if isinstance(obj, str):
        if key in NAME_KEYS and obj == old:
            return new
        return rewrite_path(obj)
    return obj


def rename_in_launch_config(raw: dict[str, Any], old: str, new: str) -> bool:
    """Rename OLD's entry in a launch config dict in place; True if anything changed."""
    changed = False
    for list_key in ("models", "ablations"):
        for item in raw.get(list_key, []) or []:
            if item.get("name") != old:
                continue
            item["name"] = new
            trainer = (item.get("overrides") or {}).get("trainer") or {}
            if trainer.get("wandb_name") == old:
                trainer["wandb_name"] = new
            changed = True
    return changed


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def live_jobs_for(model: str) -> list[str] | None:
    """Names of the user's queued/running SLURM jobs for ``model``; None if squeue is missing."""
    if shutil.which("squeue") is None:
        return None
    out = subprocess.run(
        ["squeue", "--me", "--noheader", "--format=%j"],
        capture_output=True, text=True, check=False,
    )
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return [n for n in names if n == model or n.endswith(f"_{model}")]


def build_plan(
    layout: ExperimentLayout,
    old: str,
    new: str,
    launch_configs: list[Path] | None = None,
    job_lookup: Callable[[str], list[str] | None] = live_jobs_for,
) -> RenamePlan:
    validate_model_name(old)
    validate_model_name(new)
    if old == new:
        raise RenameError("OLD and NEW are the same name")

    plan = RenamePlan(old=old, new=new)
    old_dirs = layout.model_owned_dirs(old)
    new_dirs = layout.model_owned_dirs(new)  # same order: task dirs come from the same listing

    clashes = [d for d in new_dirs if d.exists()]
    if clashes:
        raise RenameError(
            f"'{new}' already owns: " + ", ".join(str(d) for d in clashes)
        )
    plan.moves = [(src, dst) for src, dst in zip(old_dirs, new_dirs) if src.exists()]
    if not plan.moves:
        raise RenameError(f"'{old}' owns nothing under {layout.root}")

    jobs = job_lookup(old)
    if jobs is None:
        plan.warnings.append("squeue not found: could not check for a running job of this model")
    elif jobs:
        raise RenameError(
            f"'{old}' has SLURM jobs queued or running ({', '.join(jobs)}); "
            "wait for them or cancel them first, their outputs would land in the old directory"
        )

    for src, dst in plan.moves:
        for path in sorted(src.rglob("*.json")):
            if path.stat().st_size <= MAX_JSON_BYTES:
                plan.json_files.append(dst / path.relative_to(src))
    plan.json_files.extend(layout.model_reference_files())

    for cfg_path in launch_configs or []:
        if cfg_path.suffix.lower() != ".json":
            raise RenameError(f"--config {cfg_path}: only JSON launch configs can be rewritten")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not rename_in_launch_config(raw, old, new):
            plan.warnings.append(f"{cfg_path} has no model named '{old}'; left unchanged")
        else:
            plan.launch_configs.append(cfg_path)
    return plan


# --------------------------------------------------------------------------- #
# Applying (with rollback)
# --------------------------------------------------------------------------- #


def apply_plan(layout: ExperimentLayout, plan: RenamePlan, _fail_after_moves: bool = False) -> list[str]:
    """Execute ``plan``; on any failure undo the moves and restore rewritten files."""
    moved: list[tuple[Path, Path]] = []
    originals: dict[Path, str] = {}
    touched: list[str] = []
    rewrite_path = _path_rewriter(plan.moves)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        for src, dst in plan.moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            moved.append((src, dst))
            touched.append(f"moved {src} -> {dst}")
        if _fail_after_moves:
            raise RuntimeError("injected failure (self-check)")

        for path in plan.json_files:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                plan.warnings.append(f"{path} is not valid JSON; left unchanged")
                continue
            updated = rewrite_json_value(data, plan.old, plan.new, rewrite_path)
            if path.name == RUN_CARD and path.parent == layout.checkpoint_dir(plan.new):
                history = list(updated.get("previous_names", []))
                history.append({"name": plan.old, "renamed": now})
                updated["previous_names"] = history
                updated["name"] = plan.new
            if updated != data:
                originals[path] = text
                path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
                touched.append(f"rewrote {path}")

        for cfg_path in plan.launch_configs:
            text = cfg_path.read_text(encoding="utf-8")
            raw = json.loads(text)
            rename_in_launch_config(raw, plan.old, plan.new)
            originals[cfg_path] = text
            cfg_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            touched.append(f"rewrote {cfg_path}")
    except BaseException:
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")
        for src, dst in reversed(moved):
            os.rename(dst, src)
        print("Rename failed; every change was rolled back.", file=sys.stderr)
        raise

    layout.rename_log.parent.mkdir(parents=True, exist_ok=True)
    with layout.rename_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"old": plan.old, "new": plan.new, "time": now, "changes": touched}) + "\n")
    return touched


def _wandb_hint(layout: ExperimentLayout, model: str) -> str | None:
    """A ready-to-run command that renames the W&B run, if the run id can be found."""
    wandb_dir = layout.checkpoint_dir(model) / "wandb"
    runs = sorted(wandb_dir.glob("run-*")) if wandb_dir.is_dir() else []
    if not runs:
        return None
    run_id = runs[-1].name.rsplit("-", 1)[-1]
    card = layout.run_card(model)
    trainer = {}
    if card.exists():
        trainer = (json.loads(card.read_text(encoding="utf-8")).get("model_config") or {}).get("trainer") or {}
    entity = trainer.get("wandb_entity", "<entity>")
    project = trainer.get("wandb", "<project>")
    return (
        "python -c \"import wandb; r = wandb.Api().run('"
        f"{entity}/{project}/{run_id}'); r.name = '{model}'; r.update()\""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename a model: move its directories and rewrite the files that name it."
    )
    parser.add_argument("old", help="Current model name")
    parser.add_argument("new", help="New model name")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument(
        "--config", type=Path, action="append", default=[],
        help="Launch config(s) whose entry for OLD should be renamed too (repeatable)",
    )
    parser.add_argument("--apply", action="store_true", help="Perform the rename (default: dry run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ExperimentLayout.from_root(args.root)
    try:
        plan = build_plan(layout, args.old, args.new, args.config)
    except RenameError as exc:
        sys.exit(f"Refusing to rename: {exc}")

    print(f"Rename '{plan.old}' -> '{plan.new}' under {layout.root}")
    for src, dst in plan.moves:
        print(f"  move    {src}\n       -> {dst}")
    print(f"  rewrite up to {len(plan.json_files)} JSON file(s) (only those that mention the model change)")
    for cfg_path in plan.launch_configs:
        print(f"  rewrite launch config {cfg_path}")
    for warning in plan.warnings:
        print(f"  [warn] {warning}")

    if not args.apply:
        print("\nDry run: nothing changed. Re-run with --apply.")
        return

    touched = apply_plan(layout, plan)
    print(f"\nDone: {len(touched)} change(s), logged to {layout.rename_log}")
    print(
        "  note: .ckpt files were not modified. They hold no model name; Lightning's stored\n"
        "        checkpoint dirpath points at the old directory and is replaced on resume."
    )
    hint = _wandb_hint(layout, plan.new)
    print(
        "  note: the W&B run keeps its old display name"
        + (f"; to rename it:\n        {hint}" if hint else ".")
    )


if __name__ == "__main__":
    main()
