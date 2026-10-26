from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Union, Dict, Any

from experiments.layout import ExperimentLayout

TrainFn = Callable[[Path], Union[str, Path]]
EvalFn = Callable[[Union[str, Path], Dict[str, Any]], Dict[str, float]]

MODEL_CONFIG_FILE = "model_config.json"
DRY_RUN_CHECKPOINT = "dry_run.ckpt"


@dataclass
class TrainResult:
    model_name: str
    checkpoint_path: Path


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries recursively without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """Resolve a dotted entrypoint string like 'pkg.module:function'."""
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint '{entrypoint}'. Expected 'module.submodule:function'."
        )
    module_name, fn_name = entrypoint.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"Entrypoint function '{fn_name}' not found in {module_name}")
    if not callable(fn):
        raise TypeError(f"Entrypoint '{entrypoint}' is not callable")
    return fn


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_train_path(layout: ExperimentLayout, value: Any) -> tuple[str, dict[str, str] | None]:
    """Turn a ``data.train_path`` value into a path plus the layout reference it came from.

    Accepts a literal path, ``{"pipeline_ready": "<modality>:<split>"}`` or
    ``{"mix": "<name>"}``.
    """
    if isinstance(value, dict):
        if "pipeline_ready" in value:
            modality, sep, split = str(value["pipeline_ready"]).partition(":")
            if not sep or not split:
                raise ValueError(
                    f"train_path {value}: expected {{'pipeline_ready': '<modality>:<split>'}}"
                )
            return str(layout.pipeline_ready(modality, split)), {
                "pipeline_ready": f"{modality}:{split}"
            }
        if "mix" in value:
            return str(layout.mix_dir(str(value["mix"]))), {"mix": str(value["mix"])}
        raise ValueError(f"train_path {value}: expected a 'pipeline_ready' or 'mix' key")
    return str(value), None


def resolve_model_config(
    layout: ExperimentLayout,
    model_name: str,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Fill in everything the layout decides: save_dir, wandb_name and train_path.

    A hand-written ``trainer.save_dir`` is refused: it is what used to let a model
    land in an ablation directory, or two models share one.
    """
    cfg = deepcopy(model_config)
    trainer = cfg.setdefault("trainer", {})
    if trainer.get("save_dir"):
        raise ValueError(
            f"Model '{model_name}' sets trainer.save_dir={trainer['save_dir']!r}. "
            f"The save dir is derived from the name ({layout.checkpoint_dir(model_name)}); "
            "remove the key."
        )
    trainer["save_dir"] = str(layout.checkpoint_dir(model_name))
    trainer.setdefault("wandb_name", model_name)

    data = cfg.setdefault("data", {})
    train_ref = None
    if "train_path" in data:
        data["train_path"], train_ref = resolve_train_path(layout, data["train_path"])

    cfg["model_name"] = model_name
    return cfg, train_ref


def check_checkpoint_dir_free(layout: ExperimentLayout, model_name: str, model_config: dict[str, Any]) -> None:
    """Refuse to reuse a model's directory unless the run resumes it."""
    ckpt_dir = layout.checkpoint_dir(model_name)
    occupied = False
    if ckpt_dir.is_dir():
        real_ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != DRY_RUN_CHECKPOINT]
        card = ckpt_dir / "run.json"
        card_is_real = False
        if card.exists():
            try:
                card_is_real = not json.loads(card.read_text(encoding="utf-8")).get("dry_run", False)
            except json.JSONDecodeError:
                card_is_real = True
        occupied = bool(real_ckpts) or card_is_real
    resume = (model_config.get("trainer") or {}).get("resume_from_checkpoint")
    if occupied and not resume:
        raise FileExistsError(
            f"Model '{model_name}' already exists at {ckpt_dir}. Pick another name, set "
            "trainer.resume_from_checkpoint to continue it, or rename the old one with "
            "experiments/rename_model.py."
        )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def write_run_card(
    layout: ExperimentLayout,
    model_name: str,
    model_config: dict[str, Any],
    launch_name: str,
    launch_config_path: Path | None,
    train_ref: dict[str, str] | None,
    dry_run: bool = False,
) -> Path:
    """Write ``checkpoints/<model>/run.json``, the model card.

    An existing card keeps its ``created`` time and ``previous_names`` history, so
    resuming a renamed model does not lose where it came from. A dry-run card is
    flagged, so it does not count as an existing model on the next real launch.
    """
    path = layout.run_card(model_name)
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    card = {
        "name": model_name,
        "created": previous.get("created", now),
        "updated": now,
        "launch_name": launch_name,
        "launch_config": str(launch_config_path) if launch_config_path else None,
        "git_commit": _git_commit(),
        "train_path": (model_config.get("data") or {}).get("train_path"),
        "train_data": train_ref,
        "previous_names": previous.get("previous_names", []),
        "dry_run": dry_run,
        "model_config": model_config,
    }
    write_json(card, path)
    return path


def run_training_from_config(
    model_name: str,
    model_config: dict[str, Any],
    model_config_path: Path,
    train_entrypoint: str,
    dry_run: bool,
) -> TrainResult:
    """Launch training by passing a model config file path to a configurable entrypoint."""
    write_json(model_config, model_config_path)

    if dry_run:
        fake_ckpt = model_config_path.parent / DRY_RUN_CHECKPOINT
        fake_ckpt.write_text("dry-run checkpoint placeholder\n", encoding="utf-8")
        return TrainResult(model_name=model_name, checkpoint_path=fake_ckpt)

    # Imported here so the launcher (and its dry runs) stay torch-free.
    from utils import save_resolved_config
    from utils_config import build_parser, _load_json_config, _flatten_sectioned_config, _filter_known_config_keys, expand_env_vars, pretty_print_args

    # Generate args Namespace compatible with model input
    parser = build_parser()

    nested_config = _load_json_config(model_config_path)
    flat_config = _flatten_sectioned_config(nested_config, ignore_unexpected=True)
    filtered_config = _filter_known_config_keys(parser, flat_config)

    parser.set_defaults(**filtered_config)
    model_args = parser.parse_args([])
    model_args = expand_env_vars(model_args)

    pretty_print_args(model_args)
    save_resolved_config(model_args, Path(model_args.save_dir) / "config.resolved.json")

    train_fn = resolve_entrypoint(train_entrypoint)
    ckpt_path = train_fn(model_args)
    return TrainResult(model_name=model_name, checkpoint_path=Path(ckpt_path))


def run_downstream_tasks(
    checkpoint_path: Path,
    task_specs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Evaluate one checkpoint on all configured downstream tasks."""

    print(f"Running downstream tasks:\n {task_specs}")

    results: dict[str, dict[str, float]] = {}
    for task in task_specs:
        eval_fn: EvalFn = resolve_entrypoint(task["entrypoint"])
        metrics = eval_fn(task.get("config", ""), checkpoint_path=checkpoint_path)
        results[task["name"]] = {k: float(v) for k, v in metrics.items()}
    return results
