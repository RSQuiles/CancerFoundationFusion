from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Callable, Union, Dict, Any

from utils_config import build_parser, _load_json_config, _flatten_sectioned_config, _filter_known_config_keys, expand_env_vars, pretty_print_args
from argparse import Namespace
import torch.multiprocessing as mp

TrainFn = Callable[[Path], Union[str, Path]]
EvalFn = Callable[[Union[str, Path], Dict[str, Any]], Dict[str, float]]

@dataclass
class TrainResult:
    run_name: str
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


def write_model_config(config: dict[str, Any], output_path: Path) -> None:
    """Persist per-run model config. This is the handoff point to config-driven training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_training_from_config(
    run_name: str,
    model_config: dict[str, Any],
    model_config_path: Path,
    train_entrypoint: str,
    dry_run: bool,
) -> TrainResult:
    """Launch training by passing a model config file path to a configurable entrypoint."""
    write_model_config(model_config, model_config_path)

    if dry_run:
        fake_ckpt = model_config_path.parent / "checkpoints" / "model.ckpt"
        fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
        fake_ckpt.write_text("dry-run checkpoint placeholder\n", encoding="utf-8")
        return TrainResult(run_name=run_name, checkpoint_path=fake_ckpt)

    # Generate args Namespace compatible with model input
    parser = build_parser()

    nested_config = _load_json_config(model_config_path)
    flat_config = _flatten_sectioned_config(nested_config, ignore_unexpected=True)
    filtered_config = _filter_known_config_keys(parser, flat_config)

    parser.set_defaults(**filtered_config)
    model_args = parser.parse_args([])
    model_args = expand_env_vars(model_args)

    pretty_print_args(model_args)

    train_fn = resolve_entrypoint(train_entrypoint)
    ckpt_path = train_fn(model_args)
    return TrainResult(run_name=run_name, checkpoint_path=Path(ckpt_path))


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


def stable_run_id(experiment_name: str, run_name: str) -> str:
    payload = f"{experiment_name}:{run_name}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
