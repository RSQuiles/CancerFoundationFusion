from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import AblationExperimentConfig, load_experiment_config
from .runtime import (
	deep_merge_dict,
	run_downstream_tasks,
	run_training_from_config,
	stable_run_id,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run config-driven ablation experiments")
	parser.add_argument(
		"--config",
		type=Path,
		required=True,
		help="Path to ablation experiment config (.json/.yaml/.yml)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Override output_dir from config",
	)
	parser.add_argument(
		"--max-runs",
		type=int,
		default=None,
		help="Optional limit for quick debugging of ablation setup",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not launch real training; create placeholder checkpoint per run",
	)
	return parser.parse_args()


def _load_model_base_config(model_config_path: Path) -> dict[str, Any]:
	"""Read model config file to serve as baseline for all ablations.

	The training code is expected to consume this config file in a future
	config-driven pipeline.
	"""
	if not model_config_path.exists():
		raise FileNotFoundError(
			f"Base model config not found at {model_config_path}. "
			"Create it first or point model_config_path to a valid file."
		)

	if model_config_path.suffix.lower() != ".json":
		raise ValueError(
			"Only JSON model config is supported in this scaffold. "
			"Use a .json model config file."
		)
	return json.loads(model_config_path.read_text(encoding="utf-8"))


def _prepare_runs(cfg: AblationExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
	base_model_cfg = _load_model_base_config(cfg.model_config_path)
	base_model_cfg = deep_merge_dict(base_model_cfg, cfg.base_overrides)

	runs: list[tuple[str, dict[str, Any]]] = []
	for ablation in cfg.ablations:
		run_cfg = deep_merge_dict(base_model_cfg, ablation.overrides)
		run_cfg["ablation_name"] = ablation.name
		runs.append((ablation.name, run_cfg))
	return runs


def run_ablation_experiment(
	cfg: AblationExperimentConfig,
	output_dir_override: Path | None = None,
	max_runs: int | None = None,
	cli_dry_run: bool = False,
) -> dict[str, Any]:
	output_dir = output_dir_override if output_dir_override is not None else cfg.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	task_specs = [
		{"name": task.name, "entrypoint": task.entrypoint, "config": task.config}
		for task in cfg.downstream_tasks
	]

	runs = _prepare_runs(cfg)
	if max_runs is not None:
		runs = runs[: max(0, max_runs)]

	run_outputs: list[dict[str, Any]] = []
	dry_run = cfg.dry_run or cli_dry_run

	for run_name, run_model_cfg in runs:
		run_id = stable_run_id(cfg.experiment_name, run_name)
		run_dir = output_dir / f"{run_name}_{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)

		model_cfg_path = run_dir / "model_config.json"
		run_model_cfg = deepcopy(run_model_cfg)
		run_model_cfg["run_name"] = run_name

		train_result = run_training_from_config(
			run_name=run_name,
			model_config=run_model_cfg,
			model_config_path=model_cfg_path,
			train_entrypoint=cfg.train_entrypoint,
			dry_run=dry_run,
		)

		metrics = run_downstream_tasks(
			checkpoint_path=train_result.checkpoint_path,
			task_specs=task_specs,
		)

		run_outputs.append(
			{
				"run_name": run_name,
				"run_id": run_id,
				"checkpoint_path": str(train_result.checkpoint_path),
				"metrics": metrics,
			}
		)

	summary = {
		"experiment_name": cfg.experiment_name,
		"output_dir": str(output_dir),
		"num_runs": len(run_outputs),
		"runs": run_outputs,
	}
	(output_dir / "summary.json").write_text(
		json.dumps(summary, indent=2),
		encoding="utf-8",
	)
	return summary


def main() -> None:
	args = parse_args()
	cfg = load_experiment_config(args.config)
	summary = run_ablation_experiment(
		cfg=cfg,
		output_dir_override=args.output_dir,
		max_runs=args.max_runs,
		cli_dry_run=args.dry_run,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
