"""Train named models from one launch config, locally or one SLURM job per model.

Each model's files go to ``<experiments_root>/checkpoints/<name>/``; there is no
per-launch or per-ablation directory. A launch config only groups models that are
started together: the names are what identify them afterwards.

    python -m experiments.launch --config experiments/configs/example_unifiedfm.json --dry-run
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
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from typing import Any

from experiments.config import LaunchConfig, load_launch_config
from experiments.runtime import (
	describe_resume,
	MODEL_CONFIG_FILE,
	check_checkpoint_dir_free,
	deep_merge_dict,
	resolve_model_config,
	run_downstream_tasks,
	run_training_from_config,
	write_json,
	write_run_card,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train named models from a launch config")
	parser.add_argument(
		"--config",
		type=Path,
		required=True,
		help="Path to launch config (.json/.yaml/.yml)",
	)
	parser.add_argument(
		"--root",
		type=Path,
		default=None,
		help="Experiments root (overrides the config's experiments_root and $UNIFIEDFM_ROOT)",
	)
	parser.add_argument(
		"--only",
		nargs="+",
		default=None,
		help="Launch only these model names from the config",
	)
	parser.add_argument(
		"--max-runs",
		type=int,
		default=None,
		help="Optional limit for quick debugging of the launch setup",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not launch real training; create a placeholder checkpoint per model",
	)
	parser.add_argument(
		"--slurm",
		action="store_true",
		help="Submit each model as an independent SBATCH job",
	)
	return parser.parse_args()


def _resolve_model_runs(
	cfg: LaunchConfig,
	only: list[str] | None = None,
) -> list[tuple[str, dict[str, Any], dict[str, str] | None]]:
	"""Merge base config, base overrides and per-model overrides; fill in layout paths."""
	base_model_cfg = deep_merge_dict(deepcopy(cfg.base_config), cfg.base_overrides)
	if base_model_cfg.get("trainer", {}).get("save_dir"):
		raise ValueError(
			"base_config sets trainer.save_dir; the save dir is derived from each model's "
			"name now. Remove the key."
		)

	if only:
		unknown = sorted(set(only) - {m.name for m in cfg.models})
		if unknown:
			raise ValueError(f"--only names not in the config: {unknown}")

	runs: list[tuple[str, dict[str, Any], dict[str, str] | None]] = []
	for spec in cfg.models:
		if only and spec.name not in only:
			continue
		run_cfg = deep_merge_dict(base_model_cfg, spec.overrides)
		run_cfg, train_ref = resolve_model_config(cfg.layout, spec.name, run_cfg)
		check_checkpoint_dir_free(cfg.layout, spec.name, run_cfg)
		runs.append((spec.name, run_cfg, train_ref))
	return runs


def _prepare_model_dir(
	cfg: LaunchConfig,
	model_name: str,
	run_model_cfg: dict[str, Any],
	train_ref: dict[str, str] | None,
	dry_run: bool,
) -> Path:
	model_dir = cfg.layout.checkpoint_dir(model_name)
	model_dir.mkdir(parents=True, exist_ok=True)
	write_run_card(
		cfg.layout,
		model_name,
		run_model_cfg,
		launch_name=cfg.launch_name,
		launch_config_path=cfg.config_path,
		train_ref=train_ref,
		dry_run=dry_run,
	)
	return model_dir


def _write_launch_summary(cfg: LaunchConfig, summary: dict[str, Any]) -> Path:
	stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	path = cfg.layout.launches_dir / f"{cfg.launch_name}_{stamp}.json"
	suffix = 1
	while path.exists():  # two launches within the same second
		path = cfg.layout.launches_dir / f"{cfg.launch_name}_{stamp}_{suffix}.json"
		suffix += 1
	write_json(summary, path)
	print(f"Launch summary: {path}")
	return path


def _submit_models_to_slurm(
	cfg: LaunchConfig,
	runs: list[tuple[str, dict[str, Any], dict[str, str] | None]],
	task_specs: list[dict[str, Any]],
	dry_run: bool,
) -> dict[str, Any]:
	run_outputs: list[dict[str, Any]] = []

	for model_name, run_model_cfg, train_ref in runs:
		model_dir = _prepare_model_dir(cfg, model_name, run_model_cfg, train_ref, dry_run)

		model_cfg_path = model_dir / MODEL_CONFIG_FILE
		run_result_path = model_dir / "run_result.json"
		write_json(run_model_cfg, model_cfg_path)

		payload = {
			"launch_name": cfg.launch_name,
			"model_name": model_name,
			"model_config": run_model_cfg,
			"model_config_path": str(model_cfg_path),
			"train_entrypoint": cfg.train_entrypoint,
			"downstream_tasks": task_specs,
			"dry_run": dry_run,
			"result_path": str(run_result_path),
		}
		payload_path = model_dir / "job_payload.json"
		write_json(payload, payload_path)

		python_executable = cfg.slurm.python_executable or "python"

		# Generate command to run inside SLURM job
		project_root = Path(cfg.slurm.project_root)
		train_dir = Path(run_model_cfg["data"]["train_path"])
		binds = list(dict.fromkeys([*cfg.slurm.binds, str(cfg.experiments_root), str(train_dir)]))

		worker_inner_cmd = shlex.join(
			[
				"CUDA_LAUNCH_BLOCKING=1",
				python_executable,
				"-u",
				str(project_root / "experiments" / "slurm_worker.py"),
				"--payload",
				str(payload_path),
			]
		)

		bind_args: list[str] = []
		for bind in binds:
			bind_args.extend(["--bind", f"{bind}:{bind}"])

		wrapped_cmd = shlex.join(
			[
				"srun",
				"singularity",
				"run",
				"--pwd",
				str(project_root),
				*bind_args,
				"--nv",
				cfg.slurm.singularity_image,
				"bash",
				"-c",
				# expandable_segments cuts fragmentation from the large, short-lived
				# attention allocations, which is what turns a "there is free VRAM but
				# no contiguous block" failure into a successful step.
				f"export PATH=/usr/bin:/bin && export MASTER_ADDR=localhost && export MASTER_PORT=$(( SLURM_JOB_ID % 16384 + 20000 )) && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && {worker_inner_cmd}",
			]
		)

		job_name = cfg.job_name(model_name)

		sbatch_cmd = ["sbatch", "--parsable"] # makes sbatch output only the job ID
		sbatch_cmd.extend(cfg.slurm.sbatch_args)
		sbatch_cmd.extend(
			[
				"--job-name",
				job_name,
				"--output",
				str(model_dir / "slurm-%j.out"),
				"--wrap", # directly pass a command string
				wrapped_cmd,
			]
		)

		if dry_run:
			# Show what would be submitted; a dry run never queues a job.
			print(f"[dry-run] {shlex.join(sbatch_cmd)}")
			run_outputs.append(
				{
					"model_name": model_name,
					"job_id": None,
					"job_name": job_name,
					"checkpoint_dir": str(model_dir),
					"payload_path": str(payload_path),
					"status": "dry-run",
				}
			)
			continue

		env = os.environ.copy()
		env.update(cfg.slurm.environment)
		try:
			# Submit SBATCH job using the interface in slurm_worker.py
			completed = subprocess.run(
				sbatch_cmd,
				capture_output=True,
				text=True,
				check=True,
				env=env,
			)
		except subprocess.CalledProcessError as exc:
			raise RuntimeError(
				f"Failed to submit model '{model_name}' with sbatch. "
				f"stderr: {exc.stderr.strip() or '<empty>'}"
			) from exc

		raw_sbatch_output = completed.stdout.strip()
		job_id = raw_sbatch_output.split(";", maxsplit=1)[0].strip()
		run_outputs.append(
			{
				"model_name": model_name,
				"job_id": job_id,
				"job_name": job_name,
				"checkpoint_dir": str(model_dir),
				"payload_path": str(payload_path),
				"result_path": str(run_result_path),
				"model_config_path": str(model_cfg_path),
				"status": "submitted",
			}
		)

	summary = {
		"launch_name": cfg.launch_name,
		"launch_config": str(cfg.config_path),
		"experiments_root": str(cfg.experiments_root),
		"execution_mode": "slurm",
		"models": [run["model_name"] for run in run_outputs],
		"runs": run_outputs,
	}
	_write_launch_summary(cfg, summary)
	return summary


def launch_models(
	cfg: LaunchConfig,
	only: list[str] | None = None,
	max_runs: int | None = None,
	cli_dry_run: bool = False,
	force_slurm: bool = False,
) -> dict[str, Any]:
	task_specs = [
		{"name": task.name, "entrypoint": task.entrypoint, "config": task.config}
		for task in cfg.downstream_tasks
	]

	runs = _resolve_model_runs(cfg, only)
	if max_runs is not None:
		runs = runs[: max(0, max_runs)]

	dry_run = cfg.dry_run or cli_dry_run
	print(f"Experiments root: {cfg.experiments_root}")
	for model_name, run_model_cfg, _ in runs:
		print(f"  {model_name}: save_dir={run_model_cfg['trainer']['save_dir']}")
		print(f"  {' ' * len(model_name)}  train_path={run_model_cfg.get('data', {}).get('train_path')}")
		resume_line = describe_resume(cfg.layout, model_name, run_model_cfg)
		if resume_line:
			print(f"  {' ' * len(model_name)}  {resume_line}")

	execution_mode = "slurm" if (force_slurm or cfg.slurm.enabled) else "local"
	if execution_mode == "slurm":
		return _submit_models_to_slurm(
			cfg=cfg,
			runs=runs,
			task_specs=task_specs,
			dry_run=dry_run,
		)

	run_outputs: list[dict[str, Any]] = []

	for model_name, run_model_cfg, train_ref in runs:
		print(f"Running {model_name}!")
		model_dir = _prepare_model_dir(cfg, model_name, run_model_cfg, train_ref, dry_run)
		model_cfg_path = model_dir / MODEL_CONFIG_FILE

		print("Training...")
		train_result = run_training_from_config(
			model_name=model_name,
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
				"model_name": model_name,
				"checkpoint_dir": str(model_dir),
				"checkpoint_path": str(train_result.checkpoint_path),
				"metrics": metrics,
			}
		)

	summary = {
		"launch_name": cfg.launch_name,
		"launch_config": str(cfg.config_path),
		"experiments_root": str(cfg.experiments_root),
		"execution_mode": "local",
		"models": [run["model_name"] for run in run_outputs],
		"runs": run_outputs,
	}
	_write_launch_summary(cfg, summary)
	return summary


def main() -> None:
	args = parse_args()
	cfg = load_launch_config(args.config, root_override=args.root)
	summary = launch_models(
		cfg=cfg,
		only=args.only,
		max_runs=args.max_runs,
		cli_dry_run=args.dry_run,
		force_slurm=args.slurm,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
