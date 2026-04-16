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
from typing import Any

from config import AblationExperimentConfig, load_experiment_config
from runtime import (
	deep_merge_dict,
	run_downstream_tasks,
	run_training_from_config,
	stable_run_id,
	write_model_config,
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
	parser.add_argument(
		"--slurm",
		action="store_true",
		help="Submit each ablation run as an independent SBATCH job",
	)
	return parser.parse_args()


def _load_base_config(cfg: AblationExperimentConfig) -> dict[str, Any]:
	"""Return a deep copy of the baseline config used for every ablation."""
	return deepcopy(cfg.base_config)


def _prepare_runs(cfg: AblationExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
	base_model_cfg = _load_base_config(cfg)
	base_model_cfg = deep_merge_dict(base_model_cfg, cfg.base_overrides)

	runs: list[tuple[str, dict[str, Any]]] = []
	for ablation in cfg.ablations:
		run_cfg = deep_merge_dict(base_model_cfg, ablation.overrides)
		run_cfg["ablation_name"] = ablation.name
		runs.append((ablation.name, run_cfg))
	return runs


def _submit_runs_to_slurm(
	cfg: AblationExperimentConfig,
	runs: list[tuple[str, dict[str, Any]]],
	output_dir: Path,
	task_specs: list[dict[str, Any]],
	dry_run: bool,
) -> dict[str, Any]:
	run_outputs: list[dict[str, Any]] = []

	for run_name, run_model_cfg in runs:
		run_id = stable_run_id(cfg.experiment_name, run_name)
		run_dir = output_dir / f"{run_name}_{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)

		model_cfg_path = run_dir / "model_config.json"
		run_result_path = run_dir / "run_result.json"
		run_model_cfg = deepcopy(run_model_cfg)
		run_model_cfg["run_name"] = run_name
		write_model_config(run_model_cfg, model_cfg_path)

		payload = {
			"experiment_name": cfg.experiment_name,
			"run_name": run_name,
			"run_id": run_id,
			"model_config": run_model_cfg,
			"model_config_path": str(model_cfg_path),
			"train_entrypoint": cfg.train_entrypoint,
			"downstream_tasks": task_specs,
			"dry_run": dry_run,
			"result_path": str(run_result_path),
		}
		payload_path = run_dir / "job_payload.json"
		payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

		python_executable = cfg.slurm.python_executable or "python"

		# Generate command to run inside SLURM job
		project_root = Path("/cluster/work/boeva/rquiles/CancerFoundation")
		current_dir = project_root / "ablate"
		train_dir = Path(run_model_cfg["data"]["train_path"])
		singularity_image = "/cluster/customapps/biomed/boeva/fbarkmann/bionemo-framework_nightly.sif"

		worker_inner_cmd = shlex.join(
			[
				"CUDA_LAUNCH_BLOCKING=1",
				python_executable,
				"-u",
				str(project_root / "ablate" / "slurm_worker.py"),
				"--payload",
				str(payload_path),
			]
		)

		wrapped_cmd = shlex.join(
			[
				"srun",
				"singularity",
				"run",
				"--pwd",
				str(current_dir),
				"--bind",
				f"{project_root}:{project_root}",
				"--bind",
				f"{train_dir}:{train_dir}",
				"--nv",
				singularity_image,
				"bash",
				"-c",
				f"export PATH=/usr/bin:/bin && {worker_inner_cmd}",
			]
		)

		job_name_prefix = cfg.slurm.job_name_prefix or cfg.experiment_name
		job_name = f"{job_name_prefix}_{run_name}"[:128]

		sbatch_cmd = ["sbatch", "--parsable"] # makes sbatch output only the job ID
		sbatch_cmd.extend(cfg.slurm.sbatch_args)
		sbatch_cmd.extend(
			[
				"--job-name",
				job_name,
				"--output",
				str(run_dir / "slurm-%j.out"),
				# "--error",
				# str(run_dir / "slurm-%j.err"),
				"--wrap", # directly pass a command string
				wrapped_cmd,
			]
		)

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
				f"Failed to submit run '{run_name}' with sbatch. "
				f"stderr: {exc.stderr.strip() or '<empty>'}"
			) from exc

		raw_sbatch_output = completed.stdout.strip()
		job_id = raw_sbatch_output.split(";", maxsplit=1)[0].strip()
		run_outputs.append(
			{
				"run_name": run_name,
				"run_id": run_id,
				"job_id": job_id,
				"payload_path": str(payload_path),
				"result_path": str(run_result_path),
				"model_config_path": str(model_cfg_path),
				"status": "submitted",
			}
		)

	summary = {
		"experiment_name": cfg.experiment_name,
		"execution_mode": "slurm",
		"output_dir": str(output_dir),
		"num_runs": len(run_outputs),
		"runs": run_outputs,
	}
	(output_dir / "summary.json").write_text(
		json.dumps(summary, indent=2),
		encoding="utf-8",
	)
	return summary


def run_ablation_experiment(
	cfg: AblationExperimentConfig,
	output_dir_override: Path | None = None,
	max_runs: int | None = None,
	cli_dry_run: bool = False,
	force_slurm: bool = False,
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

	dry_run = cfg.dry_run or cli_dry_run
	execution_mode = "slurm" if (force_slurm or cfg.slurm.enabled) else "local"
	if execution_mode == "slurm":
		return _submit_runs_to_slurm(
			cfg=cfg,
			runs=runs,
			output_dir=output_dir,
			task_specs=task_specs,
			dry_run=dry_run,
		)

	run_outputs: list[dict[str, Any]] = []

	for run_name, run_model_cfg in runs:
		print(f"Running {run_name}!")
		run_id = stable_run_id(cfg.experiment_name, run_name)
		run_dir = output_dir / f"{run_name}_{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)

		model_cfg_path = run_dir / "model_config.json"
		run_model_cfg = deepcopy(run_model_cfg)
		run_model_cfg["run_name"] = run_name

		print("Training...")
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
		"execution_mode": "local",
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
		force_slurm=args.slurm,
	)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
