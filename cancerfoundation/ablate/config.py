from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AblationSpec:
    """One named ablation with key-value overrides on top of base model config."""

    name: str
    overrides: dict[str, Any]


@dataclass
class DownstreamTaskSpec:
    """One downstream task callable and optional task-specific configuration."""

    name: str
    entrypoint: str
    config: dict[str, Any]


@dataclass
class SlurmExecutionConfig:
    """Optional SLURM fan-out settings for one-job-per-ablation execution."""

    enabled: bool = False
    sbatch_args: list[str] = field(default_factory=list)
    python_executable: str = "python"
    environment: dict[str, str] = field(default_factory=dict)
    job_name_prefix: str | None = None


@dataclass
class AblationExperimentConfig:
    """Top-level config used by the ablation runner."""

    experiment_name: str
    train_entrypoint: str
    base_config: dict[str, Any]
    base_overrides: dict[str, Any]
    ablations: list[AblationSpec]
    downstream_tasks: list[DownstreamTaskSpec]
    output_dir: Path
    dry_run: bool = False
    slurm: SlurmExecutionConfig = field(default_factory=SlurmExecutionConfig)
    model_config_path: Path | None = None


def load_experiment_config(config_path: Path | str) -> AblationExperimentConfig:
    """Load experiment configuration from JSON (or YAML if PyYAML is installed)."""
    config_path = Path(config_path)
    raw = _read_config_file(config_path)

    ablations = [
        AblationSpec(name=item["name"], overrides=item.get("overrides", {}))
        for item in raw.get("ablations", [])
    ]
    tasks = [
        DownstreamTaskSpec(
            name=item["name"],
            entrypoint=item["entrypoint"],
            config=item.get("config", {}),
        )
        for item in raw.get("downstream_tasks", [])
    ]
    slurm_raw = raw.get("slurm", {}) or {}
    env_raw = slurm_raw.get("environment", {}) or {}
    if not isinstance(env_raw, dict):
        raise TypeError("'slurm.environment' must be a JSON object of key/value pairs")

    slurm_cfg = SlurmExecutionConfig(
        enabled=bool(slurm_raw.get("enabled", False)),
        sbatch_args=[str(arg) for arg in slurm_raw.get("sbatch_args", [])],
        python_executable=str(slurm_raw.get("python_executable", "python")),
        environment={str(k): str(v) for k, v in env_raw.items()},
        job_name_prefix=(
            str(slurm_raw["job_name_prefix"])
            if slurm_raw.get("job_name_prefix") is not None
            else None
        ),
    )

    def _resolve_relative_path(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (config_path.parent / candidate).resolve()

    base_config = raw.get("base_config")
    model_config_path_value = raw.get("model_config_path")
    if base_config is None:
        if model_config_path_value is None:
            raise KeyError(
                "Experiment config must define either 'base_config' or 'model_config_path'."
            )
        model_config_path = _resolve_relative_path(model_config_path_value)
        if model_config_path is None:
            raise ValueError("'model_config_path' could not be resolved to a valid path")
        base_config = _read_config_file(model_config_path)
    else:
        if not isinstance(base_config, dict):
            raise TypeError("'base_config' must be a JSON object/dict")
        model_config_path = _resolve_relative_path(model_config_path_value)

    base_overrides = raw.get("base_overrides", {})
    if not isinstance(base_overrides, dict):
        raise TypeError("'base_overrides' must be a JSON object/dict")

    return AblationExperimentConfig(
        experiment_name=raw.get("experiment_name", "ablation_experiment"),
        train_entrypoint=raw["train_entrypoint"],
        base_config=base_config,
        base_overrides=base_overrides,
        ablations=ablations,
        downstream_tasks=tasks,
        output_dir=Path(raw.get("output_dir", "ablation_runs")),
        dry_run=bool(raw.get("dry_run", False)),
        slurm=slurm_cfg,
        model_config_path=model_config_path,
    )


def _read_config_file(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Use JSON or install pyyaml."
            ) from exc
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raise ValueError(
        f"Unsupported config format '{suffix}'. Use .json, .yaml, or .yml."
    )
