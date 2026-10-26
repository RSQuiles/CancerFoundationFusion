from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.layout import ExperimentLayout, resolve_root, validate_model_name


@dataclass
class ModelRunSpec:
    """One named model: key-value overrides on top of the base model config.

    ``name`` is the model's identity everywhere under the experiments root
    (``checkpoints/<name>``, ``embeddings/<name>``, ``results/.../<name>``).
    """

    name: str
    overrides: dict[str, Any]


@dataclass
class DownstreamTaskSpec:
    """One downstream task callable and optional task-specific configuration."""

    name: str
    entrypoint: str
    config: dict[str, Any]


# Today's cluster setup; override any of these in the config's "slurm" block.
DEFAULT_PROJECT_ROOT = "/cluster/work/boeva/rquiles/CancerFoundationFusion"
DEFAULT_SINGULARITY_IMAGE = "/cluster/customapps/biomed/boeva/rquiles/bionemo-framework_v1.sif"
DEFAULT_BINDS = [
    "/cluster/work/boeva/rquiles",
    "/cluster/work/boeva/eheiss",
    "/cluster/work/boeva/bulkFM",
]


@dataclass
class SlurmExecutionConfig:
    """Optional SLURM fan-out settings for one-job-per-model execution."""

    enabled: bool = False
    sbatch_args: list[str] = field(default_factory=list)
    python_executable: str = "python"
    environment: dict[str, str] = field(default_factory=dict)
    job_name_prefix: str | None = None
    project_root: str = DEFAULT_PROJECT_ROOT
    singularity_image: str = DEFAULT_SINGULARITY_IMAGE
    binds: list[str] = field(default_factory=lambda: list(DEFAULT_BINDS))


@dataclass
class LaunchConfig:
    """Top-level config used by the model launcher."""

    launch_name: str
    train_entrypoint: str
    base_config: dict[str, Any]
    base_overrides: dict[str, Any]
    models: list[ModelRunSpec]
    downstream_tasks: list[DownstreamTaskSpec]
    experiments_root: Path
    dry_run: bool = False
    slurm: SlurmExecutionConfig = field(default_factory=SlurmExecutionConfig)
    model_config_path: Path | None = None
    config_path: Path | None = None

    @property
    def layout(self) -> ExperimentLayout:
        return ExperimentLayout(self.experiments_root)

    def job_name(self, model_name: str) -> str:
        """SLURM job name; ``rename_model.py`` uses it to detect a live job."""
        return f"{self.slurm.job_name_prefix or self.launch_name}_{model_name}"[:128]


def load_launch_config(
    config_path: Path | str,
    root_override: Path | str | None = None,
) -> LaunchConfig:
    """Load a launch configuration from JSON (or YAML if PyYAML is installed)."""
    config_path = Path(config_path)
    raw = _read_config_file(config_path)

    if "output_dir" in raw:
        raise ValueError(
            f"{config_path}: 'output_dir' is no longer supported; every model's files go to "
            "<experiments_root>/checkpoints/<name>/. Remove the key (and any 'save_dir')."
        )
    if "models" in raw:
        model_items = raw["models"]
    elif "ablations" in raw:
        warnings.warn(
            f"{config_path}: the 'ablations' key is deprecated, rename it to 'models'",
            stacklevel=2,
        )
        model_items = raw["ablations"]
    else:
        model_items = []

    models = [
        ModelRunSpec(name=validate_model_name(item["name"]), overrides=item.get("overrides", {}))
        for item in model_items
    ]
    names = [m.name for m in models]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{config_path}: duplicate model name(s) {duplicates}")

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
        project_root=str(slurm_raw.get("project_root", DEFAULT_PROJECT_ROOT)),
        singularity_image=str(slurm_raw.get("singularity_image", DEFAULT_SINGULARITY_IMAGE)),
        binds=[str(b) for b in slurm_raw.get("binds", DEFAULT_BINDS)],
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
                "Launch config must define either 'base_config' or 'model_config_path'."
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

    root = root_override if root_override is not None else raw.get("experiments_root")
    return LaunchConfig(
        launch_name=raw.get("launch_name", raw.get("experiment_name", config_path.stem)),
        train_entrypoint=raw["train_entrypoint"],
        base_config=base_config,
        base_overrides=base_overrides,
        models=models,
        downstream_tasks=tasks,
        experiments_root=resolve_root(root),
        dry_run=bool(raw.get("dry_run", False)),
        slurm=slurm_cfg,
        model_config_path=model_config_path,
        config_path=config_path.resolve(),
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
