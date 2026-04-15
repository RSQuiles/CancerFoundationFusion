from __future__ import annotations

import json
from dataclasses import dataclass
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
class AblationExperimentConfig:
    """Top-level config used by the ablation runner."""

    experiment_name: str
    model_config_path: Path
    train_entrypoint: str
    base_overrides: dict[str, Any]
    ablations: list[AblationSpec]
    downstream_tasks: list[DownstreamTaskSpec]
    output_dir: Path
    dry_run: bool = False


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

    return AblationExperimentConfig(
        experiment_name=raw.get("experiment_name", "ablation_experiment"),
        model_config_path=Path(raw["model_config_path"]),
        train_entrypoint=raw["train_entrypoint"],
        base_overrides=raw.get("base_overrides", {}),
        ablations=ablations,
        downstream_tasks=tasks,
        output_dir=Path(raw.get("output_dir", "ablation_runs")),
        dry_run=bool(raw.get("dry_run", False)),
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
