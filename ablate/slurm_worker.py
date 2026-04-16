from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from runtime import run_downstream_tasks, run_training_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one ablation run from a JSON payload (intended for SLURM jobs)."
    )
    parser.add_argument("--payload", type=Path, required=True, help="Path to job payload")
    return parser.parse_args()


def _load_payload(payload_path: Path) -> dict[str, Any]:
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _write_result(result_path: Path, payload: dict[str, Any], result: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    run_record = {
        "experiment_name": payload.get("experiment_name"),
        "run_name": payload.get("run_name"),
        "run_id": payload.get("run_id"),
        **result,
    }
    result_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    run_name = str(payload["run_name"])
    model_config = dict(payload["model_config"])
    model_config_path = Path(payload["model_config_path"])
    train_entrypoint = str(payload["train_entrypoint"])
    task_specs = list(payload.get("downstream_tasks", []))
    dry_run = bool(payload.get("dry_run", False))

    train_result = run_training_from_config(
        run_name=run_name,
        model_config=model_config,
        model_config_path=model_config_path,
        train_entrypoint=train_entrypoint,
        dry_run=dry_run,
    )
    metrics = run_downstream_tasks(
        checkpoint_path=train_result.checkpoint_path,
        task_specs=task_specs,
    )

    return {
        "status": "completed",
        "checkpoint_path": str(train_result.checkpoint_path),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.payload)
    result_path = Path(payload["result_path"])

    try:
        result = _execute(payload)
        _write_result(result_path, payload, result)
    except Exception as exc:  # pragma: no cover - failure path for batch jobs
        _write_result(
            result_path,
            payload,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
