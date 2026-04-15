from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def run_dummy_downstream(
    checkpoint_path: str | Path,
    task_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Dummy downstream evaluator for ablation framework integration tests.

    This function mimics a downstream task API and returns deterministic,
    pseudo-random metrics derived from checkpoint path and task config.
    Replace this with real downstream tasks later.
    """
    checkpoint_path = Path(checkpoint_path)
    cfg = task_config or {}
    seed = int(cfg.get("seed", 0))
    weight = float(cfg.get("task_weight", 1.0))

    payload = f"{checkpoint_path.as_posix()}::{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    # Deterministic score in [0, 1)
    base = int(digest[:8], 16) / 0xFFFFFFFF
    auc = max(0.0, min(1.0, 0.55 + 0.4 * base * weight))
    acc = max(0.0, min(1.0, 0.50 + 0.45 * (1.0 - base) * weight))

    return {
        "auc": float(auc),
        "accuracy": float(acc),
    }
