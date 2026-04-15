from __future__ import annotations

import json
from pathlib import Path


def train_from_config(model_config_path: str | Path) -> str:
    """Dummy config-based trainer stub.

    Replace this with your future training entrypoint that reads a config file
    and launches the model. The required interface is:
    - input: path to model config file
    - output: path to produced checkpoint
    """
    model_config_path = Path(model_config_path)
    cfg = json.loads(model_config_path.read_text(encoding="utf-8"))

    output_root = model_config_path.parent
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"

    ckpt_path.write_text(
        "stub checkpoint\n"
        f"run_name={cfg.get('run_name', 'unknown')}\n"
        f"ablation={cfg.get('ablation_name', 'unknown')}\n",
        encoding="utf-8",
    )
    return str(ckpt_path)
