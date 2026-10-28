"""Continuing a pretraining run in its own checkpoint directory.

A model directory (``checkpoints/<name>/``) can be trained in several *segments*: one
process start each. Restarting from a checkpoint continues the epoch and step
counters, adds new checkpoints next to the old ones and reuses the W&B run, so the
segments read as one run. Two modes decide what happens to the LR schedule:

- ``continue``: a fresh warmup + cosine starts at the restored step and spans the
  remaining epochs (``--epochs`` is always the total).
- ``exact``: the checkpoint's own schedule is restored, for recovering a run that was
  interrupted part-way through a segment.

Files kept beside the checkpoints:

- ``wandb_run.json``: the W&B run id, written by the first segment and reused after.
- ``segments.jsonl``: one line per process start (mode, checkpoint file name, start
  epoch and step, schedule settings, seed, commit, SLURM job, W&B id). File names
  only, never absolute paths, so ``experiments/rename_model.py`` stays exact.

Torch-free at import time: torch is imported only inside ``read_checkpoint_meta``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRY_RUN_CHECKPOINT = "dry_run.ckpt"
WANDB_RUN_FILE = "wandb_run.json"
SEGMENTS_FILE = "segments.jsonl"
RESUME_MODES = ("continue", "exact")

# A mismatch here means the checkpoint's weights cannot be loaded into the new model.
ARCHITECTURE_HPARAMS = (
    "nlayers", "embsize", "nheads", "d_hid", "max_seq_len", "n_bins", "input_style",
    "input_emb_style", "training_tasks", "do_mvc", "mvc_decoder_style", "cell_emb_style",
    "explicit_zero_prob", "esm_emb", "do_dat", "compile_model",
)
# A change here is allowed, but it is a different training setup, so it is reported.
WATCHED_HPARAMS = ("data_path", "lr", "warmup_ratio_or_step", "loss_type", "zero_percentages")
WATCHED_DATAMODULE_HPARAMS = ("batch_size",)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def find_last_checkpoint(save_dir: str | Path) -> Path | None:
    """Newest ``*.ckpt`` in *save_dir* by mtime, ignoring the dry-run placeholder."""
    ckpt_dir = Path(save_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates = [p for p in ckpt_dir.glob("*.ckpt") if p.name != DRY_RUN_CHECKPOINT]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(spec: str | None, save_dir: str | Path | None) -> Path | None:
    """``"last"`` -> the newest checkpoint in *save_dir* (None if there is none); else the path."""
    if spec is None:
        return None
    if spec == "last":
        return find_last_checkpoint(save_dir) if save_dir is not None else None
    path = Path(spec)
    if path.name == DRY_RUN_CHECKPOINT:
        raise ValueError(f"{path} is the dry-run placeholder, not a checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


@dataclass
class ResumeInfo:
    """What a checkpoint says about where training stopped."""

    checkpoint: Path
    mode: str
    epochs_completed: int
    global_step: int
    mid_epoch: bool | None
    lr_segment: dict = field(default_factory=dict)
    hparams: dict = field(default_factory=dict)
    datamodule_hparams: dict = field(default_factory=dict)

    @property
    def segment_start(self) -> int:
        """First step of the LR schedule the resumed process builds."""
        if self.mode == "continue":
            return self.global_step
        # Checkpoints written before LR segments existed belong to a schedule from 0.
        return int(self.lr_segment.get("start_step", 0))


def _is_last_batch(checkpoint: Mapping[str, Any]) -> bool | None:
    try:
        progress = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]
        return bool(progress["is_last_batch"])
    except (KeyError, TypeError):
        return None


def summarize_checkpoint(checkpoint: Mapping[str, Any], path: Path, mode: str) -> ResumeInfo:
    """``ResumeInfo`` from a loaded checkpoint dict (split out so it is testable without torch)."""
    if mode not in RESUME_MODES:
        raise ValueError(f"resume mode must be one of {RESUME_MODES}, got {mode!r}")
    last_batch = _is_last_batch(checkpoint)
    epoch = int(checkpoint.get("epoch", 0))
    # Lightning stores the 0-based epoch the checkpoint was written in; an end-of-epoch
    # checkpoint therefore has epoch + 1 epochs behind it.
    completed = epoch + 1 if last_batch else epoch
    return ResumeInfo(
        checkpoint=Path(path),
        mode=mode,
        epochs_completed=completed,
        global_step=int(checkpoint.get("global_step", 0)),
        mid_epoch=None if last_batch is None else not last_batch,
        lr_segment=dict(checkpoint.get("lr_segment") or {}),
        hparams=dict(checkpoint.get("hyper_parameters") or {}),
        datamodule_hparams=dict(checkpoint.get("datamodule_hyper_parameters") or {}),
    )


def read_checkpoint_meta(path: str | Path, mode: str) -> ResumeInfo:
    """Load *path* lazily (memory-mapped where possible) and summarize it."""
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        # Legacy (non-zip) checkpoints cannot be memory-mapped.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info = summarize_checkpoint(checkpoint, Path(path), mode)
    del checkpoint
    return info


def check_resume_epochs(info: ResumeInfo, epochs: int) -> None:
    if epochs <= info.epochs_completed:
        raise ValueError(
            f"{info.checkpoint.name} has {info.epochs_completed} completed epochs and "
            f"--epochs is {epochs}, so there is nothing left to train. --epochs is the "
            f"total, not the number of additional epochs: set it above "
            f"{info.epochs_completed}."
        )


def _vocab_size(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def compare_hparams(info: ResumeInfo, model_hparams: Mapping[str, Any],
                    datamodule_hparams: Mapping[str, Any] | None = None) -> list[str]:
    """Raise on an architecture mismatch; return messages for watched changes."""
    mismatched = [
        f"{key}: checkpoint {info.hparams[key]!r}, config {model_hparams[key]!r}"
        for key in ARCHITECTURE_HPARAMS
        if key in info.hparams and key in model_hparams and info.hparams[key] != model_hparams[key]
    ]
    old_vocab, new_vocab = _vocab_size(info.hparams.get("vocab")), _vocab_size(model_hparams.get("vocab"))
    if old_vocab is not None and new_vocab is not None and old_vocab != new_vocab:
        mismatched.append(f"vocab size: checkpoint {old_vocab}, config {new_vocab}")
    if mismatched:
        raise ValueError(
            f"Cannot resume from {info.checkpoint.name}: the config builds a different "
            "model.\n  " + "\n  ".join(mismatched)
        )
    changes = [
        f"{key}: {info.hparams[key]!r} -> {model_hparams[key]!r}"
        for key in WATCHED_HPARAMS
        if key in info.hparams and key in model_hparams and info.hparams[key] != model_hparams[key]
    ]
    for key in WATCHED_DATAMODULE_HPARAMS:
        old, new = info.datamodule_hparams.get(key), (datamodule_hparams or {}).get(key)
        if key in info.datamodule_hparams and new is not None and old != new:
            changes.append(f"datamodule {key}: {old!r} -> {new!r}")
    return changes


# --------------------------------------------------------------------------- #
# Side files
# --------------------------------------------------------------------------- #


def load_wandb_run(save_dir: str | Path) -> dict | None:
    path = Path(save_dir) / WANDB_RUN_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def find_wandb_run_id(save_dir: str | Path) -> str | None:
    """The W&B run to resume: ``wandb_run.json``, else the newest local ``wandb/run-*`` dir.

    The fallback covers runs trained before ``wandb_run.json`` was written.
    """
    recorded = (load_wandb_run(save_dir) or {}).get("id")
    if recorded:
        return recorded
    wandb_dir = Path(save_dir) / "wandb"
    runs = sorted(wandb_dir.glob("run-*")) if wandb_dir.is_dir() else []
    return runs[-1].name.rsplit("-", 1)[-1] if runs else None


def save_wandb_run(save_dir: str | Path, run_id: str, entity: str | None, project: str | None,
                   url: str | None) -> Path:
    path = Path(save_dir) / WANDB_RUN_FILE
    previous = load_wandb_run(save_dir) or {}
    record = {
        "id": run_id,
        "entity": entity,
        "project": project,
        "url": url,
        "created": previous.get("created", now_utc()) if previous.get("id") == run_id else now_utc(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_segments(save_dir: str | Path) -> list[dict]:
    path = Path(save_dir) / SEGMENTS_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_segment(save_dir: str | Path, record: dict) -> dict:
    """Append one segment line; ``index`` is its position in the file."""
    path = Path(save_dir) / SEGMENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"index": len(read_segments(save_dir)), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=False) + "\n")
    return record


def segment_record(info: ResumeInfo | None, *, epochs: int, lr: float, warmup: float,
                   seed: int | None, wandb_id: str | None = None) -> dict:
    return {
        "started": now_utc(),
        "mode": info.mode if info else "fresh",
        "from_checkpoint": info.checkpoint.name if info else None,
        "start_epoch": info.epochs_completed if info else 0,
        "start_step": info.global_step if info else 0,
        "lr_segment_start": info.segment_start if info else 0,
        "epochs": int(epochs),
        "lr": float(lr),
        "warmup_ratio_or_step": warmup,
        "seed": seed,
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wandb_id": wandb_id,
    }


def trim_metrics_csv(path: str | Path, max_step: int) -> int:
    """Move rows with ``step > max_step`` to ``metrics.discarded_<time>.csv``.

    Those rows were logged after the checkpoint being resumed and are about to be
    trained again, so keeping them would give two values for one step. Returns the
    number of rows moved (0 leaves the file untouched).
    """
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    try:
        step_col = header.index("step")
    except ValueError:
        return 0
    keep, drop = [], []
    for row in body:
        try:
            (drop if int(float(row[step_col])) > max_step else keep).append(row)
        except (ValueError, IndexError):
            keep.append(row)
    if not drop:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    discarded = path.with_name(f"{path.stem}.discarded_{stamp}{path.suffix}")
    for target, content in ((discarded, drop), (path, keep)):
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(content)
    return len(drop)


def archive_resolved_config(path: str | Path) -> Path | None:
    """Keep the previous segment's ``config.resolved.json`` as ``config.resolved.seg<k>.json``."""
    path = Path(path)
    if not path.exists():
        return None
    k = len(list(path.parent.glob(f"{path.stem}.seg*{path.suffix}")))
    target = path.with_name(f"{path.stem}.seg{k}{path.suffix}")
    path.replace(target)
    return target