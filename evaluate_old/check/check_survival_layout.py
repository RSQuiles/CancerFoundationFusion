"""Self-check for ``evaluate/finetune/survival_layout.py``.

Runs anywhere — no cluster, GPU, checkpoint, torch or pycox. That is the point of
keeping the naming in a dependency-free module: the rule that decides where survival
CSVs go is the one thing the writer and both readers must agree on, so it has to be
testable without the SurvBoard environment.

    python evaluate/check/check_survival_layout.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.finetune.survival_layout import (
    ablation_name,
    model_name_from_checkpoint,
    storage_name,
    survival_csv_dir,
)

FAILED: list[str] = []


def check(name: str, condition: bool) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        FAILED.append(name)


class _CaptureWarnings(logging.Handler):
    """Collect WARNING records emitted by the layout module."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


ABL = "/work/save_CFF/ablation_big_condition"
CKPT = f"{ABL}/unified/step_step=900000_epoch_epoch=01.ckpt"
CKPT_NESTED = f"{ABL}/unified/checkpoints/step_step=900000_epoch_epoch=01.ckpt"


def check_model_names() -> None:
    print("\n-- model name from checkpoint --")
    check("flat layout", model_name_from_checkpoint(CKPT) == "unified")
    check(
        "checkpoints/ subdir does not become the model name",
        model_name_from_checkpoint(CKPT_NESTED) == "unified",
    )
    check(
        "Path and str accepted alike",
        model_name_from_checkpoint(Path(CKPT)) == model_name_from_checkpoint(CKPT),
    )


def check_ablation_names() -> None:
    print("\n-- ablation name --")
    check("from checkpoint", ablation_name(checkpoint_path=CKPT) == "ablation_big_condition")
    check(
        "from nested checkpoint",
        ablation_name(checkpoint_path=CKPT_NESTED) == "ablation_big_condition",
    )
    check("from ablation_dir", ablation_name(ablation_dir=ABL) == "ablation_big_condition")
    check(
        "trailing slash on ablation_dir",
        ablation_name(ablation_dir=ABL + "/") == "ablation_big_condition",
    )
    check(
        "checkpoint beats a disagreeing ablation_dir",
        ablation_name(checkpoint_path=CKPT, ablation_dir="/work/save_CFF/ablation_paired_aggfix")
        == "ablation_big_condition",
    )
    check("neither -> None", ablation_name() is None)


def check_storage_names() -> None:
    print("\n-- storage name --")
    check(
        "checkpoint -> {ablation}_{model}",
        storage_name("unified", checkpoint_path=CKPT) == "ablation_big_condition_unified",
    )
    check(
        "nested checkpoint keeps the model name",
        storage_name("unified", checkpoint_path=CKPT_NESTED)
        == "ablation_big_condition_unified",
    )
    check(
        "PCA baseline: no checkpoint, ablation_dir only",
        storage_name("pca_baseline", ablation_dir=ABL)
        == "ablation_big_condition_pca_baseline",
    )
    check(
        "the stale-key case resolves to the checkpoint's ablation",
        storage_name(
            "unified",
            checkpoint_path=CKPT,
            ablation_dir="/work/save_CFF/ablation_paired_aggfix",
        ) == "ablation_big_condition_unified",
    )

    # The collision the prefix exists to prevent.
    a = storage_name("unified", checkpoint_path=f"{ABL}/unified/x.ckpt")
    b = storage_name("unified", checkpoint_path="/work/save_CFF/ablation_united_data/unified/x.ckpt")
    check("same model in two ablations -> different directories", a != b)


def check_fallback_warns() -> None:
    print("\n-- fallback when the ablation is unknown --")
    handler = _CaptureWarnings()
    log = logging.getLogger("evaluate.finetune.survival_layout")
    previous_level, previous_propagate = log.level, log.propagate
    log.addHandler(handler)
    log.setLevel(logging.WARNING)   # main() muted the root logger
    log.propagate = False           # keep the expected warning off the console
    try:
        name = storage_name("unified")
    finally:
        log.removeHandler(handler)
        log.setLevel(previous_level)
        log.propagate = previous_propagate

    check("falls back to the bare model name", name == "unified")
    check("and warns", any("Cannot determine the ablation" in m for m in handler.records))
    check("naming the model", any("unified" in m for m in handler.records))


def check_round_trip() -> None:
    """The invariant: the writer's directory is the reader's directory.

    Mirrors the two call sites without importing them — survboard_task needs torch
    and evaluate_survboard_metrics needs pycox. Both build their path from
    ``survival_csv_dir``, so composing it here the way each does is the check.
    """
    print("\n-- writer/reader round trip --")
    results_dir = Path("/work/outputs/survival/functions")

    # Writer: survboard_task.get_storage_name() -> _save_survival_csvs
    written = survival_csv_dir(
        results_dir, "TCGA", "BRCA",
        storage_name("unified", checkpoint_path=CKPT, ablation_dir=ABL),
    ) / "split_5.csv"

    # Reader: evaluate_all()'s default -> _load_survival_csv
    read = survival_csv_dir(
        results_dir, "TCGA", "BRCA", storage_name("unified", ablation_dir=ABL),
    ) / "split_5.csv"

    check("paths agree", written == read)
    check(
        "and are the documented layout",
        written == results_dir / "TCGA" / "BRCA" / "ablation_big_condition_unified" / "split_5.csv",
    )

    # Same for the PCA baseline, whose writer side has no checkpoint.
    written_pca = survival_csv_dir(
        results_dir, "TCGA", "BRCA", storage_name("pca_baseline", ablation_dir=ABL),
    )
    read_pca = survival_csv_dir(
        results_dir, "TCGA", "BRCA", storage_name("pca_baseline", ablation_dir=ABL),
    )
    check("pca_baseline paths agree", written_pca == read_pca)


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)   # keep the expected warning quiet
    check_model_names()
    check_ablation_names()
    check_storage_names()
    check_fallback_warns()
    check_round_trip()

    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
