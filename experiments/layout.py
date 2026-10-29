"""The ONE place directory names under the experiments root are composed.

Layout::

    <root>/
    ├── data/
    │   ├── {single_cell,bulk,pseudobulk}/
    │   │   ├── {modality_dir}.h5ad               raw counts in X, metadata in obs
    │   │   ├── splits/<split>/{train.txt,test.txt,split.json}[,train.h5ad,test.h5ad]
    │   │   └── pipeline_ready/<split>/           memmap store built from the train rows
    │   └── pipeline_ready/<mix>/                 multi-modality store (+ mix.json)
    ├── checkpoints/<model>/                      *.ckpt, run.json, model_config.json, ...
    │   ├── _launches/<launch>_<timestamp>.json   one summary per launcher invocation
    │   └── _renames.jsonl                        log written by rename_model.py
    ├── embeddings/<model>/{modality}_{part}.h5ad (+ embed.json)
    └── results/
        ├── representation/<model>/
        ├── representation/_alignment/<model>/
        ├── tasks/<task>/<model>/
        └── assessment/<analysis>/                (+ manifest.json listing its models)

Rules every writer follows, so ``rename_model.py`` never has to guess:
per-model output lives under a directory named after the model (never in a
filename), and an analysis spanning several models lists them in its
``manifest.json`` under ``"models"``.

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/cluster/work/boeva/rquiles/unifiedFM_experiments")
ROOT_ENV_VAR = "UNIFIEDFM_ROOT"

# Modality label (what obs["modality"] holds) -> directory name under data/.
MODALITY_DIRS: dict[str, str] = {
    "sc": "single_cell",
    "bulk": "bulk",
    "pseudobulk": "pseudobulk",
}
# Accepted spellings on the command line -> label.
_MODALITY_ALIASES: dict[str, str] = {
    **{label: label for label in MODALITY_DIRS},
    **{dirname: label for label, dirname in MODALITY_DIRS.items()},
    "single-cell": "sc",
    "pb": "pseudobulk",
}

SPLIT_PARTS = ("train", "test")
EMBEDDING_PARTS = ("test", "train_subset")

PCA_MODEL = "pca"
_RESERVED_NAMES = {"_launches", "_alignment", "_renames.jsonl"}
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

RUN_CARD = "run.json"
EMBED_CARD = "embed.json"
ASSESSMENT_MANIFEST = "manifest.json"
RENAME_LOG = "_renames.jsonl"


def resolve_root(root: str | Path | None = None) -> Path:
    """``root`` if given, else ``$UNIFIEDFM_ROOT``, else :data:`DEFAULT_ROOT`."""
    if root is not None:
        return Path(os.path.expandvars(str(root))).expanduser()
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_ROOT


def canonical_modality(modality: str) -> str:
    """Map any accepted spelling (``sc``, ``single_cell``, ``pb`` ...) to its label."""
    key = str(modality).strip().lower()
    if key not in _MODALITY_ALIASES:
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of {sorted(_MODALITY_ALIASES)}"
        )
    return _MODALITY_ALIASES[key]


def validate_model_name(name: str) -> str:
    """Return ``name`` unchanged, or raise if it cannot be used as a directory name."""
    if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid model name {name!r}: use letters, digits, '_', '.', '-' "
            "and start with a letter or digit."
        )
    if name in _RESERVED_NAMES:
        raise ValueError(f"Model name {name!r} is reserved by the layout.")
    return name


def _validate_part(part: str, allowed: tuple[str, ...]) -> str:
    if part not in allowed:
        raise ValueError(f"Unknown part {part!r}. Expected one of {allowed}")
    return part


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ExperimentLayout":
        return cls(resolve_root(root))

    # ---- data -------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    def modality_dir(self, modality: str) -> Path:
        return self.data_dir / MODALITY_DIRS[canonical_modality(modality)]

    def raw_h5ad(self, modality: str) -> Path:
        mod_dir = self.modality_dir(modality)
        return mod_dir / f"{mod_dir.name}.h5ad"

    def split_dir(self, modality: str, split: str) -> Path:
        return self.modality_dir(modality) / "splits" / split

    def split_index_file(self, modality: str, split: str, part: str) -> Path:
        return self.split_dir(modality, split) / f"{_validate_part(part, SPLIT_PARTS)}.txt"

    def split_h5ad(self, modality: str, split: str, part: str) -> Path:
        return self.split_dir(modality, split) / f"{_validate_part(part, SPLIT_PARTS)}.h5ad"

    def split_card(self, modality: str, split: str) -> Path:
        return self.split_dir(modality, split) / "split.json"

    def mask_plan(self, modality: str, split: str, part: str, name: str) -> Path:
        """Reconstruction mask plan for one split part (``evaluate/reconstruction/mask_plan.py``).

        Model-independent data, so it lives beside the split rather than under a model.
        Any part name is allowed (a split may define ``val`` and others).
        """
        if not _MODEL_NAME_RE.match(name) or not _MODEL_NAME_RE.match(part):
            raise ValueError(f"Invalid mask plan name {part!r}/{name!r}: use letters, digits, '_', '.', '-'")
        return self.split_dir(modality, split) / "masks" / f"{part}_{name}.npz"

    def pipeline_ready(self, modality: str, split: str) -> Path:
        return self.modality_dir(modality) / "pipeline_ready" / split

    def mix_dir(self, name: str) -> Path:
        return self.data_dir / "pipeline_ready" / name

    # ---- checkpoints ------------------------------------------------------

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    def checkpoint_dir(self, model: str) -> Path:
        return self.checkpoints_dir / validate_model_name(model)

    def run_card(self, model: str) -> Path:
        return self.checkpoint_dir(model) / RUN_CARD

    @property
    def launches_dir(self) -> Path:
        return self.checkpoints_dir / "_launches"

    @property
    def rename_log(self) -> Path:
        return self.checkpoints_dir / RENAME_LOG

    def latest_checkpoint(self, model: str) -> Path | None:
        """Newest ``*.ckpt`` by mtime, skipping the dry-run placeholder (the rule pretrain uses)."""
        from utils.resume import find_last_checkpoint

        return find_last_checkpoint(self.checkpoint_dir(model))

    def list_models(self) -> list[str]:
        if not self.checkpoints_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self.checkpoints_dir.iterdir()
            if p.is_dir() and p.name not in _RESERVED_NAMES
        )

    # ---- embeddings -------------------------------------------------------

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    def embedding_dir(self, model: str) -> Path:
        return self.embeddings_dir / validate_model_name(model)

    def embedding_file(self, model: str, modality: str, part: str) -> Path:
        label = canonical_modality(modality)
        part = _validate_part(part, EMBEDDING_PARTS)
        return self.embedding_dir(model) / f"{label}_{part}.h5ad"

    def embed_card(self, model: str) -> Path:
        return self.embedding_dir(model) / EMBED_CARD

    # ---- results ----------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def representation_root(self) -> Path:
        return self.results_dir / "representation"

    def representation_dir(self, model: str) -> Path:
        return self.representation_root / validate_model_name(model)

    def alignment_dir(self, model: str) -> Path:
        return self.representation_root / "_alignment" / validate_model_name(model)

    @property
    def tasks_root(self) -> Path:
        return self.results_dir / "tasks"

    def task_dir(self, task: str, model: str) -> Path:
        return self.tasks_root / task / validate_model_name(model)

    @property
    def assessment_root(self) -> Path:
        return self.results_dir / "assessment"

    def assessment_dir(self, name: str) -> Path:
        return self.assessment_root / name

    # ---- everything that belongs to / mentions one model -------------------

    def model_owned_dirs(self, model: str) -> list[Path]:
        """Every directory named after ``model``, existing or not.

        Task directories are discovered (one per task already on disk), so the
        list grows with the results instead of being hard-coded per task.
        """
        validate_model_name(model)
        dirs = [
            self.checkpoint_dir(model),
            self.embedding_dir(model),
            self.representation_dir(model),
            self.alignment_dir(model),
        ]
        if self.tasks_root.is_dir():
            dirs.extend(
                task / model for task in sorted(self.tasks_root.iterdir()) if task.is_dir()
            )
        return dirs

    def model_reference_files(self) -> list[Path]:
        """Structured (JSON) files outside a model's own directories that may name it."""
        files: list[Path] = []
        if self.launches_dir.is_dir():
            files.extend(sorted(self.launches_dir.glob("*.json")))
        if self.assessment_root.is_dir():
            files.extend(sorted(self.assessment_root.glob(f"*/{ASSESSMENT_MANIFEST}")))
        return files
