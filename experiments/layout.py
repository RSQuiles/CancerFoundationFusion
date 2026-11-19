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
    │   ├── <variant>/{modality}_{part}.h5ad      the same, embedded differently (e.g. own_panel)
    │   └── geneicl_<suite>/<dataset>.npz          GeneICL-suite embeddings (+ embed.json)
    └── results/                                  one directory per evaluation task
        ├── representation/<model>/[<variant>/]
        ├── representation/_alignment/<model>/[<variant>/]
        ├── reconstruction/<model>/<plan>[_<head>]/
        ├── reconstruction/_compare/<plan>/<name>/  model comparison tables (+ manifest.json)
        ├── reconstruction/_plots/<plan>/<name>/    model comparison figures (+ manifest.json)
        ├── geneicl_<suite>/<model>/                 e.g. geneicl_benchmark/
        ├── geneicl_<suite>/_compare/<name>/, _plots/<name>/
        ├── scib_benchmark/<name>/                   one scIB analysis (+ manifest.json)
        ├── _checks/<check>/                         data checks (e.g. store_<modality>_<split>)
        └── _eval_runs/<config>_<timestamp>/         evaluate.run_evaluation records (configs, logs)

Rules every writer follows, so ``rename_model.py`` never has to guess:
per-model output lives under a directory named after the model (never in a
filename), and an analysis spanning several models lists them in its
``manifest.json`` under ``"models"``. A task directory holds model directories and
``_``-prefixed ones only (``_compare``, ``_plots``, ``_checks``), except
``scib_benchmark``, whose subdirectories are analyses, never models.

"""

from __future__ import annotations

import json
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
_RESERVED_NAMES = {"_launches", "_alignment", "_renames.jsonl", "_plots", "_compare", "_checks"}
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

RUN_CARD = "run.json"
# Result tasks that are not "one directory per model": representation has its own
# layout, and scib_benchmark/<name>/ are analyses spanning models.
REPRESENTATION_TASK = "representation"
SCIB_TASK = "scib_benchmark"
_NON_MODEL_TASKS = {REPRESENTATION_TASK, SCIB_TASK}
DEFAULT_COMPARISON = "default"
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


def validate_variant(name: str) -> str:
    """Return an embedding variant name unchanged, or raise.

    A variant is a subdirectory of the model's embedding (and figure) directory, so it
    follows the model-name rules and may not start with ``geneicl_`` (the GeneICL-suite
    directories live at the same level).
    """
    if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid embedding variant {name!r}: use letters, digits, '_', '.', '-' "
            "and start with a letter or digit."
        )
    if name.startswith("geneicl_") or name in _RESERVED_NAMES:
        raise ValueError(f"Embedding variant {name!r} is reserved by the layout.")
    return name


def _with_variant(path: Path, variant: str | None) -> Path:
    return path if variant is None else path / validate_variant(variant)


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

    def split_source(self, modality: str, split: str) -> Path:
        """The h5ad a split's row names refer to.

        The source recorded in its ``split.json`` (a split made with ``--input``, e.g. from
        ``BULK_pretraining.h5ad``), else ``raw_h5ad``. Readers of a split's rows use this,
        so they never look the names up in another file.
        """
        card = self.split_card(modality, split)
        if card.exists():
            path = (json.loads(card.read_text(encoding="utf-8")).get("source") or {}).get("path")
            if path:
                return Path(path)
        return self.raw_h5ad(modality)

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

    def embedding_dir(self, model: str, variant: str | None = None) -> Path:
        """``embeddings/<model>/``, or ``embeddings/<model>/<variant>/`` for embeddings
        of the same model made differently (e.g. ``own_panel``: one gene panel per
        modality instead of the joint one). Inside the model's directory, so a rename
        moves it."""
        return _with_variant(self.embeddings_dir / validate_model_name(model), variant)

    def embedding_file(self, model: str, modality: str, part: str, variant: str | None = None) -> Path:
        label = canonical_modality(modality)
        part = _validate_part(part, EMBEDDING_PARTS)
        return self.embedding_dir(model, variant) / f"{label}_{part}.h5ad"

    def embed_card(self, model: str, variant: str | None = None) -> Path:
        return self.embedding_dir(model, variant) / EMBED_CARD

    def geneicl_embedding_dir(self, model: str, suite: str) -> Path:
        """``embeddings/<model>/geneicl_<suite>/``: one ``<dataset>.npz`` per GeneICL dataset."""
        if not _MODEL_NAME_RE.match(str(suite)):
            raise ValueError(f"Invalid GeneICL suite name {suite!r}")
        return self.embedding_dir(model) / f"geneicl_{suite}"

    # ---- results ----------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def representation_root(self) -> Path:
        return self.results_dir / "representation"

    def representation_dir(self, model: str, variant: str | None = None) -> Path:
        return _with_variant(self.representation_root / validate_model_name(model), variant)

    def alignment_dir(self, model: str, variant: str | None = None) -> Path:
        return _with_variant(self.representation_root / "_alignment" / validate_model_name(model), variant)

    def task_root(self, task: str) -> Path:
        """``results/<task>/`` (e.g. ``reconstruction``, ``geneicl_benchmark``, ``scib_benchmark``)."""
        if not _MODEL_NAME_RE.match(str(task)):
            raise ValueError(f"Invalid task name {task!r}")
        return self.results_dir / task

    def task_dir(self, task: str, model: str) -> Path:
        """``results/<task>/<model>/``: one model's results on a task."""
        return self.task_root(task) / validate_model_name(model)

    def _task_sub(self, task: str, kind: str, parts: tuple[str, ...]) -> Path:
        if not parts:
            raise ValueError(f"{kind} needs at least the comparison name")
        for part in parts:
            if not _MODEL_NAME_RE.match(str(part)):
                raise ValueError(f"Invalid {kind} directory name {part!r}")
        return self.task_root(task).joinpath(kind, *parts)

    def task_plot_dir(self, task: str, *parts: str) -> Path:
        """``results/<task>/_plots/<part>/...``: figures of one named model comparison on a
        task, e.g. ``(reconstruction, <plan>, <name>)`` or ``(geneicl_<suite>, <name>)``.
        The manifest there lists the models under ``"models"``. ``_plots`` is reserved, so
        it never collides with a model directory."""
        return self._task_sub(task, "_plots", parts)

    def task_compare_dir(self, task: str, *parts: str) -> Path:
        """``results/<task>/_compare/<part>/...``: the tables of one named model comparison
        (``compare.py``), laid out like ``task_plot_dir``. The manifest lists the models."""
        return self._task_sub(task, "_compare", parts)

    def scib_dir(self, name: str) -> Path:
        """``results/scib_benchmark/<name>/``: one scIB analysis over several models."""
        if not _MODEL_NAME_RE.match(str(name)):
            raise ValueError(f"Invalid scIB analysis name {name!r}")
        return self.task_root(SCIB_TASK) / name

    @property
    def checks_dir(self) -> Path:
        """``results/_checks/``: data checks that belong to no evaluation task (store checks)."""
        return self.results_dir / "_checks"

    def check_dir(self, name: str) -> Path:
        if not _MODEL_NAME_RE.match(str(name)):
            raise ValueError(f"Invalid check name {name!r}")
        return self.checks_dir / name

    def model_task_roots(self) -> list[Path]:
        """Task directories that hold one directory per model (not representation, scIB,
        nor the ``_``-prefixed infrastructure directories)."""
        if not self.results_dir.is_dir():
            return []
        return [d for d in sorted(self.results_dir.iterdir())
                if d.is_dir() and not d.name.startswith(("_", ".")) and d.name not in _NON_MODEL_TASKS]

    @property
    def eval_runs_dir(self) -> Path:
        """``results/_eval_runs/``: one directory per ``evaluate.run_evaluation`` run
        (resolved step configs, logs, summary). ``_``-prefixed, so never mistaken for a
        task or a model."""
        return self.results_dir / "_eval_runs"

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
        dirs.extend(task / model for task in self.model_task_roots())
        return dirs

    def model_reference_files(self) -> list[Path]:
        """Structured (JSON) files outside a model's own directories that may name it."""
        files: list[Path] = []
        if self.launches_dir.is_dir():
            files.extend(sorted(self.launches_dir.glob("*.json")))
        if self.checkpoints_dir.is_dir():
            # Other models' cards and configs: a warm-started model names its parent
            # (run.json "init_from", warm_start.json) and holds paths into its directory.
            for model_dir in sorted(self.checkpoints_dir.iterdir()):
                if model_dir.is_dir() and not model_dir.name.startswith("_"):
                    files.extend(sorted(model_dir.glob("*.json")))
        if self.results_dir.is_dir():
            # Comparison manifests sit at any depth under _compare / _plots (with or without
            # a plan level); scIB analyses directly under scib_benchmark/.
            for kind in ("_compare", "_plots"):
                files.extend(sorted(self.results_dir.glob(f"*/{kind}/**/{ASSESSMENT_MANIFEST}")))
            files.extend(sorted(self.task_root(SCIB_TASK).glob(f"*/{ASSESSMENT_MANIFEST}")))
        return files
