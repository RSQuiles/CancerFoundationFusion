"""Warm-starting a new model from another model's checkpoint.

A *warm start* builds a new model from its own config (name, data store, conditions,
heads, losses, sampler) and copies in the weights it shares with a parent checkpoint.
Nothing else of the parent is used: no optimizer, scheduler, epoch or step counter,
and no W&B run. It is the opposite of a resume (``utils/resume.py``), which continues
the same run.

The parent is never rebuilt. Only its ``state_dict`` and ``hyper_parameters`` are read,
so changes to ``CancerFoundation.__init__`` since it was saved cannot break a warm
start, and every existing checkpoint works without migration.

Every ``TransformerModule`` weight belongs to one group (``classify``):

- ``backbone``: gene and value encoders, the transformer, the generative flag. They
  must match the parent in name and shape (``strict_backbone``), or the warm start is
  refused.
- ``decoders``: the expression and MVC decoders. Copied when the shape matches and
  their input concatenates the same conditions in the parent and the child; else
  re-initialized.
- ``conditions``: ``condition_encoders.<col>``. ``modality`` has fixed codes (bulk 0,
  sc 1, pseudobulk 2) and is copied as is. Every other column is remapped row by row
  by *category name*, because a store's codes are its own sorted categories. New
  categories keep their fresh initialization.
- ``heads``: the DAT discriminators. Not transferred unless included, and then only
  where the class codes agree.
- anything else (e.g. CDD state): never transferred, only reported.

Torch-free at import time: torch is imported only inside ``warm_start_model``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.resume import DRY_RUN_CHECKPOINT, find_last_checkpoint, now_utc

WARM_START_FILE = "warm_start.json"
GROUPS = ("backbone", "decoders", "conditions", "heads")
DEFAULT_INCLUDE = ("backbone", "decoders", "conditions")
# Conditions whose codes are fixed by the code, not by a store's mapping.json.
FIXED_CODE_CONDITIONS = ("modality",)

_COMPILE_PREFIX = "_orig_mod."
_MODULE_PREFIX = "model."
# (prefix, group). A prefix without a trailing dot names one parameter exactly.
_PREFIX_GROUPS = (
    ("gene_encoder.", "backbone"),
    ("encoder.", "backbone"),  # gen_method="orig"
    ("value_encoder.", "backbone"),
    ("transformer_encoder.", "backbone"),
    ("flag_encoder.", "backbone"),  # gen_method="orig"
    ("generative_flag", "backbone"),
    ("mvc_decoder.", "decoders"),
    ("decoder.", "decoders"),
    ("condition_encoders.", "conditions"),
    ("grad_reverse_discriminators.", "heads"),
)
_DECODERS = ("decoder", "mvc_decoder")


class WarmStartError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Keys and groups
# --------------------------------------------------------------------------- #


def normalize_key(key: str) -> str | None:
    """``CancerFoundation`` state key -> ``TransformerModule`` key; None if outside it.

    Strips the ``torch.compile`` segment wherever it sits, then the ``model.`` prefix.
    """
    key = key.replace(_COMPILE_PREFIX, "")
    if not key.startswith(_MODULE_PREFIX):
        return None
    return key[len(_MODULE_PREFIX):]


def classify(name: str) -> str:
    """Group of a ``TransformerModule`` key: one of ``GROUPS``, or ``"other"``."""
    for prefix, group in _PREFIX_GROUPS:
        if prefix.endswith("."):
            if name.startswith(prefix):
                return group
        elif name == prefix:
            return group
    return "other"


def encoded_conditions(names: Iterable[str]) -> tuple[str, ...]:
    """Columns with a ``ConditionEncoder``, in state-dict (= construction) order."""
    out: list[str] = []
    for name in names:
        if name.startswith("condition_encoders."):
            col = name.split(".", 2)[1]
            if col not in out:
                out.append(col)
    return tuple(out)


def decoder_condition_layout(where_condition: str | None, encoded: Sequence[str]) -> dict[str, tuple]:
    """Which encoded conditions each decoder's input concatenates (``module.py``).

    The MVC decoder takes them whenever conditions are encoded and ``where_condition``
    is not ``"none"``; the expression decoder only with ``"end"``.
    """
    enc = tuple(encoded)
    if not enc or where_condition == "none":
        return {"decoder": (), "mvc_decoder": ()}
    return {"decoder": enc if where_condition == "end" else (), "mvc_decoder": enc}


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


@dataclass
class WarmStartOptions:
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = ()
    strict_backbone: bool = True
    parent_mapping: str | None = None

    def __post_init__(self) -> None:
        self.include = tuple(self.include or DEFAULT_INCLUDE)
        self.exclude = tuple(self.exclude or ())
        unknown = sorted((set(self.include) | set(self.exclude)) - set(GROUPS))
        if unknown:
            raise WarmStartError(f"unknown warm-start group(s) {unknown}; choose from {list(GROUPS)}")

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(g for g in GROUPS if g in self.include and g not in self.exclude)

    @classmethod
    def from_args(cls, args: Any) -> "WarmStartOptions":
        return cls(
            include=tuple(getattr(args, "init_include", None) or DEFAULT_INCLUDE),
            exclude=tuple(getattr(args, "init_exclude", None) or ()),
            strict_backbone=not getattr(args, "init_loose_backbone", False),
            parent_mapping=getattr(args, "init_parent_mapping", None),
        )


# --------------------------------------------------------------------------- #
# Planning (pure: shapes and mappings in, actions out)
# --------------------------------------------------------------------------- #


@dataclass
class Action:
    key: str
    group: str
    rows: list[tuple[int, int]] | None = None  # (child row, parent row); None = whole tensor


@dataclass
class WarmStartPlan:
    actions: list[Action] = field(default_factory=list)
    remapped: dict[str, dict] = field(default_factory=dict)  # key -> rows copied / new categories
    reinitialized: dict[str, str] = field(default_factory=dict)  # child key -> reason
    skipped: dict[str, str] = field(default_factory=dict)  # parent key -> reason
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for action in self.actions:
            kind = "remapped" if action.rows is not None else "copied"
            out.setdefault(action.group, {}).setdefault(kind, 0)
            out[action.group][kind] += 1
        for key in self.reinitialized:
            group = classify(key)
            out.setdefault(group, {}).setdefault("reinitialized", 0)
            out[group]["reinitialized"] += 1
        for key in self.skipped:
            group = classify(key)
            out.setdefault(group, {}).setdefault("skipped", 0)
            out[group]["skipped"] += 1
        return out


def _condition_rows(
    col: str,
    parent_rows: int,
    parent_mapping: Mapping[str, Mapping[str, int]] | None,
    child_mapping: Mapping[str, Mapping[str, int]] | None,
) -> tuple[list[tuple[int, int]] | None, str, list[str]]:
    """(row pairs, reason if None, new category names) for one condition column."""
    pm = (parent_mapping or {}).get(col)
    if pm is None:
        return None, (
            "the parent's category names are unknown (no condition_mapping in the "
            "checkpoint and no mapping.json found), so codes cannot be matched"
        ), []
    if len(pm) != parent_rows:
        return None, (
            f"the parent mapping lists {len(pm)} categories but its encoder has "
            f"{parent_rows} rows: not the mapping it was trained with"
        ), []
    cm = (child_mapping or {}).get(col)
    if cm is None:
        return None, "the column is not in this store's mapping.json", []
    pairs = sorted((int(code), int(pm[cat])) for cat, code in cm.items() if cat in pm)
    new = sorted(str(cat) for cat in cm if cat not in pm)
    if not pairs:
        return None, "no category in common with the parent", new
    return pairs, "", new


def plan_warm_start(
    parent_shapes: Mapping[str, tuple],
    child_shapes: Mapping[str, tuple],
    *,
    options: WarmStartOptions,
    parent_mapping: Mapping[str, Mapping[str, int]] | None = None,
    child_mapping: Mapping[str, Mapping[str, int]] | None = None,
    parent_where: str | None = None,
    child_where: str | None = None,
) -> WarmStartPlan:
    """Decide, key by key, what the child takes from the parent.

    Both shape maps use ``TransformerModule`` keys (see ``normalize_key``). Raises
    ``WarmStartError`` when the backbone does not match and ``strict_backbone`` is set.
    """
    groups = set(options.groups)
    plan = WarmStartPlan()
    errors: list[str] = []

    parent_layout = (
        decoder_condition_layout(parent_where, encoded_conditions(parent_shapes))
        if parent_where is not None else None
    )
    child_layout = decoder_condition_layout(child_where, encoded_conditions(child_shapes))
    if parent_layout is None:
        plan.notes.append(
            "the parent stores no where_condition; decoders are matched on shape only"
        )
    row_cache: dict[str, tuple] = {}

    def condition_rows(col: str, table_key: str):
        if col not in row_cache:
            rows = parent_shapes.get(table_key, (0,))[0]
            row_cache[col] = _condition_rows(col, rows, parent_mapping, child_mapping)
        return row_cache[col]

    for name, shape in child_shapes.items():
        shape = tuple(shape)
        group = classify(name)
        if group == "other":
            plan.reinitialized[name] = "not a transferable weight; kept as built"
            continue
        if group not in groups:
            plan.reinitialized[name] = f"group '{group}' is not transferred"
            continue
        if name not in parent_shapes:
            if group == "backbone" and options.strict_backbone:
                errors.append(f"{name}: not in the parent checkpoint")
            else:
                plan.reinitialized[name] = "not in the parent checkpoint (new)"
            continue
        pshape = tuple(parent_shapes[name])

        if group == "backbone":
            if pshape != shape:
                if options.strict_backbone:
                    errors.append(f"{name}: parent {pshape}, child {shape}")
                else:
                    plan.reinitialized[name] = f"shape differs: parent {pshape}, child {shape}"
            else:
                plan.actions.append(Action(name, group))
            continue

        if group == "decoders":
            decoder = name.split(".", 1)[0]
            if parent_layout is not None and parent_layout[decoder] != child_layout[decoder]:
                plan.reinitialized[name] = (
                    f"its input concatenates conditions {list(parent_layout[decoder])} in "
                    f"the parent and {list(child_layout[decoder])} here"
                )
            elif pshape != shape:
                plan.reinitialized[name] = f"shape differs: parent {pshape}, child {shape}"
            else:
                plan.actions.append(Action(name, group))
            continue

        col = name.split(".", 2)[1]
        if group == "conditions":
            if col in FIXED_CODE_CONDITIONS:
                if pshape != shape:
                    plan.reinitialized[name] = f"shape differs: parent {pshape}, child {shape}"
                else:
                    plan.actions.append(Action(name, group))
                continue
            table_key = f"condition_encoders.{col}.embedding.weight"
            pairs, reason, new = condition_rows(col, table_key)
            if pairs is None:
                plan.reinitialized[name] = reason
                continue
            if name == table_key:
                if pshape[1:] != shape[1:]:
                    plan.reinitialized[name] = f"shape differs: parent {pshape}, child {shape}"
                    continue
                pairs = [(c, p) for c, p in pairs if c < shape[0] and p < pshape[0]]
                plan.actions.append(Action(name, group, rows=pairs))
                plan.remapped[name] = {
                    "rows_copied": len(pairs),
                    "rows": shape[0],
                    "new_categories": new,
                }
            elif pshape != shape:
                plan.reinitialized[name] = f"shape differs: parent {pshape}, child {shape}"
            else:
                plan.actions.append(Action(name, group))
            continue

        # heads: a discriminator's output rows are the column's class codes.
        if pshape != shape:
            plan.reinitialized[name] = (
                f"shape differs: parent {pshape}, child {shape}"
                + (" (e.g. the old 3-class modality head)" if col == "modality" else "")
            )
        elif col in FIXED_CODE_CONDITIONS or (
            parent_mapping is not None
            and child_mapping is not None
            and col in parent_mapping
            and dict(parent_mapping[col]) == dict(child_mapping.get(col) or {})
        ):
            plan.actions.append(Action(name, group))
        else:
            plan.reinitialized[name] = "class codes differ between the parent's store and this one"

    if options.strict_backbone and "backbone" in groups:
        for name in parent_shapes:
            if classify(name) == "backbone" and name not in child_shapes:
                errors.append(f"{name}: in the parent but not built here")

    if errors:
        raise WarmStartError(
            "the backbone does not match the parent checkpoint (set init_loose_backbone "
            "to re-initialize the mismatched weights instead):\n  " + "\n  ".join(errors)
        )

    for name in parent_shapes:
        if name in child_shapes:
            continue
        group = classify(name)
        if group == "other":
            plan.skipped[name] = "not a transferable weight"
        elif group not in groups:
            plan.skipped[name] = f"group '{group}' is not transferred"
        else:
            plan.skipped[name] = "not built in this model"
    return plan


# --------------------------------------------------------------------------- #
# Parent lookup and metadata
# --------------------------------------------------------------------------- #


def resolve_init_checkpoint(spec: str | Path) -> Path:
    """A ``.ckpt`` path, or a model's checkpoint directory -> its newest checkpoint."""
    path = Path(spec)
    if path.is_dir():
        found = find_last_checkpoint(path)
        if found is None:
            raise FileNotFoundError(f"init_from: no checkpoint in {path}")
        return found
    if path.name == DRY_RUN_CHECKPOINT:
        raise ValueError(f"init_from: {path} is the dry-run placeholder, not a checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"init_from: checkpoint not found: {path}")
    return path


def _read_mapping_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_parent_mapping(
    checkpoint: Mapping[str, Any], checkpoint_path: Path, explicit: str | None = None
) -> tuple[dict | None, str]:
    """The parent's ``{column: {category: code}}`` and where it came from.

    Order: an explicit ``mapping.json``; the checkpoint's own ``condition_mapping``
    (written since warm starts exist); ``mapping.json`` of the store in the parent's
    ``data_path`` hparam; the ``train_path`` in its directory's ``config.resolved.json``.
    """
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            path = path / "mapping.json"
        found = _read_mapping_file(path)
        if found is None:
            raise FileNotFoundError(f"init_parent_mapping: cannot read {path}")
        return found, str(path)
    stored = checkpoint.get("condition_mapping")
    if stored:
        return dict(stored), "checkpoint"
    data_path = (checkpoint.get("hyper_parameters") or {}).get("data_path")
    if data_path:
        path = Path(str(data_path)) / "mapping.json"
        found = _read_mapping_file(path)
        if found is not None:
            return found, str(path)
    resolved = _read_mapping_file(Path(checkpoint_path).parent / "config.resolved.json") or {}
    train_path = resolved.get("train_path")
    if train_path:
        path = Path(str(train_path)) / "mapping.json"
        found = _read_mapping_file(path)
        if found is not None:
            return found, str(path)
    return None, "none"


def check_vocab(parent_vocab: Any, child_vocab: Any) -> str | None:
    """Raise if both vocabularies are known and differ; a note if the parent's is unknown."""
    try:
        parent = dict(parent_vocab) if parent_vocab is not None else None
        child = dict(child_vocab) if child_vocab is not None else None
    except (TypeError, ValueError):
        return "could not compare the vocabularies"
    if parent is None or child is None:
        return "the parent stores no vocabulary; gene ids are assumed to agree"
    if parent != child:
        only_parent = len(set(parent) - set(child))
        only_child = len(set(child) - set(parent))
        moved = sum(1 for g in set(parent) & set(child) if parent[g] != child[g])
        raise WarmStartError(
            "the parent's gene vocabulary differs from this store's "
            f"({only_parent} genes only in the parent, {only_child} only here, "
            f"{moved} with another token id); the gene embeddings would be scrambled"
        )
    return None


def parent_model_name(checkpoint_path: Path) -> str | None:
    """The model a checkpoint belongs to, when it sits in ``checkpoints/<model>/``."""
    card = _read_mapping_file(Path(checkpoint_path).parent / "run.json")
    if card and card.get("name"):
        return str(card["name"])
    return None


def build_record(
    plan: WarmStartPlan,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    options: WarmStartOptions,
    mapping_source: str,
) -> dict:
    """Everything ``warm_start.json`` holds; ``compact_record`` goes into checkpoints."""
    notes = list(plan.notes)
    return {
        "created": now_utc(),
        "from": {
            "model": parent_model_name(checkpoint_path),
            "checkpoint": str(checkpoint_path),
            "file": Path(checkpoint_path).name,
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        },
        "groups": list(options.groups),
        "strict_backbone": options.strict_backbone,
        "mapping_source": mapping_source,
        "summary": plan.summary(),
        "remapped": plan.remapped,
        "reinitialized": plan.reinitialized,
        "skipped": plan.skipped,
        "notes": notes,
    }


def compact_record(record: Mapping[str, Any]) -> dict:
    keep = ("created", "from", "groups", "strict_backbone", "mapping_source", "summary", "notes")
    return {k: record[k] for k in keep if k in record}


def write_record(save_dir: str | Path, record: Mapping[str, Any]) -> Path:
    path = Path(save_dir) / WARM_START_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def format_summary(record: Mapping[str, Any]) -> str:
    src = record.get("from") or {}
    head = (
        f"[warm start] from {src.get('model') or '?'}/{src.get('file')} "
        f"(epoch {src.get('epoch')}, step {src.get('global_step')}); "
        f"groups {record.get('groups')}; category names from {record.get('mapping_source')}"
    )
    lines = [head]
    for group, counts in sorted((record.get("summary") or {}).items()):
        lines.append(f"  {group}: " + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())))
    for key, info in (record.get("remapped") or {}).items():
        new = info.get("new_categories") or []
        lines.append(
            f"  remapped {key}: {info['rows_copied']}/{info['rows']} rows"
            + (f", new: {new[:8]}{' ...' if len(new) > 8 else ''}" if new else "")
        )
    shown = [
        (k, why) for k, why in (record.get("reinitialized") or {}).items()
        if classify(k) in ("decoders", "conditions", "heads")
    ]
    for key, why in shown[:20]:
        lines.append(f"  re-initialized {key}: {why}")
    for note in record.get("notes") or []:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Applying (torch)
# --------------------------------------------------------------------------- #


def _load_checkpoint(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        # Legacy (non-zip) checkpoints cannot be memory-mapped.
        return torch.load(path, map_location="cpu", weights_only=False)


def warm_start_model(
    model: Any,
    spec: str | Path,
    child_mapping: Mapping[str, Mapping[str, int]] | None,
    options: WarmStartOptions | None = None,
) -> dict:
    """Copy the parent's shared weights into ``model`` (a ``CancerFoundation``) in place.

    Returns the full record (see ``build_record``). Raises ``WarmStartError`` on a
    vocabulary or backbone mismatch, before any weight is changed.
    """
    import torch

    options = options or WarmStartOptions()
    path = resolve_init_checkpoint(spec)
    checkpoint = _load_checkpoint(path)
    hparams = dict(checkpoint.get("hyper_parameters") or {})

    notes = []
    vocab_note = check_vocab(hparams.get("vocab"), getattr(model, "vocab", None))
    if vocab_note:
        notes.append(vocab_note)

    parent = {}
    for key, value in (checkpoint.get("state_dict") or {}).items():
        name = normalize_key(key)
        if name is not None and hasattr(value, "shape"):
            parent[name] = value
    child = {}
    for key, value in model.state_dict().items():
        name = normalize_key(key)
        if name is not None:
            child[name] = value  # shares storage with the parameter or buffer

    parent_mapping, mapping_source = find_parent_mapping(checkpoint, path, options.parent_mapping)
    plan = plan_warm_start(
        {k: tuple(v.shape) for k, v in parent.items()},
        {k: tuple(v.shape) for k, v in child.items()},
        options=options,
        parent_mapping=parent_mapping,
        child_mapping=child_mapping,
        parent_where=hparams.get("where_condition"),
        child_where=getattr(model, "where_condition", None),
    )
    plan.notes.extend(notes)

    with torch.no_grad():
        for action in plan.actions:
            src, dst = parent[action.key], child[action.key]
            if action.rows is None:
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            elif action.rows:
                c_idx = torch.tensor([c for c, _ in action.rows], dtype=torch.long, device=dst.device)
                p_idx = torch.tensor([p for _, p in action.rows], dtype=torch.long)
                dst[c_idx] = src[p_idx].to(device=dst.device, dtype=dst.dtype)

    record = build_record(plan, path, checkpoint, options, mapping_source)
    del checkpoint
    return record
