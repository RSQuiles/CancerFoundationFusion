"""CLI plumbing shared by the representation scripts: YAML defaults and key=value lists."""

from __future__ import annotations

import argparse
from pathlib import Path


SECTIONS = ("umap", "alignment")


def parse_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse ``argv``; a ``--config`` YAML supplies defaults, the command line wins.

    Top-level keys apply to every representation script; a ``umap:`` or
    ``alignment:`` block applies only to that script and overrides the top level.
    Keys are the long option names with ``-`` or ``_`` (``stratify_by`` or
    ``stratify-by``). An unknown key is an error rather than silently ignored.
    """
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file of defaults (keys = option names); CLI flags override it")
    first, _ = parser.parse_known_args(argv)
    if first.config is not None:
        import yaml

        raw = yaml.safe_load(first.config.read_text(encoding="utf-8")) or {}
        merged = {k: v for k, v in raw.items() if k not in SECTIONS}
        merged.update(raw.get(section) or {})
        known = {a.dest for a in parser._actions}
        defaults = {str(k).replace("-", "_"): v for k, v in merged.items()}
        unknown = sorted(set(defaults) - known)
        if unknown:
            parser.error(f"{first.config}: unknown key(s) {unknown} for the '{section}' script")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def parse_quotas(specs: list[str] | None, modalities: list[str]) -> dict[str, int | None]:
    """``["sc=5000", "bulk=all"]`` -> ``{"sc": 5000, "bulk": None}``; unspecified = all."""
    from experiments.layout import canonical_modality

    quotas: dict[str, int | None] = {m: None for m in modalities}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--per-modality {spec!r} must look like MODALITY=N or MODALITY=all")
        mod, value = spec.split("=", 1)
        label = canonical_modality(mod)
        quotas[label] = None if value.strip().lower() in {"all", "none", ""} else int(value.replace("_", ""))
    return quotas
