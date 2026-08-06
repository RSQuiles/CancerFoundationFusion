"""
Shared config machinery for the comparison plots.

Both ``plot_ablation_benchmark.py`` (downstream task results) and
``plot_unified_metrics_table.py`` (internal unified-FM metrics) are driven by a YAML
config that hand-picks runs — possibly from different ablation directories — gives
them display names, and optionally arranges them into groups.  The two differ only in
which metrics they read; the run-selection grammar below is identical, so it lives
here instead of being copied.

Config grammar (the part parsed by :func:`parse_groups`)
--------------------------------------------------------
::

    # Either a flat list of runs …
    experiments:
      - name: "Baseline"
        path: /abs/path/to/ablation_a/baseline      # a model dir
      - name: "PCA"
        dir:  /abs/path/to/ablation_a               # an ablation dir …
        model: pca_baseline                         # … plus a model in it

    # … or groups of runs, possibly from different ablations:
    groups:
      - name: "Big condition"
        dir: /abs/path/to/ablation_big_condition
        experiments:
          - {name: Baseline, model: baseline}
          - {name: "No DAT", model: no_dat}
      - name: "United data"
        dir: /abs/path/to/ablation_united_data
        all_models: true            # add every model dir found there
        exclude: [pca_baseline]     # optional, applies to all_models

``name`` defaults to the model directory's name.  Display names must be unique across
the whole config, since they label the rows/bars.

What counts as a "model dir" differs per script — a downstream benchmark needs
``metrics/results_*.json``, the unified-metrics table needs
``metrics/unified_metrics.json`` — so :func:`is_model_dir` takes the glob patterns and
:func:`parse_groups` takes the resulting predicate.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.colors as mcolors

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

# Grouped mode: one base hue per group, lightened across its members.
# (Hard-coded copy of seaborn's "deep" palette — seaborn itself is not a dependency.)
GROUP_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
GROUP_GAP = 1.2                                  # bar-width units between groups
GROUP_SHADES = ("lightsteelblue", "lightyellow")  # alternating band colours


def group_color(group_idx: int, member_idx: int, n_members: int) -> tuple:
    """Base hue from the group; successive members within it are lightened."""
    base = mcolors.to_rgba(GROUP_PALETTE[group_idx % len(GROUP_PALETTE)])
    lighten = 0.45 * (member_idx / max(n_members - 1, 1)) if n_members > 1 else 0.0
    return (
        base[0] + (1 - base[0]) * lighten,
        base[1] + (1 - base[1]) * lighten,
        base[2] + (1 - base[2]) * lighten,
        1.0,
    )


def grouped_layout(
    names: list[str],
    groups: list[tuple[str | None, list[str]]],
) -> tuple[list[float], list[tuple], list[tuple[float, float, str | None]]]:
    """
    Lay items out group by group, separated by ``GROUP_GAP``.

    Returns (x per item, colour per item, [(x_lo, x_hi, group_name), ...]), the first
    two in the same order as *names*.
    """
    index = {name: i for i, name in enumerate(names)}
    xs = [0.0] * len(names)
    colors = [(0.0, 0.0, 0.0, 1.0)] * len(names)
    spans: list[tuple[float, float, str | None]] = []

    cursor = 0.0
    for g_idx, (group_name, members) in enumerate(groups):
        span_lo = cursor
        for m_idx, name in enumerate(members):
            i = index[name]
            xs[i] = cursor
            colors[i] = group_color(g_idx, m_idx, len(members))
            cursor += 1.0
        spans.append((span_lo, cursor - 1.0, group_name))
        cursor += GROUP_GAP

    return xs, colors, spans


# --------------------------------------------------------------------------- #
# Model-dir detection
# --------------------------------------------------------------------------- #

def is_model_dir(
    path: Path,
    patterns: Iterable[str] = ("results_*.json",),
) -> bool:
    """True if *path* looks like a model directory.

    A model directory has a ``metrics/`` subfolder containing at least one file
    matching any of *patterns*.
    """
    metrics_dir = path / "metrics"
    if not metrics_dir.is_dir():
        return False
    return any(any(metrics_dir.glob(pat)) for pat in patterns)


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

def as_str_list(value, where: str) -> list[str]:
    """Coerce a config entry to a list of strings (list, or comma-separated string)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [m.strip() for m in value.split(",") if m.strip()]
    if isinstance(value, (list, tuple)):
        return [str(m) for m in value]
    sys.exit(f"ERROR: {where} must be a list or a comma-separated string.")


def resolve_experiment(entry, group_dir: Path | None, where: str) -> tuple[str, Path]:
    """Turn one experiment entry into a (display_name, model_dir) pair."""
    if isinstance(entry, str):
        entry = {"model": entry}
    if not isinstance(entry, dict):
        sys.exit(f"ERROR: {where} must be a mapping or a model name.")

    path = entry.get("path")
    model = entry.get("model") or entry.get("name")
    base = entry.get("dir", group_dir)

    if path is not None:
        model_dir = Path(path).expanduser()
    elif model is not None and base is not None:
        model_dir = Path(base).expanduser() / str(model)
    else:
        sys.exit(
            f"ERROR: {where} needs either 'path', or 'model' plus a 'dir' "
            "(on the entry or its group)."
        )

    name = str(entry.get("name") or model_dir.name)
    return name, model_dir.resolve()


def discover_group_models(
    group_dir: Path,
    exclude: set[str],
    already: set[Path],
    is_model: Callable[[Path], bool] = is_model_dir,
) -> list[tuple[str, Path]]:
    """All model dirs under *group_dir*, minus exclusions and ones already listed."""
    found: list[tuple[str, Path]] = []
    if not group_dir.is_dir():
        sys.exit(f"ERROR: group dir does not exist: {group_dir}")

    for child in sorted(group_dir.iterdir()):
        if not child.is_dir() or child.name in exclude:
            continue
        if not is_model(child):
            continue
        if child.resolve() in already:
            continue
        found.append((child.name, child.resolve()))
    return found


def parse_groups(
    raw: dict,
    path: Path,
    is_model: Callable[[Path], bool] = is_model_dir,
) -> list[tuple[str | None, list[tuple[str, Path]]]]:
    """
    Parse the ``groups:`` / ``experiments:`` half of a comparison config.

    Returns the plot-order list of ``(group_name, [(display_name, model_dir), ...])``.
    A flat ``experiments:`` list is modelled as a single unnamed group.

    *path* is only used for error messages; *is_model* decides what ``all_models``
    discovery accepts.
    """
    if "groups" in raw and "experiments" in raw:
        sys.exit(
            f"ERROR: {path} defines both 'groups' and 'experiments' at the top "
            "level — use one or the other (put per-group runs under "
            "groups[].experiments)."
        )

    if "groups" in raw:
        raw_groups = raw["groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            sys.exit(f"ERROR: 'groups' in {path} must be a non-empty list.")
    elif "experiments" in raw:
        raw_groups = [{"name": None, "experiments": raw["experiments"]}]
    else:
        sys.exit(f"ERROR: {path} must define either 'experiments' or 'groups'.")

    groups: list[tuple[str | None, list[tuple[str, Path]]]] = []
    seen_names: dict[str, Path] = {}

    for g_idx, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            sys.exit(f"ERROR: groups[{g_idx}] in {path} must be a mapping.")

        group_name = group.get("name")
        group_name = None if group_name is None else str(group_name)
        group_dir = group.get("dir")
        group_dir = Path(group_dir).expanduser() if group_dir else None
        where = f"groups[{g_idx}]" + (f" ('{group_name}')" if group_name else "")

        members: list[tuple[str, Path]] = []
        for e_idx, entry in enumerate(group.get("experiments") or []):
            members.append(
                resolve_experiment(entry, group_dir, f"{where}.experiments[{e_idx}]")
            )

        if group.get("all_models"):
            if group_dir is None:
                sys.exit(f"ERROR: {where} sets 'all_models' but has no 'dir'.")
            exclude = {str(x) for x in (group.get("exclude") or [])}
            listed = {d for _, d in members}
            members.extend(
                discover_group_models(group_dir, exclude, listed, is_model)
            )

        if not members:
            sys.exit(f"ERROR: {where} selects no experiments.")

        for name, model_dir in members:
            if name in seen_names:
                sys.exit(
                    f"ERROR: duplicate experiment name '{name}' "
                    f"({seen_names[name]} and {model_dir}). Give one an "
                    "explicit, unique 'name'."
                )
            seen_names[name] = model_dir

        groups.append((group_name, members))

    return groups


def parse_figsize(raw: dict, path: Path) -> tuple[float, float] | None:
    """Parse an optional ``figsize: [W, H]`` config key."""
    figsize = raw.get("figsize")
    if figsize is None:
        return None
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        sys.exit(f"ERROR: 'figsize' in {path} must be [width, height].")
    return (float(figsize[0]), float(figsize[1]))


def load_raw_config(path: Path) -> dict:
    """Read a YAML (or JSON) config and return its top-level mapping."""
    text = path.read_text()
    if path.suffix.lower() in {".json"}:
        import json

        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    if not isinstance(raw, dict):
        sys.exit(f"ERROR: {path} must contain a mapping at the top level.")
    return raw
