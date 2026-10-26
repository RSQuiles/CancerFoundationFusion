"""Obs-level filtering, stratified subsampling and group-aware splits.

Everything here works on an ``obs`` DataFrame and returns obs names or masks, so
callers can apply the result to a backed AnnData without loading X. Shared by
``data_preprocess/split_train_test.py``, ``embed/embed_splits.py`` and the new
``evaluate/`` UMAP scripts. Needs only numpy and pandas.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

UNKNOWN = "unknown"
_SEP = " | "


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def fold_label(value) -> str:
    """Casefold a label so 'Lung ', 'lung' and 'LUNG' compare equal (also - vs _)."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def parse_filter_specs(specs: Iterable[str] | None) -> dict[str, list[str]]:
    """``["tissue_general=lung,breast", "assay=10x"]`` -> ``{col: [values]}``.

    The same column given twice merges its values.
    """
    filters: dict[str, list[str]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Filter {spec!r} must look like COLUMN=value[,value...]")
        col, raw = spec.split("=", 1)
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if not col.strip() or not values:
            raise ValueError(f"Filter {spec!r} has an empty column or value list")
        filters.setdefault(col.strip(), []).extend(values)
    return filters


def filter_mask(
    obs: pd.DataFrame,
    filters: dict[str, list[str]] | None,
    label: str = "obs",
) -> np.ndarray:
    """Boolean mask of rows matching every ``{column: [values]}`` filter (AND across columns).

    Matching is case-, space- and hyphen-insensitive. Values with no matching row
    are reported with the labels that do exist. Raises when a column is missing or
    nothing is left: plotting everything under a filtered filename would
    misrepresent the figure.
    """
    mask = np.ones(len(obs), dtype=bool)
    for col, wanted_raw in (filters or {}).items():
        if col not in obs.columns:
            raise ValueError(
                f"Filter column {col!r} not in {label}; available: {sorted(obs.columns)[:40]}"
            )
        values = obs[col].astype(str)
        folded = values.map(fold_label)
        wanted = {fold_label(v) for v in wanted_raw}
        missing = sorted(wanted - set(folded.unique()))
        if missing:
            available = sorted(values.unique())
            shown = ", ".join(available[:30]) + (" ..." if len(available) > 30 else "")
            print(f"  [warn] no rows in {label}['{col}'] for: {', '.join(missing)}")
            print(f"         available values: {shown}")
        mask &= folded.isin(wanted).to_numpy()

    if filters and not mask.any():
        raise ValueError(f"Filters {filters} matched 0 of {len(obs)} rows in {label}.")
    if filters:
        print(f"  filter {filters}: {len(obs)} -> {int(mask.sum())} rows")
    return mask


def filter_tag(filters: dict[str, list[str]] | None) -> str:
    """Short filename-safe tag for a filter, e.g. ``'lung-breast'``; ``''`` when unfiltered."""
    if not filters:
        return ""
    slugs = list(
        dict.fromkeys(
            re.sub(r"[^0-9a-z]+", "_", str(v).strip().lower()).strip("_") or "x"
            for values in filters.values()
            for v in values
        )
    )
    joined = "-".join(slugs)
    if len(joined) > 60:
        joined = f"{slugs[0]}-and-{len(slugs) - 1}-more"
    return joined


# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #


def stratum_labels(obs: pd.DataFrame, strata: Iterable[str] | None) -> pd.Series:
    """Joint label of the ``strata`` columns per row; missing values become ``'unknown'``.

    No strata gives a single stratum, so every caller can treat the unstratified
    case the same way.
    """
    strata = list(strata or [])
    if not strata:
        return pd.Series("all", index=obs.index, dtype=object)
    missing = [c for c in strata if c not in obs.columns]
    if missing:
        raise ValueError(f"Stratify column(s) {missing} not in obs")
    parts = []
    for col in strata:
        values = obs[col].astype(object).where(obs[col].notna(), UNKNOWN).astype(str)
        parts.append(values.replace({"": UNKNOWN, "nan": UNKNOWN}))
    label = parts[0]
    for part in parts[1:]:
        label = label + _SEP + part
    return label


def allocate(sizes: pd.Series, n: int, min_per_stratum: int = 1) -> pd.Series:
    """Per-stratum sample sizes summing to ``min(n, total)``, proportional to ``sizes``.

    Each stratum gets at least ``min(min_per_stratum, size)`` (so rare strata stay
    visible) and never more than its size. Largest-remainder rounding hands out the
    rest. If the floors alone exceed ``n``, the floors win and the total exceeds
    ``n``; that is reported rather than silently dropping strata.
    """
    sizes = sizes.astype("int64")
    total = int(sizes.sum())
    if n >= total:
        return sizes.copy()

    floor = np.minimum(sizes.to_numpy(), max(0, int(min_per_stratum))).astype("int64")
    alloc = floor.copy()
    remaining = n - int(floor.sum())
    if remaining < 0:
        print(
            f"  [warn] {len(sizes)} strata x min_per_stratum={min_per_stratum} exceeds "
            f"n={n}; drawing {int(floor.sum())} rows instead"
        )
        return pd.Series(alloc, index=sizes.index)

    room = sizes.to_numpy() - alloc
    while remaining > 0 and room.sum() > 0:
        share = room / room.sum() * remaining
        step = np.minimum(np.floor(share).astype(int), room)
        if step.sum() == 0:
            # Hand the last few rows to the largest remainders.
            order = np.argsort(-(share - np.floor(share)), kind="stable")
            for i in order:
                if remaining == 0:
                    break
                if room[i] > 0:
                    alloc[i] += 1
                    room[i] -= 1
                    remaining -= 1
            break
        alloc += step
        room -= step
        remaining -= int(step.sum())
    return pd.Series(alloc, index=sizes.index)


def stratified_subsample(
    obs: pd.DataFrame,
    n: int | None,
    strata: Iterable[str] | None = None,
    min_per_stratum: int = 1,
    seed: int = 0,
) -> pd.Index:
    """Obs names of a stratified random subsample of ``n`` rows, in original order.

    ``n=None`` (or ``n >= len(obs)``) returns every row. Without strata it is a
    plain uniform subsample.
    """
    if n is None or n >= len(obs):
        return obs.index.copy()
    labels = stratum_labels(obs, strata)
    quotas = allocate(labels.value_counts(), int(n), min_per_stratum)
    rng = np.random.default_rng(seed)
    positions = np.arange(len(obs))
    chosen: list[np.ndarray] = []
    for stratum, quota in quotas.items():
        if quota <= 0:
            continue
        pool = positions[(labels == stratum).to_numpy()]
        chosen.append(rng.choice(pool, size=int(quota), replace=False))
    picked = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
    return obs.index[picked]


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #


NO_GROUP_VALUES: frozenset[str] = frozenset({"", "nan", "none", "na", "<na>"})


def parse_group_specs(specs: Iterable[str] | None) -> dict[str, frozenset[str]]:
    """``["donor_id", "paired:0"]`` -> ``{column: values meaning 'no group'}``.

    ``COL:V1,V2`` adds V1, V2 to the default no-group values (empty / NaN) for that
    column only, e.g. ``paired:0`` where 0 means unpaired. A donor id ``0`` stays a
    real donor unless said so.
    """
    out: dict[str, frozenset[str]] = {}
    for spec in specs or []:
        col, _, extra = spec.partition(":")
        values = {fold_label(v) for v in extra.split(",") if v.strip()} if extra else set()
        out[col.strip()] = NO_GROUP_VALUES | values
    return out


def _group_keys(series: pd.Series, col: str, no_group: frozenset[str]) -> pd.Series:
    """``'col=value'`` per row, NaN where the value means 'no group'."""
    values = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_null = series.isna().to_numpy() | values.map(fold_label).isin(no_group).to_numpy()
    return (col + "=" + values).where(~is_null)


def split_groups(
    obs: pd.DataFrame,
    group_cols: dict[str, frozenset[str]] | Iterable[str] | str | None = None,
) -> pd.Series:
    """Unit that must not straddle the split, per row.

    Rows sharing a value in ANY of ``group_cols`` belong to one unit, and so do units
    linked through different columns: a pair whose cells come from two donors joins
    both donors' rows. That is the connected component of the row/value graph;
    grouping on one column at a time would let the others leak across the split.
    Rows with no group value in any column are units of their own.

    ``group_cols`` is a list of columns (empty / NaN = no group) or the mapping
    returned by ``parse_group_specs``.
    """
    if group_cols is None:
        group_cols = {}
    elif isinstance(group_cols, str):
        group_cols = {group_cols: NO_GROUP_VALUES}
    elif not isinstance(group_cols, dict):
        group_cols = {c: NO_GROUP_VALUES for c in group_cols}
    missing = [c for c in group_cols if c not in obs.columns]
    if missing:
        raise ValueError(f"Group column(s) {missing} not in obs")
    keys = [_group_keys(obs[c], c, nulls) for c, nulls in group_cols.items()]

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            both = pd.concat([keys[i], keys[j]], axis=1).dropna().drop_duplicates()
            for a, b in both.itertuples(index=False):
                union(a, b)

    groups = pd.Series("row=" + obs.index.astype(str), index=obs.index, dtype=object)
    for key in keys:  # linked keys share a root, so the order does not matter
        has = key.notna().to_numpy()
        groups[has] = [find(k) for k in key[has]]
    return groups


def parse_fractions(specs: Iterable[str] | float | None, default_test: float = 0.2) -> dict[str, float]:
    """``["train=0.7", "val=0.1", "test=0.2"]`` -> ``{part: fraction}`` (must sum to 1).

    A single float is the test fraction of a train/test split.
    """
    if specs is None or isinstance(specs, (int, float)):
        test = default_test if specs is None else float(specs)
        fractions = {"train": 1.0 - test, "test": test}
    else:
        fractions = {}
        for spec in specs:
            name, sep, value = spec.partition("=")
            if not sep:
                raise ValueError(f"Fraction {spec!r} must look like PART=FRACTION")
            fractions[name.strip()] = float(value)
    if len(fractions) < 2 or any(not 0.0 < f < 1.0 for f in fractions.values()):
        raise ValueError(f"Need at least two parts, each with a fraction in (0, 1): {fractions}")
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1, got {sum(fractions.values()):.4f}: {fractions}")
    return fractions


def stratified_group_split(
    obs: pd.DataFrame,
    strata: Iterable[str] | None,
    fractions: dict[str, float] | float,
    seed: int = 0,
    group_cols: dict[str, frozenset[str]] | Iterable[str] | str | None = None,
) -> pd.Series:
    """Part name per row, stratified on ``strata`` and never splitting a group.

    ``fractions`` is ``{part: fraction}`` (or a float: the test fraction of a
    train/test split). Each group takes the majority stratum of its rows. Within a
    stratum, groups are shuffled and laid end to end; a group goes to the part whose
    share of the stratum's rows its midpoint falls in, so every part lands near its
    fraction of the rows. Each part gets at least one group when the stratum has
    enough of them. Strata with fewer groups than parts are pooled and split together,
    so they are not all forced into one part.
    """
    if not isinstance(fractions, dict):
        fractions = parse_fractions(float(fractions))
    parts = list(fractions)
    bounds = np.cumsum([fractions[p] for p in parts])
    labels = stratum_labels(obs, strata)
    groups = split_groups(obs, group_cols)

    frame = pd.DataFrame({"group": groups.to_numpy(), "stratum": labels.to_numpy()})
    group_stratum = frame.groupby("group")["stratum"].agg(lambda s: s.value_counts().index[0])
    group_size = frame.groupby("group").size()

    largest = int(group_size.max())
    if largest > 0.5 * len(obs):
        print(
            f"  [warn] one unit holds {largest} of {len(obs)} rows (groups linked through "
            f"{list(group_cols or [])}); the split cannot reach {fractions}"
        )
    n_groups = group_stratum.value_counts()
    rare = n_groups.index[n_groups < len(parts)]
    if len(rare):
        print(f"  {len(rare)} strata have fewer groups than parts; pooling them into one stratum")
        group_stratum = group_stratum.where(~group_stratum.isin(rare), "__rare__")

    rng = np.random.default_rng(seed)
    part_of: dict[str, str] = {}
    for stratum in sorted(group_stratum.unique()):
        members = group_stratum.index[group_stratum == stratum].to_numpy()
        members = members[rng.permutation(len(members))]
        sizes = group_size.loc[members].to_numpy()
        mids = (np.cumsum(sizes) - sizes / 2) / sizes.sum()
        chosen = np.minimum(np.searchsorted(bounds, mids, side="right"), len(parts) - 1)
        if len(members) >= len(parts):
            # A part left empty takes a group from the part holding the most groups.
            for k in range(len(parts)):
                if not (chosen == k).any():
                    donor = np.bincount(chosen, minlength=len(parts)).argmax()
                    chosen[np.flatnonzero(chosen == donor)[-1 if k > donor else 0]] = k
        part_of.update(zip(members, (parts[k] for k in chosen)))

    return pd.Series(groups.map(part_of).to_numpy(), index=obs.index, name="split")


def split_summary(assignment: pd.Series, labels: pd.Series, parts: list[str] | None = None) -> pd.DataFrame:
    """Rows per stratum in each part, plus each part's realized fraction."""
    parts = parts or sorted(assignment.dropna().unique())
    table = pd.crosstab(labels.rename("stratum"), assignment.rename("split"))
    for part in parts:
        if part not in table.columns:
            table[part] = 0
    table = table[parts]
    totals = table.sum(axis=1)
    for part in parts:
        table[f"{part}_frac"] = (table[part] / totals).round(3)
    return table
