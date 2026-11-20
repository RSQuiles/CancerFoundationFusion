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


STRATIFY_MODES = ("proportional", "balanced", "sqrt")


def allocate(
    sizes: pd.Series,
    n: int,
    min_per_stratum: int = 1,
    mode: str = "proportional",
    caps: dict | None = None,
) -> pd.Series:
    """Per-stratum sample sizes summing to ``min(n, total)`` (less if caps bind).

    ``mode`` decides how ``n`` is shared out:

    - ``proportional`` (default): in proportion to the stratum sizes, so the subsample
      keeps the data's composition;
    - ``balanced``: equal quotas, each capped at its stratum's size, with what small
      strata cannot take passed on to the others (water-filling);
    - ``sqrt``: in proportion to sqrt(size), a compromise between the two.

    Each stratum gets at least ``min(min_per_stratum, size, cap)`` (so rare strata stay
    visible) and never more than its size or its ``caps`` entry. Caps apply before the
    rest is handed out, so a capped stratum's unused share goes to the others. If the
    floors alone exceed ``n``, the floors win and the total exceeds ``n``; that is
    reported rather than silently dropping strata.
    """
    if mode not in STRATIFY_MODES:
        raise ValueError(f"Unknown stratify mode {mode!r}; use one of {STRATIFY_MODES}")
    sizes = sizes.astype("int64")
    limit = sizes.to_numpy().copy()
    if caps:
        for i, key in enumerate(sizes.index):
            if key in caps:
                limit[i] = min(limit[i], max(0, int(caps[key])))
    if n >= int(limit.sum()):
        return pd.Series(limit, index=sizes.index)

    floor = np.minimum(limit, max(0, int(min_per_stratum))).astype("int64")
    alloc = floor.copy()
    remaining = n - int(floor.sum())
    if remaining < 0:
        print(
            f"  [warn] {len(sizes)} strata x min_per_stratum={min_per_stratum} exceeds "
            f"n={n}; drawing {int(floor.sum())} rows instead"
        )
        return pd.Series(alloc, index=sizes.index)

    room = limit - alloc
    if mode == "proportional" and not caps:
        weights = None          # the original rule: shares proportional to the room left
    elif mode == "proportional":
        weights = sizes.to_numpy().astype(float)
    elif mode == "sqrt":
        weights = np.sqrt(sizes.to_numpy().astype(float))
    else:
        weights = np.ones(len(sizes), dtype=float)
    while remaining > 0 and room.sum() > 0:
        if weights is None:
            share = room / room.sum() * remaining
        else:
            w = np.where(room > 0, weights, 0.0)
            share = w / w.sum() * remaining
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
            if weights is None:
                break
            continue
        alloc += step
        room -= step
        remaining -= int(step.sum())
    return pd.Series(alloc, index=sizes.index)


UNKNOWN_STRATUM_LABELS: frozenset[str] = frozenset({"unknown", "nan", "none", "n/a", ""})


def unknown_caps(
    sizes: pd.Series,
    n: int,
    unknown_frac: float | None,
    unknown_labels: Iterable[str] = (),
) -> dict:
    """``{stratum: max rows}`` for strata with an unknown part (``unknown_frac * n`` in total).

    A joint stratum (``tissue | assay``) counts as unknown when any part is. The cap is
    shared by all unknown strata in proportion to their sizes. ``None`` = no cap.
    """
    if unknown_frac is None:
        return {}
    labels = UNKNOWN_STRATUM_LABELS | {fold_label(v) for v in unknown_labels}
    unknown = [k for k in sizes.index
               if any(fold_label(part) in labels for part in str(k).split(_SEP))]
    if not unknown:
        return {}
    budget = max(0, int(np.floor(float(unknown_frac) * n)))
    sub = sizes.loc[unknown].astype("int64")
    shares = allocate(sub, budget, min_per_stratum=0) if budget < int(sub.sum()) else sub
    return {k: int(v) for k, v in shares.items()}


def label_caps(sizes: pd.Series, n: int, caps: dict | None) -> dict:
    """``{stratum: max rows}`` for ``caps`` = ``{label: frac}``: a label's strata get ``frac * n``.

    A stratum carries a label when its label, or one part of a joint stratum
    (``tissue | assay``), equals it (case/space/hyphen-insensitive). The share is split
    among a label's strata in proportion to their sizes. Used for large categories that
    are not tissues (e.g. ``paired``, ``cell line``) so they cannot dominate a subsample.
    """
    out: dict = {}
    for label, frac in (caps or {}).items():
        target = fold_label(label)
        hit = [k for k in sizes.index if any(fold_label(part) == target for part in str(k).split(_SEP))]
        if not hit:
            continue
        budget = max(0, int(np.floor(float(frac) * n)))
        part = sizes.loc[hit].astype("int64")
        shares = allocate(part, budget, min_per_stratum=0) if budget < int(part.sum()) else part
        for k, v in shares.items():
            out[k] = min(out.get(k, int(v)), int(v))
    return out


def parse_label_caps(spec) -> dict:
    """``{"cell line": 0.05}`` or ``["cell line=0.05", "paired=0.05"]`` -> ``{label: frac}``."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        items = spec.items()
    else:
        items = []
        for entry in spec:
            label, sep, frac = str(entry).rpartition("=")
            if not sep or not label.strip():
                raise ValueError(f"Label cap {entry!r} must look like LABEL=FRACTION")
            items.append((label.strip(), frac))
    out = {}
    for label, frac in items:
        value = float(frac)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Label cap for {label!r} must be in [0, 1], got {value}")
        out[str(label)] = value
    return out


def _all_caps(sizes, n, unknown_frac, unknown_labels, caps) -> dict:
    merged = dict(unknown_caps(sizes, n, unknown_frac, unknown_labels))
    for k, v in label_caps(sizes, n, caps).items():
        merged[k] = min(merged.get(k, v), v)
    return merged


def stratum_quotas(
    labels: pd.Series,
    n: int,
    min_per_stratum: int = 1,
    mode: str = "proportional",
    unknown_frac: float | None = None,
    unknown_labels: Iterable[str] = (),
    caps: dict | None = None,
) -> pd.Series:
    """Rows per stratum for a subsample of ``n`` (``allocate`` with the unknown and label caps)."""
    sizes = labels.value_counts()
    return allocate(sizes, int(n), min_per_stratum, mode=mode,
                    caps=_all_caps(sizes, n, unknown_frac, unknown_labels, caps))


def stratified_subsample(
    obs: pd.DataFrame,
    n: int | None,
    strata: Iterable[str] | None = None,
    min_per_stratum: int = 1,
    seed: int = 0,
    mode: str = "proportional",
    unknown_frac: float | None = None,
    unknown_labels: Iterable[str] = (),
    keep: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    caps: dict | None = None,
) -> pd.Index:
    """Obs names of a stratified random subsample of ``n`` rows, in original order.

    ``n=None`` (or ``n >= len(obs)``) returns every row. Without strata it is a
    plain uniform subsample. ``mode`` and ``unknown_frac`` shape the quotas (see
    ``allocate``, ``unknown_caps``). ``keep`` names rows that are always taken (e.g. the
    linked rows from ``select_linked_rows``); they count toward their stratum's quota and
    the rest of each quota is drawn from the other rows. ``exclude`` names rows that are
    never drawn (unless also in ``keep``), e.g. the linked rows not selected, so a pair
    cannot enter one modality through the random fill; quotas are computed on the rows
    left. ``caps`` (``{label: frac}``, see ``label_caps``) bounds named categories. When
    the kept rows of a stratum exceed its quota they are all taken and the other strata
    are re-allocated over what is left, so the total stays ``n``. The defaults reproduce
    the original draw exactly.
    """
    if exclude is not None:
        out = obs.index.isin(list(exclude))
        if keep is not None:
            out &= ~obs.index.isin(list(keep))
        obs = obs[~out]
    if n is None or n >= len(obs):
        return obs.index.copy()
    labels = stratum_labels(obs, strata)
    quotas = stratum_quotas(labels, int(n), min_per_stratum, mode, unknown_frac, unknown_labels, caps)
    rng = np.random.default_rng(seed)
    positions = np.arange(len(obs))
    kept = obs.index.isin(list(keep)) if keep is not None else np.zeros(len(obs), dtype=bool)
    if kept.any():
        kept_counts = labels[kept].value_counts().reindex(quotas.index, fill_value=0)
        over = kept_counts > quotas
        if over.any():
            rest = labels[~labels.isin(quotas.index[over])]
            n_rest = max(0, int(n) - int(kept_counts[over].sum()))
            quotas = pd.concat([
                kept_counts[over],
                stratum_quotas(rest, n_rest, min_per_stratum, mode, unknown_frac, unknown_labels, caps)
                if len(rest) else pd.Series(dtype="int64"),
            ])
    chosen: list[np.ndarray] = [positions[kept]] if kept.any() else []
    for stratum, quota in quotas.items():
        in_stratum = (labels == stratum).to_numpy()
        quota = int(quota) - int((in_stratum & kept).sum())
        if quota <= 0:
            continue
        pool = positions[in_stratum & ~kept]
        chosen.append(rng.choice(pool, size=min(quota, len(pool)), replace=False))
    picked = np.unique(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
    return obs.index[picked]


def _hash_rank(values: Iterable[str], seed: int) -> list[str]:
    """``values`` in a seeded order that does not depend on which others are present."""
    import hashlib

    return sorted(values, key=lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())


def parse_link_spec(spec: str) -> tuple[str, list[str]]:
    """``"paired:0"`` -> ``("paired", ["0"])``: the column and its extra no-link values."""
    col, _, extra = spec.partition(":")
    return col.strip(), [v.strip() for v in extra.split(",") if v.strip()]


def row_link_keys(obs: pd.DataFrame, link_specs: Iterable[str]) -> pd.Series:
    """``<col>:<key>`` per row for the first ``link_specs`` column that links it, else NaN."""
    keys = pd.Series(np.nan, index=obs.index, dtype=object)
    for col, extra in (parse_link_spec(s) for s in link_specs):
        if col not in obs.columns:
            continue
        k = link_keys(obs[col], extra)
        fill = (col + ":" + k.astype(str)).where(k.notna())
        keys = keys.where(keys.notna(), fill)
    return keys


def select_linked_rows(
    pools: dict[str, pd.DataFrame],
    budgets: dict[str, int | None],
    link_specs: Iterable[str],
    strata: Iterable[str] | None = None,
    min_per_stratum: int = 1,
    mode: str = "proportional",
    unknown_frac: float | None = None,
    unknown_labels: Iterable[str] = (),
    linked_frac: float = 0.25,
    per_key: int | None = None,
    seed: int = 0,
    caps: dict | None = None,
    linked_quota: str = "budget",
) -> tuple[dict[str, pd.Index], dict]:
    """Rows to keep because they carry a link (a pair id), chosen jointly over ``pools``.

    ``pools`` maps a modality to its obs (the rows a subsample is drawn from) and
    ``budgets`` to its subsample size (None = every row). Link keys are visited in a
    seeded hash order, so every call sees them in the same order whatever the pools. A
    key is accepted only if, in **every** pool that has it, its rows (at most
    ``per_key``, picked in a seeded order) fit both

    - the pool's linked budget, ``linked_frac * budget`` rows in total, and
    - the quota of each stratum they fall in (``stratum_quotas`` with the same mode and
      unknown cap), counting the linked rows already accepted,

    so links never take more than a fraction of the sample nor more than a stratum's
    share, and a pair is kept in all modalities or in none. Returns the kept obs names
    per pool and a summary (keys available / accepted, rows per pool).

    ``linked_quota="budget"`` (default) drops the per-stratum condition: only the linked
    budget bounds the pairs. Their rows still count toward their stratum's quota in
    ``stratified_subsample``, so a stratum exceeds its quota only through pairs. This
    matters when the pairs make up a stratum of their own (single-cell ``paired``), whose
    balanced or capped quota would otherwise admit only a few pairs. ``"stratum"`` keeps
    the per-stratum condition.
    """
    if linked_quota not in ("budget", "stratum"):
        raise ValueError(f"linked_quota must be 'budget' or 'stratum', got {linked_quota!r}")
    link_specs = list(link_specs)
    state = {}
    all_keys: set[str] = set()
    for name, obs in pools.items():
        n = budgets.get(name)
        full = n is None or n >= len(obs)
        labels = stratum_labels(obs, strata)
        quotas = None if full else stratum_quotas(labels, int(n), min_per_stratum, mode,
                                                  unknown_frac, unknown_labels, caps)
        keys = row_link_keys(obs, link_specs).dropna()
        rows_by_key: dict[str, list[str]] = {}
        for key, idx in keys.groupby(keys).groups.items():
            names = _hash_rank([str(i) for i in idx], seed)
            rows_by_key[str(key)] = names[:per_key] if per_key else names
        all_keys |= set(rows_by_key)
        state[name] = {
            "full": full,
            "labels": labels.set_axis(obs.index.astype(str)),
            "quota": quotas,
            "used": pd.Series(0, index=quotas.index if quotas is not None else [], dtype="int64"),
            "budget": None if full else int(np.floor(linked_frac * int(n))),
            "total": 0,
            "rows_by_key": rows_by_key,
            "kept": [],
        }
    accepted = 0
    for key in _hash_rank(all_keys, seed):
        plan = {}
        fits = True
        for name, st in state.items():
            rows = st["rows_by_key"].get(key)
            if not rows:
                continue
            if st["full"]:
                plan[name] = (rows, None)
                continue
            if st["total"] + len(rows) > st["budget"]:
                fits = False
                break
            counts = st["labels"].loc[rows].value_counts()
            if linked_quota == "stratum":
                over = (st["used"].reindex(counts.index, fill_value=0) + counts
                        > st["quota"].reindex(counts.index, fill_value=0))
                if over.any():
                    fits = False
                    break
            plan[name] = (rows, counts)
        if not fits or not plan:
            continue
        accepted += 1
        for name, (rows, counts) in plan.items():
            st = state[name]
            st["kept"].extend(rows)
            st["total"] += len(rows)
            if counts is not None:
                st["used"] = st["used"].add(counts, fill_value=0).astype("int64")
    kept = {name: pd.Index(st["kept"]) for name, st in state.items()}
    info = {
        "link_specs": link_specs,
        "linked_quota": linked_quota,
        "keys_available": len(all_keys),
        "keys_accepted": accepted,
        "rows": {name: len(v) for name, v in kept.items()},
    }
    return kept, info


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


# Values of a link column (pseudobulk_id, paired) that mean "linked to nothing". 0 and -1
# are the fills the producers write for unlinked rows; a genuine id 0 is therefore lost
# (the Census pseudobulk ``pseudo_bulk:000000``), which is accepted.
NO_LINK_VALUES: frozenset[str] = frozenset({"", "0", "-1", "nan", "none", "na", "<na>", "unknown"})


def link_keys(series: pd.Series, extra_no_link: Iterable[str] = ()) -> pd.Series:
    """Link key per row as a string, NaN where the row is linked to nothing.

    Shared by the splits (inheritance, ``--stratify-links``) and the stores
    (``--pb-id-column``), so both agree on which rows carry a link. Numeric ids lose a
    trailing ``.0`` (an int column read back as float).
    """
    # Plain lower-casing, not fold_label: fold_label maps "-" to "_", so "-1" would never match.
    no_link = NO_LINK_VALUES | {str(v).strip().lower() for v in extra_no_link}
    values = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_null = series.isna().to_numpy() | values.str.lower().isin(no_link).to_numpy()
    return values.where(~is_null)


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


def class_aware_allocation(
    n_pb: int,
    n_bulk: int,
    n_shared: int,
    n_classes: int,
    min_count: int,
    pb_frac: float = 0.6,
    bulk_frac: float = 0.6,
) -> dict:
    """Slots of one class-aware batch: matched / free per domain and the tissues K.

    K is lowered from ``n_classes`` until every chosen tissue gets >= ``min_count``
    matched rows in each domain that has matched slots. Round-robin over too many
    tissues used to leave some with a single bulk row: they counted in neither
    contrastive nor CDD, and their pseudobulk slots were wasted. Shared by the
    sampler (``BulkSCSampler``) and ``data_preprocess/check_store_losses.py``.
    """
    m = max(1, int(min_count))
    n_pb_matched = int(round(n_pb * pb_frac))
    n_bulk_matched = int(round(n_bulk * bulk_frac))
    k_requested = max(0, min(int(n_classes), int(n_shared)))
    k = k_requested
    for n in (n_pb_matched, n_bulk_matched):
        if n > 0:
            k = min(k, n // m)
    k = max(1, k) if k_requested > 0 else 0

    def spread(n):   # rows per chosen tissue, as round-robin over k gives them
        return (n // k, -(-n // k)) if k else (0, 0)

    return {
        "k_requested": k_requested,
        "k": k,
        "n_pb_matched": n_pb_matched,
        "n_pb_free": n_pb - n_pb_matched,
        "n_bulk_matched": n_bulk_matched,
        "n_bulk_free": n_bulk - n_bulk_matched,
        "pb_per_tissue": spread(n_pb_matched),
        "bulk_per_tissue": spread(n_bulk_matched),
    }
