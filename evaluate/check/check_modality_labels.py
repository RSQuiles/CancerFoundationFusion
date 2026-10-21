"""Self-check for the ``_eval_modality`` vocabulary in ``evaluate/utils.py``.

Runs anywhere — no cluster, GPU, checkpoint or model, and not even anndata: the
rewrite only ever touches ``adata.obs``, so a stand-in with an obs DataFrame is a
faithful subject. Real AnnData objects are used when anndata happens to be installed.

    python evaluate/check/check_modality_labels.py

The rule under test is the one that made an eval.h5ad report
``Modality rows available: {'subsampled': 5000, ...}`` and then skip ``agg_synth_*``
with "missing 'sc' rows" in the same log: eval.h5ad files built before June 2026
wrote the *filename prefix* into the obs column, while every consumer matches on the
*label*. ``canonicalize_modality_column`` translates on read, so the two vocabularies
can never drift apart again without this check failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:                      # plain env: exercise the same code on a stub
    ad = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.utils import (
    CANONICAL_MODALITIES,
    MODALITY_ALIASES,
    MODALITY_COL,
    MODALITY_FILE_PREFIXES,
    MOD_BULK,
    MOD_PAIRED_BULK,
    MOD_PAIRED_PB,
    MOD_PAIRED_SC,
    MOD_PB,
    MOD_SC,
    MOD_SYNTH_PB,
    canonical_modality,
    canonicalize_modality_column,
)

FAILED: list[str] = []


def check(name: str, condition: bool) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + name)
    if not condition:
        FAILED.append(name)


class _CaptureLog:
    """Stand-in logger: canonicalize_modality_column takes one or falls back to print."""

    def __init__(self) -> None:
        self.info_msgs: list[str] = []
        self.warn_msgs: list[str] = []

    def info(self, msg: str) -> None:
        self.info_msgs.append(str(msg))

    def warning(self, msg: str) -> None:
        self.warn_msgs.append(str(msg))


class _FakeAnnData:
    """Everything canonicalize_modality_column can see of an AnnData."""

    def __init__(self, obs: pd.DataFrame) -> None:
        self.obs = obs


def _adata(labels: list[str] | None):
    """Minimal eval.h5ad stand-in: only the modality column matters here.

    ``labels=None`` builds one with no modality column at all.
    """
    if labels is None:
        obs = pd.DataFrame(index=["c0", "c1"])
        n = 2
    else:
        n = len(labels)
        obs = pd.DataFrame({MODALITY_COL: labels}, index=[f"c{i}" for i in range(n)])
    if ad is None:
        return _FakeAnnData(obs)
    return ad.AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs)


def check_prefix_table() -> None:
    print("\n-- filename prefixes -> labels --")
    labels = {label for _, label in MODALITY_FILE_PREFIXES}
    check(
        "every prefix maps to a canonical label",
        labels <= set(CANONICAL_MODALITIES),
    )
    check(
        "the SC filename conventions are all covered",
        {p for p, lab in MODALITY_FILE_PREFIXES if lab == MOD_SC}
        >= {"subsampled", "partition", "sc", "pretraining_sc"},
    )
    check(
        "no prefix is listed twice",
        len({p for p, _ in MODALITY_FILE_PREFIXES}) == len(MODALITY_FILE_PREFIXES),
    )
    # A prefix ordered before one it is a prefix *of* would swallow those files.
    bad = [
        (a, b)
        for i, (a, _) in enumerate(MODALITY_FILE_PREFIXES)
        for b, _ in MODALITY_FILE_PREFIXES[i + 1:]
        if b.startswith(a)
    ]
    check(f"no prefix shadows a later one {bad or ''}".strip(), not bad)
    check(
        "every prefix is also an alias, so a legacy file reads back",
        all(canonical_modality(p) == lab for p, lab in MODALITY_FILE_PREFIXES),
    )


def check_canonical_modality() -> None:
    print("\n-- canonical_modality --")
    check("the reported bug: 'subsampled' -> 'sc'", canonical_modality("subsampled") == MOD_SC)
    check("'partition' -> 'sc'", canonical_modality("partition") == MOD_SC)
    check("'pretraining_sc' -> 'sc'", canonical_modality("pretraining_sc") == MOD_SC)
    check("'single_cell' -> 'sc'", canonical_modality("single_cell") == MOD_SC)
    check("'single-cell' -> 'sc' (dash folded)", canonical_modality("single-cell") == MOD_SC)
    check("'Subsampled ' -> 'sc' (case and space)", canonical_modality("Subsampled ") == MOD_SC)
    check("'pretraining_bulk' -> 'bulk'", canonical_modality("pretraining_bulk") == MOD_BULK)
    check("'pseudo_bulk' -> 'pseudobulk'", canonical_modality("pseudo_bulk") == MOD_PB)
    check("'synthetic_pb' -> 'synth_pb'", canonical_modality("synthetic_pb") == MOD_SYNTH_PB)

    check(
        "canonical labels pass through unchanged",
        all(canonical_modality(m) == m for m in CANONICAL_MODALITIES),
    )
    check(
        "aliases never point outside the canonical vocabulary",
        set(MODALITY_ALIASES.values()) <= set(CANONICAL_MODALITIES),
    )
    check("an unknown label is left alone", canonical_modality("xenograft") == "xenograft")
    check("no alias rewrites a canonical label", all(
        canonical_modality(m) == m for m in CANONICAL_MODALITIES
    ))


def check_column_rewrite() -> None:
    print("\n-- canonicalize_modality_column --")
    # Exactly the vocabulary of the eval.h5ad that triggered this: SC rows labelled
    # with the prefix, everything else already canonical.
    adata = _adata(
        ["subsampled"] * 5 + ["bulk"] * 3 + ["paired_sc"] * 2
        + ["paired_pb"] * 2 + ["paired_bulk"] * 2 + ["synth_pb"] * 4
    )
    logger = _CaptureLog()
    renames = canonicalize_modality_column(adata, log=logger)

    check("only the legacy label is rewritten", renames == {"subsampled": MOD_SC})
    labels = adata.obs[MODALITY_COL].astype(str)
    check("the SC rows are now 'sc'", int((labels == MOD_SC).sum()) == 5)
    check("no 'subsampled' rows remain", not (labels == "subsampled").any())
    check("the other modalities are untouched", (
        int((labels == MOD_BULK).sum()) == 3
        and int((labels == MOD_PAIRED_SC).sum()) == 2
        and int((labels == MOD_PAIRED_PB).sum()) == 2
        and int((labels == MOD_PAIRED_BULK).sum()) == 2
        and int((labels == MOD_SYNTH_PB).sum()) == 4
    ))
    check("the rewrite is reported", any("subsampled" in m and "sc" in m
                                         for m in logger.info_msgs))
    check("and nothing is warned about", not logger.warn_msgs)

    # This is literally unified_metrics._get(MOD_SC)'s predicate, and the condition
    # has_precomputed_agg / agg_synth_* hangs off.
    mask = (adata.obs.get(MODALITY_COL, pd.Series(dtype=str)) == MOD_SC).values
    check("unified_metrics._get(MOD_SC) now finds rows", bool(mask.any()))

    second = canonicalize_modality_column(adata, log=_CaptureLog())
    check("idempotent: a second pass rewrites nothing", second == {})


def check_already_canonical_and_edges() -> None:
    print("\n-- no-ops and edge cases --")
    adata = _adata([MOD_SC] * 2 + [MOD_BULK] * 2)
    logger = _CaptureLog()
    check("a canonical file is left alone",
          canonicalize_modality_column(adata, log=logger) == {})
    check("silently", not logger.info_msgs and not logger.warn_msgs)

    check("a missing modality column is not an error",
          canonicalize_modality_column(_adata(None)) == {})

    unknown = _adata(["subsampled", "xenograft"])
    logger = _CaptureLog()
    renames = canonicalize_modality_column(unknown, log=logger)
    labels = unknown.obs[MODALITY_COL].astype(str).tolist()
    check("an unknown label survives the rewrite", labels == [MOD_SC, "xenograft"])
    check("and is warned about", any("xenograft" in m for m in logger.warn_msgs))
    check("while the known one is still fixed", renames == {"subsampled": MOD_SC})

    # Categorical dtype is what h5ad round-trips give back for a string obs column.
    cat = _adata(["subsampled"] * 3 + ["bulk"])
    cat.obs[MODALITY_COL] = cat.obs[MODALITY_COL].astype("category")
    canonicalize_modality_column(cat, log=_CaptureLog())
    check("a categorical column is rewritten too",
          int((cat.obs[MODALITY_COL].astype(str) == MOD_SC).sum()) == 3)


def main() -> None:
    check_prefix_table()
    check_canonical_modality()
    check_column_rewrite()
    check_already_canonical_and_edges()

    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
