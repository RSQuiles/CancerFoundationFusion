"""Add back the obs columns a joint-panel embedding file lost, without re-embedding.

Before the fix in ``embed_splits.py``, a call embedding several modalities together wrote
each file from the pooled obs, which keeps only the columns EVERY modality has (an inner
``ad.concat``). So ``pseudobulk_id`` (not in bulk) was missing from the sc and pseudobulk
files, and the integration ``pb_cells`` edge and ``plot_alignment --cell-means`` had
nothing to link on.

For each ``<modality>_<part>.h5ad`` of a model (or one variant), this reads the split's
source h5ad (``split.json``, the file ``embed_splits`` read, recorded per file in
``embed.json``), takes the wanted source obs columns the file lacks for the file's rows
(matched by obs name), and rewrites the file (obs + obsm only; small) through a temporary
file. Wanted = ``DEFAULT_COLUMNS`` (the link keys and the labels the evaluation reads),
``--columns`` to name others, ``--all-columns`` for every source column.
Existing columns and the embedding are untouched; ``embed.json`` records ``obs_restored``.

    python -m embed.restore_obs --model unified_baseline pca_leiden_s1 --dry-run
    python -m embed.restore_obs --model unified_baseline pca_leiden_s1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.layout import ExperimentLayout, canonical_modality  # noqa: E402

EMBED_CARD = "embed.json"
# What the evaluation reads from an embedding's obs, beyond what every modality has:
# the link keys (integration, plot_alignment --cell-means / --connect-pairs) and the
# label and filter columns of the figures and scIB. A column the source lacks is skipped.
DEFAULT_COLUMNS = (
    "pseudobulk_id", "paired",                                          # link keys
    "tissue_general", "tissue_label_source", "tissue_noncurated",       # tissue labels + curation
    "cell_type", "assay", "dataset_id", "donor_id",                     # figure / scIB labels
)
# A name made unique by embed_splits (obs_names_make_unique) is not in the source; a few
# are tolerated (left NaN), more mean the file does not belong to this source.
MAX_UNMATCHED = 0.001


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restore the obs columns joint-panel embedding files lost.")
    p.add_argument("--root", type=Path, default=None, help="Experiments root")
    p.add_argument("--model", nargs="+", required=True, help="Model names (embeddings/<model>/)")
    p.add_argument("--variant", default=None, help="Embedding variant (default: the joint-panel files)")
    cols = p.add_mutually_exclusive_group()
    cols.add_argument("--columns", nargs="+", default=None,
                      help=f"Source columns to add when missing (default: {' '.join(DEFAULT_COLUMNS)})")
    cols.add_argument("--all-columns", action="store_true",
                      help="Add every source obs column the file lacks")
    p.add_argument("--dry-run", action="store_true", help="Report what would be added; write nothing")
    return p


def _source_obs(layout: ExperimentLayout, modality: str, split: str, cache: dict):
    import anndata as ad

    key = (modality, split)
    if key not in cache:
        src = layout.split_source(modality, split)
        backed = ad.read_h5ad(src, backed="r")
        try:
            cache[key] = (src, backed.obs.copy())
        finally:
            backed.file.close()
    return cache[key]


_UNIQUE_SUFFIX = re.compile(r"-\d+$")


def source_names(names, source_index) -> tuple[pd.Index, int]:
    """The source obs name of each row: the name itself, else the name without the ``-<n>``
    suffix anndata's ``obs_names_make_unique`` added when ``embed_splits`` pooled
    modalities sharing a name (e.g. a pair's bulk and pseudobulk). Returns the names and
    how many needed the stripped form."""
    names = pd.Index(names).astype(str)
    found = names.isin(source_index)
    stripped = pd.Index([_UNIQUE_SUFFIX.sub("", n) for n in names])
    use = ~found & stripped.isin(source_index)
    return pd.Index(np.where(use, stripped, names)), int(use.sum())


def restore_file(path: Path, entry: dict, layout: ExperimentLayout, columns: list[str] | None,
                 dry_run: bool, cache: dict) -> list[str]:
    """Add the missing source columns to one embedding file; returns the columns added."""
    import anndata as ad

    split, modality = entry.get("split"), entry.get("modality")
    if not split or not modality:
        print(f"  [skip] {path.name}: embed.json has no split/modality (not made by embed_splits)")
        return []
    src, src_obs = _source_obs(layout, canonical_modality(modality), split, cache)
    adata = ad.read_h5ad(path)
    wanted = [c for c in (columns if columns is not None else src_obs.columns)
              if c in src_obs.columns and c not in adata.obs.columns]
    if not wanted:
        print(f"  {path.name}: nothing missing")
        return []

    keys, n_suffixed = source_names(adata.obs_names, src_obs.index)
    pos = src_obs.index.get_indexer(keys)
    unmatched = int((pos < 0).sum())
    rate = unmatched / max(1, adata.n_obs)
    print(f"  {path.name}: add {wanted} from {src} "
          f"({adata.n_obs - unmatched}/{adata.n_obs} rows matched"
          + (f", {n_suffixed} of them by their name without the make-unique suffix" if n_suffixed else "")
          + ")")
    if rate > MAX_UNMATCHED:
        raise SystemExit(f"Error: {path.name}: {unmatched} of {adata.n_obs} rows are not in {src}, "
                         f"e.g. {list(adata.obs_names[pos < 0][:5])}; "
                         "is this the source the file was embedded from?")
    if dry_run:
        return wanted

    for col in wanted:  # source dtype kept (categoricals stay categorical); unmatched rows NaN
        adata.obs[col] = src_obs[col].reindex(keys).values
    tmp = path.with_suffix(".restore_tmp.h5ad")
    adata.write_h5ad(tmp)
    os.replace(tmp, path)
    return wanted


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = ExperimentLayout.from_root(args.root)
    columns = None if args.all_columns else list(args.columns or DEFAULT_COLUMNS)
    print(f"columns to add when missing: {'every source column' if columns is None else columns}")
    cache: dict = {}
    for model in args.model:
        out_dir = layout.embedding_dir(model, args.variant)
        card_path = out_dir / EMBED_CARD
        if not card_path.exists():
            print(f"[{model}] no {card_path}; skipped")
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        print(f"[{model}]{' (' + args.variant + ')' if args.variant else ''}")
        changed = False
        for name, entry in sorted(card.get("files", {}).items()):
            path = out_dir / name
            if not path.exists():
                continue
            added = restore_file(path, entry, layout, columns, args.dry_run, cache)
            if added and not args.dry_run:
                entry["obs_restored"] = sorted(set(entry.get("obs_restored", [])) | set(added))
                entry["obs_restored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                changed = True
        if changed:
            card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
