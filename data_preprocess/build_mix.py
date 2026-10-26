"""Build one training store from parts of several splits, e.g. for --unified runs.

Unified training reads ONE store whose obs carries a ``modality`` column, so the
chosen part of each source split goes into the same store. Rows are read straight
from the source h5ads: nothing is staged or copied besides the one transient file per
source that SCDL needs (see ``bulk_sc_data_preprocessing.build_pipeline_ready``).

Sources, repeatable, in either form:

- ``MODALITY:SPLIT[:PART]`` under the experiments layout (part defaults to ``train``);
- ``PATH.h5ad=SPLIT_DIR[:PART]`` for any h5ad and split directory.

    python build_mix.py --name sc_bulk_pb_tissue_s0 \\
        --source single_cell:tissue_s0 bulk:tissue_s0 pseudobulk:tissue_s0 \\
        --obs-columns tissue_general assay paired --pb-id-column pseudobulk_id

The store goes to ``data/pipeline_ready/<name>/`` (or ``--out-dir``); a launch config
then points at it with ``"train_path": {"mix": "<name>"}``. The genes are the
intersection across sources unless ``--vocab-mode union`` or ``--vocab-path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import shutil
from argparse import ArgumentParser
from datetime import datetime, timezone

from experiments.layout import ExperimentLayout, canonical_modality
from split_train_test import read_parts


def resolve_source(spec: str, layout: ExperimentLayout) -> dict:
    """``{path, split_dir, part, modality}`` for one ``--source`` entry."""
    if "=" in spec:
        path, _, rest = spec.partition("=")
        split_dir, _, part = rest.partition(":")
        return {"path": Path(path), "split_dir": Path(split_dir), "part": part or "train", "modality": None}
    fields = spec.split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"--source {spec!r} must be MODALITY:SPLIT[:PART] or PATH.h5ad=SPLIT_DIR[:PART]")
    label = canonical_modality(fields[0])
    return {"path": layout.raw_h5ad(label), "split_dir": layout.split_dir(label, fields[1]),
            "part": fields[2] if len(fields) == 3 else "train", "modality": label}


def main(args):
    from bulk_sc_data_preprocessing import Source, build_pipeline_ready

    layout = ExperimentLayout.from_root(args.root)
    store = Path(args.out_dir) if args.out_dir else layout.mix_dir(args.name)
    if store.exists() and any(store.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{store} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(store)

    resolved, sources = [], []
    for spec in args.source:
        src = resolve_source(spec, layout)
        parts = read_parts(src["split_dir"])
        if src["part"] not in parts:
            raise ValueError(f"{src['split_dir']} has no part {src['part']!r}; it has {list(parts)}")
        rows = parts[src["part"]]
        print(f"  {src['path'].name}: {len(rows)} rows from {src['split_dir'].name}/{src['part']}")
        sources.append(Source(path=src["path"], rows=rows, modality=src["modality"]))
        resolved.append({**{k: str(v) if v is not None else None for k, v in src.items()}, "n_rows": len(rows)})
    paths = [s.path for s in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("The same h5ad appears in several --source entries; give it once")

    summary = build_pipeline_ready(
        sources,
        store,
        obs_columns=args.obs_columns,
        vocab_path=args.vocab_path,
        vocab_mode=args.vocab_mode,
        gene_column=args.gene_column,
        pb_id_column=args.pb_id_column,
    )
    (store / "mix.json").write_text(json.dumps({
        "name": args.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": resolved,
        **summary,
    }, indent=2), encoding="utf-8")
    print("Done.")


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="Mix name; becomes data/pipeline_ready/<name>/")
    parser.add_argument("--source", nargs="+", required=True,
                        metavar="MODALITY:SPLIT[:PART] | PATH.h5ad=SPLIT_DIR[:PART]",
                        help="Split parts to combine (default part: train)")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--out-dir", type=Path, default=None, help="Store directory instead of the layout's")
    parser.add_argument("--obs-columns", nargs="+", default=None,
                        help="obs columns kept in the store (modality is always kept)")
    parser.add_argument("--vocab-path", type=Path, default=None, help="Existing vocab.json")
    parser.add_argument("--vocab-mode", choices=["intersection", "union"], default="intersection",
                        help="Genes in the new vocabulary when --vocab-path is not given")
    parser.add_argument("--gene-column", default=None, help="var column holding the gene names")
    parser.add_argument("--pb-id-column", default=None,
                        help="Keep this pseudobulk -> cell link column in the store")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
