"""Embed the test rows and/or a train subset of each modality's split for one model.

For every requested modality, reads ``data/<modality>/<modality>.h5ad``, takes the rows
of the requested ``--parts`` of ``--split`` (``test``, ``train_subset`` or both; default
``test`` only), optionally subsampled with stratification (``--test-subset``,
``--train-subset``, ``--stratify-by``), and writes them to ``embeddings/<model>/`` with
the machinery in ``embed.py``::

    embeddings/<model>/{sc,bulk,pseudobulk}_{test,train_subset}.h5ad   obs + obsm["X_emb"], no X
    embeddings/<model>/embed.json                                     provenance per file
    embeddings/<model>/panel_<hash>.txt                               gene panel used

All requested modalities and parts are embedded **together, through one gene
panel**, so an alignment UMAP never compares embeddings fitted on different inputs.

CancerFoundation model (checkpoint from ``checkpoints/<model>/``, newest ``*.ckpt``
unless ``--ckpt`` names one):

    python embed_splits.py --model unified_paired_tissue_s0 --split tissue_s0 \\
        --modalities sc bulk pseudobulk --parts test --test-subset 20000 --stratify-by tissue_general
    python embed_splits.py --model unified_paired_tissue_s0 --split tissue_s0 \\
        --parts test train_subset --train-subset 5000

Any embedder registered in ``embed_any.py`` (written as ``embeddings/<name>/``, ``--name``
defaulting to the embedder's name). It is fitted on the train-subset rows, so request
``train_subset`` too; with ``test`` only it is fitted on the test rows, with a warning:

    python embed_splits.py --embedder pca --split tissue_s0 --parts test train_subset --option n_pcs=50
    python embed_splits.py --embedder bulkformer --name bulkformer_v1 --split tissue_s0 \\
        --modalities bulk --option repo=/path/to/bulkformer-latest --option gene_column=gene_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed.embed import (  # noqa: E402
    MODALITY_COL,
    add_model_run_args,
    add_normalization_args,
    embed_adata,
    load_model,
    set_card_fields,
    write_embeddings,
)
from experiments.layout import EMBEDDING_PARTS, ExperimentLayout, canonical_modality  # noqa: E402
from utils.sampling import stratified_subsample  # noqa: E402

PART_COL = "_embed_part"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the test split and a stratified train subset of each modality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--model", default=None, help="CancerFoundation model name (checkpoints/<model>/).")
    backend.add_argument("--embedder", default=None, help="Embedder registered in embed_any.py (e.g. pca).")
    parser.add_argument("--name", default=None,
                        help="Output model name with --embedder (default: the embedder's name).")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Keyword argument for --embedder (repeatable).")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint file in checkpoints/<model>/ or a full path (default: newest *.ckpt).")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root.")
    parser.add_argument("--split", required=True, help="Split name (data/<modality>/splits/<split>/).")
    parser.add_argument("--modalities", nargs="+", default=["sc", "bulk", "pseudobulk"],
                        help="Modalities to embed.")
    parser.add_argument("--parts", nargs="+", default=["test"], choices=list(EMBEDDING_PARTS),
                        help="What to embed per modality: test, train_subset, or both.")
    parser.add_argument("--test-subset", "--max-test", dest="test_subset", type=int, default=None,
                        help="Test rows per modality, stratified (default: all).")
    parser.add_argument("--train-subset", type=int, default=5000,
                        help="Train rows per modality, stratified (0 or less = all); used with --parts train_subset.")
    parser.add_argument("--stratify-by", "--train-stratify-by", dest="stratify_by", nargs="*",
                        default=["tissue_general"],
                        help="obs columns stratifying both subsets (missing ones are skipped).")
    add_normalization_args(parser)
    add_model_run_args(parser)
    args = parser.parse_args()
    if args.embedder is None and (args.name or args.option):
        parser.error("--name and --option only apply with --embedder")
    return args


def resolve_checkpoint(layout: ExperimentLayout, model: str, ckpt: str | None) -> Path:
    if ckpt is None:
        path = layout.latest_checkpoint(model)
        if path is None:
            sys.exit(f"Error: no *.ckpt in {layout.checkpoint_dir(model)}")
        return path
    path = Path(ckpt)
    if not path.is_absolute() and not path.exists():
        path = layout.checkpoint_dir(model) / ckpt
    if not path.exists():
        sys.exit(f"Error: checkpoint not found: {path}")
    return path


def _size(n: int | None) -> int | None:
    """A subset size; None, 0 or less mean all rows."""
    return n if n is not None and n > 0 else None


def collect_rows(layout: ExperimentLayout, args, label: str):
    """The rows of one modality to embed, in memory, tagged with modality and part."""
    import anndata as ad

    raw = layout.raw_h5ad(label)
    if not raw.exists():
        sys.exit(f"Error: {raw} not found")
    lists = {}
    split_parts = {"test": "test", "train_subset": "train"}
    for part in sorted({split_parts[p] for p in args.parts}):
        path = layout.split_index_file(label, args.split, part)
        if not path.exists():
            sys.exit(f"Error: split '{args.split}' has no {path.name} for {label} ({path.parent})")
        lists[part] = set(path.read_text(encoding="utf-8").split())

    backed = ad.read_h5ad(raw, backed="r")
    try:
        obs = backed.obs
        strata = [c for c in args.stratify_by if c in obs.columns]
        skipped = sorted(set(args.stratify_by) - set(strata))
        if skipped:
            print(f"  [{label}] stratify columns not in obs, skipped: {skipped}")

        selected = {}
        if "test" in args.parts:
            test_obs = obs[obs.index.isin(lists["test"])]
            selected["test"] = stratified_subsample(test_obs, _size(args.test_subset), strata, seed=args.seed)
        if "train_subset" in args.parts:
            train_obs = obs[obs.index.isin(lists["train"])]
            selected["train_subset"] = stratified_subsample(train_obs, _size(args.train_subset), strata,
                                                            seed=args.seed)

        pieces = []
        for part, names in selected.items():
            if len(names) == 0:
                print(f"  [{label}] no {part} rows; skipped")
                continue
            piece = backed[obs.index.isin(names)].to_memory()
            piece.obs[MODALITY_COL] = label
            piece.obs[PART_COL] = part
            pieces.append(piece)
            print(f"  [{label}] {part}: {piece.n_obs} rows")
    finally:
        backed.file.close()
    return pieces


def pool(pieces):
    """One AnnData over every modality and part, on the genes they all share."""
    import anndata as ad

    adata = ad.concat(pieces, join="inner", merge="same")
    if not adata.obs_names.is_unique:
        print("  [warn] obs names repeat across modalities; suffixing the duplicates")
        adata.obs_names_make_unique()
    dropped = max(p.n_vars for p in pieces) - adata.n_vars
    print(f"{adata.n_obs} rows x {adata.n_vars} genes shared by all modalities"
          + (f" ({dropped} genes not shared were dropped)" if dropped else ""))
    return adata


def embed_with_model(args, layout: ExperimentLayout, adata):
    checkpoint = resolve_checkpoint(layout, args.model, args.ckpt)
    model = load_model(checkpoint, args.cpu)
    emb, genes = embed_adata(
        model, adata,
        modality_col=MODALITY_COL,
        batch_size=args.batch_size, normalize=args.normalize,
        shared_panel=args.shared_panel, chunk_rows=args.chunk_rows,
        panel_sample=args.panel_sample, seed=args.seed,
    )
    provenance = {"method": "cancerfoundation", "checkpoint": str(checkpoint),
                  "normalize": args.normalize}
    return emb, genes, provenance


def embed_with_embedder(args, adata):
    from embed.embed_any import build_embedder, parse_options, run_embedder

    embedder = build_embedder(args.embedder, parse_options(args.option))
    # Fit on train rows only, so nothing about the test rows leaks into the embedding.
    fit_rows = (adata.obs[PART_COL] == "train_subset").to_numpy()
    fitted_on = "train_subset"
    if not fit_rows.any():
        print("  [warn] no train-subset rows: fitting the embedder on the test rows")
        fit_rows[:] = True
        fitted_on = "test"
    emb = run_embedder(embedder, adata, adata[fit_rows].copy())
    return emb, embedder.genes, {**embedder.provenance(), "fitted_on": fitted_on}


def main() -> None:
    args = parse_args()
    layout = ExperimentLayout.from_root(args.root)
    model_name = args.model or args.name or args.embedder
    labels = list(dict.fromkeys(canonical_modality(m) for m in args.modalities))

    print(f"Collecting rows for {labels} from split '{args.split}'")
    pieces = [p for label in labels for p in collect_rows(layout, args, label)]
    if not pieces:
        sys.exit("Error: nothing to embed")
    adata = pool(pieces)

    if args.model:
        emb, genes, provenance = embed_with_model(args, layout, adata)
    else:
        emb, genes, provenance = embed_with_embedder(args, adata)

    provenance.update({
        "split": args.split,
        "modalities_embedded_together": labels,
        "shared_panel": bool(args.shared_panel or len(labels) == 1),
        "seed": args.seed,
        "parts": args.parts,
        "test_subset": _size(args.test_subset),
        "train_subset": _size(args.train_subset) if "train_subset" in args.parts else None,
        "stratify_by": args.stratify_by,
    })

    out_dir = layout.embedding_dir(model_name)
    print(f"Writing embeddings for '{model_name}' to {out_dir}")
    keys = adata.obs[MODALITY_COL].astype(str) + "|" + adata.obs[PART_COL].astype(str)
    for key in keys.unique():
        label, part = key.split("|")
        mask = (keys == key).to_numpy()
        obs = adata.obs.loc[mask].drop(columns=[PART_COL])
        name = layout.embedding_file(model_name, label, part).stem
        write_embeddings(out_dir, name, obs, emb[mask], genes,
                         provenance={**provenance, "modality": label, "part": part})
    set_card_fields(out_dir, model=model_name)
    print("Done.")


if __name__ == "__main__":
    main()
