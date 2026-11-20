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
    add_input_mode_args,
    add_model_run_args,
    add_normalization_args,
    embed_adata,
    input_mode_provenance,
    load_model,
    set_card_fields,
    write_embeddings,
)
from experiments.layout import EMBEDDING_PARTS, ExperimentLayout, canonical_modality  # noqa: E402
from utils.sampling import (  # noqa: E402
    STRATIFY_MODES, parse_label_caps, row_link_keys, select_linked_rows, stratified_subsample,
)

PART_COL = "_embed_part"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed the test split and a stratified train subset of each modality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Not required=True: a --config default does not satisfy argparse's own check, so
    # parse_args() checks that one of the two is given.
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--model", default=None, help="CancerFoundation model name (checkpoints/<model>/).")
    backend.add_argument("--embedder", default=None, help="Embedder registered in embed_any.py (e.g. pca).")
    parser.add_argument("--name", default=None,
                        help="Output model name with --embedder (default: the embedder's name).")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Keyword argument for --embedder (repeatable).")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint file in checkpoints/<model>/ or a full path (default: newest *.ckpt).")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root.")
    parser.add_argument("--variant", default=None,
                        help="Write to embeddings/<model>/<variant>/ instead of embeddings/<model>/, "
                             "e.g. own_panel for one call per modality (each with its own gene panel) "
                             "next to the joint-panel files.")
    parser.add_argument("--split", default=None, help="Required. Split name (data/<modality>/splits/<split>/).")
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
    parser.add_argument("--stratify-mode", default="proportional", choices=list(STRATIFY_MODES),
                        help="How a subset is shared among strata: proportional to their sizes, "
                             "balanced (equal per stratum, capped at its size) or sqrt.")
    parser.add_argument("--unknown-frac", type=float, default=None,
                        help="At most this share of a subset for unknown strata (default: no cap; "
                             "0 leaves them out). Their unused share goes to the other strata.")
    parser.add_argument("--unknown-labels", nargs="*", default=[],
                        help="Extra labels that count as unknown (unknown, nan, none, n/a and empty always do).")
    parser.add_argument("--cap-labels", nargs="*", default=[], metavar="LABEL=FRACTION",
                        help="At most this share of the subset for strata carrying LABEL (e.g. "
                             "paired=0.05 \"cell line=0.05\"); YAML may give a mapping. The freed share "
                             "goes to the other strata.")
    parser.add_argument("--keep-linked", nargs="*", default=[], metavar="COL[:NOLINK,...]",
                        help="Test part: keep rows linked by these columns (e.g. paired:0), the same "
                             "keys in every modality, within --linked-frac and each stratum's quota.")
    parser.add_argument("--linked-frac", type=float, default=0.25,
                        help="With --keep-linked: at most this share of a modality's test subset is linked rows.")
    parser.add_argument("--linked-per-key", type=int, default=None,
                        help="With --keep-linked: at most this many rows per link key (e.g. cells per pair).")
    parser.add_argument("--linked-quota", default="budget", choices=["budget", "stratum"],
                        help="With --keep-linked: budget = pairs bounded only by --linked-frac (their rows "
                             "still count toward their stratum, which may then exceed its quota); stratum = "
                             "also within each stratum's quota (fewer pairs when they form a stratum, "
                             "e.g. single-cell 'paired').")
    parser.add_argument("--link-modalities", nargs="+", default=["sc", "bulk", "pseudobulk"],
                        help="With --keep-linked: modalities whose test parts decide the kept keys "
                             "(the same in every call, also one-modality own_panel calls).")
    add_normalization_args(parser)
    add_input_mode_args(parser)
    add_model_run_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags, with defaults from a ``--config`` YAML (top level + ``embed_splits:`` block)."""
    from evaluate._cli import parse_with_config

    parser = build_parser()
    args = parse_with_config(parser, "embed_splits", argv)
    if not (args.model or args.embedder):
        parser.error("one of the arguments --model --embedder is required")
    if args.model and args.embedder:
        parser.error("--model and --embedder are mutually exclusive")
    if not args.split:
        parser.error("--split is required")
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


def _test_obs(layout: ExperimentLayout, args, label: str):
    """obs of one modality's test part (None when the split has no test list for it)."""
    import anndata as ad

    raw = layout.split_source(label, args.split)
    path = layout.split_index_file(label, args.split, "test")
    if not raw.exists() or not path.exists():
        return None
    names = set(path.read_text(encoding="utf-8").split())
    backed = ad.read_h5ad(raw, backed="r")
    try:
        obs = backed.obs
        return obs[obs.index.isin(names)].copy()
    finally:
        backed.file.close()


def linked_test_rows(layout: ExperimentLayout, args) -> tuple[dict, dict]:
    """The linked test rows to keep per modality, chosen jointly (``select_linked_rows``)."""
    pools = {}
    for m in dict.fromkeys(canonical_modality(x) for x in args.link_modalities):
        obs = _test_obs(layout, args, m)
        if obs is None:
            print(f"  [keep-linked] no test part for {m}; not used to choose keys")
            continue
        pools[m] = obs
    strata = [c for c in args.stratify_by if all(c in o.columns for o in pools.values())]
    budgets = {m: _size(args.test_subset) for m in pools}
    kept, info = select_linked_rows(
        pools, budgets, args.keep_linked, strata, mode=args.stratify_mode,
        unknown_frac=args.unknown_frac, unknown_labels=args.unknown_labels,
        linked_frac=args.linked_frac, per_key=args.linked_per_key, seed=args.seed,
        caps=parse_label_caps(args.cap_labels), linked_quota=args.linked_quota,
    )
    print(f"  [keep-linked] {info['keys_accepted']} of {info['keys_available']} link keys kept; "
          f"rows per modality: {info['rows']}")
    return kept, info


def collect_rows(layout: ExperimentLayout, args, label: str, keep=None):
    """The rows of one modality to embed, in memory, tagged with modality and part."""
    import anndata as ad

    raw = layout.split_source(label, args.split)
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
            # With --keep-linked, linked rows enter only through the joint selection (keep):
            # the others are excluded, so the per-key cap holds and no pair appears in one
            # modality but not in another.
            linked_rows = (row_link_keys(test_obs, args.keep_linked).dropna().index
                           if args.keep_linked else None)
            selected["test"] = stratified_subsample(
                test_obs, _size(args.test_subset), strata, seed=args.seed, mode=args.stratify_mode,
                unknown_frac=args.unknown_frac, unknown_labels=args.unknown_labels, keep=keep,
                exclude=linked_rows, caps=parse_label_caps(args.cap_labels),
            )
        if "train_subset" in args.parts:
            train_obs = obs[obs.index.isin(lists["train"])]
            selected["train_subset"] = stratified_subsample(
                train_obs, _size(args.train_subset), strata, seed=args.seed, mode=args.stratify_mode,
                unknown_frac=args.unknown_frac, unknown_labels=args.unknown_labels,
                caps=parse_label_caps(args.cap_labels),
            )

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
        input_mode=args.input_mode, zero_fraction=args.zero_fraction, panel_size=args.panel_size,
    )
    provenance = {"method": "cancerfoundation", "checkpoint": str(checkpoint),
                  "normalize": args.normalize, **input_mode_provenance(args)}
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    layout = ExperimentLayout.from_root(args.root)
    model_name = args.model or args.name or args.embedder
    labels = list(dict.fromkeys(canonical_modality(m) for m in args.modalities))

    print(f"Collecting rows for {labels} from split '{args.split}'")
    linked, linked_info = ({}, None)
    if args.keep_linked and "test" in args.parts:
        linked, linked_info = linked_test_rows(layout, args)
    pieces = [p for label in labels for p in collect_rows(layout, args, label, keep=linked.get(label))]
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
        "stratify_mode": args.stratify_mode,
        "unknown_frac": args.unknown_frac,
        "unknown_labels": args.unknown_labels,
        "cap_labels": parse_label_caps(args.cap_labels),
        "keep_linked": args.keep_linked,
        "linked_quota": args.linked_quota if args.keep_linked else None,
        "linked_frac": args.linked_frac if args.keep_linked else None,
        "linked_per_key": args.linked_per_key if args.keep_linked else None,
        "linked_keys": ({"accepted": linked_info["keys_accepted"], "available": linked_info["keys_available"],
                         "rows": linked_info["rows"], "modalities": list(linked_info["rows"])}
                        if linked_info else None),
        "variant": args.variant,
    })

    out_dir = layout.embedding_dir(model_name, args.variant)
    print(f"Writing embeddings for '{model_name}' to {out_dir}")
    keys = adata.obs[MODALITY_COL].astype(str) + "|" + adata.obs[PART_COL].astype(str)
    for key in keys.unique():
        label, part = key.split("|")
        mask = (keys == key).to_numpy()
        obs = adata.obs.loc[mask].drop(columns=[PART_COL])
        name = layout.embedding_file(model_name, label, part, args.variant).stem
        write_embeddings(out_dir, name, obs, emb[mask], genes,
                         provenance={**provenance, "modality": label, "part": part})
    set_card_fields(out_dir, model=model_name, variant=args.variant)
    print("Done.")


if __name__ == "__main__":
    main()
