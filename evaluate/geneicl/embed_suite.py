"""Embed every dataset of a GeneICL task suite, in the format GeneICL's probe reads.

    # CancerFoundation model (newest checkpoint in checkpoints/<model>/ unless --ckpt)
    python -m evaluate.geneicl.embed_suite --model unified_v1 [--suite benchmark] [--datasets GSE107422 ...]
    # any embedder registered in embed/embed_any.py, written as --name (default: its name)
    python -m evaluate.geneicl.embed_suite --embedder pca --option n_pcs=50

Writes ``embeddings/<model>/geneicl_<suite>/``:

- ``<dataset>.npz`` with ``emb`` (n, D) float32 and ``obs_names``, exactly what GeneICL's
  ``benchmark.py emb_dir=`` and its ``fm_baselines/*_extract.py`` use;
- ``embed.json``: model, checkpoint or options, suite, GeneICL commit, gene space and,
  per dataset, rows, dim and ``panel_hash``; ``panel_<hash>.txt`` holds each panel.

GeneICL datasets are log1p-CPM over gene symbols (``reference_genes.txt``). They are
embedded as stored by default (``--normalize`` off): CancerFoundation bins each row by
quantiles, which a monotone transform does not change, and gene-panel selection
normalizes for itself. The panel is fitted per dataset, on all of its rows and without
labels, as GeneICL's FM extractors do. ``--gene-ids auto`` (the default for a model)
renames the symbols to Ensembl IDs when that overlaps the model vocabulary more.

Datasets whose ``.npz`` exists are skipped unless ``--overwrite``; a directory written
by another checkpoint or embedder is refused. Rows the embedder could not embed
(non-finite) stop the run, since the probe would silently impute them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed.embed import (  # noqa: E402
    EMBED_CARD,
    add_model_run_args,
    add_normalization_args,
    flat_genes,
    panel_hash,
)
from evaluate._cli import parse_with_config  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402

# Card fields that must agree before existing files are reused.
IDENTITY_FIELDS = ("method", "checkpoint", "options", "suite", "gene_ids", "normalize", "modality")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed the datasets of a GeneICL task suite for GeneICL's frozen-embedding probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--geneicl-repo", type=Path, default=None,
                        help=f"GeneICL checkout (default: ${S.GENEICL_ENV_VAR}, else ../GeneICL)")
    parser.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--model", default=None, help="CancerFoundation model name (checkpoints/<model>/)")
    backend.add_argument("--embedder", default=None, help="Embedder registered in embed/embed_any.py")
    parser.add_argument("--name", default=None, help="Output model name with --embedder (default: its name)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Keyword argument for --embedder (repeatable, values parsed as YAML)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint in checkpoints/<model>/ or a full path (default: newest *.ckpt)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only these datasets (default: all)")
    parser.add_argument("--gene-ids", default=None, choices=list(S.GENE_IDS),
                        help="Gene names given to the model: symbols, Ensembl IDs, or auto "
                             "(by vocabulary overlap). Default: auto for --model, symbol for --embedder")
    parser.add_argument("--modality", default="bulk", choices=["sc", "bulk", "pseudobulk"],
                        help="Modality the gene panel is selected for (GeneICL data is bulk)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: embeddings/<model>/geneicl_<suite>/)")
    parser.add_argument("--overwrite", action="store_true", help="Re-embed datasets that already have a file")
    add_normalization_args(parser)
    add_model_run_args(parser)
    return parser


# --------------------------------------------------------------------------- #
# Machinery (no model needed; the self-check drives it with a fake embed_fn)
# --------------------------------------------------------------------------- #


def dataset_adata(X: np.ndarray, obs_names: list[str], cols: np.ndarray, genes: list[str], modality: str):
    """One GeneICL dataset as an AnnData: the kept columns, renamed, with obs ``modality``."""
    import anndata as ad
    import pandas as pd

    obs = pd.DataFrame({"modality": modality}, index=pd.Index(obs_names, name=None))
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    return ad.AnnData(X=np.ascontiguousarray(X[:, cols]), obs=obs, var=var)


def read_card(out_dir: Path) -> dict:
    path = Path(out_dir) / EMBED_CARD
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_identity(card: dict, identity: dict, overwrite: bool) -> None:
    """Refuse to mix files from different checkpoints / embedders in one directory."""
    if not card.get("files") or overwrite:
        return
    differ = {k: (card.get(k), identity.get(k)) for k in IDENTITY_FIELDS if card.get(k) != identity.get(k)}
    if differ:
        details = "; ".join(f"{k}: on disk {a!r}, now {b!r}" for k, (a, b) in differ.items())
        raise SystemExit(f"Error: the existing GeneICL embeddings were made differently ({details}). "
                         "Pass --overwrite to re-embed every dataset.")


def embed_suite(
    suite_path: Path,
    datasets: list[str],
    out_dir: Path,
    embed_fn,
    gene_cols: np.ndarray,
    gene_names: list[str],
    identity: dict,
    modality: str = "bulk",
    overwrite: bool = False,
) -> dict:
    """Embed each dataset with ``embed_fn(adata) -> (emb, panel genes or None)``.

    Writes ``<out_dir>/<dataset>.npz`` (``emb``, ``obs_names``) and keeps
    ``<out_dir>/embed.json`` current after every dataset, so an interrupted run
    resumes where it stopped. Returns the card.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card = read_card(out_dir)
    check_identity(card, identity, overwrite)
    if overwrite:
        card = {}
    card.update(identity)
    card.setdefault("files", {})

    for ds in datasets:
        path = out_dir / f"{ds}.npz"
        if path.exists() and not overwrite:
            print(f"[skip] {ds} (exists)")
            continue
        X, obs = S.load_dataset(suite_path, ds)
        adata = dataset_adata(X, obs, gene_cols, gene_names, modality)
        print(f"[{ds}] {adata.n_obs} samples x {adata.n_vars} genes")
        emb, genes = embed_fn(adata)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(obs):
            raise ValueError(f"{ds}: embedding shape {emb.shape} for {len(obs)} samples")
        bad = ~np.isfinite(emb).all(axis=1)
        if bad.any():
            names = [obs[i] for i in np.flatnonzero(bad)[:5]]
            raise ValueError(f"{ds}: {int(bad.sum())} of {len(obs)} rows have non-finite embeddings "
                             f"(e.g. {names}); the probe would impute them, so nothing was written")
        np.savez(path, emb=emb, obs_names=np.asarray(obs, dtype=object))

        entry = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_rows": int(emb.shape[0]), "dim": int(emb.shape[1]),
                 "panel_hash": None, "n_panel_genes": None}
        if genes is not None:
            genes = flat_genes(genes)
            entry["panel_hash"] = panel_hash(genes)
            entry["n_panel_genes"] = len(genes)
            (out_dir / f"panel_{entry['panel_hash']}.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
        card["files"][path.name] = entry
        card["updated"] = entry["created"]
        (out_dir / EMBED_CARD).write_text(json.dumps(card, indent=2), encoding="utf-8")
        print(f"  wrote {path} ({emb.shape[0]} rows, dim {emb.shape[1]})")
    return card


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def model_backend(args, layout):
    """``(name, embed_fn, vocab, identity)`` for a CancerFoundation checkpoint."""
    from embed.embed import embed_adata, load_model
    from embed.embed_splits import resolve_checkpoint
    from experiments.layout import validate_model_name

    name = validate_model_name(args.model)
    checkpoint = resolve_checkpoint(layout, name, args.ckpt)
    model = load_model(checkpoint, args.cpu)

    def embed_fn(adata):
        return embed_adata(model, adata, modality_col=None, modality=args.modality,
                           batch_size=args.batch_size, normalize=args.normalize,
                           shared_panel=args.shared_panel, chunk_rows=args.chunk_rows,
                           panel_sample=args.panel_sample, seed=args.seed)

    identity = {"method": "cancerfoundation", "checkpoint": str(checkpoint), "options": None}
    return name, embed_fn, model.vocab, identity


def embedder_backend(args):
    """``(name, embed_fn, None, identity)`` for an ``embed_any`` embedder, refitted per dataset."""
    from embed.embed_any import build_embedder, parse_options, run_embedder
    from experiments.layout import validate_model_name

    options = parse_options(args.option)
    if args.embedder == "pca" and "normalize" not in options:
        options["normalize"] = "none"
        print("pca: GeneICL data is already log1p-CPM, so normalize=none (pass --option normalize=... to change)")
    embedder = build_embedder(args.embedder, options)
    name = validate_model_name(args.name or args.embedder)

    def embed_fn(adata):
        return run_embedder(embedder, adata), embedder.genes

    identity = {"method": args.embedder, "checkpoint": None, "options": options}
    return name, embed_fn, None, identity


def main(argv: list[str] | None = None) -> dict:
    from experiments.layout import ExperimentLayout

    args = parse_with_config(build_parser(), "geneicl_embed", argv)
    if not (args.model or args.embedder):
        sys.exit("Error: give --model or --embedder")
    if args.model and (args.name or args.option):
        sys.exit("Error: --name and --option only apply with --embedder")
    layout = ExperimentLayout.from_root(args.root)
    repo = S.resolve_geneicl(args.geneicl_repo)
    suite_path = S.suite_dir(repo, args.suite)
    datasets = S.list_datasets(suite_path, args.datasets)

    if args.model:
        name, embed_fn, vocab, identity = model_backend(args, layout)
    else:
        name, embed_fn, vocab, identity = embedder_backend(args)
    gene_ids = args.gene_ids or ("auto" if args.model else "symbol")
    symbols = S.reference_genes(suite_path)
    cols, genes, space = S.map_genes(symbols, gene_ids, vocab=vocab)
    covered = f", {len(set(vocab).intersection(genes))} in the model vocabulary" if vocab is not None else ""
    print(f"Genes: {len(genes)} of {len(symbols)} GeneICL symbols as {space}{covered}")

    identity.update(model=name, suite=args.suite, gene_ids=space,
                    normalize=args.normalize if args.model else None,
                    modality=args.modality, geneicl_repo=str(repo), geneicl_commit=S.geneicl_commit(repo),
                    n_input_genes=len(genes))
    out_dir = args.out_dir or layout.geneicl_embedding_dir(name, args.suite)
    print(f"{name}: {len(datasets)} {args.suite} datasets -> {out_dir}")
    card = embed_suite(suite_path, datasets, out_dir, embed_fn, cols, genes, identity,
                       modality=args.modality, overwrite=args.overwrite)
    print("Done.")
    return card


if __name__ == "__main__":
    main()
