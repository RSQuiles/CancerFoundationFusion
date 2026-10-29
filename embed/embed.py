"""Embed an h5ad with a CancerFoundation checkpoint and save the embeddings to a directory.

General-purpose: any h5ad in, ``<out-dir>/<name>.h5ad`` out, holding the input's obs
plus ``obsm["X_emb"]`` and no X. ``<out-dir>/embed.json`` records, per file, the
checkpoint, normalization and gene panel, and the panel itself is saved as
``<out-dir>/panel_<hash>.txt``. Several files written to the same directory share one
``embed.json``.

    python embed.py --checkpoint path/to/model.ckpt --input data.h5ad --out-dir out/

    # Normalize the model input first (default: values as stored, as in training)
    python embed.py --checkpoint model.ckpt --input data.h5ad --out-dir out/ --normalize cpm_log1p

The obs column ``modality`` (``--modality-col``) decides the gene-selection scheme: a
single value names the file's modality, several values share one gene panel. Without
the column, ``--modality`` (default ``sc``) applies to the whole file. Gene selection
scores genes on its own normalized view of the data whatever ``--normalize`` is (see
``cancerfoundation/data/gene_panel.py``).

The functions here are the machinery ``embed_splits.py`` and ``embed_any.py`` build on:
``load_model``, ``embed_adata`` and ``write_embeddings``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMB_KEY = "X_emb"
EMBED_CARD = "embed.json"
MODALITY_COL = "modality"
NORMALIZE_CHOICES = ("log1p", "cpm_log1p", "cp10k_log1p")


# --------------------------------------------------------------------------- #
# Machinery
# --------------------------------------------------------------------------- #


def load_model(checkpoint_path: Path, cpu: bool = False):
    """``CancerFoundation.load_for_inference`` on GPU when available, gradients off."""
    import torch
    from cancerfoundation.model.model import CancerFoundation

    print(f"Loading checkpoint: {checkpoint_path}")
    model = CancerFoundation.load_for_inference(checkpoint_path)
    for param in model.parameters():
        param.requires_grad = False
    if not cpu and torch.cuda.is_available():
        model = model.cuda()
        print("Inference device: GPU")
    else:
        print("Inference device: CPU")
    return model


def panel_hash(genes: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(genes)).encode("utf-8")).hexdigest()[:12]


def flat_genes(genes) -> list[str]:
    """``model.embed`` returns a list, or a dict per modality for per-modality panels."""
    if isinstance(genes, dict):
        return sorted({g for v in genes.values() for g in v})
    return list(genes)


def embed_adata(
    model,
    adata,
    modality_col: str | None = MODALITY_COL,
    modality: str | None = None,
    batch_size: int = 64,
    normalize: str | None = None,
    shared_panel: bool = True,
    chunk_rows: int = 100_000,
    panel_sample: int = 50_000,
    seed: int = 0,
) -> tuple[np.ndarray, list[str] | dict[str, list[str]]]:
    """Embed every row of ``adata`` through one gene panel, in chunks if it is large.

    With ``modality_col`` naming a column with several values, the panel is fitted
    across them (``shared_panel``); a single value names the modality; without the
    column, ``modality`` (default ``"sc"``) drives gene selection. ``normalize`` is
    the model-input normalization (None, "log1p", "cpm_log1p", "cp10k_log1p"). When
    ``adata`` has more than ``chunk_rows`` rows, the panel is fitted once on a
    ``panel_sample``-row sample (stratified on ``modality_col``) and reused for every
    chunk, so all rows still see the same input genes.
    """
    from utils.sampling import stratified_subsample

    has_col = modality_col is not None and modality_col in adata.obs.columns
    multi = has_col and adata.obs[modality_col].nunique() > 1
    if not has_col and modality_col is not None:
        print(f"No obs column '{modality_col}': treating every row as '{modality or 'sc'}'")
    common = dict(batch_size=batch_size, normalize=normalize, modality=modality or "sc",
                  modality_col=modality_col if has_col else None, shared_panel=shared_panel)

    def run(sub, gene_subset=None):
        kwargs = dict(common)
        if gene_subset is not None:
            kwargs["gene_subset"] = gene_subset
        df, genes = model.embed(sub, **kwargs)
        return df.to_numpy().astype(np.float32), genes

    if adata.n_obs <= chunk_rows:
        return run(adata)

    if multi and not shared_panel:
        raise ValueError("Chunked embedding needs a shared panel; drop shared_panel=False or raise chunk_rows")
    strata = [modality_col] if multi else None
    fit_names = stratified_subsample(adata.obs, panel_sample, strata, seed=seed)
    print(f"Fitting the gene panel on {len(fit_names)} of {adata.n_obs} rows")
    _, genes = run(adata[adata.obs_names.isin(fit_names)].copy())
    blocks = []
    for start in range(0, adata.n_obs, chunk_rows):
        stop = min(start + chunk_rows, adata.n_obs)
        print(f"Embedding rows {start}-{stop} of {adata.n_obs}")
        emb, _ = run(adata[start:stop].copy(), gene_subset=genes)
        blocks.append(emb)
    return np.concatenate(blocks, axis=0), genes


def write_embeddings(
    out_dir: Path,
    name: str,
    obs,
    emb: np.ndarray,
    genes=None,
    provenance: dict | None = None,
    obsm_key: str = EMB_KEY,
) -> Path:
    """Write ``<out_dir>/<name>.h5ad`` (obs + obsm, no X) and record it in ``embed.json``.

    ``genes`` (the input panel, if the embedder has one) is saved once per distinct
    panel as ``panel_<hash>.txt``; the hash goes into the file's ``embed.json`` entry,
    which is what tells a reader whether two files saw the same inputs.
    """
    import anndata as ad

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.h5ad"
    ad.AnnData(obs=obs.copy(), obsm={obsm_key: np.asarray(emb, dtype=np.float32)}).write_h5ad(path)

    entry = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(provenance or {}),
        "n_rows": int(len(obs)),
        "dim": int(np.asarray(emb).shape[1]),
        "obsm_key": obsm_key,
        "panel_hash": None,
        "n_panel_genes": None,
    }
    if genes is not None:
        genes = flat_genes(genes)
        entry["panel_hash"] = panel_hash(genes)
        entry["n_panel_genes"] = len(genes)
        (out_dir / f"panel_{entry['panel_hash']}.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")

    card_path = out_dir / EMBED_CARD
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    card.setdefault("files", {})[path.name] = entry
    card["updated"] = entry["created"]
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"  wrote {path} ({len(obs)} rows, dim {entry['dim']})")
    return path


def set_card_fields(out_dir: Path, **fields) -> None:
    """Top-level fields of ``<out_dir>/embed.json`` (e.g. the model name)."""
    card_path = Path(out_dir) / EMBED_CARD
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    card.update(fields)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")


def add_normalization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--normalize",
        default=None,
        choices=NORMALIZE_CHOICES,
        help=(
            "Normalize the model input before embedding: log1p, or CPM / CP10K by each "
            "row's total over all input genes, then log1p. Default: none, the values are "
            "embedded as stored, as in training. Gene-panel selection does its own "
            "normalization either way."
        ),
    )


def add_model_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU inference even when a GPU is available.")
    parser.add_argument("--chunk-rows", type=int, default=100_000,
                        help="Embed at most this many rows per call; the panel is fitted once.")
    parser.add_argument("--panel-sample", type=int, default=50_000,
                        help="Rows the gene panel is fitted on when the data needs chunking.")
    parser.add_argument("--no-shared-panel", dest="shared_panel", action="store_false",
                        help="Fit a separate panel per modality (not comparable across modalities).")
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(shared_panel=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed an h5ad with a CancerFoundation checkpoint into a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt model checkpoint.")
    parser.add_argument("--input", required=True, help="Path to the input .h5ad file.")
    parser.add_argument("--out-dir", required=True, help="Directory the embeddings are written to.")
    parser.add_argument("--name", default=None, help="Output file stem (default: the input's stem).")
    parser.add_argument("--obsm-key", default=EMB_KEY, help="obsm key of the saved embedding.")
    parser.add_argument("--modality-col", default=MODALITY_COL,
                        help="obs column with per-row modality; several values share one gene panel.")
    parser.add_argument("--modality", default=None, choices=["sc", "bulk", "pseudobulk"],
                        help="Modality of the whole file when it has no --modality-col column (default: sc).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Embed only this many rows, stratified by --stratify-by (default: all rows).")
    parser.add_argument("--stratify-by", nargs="*", default=["tissue_general"],
                        help="obs columns the --subsample is stratified on (missing ones are skipped).")
    add_normalization_args(parser)
    add_model_run_args(parser)
    return parser.parse_args()


def read_input(path: Path, subsample: int | None = None, stratify_by: list[str] | None = None, seed: int = 0):
    """An h5ad in memory, or a stratified subsample of its rows.

    With ``subsample``, the file is opened backed and only the chosen rows are read, so a
    large file never has to fit in memory. Strata not in obs are skipped with a note.
    """
    import anndata as ad

    if subsample is None or subsample <= 0:
        return ad.read_h5ad(path)
    from utils.sampling import stratified_subsample

    backed = ad.read_h5ad(path, backed="r")
    try:
        strata = [c for c in (stratify_by or []) if c in backed.obs.columns]
        skipped = sorted(set(stratify_by or []) - set(strata))
        if skipped:
            print(f"  stratify columns not in obs, skipped: {skipped}")
        n_total = backed.n_obs
        names = stratified_subsample(backed.obs, subsample, strata, seed=seed)
        adata = backed[backed.obs_names.isin(names)].to_memory()
    finally:
        backed.file.close()
    print(f"  subsampled {adata.n_obs} of {n_total} rows" + (f" (stratified on {strata})" if strata else ""))
    return adata


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    checkpoint = Path(args.checkpoint)
    for path in (input_path, checkpoint):
        if not path.exists():
            sys.exit(f"Error: not found: {path}")

    print(f"Loading data: {input_path}")
    adata = read_input(input_path, args.subsample, args.stratify_by, args.seed)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    model = load_model(checkpoint, args.cpu)
    emb, genes = embed_adata(
        model, adata,
        modality_col=args.modality_col, modality=args.modality,
        batch_size=args.batch_size, normalize=args.normalize,
        shared_panel=args.shared_panel, chunk_rows=args.chunk_rows,
        panel_sample=args.panel_sample, seed=args.seed,
    )
    write_embeddings(
        Path(args.out_dir), args.name or input_path.stem, adata.obs, emb, genes,
        provenance={
            "method": "cancerfoundation",
            "checkpoint": str(checkpoint),
            "input": str(input_path),
            "modality_col": args.modality_col,
            "shared_panel": args.shared_panel,
            "normalize": args.normalize,
            "subsample": args.subsample if args.subsample and args.subsample > 0 else None,
            "stratify_by": args.stratify_by if args.subsample and args.subsample > 0 else None,
            "seed": args.seed,
        },
        obsm_key=args.obsm_key,
    )
    print("Done.")


if __name__ == "__main__":
    main()
