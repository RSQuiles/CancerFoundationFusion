"""UMAP evaluation utilities.

CLI usage (example):

	python umaps.py \
		--run-name experiment_name \
		--adata ./my_query.h5ad \
		--out-dir ./umap_outputs

This script:
1) resolves a checkpoint from the run directory,
2) loads the `CancerFoundation` LightningModule,
3) embeds the provided AnnData via `CancerFoundation.embed(adata)`,
4) computes UMAP on the embeddings,
5) saves a UMAP plot and an annotated `.h5ad`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import sys
sys.path.insert(0, "../")
from cancerfoundation.model.model import CancerFoundation


_EPOCH_RE = re.compile(r"epoch_(\d+)")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_path: Path


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def resolve_run_checkpoint(
    run_name: str,
    save_root: str | Path = "./save",
    ckpt: str | Path | None = None,
) -> RunPaths:
    """Resolve a run name to a checkpoint path.

    If `ckpt` is provided it is used directly; otherwise we look in
    `{save_root}/{run_name}` for `*.ckpt`.

    Heuristic:
    - prefer the highest `epoch_XX*.ckpt` if present,
    - otherwise fall back to newest-mtime `*.ckpt`.
    """
    save_root = _as_path(save_root)
    run_dir = save_root / run_name
    if ckpt is not None:
        ckpt_path = _as_path(ckpt)
        if ckpt_path.is_dir():
            raise ValueError(f"--ckpt must be a file, got directory: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return RunPaths(run_dir=ckpt_path.parent, ckpt_path=ckpt_path)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpts = sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in: {run_dir}")

    def epoch_num(p: Path) -> int | None:
        m = _EPOCH_RE.search(p.stem)
        return int(m.group(1)) if m else None

    ckpts_with_epoch = [(epoch_num(p), p) for p in ckpts]
    epoch_candidates = [pair for pair in ckpts_with_epoch if pair[0] is not None]
    if epoch_candidates:
        _, best = max(epoch_candidates, key=lambda t: t[0])
        return RunPaths(run_dir=run_dir, ckpt_path=best)

    best = max(ckpts, key=lambda p: p.stat().st_mtime)
    return RunPaths(run_dir=run_dir, ckpt_path=best)


def load_vocab_from_json(vocab_json: str | Path) -> dict:
    """Load a vocab mapping from a `vocab.json` file."""
    import json

    vocab_json = _as_path(vocab_json)
    with vocab_json.open("r") as f:
        return json.load(f)


def load_model_for_inference(
    ckpt_path: str | Path,
    vocab: dict | None = None,
    device: str | None = None,
) -> CancerFoundation:
    """Load a trained model checkpoint and move to the requested device."""
    ckpt_path = _as_path(ckpt_path)
    if vocab is None:
        model = CancerFoundation.load_from_checkpoint(str(ckpt_path))
    else:
        model = CancerFoundation.load_from_checkpoint(str(ckpt_path), vocab=vocab)

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


def embed_adata(
    model: CancerFoundation,
    adata: sc.AnnData,
    batch_size: int = 64,
    obsm_key: str = "X_cf",
) -> sc.AnnData:
    """Compute embeddings and store them in `adata.obsm[obsm_key]`."""
    emb_df: pd.DataFrame = model.embed(adata, flavor=args.flavor, batch_size=batch_size)
    adata.obsm[obsm_key] = emb_df.to_numpy(dtype=np.float32)
    return adata


def compute_umap(
    adata: sc.AnnData,
    use_rep: str = "X_cf",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 0,
) -> sc.AnnData:
    """Compute UMAP on `adata.obsm[use_rep]` and store in `adata.obsm['X_umap']`."""
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    return adata


def save_umap_plot(
    adata: sc.AnnData,
    out_png: str | Path,
    color: str | Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Save a UMAP plot to a PNG file."""
    out_png = _as_path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    sc.pl.umap(adata, color=color, title=title, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png


def _parse_color_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for v in values:
        if "," in v:
            out.extend([x for x in (s.strip() for s in v.split(",")) if x])
        else:
            out.append(v)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)

    paths = resolve_run_checkpoint(
        run_name=args.run_name,
        save_root=args.save_root,
        ckpt=args.ckpt,
    )
    out_dir = _as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix or args.run_name
    out_png = out_dir / f"{prefix}.umap.png"
    out_h5ad = out_dir / f"{prefix}.umap.h5ad"

    vocab = None
    if args.vocab_json is not None:
        vocab = load_vocab_from_json(args.vocab_json)

    try:
        print("Loading model...")
        model = load_model_for_inference(
            paths.ckpt_path, vocab=vocab, device=args.device
        )
    except TypeError as e:
        # Common failure mode: checkpoint does not contain vocab and user didn't pass --vocab-json.
        raise RuntimeError(
            "Failed to load model from checkpoint. If this checkpoint was saved without vocab in hparams, "
            "pass --vocab-json pointing to the training data vocab.json."
        ) from e

    adata_path = _as_path(args.adata)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    print("Loading AnnData...")
    adata = sc.read_h5ad(adata_path)

    print("Embedding...")
    embed_adata(model, adata, batch_size=args.embed_batch_size, obsm_key="X_cf")
    
    print("Computing UMAP...")
    compute_umap(
        adata,
        use_rep="X_cf",
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )

    color = _parse_color_list(args.color)
    title = prefix
    save_umap_plot(adata, out_png=out_png, color=color, title=title)

    adata.write_h5ad(out_h5ad)
    print(f"Checkpoint: {paths.ckpt_path}")
    print(f"Saved plot:  {out_png}")
    print(f"Saved h5ad:  {out_h5ad}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed AnnData and save UMAP plot.")
    p.add_argument("--run-name", required=True, help="Run name under --save-root")
    p.add_argument(
        "--save-root",
        default="../submits_biomed/save",
        help="Root folder containing run directories (default: ./save)",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional explicit checkpoint path (overrides run-name resolution)",
    )
    p.add_argument("--adata", required=True, help="Path to input .h5ad")
    p.add_argument(
        "--out-dir",
        default="./umap",
        help="Output directory for plot and annotated h5ad",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Optional prefix for output files (defaults to run-name)",
    )
    p.add_argument(
        "--vocab-json",
        default=None,
        help="Optional vocab.json path (only needed if checkpoint lacks vocab)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for inference: cuda, cpu, cuda:0, ... (default: auto)",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size used inside model.embed (default: 64)",
    )
    p.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="n_neighbors for UMAP graph (default: 15)",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="min_dist for UMAP (default: 0.5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for UMAP (default: 0)",
    )
    p.add_argument(
        "--color",
        nargs="*",
        default=None,
        help="Obs column(s) to color by (space-separated or comma-separated)",
    )
    p.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        help="Flavor used for HVG selection",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
