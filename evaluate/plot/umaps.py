"""UMAP evaluation utilities.

Single-model CLI usage:

    python umaps.py \
        --run-name experiment_name \
        --adata ./my_query.h5ad \
        --out-dir ./umap_outputs

Ablation CLI usage (generates one UMAP per model, saved inside each model dir):

    python umaps.py \
        --ablation-dir ./save/my_ablation \
        --adata ./my_query.h5ad \
        --color cancer_type modality

This script:
1) resolves a checkpoint from the run directory,
2) loads the `CancerFoundation` LightningModule,
3) embeds the provided AnnData via `CancerFoundation.embed(adata)`,
4) computes UMAP on the embeddings,
5) saves a UMAP plot and an annotated `.h5ad`.

When the AnnData contains a ``modality`` column, single-cell observations are
drawn as small transparent dots and bulk observations as larger stars, so both
populations are visually distinguishable while sharing the same colour scale.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch
from utils import sample_h5ad_subset_from_prefix, subsample_adata

import sys
sys.path.insert(0, "../")
from cancerfoundation.model.model import CancerFoundation


_EPOCH_RE = re.compile(r"epoch_(\d+)")

# Modality aliases recognised as single-cell.
_SC_MODALITY_ALIASES: frozenset[str] = frozenset({
    "sc", "singlecell", "scrna", "scrnaseq",
})


# --------------------------------------------------------------------------- #
# Checkpoint / model helpers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_path: Path


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _epoch_num(p: Path) -> int:
    m = _EPOCH_RE.search(p.stem)
    return int(m.group(1)) if m else -1


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

    ckpts_with_epoch = [(p, _epoch_num(p)) for p in ckpts]
    epoch_candidates = [(p, e) for p, e in ckpts_with_epoch if e >= 0]
    if epoch_candidates:
        best = max(epoch_candidates, key=lambda t: t[1])[0]
        return RunPaths(run_dir=run_dir, ckpt_path=best)

    best = max(ckpts, key=lambda p: p.stat().st_mtime)
    return RunPaths(run_dir=run_dir, ckpt_path=best)


def _find_best_ckpt(model_dir: Path) -> Path | None:
    """Search model_dir and its checkpoints/ subdirectory for the best ckpt."""
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "checkpoints/*.ckpt"):
        candidates.extend(model_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=_epoch_num)


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


# --------------------------------------------------------------------------- #
# Embedding / UMAP
# --------------------------------------------------------------------------- #

def embed_adata(
    model: CancerFoundation,
    adata: sc.AnnData,
    batch_size: int = 64,
    flavor: str = "seurat",
    obsm_key: str = "X_cf",
) -> sc.AnnData:
    """Compute embeddings and store them in `adata.obsm[obsm_key]`."""
    emb_df: pd.DataFrame = model.embed(adata, flavor=flavor, batch_size=batch_size)
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


# --------------------------------------------------------------------------- #
# Modality-aware UMAP plot
# --------------------------------------------------------------------------- #

def _is_sc_modality(val: str) -> bool:
    """Return True if the modality string refers to single-cell data."""
    normalised = val.lower().replace(" ", "").replace("-", "").replace("_", "")
    return normalised in _SC_MODALITY_ALIASES or normalised.startswith("sc")


def _assign_colors(
    col: pd.Series,
) -> tuple[np.ndarray, dict[str, tuple] | None, list[str] | None]:
    """
    Map a Series to RGBA colours.

    Returns
    -------
    rgba       : (N, 4) float32 array of RGBA colours.
    cat_colors : category → RGBA mapping, or None for continuous data.
    categories : sorted category list, or None for continuous data.
    """
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        vals = col.to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(vmin, vmax) if vmin != vmax else plt.Normalize(0, 1)
        rgba = plt.get_cmap("viridis")(norm(vals)).astype(np.float32)
        return rgba, None, None

    categories = sorted(col.astype(str).unique())
    n = len(categories)
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    cat_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}
    rgba = np.array([cat_colors[str(v)] for v in col], dtype=np.float32)
    return rgba, cat_colors, categories


def _plot_umap_modality_aware(
    adata: sc.AnnData,
    color_keys: list[str | None],
    title: str | None,
) -> plt.Figure:
    """
    UMAP plot with modality-sensitive markers.

    Single-cell observations → small transparent dots (marker "o").
    Bulk observations        → larger stars with dark edge (marker "*").

    One panel is produced per entry in ``color_keys``.
    """
    umap_coords = adata.obsm["X_umap"]
    modality_vals = adata.obs["modality"].astype(str).to_numpy()
    sc_mask   = np.array([_is_sc_modality(v) for v in modality_vals])
    bulk_mask = ~sc_mask

    n_panels = max(len(color_keys), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.5), squeeze=False)
    axes_flat = axes[0]

    for ax, color_key in zip(axes_flat, color_keys):
        if color_key is not None and color_key in adata.obs:
            point_colors, cat_colors, categories = _assign_colors(adata.obs[color_key])
        else:
            fallback = np.array([0.35, 0.55, 0.80, 1.0], dtype=np.float32)
            point_colors = np.tile(fallback, (len(adata), 1))
            cat_colors, categories = None, None

        # SC: small translucent dots (plotted first so bulk sits on top)
        if sc_mask.any():
            ax.scatter(
                umap_coords[sc_mask, 0],
                umap_coords[sc_mask, 1],
                c=point_colors[sc_mask],
                s=4, alpha=0.45, marker="o", linewidths=0,
                rasterized=True,
            )

        # Bulk: larger stars with dark edge (plotted on top for visibility)
        if bulk_mask.any():
            ax.scatter(
                umap_coords[bulk_mask, 0],
                umap_coords[bulk_mask, 1],
                c=point_colors[bulk_mask],
                s=70, alpha=0.90, marker="*",
                linewidths=0.4, edgecolors="black",
                rasterized=True,
            )

        # Axis styling
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(color_key or "", fontsize=10, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Category colour legend (placed outside to the right)
        if cat_colors and categories:
            color_handles = [
                mpatches.Patch(color=cat_colors[cat], label=cat)
                for cat in categories
            ]
            ncol = max(1, len(categories) // 20)
            ax.legend(
                handles=color_handles,
                title=color_key, fontsize=7, title_fontsize=8,
                bbox_to_anchor=(1.02, 1.0), loc="upper left",
                frameon=True, ncol=ncol,
            )

    # Modality legend anchored to the first panel (bottom-left).
    modality_handles = [
        mlines.Line2D(
            [], [], marker="o", color="grey", markersize=5, alpha=0.6,
            linestyle="None", label="Single-cell",
        ),
        mlines.Line2D(
            [], [], marker="*", color="grey", markersize=10, alpha=0.9,
            markeredgecolor="black", markeredgewidth=0.4,
            linestyle="None", label="Bulk",
        ),
    ]
    axes_flat[0].add_artist(
        axes_flat[0].legend(
            handles=modality_handles,
            title="Modality", fontsize=8, title_fontsize=8,
            loc="lower left", frameon=True, framealpha=0.9,
        )
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Public save helper
# --------------------------------------------------------------------------- #

def save_umap_plot(
    adata: sc.AnnData,
    out_png: str | Path,
    color: str | Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Save a UMAP plot to a PNG/PDF file.

    If ``adata.obs`` contains a ``modality`` column the plot uses
    modality-aware markers (dots for single-cell, stars for bulk).
    Otherwise the standard scanpy renderer is used.
    """
    out_png = _as_path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    has_modality = "modality" in adata.obs.columns

    if has_modality:
        if color is None:
            color_keys: list[str | None] = [None]
        elif isinstance(color, str):
            color_keys = [color]
        else:
            color_keys = list(color)

        fig = _plot_umap_modality_aware(adata, color_keys, title)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        sc.pl.umap(adata, color=color, title=title, show=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi)
        plt.close()

    return out_png


# --------------------------------------------------------------------------- #
# Ablation-level UMAP generation
# --------------------------------------------------------------------------- #

def run_ablation_umaps(
    ablation_dir: str | Path,
    adata: AnnData,
    color: list[str] | None = None,
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    seed: int = 0,
    device: str | None = None,
    vocab: dict | None = None,
) -> None:
    """Generate and save a UMAP for every model inside an ablation directory.

    For each ``{model_dir}`` found under ``ablation_dir`` that contains a
    ``.ckpt`` file (directly or inside a ``checkpoints/`` subdirectory), the
    function:

    1. Loads the model from the best-epoch checkpoint.
    2. Embeds a copy of ``adata``.
    3. Computes UMAP coordinates.
    4. Saves the figure to ``{model_dir}/umap.png``.

    Parameters
    ----------
    ablation_dir : path to the top-level ablation experiment directory.
    adata        : AnnData to embed (a copy is made per model).
    color        : obs column(s) to colour by.
    """
    ablation_dir = _as_path(ablation_dir)

    model_dirs = sorted(d for d in ablation_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No subdirectories found in {ablation_dir}.")
        return

    for model_dir in model_dirs:
        ckpt_path = _find_best_ckpt(model_dir)
        if ckpt_path is None:
            print(f"[skip] {model_dir.name} — no checkpoint found")
            continue

        print(f"[{model_dir.name}] checkpoint: {ckpt_path.name}")

        try:
            model = load_model_for_inference(ckpt_path, vocab=vocab, device=device)
        except Exception as exc:
            print(f"  [error] could not load model: {exc}")
            continue

        adata_copy = adata.copy()
        try:
            embed_adata(model, adata_copy, batch_size=embed_batch_size,
                        flavor=flavor, obsm_key="X_cf")
            compute_umap(adata_copy, use_rep="X_cf", n_neighbors=n_neighbors,
                         min_dist=min_dist, random_state=seed)
            out_png = model_dir / "umap.png"
            save_umap_plot(adata_copy, out_png=out_png, color=color,
                           title=model_dir.name)
            print(f"  saved → {out_png}")
        except Exception as exc:
            print(f"  [error] UMAP generation failed: {exc}")
        finally:
            del model, adata_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

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

    # ---- common: must supply some adata source ----
    assert args.adata_dir is not None or args.adata is not None, (
        "A path to an AnnData (--adata) or a directory (--adata-dir) must be provided."
    )
    if args.adata_dir is not None:
        assert args.adata is None, (
            "Provide either --adata or --adata-dir, not both."
        )

    vocab = None
    if args.vocab_json is not None:
        vocab = load_vocab_from_json(args.vocab_json)

    color = _parse_color_list(args.color)

    # ---- load adata ----
    sample_size = args.sample_size
    if args.adata is not None:
        adata_path = _as_path(args.adata)
        if not adata_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {adata_path}")
        print(f"Loading {sample_size} cells from {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        adata = subsample_adata(adata, sample_size)
    else:
        adata_dir  = args.adata_dir
        adata_prefix = args.adata_prefix  # may be None → use all .h5ad files
        print(
            f"Loading {sample_size} cells from "
            f"{'all' if not adata_prefix else repr(adata_prefix)} "
            f".h5ad files in {adata_dir}..."
        )
        adata = sample_h5ad_subset_from_prefix(adata_prefix, adata_dir, sample_size)

    # ---- ablation mode ----
    if args.ablation_dir is not None:
        run_ablation_umaps(
            ablation_dir=args.ablation_dir,
            adata=adata,
            color=color,
            embed_batch_size=args.embed_batch_size,
            flavor=args.flavor,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
            device=args.device,
            vocab=vocab,
        )
        return 0

    # ---- single-model mode ----
    assert args.run_name is not None, (
        "--run-name is required unless --ablation-dir is provided."
    )

    paths = resolve_run_checkpoint(
        run_name=args.run_name,
        save_root=args.save_root,
        ckpt=args.ckpt,
    )
    out_dir = _as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix   = args.out_prefix or args.run_name
    out_png  = out_dir / f"{prefix}.umap.png"
    out_h5ad = out_dir / f"{prefix}.umap.h5ad"

    try:
        print("Loading model...")
        model = load_model_for_inference(
            paths.ckpt_path, vocab=vocab, device=args.device
        )
    except TypeError as e:
        raise RuntimeError(
            "Failed to load model from checkpoint. If the checkpoint was saved "
            "without vocab in hparams, pass --vocab-json with the training vocab.json."
        ) from e

    print("Embedding...")
    print(adata)
    embed_adata(model, adata, batch_size=args.embed_batch_size,
                flavor=args.flavor, obsm_key="X_cf")

    print("Computing UMAP...")
    compute_umap(
        adata,
        use_rep="X_cf",
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )

    save_umap_plot(adata, out_png=out_png, color=color, title=prefix)

    adata.write_h5ad(out_h5ad)
    print(f"Checkpoint: {paths.ckpt_path}")
    print(f"Saved plot:  {out_png}")
    print(f"Saved h5ad:  {out_h5ad}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed AnnData and save UMAP plot.")

    # --- run identification (single-model mode) ---
    p.add_argument(
        "--run-name",
        default=None,
        help="Run name under --save-root (required for single-model mode).",
    )
    p.add_argument(
        "--save-root",
        default="../submits_biomed/save",
        help="Root folder containing run directories (default: ../submits_biomed/save).",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional explicit checkpoint path (overrides run-name resolution).",
    )

    # --- ablation mode ---
    p.add_argument(
        "--ablation-dir",
        default=None,
        help=(
            "Generate one UMAP per model found inside this ablation directory. "
            "Each UMAP is saved to {model_dir}/umap.png. "
            "Mutually exclusive with --run-name."
        ),
    )

    # --- data source ---
    p.add_argument("--adata", help="Path to input .h5ad.")
    p.add_argument(
        "--adata-dir",
        help="Directory with .h5ad files to virtually concatenate.",
    )
    p.add_argument(
        "--adata-prefix",
        type=str,
        default=None,
        help=(
            "Filename prefix for virtual concatenation (e.g. 'train'). "
            "If omitted, all .h5ad files in --adata-dir are used."
        ),
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help="Number of cells to sample from the input AnnData (default: 50000).",
    )

    # --- single-model output ---
    p.add_argument(
        "--out-dir",
        default="./umap",
        help="Output directory for plot and annotated h5ad (single-model mode).",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Optional prefix for output files (defaults to run-name).",
    )

    # --- misc ---
    p.add_argument(
        "--vocab-json",
        default=None,
        help="Optional vocab.json path (only needed if checkpoint lacks vocab).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: cuda, cpu, cuda:0, ... (default: auto).",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size inside model.embed (default: 64).",
    )
    p.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="n_neighbors for UMAP graph (default: 15).",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="min_dist for UMAP (default: 0.5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for UMAP (default: 0).",
    )
    p.add_argument(
        "--color",
        nargs="*",
        default=None,
        help="obs column(s) to colour by (space- or comma-separated).",
    )
    p.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        help="Flavor used for HVG selection (default: seurat).",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
