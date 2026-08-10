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

``--tissues`` restricts every figure (joint, SC-only, bulk-only, pseudobulk) to
observations from the listed tissues, and prefixes the output filenames with them:

    python umaps.py --ablation-dir ./save/my_ablation --eval-adata eval.h5ad \
        --tissues lung,breast
    # → {model_dir}/lung-breast_umap.png, lung-breast_umap_sc.png, ...
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import torch
from utils import sample_h5ad_subset_from_prefix, subsample_adata

# Model-free UMAP computation/rendering, re-exported so this module's API is unchanged.
from umap_render import (  # noqa: F401
    _as_path,
    _assign_colors,
    _canonical_modality,
    _is_pb_modality,
    _is_sc_modality,
    _plot_umap_modality_aware,
    compute_umap,
    save_umap_plot,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.model.model import CancerFoundation
from evaluate.utils import (
    BULK_MODALITIES,
    MOD_BULK,
    MOD_PAIRED_BULK,
    MOD_PAIRED_PB,
    MOD_PAIRED_SC,
    MOD_PB,
    MOD_SC,
    MOD_SYNTH_PB,
    SC_MODALITIES,
    canonicalize_modality_column,
    generate_pseudobulk_adata,
)


_EPOCH_RE = re.compile(r"epoch_(\d+)")


# --------------------------------------------------------------------------- #
# Checkpoint / model helpers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_path: Path


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
    - Always prefer the newest-mtime (most recently modified) `*.ckpt`.
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

    # Always use the most recently modified checkpoint
    best = max(ckpts, key=lambda p: p.stat().st_mtime)
    return RunPaths(run_dir=run_dir, ckpt_path=best)


def _find_best_ckpt(model_dir: Path) -> Path | None:
    """Search model_dir and its checkpoints/ subdirectory for the best ckpt.
    
    Returns the most recently modified checkpoint.
    """
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "checkpoints/*.ckpt"):
        candidates.extend(model_dir.glob(pattern))
    if not candidates:
        return None
    # Return the most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
        model = CancerFoundation.load_for_inference(ckpt_path)
    else:
        model = CancerFoundation.load_for_inference(ckpt_path, vocab=vocab)

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
    normalized: bool = True,
    modality: str = "sc",
    modality_col: str | None = None,
) -> sc.AnnData:
    """Compute embeddings and store them in `adata.obsm[obsm_key]`.

    Gene selection is deterministic and modality-aware (see ``CancerFoundation.embed``):
    scanpy ``seurat`` HVG for single-cell, log1p + MAD for bulk/pseudobulk. When a
    modality column is present (``modality_col``, else auto-detected as "_eval_modality"
    or "modality"), selection is fitted per modality group; otherwise the scalar
    ``modality`` applies to all cells.
    """
    if modality_col is None:
        modality_col = next(
            (c for c in ("_eval_modality", "modality") if c in adata.obs.columns), None
        )
    kwargs = dict(
        flavor=flavor,
        batch_size=batch_size,
        normalized=normalized,
        hvg_select=True,
        modality=modality,
    )
    if modality_col is not None and modality_col in adata.obs.columns:
        kwargs["modality_col"] = modality_col
    result = model.embed(adata, **kwargs)
    emb_df = result[0] if isinstance(result, tuple) else result
    adata.obsm[obsm_key] = emb_df.to_numpy(dtype=np.float32)
    return adata


# --------------------------------------------------------------------------- #
# Tissue filtering
# --------------------------------------------------------------------------- #

# Tried in order when no explicit tissue column is given.
_TISSUE_COLUMN_FALLBACKS: tuple[str, ...] = ("tissue_general", "tissue")


def _normalize_tissue(value) -> str:
    """Casefold a tissue label so 'Lung ' and 'lung' compare equal."""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_tissue_column(
    adata: AnnData,
    tissue_column: str | None = None,
) -> str | None:
    """Return the obs column holding tissue labels, or None if there is none.

    An explicit ``tissue_column`` is used as-is (and must exist); otherwise the
    first of ``_TISSUE_COLUMN_FALLBACKS`` present in ``adata.obs`` wins.
    """
    candidates = (tissue_column,) if tissue_column else _TISSUE_COLUMN_FALLBACKS
    return next((c for c in candidates if c in adata.obs.columns), None)


def filter_adata_by_tissues(
    adata: AnnData,
    tissues: list[str] | None,
    tissue_column: str | None = None,
    label: str = "adata",
) -> AnnData:
    """Subset *adata* to observations whose tissue label is in *tissues*.

    Matching is case-, space- and hyphen-insensitive. Returns ``adata`` unchanged
    when ``tissues`` is empty/None, so callers can pass the flag through blindly.

    Raises ``ValueError`` when no tissue column exists or when the selection is
    empty — silently plotting everything under a tissue-prefixed filename would
    misrepresent the figure.
    """
    if not tissues:
        return adata

    col = resolve_tissue_column(adata, tissue_column)
    if col is None:
        raise ValueError(
            f"--tissues given but no tissue column found in {label}.obs. "
            f"Looked for {tissue_column or list(_TISSUE_COLUMN_FALLBACKS)}; "
            f"available columns: {sorted(adata.obs.columns)[:40]}"
        )

    values = adata.obs[col].astype(str)
    normalised = values.map(_normalize_tissue)
    wanted = {_normalize_tissue(t) for t in tissues}
    mask = normalised.isin(wanted).to_numpy()

    missing = sorted(wanted - set(normalised.unique()))
    if missing:
        # Report the raw labels — those are what the user has to type back.
        available = sorted(values.unique())
        shown = ", ".join(available[:30]) + (" ..." if len(available) > 30 else "")
        print(f"  [warn] no rows in {label}.obs['{col}'] for: {', '.join(missing)}")
        print(f"         available values: {shown}")

    if not mask.any():
        raise ValueError(
            f"Tissue filter {sorted(wanted)} matched 0 of {adata.n_obs} rows in "
            f"{label}.obs['{col}']."
        )

    out = adata[mask].copy()

    # Report per-modality survival: a filter that wipes out bulk (e.g. because the
    # bulk rows carry no tissue label) would otherwise silently yield an SC-only plot.
    mod_col = next(
        (c for c in ("_eval_modality", "modality") if c in out.obs.columns), None
    )
    breakdown = ""
    if mod_col is not None:
        counts = out.obs[mod_col].astype(str).value_counts()
        breakdown = " [" + ", ".join(f"{k}:{v}" for k, v in counts.items()) + "]"
    matched = sorted(wanted - set(missing))
    print(
        f"  tissue filter on '{col}': {adata.n_obs} → {out.n_obs} cells "
        f"({', '.join(matched)}){breakdown}"
    )
    return out


def tissue_file_prefix(tissues: list[str] | None) -> str:
    """Filename prefix encoding the selected tissues, e.g. ``'lung-breast_'``.

    Empty string when no filter is active, so callers can always concatenate it.
    """
    if not tissues:
        return ""
    slugs = list(dict.fromkeys(
        re.sub(r"[^0-9a-z]+", "_", str(t).strip().lower()).strip("_") or "tissue"
        for t in tissues
    ))
    joined = "-".join(slugs)
    if len(joined) > 60:  # keep filenames sane when many tissues are selected
        joined = f"{slugs[0]}-and-{len(slugs) - 1}-more"
    return f"{joined}_"


# --------------------------------------------------------------------------- #
# Modality-split UMAP helper
# --------------------------------------------------------------------------- #

# Per-modality marker styling shared by the modality- and tissue-coloured panels.
# Colour is only used by the modality panel; the tissue panel overrides it per point.
_MODALITY_ORDER: tuple[str, ...] = ("sc", "pseudobulk", "bulk")
_MODALITY_STYLE: dict[str, dict] = {
    "sc":         dict(color="#4393c3", s=4,  marker="o", linewidths=0,   label="Single-cell"),
    "pseudobulk": dict(color="#4dac26", s=30, marker="^", linewidths=0.5, edgecolors="black", label="Pseudobulk"),
    "bulk":       dict(color="#d6604d", s=30, marker="D", linewidths=0.5, edgecolors="black", label="Bulk"),
}


def _scatter_by_modality(
    ax,
    coords: np.ndarray,
    modality_vals: np.ndarray,
    point_colors: np.ndarray | None = None,
) -> None:
    """Scatter sc/pseudobulk/bulk points with their per-modality marker styles.

    When ``point_colors`` is None each modality uses its fixed palette colour
    (modality panel). When given an (N, 4) RGBA array the marker colours are taken
    from it per point (tissue panel), keeping the marker *shape* per modality.
    """
    for mod in _MODALITY_ORDER:
        mask = modality_vals == mod
        if not mask.any():
            continue
        style = dict(_MODALITY_STYLE[mod])
        if point_colors is not None:
            style.pop("color", None)
            style["c"] = point_colors[mask]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            alpha=0.4, rasterized=True, **style,
        )


def _plot_pseudobulk_figure(
    adata: sc.AnnData,
    title: str | None = None,
    tissue_key: str | None = "tissue",
    skip_unknown: bool = False,
) -> plt.Figure:
    """Pseudobulk UMAP figure: one modality-coloured panel, plus a tissue-coloured
    panel when ``tissue_key`` is present in ``adata.obs``.

    Both panels share the same layout and per-modality marker shapes (dots for SC,
    triangles for pseudobulk, diamonds for bulk); the tissue panel recolours points
    by tissue while keeping the shape encoding so modalities stay distinguishable.
    """
    umap_coords = adata.obsm["X_umap"]
    modality_vals = adata.obs["modality"].astype(str).to_numpy()

    has_tissue = tissue_key is not None and tissue_key in adata.obs.columns
    n_panels = 2 if has_tissue else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.0), squeeze=False)
    axes_flat = axes[0]

    def _style_axes(ax, panel_title: str) -> None:
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Marker-shape handles reused for the modality legend on both panels.
    modality_handles = [
        mlines.Line2D(
            [], [], linestyle="None", markersize=6, alpha=0.6,
            marker=_MODALITY_STYLE[m]["marker"],
            color=_MODALITY_STYLE[m]["color"],
            markeredgecolor="black" if m != "sc" else _MODALITY_STYLE[m]["color"],
            label=_MODALITY_STYLE[m]["label"],
        )
        for m in _MODALITY_ORDER
        if (modality_vals == m).any()
    ]

    # ── Panel 1: colour by modality ──────────────────────────────────────
    ax0 = axes_flat[0]
    _scatter_by_modality(ax0, umap_coords, modality_vals, point_colors=None)
    _style_axes(ax0, "modality")
    ax0.legend(
        handles=modality_handles, title="Modality", fontsize=8, title_fontsize=8,
        loc="lower left", frameon=True, framealpha=0.9,
    )

    # ── Panel 2: colour by tissue (shape still encodes modality) ─────────
    if has_tissue:
        ax1 = axes_flat[1]
        tissue_col = adata.obs[tissue_key].astype(str)
        point_colors, cat_colors, categories = _assign_colors(
            tissue_col, skip_unknown=skip_unknown
        )
        _scatter_by_modality(ax1, umap_coords, modality_vals, point_colors=point_colors)
        _style_axes(ax1, str(tissue_key))
        if cat_colors and categories:
            n_cats = len(categories)
            leg_fontsize = 3 if n_cats > 50 else 5 if n_cats > 20 else 6
            ncol = max(1, n_cats // 25)
            tissue_legend = ax1.legend(
                handles=[mpatches.Patch(color=cat_colors[c], label=c) for c in categories],
                title=tissue_key, fontsize=leg_fontsize, title_fontsize=8,
                loc="lower right", frameon=True, framealpha=0.85, ncol=ncol,
                borderpad=0.5, labelspacing=0.3, handlelength=1.0,
            )
            ax1.add_artist(tissue_legend)  # pin so the modality legend doesn't overwrite
        # Modality (shape) legend so triangles/diamonds/dots remain readable.
        ax1.legend(
            handles=modality_handles, title="Modality", fontsize=8, title_fontsize=8,
            loc="lower left", frameon=True, framealpha=0.9,
        )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _save_pseudobulk_umap(
    adata: sc.AnnData,
    model: CancerFoundation | None,
    joint_out_png: Path,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    dpi: int = 200,
    use_sc: bool = False,
    skip_unknown: bool = False,
) -> None:
    """Compute and save a UMAP with SC, bulk, and pseudobulk observations.

    Three code paths, tried in order of preference:

    **Eval-precomputed**: when ``adata.obs`` contains ``_eval_modality`` (written by
    ``build_eval_adata.py``), embeddings are read directly from ``adata.obsm["X_cf"]``
    using the ``synth_pb`` rows — no model needed (``model`` may be ``None``).

    **Modality-precomputed**: when the raw ``modality`` column already contains
    pseudobulk rows (the ``--precomputed-pb`` training feature), those rows' existing
    embeddings in ``adata.obsm["X_cf"]`` are used directly instead of regenerating
    pseudobulks — also needs no model.

    **Live**: ``model`` must be provided; pseudobulk profiles are generated from SC
    expression and embedded on the fly. On-the-fly pseudobulks are labelled with the
    ``group_column`` tissue they were aggregated from.

    Each pseudobulk point carries its tissue (from ``group_column``) when available,
    so the saved figure gets both a modality-coloured and a tissue-coloured panel.

    Output is named after ``joint_out_png`` with ``_pseudobulk`` or
    ``_pseudobulk_with_sc`` inserted before the extension.
    """
    if "X_cf" not in adata.obsm:
        print("  [warn] pseudobulk UMAP skipped — 'X_cf' not in obsm")
        return

    # Per-subset tissue labels (aligned with sc_emb / pb_emb / bulk_emb), or None
    # for any subset whose tissue is unavailable. All-None → modality-only plot.
    sc_tissue = pb_tissue = bulk_tissue = None

    def _tissue_for(a: sc.AnnData, mask) -> np.ndarray | None:
        if group_column not in a.obs.columns:
            return None
        return a.obs[group_column].astype(str).to_numpy()[mask] if mask is not None \
            else a.obs[group_column].astype(str).to_numpy()

    # ── Path 1 — eval-precomputed: synth_pb already embedded via build_eval_adata ──
    if "_eval_modality" in adata.obs.columns:
        eval_mod      = adata.obs["_eval_modality"].astype(str)
        # These are _eval_modality labels, not build_eval_adata's filename prefixes.
        # This used to match "subsampled", which is only ever a prefix, so every
        # plain "sc" row was dropped from the SC layer of the plot.
        sc_mask       = eval_mod.isin(SC_MODALITIES).values
        bulk_mask     = eval_mod.isin(BULK_MODALITIES).values
        synth_pb_mask = (eval_mod == MOD_SYNTH_PB).values

        if not synth_pb_mask.any():
            print("  [warn] pseudobulk UMAP skipped — no 'synth_pb' rows in _eval_modality")
            return

        emb = adata.obsm["X_cf"]
        sc_emb   = emb[sc_mask]
        bulk_emb = emb[bulk_mask]
        pb_emb   = emb[synth_pb_mask]
        sc_tissue   = _tissue_for(adata, sc_mask)
        bulk_tissue = _tissue_for(adata, bulk_mask)
        pb_tissue   = _tissue_for(adata, synth_pb_mask)

    # ── Path 2 — modality-precomputed: real pseudobulk rows already in the data ──
    elif "modality" in adata.obs.columns and adata.obs["modality"].astype(str).map(
        _is_pb_modality
    ).any():
        print("  using precomputed pseudobulk rows (modality == 'pseudobulk')")
        canon    = adata.obs["modality"].astype(str).map(_canonical_modality).to_numpy()
        sc_mask   = canon == "sc"
        bulk_mask = canon == "bulk"
        pb_mask   = canon == "pseudobulk"

        emb = adata.obsm["X_cf"]
        sc_emb   = emb[sc_mask]
        bulk_emb = emb[bulk_mask]
        pb_emb   = emb[pb_mask]
        sc_tissue   = _tissue_for(adata, sc_mask)
        bulk_tissue = _tissue_for(adata, bulk_mask)
        pb_tissue   = _tissue_for(adata, pb_mask)

    # ── Path 3 — live: generate and embed pseudobulks on the fly ────────────────
    else:
        if model is None:
            print("  [warn] pseudobulk UMAP skipped — no model and no precomputed pseudobulk")
            return
        if "modality" not in adata.obs.columns:
            print("  [warn] pseudobulk UMAP skipped — 'modality' not in obs")
            return
        if group_column not in adata.obs.columns:
            print(f"  [warn] pseudobulk UMAP skipped — '{group_column}' not in obs")
            return

        modality_vals = adata.obs["modality"].astype(str).replace("nan", "sc").to_numpy()
        sc_mask_bool  = np.array([_is_sc_modality(v) for v in modality_vals])
        sc_sub        = adata[sc_mask_bool]
        bulk_sub      = adata[~sc_mask_bool]

        if sc_sub.n_obs == 0:
            print("  [warn] pseudobulk UMAP skipped — no SC cells found")
            return

        pb_adata = generate_pseudobulk_adata(
            sc_sub,
            group_column=group_column,
            n_sc_per_pb=n_sc_per_pb,
            agg_method=agg_method,
            n_pb=bulk_sub.n_obs if bulk_sub.n_obs > 0 else None,
            seed=seed,
        )
        if pb_adata is None:
            return
        try:
            embed_adata(model, pb_adata, batch_size=embed_batch_size,
                        flavor=flavor, obsm_key="X_cf", modality="pseudobulk")
        except Exception as exc:
            print(f"  [warn] pseudobulk embedding failed: {exc}")
            return

        sc_emb   = sc_sub.obsm["X_cf"]
        pb_emb   = pb_adata.obsm["X_cf"]
        bulk_emb = (
            bulk_sub.obsm["X_cf"] if bulk_sub.n_obs > 0
            else np.empty((0, sc_emb.shape[1]), dtype=np.float32)
        )
        # SC/bulk tissue from the source obs; PB tissue from the group each was built
        # from (generate_pseudobulk_adata writes it into obs[group_column]).
        sc_tissue   = _tissue_for(sc_sub, None)
        bulk_tissue = _tissue_for(bulk_sub, None) if bulk_sub.n_obs > 0 else np.empty(0, dtype=object)
        pb_tissue   = _tissue_for(pb_adata, None)

    if use_sc:
        combined_emb = np.vstack([sc_emb, pb_emb, bulk_emb])
        modality_col = (
            ["sc"]           * len(sc_emb)
            + ["pseudobulk"] * len(pb_emb)
            + ["bulk"]       * len(bulk_emb)
        )
        tissue_parts = [sc_tissue, pb_tissue, bulk_tissue]
    else:
        combined_emb = np.vstack([pb_emb, bulk_emb])
        modality_col = (
            ["pseudobulk"] * len(pb_emb)
            + ["bulk"]       * len(bulk_emb)
        )
        tissue_parts = [pb_tissue, bulk_tissue]

    obs_dict = {"modality": modality_col}
    # Only add tissue if every included subset supplied it (keeps alignment exact).
    if all(t is not None for t in tissue_parts):
        obs_dict["tissue"] = np.concatenate(
            [np.asarray(t, dtype=object) for t in tissue_parts]
        )
    combined = AnnData(obs=pd.DataFrame(obs_dict))
    combined.obsm["X_cf"] = combined_emb

    n_cells = combined.n_obs
    try:
        compute_umap(
            combined,
            use_rep="X_cf",
            n_neighbors=min(n_neighbors, n_cells - 1),
            min_dist=min_dist,
            random_state=seed,
        )
    except Exception as exc:
        print(f"  [warn] pseudobulk UMAP computation failed: {exc}")
        return

    stem   = joint_out_png.stem
    suffix = joint_out_png.suffix
    tag    = "pseudobulk_with_sc" if use_sc else "pseudobulk"
    out_png = joint_out_png.parent / f"{stem}_{tag}{suffix}"

    try:
        fig = _plot_pseudobulk_figure(
            combined, title=f"{stem} ({tag})",
            tissue_key="tissue", skip_unknown=skip_unknown,
        )
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {out_png}")
    except Exception as exc:
        print(f"  [warn] pseudobulk UMAP plot failed: {exc}")


def _save_modality_split_umaps(
    adata: sc.AnnData,
    joint_out_png: Path,
    color: list[str] | None,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    dpi: int = 200,
    skip_unknown: bool = False,
    model: CancerFoundation | None = None,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    only_pseudobulk: bool = False,
) -> None:
    """Compute and save separate UMAP plots for SC-only and bulk-only subsets.

    Each subset re-runs ``sc.pp.neighbors`` + ``sc.tl.umap`` on its own cells
    (using the already-embedded ``X_cf`` vectors) so the UMAP layout reflects
    the internal structure of that modality rather than the shared joint space.

    When ``model`` is provided, also generates a pseudobulk UMAP (see
    ``_save_pseudobulk_umap``).

    Output files are named after ``joint_out_png`` with ``_sc`` / ``_bulk``
    inserted before the extension, e.g.:
        umap.png  →  umap_sc.png   umap_bulk.png   umap_pseudobulk.png
    """
    if "modality" not in adata.obs.columns:
        return

    if not only_pseudobulk:
        modality_vals = adata.obs["modality"].astype(str).replace("nan", "sc").to_numpy()
        sc_mask   = np.array([_is_sc_modality(v) for v in modality_vals])
        bulk_mask = ~sc_mask

        stem = joint_out_png.stem
        suffix = joint_out_png.suffix
        out_dir = joint_out_png.parent

        for label, mask in [("sc", sc_mask), ("bulk", bulk_mask)]:
            n = int(mask.sum())
            if n == 0:
                continue

            subset = adata[mask].copy()
            out_png = out_dir / f"{stem}_{label}{suffix}"
            try:
                compute_umap(
                    subset,
                    use_rep="X_cf",
                    n_neighbors=min(n_neighbors, n - 1),
                    min_dist=min_dist,
                    random_state=seed,
                )
                save_umap_plot(
                    subset,
                    out_png=out_png,
                    color=color,
                    title=f"{stem} ({label})",
                    dpi=dpi,
                    skip_unknown=skip_unknown,
                )
                print(f"  saved → {out_png}")
            except Exception as exc:
                print(f"  [warn] {label}-only UMAP failed: {exc}")

    # Pseudobulk points can come from precomputed eval synth_pb rows, from real
    # precomputed pseudobulk rows in the raw modality column (--precomputed-pb), or
    # be generated on the fly (needs a model). The first two need no model.
    has_eval_synth_pb = (
        "_eval_modality" in adata.obs.columns
        and (adata.obs["_eval_modality"] == MOD_SYNTH_PB).any()
    )
    has_modality_pb = (
        "modality" in adata.obs.columns
        and adata.obs["modality"].astype(str).map(_is_pb_modality).any()
    )
    has_precomputed_pb = has_eval_synth_pb or has_modality_pb
    if model is not None or has_precomputed_pb:
        # Use n_sc_per_pb from model hparams when available
        n_sc_per_pb_actual = (
            int(getattr(model.hparams, "n_sc_per_pseudobulk", n_sc_per_pb))
            if model is not None else n_sc_per_pb
        )
        _common = dict(
            n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
            n_sc_per_pb=n_sc_per_pb_actual, group_column=group_column,
            agg_method=agg_method, embed_batch_size=embed_batch_size,
            flavor=flavor, dpi=dpi, skip_unknown=skip_unknown,
        )
        # UMAP computed with bulk + pseudobulk only (no SC)
        _save_pseudobulk_umap(adata, model, joint_out_png, use_sc=False, **_common)
        # UMAP computed with SC + bulk + pseudobulk
        _save_pseudobulk_umap(adata, model, joint_out_png, use_sc=True, **_common)


# --------------------------------------------------------------------------- #
# Precomputed-embedding helper
# --------------------------------------------------------------------------- #

_EVAL_MOD_TO_MODALITY: dict[str, str] = {
    MOD_SC:          "sc",
    MOD_PAIRED_SC:   "sc",
    MOD_BULK:        "bulk",
    MOD_PAIRED_BULK: "bulk",
    MOD_PB:          "pseudobulk",
    MOD_PAIRED_PB:   "pseudobulk",
    MOD_SYNTH_PB:    "synth_pb",
}


def _build_from_eval_adata(
    eval_adata: AnnData,
    model_name: str,
) -> AnnData | None:
    """Return a copy of *eval_adata* ready for UMAP plotting.

    Copies ``obsm["X_cf_{model_name}"]`` → ``obsm["X_cf"]`` and adds a
    simplified ``modality`` column derived from ``_eval_modality``.
    Returns ``None`` if the embedding key is absent.
    """
    emb_key = f"X_cf_{model_name}"
    if emb_key not in eval_adata.obsm:
        return None

    adata_copy = eval_adata.copy()
    adata_copy.obsm["X_cf"] = eval_adata.obsm[emb_key].copy()

    if "_eval_modality" in adata_copy.obs.columns:
        adata_copy.obs["modality"] = (
            adata_copy.obs["_eval_modality"]
            .astype(str)
            .map(_EVAL_MOD_TO_MODALITY)
            .fillna(adata_copy.obs["_eval_modality"])
        )
    return adata_copy


# --------------------------------------------------------------------------- #
# Ablation-level UMAP generation
# --------------------------------------------------------------------------- #

def run_ablation_umaps(
    ablation_dir: str | Path,
    adata: AnnData,
    eval_adata: AnnData | None = None,
    color: list[str] | None = None,
    embed_batch_size: int = 64,
    flavor: str = "seurat",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    seed: int = 0,
    device: str | None = None,
    vocab: dict | None = None,
    skip_unknown: bool = False,
    modality_split: bool = True,
    n_sc_per_pb: int = 10,
    group_column: str = "tissue_general",
    agg_method: str = "mean",
    only_pseudobulk: bool = False,
    tissues: list[str] | None = None,
    tissue_column: str | None = None,
) -> None:
    """Generate and save a UMAP for every model inside an ablation directory.

    For each ``{model_dir}`` found under ``ablation_dir`` that contains a
    ``.ckpt`` file (directly or inside a ``checkpoints/`` subdirectory), the
    function:

    1. Loads the model from the best-epoch checkpoint (skipped when
       ``eval_adata`` is provided and already contains the model's embeddings).
    2. Embeds a copy of ``adata`` (or reads embeddings from ``eval_adata``).
    3. Computes UMAP coordinates.
    4. Saves the joint figure to ``{model_dir}/umap.png``.
    5. If ``modality_split`` is True, also saves per-modality UMAPs.

    Parameters
    ----------
    ablation_dir : path to the top-level ablation experiment directory.
    adata        : AnnData to embed when model loading is needed.
    eval_adata   : optional pre-built AnnData from ``build_eval_adata.py``
                   that already contains per-model embeddings in
                   ``obsm["X_cf_{model_name}"]``.  When provided, model
                   loading and embedding are skipped for any model whose
                   embedding key is present.
    color        : obs column(s) to colour by.
    modality_split : when True (default), generate per-modality UMAPs.
    tissues      : optional tissue whitelist; every figure is restricted to these
                   tissues and output filenames get them as a prefix
                   (``lung-breast_umap.png``). On the live path the filter is
                   applied *after* embedding, so the gene panel stays the one the
                   full dataset would have produced.
    tissue_column : obs column holding the tissue labels (default: auto-detect
                   ``tissue_general`` then ``tissue``).
    """
    ablation_dir = _as_path(ablation_dir)

    model_dirs = sorted(d for d in ablation_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No subdirectories found in {ablation_dir}.")
        return

    # Embeddings in eval_adata are precomputed, so filtering up front is free and
    # saves repeating the mask for every model.
    if tissues and eval_adata is not None:
        eval_adata = filter_adata_by_tissues(
            eval_adata, tissues, tissue_column, label="eval_adata"
        )

    file_prefix = tissue_file_prefix(tissues)
    title_note = f" ({', '.join(tissues)})" if tissues else ""

    for model_dir in model_dirs:
        model_name = model_dir.name
        emb_key    = f"X_cf_{model_name}"

        # ── Precomputed path: use eval_adata embeddings ─────────────────
        if eval_adata is not None:
            adata_copy = _build_from_eval_adata(eval_adata, model_name)
        else:
            adata_copy = None

        if adata_copy is not None:
            print(f"[{model_name}] using precomputed embeddings from eval_adata")

            try:
                compute_umap(adata_copy, use_rep="X_cf",
                             n_neighbors=min(n_neighbors, adata_copy.n_obs - 1),
                             min_dist=min_dist, random_state=seed)
                out_png = model_dir / f"{file_prefix}umap.png"
                save_umap_plot(adata_copy, out_png=out_png, color=color,
                               title=f"{model_name}{title_note}", skip_unknown=skip_unknown)
                print(f"  saved → {out_png}")

                if modality_split:
                    _save_modality_split_umaps(
                        adata_copy, out_png, color,
                        n_neighbors, min_dist, seed,
                        skip_unknown=skip_unknown,
                        model=None,
                        n_sc_per_pb=n_sc_per_pb,
                        group_column=group_column,
                        agg_method=agg_method,
                        embed_batch_size=embed_batch_size,
                        flavor=flavor,
                        only_pseudobulk=only_pseudobulk
                    )
            except Exception as exc:
                print(f"  [error] UMAP generation failed: {exc}")
            finally:
                del adata_copy
            continue

        # ── Live path: load model and embed ────────────────────────────
        ckpt_path = _find_best_ckpt(model_dir)
        if ckpt_path is None:
            print(f"[skip] {model_name} — no checkpoint found and no precomputed embedding")
            continue

        print(f"[{model_name}] checkpoint: {ckpt_path.name}")

        try:
            model = load_model_for_inference(ckpt_path, vocab=vocab, device=device)
        except Exception as exc:
            print(f"  [error] could not load model: {exc}")
            continue

        adata_copy = adata.copy()
        try:
            embed_adata(model, adata_copy, batch_size=embed_batch_size,
                        flavor=flavor, obsm_key="X_cf")
            # After embedding: the gene panel must be fitted on the full dataset,
            # otherwise a tissue subset silently changes the model's input space.
            adata_copy = filter_adata_by_tissues(
                adata_copy, tissues, tissue_column, label=model_name
            )
            compute_umap(adata_copy, use_rep="X_cf",
                         n_neighbors=min(n_neighbors, adata_copy.n_obs - 1),
                         min_dist=min_dist, random_state=seed)
            out_png = model_dir / f"{file_prefix}umap.png"
            save_umap_plot(adata_copy, out_png=out_png, color=color,
                           title=f"{model_name}{title_note}", skip_unknown=skip_unknown)
            print(f"  saved → {out_png}")

            if modality_split:
                _save_modality_split_umaps(
                    adata_copy, out_png, color,
                    n_neighbors, min_dist, seed,
                    skip_unknown=skip_unknown,
                    model=model,
                    n_sc_per_pb=n_sc_per_pb,
                    group_column=group_column,
                    agg_method=agg_method,
                    embed_batch_size=embed_batch_size,
                    flavor=flavor,
                    only_pseudobulk=only_pseudobulk
                )
        except Exception as exc:
            print(f"  [error] UMAP generation failed: {exc}")
        finally:
            del model, adata_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_str_list(values: list[str] | None) -> list[str] | None:
    """Flatten a space-separated argparse list that may also use commas."""
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
    # --adata / --adata-dir are optional when --eval-adata is provided, because
    # eval_adata already contains the cells with precomputed embeddings.
    has_raw_adata = args.adata is not None or args.adata_dir is not None
    if not has_raw_adata and args.eval_adata is None:
        raise ValueError(
            "Provide a raw AnnData (--adata / --adata-dir) or a precomputed "
            "eval AnnData (--eval-adata)."
        )
    if args.adata_dir is not None:
        assert args.adata is None, "Provide either --adata or --adata-dir, not both."

    vocab = None
    if args.vocab_json is not None:
        vocab = load_vocab_from_json(args.vocab_json)

    color   = _parse_str_list(args.color)
    tissues = _parse_str_list(args.tissues)

    # ---- load raw adata (optional when eval_adata covers everything) ----
    adata = None
    sample_size = args.sample_size
    if args.adata is not None:
        adata_path = _as_path(args.adata)
        if not adata_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {adata_path}")
        print(f"Loading {sample_size} cells from {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        adata = subsample_adata(adata, sample_size)
    elif args.adata_dir is not None:
        adata_dir    = args.adata_dir
        adata_prefix = args.adata_prefix
        print(
            f"Loading {sample_size} cells from "
            f"{'all' if not adata_prefix else repr(adata_prefix)} "
            f".h5ad files in {adata_dir}..."
        )
        adata = sample_h5ad_subset_from_prefix(adata_prefix, adata_dir, sample_size)

    # ---- load eval_adata (precomputed embeddings) if provided ----
    eval_adata = None
    if args.eval_adata is not None:
        eval_adata_path = _as_path(args.eval_adata)
        if not eval_adata_path.exists():
            raise FileNotFoundError(f"eval_adata file not found: {eval_adata_path}")
        print(f"Loading eval_adata from {eval_adata_path} ...")
        import anndata as ad
        eval_adata = ad.read_h5ad(eval_adata_path)
        print(f"  {eval_adata.n_obs} cells, obsm keys: {list(eval_adata.obsm.keys())}")
        # Older eval.h5ad files label SC rows with the filename prefix
        # ("subsampled"); _EVAL_MOD_TO_MODALITY would pass that straight through and
        # the plot would carry a modality the legend does not know.
        canonicalize_modality_column(eval_adata)

    # ---- ablation mode ----
    if args.ablation_dir is not None:
        run_ablation_umaps(
            ablation_dir=args.ablation_dir,
            adata=adata,
            eval_adata=eval_adata,
            color=color,
            embed_batch_size=args.embed_batch_size,
            flavor=args.flavor,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
            device=args.device,
            vocab=vocab,
            skip_unknown=args.skip_unknown,
            modality_split=not args.no_modality_split,
            n_sc_per_pb=args.n_sc_per_pb,
            group_column=args.pb_group_column,
            agg_method=args.pb_agg_method,
            only_pseudobulk=args.plot_pb_only,
            tissues=tissues,
            tissue_column=args.tissue_column,
        )
        return 0

    # ---- single-model mode ----
    assert args.run_name is not None, (
        "--run-name is required unless --ablation-dir is provided."
    )

    out_dir = _as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tissue selection is part of the file identity, so it leads the name and is
    # inherited by the per-modality figures (they derive theirs from out_png.stem).
    prefix   = f"{tissue_file_prefix(tissues)}{args.out_prefix or args.run_name}"
    out_png  = out_dir / f"{prefix}.umap.png"
    out_h5ad = out_dir / f"{prefix}.umap.h5ad"

    model     = None
    ckpt_path = None

    # ── Try precomputed embeddings first ──────────────────────────────────
    adata_to_use = None
    if eval_adata is not None:
        adata_to_use = _build_from_eval_adata(eval_adata, args.run_name)
        if adata_to_use is not None:
            print(f"Using precomputed embeddings (X_cf_{args.run_name}) from eval_adata.")
            adata_to_use = filter_adata_by_tissues(
                adata_to_use, tissues, args.tissue_column, label="eval_adata"
            )
        else:
            print(
                f"Warning: 'X_cf_{args.run_name}' not found in eval_adata — "
                "falling back to model embedding."
            )

    # ── Fall back to live embedding ───────────────────────────────────────
    if adata_to_use is None:
        if adata is None:
            raise ValueError(
                f"No precomputed embedding for '{args.run_name}' in eval_adata "
                "and no --adata provided for live embedding."
            )
        paths = resolve_run_checkpoint(
            run_name=args.run_name,
            save_root=args.save_root,
            ckpt=args.ckpt,
        )
        ckpt_path = paths.ckpt_path
        try:
            print("Loading model...")
            model = load_model_for_inference(ckpt_path, vocab=vocab, device=args.device)
        except TypeError as e:
            raise RuntimeError(
                "Failed to load model from checkpoint. If the checkpoint was saved "
                "without vocab in hparams, pass --vocab-json with the training vocab.json."
            ) from e

        print("Embedding...")
        print(adata)
        embed_adata(model, adata, batch_size=args.embed_batch_size,
                    flavor=args.flavor, obsm_key="X_cf")
        # Filter after embedding so the gene panel is the full dataset's, not the
        # tissue subset's (a different panel behaves like a batch effect).
        adata_to_use = filter_adata_by_tissues(
            adata, tissues, args.tissue_column, label="adata"
        )

    # ── UMAP + plots ──────────────────────────────────────────────────────
    print("Computing UMAP...")
    compute_umap(
        adata_to_use,
        use_rep="X_cf",
        n_neighbors=min(args.neighbors, adata_to_use.n_obs - 1),
        min_dist=args.min_dist,
        random_state=args.seed,
    )

    save_umap_plot(adata_to_use, out_png=out_png, color=color,
                   title=prefix, skip_unknown=args.skip_unknown)

    if not args.no_modality_split:
        _save_modality_split_umaps(
            adata_to_use, out_png, color,
            args.neighbors, args.min_dist, args.seed,
            skip_unknown=args.skip_unknown,
            model=model,
            n_sc_per_pb=args.n_sc_per_pb,
            group_column=args.pb_group_column,
            agg_method=args.pb_agg_method,
            embed_batch_size=args.embed_batch_size,
            flavor=args.flavor,
            only_pseudobulk=args.plot_pb_only
        )

    adata_to_use.write_h5ad(out_h5ad)
    if ckpt_path is not None:
        print(f"Checkpoint:  {ckpt_path}")
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
    p.add_argument(
        "--eval-adata",
        default=None,
        help=(
            "Path to a pre-built eval.h5ad produced by build_eval_adata.py. "
            "When provided, per-model embeddings are read from "
            "obsm['X_cf_{model_name}'] instead of re-embedding on the fly. "
            "Synth-pseudobulk cells (obs._eval_modality == 'synth_pb') are "
            "used directly for the pseudobulk UMAP. Works with --ablation-dir."
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
        "--tissues",
        nargs="*",
        default=None,
        help=(
            "Restrict every UMAP (joint, sc, bulk, pseudobulk) to observations "
            "from these tissues (space- or comma-separated, case-insensitive). "
            "Output filenames are prefixed with them, e.g. "
            "'lung-breast_umap_pseudobulk.png'. On the live-embedding path the "
            "filter is applied after embedding, so the gene panel is unaffected."
        ),
    )
    p.add_argument(
        "--tissue-column",
        type=str,
        default=None,
        help=(
            "obs column holding the tissue labels used by --tissues "
            "(default: first of 'tissue_general', 'tissue' present in obs)."
        ),
    )
    p.add_argument(
        "--flavor",
        type=str,
        default="seurat",
        help="Flavor used for HVG selection (default: seurat).",
    )
    p.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Whether to skip the unknown / nan categories.",
    )
    p.add_argument(
        "--no-modality-split",
        action="store_true",
        help=(
            "Disable per-modality UMAPs. By default, when the AnnData contains "
            "a 'modality' column, separate SC-only and bulk-only UMAPs are "
            "computed and saved alongside the joint UMAP."
        ),
    )
    p.add_argument(
        "--n-sc-per-pb",
        type=int,
        default=10,
        help=(
            "Number of single-cell samples aggregated into each pseudobulk "
            "(default: 10)."
        ),
    )
    p.add_argument(
        "--pb-group-column",
        type=str,
        default="tissue_general",
        help=(
            "obs column used to group SC cells when sampling pseudobulks. "
            "Cells composing one pseudobulk are drawn exclusively from the "
            "same group (default: tissue_general)."
        ),
    )
    p.add_argument(
        "--pb-agg-method",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help=(
            "Aggregation method for combining SC expression into a pseudobulk "
            "profile before embedding: 'mean' (default) or 'sum'."
        ),
    )
    p.add_argument(
        "--plot-pb-only",
        action="store_true",
        help=(
            "Whether to only ouput the pseudobulk UMAP"
        ),
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
