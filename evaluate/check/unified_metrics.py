"""Unified FM evaluation metrics for CancerFoundation models.

Reads pre-computed cell embeddings from an evaluation AnnData built by
``build_eval_adata.py`` and quantifies how well each unified-FM loss objective
is achieved at inference time.

Metrics
-------
  recon_*       Reconstruction accuracy (masked forward pass, SC cells)
  paired_*      Paired alignment  — pseudobulk ↔ real bulk by "paired" column
  agg_paired_*  Aggregation consistency from paired SC → paired PB
  agg_synth_*   Aggregation consistency from on-the-fly synthetic pseudobulks
  contrastive_* Distribution-level bulk / pseudobulk alignment

Usage
-----
Single model:
    python unified_metrics.py \\
        --eval-adata path/to/eval.h5ad \\
        --ckpt path/to/epoch_05.ckpt [--name my_model] \\
        [--out-dir path/to/output]

Ablation directory:
    python unified_metrics.py \\
        --eval-adata path/to/eval.h5ad \\
        --ablation-dir path/to/ablation \\
        [--skip-existing]

The obsm key used to look up embeddings is ``X_cf_{model_name}``, where
``model_name`` is the value of ``--name`` (or the checkpoint's parent
directory name, or the ablation model directory name).

The model checkpoint is required to compute reconstruction and synthetic
aggregation metrics (forward passes).  All other metrics are derived from
the pre-computed embeddings in the eval AnnData.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.model.model import CancerFoundation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

_MODALITY_COL = "_eval_modality"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_best_ckpt(model_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "checkpoints/*.ckpt"):
        candidates.extend(model_dir.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _load_model(ckpt_path: Path, device: str) -> CancerFoundation:
    model = CancerFoundation.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Metric 1: masked reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_reconstruction_metrics(
    model: CancerFoundation,
    sc_adata: ad.AnnData,
    batch_size: int = 64,
    mask_ratio: float = 0.15,
    n_cells: int = 1000,
    seed: int = 0,
    normalized: bool = True,
) -> dict:
    """Pearson R and MSE between predicted and actual expression at masked positions."""
    device = next(model.model.parameters()).device
    rng = np.random.default_rng(seed)

    data = model.preprocess_for_embedding(sc_adata, normalized=normalized)
    if data.n_obs == 0:
        return {}

    n = min(data.n_obs, n_cells)
    idx = rng.choice(data.n_obs, size=n, replace=False)
    data = data[idx].copy()

    X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
    X = X.astype(np.float32)
    gene_ids = torch.LongTensor([model.vocab[g] for g in data.var.index]).to(device)
    n_genes = X.shape[1]

    # Effective mask value after optional bin normalisation
    if model.input_emb_style == "category":
        is_normalised = (
            hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "normalise_bins")
            and model.model.decoder.normalise_bins
        )
        eff_mask = float(model.mask_value) / model.n_bins if is_normalised else float(model.mask_value)
    else:
        eff_mask = float(model.mask_value)  # -1 for continuous

    pearson_rs: list[float] = []
    mse_vals: list[float] = []

    for start in range(0, n, batch_size):
        batch_expr = torch.from_numpy(X[start : start + batch_size]).to(device)
        bs = batch_expr.shape[0]
        batch_genes = gene_ids.unsqueeze(0).expand(bs, -1)

        cls_g = torch.full((bs, 1), model.cls_token_id, dtype=torch.long, device=device)
        cls_e = torch.full((bs, 1), float(model.pad_value), device=device)
        genes_full = torch.cat([cls_g, batch_genes], dim=1)
        expr_full  = torch.cat([cls_e, batch_expr],  dim=1)

        gene_mask = torch.rand(bs, n_genes, device=device) < mask_ratio
        masked_expr = expr_full.clone()
        masked_expr[:, 1:][gene_mask] = eff_mask

        padding_mask = torch.zeros(genes_full.shape, dtype=torch.bool, device=device)

        if model.model.use_generative_training:
            output = model.model.embed(
                src=genes_full, values=masked_expr, src_key_padding_mask=padding_mask
            )
            transformer_out = output[0]
        else:
            transformer_out = model.model.encode(
                src=genes_full, values=masked_expr,
                src_key_padding_mask=padding_mask, check_conditions=False,
            )

        decoder_out = model.model.decoder(transformer_out)
        pred = decoder_out["pred"]
        if pred.dim() == 3:
            pred = pred[:, 1:, 0]
        else:
            pred = pred[:, 1:]

        target = batch_expr
        for i in range(bs):
            m = gene_mask[i]
            if m.sum() < 2:
                continue
            p = pred[i][m].float().cpu().numpy()
            t = target[i][m].float().cpu().numpy()
            if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                continue
            r = float(np.corrcoef(p, t)[0, 1])
            if not np.isnan(r):
                pearson_rs.append(r)
            mse_vals.append(float(np.mean((p - t) ** 2)))

    return {
        "recon_pearson_r":     float(np.mean(pearson_rs)) if pearson_rs else float("nan"),
        "recon_pearson_r_std": float(np.std(pearson_rs))  if pearson_rs else float("nan"),
        "recon_mse":           float(np.mean(mse_vals))   if mse_vals   else float("nan"),
        "recon_n_cells":       len(pearson_rs),
    }


# ---------------------------------------------------------------------------
# Metric 2: paired alignment
# ---------------------------------------------------------------------------

def compute_paired_alignment_metrics(
    pb_emb: np.ndarray,
    bulk_emb: np.ndarray,
    pb_pair_ids: np.ndarray,
    bulk_pair_ids: np.ndarray,
) -> dict:
    """Cosine similarity and retrieval rank between matched pseudobulk–bulk pairs."""
    common = sorted(set(pb_pair_ids.tolist()) & set(bulk_pair_ids.tolist()) - {0})
    if not common:
        log.warning("No common pair IDs — skipping paired alignment metrics.")
        return {}

    pb_dict   = {p: pb_emb  [pb_pair_ids   == p][0] for p in common if (pb_pair_ids   == p).any()}
    bulk_dict = {p: bulk_emb[bulk_pair_ids  == p][0] for p in common if (bulk_pair_ids  == p).any()}
    common = [p for p in common if p in pb_dict and p in bulk_dict]
    if not common:
        return {}

    pb_arr   = np.stack([pb_dict[p]   for p in common])
    bulk_arr = np.stack([bulk_dict[p] for p in common])
    N = len(common)

    pb_n   = pb_arr   / (np.linalg.norm(pb_arr,   axis=1, keepdims=True) + 1e-8)
    bulk_n = bulk_arr / (np.linalg.norm(bulk_arr, axis=1, keepdims=True) + 1e-8)

    sim = pb_n @ bulk_n.T
    paired_sims = np.diag(sim)
    ranks = [int((sim[i] > sim[i, i]).sum()) + 1 for i in range(N)]

    off = ~np.eye(N, dtype=bool)
    return {
        "paired_cosine_sim_mean":       float(paired_sims.mean()),
        "paired_cosine_sim_std":        float(paired_sims.std()),
        "paired_rank_mean":             float(np.mean(ranks)),
        "paired_rank_median":           float(np.median(ranks)),
        "paired_n_pairs":               N,
        "paired_random_baseline_cosine": float(sim[off].mean()) if N > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Metric 3a: aggregation consistency — paired SC → paired PB
# ---------------------------------------------------------------------------

def _agg_from_paired_sc(
    paired_sc_adata: ad.AnnData,
    paired_sc_emb: np.ndarray,
    paired_pb_adata: ad.AnnData,
    paired_pb_emb: np.ndarray,
) -> dict:
    """Cosine sim between paired_pb embedding and mean embedding of its SC constituents."""
    sc_ids = np.asarray(paired_sc_adata.obs["paired"])
    pb_ids = np.asarray(paired_pb_adata.obs["paired"])
    common = sorted(set(sc_ids.tolist()) & set(pb_ids.tolist()) - {0})
    if not common:
        return {}

    sims: list[float] = []
    for pid in common:
        sc_mask = sc_ids == pid
        pb_mask = pb_ids == pid
        if not sc_mask.any() or not pb_mask.any():
            continue
        mean_sc = paired_sc_emb[sc_mask].mean(axis=0)
        pb_e    = paired_pb_emb[pb_mask][0]
        pb_n    = pb_e    / (np.linalg.norm(pb_e)    + 1e-8)
        sc_n    = mean_sc / (np.linalg.norm(mean_sc) + 1e-8)
        sims.append(float(np.dot(pb_n, sc_n)))

    if not sims:
        return {}
    return {
        "agg_paired_cosine_pb_to_mean_sc":     float(np.mean(sims)),
        "agg_paired_cosine_pb_to_mean_sc_std": float(np.std(sims)),
        "agg_paired_n_pairs":                  len(sims),
    }


# ---------------------------------------------------------------------------
# Metric 3b: aggregation consistency — synthetic pseudobulks
# ---------------------------------------------------------------------------

def compute_aggregation_metrics(
    model: CancerFoundation,
    sc_adata: ad.AnnData,
    group_column: str = "tissue_general",
    n_sc_per_pb: int = 10,
    n_pb: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    normalized: bool = True,
) -> dict:
    """
    Sample groups of SC cells, aggregate into pseudobulks, embed both, then
    measure cosine similarity between the PB embedding and mean SC embedding.
    """
    if group_column not in sc_adata.obs.columns:
        log.warning("group_column '%s' not found — skipping synthetic aggregation.", group_column)
        return {}

    rng = np.random.default_rng(seed)
    group_vals = sc_adata.obs[group_column].astype(str).to_numpy()
    group_idx  = {g: np.where(group_vals == g)[0] for g in np.unique(group_vals)}
    valid      = [g for g, idx in group_idx.items() if len(idx) >= n_sc_per_pb]

    if not valid:
        log.warning("No group with >= %d SC cells — skipping synthetic aggregation.", n_sc_per_pb)
        return {}

    n_pb = min(n_pb, max(1, len(valid) * 5))
    sims: list[float] = []

    for g in rng.choice(valid, size=n_pb, replace=True).tolist():
        pool = group_idx[g]
        sc_idx = rng.choice(pool, size=n_sc_per_pb, replace=len(pool) < n_sc_per_pb)
        sc_sub = sc_adata[sc_idx].copy()

        try:
            emb_df, gene_set = model.embed(sc_sub, batch_size=batch_size, normalized=normalized)
            sc_embs = emb_df.to_numpy(dtype=np.float32)
        except Exception:
            log.warning("SC embed failed:\n%s", traceback.format_exc())
            continue

        mean_sc = sc_embs.mean(axis=0)

        sc_exprs = sc_sub.X if isinstance(sc_sub.X, np.ndarray) else sc_sub.X.toarray()
        pb_expr  = np.expm1(sc_exprs.astype(np.float64)).sum(axis=0)
        total = pb_expr.sum()
        if total > 0:
            pb_expr = pb_expr / total * 1e6
        pb_expr = np.log1p(pb_expr).astype(np.float32)

        pb_adata = ad.AnnData(X=pb_expr[np.newaxis, :], var=sc_sub.var.copy())
        pb_adata.obs_names = ["pb_synth_0"]

        try:
            pb_df, _ = model.embed(pb_adata, batch_size=1, normalized=True, gene_subset=gene_set)
            pb_emb   = pb_df.to_numpy(dtype=np.float32)[0]
        except Exception:
            log.warning("PB embed failed:\n%s", traceback.format_exc())
            continue

        pb_n = pb_emb  / (np.linalg.norm(pb_emb)  + 1e-8)
        sc_n = mean_sc / (np.linalg.norm(mean_sc) + 1e-8)
        sims.append(float(np.dot(pb_n, sc_n)))

    if not sims:
        return {}
    return {
        "agg_synth_cosine_pb_to_mean_sc":     float(np.mean(sims)),
        "agg_synth_cosine_pb_to_mean_sc_std": float(np.std(sims)),
        "agg_synth_n_pseudobulks":            len(sims),
    }


# ---------------------------------------------------------------------------
# Metric 4: contrastive / distribution-level alignment
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bw: float) -> float:
    diff  = X[:, None, :] - Y[None, :, :]
    sq    = (diff ** 2).sum(axis=2)
    return float(np.exp(-sq / (2 * bw ** 2)).mean())


def _mmd(X: np.ndarray, Y: np.ndarray) -> float:
    XY   = np.vstack([X, Y])
    diff = XY[:, None, :] - XY[None, :, :]
    sq   = (diff ** 2).sum(axis=2)
    pos  = sq[sq > 0]
    bw   = float(np.sqrt(np.median(pos) / 2)) if len(pos) > 0 else 1.0
    return _rbf_kernel(X, X, bw) + _rbf_kernel(Y, Y, bw) - 2 * _rbf_kernel(X, Y, bw)


def compute_contrastive_metrics(
    bulk_emb: np.ndarray,
    pb_emb: np.ndarray,
    n_max: int = 500,
    seed: int = 0,
) -> dict:
    """Cross-modal cosine similarity and MMD between bulk and pseudobulk distributions."""
    rng = np.random.default_rng(seed)
    if len(bulk_emb) > n_max:
        bulk_emb = bulk_emb[rng.choice(len(bulk_emb), n_max, replace=False)]
    if len(pb_emb) > n_max:
        pb_emb = pb_emb[rng.choice(len(pb_emb), n_max, replace=False)]

    bn = bulk_emb / (np.linalg.norm(bulk_emb, axis=1, keepdims=True) + 1e-8)
    pn = pb_emb   / (np.linalg.norm(pb_emb,   axis=1, keepdims=True) + 1e-8)

    def _within(e: np.ndarray) -> float:
        if len(e) < 2:
            return float("nan")
        s = e @ e.T
        return float(s[~np.eye(len(e), dtype=bool)].mean())

    return {
        "contrastive_cross_cosine_mean":   float((bn @ pn.T).mean()),
        "contrastive_within_bulk_cosine":  _within(bn),
        "contrastive_within_pb_cosine":    _within(pn),
        "contrastive_mmd":                 _mmd(bulk_emb, pb_emb),
        "contrastive_n_bulk":              len(bulk_emb),
        "contrastive_n_pb":                len(pb_emb),
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_single_model(
    model_name: str,
    eval_adata: ad.AnnData,
    out_dir: Path,
    ckpt_path: Path | None = None,
    batch_size: int = 64,
    mask_ratio: float = 0.15,
    n_sc_per_pb: int = 10,
    n_synth_pb: int = 200,
    group_column: str = "tissue_general",
    device: str = "cpu",
    seed: int = 0,
    skip_existing: bool = False,
    normalized: bool = True,
) -> dict:
    """Compute all unified FM metrics for one model. Returns a flat metric dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "unified_metrics.json"

    if skip_existing and metrics_file.exists():
        log.info("  Cached — loading %s", metrics_file)
        with metrics_file.open() as f:
            return json.load(f)

    obsm_key = f"X_cf_{model_name}"
    if obsm_key not in eval_adata.obsm:
        log.error(
            "obsm key '%s' not found in eval AnnData. Available: %s",
            obsm_key, list(eval_adata.obsm.keys()),
        )
        return {}

    # ── Split eval_adata by modality and read pre-computed embeddings ─────
    def _get(label: str):
        mask = (eval_adata.obs.get(_MODALITY_COL, pd.Series(dtype=str)) == label).values
        if not mask.any():
            return None, None
        sub = eval_adata[mask]
        return sub, sub.obsm[obsm_key].astype(np.float32)

    sc_adata,          sc_emb          = _get("subsampled")
    bulk_adata,        bulk_emb        = _get("bulk")
    paired_sc_adata,   paired_sc_emb   = _get("paired_sc")
    paired_pb_adata,   paired_pb_emb   = _get("paired_pb")
    paired_bulk_adata, paired_bulk_emb = _get("paired_bulk")

    metrics: dict = {}

    # ── Load model (needed for reconstruction + synthetic aggregation) ────
    model = None
    if ckpt_path is not None:
        log.info("  Loading model from %s ...", ckpt_path.name)
        try:
            model = _load_model(ckpt_path, device)
        except Exception:
            log.error("  Failed to load model:\n%s", traceback.format_exc())

    # ── Metric 1: reconstruction ──────────────────────────────────────────
    if model is not None and sc_adata is not None:
        log.info("  Computing reconstruction metrics (%d SC cells) ...", sc_adata.n_obs)
        metrics.update(
            compute_reconstruction_metrics(
                model, sc_adata,
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                n_cells=min(sc_adata.n_obs, 1000),
                seed=seed,
                normalized=normalized,
            )
        )

    # ── Metric 2: paired alignment ────────────────────────────────────────
    if paired_pb_emb is not None and paired_bulk_emb is not None:
        pb_pairs   = paired_pb_adata.obs.get("paired",   None)
        bulk_pairs = paired_bulk_adata.obs.get("paired", None)
        if pb_pairs is not None and bulk_pairs is not None:
            log.info("  Computing paired alignment metrics ...")
            metrics.update(
                compute_paired_alignment_metrics(
                    paired_pb_emb, paired_bulk_emb,
                    pb_pair_ids=np.asarray(pb_pairs),
                    bulk_pair_ids=np.asarray(bulk_pairs),
                )
            )
        else:
            log.warning("  'paired' column missing — skipping paired alignment.")

    # ── Metric 3a: aggregation consistency from paired data ───────────────
    if (
        paired_sc_emb is not None
        and paired_pb_emb is not None
        and "paired" in paired_sc_adata.obs.columns
        and "paired" in paired_pb_adata.obs.columns
    ):
        log.info("  Computing aggregation consistency (paired) ...")
        metrics.update(
            _agg_from_paired_sc(
                paired_sc_adata, paired_sc_emb,
                paired_pb_adata, paired_pb_emb,
            )
        )

    # ── Metric 3b: aggregation consistency from synthetic pseudobulks ─────
    if model is not None and sc_adata is not None and group_column in sc_adata.obs.columns:
        log.info("  Computing aggregation consistency (synthetic) ...")
        metrics.update(
            compute_aggregation_metrics(
                model, sc_adata,
                group_column=group_column,
                n_sc_per_pb=n_sc_per_pb,
                n_pb=n_synth_pb,
                batch_size=batch_size,
                seed=seed,
                normalized=normalized,
            )
        )

    # ── Metric 4: contrastive / distributional alignment ──────────────────
    pb_for_contrast   = paired_pb_emb
    bulk_for_contrast = bulk_emb if bulk_emb is not None else paired_bulk_emb

    if pb_for_contrast is not None and bulk_for_contrast is not None:
        log.info("  Computing contrastive alignment metrics ...")
        metrics.update(compute_contrastive_metrics(bulk_for_contrast, pb_for_contrast, seed=seed))

    # ── Save ──────────────────────────────────────────────────────────────
    if ckpt_path is not None:
        metrics["checkpoint"] = str(ckpt_path)

    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2)
    log.info("  Metrics → %s", metrics_file)

    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Ablation-level evaluation
# ---------------------------------------------------------------------------

def run_ablation(
    ablation_dir: Path,
    eval_adata: ad.AnnData,
    batch_size: int,
    mask_ratio: float,
    n_sc_per_pb: int,
    n_synth_pb: int,
    group_column: str,
    device: str,
    seed: int,
    skip_existing: bool,
    normalized: bool,
) -> None:
    model_dirs = sorted(
        d for d in ablation_dir.iterdir()
        if d.is_dir() and f"X_cf_{d.name}" in eval_adata.obsm
    )
    if not model_dirs:
        log.error(
            "No matching model dirs found. eval_adata obsm keys: %s",
            list(eval_adata.obsm.keys()),
        )
        return

    log.info("Found %d model(s): %s", len(model_dirs), [d.name for d in model_dirs])

    all_metrics: list[dict] = []
    for model_dir in model_dirs:
        ckpt = _find_best_ckpt(model_dir)
        log.info("[%s]%s", model_dir.name, f" ckpt: {ckpt.name}" if ckpt else " (no ckpt — skipping recon/agg_synth)")
        m = run_single_model(
            model_name=model_dir.name,
            eval_adata=eval_adata,
            out_dir=model_dir / "metrics",
            ckpt_path=ckpt,
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            n_sc_per_pb=n_sc_per_pb,
            n_synth_pb=n_synth_pb,
            group_column=group_column,
            device=device,
            seed=seed,
            skip_existing=skip_existing,
            normalized=normalized,
        )
        if m:
            m["model"] = model_dir.name
            all_metrics.append(m)

    if not all_metrics:
        log.warning("No metrics collected.")
        return

    df = pd.DataFrame(all_metrics).set_index("model")
    csv_path = ablation_dir / "unified_metrics.csv"
    df.to_csv(csv_path)
    log.info("Summary CSV → %s", csv_path)
    _plot_metrics(df, ablation_dir / "unified_metrics.png")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metrics(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_meta = [
        ("recon_pearson_r",                    "Reconstruction\nPearson R ↑"),
        ("recon_mse",                          "Reconstruction\nMSE ↓"),
        ("paired_cosine_sim_mean",             "Paired Alignment\nCosine Sim ↑"),
        ("paired_rank_mean",                   "Paired Alignment\nRank ↓"),
        ("agg_paired_cosine_pb_to_mean_sc",    "Agg Consistency\n(paired) ↑"),
        ("agg_synth_cosine_pb_to_mean_sc",     "Agg Consistency\n(synth) ↑"),
        ("contrastive_cross_cosine_mean",      "Contrastive\nCross Cosine ↑"),
        ("contrastive_mmd",                    "Contrastive\nMMD ↓"),
    ]
    available = [(col, lbl) for col, lbl in metric_meta if col in df.columns]
    if not available:
        log.warning("No plottable metrics found.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5), squeeze=False)
    axes = axes[0]
    models = df.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(models), 1)))

    for ax, (col, lbl) in zip(axes, available):
        vals = df[col].to_numpy(dtype=float)
        ax.bar(range(len(models)), vals, color=colors[: len(models)])
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Unified FM evaluation metrics", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot → %s", out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified FM evaluation metrics (reads embeddings from eval AnnData).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--eval-adata", type=Path, required=True,
                   help="Pre-built evaluation AnnData from build_eval_adata.py.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ckpt", type=Path, default=None,
                      help="Checkpoint for a single model (used for reconstruction metric).")
    mode.add_argument("--ablation-dir", type=Path, default=None,
                      help="Ablation root; evaluates every model sub-directory.")

    p.add_argument("--name", type=str, default=None,
                   help="Model name (obsm key = X_cf_<name>). "
                        "Defaults to the checkpoint's parent directory name. "
                        "Ignored in ablation mode.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (single-model mode; defaults to <ckpt-dir>/metrics).")

    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--mask-ratio",   type=float, default=0.15,
                   help="Fraction of genes masked for reconstruction (default: 0.15).")
    p.add_argument("--n-sc-per-pb",  type=int,   default=10,
                   help="SC cells per synthetic pseudobulk (default: 10).")
    p.add_argument("--n-synth-pb",   type=int,   default=200,
                   help="Number of synthetic pseudobulks for aggregation (default: 200).")
    p.add_argument("--group-column", type=str,   default="tissue_general",
                   help="obs column for SC grouping (synthetic aggregation).")

    p.add_argument("--device", type=str, default=None,
                   help="Inference device (default: auto-detect cuda/cpu).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models that already have unified_metrics.json.")
    p.add_argument("--not-normalized", action="store_true",
                   help="h5ad files contain raw counts (not log1p-normalised).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_path = args.eval_adata.expanduser().resolve()
    if not eval_path.exists():
        log.error("--eval-adata not found: %s", eval_path)
        return 1

    log.info("Loading eval AnnData from %s ...", eval_path)
    eval_adata = ad.read_h5ad(eval_path)
    log.info("  %d cells, obsm keys: %s", eval_adata.n_obs, list(eval_adata.obsm.keys()))

    normalized = not args.not_normalized
    kwargs = dict(
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        n_sc_per_pb=args.n_sc_per_pb,
        n_synth_pb=args.n_synth_pb,
        group_column=args.group_column,
        device=args.device,
        seed=args.seed,
        skip_existing=args.skip_existing,
        normalized=normalized,
    )

    if args.ablation_dir is not None:
        ablation_dir = args.ablation_dir.expanduser().resolve()
        if not ablation_dir.is_dir():
            log.error("--ablation-dir not found: %s", ablation_dir)
            return 1
        run_ablation(ablation_dir=ablation_dir, eval_adata=eval_adata, **kwargs)

    else:
        ckpt = args.ckpt.expanduser().resolve()
        if not ckpt.exists():
            log.error("Checkpoint not found: %s", ckpt)
            return 1
        model_name = args.name if args.name else ckpt.parent.name
        out_dir = (args.out_dir or ckpt.parent / "metrics").expanduser().resolve()

        metrics = run_single_model(
            model_name=model_name,
            eval_adata=eval_adata,
            out_dir=out_dir,
            ckpt_path=ckpt,
            **kwargs,
        )
        if metrics:
            print("\nMetrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif not isinstance(v, str):
                    print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
