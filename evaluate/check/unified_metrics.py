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

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import gc
import json
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

try:
    from cancerfoundation.model.model import CancerFoundation
    from utils_config import LossType
except ImportError:
    log.warning("Could not load CancerFoundation")

from evaluate.utils import generate_pseudobulk_adata

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

def _bin_to_expr(bin_idx: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map integer bin indices → log1p expression values using per-cell quantile edges.

    bin 0       → 0.0  (zero-expression genes)
    bin k > 0   → edges[k-1]  (left quantile boundary of that bin)

    When edges are repeated (edges[k-1] == edges[j-1]), different bin indices
    that share the same edge value map to the same expression level, so the
    resulting error contribution is correctly 0.
    """
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    out = np.zeros(len(bin_idx), dtype=np.float32)
    pos = bin_idx > 0
    if pos.any():
        k = bin_idx[pos]
        # Clip to valid edge range (bin k maps to edges[k-1], max index = len(edges)-1)
        k_clipped = np.clip(k - 1, 0, len(edges) - 1)
        out[pos] = edges[k_clipped]
    return out


def _recon_metrics_from_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    bin_edges: np.ndarray | None = None,
    orig_expr: np.ndarray | None = None,  # retained for cache compat; unused
    n_bins: int | None = None,
) -> dict:
    """Compute reconstruction metrics from cached (pred, target, mask) arrays.

    MAE is computed in bin space.  When ``bin_edges`` is provided, the
    repeated-edge correction is applied: if pred_bin and target_bin map to the
    same quantile edge value (i.e. the same expression level), their error
    contribution is set to 0 instead of |pred_bin - target_bin|.

    Falls back to plain bin-space MAE when ``bin_edges`` is absent (older caches).

    Per-5-bin MAE is accumulated for the line plot (``recon_mae_per_bin``).
    """
    use_edges = bin_edges is not None

    pearson_rs: list[float] = []
    mae_vals:   list[float] = []
    # per-bin-group accumulator: window_start (0, 5, 10, …) → list of per-gene errors
    per_bin: dict[int, list[float]] = {}

    for i in range(len(pred)):
        m = mask[i]
        if m.sum() < 2:
            continue

        p = pred[i][m].astype(np.float32)
        t = target[i][m].astype(np.float32)

        # Pearson R (bin space)
        if np.std(p) >= 1e-8 and np.std(t) >= 1e-8:
            r = float(np.corrcoef(p, t)[0, 1])
            if not np.isnan(r):
                pearson_rs.append(r)

        # De-normalise bin indices if targets are in [0,1] (normalise_bins=True)
        if n_bins is not None and t.max() <= 1.0 and t.max() > 0:
            p_bins = np.round(p * n_bins).astype(np.int64)
            t_bins = np.round(t * n_bins).astype(np.int64)
        else:
            p_bins = np.round(p).astype(np.int64)
            t_bins = t.astype(np.int64)

        abs_err = np.abs(p_bins - t_bins).astype(np.float32)

        # Repeated-edge correction: zero out errors where pred and target share
        # the same quantile edge value (same expression → not a real error)
        if use_edges:
            edges = bin_edges[i]
            p_expr = _bin_to_expr(p_bins, edges)
            t_expr = _bin_to_expr(t_bins, edges)
            abs_err[p_expr == t_expr] = 0.0

        mae_vals.append(float(abs_err.mean()))

        # Per-5-bin accumulation (stratify by true bin index)
        for err_val, tb in zip(abs_err, t_bins):
            window = int((int(tb) // 5) * 5)
            per_bin.setdefault(window, []).append(float(err_val))

    per_bin_means = {
        str(w + 2): float(np.mean(v))   # key = centre of each 5-bin window
        for w, v in sorted(per_bin.items()) if v
    }

    return {
        "recon_pearson_r":     float(np.mean(pearson_rs)) if pearson_rs else float("nan"),
        "recon_pearson_r_std": float(np.std(pearson_rs))  if pearson_rs else float("nan"),
        "recon_mae_bins":      float(np.mean(mae_vals))   if mae_vals   else float("nan"),
        "recon_mae_per_bin":   per_bin_means,
        "recon_n_cells":       len(pearson_rs),
    }


@torch.no_grad()
def compute_reconstruction_metrics(
    model: CancerFoundation,
    adata: ad.AnnData,
    batch_size: int = 64,
    mask_ratio: float = 0.15,
    n_cells: int = 1000,
    seed: int = 0,
    normalized: bool = True,
    loss: LossType | None = None,
) -> dict:
    """Pearson R and mean absolute bin error between predicted and actual expression at masked positions."""
    device = next(model.model.parameters()).device
    rng = np.random.default_rng(seed)

    data = model.preprocess_for_embedding(adata, normalized=normalized)
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

    effective_loss = loss if loss is not None else model.loss_type

    pearson_rs: list[float] = []
    mae_vals: list[float] = []

    for start in range(0, n, batch_size):
        batch_expr = torch.from_numpy(X[start : start + batch_size]).to(device)
        bs = batch_expr.shape[0]
        batch_genes = gene_ids.unsqueeze(0).expand(bs, -1)

        # Prepend CLS token
        cls_g = torch.full((bs, 1), model.cls_token_id, dtype=torch.long, device=device)
        cls_e = torch.full((bs, 1), float(model.pad_value), device=device)
        genes_full = torch.cat([cls_g, batch_genes], dim=1)
        expr_full  = torch.cat([cls_e, batch_expr],  dim=1)

        gene_mask = (torch.rand(bs, n_genes, device=device) < mask_ratio)
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
        raw_pred = decoder_out["pred"]

        if effective_loss == LossType.CORN:
            logits = raw_pred[:, 1:, :]
            pred = (torch.cumprod(torch.sigmoid(logits), -1) > 0.5).sum(-1)
        elif effective_loss == LossType.ORDINALCROSSENTROPY:
            pred = torch.sigmoid(raw_pred[:, 1:, :]).sum(dim=-1)
        else:
            pred = raw_pred[:, 1:, 0] if raw_pred.dim() == 3 else raw_pred[:, 1:]

        target = batch_expr
        for i in range(bs):
            m = gene_mask[i]
            if m.sum() < 2:
                continue
            p = pred[i][m].float().cpu().numpy()
            t = target[i][m].float().cpu().numpy()
            mae_vals.append(float(np.abs(p - t).mean()))
            if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                continue
            r = float(np.corrcoef(p, t)[0, 1])
            if not np.isnan(r):
                pearson_rs.append(r)

    return {
        "recon_pearson_r":     float(np.mean(pearson_rs)) if pearson_rs else float("nan"),
        "recon_pearson_r_std": float(np.std(pearson_rs))  if pearson_rs else float("nan"),
        "recon_mae_bins":      float(np.mean(mae_vals))   if mae_vals   else float("nan"),
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
    """Cosine similarity, L2 distance, and retrieval rank between matched pseudobulk–bulk pairs."""
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

    # ── Cosine similarity (unit-sphere projected) ──────────────────────────
    pb_n   = pb_arr   / (np.linalg.norm(pb_arr,   axis=1, keepdims=True) + 1e-8)
    bulk_n = bulk_arr / (np.linalg.norm(bulk_arr, axis=1, keepdims=True) + 1e-8)

    cos_sim = pb_n @ bulk_n.T
    paired_sims = np.diag(cos_sim)
    ranks_cos = [int((cos_sim[i] > cos_sim[i, i]).sum()) + 1 for i in range(N)]

    off = ~np.eye(N, dtype=bool)

    # ── L2 distance (raw embedding space) ─────────────────────────────────
    # Pairwise squared distances via ||a-b||² = ||a||² + ||b||² - 2a·b
    sq_a = (pb_arr   ** 2).sum(axis=1)
    sq_b = (bulk_arr ** 2).sum(axis=1)
    sq_dist_mat = np.clip(sq_a[:, None] + sq_b[None, :] - 2 * (pb_arr @ bulk_arr.T), 0, None)
    l2_mat = np.sqrt(sq_dist_mat)
    paired_l2 = np.diag(l2_mat)
    # rank by L2: 1 = matched pair is the nearest bulk neighbour for this PB
    ranks_l2 = [int((l2_mat[i] < l2_mat[i, i]).sum()) + 1 for i in range(N)]

    return {
        "paired_cosine_sim_mean":        float(paired_sims.mean()),
        "paired_cosine_sim_std":         float(paired_sims.std()),
        "paired_rank_mean":              float(np.mean(ranks_cos)),
        "paired_rank_median":            float(np.median(ranks_cos)),
        "paired_l2_mean":                float(paired_l2.mean()),
        "paired_l2_std":                 float(paired_l2.std()),
        "paired_rank_l2_mean":           float(np.mean(ranks_l2)),
        "paired_rank_l2_median":         float(np.median(ranks_l2)),
        "paired_n_pairs":                N,
        "paired_random_baseline_cosine": float(cos_sim[off].mean()) if N > 1 else float("nan"),
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
    l2s:  list[float] = []
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
        l2s.append(float(np.linalg.norm(pb_e - mean_sc)))

    if not sims:
        return {}
    return {
        "agg_paired_cosine_pb_to_mean_sc":     float(np.mean(sims)),
        "agg_paired_cosine_pb_to_mean_sc_std": float(np.std(sims)),
        "agg_paired_l2_pb_to_mean_sc":         float(np.mean(l2s)),
        "agg_paired_l2_pb_to_mean_sc_std":     float(np.std(l2s)),
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
    l2s:  list[float] = []

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
        l2s.append(float(np.linalg.norm(pb_emb - mean_sc)))

    if not sims:
        return {}
    return {
        "agg_synth_cosine_pb_to_mean_sc":     float(np.mean(sims)),
        "agg_synth_cosine_pb_to_mean_sc_std": float(np.std(sims)),
        "agg_synth_l2_pb_to_mean_sc":         float(np.mean(l2s)),
        "agg_synth_l2_pb_to_mean_sc_std":     float(np.std(l2s)),
        "agg_synth_n_pseudobulks":            len(sims),
    }


def _agg_metrics_from_precomputed(
    sc_emb: np.ndarray,
    sc_groups: np.ndarray,
    synth_pb_emb: np.ndarray,
    synth_pb_groups: np.ndarray,
    n_sc_per_pb: int = 10,
    seed: int = 0,
) -> dict:
    """Aggregation consistency from precomputed SC and synth_pb embeddings.

    For each synth_pb, randomly samples ``n_sc_per_pb`` SC embeddings from
    the same group, takes their mean, and measures cosine similarity and L2
    distance to the pseudobulk embedding.  Returns the same keys as
    ``compute_aggregation_metrics``.
    """
    rng = np.random.default_rng(seed)

    group_to_sc_idx: dict[str, np.ndarray] = {}
    for g in np.unique(sc_groups):
        group_to_sc_idx[g] = np.where(sc_groups == g)[0]

    sims: list[float] = []
    l2s:  list[float] = []

    for pb_emb_i, g in zip(synth_pb_emb, synth_pb_groups):
        sc_idx = group_to_sc_idx.get(str(g))
        if sc_idx is None or len(sc_idx) == 0:
            continue
        chosen  = rng.choice(sc_idx, size=min(n_sc_per_pb, len(sc_idx)),
                             replace=len(sc_idx) < n_sc_per_pb)
        mean_sc = sc_emb[chosen].mean(axis=0)

        pb_n = pb_emb_i / (np.linalg.norm(pb_emb_i) + 1e-8)
        sc_n = mean_sc   / (np.linalg.norm(mean_sc)  + 1e-8)
        sims.append(float(np.dot(pb_n, sc_n)))
        l2s.append(float(np.linalg.norm(pb_emb_i - mean_sc)))

    if not sims:
        return {}
    return {
        "agg_synth_cosine_pb_to_mean_sc":     float(np.mean(sims)),
        "agg_synth_cosine_pb_to_mean_sc_std": float(np.std(sims)),
        "agg_synth_l2_pb_to_mean_sc":         float(np.mean(l2s)),
        "agg_synth_l2_pb_to_mean_sc_std":     float(np.std(l2s)),
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
    # Sets the kernel width to the median pairwise distance of the combined cloud
    XY   = np.vstack([X, Y])
    diff = XY[:, None, :] - XY[None, :, :]
    sq   = (diff ** 2).sum(axis=2)
    pos  = sq[sq > 0]
    bw   = float(np.sqrt(np.median(pos) / 2)) if len(pos) > 0 else 1.0
    return _rbf_kernel(X, X, bw) + _rbf_kernel(Y, Y, bw) - 2 * _rbf_kernel(X, Y, bw)


def _sliced_wasserstein(
    X: np.ndarray, 
    Y: np.ndarray,
    project: bool = False,
    n_projections: int = 50, 
    seed: int = 0
) -> float:
    """Wasserstein distance between X and Y. Optionally projects onto random 1D directions and averages the 1D Wasserstein distances.
    """
    from scipy.stats import wasserstein_distance

    if project:
        rng = np.random.default_rng(seed)
        directions = rng.standard_normal((n_projections, X.shape[1]))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        dists = [wasserstein_distance(X @ d, Y @ d) for d in directions]
        return float(np.mean(dists))
    else:
        return float(wasserstein_distance(X.flatten(), Y.flatten()))


def _pairwise_l2_mean(X: np.ndarray) -> float:
    """Mean pairwise L2 distance between all rows of X, excluding self-pairs."""
    if len(X) < 2:
        return float("nan")
    sq = (X ** 2).sum(axis=1)
    sq_dist = np.clip(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0, None)
    mask = ~np.eye(len(X), dtype=bool)
    return float(np.sqrt(sq_dist[mask]).mean())


def _cross_l2_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean L2 distance between all pairs (x_i, y_j) across two sets."""
    sq_x = (X ** 2).sum(axis=1)
    sq_y = (Y ** 2).sum(axis=1)
    sq_dist = np.clip(sq_x[:, None] + sq_y[None, :] - 2 * (X @ Y.T), 0, None)
    return float(np.sqrt(sq_dist).mean())


def compute_contrastive_metrics(
    bulk_emb: np.ndarray,
    pb_emb: np.ndarray,
    n_max: int = 500,
    seed: int = 0,
) -> dict:
    """Cosine similarity, L2 distances, MMD, and Sliced Wasserstein between bulk and pseudobulk distributions."""
    rng = np.random.default_rng(seed)
    if len(bulk_emb) > n_max:
        bulk_emb = bulk_emb[rng.choice(len(bulk_emb), n_max, replace=False)]
    if len(pb_emb) > n_max:
        pb_emb = pb_emb[rng.choice(len(pb_emb), n_max, replace=False)]

    bn = bulk_emb / (np.linalg.norm(bulk_emb, axis=1, keepdims=True) + 1e-8)
    pn = pb_emb   / (np.linalg.norm(pb_emb,   axis=1, keepdims=True) + 1e-8)

    def _within_cos(e: np.ndarray) -> float:
        if len(e) < 2:
            return float("nan")
        s = e @ e.T
        return float(s[~np.eye(len(e), dtype=bool)].mean())

    return {
        # cosine-based (unit-sphere projected)
        "contrastive_cross_cosine_mean":   float((bn @ pn.T).mean()),
        "contrastive_within_bulk_cosine":  _within_cos(bn),
        "contrastive_within_pb_cosine":    _within_cos(pn),
        # L2-based (raw embedding space)
        "contrastive_cross_l2_mean":       _cross_l2_mean(bulk_emb, pb_emb),
        "contrastive_within_bulk_l2":      _pairwise_l2_mean(bulk_emb),
        "contrastive_within_pb_l2":        _pairwise_l2_mean(pb_emb),
        # distributional
        "contrastive_mmd":                 _mmd(bulk_emb, pb_emb),
        "contrastive_wasserstein":         _sliced_wasserstein(bulk_emb, pb_emb, seed=seed),
        "contrastive_n_bulk":              len(bulk_emb),
        "contrastive_n_pb":                len(pb_emb),
    }


# ---------------------------------------------------------------------------
# Metric 5: scIB batch integration benchmark (ablation-level, all models at once)
# ---------------------------------------------------------------------------

def run_scib_benchmark(
    eval_adata: ad.AnnData,
    model_names: list[str],
    group_column: str = "tissue_general",
    n_max: int = 500,
    out_dir: Path | None = None,
    seed: int = 0,
) -> None:
    """Run four scIB batch integration benchmarks across all models.

    Synthetic pseudobulks (``_eval_modality = "synth_pb"``) are expected to be
    already present in ``eval_adata`` with their embeddings pre-computed by
    ``build_eval_adata.py``; no model loading happens here.

    Benchmarks
    ----------
    1. **bulk_vs_pb** — all bulk (bulk + paired_bulk) vs all pseudobulk
       (paired_pb + synth_pb).  batch_key = "modality_type".
    2. **paired_vs_nonpaired** — paired samples (paired_bulk + paired_pb) vs
       non-paired samples (bulk + synth_pb).  batch_key = "pairing".
    3. **paired_bulk_vs_paired_pb** — paired_bulk vs paired_pb only.
       batch_key = "modality_type".
    4. **nonpaired_bulk_vs_synth_pb** — non-paired bulk vs synth_pb only.
       batch_key = "modality_type".
    """
    try:
        from scib_metrics.benchmark import Benchmarker, BatchCorrection
    except ImportError:
        log.warning("scib_metrics not installed — skipping batch integration benchmark.")
        return

    rng = np.random.default_rng(seed)

    # ── Modality masks ──────────────────────────────────────────────────────
    mod_col = eval_adata.obs.get(
        _MODALITY_COL, pd.Series("", index=eval_adata.obs_names, dtype=str)
    )
    bulk_mask        = (mod_col == "bulk").values
    paired_bulk_mask = (mod_col == "paired_bulk").values
    paired_pb_mask   = (mod_col == "paired_pb").values
    synth_pb_mask    = (mod_col == "synth_pb").values
    all_bulk_mask    = bulk_mask | paired_bulk_mask
    all_pb_mask      = paired_pb_mask | synth_pb_mask

    # ── Helper: subsample a boolean mask to at most n indices ───────────────
    def _sample_idx(mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        return np.sort(idx)

    # ── Helper: build a minimal AnnData for scIB from eval_adata subsets ───
    # part_specs: list of (row_indices_into_eval_adata, batch_label)
    def _build_sub(
        part_specs: list[tuple[np.ndarray, str]],
        batch_col: str,
    ) -> ad.AnnData | None:
        obs_rows: list[dict] = []
        emb_accum: dict[str, list[np.ndarray]] = {}
        emb_counts: dict[str, int] = {}

        for idx, label in part_specs:
            if len(idx) == 0:
                continue
            grp = (
                eval_adata.obs[group_column].iloc[list(idx)].astype(str).values
                if group_column in eval_adata.obs.columns
                else np.full(len(idx), "unknown")
            )
            for g in grp:
                obs_rows.append({group_column: g, batch_col: label})
            for name in model_names:
                key = f"X_cf_{name}"
                if key in eval_adata.obsm:
                    emb_accum.setdefault(key, []).append(eval_adata.obsm[key][list(idx)])
                    emb_counts[key] = emb_counts.get(key, 0) + len(idx)
            if "X_pca" in eval_adata.obsm:
                emb_accum.setdefault("X_pca", []).append(eval_adata.obsm["X_pca"][list(idx)])
                emb_counts["X_pca"] = emb_counts.get("X_pca", 0) + len(idx)

        if not obs_rows:
            return None

        n_total = len(obs_rows)
        sub = ad.AnnData(obs=pd.DataFrame(obs_rows))
        for key, arrs in emb_accum.items():
            if emb_counts[key] == n_total:
                sub.obsm[key] = np.vstack(arrs).astype(np.float32)

        return sub if sub.obsm else None

    # ── Helper: run one Benchmarker and save results ────────────────────────
    def _run_one(sub: ad.AnnData, batch_col: str, tag: str) -> None:
        sub.obs[group_column] = sub.obs[group_column].fillna("unknown").astype(str)
        embedding_keys = [k for k in sub.obsm if k.startswith("X_cf_")]
        if not embedding_keys:
            log.warning("[scIB %s] No valid embedding keys — skipping.", tag)
            return
        if sub.obs[batch_col].nunique() < 2:
            log.warning("[scIB %s] Only 1 batch value — skipping.", tag)
            return

        log.info(
            "[scIB %s] %d cells, %d models, batch_key='%s' %s ...",
            tag, sub.n_obs, len(embedding_keys), batch_col,
            dict(sub.obs[batch_col].value_counts().items()),
        )

        print(f"sub.n_vars: {sub.n_vars}")
        print(f"sub.n_obs: {sub.n_obs}")
        # print(f"sub.X: {sub.X.shape}")

        pre_integrated_key = "X_pca" if "X_pca" in sub.obsm else None

        bm = Benchmarker(
            sub,
            batch_key=batch_col,
            label_key=group_column,
            embedding_obsm_keys=embedding_keys,
            bio_conservation_metrics=None,
            batch_correction_metrics=BatchCorrection(
                bras=True,
                ilisi_knn=True,
                pcr_comparison=pre_integrated_key is not None,
                graph_connectivity=False,
            ),
            pre_integrated_embedding_obsm_key=pre_integrated_key,
            n_jobs=1,
        )
        bm.benchmark()
        results = bm.get_results(min_max_scale=False, clean_names=True)
        log.info("[scIB %s] complete.\n%s", tag, results.to_string())

        if out_dir is not None:
            scib_out_dir = out_dir / "_scib_metrics"
            scib_out_dir.mkdir(parents=True, exist_ok=True)
            results.to_csv(scib_out_dir / f"scib_{tag}.csv")
            png_dir = scib_out_dir / f"scib_{tag}"
            png_dir.mkdir(parents=True, exist_ok=True)
            bm.plot_results_table(min_max_scale=False, show=False, save_dir=png_dir)
            log.info("[scIB %s] → %s/", tag, png_dir)

    # ── Benchmark 1: all bulk vs all pseudobulk ─────────────────────────────
    sub1 = _build_sub([
        (_sample_idx(all_bulk_mask, n_max), "bulk"),
        (_sample_idx(all_pb_mask,   n_max), "pseudobulk"),
    ], "modality_type")
    if sub1 is not None:
        _run_one(sub1, "modality_type", "bulk_vs_pb")
    else:
        log.warning("[scIB] bulk_vs_pb: could not build sub-adata — skipping.")

    # ── Benchmark 2: paired vs non-paired ──────────────────────────────────
    paired_mask    = paired_bulk_mask | paired_pb_mask
    nonpaired_mask = bulk_mask | synth_pb_mask
    sub2 = _build_sub([
        (_sample_idx(paired_mask,    n_max), "paired"),
        (_sample_idx(nonpaired_mask, n_max), "non_paired"),
    ], "pairing")
    if sub2 is not None:
        _run_one(sub2, "pairing", "paired_vs_nonpaired")
    else:
        log.warning("[scIB] paired_vs_nonpaired: could not build sub-adata — skipping.")

    # ── Benchmark 3: paired bulk vs paired pseudobulk ──────────────────────
    sub3 = _build_sub([
        (_sample_idx(paired_bulk_mask, n_max), "bulk"),
        (_sample_idx(paired_pb_mask,   n_max), "pseudobulk"),
    ], "modality_type")
    if sub3 is not None:
        _run_one(sub3, "modality_type", "paired_bulk_vs_paired_pb")
    else:
        log.warning("[scIB] paired_bulk_vs_paired_pb: could not build sub-adata — skipping.")

    # ── Benchmark 4: non-paired bulk vs synthetic pseudobulk ───────────────
    sub4 = _build_sub([
        (_sample_idx(bulk_mask,      n_max), "bulk"),
        (_sample_idx(synth_pb_mask,  n_max), "pseudobulk"),
    ], "modality_type")
    if sub4 is not None:
        _run_one(sub4, "modality_type", "nonpaired_bulk_vs_synth_pb")
    else:
        log.warning("[scIB] nonpaired_bulk_vs_synth_pb: could not build sub-adata — skipping.")


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_single_model(
    model_name: str,
    eval_adata: ad.AnnData,
    out_dir: Path,
    ckpt_path: Path | None = None,
    batch_size: int = 64,
    mask_ratio: float = 0.4,
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

    metrics: dict = {}

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
    synth_pb_adata,    synth_pb_emb    = _get("synth_pb")

    # ── Metric 1: reconstruction ──────────────────────────────────────────
    recon_cache_key = f"recon_{model_name}"
    if recon_cache_key in eval_adata.uns:
        log.info("  Using cached reconstruction arrays from eval_adata.uns ...")
        c = eval_adata.uns[recon_cache_key]
        metrics.update(_recon_metrics_from_arrays(
            c["pred"], c["target"], c["mask"],
            bin_edges=c.get("bin_edges"),
            orig_expr=c.get("orig_expr"),
            n_bins=c.get("n_bins"),
        ))
        recon_done = True
    else:
        recon_done = False

    # ── Load model if still needed (reconstruction not cached, or synth agg) ─
    has_precomputed_agg = (
        synth_pb_emb is not None
        and sc_emb is not None
        and sc_adata is not None
        and synth_pb_adata is not None
        and group_column in sc_adata.obs.columns
        and group_column in synth_pb_adata.obs.columns
    )
    model = None
    need_model = (not recon_done and bulk_adata is not None) or (
        not has_precomputed_agg
        and sc_adata is not None
        and group_column in sc_adata.obs.columns
    )
    if need_model and ckpt_path is not None:
        log.info("  Loading model from %s ...", ckpt_path.name)
        try:
            model = _load_model(ckpt_path, device)
        except Exception:
            log.error("  Failed to load model:\n%s", traceback.format_exc())

    if not recon_done and model is not None and bulk_adata is not None:
        log.info("  Computing reconstruction metrics (%d bulk cells) ...", bulk_adata.n_obs)
        metrics.update(
            compute_reconstruction_metrics(
                model, bulk_adata,
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                n_cells=min(bulk_adata.n_obs, 1000),
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
    if has_precomputed_agg:
        log.info("  Computing aggregation consistency from precomputed synth_pb embeddings ...")
        metrics.update(
            _agg_metrics_from_precomputed(
                sc_emb=sc_emb,
                sc_groups=sc_adata.obs[group_column].astype(str).to_numpy(),
                synth_pb_emb=synth_pb_emb,
                synth_pb_groups=synth_pb_adata.obs[group_column].astype(str).to_numpy(),
                n_sc_per_pb=n_sc_per_pb,
                seed=seed,
            )
        )
    elif model is not None and sc_adata is not None and group_column in sc_adata.obs.columns:
        log.info("  Computing aggregation consistency (synthetic, live embedding) ...")
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

    # print(metrics)

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
    do_scib: bool = False,
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
    # recon_mae_per_bin is a nested dict — keep it out of the CSV
    per_bin_col = "recon_mae_per_bin"
    per_model_per_bin: dict[str, dict] = {}
    for row in all_metrics:
        name = row.get("model", "")
        if per_bin_col in row and isinstance(row[per_bin_col], dict):
            per_model_per_bin[name] = row[per_bin_col]
    if per_bin_col in df.columns:
        df = df.drop(columns=[per_bin_col])

    csv_path = ablation_dir / "unified_metrics.csv"
    df.to_csv(csv_path)
    log.info("Summary CSV → %s", csv_path)
    _plot_metrics(df, ablation_dir / "unified_metrics.png")

    if per_model_per_bin:
        plot_reconstruction_error(per_model_per_bin, ablation_dir / "reconstruction_error.png")

    if do_scib:
        run_scib_benchmark(
            eval_adata,
            model_names=[d.name for d in model_dirs],
            group_column=group_column,
            out_dir=ablation_dir,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstruction_error(
    per_model_per_bin: dict[str, dict],
    out_path: Path,
) -> None:
    """Line plot of per-5-bin edge-aware MAE for all models on one axes.

    Args:
        per_model_per_bin: ``{model_name: {bin_centre_str: mean_mae, ...}, ...}``
        out_path: destination PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, bin_dict in per_model_per_bin.items():
        if not bin_dict:
            continue
        xs = sorted(int(k) for k in bin_dict)
        ys = [bin_dict[str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", markersize=4, label=model_name)

    ax.set_xlabel("True bin index (group centre, every 5 bins)")
    ax.set_ylabel("Mean MAE (bins, edge-aware)")
    ax.set_title("Reconstruction error by bin group")
    if per_model_per_bin:
        ax.legend(fontsize="small", loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Reconstruction error line plot → %s", out_path)


def _plot_metrics(df: pd.DataFrame, out_png: Path, ncols: int = 5) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_meta = [
        # ── Reconstruction ─────────────────────────────────────────────────
        ("recon_pearson_r",                    "Reconstruction\nPearson R ↑"),
        ("recon_mae_bins",                     "Reconstruction\nMAE (bins, edge-aware) ↓"),
        # ── Paired alignment — cosine ───────────────────────────────────────
        ("paired_cosine_sim_mean",             "Paired Cosine Sim ↑"),
        ("paired_rank_mean",                   "Paired Rank\n(cosine) ↓"),
        # ── Paired alignment — L2 ──────────────────────────────────────────
        ("paired_l2_mean",                     "Paired L2 Dist ↓"),
        ("paired_rank_l2_mean",                "Paired Rank\n(L2) ↓"),
        # ── Aggregation — cosine ───────────────────────────────────────────
        ("agg_paired_cosine_pb_to_mean_sc",    "Agg Consistency\n(paired, cosine) ↑"),
        ("agg_synth_cosine_pb_to_mean_sc",     "Agg Consistency\n(synth, cosine) ↑"),
        # ── Aggregation — L2 ───────────────────────────────────────────────
        ("agg_paired_l2_pb_to_mean_sc",        "Agg Consistency\n(paired, L2) ↓"),
        ("agg_synth_l2_pb_to_mean_sc",         "Agg Consistency\n(synth, L2) ↓"),
        # ── Contrastive — cosine ────────────────────────────────────────────
        ("contrastive_cross_cosine_mean",      "Cross Cosine\nSim ↑"),
        ("contrastive_within_bulk_cosine",     "Within-Bulk\nCosine Sim"),
        ("contrastive_within_pb_cosine",       "Within-PB\nCosine Sim"),
        # ── Contrastive — L2 ───────────────────────────────────────────────
        ("contrastive_cross_l2_mean",          "Cross L2 Dist ↓"),
        ("contrastive_within_bulk_l2",         "Within-Bulk\nL2 Spread ↑"),
        ("contrastive_within_pb_l2",           "Within-PB\nL2 Spread ↑"),
        # ── Distributional ─────────────────────────────────────────────────
        ("contrastive_mmd",                    "MMD ↓"),
        ("contrastive_wasserstein",            "Wasserstein ↓"),
    ]

    # Groups of columns that must share the same Y-axis for direct comparison.
    # Each entry is a frozenset of column names; any subset that appears in the
    # data will have its Y limits unified.
    _SHARED_YLIM_GROUPS: list[frozenset] = [
        frozenset({
            "paired_rank_mean",
            "paired_rank_mean",
            "paired_rank_median",
            "paired_rank_l2_mean",
            "paired_rank_l2_median",
        }),
    ]

    available = [(col, lbl) for col, lbl in metric_meta if col in df.columns]
    if not available:
        log.warning("No plottable metrics found.")
        return

    n = len(available)
    nrows = math.ceil(n / ncols)
    models = df.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(models), 1)))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    _BASELINE_COL = "paired_random_baseline_cosine"
    col_to_ax: dict[str, object] = {}

    for idx, (col, lbl) in enumerate(available):
        row, col_idx = divmod(idx, ncols)
        ax = axes[row][col_idx]
        col_to_ax[col] = ax

        vals = df[col].to_numpy(dtype=float)
        ax.bar(range(len(models)), vals, color=colors[: len(models)])
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if col == "paired_cosine_sim_mean" and _BASELINE_COL in df.columns:
            baseline = df[_BASELINE_COL].to_numpy(dtype=float)
            ax.scatter(
                range(len(models)), baseline,
                color="black", marker="_", s=300, linewidths=2.5,
                zorder=5, label="random baseline",
            )
            ax.legend(fontsize=6, loc="lower right")

    # Apply shared Y-limits across comparable metric groups
    for group in _SHARED_YLIM_GROUPS:
        group_cols = [c for c in group if c in col_to_ax]
        if len(group_cols) < 2:
            continue
        all_vals = np.concatenate([
            df[c].to_numpy(dtype=float) for c in group_cols
        ])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        pad = (hi - lo) * 0.08 if hi > lo else 0.5
        ymin, ymax = lo - pad, hi + pad
        for c in group_cols:
            col_to_ax[c].set_ylim(ymin, ymax)

    # Hide unused subplot cells in the last row
    for idx in range(n, nrows * ncols):
        row, col_idx = divmod(idx, ncols)
        axes[row][col_idx].set_visible(False)

    fig.suptitle("Unified FM evaluation metrics", fontsize=11, fontweight="bold")
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
    p.add_argument("--eval-adata", type=Path, default=None,
                   help="Pre-built evaluation AnnData from build_eval_adata.py. "
                        "Not required when using --plot-csv.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ckpt", type=Path, default=None,
                      help="Checkpoint for a single model (used for reconstruction metric).")
    mode.add_argument("--ablation-dir", type=Path, default=None,
                      help="Ablation root; evaluates every model sub-directory.")
    mode.add_argument("--plot-csv", type=Path, default=None,
                      help="Re-plot directly from an existing unified_metrics.csv "
                           "without recomputing any metrics.")

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
    p.add_argument("--scib", action="store_true",
                   help="Compute scIB batch integration metrics (requires scib + scanpy). "
                        "Intended for a second pass after the main metrics have been cached "
                        "with --skip-existing, using an environment that has scib installed.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # ── Plot-only shortcut ────────────────────────────────────────────────────
    if args.plot_csv is not None:
        csv_path = args.plot_csv.expanduser().resolve()
        if not csv_path.exists():
            log.error("--plot-csv not found: %s", csv_path)
            return 1
        df = pd.read_csv(csv_path, index_col=0)
        out_png = csv_path.with_suffix(".png")
        log.info("Plotting %d models from %s ...", len(df), csv_path)
        _plot_metrics(df, out_png)
        return 0

    if args.eval_adata is None:
        log.error("--eval-adata is required unless using --plot-csv.")
        return 1

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
    # Common kwargs for run_single_model (do_scib is ablation-level only)
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
        run_ablation(ablation_dir=ablation_dir, eval_adata=eval_adata, do_scib=args.scib, **kwargs)

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
