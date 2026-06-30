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

_MODALITY_COL = "_eval_modality"

# (lo_inclusive, hi_exclusive, label_suffix)
_BIN_STRATA = [
    (0,  5,  "0_5"),
    (5,  15, "5_15"),
    (15, 30, "15_30"),
    (30, 51, "30_50"),  # hi=51 to include bin 50
]


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
    strat_mae: dict[str, list[float]] = {s: [] for _, _, s in _BIN_STRATA}

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
            # CORN logits (batch, seq, num_classes-1): decode to expected bin via
            # cumulative product of conditional sigmoid probabilities P(Y>k | Y>=k).
            logits = raw_pred[:, 1:, :]
            hard_pred = (torch.cumprod(torch.sigmoid(logits), -1) > 0.5).sum(-1)  # integer
            soft_pred = torch.cumprod(torch.sigmoid(logits), -1).sum(-1)          # expected
            pred = hard_pred
        elif effective_loss == LossType.ORDINALCROSSENTROPY:
            # Each logit is P(Y > k) directly; expected bin = sum of sigmoids.
            pred = torch.sigmoid(raw_pred[:, 1:, :]).sum(dim=-1)
        else:
            # MSE / ZINB: decoder emits a scalar (or mu as first channel) per gene.
            pred = raw_pred[:, 1:, 0] if raw_pred.dim() == 3 else raw_pred[:, 1:]

        target = batch_expr
        for i in range(bs):
            m = gene_mask[i]
            if m.sum() < 2:
                continue
            p = pred[i][m].float().cpu().numpy()
            t = target[i][m].float().cpu().numpy()
            abs_err = np.abs(p - t)
            mae_vals.append(float(abs_err.mean()))
            for lo, hi, suffix in _BIN_STRATA:
                stratum = (t >= lo) & (t < hi)
                if stratum.any():
                    # print("Adding stratified MAE")
                    strat_mae[suffix].append(float(abs_err[stratum].mean()))
            if np.std(p) < 1e-8 or np.std(t) < 1e-8:
                continue
            r = float(np.corrcoef(p, t)[0, 1])
            if not np.isnan(r):
                pearson_rs.append(r)

    out_dict = {
        "recon_pearson_r":     float(np.mean(pearson_rs)) if pearson_rs else float("nan"),
        "recon_pearson_r_std": float(np.std(pearson_rs))  if pearson_rs else float("nan"),
        "recon_mae_bins":      float(np.mean(mae_vals))   if mae_vals   else float("nan"),
        **{
            f"recon_mae_bins_{s}": float(np.mean(v)) if v else float("nan")
            for _, _, s in _BIN_STRATA
            for v in [strat_mae[s]]
        },
        "recon_n_cells":       len(pearson_rs),
    }

    # print(out_dict)
    return out_dict


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
# Metric 5: scIB batch integration (bulk vs pseudobulk)
# ---------------------------------------------------------------------------

def compute_scib_metrics(
    bulk_emb: np.ndarray,
    pb_emb: np.ndarray,
    bulk_labels: np.ndarray | None = None,
    pb_labels: np.ndarray | None = None,
    label_col: str = "tissue_general",
    n_neighbors: int = 15,
    n_max: int = 2000,
    seed: int = 0,
) -> dict:
    """scIB batch integration metrics treating modality (bulk vs PB) as the batch variable.

    bulk_labels / pb_labels : per-cell biological group (e.g. tissue type).
        Required for silhouette_batch (which conditions on the biological label).
        If omitted, only iLISI is computed.
    """
    try:
        import scib
        import scanpy as sc
    except ImportError:
        log.warning("scib or scanpy not installed — skipping scIB metrics.")
        return {}

    # Downsample if necessary
    rng = np.random.default_rng(seed)
    if len(bulk_emb) > n_max:
        bulk_emb = bulk_emb[rng.choice(len(bulk_emb), n_max, replace=False)]
        if bulk_labels is not None:
            bulk_labels = bulk_labels[rng.choice(len(bulk_labels), n_max, replace=False)]
    if len(pb_emb) > n_max:
        pb_emb = pb_emb[rng.choice(len(pb_emb), n_max, replace=False)]
        if pb_labels is not None:
            pb_labels = pb_labels[rng.choice(len(pb_labels), n_max, replace=False)]

    combined = np.vstack([bulk_emb, pb_emb]).astype(np.float32)
    n_bulk, n_pb = len(bulk_emb), len(pb_emb)
    batch_col = np.array(["bulk"] * n_bulk + ["pb"] * n_pb)

    adata = ad.AnnData(np.zeros((n_bulk + n_pb, 1), dtype=np.float32))
    adata.obs["modality"] = pd.Categorical(batch_col)
    adata.obsm["X_emb"]   = combined

    has_labels = bulk_labels is not None and pb_labels is not None
    if has_labels:
        adata.obs["label"] = pd.Categorical(
            np.concatenate([bulk_labels.astype(str), pb_labels.astype(str)])
        )
    else:
        adata.obs["label"] = adata.obs["modality"].copy()

    try:
        sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)
    except Exception as exc:
        log.warning("scIB: scanpy neighbors failed (%s) — skipping scIB metrics.", exc)
        return {}

    out: dict = {}

    # Batch ASW: 1 - mean |ASW_batch| per label group, scaled to [0,1]. Higher = better mixing.
    if has_labels:
        try:
            asw = scib.metrics.silhouette_batch(
                adata, batch_key="modality", label_key="label",
                embed="X_emb", scale=True,
            )
            out["scib_batch_asw"] = float(asw)
        except Exception as exc:
            log.warning("scib silhouette_batch failed: %s", exc)

    # iLISI: higher values = better batch mixing.
    try:
        ilisi = scib.metrics.ilisi_graph(
            adata, batch_key="modality", type_="embed",
            use_rep="X_emb", scale=True,
        )
        out["scib_ilisi"] = float(ilisi)
    except Exception as exc:
        log.warning("scib ilisi_graph failed: %s", exc)

    # Graph connectivity on biological label: higher = better preservation.
    if has_labels:
        try:
            gc = scib.metrics.graph_connectivity(adata, label_key="label")
            out["scib_graph_connectivity"] = float(gc)
        except Exception as exc:
            log.warning("scib graph_connectivity failed: %s", exc)

    return out


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
    do_scib: bool = False,
) -> dict:
    """Compute all unified FM metrics for one model. Returns a flat metric dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "unified_metrics.json"

    # Decide whether to use a cached JSON.
    # When --scib is active and scIB keys are absent from the cache, fall
    # through so we can append them without recomputing everything else.
    _from_cache = False
    metrics: dict = {}

    if skip_existing and metrics_file.exists():
        with metrics_file.open() as f:
            cached = json.load(f)
        # scib_already_done = any(k.startswith("scib_") for k in cached)
        scib_already_done = False
        if not do_scib or scib_already_done:
            log.info("  Cached — loading %s", metrics_file)
            return cached
        log.info("  Cached (scIB missing) — computing scIB on top of %s", metrics_file)
        metrics = cached
        _from_cache = True

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

    # ── Load model (needed for reconstruction + synthetic aggregation) ────
    model = None
    if not _from_cache and ckpt_path is not None:
        log.info("  Loading model from %s ...", ckpt_path.name)
        try:
            model = _load_model(ckpt_path, device)
        except Exception:
            log.error("  Failed to load model:\n%s", traceback.format_exc())

    # ── Metric 1: reconstruction ──────────────────────────────────────────
    if not _from_cache and model is not None and sc_adata is not None:
        log.info("  Computing reconstruction metrics (%d SC cells) ...", sc_adata.n_obs)
        metrics.update(
            compute_reconstruction_metrics(
                model, bulk_adata,
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                n_cells=min(sc_adata.n_obs, 1000),
                seed=seed,
                normalized=normalized,
            )
        )

    # ── Metric 2: paired alignment ────────────────────────────────────────
    if not _from_cache and paired_pb_emb is not None and paired_bulk_emb is not None:
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
    if not _from_cache and (
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
    if not _from_cache and model is not None and sc_adata is not None and group_column in sc_adata.obs.columns:
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
    bulk_adata_for_contrast = bulk_adata if bulk_emb is not None else paired_bulk_adata
    pb_for_contrast   = paired_pb_emb
    bulk_for_contrast = bulk_emb if bulk_emb is not None else paired_bulk_emb

    if not _from_cache and pb_for_contrast is not None and bulk_for_contrast is not None:
        log.info("  Computing contrastive alignment metrics ...")
        metrics.update(compute_contrastive_metrics(bulk_for_contrast, pb_for_contrast, seed=seed))

    # ── Metric 5: scIB batch integration (bulk vs pseudobulk) ────────────
    if do_scib and pb_for_contrast is not None and bulk_for_contrast is not None:
        log.info("  Computing scIB batch integration metrics ...")
        bulk_labels_scib = bulk_adata_for_contrast.obs[group_column]
        pb_labels_scib   = paired_pb_adata.obs[group_column]
        if bulk_adata_for_contrast is not None and group_column in bulk_adata_for_contrast.obs.columns:
            bulk_labels_scib = bulk_adata_for_contrast.obs[group_column].to_numpy().astype(str)
        if paired_pb_adata is not None and group_column in paired_pb_adata.obs.columns:
            pb_labels_scib   = paired_pb_adata.obs[group_column].to_numpy().astype(str)
        metrics.update(
            compute_scib_metrics(
                bulk_for_contrast, pb_for_contrast,
                bulk_labels=bulk_labels_scib,
                pb_labels=pb_labels_scib,
                label_col=group_column,
                seed=seed,
            )
        )

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
            do_scib=do_scib,
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
    if do_scib:
        _plot_batch_integration(df, ablation_dir / "batch_integration.png")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metrics(df: pd.DataFrame, out_png: Path, ncols: int = 5) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_meta = [
        # ── Reconstruction ─────────────────────────────────────────────────
        ("recon_pearson_r",                    "Reconstruction\nPearson R ↑"),
        ("recon_mae_bins",                     "Reconstruction\nMAE Bins ↓"),
        ("recon_mae_bins_0_5",                 "MAE Bins [0–5] ↓"),
        ("recon_mae_bins_5_15",                "MAE Bins [5–15] ↓"),
        ("recon_mae_bins_15_30",               "MAE Bins [15–30] ↓"),
        ("recon_mae_bins_30_50",               "MAE Bins [30–50] ↓"),
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
            "recon_mae_bins",
            "recon_mae_bins_0_5",
            "recon_mae_bins_5_15",
            "recon_mae_bins_15_30",
            "recon_mae_bins_30_50",
        }),
        frozenset({
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
# Batch integration plot (scIB metrics — separate figure)
# ---------------------------------------------------------------------------

def _plot_batch_integration(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_meta = [
        ("scib_batch_asw",          "Batch ASW ↑\n(higher = better mixing)"),
        ("scib_ilisi",              "iLISI ↑\n(higher = better mixing)"),
        ("scib_graph_connectivity", "Graph Connectivity ↑\n(biological preservation)"),
    ]

    available = [(col, lbl) for col, lbl in metric_meta if col in df.columns]
    if not available:
        log.warning("No scIB metrics found — skipping batch_integration.png.")
        return

    n = len(available)
    models = df.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(models), 1)))

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (col, lbl) in zip(axes, available):
        vals = df[col].to_numpy(dtype=float)
        ax.bar(range(len(models)), vals, color=colors[: len(models)])
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("scIB batch integration: bulk vs pseudobulk", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Batch integration plot → %s", out_png)


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
        _plot_batch_integration(df, csv_path.parent / "batch_integration.png")
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
        do_scib=args.scib,
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
