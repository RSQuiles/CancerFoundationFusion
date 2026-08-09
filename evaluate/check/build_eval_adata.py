"""Build a combined evaluation AnnData with per-model cell embeddings.

Loads all h5ad modalities from ``--adata-dir``, concatenates them into one
AnnData with a ``_eval_modality`` obs column, then embeds all cells through
one or more CancerFoundation checkpoints.  Each model's embeddings are stored
in a separate obsm key (``X_cf_{model_name}``), so multiple models can be
compared side-by-side from the same file.

In addition, for each model:
  - Synthetic pseudobulks are generated from SC cells (n = number of non-paired
    bulk samples) and embedded.  The ``n_sc_per_pseudobulk`` used during
    generation is read from the model's saved hyperparameters so it matches
    training.  The synthetic PBs are added as new rows with
    ``_eval_modality = "synth_pb"``.
  - A masked forward pass is run on SC cells with a fixed seed and the raw
    (pred, target, mask) arrays are cached in
    ``adata.uns["recon_{model_name}"]``.  ``unified_metrics.py`` reads these
    to compute reconstruction metrics without loading the model again.

The output is consumed by ``unified_metrics.py``.

Usage
-----
Single checkpoint:
    python build_eval_adata.py \\
        --adata-dir path/to/h5ads \\
        --out path/to/eval.h5ad \\
        --ckpt path/to/epoch_05.ckpt [--name my_model]

Multiple checkpoints (--ckpt / --name are repeatable):
    python build_eval_adata.py \\
        --adata-dir path/to/h5ads \\
        --out path/to/eval.h5ad \\
        --ckpt a/epoch.ckpt --name model_a \\
        --ckpt b/epoch.ckpt --name model_b

Ablation directory:
    python build_eval_adata.py \\
        --adata-dir path/to/h5ads \\
        --out path/to/eval.h5ad \\
        --ablation-dir path/to/ablation

Expected files in ``--adata-dir``
----------------------------------
    subsampled*.h5ad  Regular single-cell samples (log1p-normalised)
    bulk*.h5ad        Unpaired bulk RNA-seq samples
    paired_sc*.h5ad   SC constituent cells for paired pseudobulks
    paired_pb*.h5ad   Pre-computed pseudobulk rows
    paired_bulk*.h5ad Matched real-bulk rows
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
import scanpy as sc
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.checkpoint import read_checkpoint_hparams
from cancerfoundation.data.bulk_sc_collator import BULK_MODALITY
from cancerfoundation.model.model import CancerFoundation
from evaluate.utils import (
    MODALITY_COL as _MODALITY_COL,
    MOD_BULK,
    MOD_PAIRED_BULK,
    MOD_PAIRED_PB,
    MOD_PAIRED_SC,
    MOD_PB,
    MOD_SC,
    MOD_SYNTH_PB,
    SC_MODALITIES,
    detect_scale_by_modality,
    generate_pseudobulk_adata,
    log_scale_table,
    looks_like_counts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_ckpt(model_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.ckpt", "checkpoints/*.ckpt"):
        candidates.extend(model_dir.glob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _load_prefix(
    adata_dir: Path,
    prefix: str,
    n_max: int,
    seed: int,
) -> ad.AnnData | None:
    files = sorted(adata_dir.glob(f"{prefix}*.h5ad"))
    if not files:
        return None
    rng = np.random.default_rng(seed)
    parts: list[ad.AnnData] = []
    total = 0
    for f in files:
        if total >= n_max:
            break
        a = ad.read_h5ad(f)
        if a.n_obs == 0:
            continue
        keep = min(a.n_obs, n_max - total)
        if keep < a.n_obs:
            a = a[rng.choice(a.n_obs, size=keep, replace=False)].copy()
        a.obs["_source_file"] = f.name
        parts.append(a)
        total += keep
    if not parts:
        return None
    out = ad.concat(parts, join="outer", merge="same")
    out.obs_names_make_unique()
    return out


def load_all_modalities(
    adata_dir: Path,
    sample_size: int,
    seed: int,
) -> ad.AnnData:
    """Load all modality h5ad files and return one AnnData with _eval_modality column.

    Note the left column is a *filename prefix* and the right one is the
    ``_eval_modality`` label written to obs — they are not the same vocabulary.
    Consumers must match on the labels (imported from ``evaluate.utils``), never on
    the prefixes; matching on "subsampled" is what silently disabled the
    ``agg_synth_*`` metrics.
    """
    prefixes = [
        ("subsampled",  MOD_SC),
        ("partition",  MOD_SC),
        ("sc",  MOD_SC),
        ("pretraining_sc", MOD_SC),
        ("pretraining_bulk", MOD_BULK),
        ("pseudo_bulk", MOD_PB),
        ("bulk",        MOD_BULK),
        ("paired_sc",   MOD_PAIRED_SC),
        ("paired_pb",   MOD_PAIRED_PB),
        ("paired_bulk", MOD_PAIRED_BULK),
    ]
    parts: list[ad.AnnData] = []
    for prefix, label in prefixes:
        adata = _load_prefix(adata_dir, prefix, sample_size, seed)
        if adata is None:
            log.info("No %s*.h5ad files found — skipping.", prefix)
            continue
        adata.obs["_eval_modality"] = label
        log.info("Loaded %-12s : %d cells", label, adata.n_obs)
        parts.append(adata)

    if not parts:
        raise FileNotFoundError(f"No recognised h5ad files found in {adata_dir}")

    combined = ad.concat(parts, join="outer", merge="same")
    combined.obs_names_make_unique()
    return combined


# ---------------------------------------------------------------------------
# Masked forward pass — returns raw (pred, target, mask) arrays for caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_masked_forward(
    model: CancerFoundation,
    sc_adata: ad.AnnData,
    batch_size: int,
    seed: int,
    normalized: bool,
    mask_ratio: float = 0.15,
    n_cells: int = 1000,
    recon_conditions: dict[str, str] | None = None,
) -> dict | None:
    """Run a masked forward pass on SC cells and return raw arrays.

    Returns a dict ``{"pred": ndarray, "target": ndarray, "mask": ndarray}``
    where all arrays have shape ``(n_sampled_cells, n_genes)`` with the same
    gene ordering as ``sc_adata.var``.  ``unified_metrics.py`` reads these
    to compute reconstruction metrics without reloading the model.

    ``recon_conditions`` maps an encoded condition name -> the obs column to read
    its per-cell label from (default: the condition's own name).  It only applies
    to non-``modality`` conditions of a ``where_condition == "end"`` model, whose
    conditional decoder needs the condition embedding concatenated onto the
    transformer output.  ``modality`` always uses the canonical ``BULK_MODALITY``
    code (recon cells are all bulk); every other condition's integer code is read
    from the model's ``mapping.json``.
    """
    from utils_config import LossType

    device = next(model.model.parameters()).device
    rng = np.random.default_rng(seed)

    use_edges = model.input_style == "binned"
    if use_edges:
        preprocess_result = model.preprocess_for_embedding(
            sc_adata, normalized=normalized, return_edges=True, modality="bulk"
        )
        data, orig_X_full, bin_edges_full = preprocess_result
    else:
        data = model.preprocess_for_embedding(sc_adata, normalized=normalized, modality="bulk")

    if data.n_obs == 0:
        return None

    n = min(data.n_obs, n_cells)
    idx = rng.choice(data.n_obs, size=n, replace=False)
    data = data[idx].copy()

    X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
    X = X.astype(np.float32)

    if use_edges:
        all_orig_expr = orig_X_full[list(idx)].astype(np.float32)
        all_bin_edges = bin_edges_full[list(idx)]
    gene_ids = torch.LongTensor([model.vocab[g] for g in data.var.index]).to(device)
    n_genes = X.shape[1]

    if model.input_emb_style == "category":
        is_normalised = (
            hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "normalise_bins")
            and model.model.decoder.normalise_bins
        )
        eff_mask = float(model.mask_value) / model.n_bins if is_normalised else float(model.mask_value)
    else:
        eff_mask = float(model.mask_value)

    effective_loss = model.loss_type

    # Condition codes for a `where_condition == "end"` model: its ExprDecoder was built
    # to take [condition_emb || transformer_output], so the bare encoder output is too
    # narrow. Rebuild the condition embedding here (mirrors module.py's "end" branch).
    # `modality` uses the canonical model-facing code (decoupled from mapping.json, see
    # bulk_sc_collator.py); every other condition's integer code comes from mapping.json.
    mod = model.model
    cond_codes: dict[str, torch.Tensor] = {}
    if getattr(mod, "_use_condition_encoders", False) and mod.where_condition == "end":
        recon_conditions = recon_conditions or {}
        mapping = None  # loaded lazily, only if a non-modality condition needs it
        for cname in mod.encoded_conditions:
            if cname == "modality":
                # recon cells are all bulk
                codes = torch.full((n,), BULK_MODALITY, dtype=torch.long, device=device)
            else:
                if mapping is None:
                    map_path = Path(model.data_path) / "mapping.json"
                    if not map_path.is_file():
                        raise FileNotFoundError(
                            f"condition '{cname}' needs codes from {map_path} "
                            f"(model.data_path) but it is not present"
                        )
                    mapping = json.loads(map_path.read_text())
                obs_col = recon_conditions.get(cname, cname)
                if obs_col not in data.obs:
                    raise ValueError(
                        f"condition '{cname}' needs obs column '{obs_col}'; not in eval "
                        f"obs. Pass recon_conditions={{'{cname}': '<obs column>'}}"
                    )
                table = mapping[cname]  # {label: int code}
                labels = data.obs[obs_col].astype(str).to_numpy()
                try:
                    codes = torch.as_tensor(
                        [table[lbl] for lbl in labels], dtype=torch.long, device=device
                    )
                except KeyError as e:
                    raise ValueError(
                        f"label {e} in obs['{obs_col}'] is absent from "
                        f"mapping.json['{cname}']"
                    ) from e
            cond_codes[cname] = codes

    all_pred   = np.zeros((n, n_genes), dtype=np.float32)
    all_target = X.copy()
    all_mask   = np.zeros((n, n_genes), dtype=bool)

    for start in range(0, n, batch_size):
        batch_expr = torch.from_numpy(X[start: start + batch_size]).to(device)
        bs = batch_expr.shape[0]
        batch_genes = gene_ids.unsqueeze(0).expand(bs, -1)

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

        if cond_codes:
            # Concatenate the condition embedding onto every token, matching the
            # training "end" branch in module.py (generative/perceptual forward).
            condition_emb = torch.cat(
                [
                    mod.condition_encoders[c](cond_codes[c][start:start + bs]).unsqueeze(1)
                    for c in mod.encoded_conditions
                ],
                dim=1,
            )  # (bs, n_cond, d_model)
            decoder_input = torch.cat(
                [
                    condition_emb.view(bs, -1)
                    .unsqueeze(1)
                    .repeat(1, transformer_out.shape[1], 1),
                    transformer_out,
                ],
                dim=2,
            )
        else:
            decoder_input = transformer_out

        decoder_out = model.model.decoder(decoder_input)
        raw_pred = decoder_out["pred"]

        if effective_loss == LossType.CORN:
            logits = raw_pred[:, 1:, :]
            pred = (torch.cumprod(torch.sigmoid(logits), -1) > 0.5).sum(-1).float()
        elif effective_loss == LossType.ORDINALCROSSENTROPY:
            pred = torch.sigmoid(raw_pred[:, 1:, :]).sum(dim=-1)
        else:
            pred = raw_pred[:, 1:, 0] if raw_pred.dim() == 3 else raw_pred[:, 1:]

        end = start + bs
        all_pred[start:end] = pred.float().cpu().numpy()
        all_mask[start:end] = gene_mask.cpu().numpy()

    cache: dict = {"pred": all_pred, "target": all_target, "mask": all_mask}
    if use_edges:
        cache["orig_expr"] = all_orig_expr
        cache["bin_edges"] = all_bin_edges
        cache["n_bins"]    = model.n_bins
    return cache


# ---------------------------------------------------------------------------
# PCA baseline
# ---------------------------------------------------------------------------

def _cp10k_log1p_rows(X, rows: np.ndarray, target_sum: float = 1e4) -> None:
    """CP10K + log1p, in place, for ``rows`` only.

    Works on the CSR ``data`` buffer directly. The obvious spelling —
    ``X[rows] = normalized_submatrix`` — is fancy assignment into a sparse matrix,
    which makes scipy rebuild the entire sparsity structure and allocates several
    times the matrix; row-slicing ``data[indptr[r]:indptr[r+1]]`` is O(nnz) and
    allocates nothing. Dense input is handled directly.
    """
    import scipy.sparse as sp

    if len(rows) == 0:
        return
    if not sp.issparse(X):
        block = X[rows]
        totals = block.sum(axis=1, keepdims=True)
        np.divide(block, np.where(totals > 0, totals, 1.0), out=block)
        np.multiply(block, target_sum, out=block)
        X[rows] = np.log1p(block)
        return

    indptr, data = X.indptr, X.data
    for r in rows:
        lo, hi = indptr[r], indptr[r + 1]
        if hi <= lo:
            continue
        row = data[lo:hi]
        total = row.sum()
        if total > 0:
            row *= target_sum / total
        data[lo:hi] = np.log1p(row)


def _to_log1p_scale(adata: ad.AnnData, scale: dict) -> ad.AnnData:
    """CP10K + log1p only the modality groups detected as raw counts.

    Returns ``adata`` with every row on the log1p scale, so a downstream ``expm1``
    is valid for all of them. Mutates the passed object (callers hand in a copy).
    """
    if _MODALITY_COL not in adata.obs.columns:
        if looks_like_counts(adata.X):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        return adata

    labels = adata.obs[_MODALITY_COL].astype(str).to_numpy()
    for group, is_counts in scale.items():
        if not is_counts:
            continue
        rows = np.where(labels == group)[0]
        if len(rows) == 0:
            continue
        log.info("  '%s': %d rows are raw counts -> CP10K + log1p before aggregating",
                 group, len(rows))
        _cp10k_log1p_rows(adata.X, rows)
    return adata


def _pca_baseline(
    combined: ad.AnnData,
    n_pca_components: int,
    n_hvg_baseline: int,
    scale: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """scIB-style "unintegrated" reference: CP10K + log1p -> HVGs -> PCA.

    This is what PCR comparison scores every embedding against, so it must be a fair
    baseline: PCA straight on raw counts is dominated by library size, which for bulk
    vs pseudobulk *is* the batch, and pcr_comparison being
    ``max(0, (pre - post)/pre)`` then clips to 0 for every model.

    Order matters for memory, not just correctness. Everything stays sparse until the
    gene set has been cut to ``n_hvg_baseline``; only that block is densified. The
    previous version densified the full matrix and held ~3 copies of it before
    selecting HVGs, which at 300k cells x 28.7k genes is ~100 GB and was reliably
    OOM-killed. scanpy's normalize_total / log1p / highly_variable_genes all accept
    CSR, so nothing is lost by deferring the densification.
    """
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    recipe: dict = {"n_components_requested": int(n_pca_components), "seed": int(seed)}

    # Sparse copy: cheap relative to the dense form, and keeps combined.X untouched
    # since normalize_total/log1p mutate in place.
    X = combined.X
    X = X.copy() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float32))
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    work = ad.AnnData(X=X, obs=combined.obs[[]].copy(), var=combined.var[[]].copy())
    del X

    # Bring the count groups onto the log1p scale the others are already on. A single
    # global verdict is wrong here — see detect_scale_by_modality.
    counts_groups = [g for g, is_counts in scale.items() if is_counts]
    recipe["scale_by_modality"] = {str(k): bool(v) for k, v in scale.items()}
    if counts_groups and _MODALITY_COL in combined.obs.columns:
        labels = combined.obs[_MODALITY_COL].astype(str).to_numpy()
        rows = np.where(np.isin(labels, counts_groups))[0]
        log.info(
            "  CP10K + log1p for %d rows in count-scale group(s): %s",
            len(rows), ", ".join(counts_groups),
        )
        _cp10k_log1p_rows(work.X, rows)
        recipe["normalisation"] = f"normalize_total(1e4)+log1p on {counts_groups}"
    elif not counts_groups:
        log.info("  every modality already looks log1p -> PCA on the values as-is")
        recipe["normalisation"] = "none"
    else:
        # No modality column: fall back to one global decision.
        if looks_like_counts(work.X):
            sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
            recipe["normalisation"] = "normalize_total(1e4)+log1p (global)"
        else:
            recipe["normalisation"] = "none"

    # Safety net before HVG. scanpy's "seurat" flavor expm1's its input, so any row
    # still on the count scale here overflows to inf and pd.cut then raises
    # "cannot specify integer `bins` when input data contains infinity". That can only
    # happen if looks_like_counts returned a false negative for some group, so log1p
    # such rows rather than letting the whole baseline die.
    _mx = float(work.X.max()) if work.X.nnz else 0.0
    if _mx > 20:
        log.warning(
            "  values up to %.1f survived normalisation - some group was detected as "
            "log1p but looks like counts. Applying log1p globally so the HVG step "
            "(which expm1's internally) does not overflow; the baseline will be "
            "approximate.", _mx,
        )
        sc.pp.log1p(work)
        recipe["normalisation"] += " (+global log1p fallback)"

    # HVGs while still sparse, then cut the gene axis before densifying.
    n_hvg = min(n_hvg_baseline, work.n_vars)
    if n_hvg < work.n_vars:
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="seurat")
        work = work[:, work.var["highly_variable"].values].copy()
    recipe["n_hvg"] = int(work.n_vars)

    dense = work.X.toarray() if sp.issparse(work.X) else np.asarray(work.X)
    del work
    dense = np.nan_to_num(dense, copy=False).astype(np.float32, copy=False)
    log.info("  densified %d x %d block (%.2f GB) for PCA",
             dense.shape[0], dense.shape[1], dense.nbytes / 1e9)

    n_components = int(min(n_pca_components, dense.shape[0] - 1, dense.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(dense).astype(np.float32)
    recipe["n_components"] = n_components
    recipe["explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
    return coords, recipe


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def panel_hash(panel) -> str:
    """Stable short hash of a gene panel (flat list or per-modality dict).

    Used as a cache key: metrics computed under one panel must not be reused for
    another. Shared with ``unified_metrics.py`` via ``uns["eval_provenance"]``.
    """
    import hashlib
    import json

    if panel is None:
        return "none"
    if isinstance(panel, dict):
        payload = {str(k): sorted(map(str, v)) for k, v in sorted(panel.items())}
    else:
        payload = sorted(map(str, panel))
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _build_provenance(
    combined: ad.AnnData,
    panels: dict,
    normalized: bool,
    shared_panel: bool,
    panel_strategy: str,
    sample_size: int,
    seed: int,
    group_column: str,
    recon_failures: dict | None = None,
    scale_by_modality: dict | None = None,
) -> dict:
    """Assemble the ``uns["eval_provenance"]`` record."""
    counts = (
        combined.obs["_eval_modality"].astype(str).value_counts().to_dict()
        if "_eval_modality" in combined.obs.columns
        else {}
    )
    # One hash over all models' panels: under shared_panel they are refitted per
    # model (n_top_genes is a model hparam), so a difference in any of them means
    # the cached metrics are not comparable.
    combined_hash = panel_hash(
        {str(k): (v if not isinstance(v, dict) else sorted(sum(v.values(), [])))
         for k, v in sorted(panels.items())}
    ) if panels else "none"

    return {
        "normalized_flag": bool(normalized),
        # Per group, because a single global verdict is not meaningful when the
        # modalities disagree on scale (see detect_scale_by_modality).
        "scale_by_modality": {
            str(k): bool(v) for k, v in (scale_by_modality or {}).items()
        },
        "shared_panel": bool(shared_panel),
        "panel_strategy": str(panel_strategy),
        "panel_hash": combined_hash,
        "panel_sizes": {
            k: (len(v) if not isinstance(v, dict)
                else {kk: len(vv) for kk, vv in v.items()})
            for k, v in panels.items()
        },
        "modality_counts": {str(k): int(v) for k, v in counts.items()},
        # Empty when every model's masked forward pass succeeded. Non-empty means
        # recon_* will be missing downstream, with the reason recorded here.
        "recon_failures": {str(k): str(v) for k, v in (recon_failures or {}).items()},
        "sample_size_per_prefix": int(sample_size),
        "seed": int(seed),
        "group_column": str(group_column),
        "n_obs": int(combined.n_obs),
        "n_vars": int(combined.n_vars),
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_into_adata(
    model: CancerFoundation,
    adata: ad.AnnData,
    obsm_key: str,
    batch_size: int,
    normalized: bool,
    shared_panel: bool = True,
    panel_strategy: str = "consensus",
) -> list[str] | dict | None:
    """Embed every cell in adata and store the result in adata.obsm[obsm_key].

    Returns the gene panel actually used — a flat list under ``shared_panel``, a
    per-modality dict otherwise — so the caller can record it as provenance.
    """
    try:
        emb_df, gene_set = model.embed(
            adata, batch_size=batch_size, normalized=normalized,
            modality_col="_eval_modality",
            shared_panel=shared_panel, panel_strategy=panel_strategy,
        )
        adata.obsm[obsm_key] = emb_df.to_numpy(dtype=np.float32)
        log.info(
            "  obsm['%s'] stored  (%d cells x %d dims)",
            obsm_key, adata.n_obs, adata.obsm[obsm_key].shape[1],
        )
        return gene_set
    except Exception:
        log.error("  Embedding failed for key '%s':\n%s", obsm_key, traceback.format_exc())
        return None


def build(
    adata_dir: Path,
    out_path: Path,
    ckpt_name_pairs: list[tuple[Path, str]],
    sample_size: int = 2000,
    batch_size: int = 64,
    seed: int = 0,
    normalized: bool = True,
    group_column: str = "tissue_general",
    n_pca_components: int = 50,
    n_hvg_baseline: int = 2000,
    precomputed_pb: bool = False,
    shared_panel: bool = True,
    panel_strategy: str = "consensus",
    recon_conditions: dict[str, str] | None = None,
) -> ad.AnnData:
    """
    Load data, generate synthetic pseudobulks, embed with each model,
    cache masked forward pass arrays, and save combined AnnData.

    Synthetic pseudobulks (``_eval_modality = "synth_pb"``) are generated once
    using ``n_sc_per_pseudobulk`` from the first model's saved hyperparameters.
    Their embeddings are stored alongside all other cells.

    All modalities are embedded through ONE shared gene panel by default
    (``shared_panel``), which is what makes the cross-modality metrics
    interpretable; see ``cancerfoundation.data.gene_panel.select_shared_panel``.

    Masked forward-pass arrays (pred, target, mask) are cached per model in
    ``adata.uns["recon_{name}"]`` so ``unified_metrics.py`` can compute
    reconstruction metrics without a second model load.
    """
    log.info("Loading data from %s ...", adata_dir)
    combined = load_all_modalities(adata_dir, sample_size, seed)
    log.info("Total cells: %d", combined.n_obs)

    # Per-modality, because these files genuinely disagree on scale — see
    # detect_scale_by_modality. Everything below that needs to know the expression
    # scale reads this rather than re-deciding globally.
    scale_by_modality = detect_scale_by_modality(combined, _MODALITY_COL)
    log_scale_table(scale_by_modality, log)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Read n_sc_per_pb from the first model's hparams ─────────────────────
    n_sc_per_pb = 10
    if ckpt_name_pairs:
        try:
            # Read the hparams straight off the checkpoint; building the whole
            # model for one scalar fails on any checkpoint with drifted hparams.
            hparams = read_checkpoint_hparams(ckpt_name_pairs[0][0])
            n_sc_per_pb = int(hparams.get("n_sc_per_pseudobulk", 10))
            log.info("n_sc_per_pseudobulk from model hparams: %d", n_sc_per_pb)
        except Exception:
            log.warning(
                "Could not read n_sc_per_pseudobulk from checkpoint — using default %d",
                n_sc_per_pb,
            )

    # ── Generate synthetic pseudobulks (expression only, same for all models)
    mod_col   = combined.obs["_eval_modality"]
    sc_mask   = mod_col.isin(SC_MODALITIES).values
    bulk_mask = (mod_col == MOD_BULK).values
    pb_mask = (mod_col == MOD_PB).values
    n_synth   = int(bulk_mask.sum())

    synth_pb_adata: ad.AnnData | None = None
    if sc_mask.any() and n_synth > 0 and group_column in combined.obs.columns and not precomputed_pb:
        log.info("Generating pseudobulk samples on-the-fly from single-cell!")
        # The SC pool can span two modalities on different scales ('sc' is raw counts
        # from cellxgene_subsample, 'paired_sc' is log1p because
        # pseudobulk_paired_generation.py log1p's it unconditionally). Aggregating
        # across that with one is_log1p verdict applies expm1 to raw counts and
        # saturates float32, so bring the count rows onto the log1p scale first and
        # then say so explicitly rather than letting the callee guess.
        sc_sub = _to_log1p_scale(combined[sc_mask].copy(), scale_by_modality)
        synth_pb_adata = generate_pseudobulk_adata(
            sc_sub,
            group_column=group_column,
            n_sc_per_pb=n_sc_per_pb,
            n_pb=n_synth,
            seed=seed,
            is_log1p=True,  # guaranteed by _to_log1p_scale above
            normalize=True,
        )
        del sc_sub
        if synth_pb_adata is not None:
            synth_pb_adata.obs["_eval_modality"] = MOD_SYNTH_PB
            combined = ad.concat([combined, synth_pb_adata], join="outer", merge="same")
            combined.obs_names_make_unique()
            log.info(
                "Generated %d synthetic pseudobulks (n_sc_per_pb=%d, group='%s')",
                synth_pb_adata.n_obs, n_sc_per_pb, group_column,
            )
    elif precomputed_pb and pb_mask.any():
        # Relabel a subset of the existing pseudobulk rows IN PLACE. Concatenating a
        # relabelled *copy* (as this used to) put the same cells in the file twice,
        # embedded twice under independently fitted panels — so the UMAP read one
        # copy and scIB the other and they disagreed for no modelling reason.
        pb_pos = np.where(pb_mask)[0]
        if len(pb_pos) > n_synth > 0:
            rng = np.random.default_rng(seed)
            pb_pos = np.sort(rng.choice(pb_pos, size=n_synth, replace=False))
        labels = combined.obs["_eval_modality"].astype(str).to_numpy()
        labels[pb_pos] = MOD_SYNTH_PB
        combined.obs["_eval_modality"] = labels
        log.info(
            "Relabelled %d of %d precomputed pseudobulk rows as '%s' in place "
            "(no duplicated cells)",
            len(pb_pos), int(pb_mask.sum()), MOD_SYNTH_PB,
        )
    else:
        log.warning(
            "Skipping synthetic pseudobulk generation "
            "(sc_cells=%d, pb_rows=%d, n_synth_target=%d, group_column_present=%s)",
            sc_mask.sum(), int(pb_mask.sum()), n_synth,
            group_column in combined.obs.columns,
        )

    # ── PCA baseline (model-independent) ────────────────────────────────────
    # This is the "unintegrated" reference that scIB's PCR comparison scores every
    # embedding against, so it has to be a *fair* baseline. PCA straight on raw
    # counts is dominated by library size — which for bulk vs pseudobulk IS the
    # batch — so pcr_pre saturates and pcr_comparison, being
    # max(0, (pcr_pre - pcr_post)/pcr_pre), can only ever clip to 0. Follow the
    # standard scIB recipe instead: CP10K + log1p (when the data is counts), HVGs,
    # then PCA.
    if n_pca_components <= 0:
        log.info("Skipping PCA baseline (n_pca_components <= 0); "
                 "scIB's PCR comparison will report no baseline.")
    else:
        log.info("Computing PCA baseline (%d components) ...", n_pca_components)
        try:
            combined.obsm["X_pca"], recipe = _pca_baseline(
                combined, n_pca_components, n_hvg_baseline, scale_by_modality, seed
            )
            combined.uns["X_pca_recipe"] = recipe
            log.info(
                "  X_pca stored (%d cells x %d components, %d HVGs, %.3f var explained)",
                combined.n_obs, combined.obsm["X_pca"].shape[1], recipe["n_hvg"],
                recipe["explained_variance_ratio_sum"],
            )
            gc.collect()
        except Exception:
            # Never fatal: the baseline only feeds scIB's PCR comparison, and losing
            # it must not cost the embeddings this build already computed.
            log.error("PCA baseline failed (X_pca will be absent):\n%s",
                      traceback.format_exc())

    # ── Per-model: embed all cells + cache masked forward pass ───────────────
    panels: dict[str, list[str] | dict] = {}
    recon_failures: dict[str, str] = {}
    for ckpt_path, name in ckpt_name_pairs:
        obsm_key = f"X_cf_{name}"
        log.info("[%s] Embedding -> obsm['%s'] ...", name, obsm_key)
        try:
            model = CancerFoundation.load_for_inference(ckpt_path)
            model.to(device)
        except Exception:
            log.error("  Failed to load %s:\n%s", ckpt_path, traceback.format_exc())
            continue

        panel = embed_into_adata(
            model, combined, obsm_key, batch_size, normalized,
            shared_panel=shared_panel, panel_strategy=panel_strategy,
        )
        if panel is not None:
            panels[name] = panel

        # Cache masked forward pass on bulk cells for reconstruction metrics
        bulk_sub = combined[combined.obs["_eval_modality"] == MOD_BULK]
        if bulk_sub.n_obs > 0:
            log.info(
                "  Caching masked forward pass for reconstruction (%d bulk cells) ...",
                bulk_sub.n_obs,
            )
            try:
                recon_cache = _run_masked_forward(
                    model, bulk_sub,
                    batch_size=batch_size, seed=seed, normalized=normalized,
                    recon_conditions=recon_conditions,
                )
                if recon_cache is not None:
                    combined.uns[f"recon_{name}"] = recon_cache
                    log.info("  Cached: pred/target/mask arrays stored in uns['recon_%s']", name)
                else:
                    recon_failures[name] = "no cells left after preprocessing"
                    log.error(
                        "  Reconstruction cache NOT written for '%s': preprocessing "
                        "left 0 bulk cells. recon_* metrics will be unavailable.", name,
                    )
            except Exception as exc:
                # Raised to error because a missing cache silently removes the whole
                # recon_* family downstream — and the live fallback in
                # unified_metrics.py cannot run in the scIB conda environment.
                recon_failures[name] = f"{type(exc).__name__}: {exc}"
                log.error(
                    "  Reconstruction cache NOT written for '%s' — recon_* metrics "
                    "will be unavailable downstream:\n%s", name, traceback.format_exc(),
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Provenance ───────────────────────────────────────────────────────────
    # Records how these embeddings were produced. unified_metrics.py stores the
    # panel hash in each metrics JSON and refuses a --skip-existing cache whose
    # hash differs, so changing the panel can never silently serve stale metrics.
    combined.uns["eval_provenance"] = _build_provenance(
        combined, panels,
        normalized=normalized,
        shared_panel=shared_panel,
        panel_strategy=panel_strategy,
        sample_size=sample_size,
        seed=seed,
        group_column=group_column,
        recon_failures=recon_failures,
        scale_by_modality=scale_by_modality,
    )
    if recon_failures:
        log.error(
            "Reconstruction cache missing for %d model(s): %s — recon_* metrics will "
            "be absent unless unified_metrics.py runs somewhere CancerFoundation can "
            "be imported (i.e. inside the container, not the scIB conda env).",
            len(recon_failures), ", ".join(sorted(recon_failures)),
        )
    log.info(
        "Provenance: shared_panel=%s strategy=%s panel_hash=%s",
        shared_panel, panel_strategy,
        combined.uns["eval_provenance"]["panel_hash"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_path)
    log.info("Saved -> %s", out_path)
    log.info("obsm keys: %s", list(combined.obsm.keys()))
    log.info("uns  keys: %s", list(combined.uns.keys()))
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build evaluation AnnData with per-model embeddings in obsm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--adata-dir", type=Path, required=True,
                   help="Directory containing subsampled/bulk/paired_* h5ad files.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output h5ad path.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ablation-dir", type=Path, default=None,
                     help="Ablation root; embeds with every model sub-directory.")
    src.add_argument("--ckpt", type=Path, action="append", dest="ckpts",
                     help="Checkpoint path (repeatable). Pair with --name.")

    p.add_argument("--name", type=str, action="append", dest="names",
                   help="Model name for the preceding --ckpt (repeatable). "
                        "Defaults to the checkpoint's parent directory name.")

    p.add_argument("--sample-size", type=int, default=2000,
                   help="Max cells per modality to load (default: 2000).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-column", type=str, default="tissue_general",
                   help="obs column used to group SC cells for synthetic pseudobulk "
                        "generation (default: tissue_general).")
    p.add_argument("--not-normalized", action="store_true",
                   help="h5ad files contain raw counts (not log1p-normalised).")
    p.add_argument("--n-pca-components", type=int, default=50,
                   help="Number of PCA components for the baseline embedding (default: 50).")
    p.add_argument("--n-hvg-baseline", type=int, default=2000,
                   help="HVGs used for the PCA baseline (default: 2000).")
    p.add_argument("--precomputed-pb", action="store_true", help="Precomputed pb in dataset")
    p.add_argument("--per-modality-panel", action="store_true",
                   help="Fit a separate gene panel per modality instead of one shared "
                        "panel. Restores the pre-fix behaviour; cross-modality metrics "
                        "computed this way are confounded by the panel difference, so "
                        "this is for comparison only.")
    p.add_argument("--panel-strategy", type=str, default="consensus",
                   choices=["consensus", "reference", "pooled"],
                   help="How to fit the shared panel (default: consensus, which ranks "
                        "genes within each modality and averages the ranks). 'pooled' "
                        "is biased toward genes that separate the modalities.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    adata_dir = args.adata_dir.expanduser().resolve()
    if not adata_dir.is_dir():
        log.error("--adata-dir not found: %s", adata_dir)
        return 1

    if args.ablation_dir is not None:
        ablation_dir = args.ablation_dir.expanduser().resolve()
        if not ablation_dir.is_dir():
            log.error("--ablation-dir not found: %s", ablation_dir)
            return 1
        model_dirs = sorted(
            d for d in ablation_dir.iterdir()
            if d.is_dir() and _find_best_ckpt(d) is not None
        )
        if not model_dirs:
            log.error("No model directories with checkpoints under %s", ablation_dir)
            return 1
        pairs = [(_find_best_ckpt(d), d.name) for d in model_dirs]
    else:
        ckpts = [p.expanduser().resolve() for p in (args.ckpts or [])]
        names = args.names or []
        pairs = [
            (ckpt, names[i] if i < len(names) else ckpt.parent.name)
            for i, ckpt in enumerate(ckpts)
        ]

    build(
        adata_dir=adata_dir,
        out_path=args.out.expanduser().resolve(),
        ckpt_name_pairs=pairs,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
        normalized=not args.not_normalized,
        group_column=args.group_column,
        n_pca_components=args.n_pca_components,
        n_hvg_baseline=args.n_hvg_baseline,
        precomputed_pb=args.precomputed_pb,
        shared_panel=not args.per_modality_panel,
        panel_strategy=args.panel_strategy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
