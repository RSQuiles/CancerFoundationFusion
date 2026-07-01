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
import logging
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cancerfoundation.model.model import CancerFoundation
from evaluate.utils import generate_pseudobulk_adata

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
    """Load all modality h5ad files and return one AnnData with _eval_modality column."""
    prefixes = [
        ("subsampled",  "subsampled"),
        ("bulk",        "bulk"),
        ("paired_sc",   "paired_sc"),
        ("paired_pb",   "paired_pb"),
        ("paired_bulk", "paired_bulk"),
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
) -> dict | None:
    """Run a masked forward pass on SC cells and return raw arrays.

    Returns a dict ``{"pred": ndarray, "target": ndarray, "mask": ndarray}``
    where all arrays have shape ``(n_sampled_cells, n_genes)`` with the same
    gene ordering as ``sc_adata.var``.  ``unified_metrics.py`` reads these
    to compute reconstruction metrics without reloading the model.
    """
    from utils_config import LossType

    device = next(model.model.parameters()).device
    rng = np.random.default_rng(seed)

    use_edges = model.input_style == "binned"
    if use_edges:
        preprocess_result = model.preprocess_for_embedding(
            sc_adata, normalized=normalized, return_edges=True
        )
        data, orig_X_full, bin_edges_full = preprocess_result
    else:
        data = model.preprocess_for_embedding(sc_adata, normalized=normalized)

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

        decoder_out = model.model.decoder(transformer_out)
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
# Core
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_into_adata(
    model: CancerFoundation,
    adata: ad.AnnData,
    obsm_key: str,
    batch_size: int,
    normalized: bool,
) -> None:
    """Embed every cell in adata and store the result in adata.obsm[obsm_key]."""
    try:
        emb_df, _ = model.embed(adata, batch_size=batch_size, normalized=normalized)
        adata.obsm[obsm_key] = emb_df.to_numpy(dtype=np.float32)
        log.info(
            "  obsm['%s'] stored  (%d cells × %d dims)",
            obsm_key, adata.n_obs, adata.obsm[obsm_key].shape[1],
        )
    except Exception:
        log.error("  Embedding failed for key '%s':\n%s", obsm_key, traceback.format_exc())


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
) -> ad.AnnData:
    """
    Load data, generate synthetic pseudobulks, embed with each model,
    cache masked forward pass arrays, and save combined AnnData.

    Synthetic pseudobulks (``_eval_modality = "synth_pb"``) are generated once
    using ``n_sc_per_pseudobulk`` from the first model's saved hyperparameters.
    Their embeddings are stored alongside all other cells.

    Masked forward-pass arrays (pred, target, mask) are cached per model in
    ``adata.uns["recon_{name}"]`` so ``unified_metrics.py`` can compute
    reconstruction metrics without a second model load.
    """
    log.info("Loading data from %s ...", adata_dir)
    combined = load_all_modalities(adata_dir, sample_size, seed)
    log.info("Total cells: %d", combined.n_obs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Read n_sc_per_pb from the first model's hparams ─────────────────────
    n_sc_per_pb = 10
    if ckpt_name_pairs:
        try:
            tmp = CancerFoundation.load_from_checkpoint(str(ckpt_name_pairs[0][0]))
            n_sc_per_pb = int(getattr(tmp.hparams, "n_sc_per_pseudobulk", 10))
            log.info("n_sc_per_pseudobulk from model hparams: %d", n_sc_per_pb)
            del tmp
            gc.collect()
        except Exception:
            log.warning(
                "Could not read n_sc_per_pseudobulk from checkpoint — using default %d",
                n_sc_per_pb,
            )

    # ── Generate synthetic pseudobulks (expression only, same for all models)
    mod_col   = combined.obs["_eval_modality"]
    sc_mask   = mod_col.isin(["subsampled", "paired_sc"]).values
    bulk_mask = (mod_col == "bulk").values
    n_synth   = int(bulk_mask.sum())

    synth_pb_adata: ad.AnnData | None = None
    if sc_mask.any() and n_synth > 0 and group_column in combined.obs.columns:
        synth_pb_adata = generate_pseudobulk_adata(
            combined[sc_mask],
            group_column=group_column,
            n_sc_per_pb=n_sc_per_pb,
            n_pb=n_synth,
            seed=seed,
            is_log1p=True,
            normalize=True,
        )
    if synth_pb_adata is not None:
        synth_pb_adata.obs["_eval_modality"] = "synth_pb"
        combined = ad.concat([combined, synth_pb_adata], join="outer", merge="same")
        combined.obs_names_make_unique()
        log.info(
            "Generated %d synthetic pseudobulks (n_sc_per_pb=%d, group='%s')",
            synth_pb_adata.n_obs, n_sc_per_pb, group_column,
        )
    else:
        log.warning(
            "Skipping synthetic pseudobulk generation "
            "(sc_cells=%d, n_synth_target=%d, group_column_present=%s)",
            sc_mask.sum(), n_synth, group_column in combined.obs.columns,
        )

    # ── PCA baseline (model-independent) ────────────────────────────────────
    log.info("Computing PCA baseline (%d components) ...", n_pca_components)
    try:
        import scipy.sparse as sp
        from sklearn.decomposition import PCA

        X_expr = combined.X
        if sp.issparse(X_expr):
            X_expr = X_expr.toarray()
        X_expr = np.nan_to_num(X_expr.astype(np.float32))
        n_components = min(n_pca_components, X_expr.shape[0] - 1, X_expr.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        combined.obsm["X_pca"] = pca.fit_transform(X_expr).astype(np.float32)
        log.info(
            "  X_pca stored (%d cells × %d components)",
            combined.n_obs, combined.obsm["X_pca"].shape[1],
        )
    except Exception:
        log.warning("PCA computation failed:\n%s", traceback.format_exc())

    # ── Per-model: embed all cells + cache masked forward pass ───────────────
    for ckpt_path, name in ckpt_name_pairs:
        obsm_key = f"X_cf_{name}"
        log.info("[%s] Embedding → obsm['%s'] ...", name, obsm_key)
        try:
            model = CancerFoundation.load_from_checkpoint(str(ckpt_path))
            model.eval().to(device)
        except Exception:
            log.error("  Failed to load %s:\n%s", ckpt_path, traceback.format_exc())
            continue

        embed_into_adata(model, combined, obsm_key, batch_size, normalized)

        # Cache masked forward pass on bulk cells for reconstruction metrics
        bulk_sub = combined[combined.obs["_eval_modality"] == "bulk"]
        if bulk_sub.n_obs > 0:
            log.info(
                "  Caching masked forward pass for reconstruction (%d bulk cells) ...",
                bulk_sub.n_obs,
            )
            try:
                recon_cache = _run_masked_forward(
                    model, bulk_sub,
                    batch_size=batch_size, seed=seed, normalized=normalized,
                )
                if recon_cache is not None:
                    combined.uns[f"recon_{name}"] = recon_cache
                    log.info("  Cached: pred/target/mask arrays stored in uns['recon_%s']", name)
            except Exception:
                log.warning("  Masked forward pass failed:\n%s", traceback.format_exc())

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_path)
    log.info("Saved → %s", out_path)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
