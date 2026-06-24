"""Build a combined evaluation AnnData with per-model cell embeddings.

Loads all h5ad modalities from ``--adata-dir``, concatenates them into one
AnnData with a ``_eval_modality`` obs column, then embeds all cells through
one or more CancerFoundation checkpoints.  Each model's embeddings are stored
in a separate obsm key (``X_cf_{model_name}``), so multiple models can be
compared side-by-side from the same file.

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


def _ckpt_name(ckpt_path: Path, name: str | None) -> str:
    return name if name else ckpt_path.parent.name


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
        ("subsampled", "subsampled"),
        ("bulk",       "bulk"),
        ("paired_sc",  "paired_sc"),
        ("paired_pb",  "paired_pb"),
        ("paired_bulk","paired_bulk"),
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
) -> ad.AnnData:
    """
    Load data, embed with each model, save combined AnnData.

    Returns the combined AnnData (already written to out_path).
    """
    log.info("Loading data from %s ...", adata_dir)
    combined = load_all_modalities(adata_dir, sample_size, seed)
    log.info("Total cells: %d", combined.n_obs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_path)
    log.info("Saved → %s", out_path)
    log.info("obsm keys: %s", list(combined.obsm.keys()))
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
    p.add_argument("--not-normalized", action="store_true",
                   help="h5ad files contain raw counts (not log1p-normalised).")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
