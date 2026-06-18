"""
Inference script for downstream fine-tuning tasks.

Loads a saved checkpoint (head + optionally a finetuned transformer) and runs
predictions on an AnnData file or a pre-computed expression matrix.

Usage
-----
# From an h5ad file
python run_predict.py \\
    --checkpoint ./checkpoints/cancer_classification/final.ckpt \\
    --data /path/to/data.h5ad \\
    --output-dir ./predictions

# Override the transformer path (useful when running on a different machine)
python run_predict.py \\
    --checkpoint ./checkpoints/cancer_classification/final.ckpt \\
    --data /path/to/data.h5ad \\
    --pretrained-model-path /new/path/to/transformer.ckpt \\
    --output-dir ./predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    required = {"task", "model_state", "config", "output_dim", "embedding_dim"}
    missing = required - ckpt.keys()
    if missing:
        raise ValueError(
            f"Checkpoint is missing keys {missing}. "
            "Was it saved by a recent version of BaseDownstreamRunner?"
        )
    return ckpt


def _load_embedder(ckpt: dict, pretrained_model_path_override: str | None):
    from cancerfoundation.model.model import CancerFoundation

    config = ckpt["config"]
    pretrained_path = pretrained_model_path_override or config.get("pretrained_model_path")
    if not pretrained_path:
        raise ValueError(
            "No pretrained_model_path found in checkpoint config. "
            "Provide one via --pretrained-model-path."
        )

    log.info(f"Loading base transformer from {pretrained_path}")
    embedder = CancerFoundation.load_from_checkpoint(str(pretrained_path), strict=False)

    if ckpt.get("finetuned_embedder") and "embedder_state" in ckpt:
        log.info("Applying finetuned embedder state from checkpoint")
        embedder.load_state_dict(ckpt["embedder_state"])

    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False

    return embedder


def _build_head(task_name: str, ckpt: dict, device: torch.device):
    # Import task implementations to trigger registry population
    from evaluate.finetune.tasks import CancTypeClassTask, DeconvTask, SurvBoardTask, DrugSensitivityV2Task
    from evaluate.finetune.downstream_task import TaskRegistry

    task = TaskRegistry.get_task(task_name)
    head_class = task.get_head_class()
    config = ckpt["config"]
    head = head_class(
        embedding_dim=ckpt["embedding_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.0)),
    )
    head.load_state_dict(ckpt["model_state"])
    head.to(device)
    head.eval()
    return head


def _embed_finetune(
    embedder,
    gene_names: list[str],
    adata,
    normalized: bool,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Preprocess adata with the training gene set, then embed via the transformer."""
    processed = embedder.preprocess_for_embedding(
        adata, normalized=normalized, gene_subset=gene_names
    )
    expr = processed.X if isinstance(processed.X, np.ndarray) else processed.X.toarray()
    expr_tensor = torch.from_numpy(expr.astype(np.float32))
    gene_ids = torch.LongTensor([embedder.vocab[g] for g in gene_names])

    embedder.to(device)
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(expr_tensor), batch_size):
            batch = expr_tensor[i : i + batch_size].to(device)
            emb = embedder.embed_for_finetune(gene_ids, batch)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def _embed_frozen(
    embedder,
    adata,
    normalized: bool,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed adata using the standard frozen embed() path."""
    embedder.eval()
    embedder.to(device)
    result = embedder.embed(adata, batch_size=batch_size, normalized=normalized)
    df_emb = result[0] if isinstance(result, tuple) else result
    return df_emb.to_numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(
    checkpoint_path: str | Path,
    data_path: str | Path,
    pretrained_model_path: str | None = None,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Run predictions on an AnnData file using a saved downstream checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to final.ckpt produced by BaseDownstreamRunner.
    data_path:
        Path to input h5ad file.
    pretrained_model_path:
        Override for the transformer checkpoint path stored inside the downstream
        checkpoint. Required when running on a different machine.
    output_dir:
        Directory to write predictions.npy. Defaults to <checkpoint_dir>/predictions.
    batch_size:
        Inference batch size. Defaults to the value stored in the checkpoint config.

    Returns
    -------
    np.ndarray
        Raw model outputs (logits / regression values) for every input sample.
    """
    import anndata as ad

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ckpt = _load_checkpoint(checkpoint_path)

    task_name: str = ckpt["task"]
    config: dict = ckpt["config"]
    finetuned_embedder: bool = ckpt.get("finetuned_embedder", False)
    gene_names: list[str] | None = ckpt.get("gene_names")
    normalized: bool = bool(config.get("normalized", False))
    bs = batch_size or int(config.get("batch_size", 32))

    log.info(f"Task: {task_name}  |  finetuned_embedder={finetuned_embedder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    log.info(f"Loading data from {data_path}")
    adata = ad.read_h5ad(data_path)
    log.info(f"Loaded AnnData: {adata.shape}")

    embedder = _load_embedder(ckpt, pretrained_model_path)
    head = _build_head(task_name, ckpt, device)

    # Embed
    if finetuned_embedder:
        if gene_names is None:
            raise ValueError("Checkpoint has finetuned_embedder=True but no gene_names saved.")
        log.info(f"Fine-tuning mode: embedding with {len(gene_names)} training genes...")
        embeddings = _embed_finetune(embedder, gene_names, adata, normalized, device, bs)
    else:
        log.info("Frozen mode: embedding with transformer...")
        embeddings = _embed_frozen(embedder, adata, normalized, device, bs)

    log.info(f"Embeddings shape: {embeddings.shape}")

    # Head inference
    log.info("Running head inference...")
    emb_tensor = torch.from_numpy(embeddings.astype(np.float32))
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(emb_tensor), bs):
            batch = emb_tensor[i : i + bs].to(device)
            logits = head(batch)
            all_preds.append(logits.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    log.info(f"Predictions shape: {predictions.shape}")

    # Save
    if output_dir is None:
        output_dir = checkpoint_path.parent / "predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "predictions.npy", predictions)
    log.info(f"Saved predictions to {output_dir / 'predictions.npy'}")

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using a saved downstream task checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to final.ckpt produced by BaseDownstreamRunner.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input h5ad file.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Override the transformer checkpoint path stored in the downstream checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write predictions.npy. "
        "Defaults to <checkpoint_dir>/predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to the value in the checkpoint config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        predict(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            pretrained_model_path=args.pretrained_model_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        sys.exit(0)
    except Exception as e:
        log.exception(f"Prediction failed: {e}")
        sys.exit(1)
