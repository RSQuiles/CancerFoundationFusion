"""Minimal cancer type classification from precomputed embeddings.

Reads embeddings from adata.obsm, fits a two-layer MLP (same architecture as
EmbeddingPredHead in the finetune pipeline), and reports the same metrics:
accuracy, weighted F1, macro precision, and macro recall.

Usage
-----
    python evaluate/test/canc_type_pred.py \\
        --tcga-data /path/to/tcga_embedded.h5ad \\
        --obsm-key X_cf

    # Custom cohorts, without GBM/LGG merge:
    python evaluate/test/canc_type_pred.py \\
        --tcga-data /path/to/tcga_embedded.h5ad \\
        --cohorts BRCA LUAD UCEC \\
        --no-merge-gbm-lgg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_COHORTS = ["BRCA", "BLCA", "GBM", "LGG", "LUAD", "UCEC"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = torch.from_numpy(embeddings.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model — mirrors EmbeddingPredHead from evaluate/finetune/tasks/components.py
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Metrics — same as CancTypeClassTask.compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    pred_labels = predictions.argmax(axis=-1)
    acc = accuracy_score(targets, pred_labels)
    f1 = f1_score(targets, pred_labels, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        targets, pred_labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for emb, labels in loader:
        emb, labels = emb.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(emb), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for emb, labels in loader:
        all_preds.append(model(emb.to(device)).cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancer type classification from precomputed embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tcga-data", required=True, help="Path to TCGA h5ad file that contains embeddings."
    )
    parser.add_argument(
        "--obsm-key", default="X_cf", help="adata.obsm key holding the precomputed embeddings."
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=DEFAULT_COHORTS,
        metavar="COHORT",
        help="Cancer cohort labels (without the TCGA- prefix).",
    )
    parser.add_argument(
        "--no-merge-gbm-lgg",
        action="store_true",
        default=False,
        help="Keep GBM and LGG as separate classes instead of merging them into GBM_LGG.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data_path = Path(args.tcga_data)
    if not data_path.exists():
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)

    log.info(f"Loading {data_path}...")
    adata = sc.read_h5ad(str(data_path))
    log.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    if args.obsm_key not in adata.obsm:
        log.error(
            f"obsm key '{args.obsm_key}' not found. Available keys: {list(adata.obsm.keys())}"
        )
        sys.exit(1)

    if "project_id" not in adata.obs:
        log.error("adata.obs must contain a 'project_id' column (e.g. 'TCGA-BRCA').")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Filter cohorts and build labels
    # -------------------------------------------------------------------------
    selected = {f"TCGA-{c}" for c in args.cohorts}
    mask = adata.obs["project_id"].astype(str).isin(selected).to_numpy()
    adata = adata[mask].copy()
    log.info(f"After cohort filter ({args.cohorts}): {adata.n_obs} samples")

    if adata.n_obs == 0:
        log.error(f"No samples found for cohorts: {args.cohorts}")
        sys.exit(1)

    adata.obs["cancer_type"] = adata.obs["project_id"].astype(str).str.removeprefix("TCGA-")

    if not args.no_merge_gbm_lgg:
        gbm_lgg = adata.obs["cancer_type"].isin(["GBM", "LGG"])
        adata.obs.loc[gbm_lgg, "cancer_type"] = "GBM_LGG"
        log.info("Merged GBM and LGG → GBM_LGG")

    label_strs = adata.obs["cancer_type"].to_numpy()
    classes, label_ints = np.unique(label_strs, return_inverse=True)
    num_classes = len(classes)
    log.info(f"Classes ({num_classes}): {classes.tolist()}")

    # -------------------------------------------------------------------------
    # Train / test split
    # -------------------------------------------------------------------------
    embeddings = adata.obsm[args.obsm_key].astype(np.float32)

    train_idx, test_idx = train_test_split(
        np.arange(len(label_ints)),
        test_size=args.test_size,
        stratify=label_ints,
        random_state=args.seed,
    )
    log.info(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    train_loader = DataLoader(
        EmbeddingDataset(embeddings[train_idx], label_ints[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        EmbeddingDataset(embeddings[test_idx], label_ints[test_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Model, optimizer, loss
    # -------------------------------------------------------------------------
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    embedding_dim = embeddings.shape[1]
    model = MLP(embedding_dim, num_classes, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    log.info(f"Device: {device}  |  embedding_dim={embedding_dim}  |  num_classes={num_classes}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        if epoch == 1 or epoch % 5 == 0:
            log.info(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    preds, targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(preds, targets)

    print("\n--- Test metrics ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
