"""
Specific downstream task implementations.

Each task defines its own head, dataset, data loading, and metrics computation.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry

log = logging.getLogger(__name__)

# ============================================================================
# Common Components
# ============================================================================


class EmbeddingPredHead(nn.Module):
    """Generic prediction head for embedding-based downstream tasks."""

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        return self.fc2(x)


# ============================================================================
# Cancer Type Classification Task
# ============================================================================


class CancTypeClassEmbeddingDataset(Dataset):
    """Dataset for cancer type classification."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return emb, label


DEFAULT_COHORTS = ["BRCA", "BLCA", "GBM", "LGG", "LUAD", "UCEC"]


@TaskRegistry.register
class CancTypeClassTask(DownstreamTask):
    """Cancer type classification from single-cell and bulk TCGA data."""

    @property
    def task_name(self) -> str:
        return "canc_type_class"

    @property
    def config_key(self) -> str:
        return "finetune.canc_type_class"

    def get_head_class(self) -> type[nn.Module]:
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return CancTypeClassEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        return nn.CrossEntropyLoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        """Validate cancer type classification config."""
        super().validate_config(task_cfg)
        required = ["tcga_data_dir"]
        missing = [key for key in required if getattr(task_cfg, key, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """Load TCGA data and split into train/test. Determine output dimension"""
        import scanpy as sc

        # Load TCGA data
        tcga_path = getattr(task_cfg, "tcga_data_dir", None)
        if not tcga_path:
            raise ValueError("finetune.canc_type_class.tcga_data_dir must be set.")

        data_path = Path(hydra.utils.to_absolute_path(str(tcga_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA data not found at {data_path}")

        adata = sc.read_h5ad(data_path)
        log.info(f"Loaded TCGA data: {adata.shape}")

        # Filter by cohorts
        if "project_id" not in adata.obs:
            raise ValueError("TCGA data must have 'project_id' in obs")

        cohorts = list(getattr(task_cfg, "cohorts", DEFAULT_COHORTS))
        selected_projects = {f"TCGA-{cohort}" for cohort in cohorts}
        keep_mask = adata.obs["project_id"].astype(str).isin(selected_projects).to_numpy()
        adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(f"No samples found for cohorts: {cohorts}")

        # Add cancer type annotation
        adata.obs["cancer_type"] = adata.obs["project_id"].astype(str).str.removeprefix("TCGA-")
        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        log.info(f"Filtered to cohorts: {cohorts}, shape: {adata.shape}")

        # Optionally merge GBM and LGG
        if bool(getattr(task_cfg, "merge_gbm_lgg", True)):
            gbm_lgg_mask = adata.obs["cancer_type"].isin(["GBM", "LGG"])
            adata.obs.loc[gbm_lgg_mask, "cancer_type"] = "GBM_LGG"
            log.info("Merged GBM and LGG into single class")

        # Preprocessing
        adata = self._preprocess_adata(adata, task_cfg)

        # Train/test split
        labels = adata.obs["cancer_type"].astype(str)
        test_size = float(getattr(task_cfg, "test_size", 0.2))
        split_version = getattr(
            task_cfg,
            "train_test_split_version",
            getattr(task_cfg, "random_seed", 42),
        )
        split_seed = self.hash_split_version(split_version)

        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            stratify=labels,
            random_state=split_seed,
        )

        # Number of classes
        num_classes = labels.nunique()

        return num_classes, adata[train_idx].copy(), adata[test_idx].copy(), labels.iloc[train_idx].to_numpy(), labels.iloc[test_idx].to_numpy()

    def _preprocess_adata(self, adata: ad.AnnData, task_cfg: DictConfig) -> ad.AnnData:
        """Preprocess and normalize data."""
        import scanpy as sc

        normalized = bool(getattr(task_cfg, "normalized", True))
        if not normalized:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            log.info("Applied log normalization")

        return adata

    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Create embeddings and datasets."""
        # Encode labels
        label_dict, train_labels_encoded = np.unique(train_targets, return_inverse=True)
        label_to_idx = {label: idx for idx, label in enumerate(label_dict.tolist())}
        test_labels_encoded = np.array(
            [label_to_idx[label] for label in test_targets],
            dtype=np.int64,
        )

        # Generate embeddings
        train_embeddings = self._embed_adata(embedder, train_adata)
        test_embeddings = self._embed_adata(embedder, test_adata)

        if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_cells, embedding_dim].")
        if train_embeddings.shape[1] != test_embeddings.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        embedding_dim = int(train_embeddings.shape[1])

        # Create datasets
        train_dataset = CancTypeClassEmbeddingDataset(train_embeddings, train_labels_encoded)
        test_dataset = CancTypeClassEmbeddingDataset(test_embeddings, test_labels_encoded)

        log.info(f"Created datasets with embedding_dim={embedding_dim}")

        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData) -> np.ndarray:
        """Generate embeddings for adata."""
        batch_size = 64
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, flavor="seurat_v3")
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Compute classification metrics."""
        # Convert logits to predictions
        pred_labels = predictions.argmax(axis=-1)
        
        acc = accuracy_score(targets, pred_labels)
        f1 = f1_score(targets, pred_labels, average="weighted", zero_division=0)
        
        metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
        }
        
        # Precision, recall per class
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            targets, pred_labels, average=None, zero_division=0
        )
        metrics["precision_macro"] = float(np.mean(precision))
        metrics["recall_macro"] = float(np.mean(recall))
        
        return metrics


# ============================================================================
# Deconvolution Task
# ============================================================================


class DeconvEmbeddingDataset(Dataset):
    """Dataset for deconvolution (cell type proportion prediction)."""

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        targets = np.asarray(targets, dtype=np.float32)
        target_sums = targets.sum(axis=1, keepdims=True)
        if np.any(target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.targets = targets / target_sums

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        target = torch.from_numpy(self.targets[index]).float()
        return emb, target


@TaskRegistry.register
class DeconvTask(DownstreamTask):
    """Cell type proportion deconvolution from bulk pseudobulk samples."""

    @property
    def task_name(self) -> str:
        return "deconv"

    @property
    def config_key(self) -> str:
        return "finetune.deconv"

    def get_head_class(self) -> type[nn.Module]:
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return DeconvEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        # Deconvolution uses KL divergence by default
        return nn.KLDivLoss(reduction="batchmean").to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        """Validate deconvolution config."""
        super().validate_config(task_cfg)
        required = ["pseudo_bulk_data_path"]
        missing = [key for key in required if getattr(task_cfg, key, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """Load pseudobulk data and cell type proportions."""
        # Load data
        pb_path = getattr(task_cfg, "pseudo_bulk_data_path", None)
        if not pb_path:
            raise ValueError("finetune.deconv.pseudo_bulk_data_path must be set.")

        data_path = Path(hydra.utils.to_absolute_path(str(pb_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"Pseudobulk data not found at {data_path}")

        adata = ad.read_h5ad(data_path)
        adata.var_names_make_unique()
        log.info(f"Loaded pseudobulk data: {adata.shape}")

        # Get cell type proportions
        targets = self._load_targets(adata, task_cfg)

        # Train/test split (optionally by context)
        test_size = float(getattr(task_cfg, "test_size", 0.2))
        split_version = getattr(
            task_cfg,
            "train_test_split_version",
            getattr(task_cfg, "random_seed", 42),
        )
        split_seed = self.hash_split_version(split_version)

        if bool(getattr(task_cfg, "split_by_context", True)):
            # Split by context variable (e.g., study)
            contexts = adata.obs.get(getattr(task_cfg, "context_col", "context"), None)
            if contexts is not None:
                unique_contexts = contexts.unique()
                split_idx = int(len(unique_contexts) * (1 - test_size))
                test_contexts = unique_contexts[split_idx:]
                test_mask = contexts.isin(test_contexts).to_numpy()
                train_idx = np.where(~test_mask)[0]
                test_idx = np.where(test_mask)[0]
            else:
                log.warning("context_col not found, falling back to random split")
                train_idx, test_idx = train_test_split(
                    np.arange(adata.n_obs),
                    test_size=test_size,
                    random_state=split_seed,
                )
        else:
            train_idx, test_idx = train_test_split(
                np.arange(adata.n_obs),
                test_size=test_size,
                random_state=split_seed,
            )

        # Number of classes
        num_classes = None

        return (
            num_classes,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _load_targets(self, adata: ad.AnnData, task_cfg: DictConfig) -> np.ndarray:
        """Load cell type proportions from adata.obs."""
        # Get cell type mapping
        cell_types = self._as_string_list(adata.uns.get("cell_type_proportion_cell_types"))
        obs_columns = self._as_string_list(adata.uns.get("cell_type_proportion_obs_columns"))

        if cell_types is not None and obs_columns is not None:
            mapping = dict(zip(cell_types, obs_columns))
        else:
            # Try to infer or use custom mapping
            mapping = getattr(task_cfg, "cell_type_mapping", None)
            if mapping is None:
                log.info("Inferring cell type proportions!")
                mapping = self._infer_proportion_mapping_from_obs(adata.obs.columns)

        self.cell_types = sorted(mapping)
        target_columns = [mapping[cell_type] for cell_type in self.cell_types]

        missing = [col for col in target_columns if col not in adata.obs]
        if missing:
            raise ValueError(f"Missing target columns in adata.obs: {missing}")

        targets = adata.obs[target_columns].to_numpy(dtype=np.float32)
        if np.any(targets < 0):
            raise ValueError("Cell type proportions must be non-negative")

        return targets

    @staticmethod
    def _as_string_list(values: Any) -> list[str] | None:
        """Convert values to string list if not None."""
        if values is None:
            return None
        array = np.asarray(values).reshape(-1)
        return [str(v) for v in array]

    @staticmethod
    def _infer_proportion_mapping_from_obs(obs_columns) -> dict[str, str]:
        """Infer cell type mapping from obs column names (e.g., B_cells, T_cells)."""
        cell_type_cols = [col for col in obs_columns if any(
            ct in col.lower() for ct in ["cell", "type", "prop"]
        )]
        return {col: col for col in cell_type_cols}

    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Create embeddings and deconvolution datasets."""
        # Generate embeddings
        train_embeddings = self._embed_adata(embedder, train_adata)
        test_embeddings = self._embed_adata(embedder, test_adata)

        if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_cells, embedding_dim].")
        if train_embeddings.shape[1] != test_embeddings.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        embedding_dim = int(train_embeddings.shape[1])

        # Create datasets
        train_dataset = DeconvEmbeddingDataset(train_embeddings, train_targets)
        test_dataset = DeconvEmbeddingDataset(test_embeddings, test_targets)

        log.info(f"Created deconv datasets with embedding_dim={embedding_dim}, num_cell_types={len(self.cell_types)}")

        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData) -> np.ndarray:
        """Generate embeddings for adata."""
        batch_size = 64
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, flavor="seurat_v3")
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Compute deconvolution regression metrics."""
        # Convert logits to proportions (softmax)
        pred_props = torch.from_numpy(predictions)
        pred_props = F.softmax(pred_props, dim=-1).numpy()

        # MAE and MSE
        mae = np.mean(np.abs(pred_props - targets))
        mse = np.mean((pred_props - targets) ** 2)

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
        }

        return metrics
