from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
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
from evaluate.finetune.utils import parquet_to_adata, translate_gene_symbols, strip_ensembl_versions, deduplicate_var_names

log = logging.getLogger(__name__)


# ============================================================================
# Deconvolution Task (Pseudobulk)
# ============================================================================
#
# Task: Given a pseudobulk RNA-seq profile aggregated from single cells, predict
# the underlying cell type proportion vector. This tests whether the foundation
# model captures cell-type-specific transcriptional signatures in bulk data.
#
# Data: Pseudobulk h5ad file where each observation is a bulk-aggregated sample.
#   - X matrix: gene expression (pseudobulk aggregated from single cells).
#   - obs columns: per-cell-type proportion values (floats summing to 1).
#   - uns key: 'cell_type_proportion_columns' — a dict mapping cell type name
#              to the obs column that stores its proportion. If absent, the
#              mapping is taken from 'cell_type_mapping' in the config, or
#              auto-detected from obs column names as a fallback.
#
#   Splitting is preferably done by context variable (e.g. study/donor) so that
#   the model is evaluated on held-out contexts rather than held-out samples from
#   seen contexts — this tests generalisation across biological conditions.
# ============================================================================


class DeconvEmbeddingDataset(Dataset):
    """Dataset for deconvolution (cell type proportion prediction)."""

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        target_sums = targets.sum(axis=1)
        valid_mask = target_sums > 0
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            log.warning(f"Dropping {n_dropped} samples with non-positive target sums.")
            self.embeddings = self.embeddings[valid_mask]
            targets = targets[valid_mask]
            target_sums = target_sums[valid_mask]

        self.targets = targets / target_sums[:, np.newaxis]

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

        # Number of classes (cell types covered by the dataset)
        num_classes = targets.shape[1]

        return (
            num_classes,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _load_targets(self, adata: ad.AnnData, task_cfg: DictConfig) -> np.ndarray:
        """Load cell type proportions from adata.obs."""

        def recurse_prop_value(value):
            if isinstance(value, dict):
                value_rec = list(value.keys())[0]
                return recurse_prop_value(value[value_rec])
            return value

        # Get cell type mapping (cell type → obs column)
        mapping = adata.uns.get("cell_type_proportion_columns")
        for key, value in mapping.items():
            mapping[key] = recurse_prop_value(value)

        if mapping is not None:
            log.info("Using cell type mapping from adata.uns['cell_type_proportion_columns']")
        else:
            # Try to infer or use custom mapping
            mapping = getattr(task_cfg, "cell_type_mapping", None)
            if mapping is None:
                log.info("Inferring cell type proportions!")
                mapping = self._infer_proportion_mapping_from_obs(adata.obs.columns)

        self.cell_types = sorted(mapping)
        target_columns = [mapping[cell_type] for cell_type in self.cell_types]

        missing = [col for col in target_columns if col not in adata.obs.columns]
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
        df_emb = embedder.embed(adata, batch_size=batch_size, normalized=True)
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