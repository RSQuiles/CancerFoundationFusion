from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry

log = logging.getLogger(__name__)


# ============================================================================
# Deconvolution Task (Pseudobulk)
# ============================================================================
#
# Task: Given a pseudobulk RNA-seq profile aggregated from single cells, predict
# the underlying cell type proportion vector. This tests whether the foundation
# model captures cell-type-specific transcriptional signatures in bulk data.
#
# Data: pseudo_bulk_RAW.h5ad — each obs is a pseudobulk sample built by summing
#   raw counts of 1000 single cells (CELLxGENE Census).
#   - X matrix: raw summed counts (sparse CSR float32). Preprocessed via
#               CP10K + log1p normalization before passing to the embedder.
#   - obs columns: prop__<cell_type> floats summing to 1.0 per row.
#   - var index: ensembl_id — Ensembl gene IDs.
#   - uns keys:
#       cell_type_proportion_columns      dict: cell_type → obs column name
#       cell_type_proportion_obs_columns  ordered array of obs column names
#       cell_type_proportion_cell_types   ordered array of cell type strings
#
#   Splitting is done by composite context (dataset_id × donor_id ×
#   tissue_general) so held-out samples come from genuinely unseen contexts,
#   not just held-out draws from seen donors/datasets.
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
    """Cell type proportion deconvolution from pseudobulk samples."""

    @property
    def task_name(self) -> str:
        return "deconv"

    @property
    def config_key(self) -> str:
        return "finetune.deconv"

    def get_head_class(self) -> type[nn.Module]:
        from evaluate.finetune.tasks.components import EmbeddingPredHead
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return DeconvEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        return nn.KLDivLoss(reduction="batchmean").to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
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
        pb_path = getattr(task_cfg, "pseudo_bulk_data_path", None)
        if not pb_path:
            raise ValueError("finetune.deconv.pseudo_bulk_data_path must be set.")

        data_path = Path(hydra.utils.to_absolute_path(str(pb_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"Pseudobulk data not found at {data_path}")

        adata = ad.read_h5ad(data_path)
        adata.var_names_make_unique()
        log.info(f"Loaded pseudobulk data: {adata.shape}")

        # Store task_cfg for use in _embed_adata
        self._task_cfg = task_cfg

        targets = self._load_targets(adata, task_cfg)

        # Train/test split
        test_size = float(getattr(task_cfg, "test_size", 0.2))
        split_version = getattr(
            task_cfg,
            "train_test_split_version",
            getattr(task_cfg, "random_seed", 42),
        )
        split_seed = self.hash_split_version(split_version)

        if bool(getattr(task_cfg, "split_by_context", True)):
            contexts = self._build_context_series(adata, task_cfg)
            if contexts is not None:
                unique_contexts = contexts.unique()
                split_idx = int(len(unique_contexts) * (1 - test_size))
                test_contexts = unique_contexts[split_idx:]
                test_mask = contexts.isin(test_contexts).to_numpy()
                train_idx = np.where(~test_mask)[0]
                test_idx = np.where(test_mask)[0]
            else:
                log.warning("No context columns found, falling back to random split.")
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

        num_classes = targets.shape[1]
        log.info(
            f"Split: {len(train_idx)} train / {len(test_idx)} test samples, "
            f"{num_classes} cell types."
        )

        return (
            num_classes,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _build_context_series(self, adata: ad.AnnData, task_cfg: DictConfig) -> pd.Series | None:
        """Build a composite context Series for context-aware splitting.

        Checks config for context_cols (list) first, then context_col (single),
        then falls back to the standard pseudobulk context columns.
        """
        # Priority 1: explicit list of columns from config
        context_cols = getattr(task_cfg, "context_cols", None)
        if context_cols is not None:
            cols = list(context_cols)
        else:
            # Priority 2: single column from config
            single = getattr(task_cfg, "context_col", None)
            if single is not None:
                cols = [str(single)]
            else:
                # Priority 3: default pseudobulk context columns
                cols = ["dataset_id", "donor_id", "tissue_general"]

        valid_cols = [c for c in cols if c in adata.obs.columns]
        if not valid_cols:
            return None

        if len(valid_cols) == 1:
            return adata.obs[valid_cols[0]].astype(str)

        return adata.obs[valid_cols].astype(str).agg("__".join, axis=1)

    def _load_targets(self, adata: ad.AnnData, task_cfg: DictConfig) -> np.ndarray:
        """Load cell type proportions from adata.obs.

        Handles both the new uns key structure (cell_type_proportion_obs_columns /
        cell_type_proportion_cell_types) and the legacy dict mapping.
        """
        # New format: explicit ordered arrays in uns
        obs_columns_arr = adata.uns.get("cell_type_proportion_obs_columns")
        cell_types_arr = adata.uns.get("cell_type_proportion_cell_types")

        if obs_columns_arr is not None and cell_types_arr is not None:
            prop_columns = list(obs_columns_arr)
            self.cell_types = list(cell_types_arr)
            log.info(
                f"Using {len(self.cell_types)} cell types from "
                "adata.uns['cell_type_proportion_obs_columns']."
            )
        else:
            # Legacy format: mapping dict (cell_type → column_name)
            mapping = adata.uns.get("cell_type_proportion_columns")
            if mapping is not None:
                mapping = {k: self._recurse_prop_value(v) for k, v in mapping.items()}
                log.info(
                    "Using cell type mapping from adata.uns['cell_type_proportion_columns']."
                )
            else:
                mapping = getattr(task_cfg, "cell_type_mapping", None)
                if mapping is None:
                    log.info("Inferring cell type proportions from obs column names.")
                    mapping = self._infer_proportion_mapping_from_obs(adata.obs.columns)
            self.cell_types = sorted(mapping)
            prop_columns = [mapping[ct] for ct in self.cell_types]

        missing = [col for col in prop_columns if col not in adata.obs.columns]
        if missing:
            raise ValueError(f"Missing target columns in adata.obs: {missing}")

        targets = adata.obs[prop_columns].to_numpy(dtype=np.float32)
        if np.any(targets < 0):
            raise ValueError("Cell type proportions must be non-negative.")

        return targets

    @staticmethod
    def _recurse_prop_value(value: Any) -> str:
        """Unwrap nested dicts to reach the leaf column name."""
        if isinstance(value, dict):
            return DeconvTask._recurse_prop_value(list(value.values())[0])
        return value

    @staticmethod
    def _infer_proportion_mapping_from_obs(obs_columns) -> dict[str, str]:
        """Infer cell type mapping from obs column names (e.g., prop__b_cell)."""
        cell_type_cols = [
            col for col in obs_columns
            if any(kw in col.lower() for kw in ["prop__", "cell", "type", "prop"])
        ]
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
        train_embeddings = self._embed_adata(embedder, train_adata)
        test_embeddings = self._embed_adata(embedder, test_adata)

        if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_samples, embedding_dim].")
        if train_embeddings.shape[1] != test_embeddings.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        embedding_dim = int(train_embeddings.shape[1])

        train_dataset = DeconvEmbeddingDataset(train_embeddings, train_targets)
        test_dataset = DeconvEmbeddingDataset(test_embeddings, test_targets)

        log.info(
            f"Created deconv datasets: embedding_dim={embedding_dim}, "
            f"num_cell_types={len(self.cell_types)}"
        )

        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData) -> np.ndarray:
        """Normalize raw counts (CP10K + log1p) then embed with the frozen model."""
        task_cfg = getattr(self, "_task_cfg", None)
        normalize = bool(getattr(task_cfg, "normalize_input", True)) if task_cfg is not None else True

        if normalize:
            adata = adata.copy()
            X = adata.X
            if sp.issparse(X):
                row_sums = np.asarray(X.sum(axis=1)).ravel()
                row_sums = np.maximum(row_sums, 1.0)
                X_norm = X.multiply(10_000.0 / row_sums[:, None]).toarray().astype(np.float32)
            else:
                row_sums = np.maximum(np.asarray(X).sum(axis=1), 1.0)
                X_norm = (np.asarray(X) * (10_000.0 / row_sums[:, np.newaxis])).astype(np.float32)
            np.log1p(X_norm, out=X_norm)
            adata.X = X_norm

        batch_size = int(getattr(task_cfg, "embed_batch_size", 64)) if task_cfg is not None else 64
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, normalized=True)
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Compute deconvolution regression metrics.

        Includes MAE, MSE, RMSE, and mean per-cell-type Pearson R (computed
        only over cell types with non-zero variance in both true and predicted).
        """
        pred_props = F.softmax(torch.from_numpy(predictions), dim=-1).numpy()

        mae = float(np.mean(np.abs(pred_props - targets)))
        mse = float(np.mean((pred_props - targets) ** 2))

        # Per-cell-type Pearson R (skip constant columns to avoid NaN)
        pearson_rs = []
        for i in range(targets.shape[1]):
            true_col = targets[:, i]
            pred_col = pred_props[:, i]
            if np.std(true_col) > 0 and np.std(pred_col) > 0:
                r, _ = pearsonr(true_col, pred_col)
                pearson_rs.append(float(r))

        mean_pearson_r = float(np.mean(pearson_rs)) if pearson_rs else float("nan")

        return {
            "mae": mae,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mean_pearson_r": mean_pearson_r,
        }
