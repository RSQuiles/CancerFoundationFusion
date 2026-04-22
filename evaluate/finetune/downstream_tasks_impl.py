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
from evaluate.finetune.utils import parquet_to_adata, CoxPHLoss, MaskedMSELoss

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
# Cancer Type Classification Task (TCGA)
# ============================================================================
#
# Task: Given a bulk RNA-seq sample from a TCGA cancer patient, predict the
# cancer type (cohort label). This tests whether the foundation model produces
# embeddings that separate tumour types without task-specific supervision.
#
# Data: TCGA (The Cancer Genome Atlas)
#   Default cohorts: BRCA, BLCA, GBM, LGG, LUAD, UCEC (configurable).
#   - Single h5ad file with obs column 'project_id' (e.g. "TCGA-BRCA").
#
#   GBM and LGG are optionally merged into a single "GBM_LGG" class (default:
#   True) because they represent histologically related glioma subtypes and the
#   distinction is driven more by grade than by transcriptional programme.
#   Train/test split is stratified by cancer type to preserve class balance.
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

        # Optionally merge GBM and LGGt. True by default
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
        # Get cell type mapping (cell type → obs column)
        mapping = adata.uns.get("cell_type_proportion_columns")

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


# ============================================================================
# Survival Prediction Task (SurvBoard)
# ============================================================================
#
# Task: Given a bulk RNA-seq sample from a cancer patient, predict a scalar
# risk score. Higher score = shorter predicted survival. The model is trained
# with the Cox Proportional Hazards partial log-likelihood and evaluated with
# Harrell's concordance index (C-index).
#
# Data: SurvBoard benchmark (Wissel, Boeva et al., Briefings in Bioinformatics 2025)
#   Multiple cohorts (e.g. TCGA, METABRIC, ICGC, TARGET) across cancer types.
#   File naming: {COHORT}_{CANCER}.parquet — expression columns + OS_days + OS_event.
#   The same cancer type may appear in several cohorts (e.g. TCGA_BRCA, METABRIC_BRCA).
#   Optional pre-made CV splits:
#     {splits_dir}/{COHORT}/{CANCER}_train_{fold_index}.csv
#     {splits_dir}/{COHORT}/{CANCER}_test_{fold_index}.csv
#     Each CSV contains integer row indices (positions in the parquet file).
#     If splits_dir is not set or any file is missing, the whole dataset falls
#     back to a stratified random split by cancer type.
# ============================================================================


class SurvivalEmbeddingDataset(Dataset):
    """
    Dataset for survival risk prediction.

    Each item is (embedding, [OS_days, OS_event]) so that CoxPHLoss can receive
    both the event time and the censoring indicator alongside the risk score in
    a single targets tensor.
    """

    def __init__(self, embeddings: np.ndarray, times: np.ndarray, events: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.times  = np.asarray(times,  dtype=np.float32)
        self.events = np.asarray(events, dtype=np.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        # Pack time + event into a 2-element vector consumed by CoxPHLoss
        target = torch.tensor([self.times[index], self.events[index]], dtype=torch.float32)
        return emb, target
    

class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood loss for the Cox Proportional Hazards model.

    For a batch of scalar risk scores h_i and corresponding (time_i, event_i)
    pairs, the Cox partial log-likelihood is:

        PLL = Σ_{i: event_i=1} [ h_i - log( Σ_{j: t_j ≥ t_i} exp(h_j) ) ]

    We negate and normalise by the number of observed events to obtain the loss.
    The inner log-sum-exp is computed via torch.logcumsumexp after sorting by
    descending event time, which is numerically stable and avoids O(n²) loops.

    After sorting descending, position k's "risk set" (all samples still alive
    at time[k]) corresponds exactly to the prefix {0 … k}, so logcumsumexp
    gives the correct denominator at every position in a single pass.

    Input
    -----
    risk    : Tensor, shape (N,) or (N, 1)
              Scalar risk scores — higher value = shorter predicted survival.
    targets : Tensor, shape (N, 2)
              Column 0 = OS_days (float), column 1 = OS_event (0 or 1 float).
    """

    def forward(self, risk: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        risk = risk.squeeze(-1)          # (N, 1) → (N,)
        time  = targets[:, 0]
        event = targets[:, 1]

        if event.sum() == 0:
            # No observed events in this batch — Cox loss is undefined.
            # Return a zero-gradient scalar so training continues gracefully.
            return risk.sum() * 0.0

        # Sort by descending event time so that a prefix sum gives the risk set.
        order       = torch.argsort(time, descending=True)
        risk_sorted = risk[order]
        event_sorted = event[order]

        # log Σ_{j in risk_set(i)} exp(h_j) for every position i
        log_risk_set = torch.logcumsumexp(risk_sorted, dim=0)

        # Partial log-likelihood, summed only over observed events
        pll      = (event_sorted * (risk_sorted - log_risk_set)).sum()
        n_events = event_sorted.sum()

        return -pll / n_events


def _c_index(
    event_times: np.ndarray,
    risk_scores: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """
    Harrell's concordance index (C-index) for survival models.

    A pair (i, j) is *permissible* when:
      - sample i had an observed event (event_observed[i] = True)
      - sample j was still alive when i died  (event_times[j] > event_times[i])

    The pair is *concordant* if the model assigned higher risk to i (who died
    sooner). A random predictor gives C = 0.5; a perfect predictor gives C = 1.

    Uses lifelines.utils.concordance_index if available (O(n log n)), otherwise
    falls back to a vectorised numpy O(n²) implementation.
    """
    try:
        from lifelines.utils import concordance_index
        # lifelines convention: higher predicted_time = longer survival.
        # Negate risk scores so "high risk" maps to "short predicted time".
        return float(concordance_index(event_times, -risk_scores, event_observed))
    except ImportError:
        pass

    # Vectorised numpy fallback — O(n²) in memory, acceptable for eval set sizes.
    event_mask = event_observed.astype(bool)
    et      = event_times[event_mask]    # (n_events,)
    rs_evt  = risk_scores[event_mask]    # (n_events,)

    # at_risk[i, j] = True if sample j was still alive when event i occurred
    at_risk   = event_times[None, :] - et[:, None] > 0      # (n_events, n_all)
    risk_diff = rs_evt[:, None] - risk_scores[None, :]       # (n_events, n_all)

    concordant  = float(((risk_diff > 0)  & at_risk).sum())
    tied        = float(((risk_diff == 0) & at_risk).sum())
    permissible = float(at_risk.sum())

    return (concordant + 0.5 * tied) / permissible if permissible > 0 else 0.5


@TaskRegistry.register
class SurvivalTask(DownstreamTask):
    """
    Survival risk prediction on the SurvBoard benchmark.

    The frozen CancerFoundation encoder produces a sample-level embedding from
    bulk RNA-seq data. A 2-layer head then predicts a scalar log-risk score
    (higher = shorter predicted survival). Training uses the Cox partial
    log-likelihood; evaluation uses Harrell's C-index.

    Data files are named {COHORT}_{CANCER}.parquet (e.g. TCGA_BRCA.parquet).
    Multiple cohorts and cancer types can be requested; all matching files are
    concatenated before training. When splits_dir is configured, pre-made CV
    folds are loaded per (cohort, cancer) block from:
        {splits_dir}/{COHORT}/{CANCER}_train_splits.csv
        {splits_dir}/{COHORT}/{CANCER}_test_splits.csv
    Each CSV has no header or row index. Each row corresponds to one CV fold
    and contains the integer row indices (into the parquet) for that fold.
    fold_index selects which row to use. If any file is missing, the whole
    dataset falls back to a stratified random split by cancer type.

    Config key: finetune.survival
    Required config fields:
        survboard_data_dir  Path to the processed SurvBoard directory.
        cancer_types        List of cancer type strings, e.g. ["BRCA", "LUAD"].
        cohorts             List of cohort names, e.g. ["TCGA", "METABRIC"].
    """

    @property
    def task_name(self) -> str:
        return "survival"

    @property
    def config_key(self) -> str:
        return "finetune.survival"

    def get_head_class(self) -> type[nn.Module]:
        # output_dim=1 → scalar risk score per sample
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return SurvivalEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        return CoxPHLoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)
        required = ["survboard_data_dir", "cancer_types", "cohorts"]
        missing = [k for k in required if getattr(task_cfg, k, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """
        Load SurvBoard parquet files for the requested cohorts and cancer types.

        Files are named {COHORT}_{CANCER}.parquet and contain gene-expression
        columns plus OS_days and OS_event. All found files are concatenated into
        a single AnnData. block_info tracks each (cohort, cancer, size) block so
        that local split indices can be mapped to global positions.

        Split priority:
          1. If splits_dir is set, load pre-made CV folds for every (cohort,
             cancer) block from {splits_dir}/{COHORT}/{CANCER}_train_splits.csv
             and {splits_dir}/{COHORT}/{CANCER}_test_splits.csv. Each CSV has no
             header or row index; each row is one fold, each value is a row index
             into the parquet. fold_index selects the row. If any file is missing,
             falls back to (2).
          2. Stratified random split by cancer type.

        Returns
        -------
        (1, train_adata, test_adata, train_targets, test_targets)
            num_classes=1 → runner sets output_dim=1 (scalar risk score).
            targets shape: (N, 2) = [OS_days, OS_event].
        """
        data_dir     = Path(hydra.utils.to_absolute_path(str(task_cfg.survboard_data_dir)))
        cancer_types = list(task_cfg.cancer_types)
        cohorts      = list(task_cfg.cohorts)

        dfs: list[pd.DataFrame] = []
        block_info: list[tuple[str, str, int]] = []  # (cohort, cancer_type, block_size)

        for cohort in cohorts:
            for ct in cancer_types:
                parquet_path = data_dir / f"{cohort}_{ct}.parquet"
                if not parquet_path.exists():
                    log.warning(f"SurvBoard parquet not found, skipping: {parquet_path}")
                    continue
                df = pd.read_parquet(parquet_path)
                df["_cohort"]      = cohort
                df["_cancer_type"] = ct
                block_info.append((cohort, ct, len(df)))
                dfs.append(df)

        if not dfs:
            raise ValueError(
                f"No SurvBoard parquet files found for cohorts={cohorts}, "
                f"cancer_types={cancer_types} in {data_dir}."
            )

        df_all = pd.concat(dfs, axis=0)
        df_all.index = df_all.index.astype(str)

        # Separate expression from survival labels
        meta_cols = {"OS_days", "OS_event", "_cancer_type", "_cohort"}
        gene_cols = [c for c in df_all.columns if c not in meta_cols]

        times  = df_all["OS_days"].to_numpy(dtype=np.float32)
        events = df_all["OS_event"].to_numpy(dtype=np.float32)

        # Targets packed as (N, 2) = [time, event] for CoxPHLoss
        targets = np.stack([times, events], axis=1)

        adata = parquet_to_adata(df_all, gene_cols)
        adata.obs["cancer_type"] = df_all["_cancer_type"].values
        adata.obs["cohort"]      = df_all["_cohort"].values
        adata.obs["OS_days"]     = times
        adata.obs["OS_event"]    = events

        train_idx, test_idx = self._make_split(adata, task_cfg, block_info)

        log.info(
            f"SurvBoard loaded: {adata.n_obs} samples, {len(gene_cols)} genes, "
            f"{int(events.sum())} events. Train: {len(train_idx)}, Test: {len(test_idx)}"
        )

        # Number of classes = 1: single scalar risk score per sample
        num_classes = 1

        return (
            num_classes,                   # num_classes=1 (risk score)
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _make_split(
        self,
        adata: ad.AnnData,
        task_cfg: DictConfig,
        block_info: list[tuple[str, str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (train_indices, test_indices) into the concatenated adata.

        For each (cohort, cancer) block, tries to load:
            {splits_dir}/{COHORT}/{CANCER}_train_splits.csv
            {splits_dir}/{COHORT}/{CANCER}_test_splits.csv
        Each CSV has no header or row index. Each row is one CV fold; each value
        in that row is an integer row index into the corresponding parquet file.
        fold_index selects which row. A running offset converts local indices to
        global positions in the concatenated array.
        If splits_dir is not set or any file is missing, falls back to a
        stratified random split by cancer type over the whole dataset.
        """
        splits_dir = getattr(task_cfg, "splits_dir", None)
        fold_index = int(getattr(task_cfg, "fold_index", 0))

        if splits_dir is not None:
            splits_base   = Path(hydra.utils.to_absolute_path(str(splits_dir)))
            all_train:    list[np.ndarray] = []
            all_test:     list[np.ndarray] = []
            global_offset = 0
            all_loaded    = True

            for cohort, ct, block_size in block_info:
                train_path = splits_base / cohort / f"{ct}_train_splits.csv"
                test_path  = splits_base / cohort / f"{ct}_test_splits.csv"

                if train_path.exists() and test_path.exists():
                    try:
                        local_train = (
                            pd.read_csv(train_path, header=None)
                            .iloc[fold_index]
                            .dropna()
                            .astype(int)
                            .to_numpy()
                        )
                        local_test = (
                            pd.read_csv(test_path, header=None)
                            .iloc[fold_index]
                            .dropna()
                            .astype(int)
                            .to_numpy()
                        )
                        all_train.append(local_train + global_offset)
                        all_test.append(local_test   + global_offset)
                    except Exception as exc:
                        log.warning(
                            f"Could not load splits for {cohort}/{ct} fold {fold_index}: {exc}"
                        )
                        all_loaded = False
                        break
                else:
                    log.warning(
                        f"Split files not found for {cohort}/{ct} fold {fold_index} "
                        f"(expected {train_path} and {test_path})"
                    )
                    all_loaded = False
                    break

                global_offset += block_size

            if all_loaded:
                log.info(f"Loaded SurvBoard pre-made splits: fold={fold_index}")
                return np.concatenate(all_train), np.concatenate(all_test)

            log.warning("Falling back to stratified random split.")

        # Stratified random split — stratify by cancer type so each type has
        # proportional representation in both train and test.
        test_size  = float(getattr(task_cfg, "test_size", 0.2))
        split_seed = self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1))
        strata     = adata.obs["cancer_type"].to_numpy()

        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            stratify=strata,
            random_state=split_seed,
        )
        return train_idx, test_idx

    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Generate CancerFoundation embeddings and wrap into SurvivalEmbeddingDataset."""
        train_emb = self._embed_adata(embedder, train_adata)
        test_emb  = self._embed_adata(embedder, test_adata)
        embedding_dim = train_emb.shape[1]

        # Unpack the (N, 2) target arrays into separate time / event arrays
        train_dataset = SurvivalEmbeddingDataset(
            train_emb, train_targets[:, 0], train_targets[:, 1]
        )
        test_dataset = SurvivalEmbeddingDataset(
            test_emb, test_targets[:, 0], test_targets[:, 1]
        )

        log.info(f"Survival datasets ready. embedding_dim={embedding_dim}")
        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData, batch_size: int = 64) -> np.ndarray:
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, flavor="seurat_v3")
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute Harrell's C-index from predicted risk scores.

        Parameters
        ----------
        predictions : np.ndarray, shape (N, 1)
            Scalar risk scores from the head (higher = worse prognosis).
        targets : np.ndarray, shape (N, 2)
            Column 0 = OS_days, column 1 = OS_event.
        """
        # Head outputs (N, 1) — squeeze to (N,)
        risk_scores    = predictions[:, 0] if predictions.ndim == 2 else predictions
        event_times    = targets[:, 0]
        event_observed = targets[:, 1].astype(bool)

        c_idx      = _c_index(event_times, risk_scores, event_observed)
        n_events   = int(event_observed.sum())
        event_rate = float(n_events / max(len(event_observed), 1))

        log.info(
            f"Survival: C-index={c_idx:.4f}, n_events={n_events}, "
            f"event_rate={event_rate:.2f}"
        )

        return {
            "c_index":    float(c_idx),
            "n_events":   float(n_events),
            "event_rate": event_rate,
        }


# ============================================================================
# Proteome Prediction Task (CPTAC)
# ============================================================================
#
# Task: Given a bulk RNA-seq sample from a cancer patient, predict the matched
# protein abundance vector. This tests whether the foundation model encodes
# RNA-to-protein relationships beyond simple transcript-level correlation.
#
# Data: CPTAC (Clinical Proteomic Tumor Analysis Consortium)
#   10 cancer types: BRCA, CCRCC, COAD, GBM, HNSCC, LUAD, LSCC, OV, PDAC, UCEC.
#   - rna_expression.parquet     (samples × genes, HGNC gene symbols as columns)
#   - protein_expression.parquet (samples × proteins, same gene-symbol names,
#                                  log2-ratio TMT values)
#   - gene_protein_mapping.csv   3 columns: (0) gene symbol, (1) has_rna (bool),
#                                  (2) has_protein (bool).
#                                  Genes with has_rna=True are used as RNA input;
#                                  genes with has_protein=True are predicted targets.
#
#   Only samples present in BOTH RNA and protein DataFrames are used (inner join).
#   Gene symbols from CPTAC (standard HGNC) may differ from the CancerFoundation
#   vocabulary (which mixes HGNC with older Ensembl lncRNA names). The embedder's
#   gene intersection handles unknown symbols silently; a translation step is
#   provided as a placeholder for improved coverage.
# ============================================================================


class ProteomeEmbeddingDataset(Dataset):
    """Dataset for RNA → protein abundance regression."""

    def __init__(self, embeddings: np.ndarray, protein_targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        # protein_targets shape: (N, n_proteins), log2-ratio TMT values
        self.targets = np.asarray(protein_targets, dtype=np.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = torch.from_numpy(self.embeddings[index]).float()
        target = torch.from_numpy(self.targets[index]).float()
        return emb, target


@TaskRegistry.register
class ProteomePredTask(DownstreamTask):
    """
    RNA-to-protein abundance prediction using CPTAC data.

    The frozen encoder embeds bulk RNA-seq samples (using only genes flagged
    has_rna in gene_protein_mapping.csv); a 2-layer regression head predicts
    log2-ratio TMT protein abundances for genes flagged has_protein.

    Evaluation: mean and median Pearson R across proteins, plus overall RMSE.
    Proteins with near-zero variance in either predictions or targets are
    excluded from the correlation average (degenerate output / flat signal).

    Config key: finetune.proteome_pred
    Required config fields:
        rna_path              Path to rna_expression.parquet
        protein_path          Path to protein_expression.parquet
        gene_protein_map_path Path to gene_protein_mapping.csv
                              Columns: (0) gene symbol, (1) has_rna, (2) has_protein
    """

    @property
    def task_name(self) -> str:
        return "proteome_pred"

    @property
    def config_key(self) -> str:
        return "finetune.proteome_pred"

    def get_head_class(self) -> type[nn.Module]:
        # output_dim = n_proteins (set by num_classes returned from load_data)
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return ProteomeEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        # MSE directly on log2-ratio protein abundances — no activation needed
        # since the head outputs unbounded floats and targets are also unbounded.
        return nn.MSELoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)
        required = ["rna_path", "protein_path", "gene_protein_map_path"]
        missing = [k for k in required if getattr(task_cfg, k, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """
        Load CPTAC RNA-seq and proteomics data.

        Steps:
          1. Load RNA and protein parquets plus gene_protein_mapping.csv.
          2. From the mapping, derive RNA input genes (has_rna=True) and protein
             target genes (has_protein=True); intersect each with the respective
             parquet columns.
          3. Optionally filter to a subset of cancer types.
          4. Inner-join RNA and protein on sample ID.
          5. Translate RNA gene symbols for the model vocabulary (placeholder).
          6. Random train/test split, stratified by cancer_type when available.

        Returns
        -------
        (n_proteins, train_adata, test_adata, train_prot, test_prot)
            n_proteins → runner sets output_dim = n_proteins.
            protein target shape: (N, n_proteins), log2-ratio TMT.
        """
        rna_path  = Path(hydra.utils.to_absolute_path(str(task_cfg.rna_path)))
        prot_path = Path(hydra.utils.to_absolute_path(str(task_cfg.protein_path)))
        map_path  = Path(hydra.utils.to_absolute_path(str(task_cfg.gene_protein_map_path)))

        df_rna  = pd.read_parquet(rna_path)
        df_prot = pd.read_parquet(prot_path)
        df_map  = pd.read_csv(map_path)

        # gene_protein_mapping.csv columns: (0) gene symbol, (1) has_rna, (2) has_protein
        gene_col     = df_map.columns[0]
        has_rna_col  = df_map.columns[1]
        has_prot_col = df_map.columns[2]

        # RNA input genes: flagged has_rna, present in parquet
        rna_genes_mapped = df_map.loc[df_map[has_rna_col].astype(bool), gene_col].astype(str).tolist()
        rna_genes = [g for g in rna_genes_mapped if g in df_rna.columns]

        # Protein target genes: flagged has_protein, present in parquet
        prot_genes_mapped = df_map.loc[df_map[has_prot_col].astype(bool), gene_col].astype(str).tolist()
        prot_genes = [g for g in prot_genes_mapped if g in df_prot.columns]

        if not rna_genes:
            raise ValueError(
                "No RNA genes from gene_protein_mapping.csv (has_rna=True) found "
                "in rna_expression.parquet column names."
            )
        if not prot_genes:
            raise ValueError(
                "No protein genes from gene_protein_mapping.csv (has_protein=True) found "
                "in protein_expression.parquet column names."
            )

        self._protein_names = prot_genes
        log.info(
            f"CPTAC: {len(rna_genes)} RNA input genes, "
            f"{len(prot_genes)} protein target genes"
        )

        # Filter to requested cancer types (optional)
        cancer_types = list(getattr(task_cfg, "cancer_types", []))
        if cancer_types and "cancer_type" in df_rna.columns:
            df_rna  = df_rna[df_rna["cancer_type"].isin(cancer_types)]
            if "cancer_type" in df_prot.columns:
                df_prot = df_prot[df_prot["cancer_type"].isin(cancer_types)]
            log.info(f"Filtered to cancer types: {cancer_types}")

        # Inner join on sample ID — only keep samples with both RNA and protein
        shared = df_rna.index.intersection(df_prot.index)
        if len(shared) == 0:
            raise ValueError(
                "No shared sample IDs between RNA and protein DataFrames after "
                "filtering. Check that both parquets use the same sample ID index."
            )
        df_rna  = df_rna.loc[shared]
        df_prot = df_prot.loc[shared]
        log.info(f"CPTAC: {len(shared)} paired RNA–protein samples retained")

        # Translate gene symbols to the model vocab convention, then build AnnData
        rna_genes_translated = self._translate_gene_symbols(rna_genes)
        adata = parquet_to_adata(df_rna, rna_genes)
        adata.var_names = rna_genes_translated

        targets = df_prot[prot_genes].to_numpy(dtype=np.float32)

        # Stratify by cancer_type if present so every type appears in both splits
        strata = None
        if "cancer_type" in df_rna.columns:
            strata = df_rna["cancer_type"].to_numpy()
            adata.obs["cancer_type"] = strata

        test_size  = float(getattr(task_cfg, "test_size", 0.2))
        split_seed = self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1))

        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            stratify=strata,
            random_state=split_seed,
        )

        n_proteins = len(prot_genes)
        log.info(
            f"CPTAC split: train={len(train_idx)}, test={len(test_idx)}, "
            f"n_proteins={n_proteins}"
        )

        return (
            n_proteins, # output_dim = n_proteins (one log2-ratio TMT value per protein)
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Generate embeddings and create ProteomeEmbeddingDataset instances."""
        train_emb = self._embed_adata(embedder, train_adata)
        test_emb  = self._embed_adata(embedder, test_adata)
        embedding_dim = train_emb.shape[1]

        train_dataset = ProteomeEmbeddingDataset(train_emb, train_targets)
        test_dataset  = ProteomeEmbeddingDataset(test_emb,  test_targets)

        log.info(
            f"Proteome prediction datasets ready. "
            f"embedding_dim={embedding_dim}, n_proteins={train_targets.shape[1]}"
        )
        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData, batch_size: int = 64) -> np.ndarray:
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, flavor="seurat_v3")
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute per-protein Pearson correlation and overall RMSE.

        Parameters
        ----------
        predictions : np.ndarray, shape (N, n_proteins)
            Raw head outputs (direct regression, no activation).
        targets : np.ndarray, shape (N, n_proteins)
            log2-ratio TMT protein abundances.

        Notes
        -----
        Proteins where either predictions or targets have near-zero variance
        are skipped — the correlation is undefined or meaningless in those cases.
        This can happen for proteins with universally constant abundance or when
        the model has collapsed to a constant prediction for a particular output.
        """
        from scipy.stats import pearsonr

        per_protein_r: list[float] = []
        for i in range(predictions.shape[1]):
            pred_i, tgt_i = predictions[:, i], targets[:, i]
            if np.std(tgt_i) < 1e-8 or np.std(pred_i) < 1e-8:
                continue
            r, _ = pearsonr(tgt_i, pred_i)
            if not np.isnan(r):
                per_protein_r.append(float(r))

        n_eval = len(per_protein_r)
        mean_r = float(np.mean(per_protein_r))   if n_eval > 0 else 0.0
        med_r  = float(np.median(per_protein_r)) if n_eval > 0 else 0.0
        rmse   = float(np.sqrt(np.mean((predictions - targets) ** 2)))

        log.info(
            f"Proteome metrics: mean_r={mean_r:.4f}, median_r={med_r:.4f}, "
            f"rmse={rmse:.4f}, evaluated {n_eval}/{predictions.shape[1]} proteins"
        )

        return {
            "mean_pearson_r":        mean_r,
            "median_pearson_r":      med_r,
            "rmse":                  rmse,
            "n_proteins_evaluated":  float(n_eval),
        }


# ============================================================================
# Drug Sensitivity Prediction Task (BeatAML)
# ============================================================================
#
# Task: Given a bulk RNA-seq profile from an AML patient specimen, predict the
# AUC drug response value (range 0–1) across a panel of drugs. This directly
# tests whether the model captures therapeutically relevant transcriptional
# programmes — high-AUC = sensitive, low-AUC = resistant.
#
# Data: BeatAML2 (Bottomly et al., Cancer Cell 2022)
#   - expression.parquet      (specimens × genes, log1p-normalised)
#   - drug_response.parquet   (specimens × drugs, AUC ∈ [0,1]; NaN = not tested)
#   - metadata.csv            (specimen_id, patient_id, diagnosis, clinical)
#
# Key challenges:
#   1. SPARSE targets — not every drug is assayed on every specimen. NaN entries
#      must not contribute to the loss or metrics. MaskedMSELoss handles this.
#   2. PATIENT-LEVEL split — one patient may contribute multiple specimens.
#      Splitting by specimen would leak information across train/test. We split
#      by patient and assign all specimens of a patient to one partition.
# ============================================================================


class DrugSensitivityEmbeddingDataset(Dataset):
    """
    Dataset for drug AUC prediction.

    NaN values (drug not tested on specimen) are preserved in the target tensor.
    MaskedMSELoss uses torch.isnan() to exclude them from the gradient during
    training. compute_metrics applies the same masking at evaluation time.
    """

    def __init__(self, embeddings: np.ndarray, drug_targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        # Preserve NaN — do NOT replace with 0 here; the loss needs the mask.
        self.targets = drug_targets.astype(np.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = torch.from_numpy(self.embeddings[index]).float()
        # torch.tensor() correctly propagates NaN in float32
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return emb, target
    

class MaskedMSELoss(nn.Module):
    """
    MSE loss that skips positions where the target is NaN.

    BeatAML drug response data is a sparse matrix: not every drug is assayed
    on every specimen. NaN encodes "not tested". This loss zeros out those
    positions so gradients only flow from positions with a real measurement.

    Sigmoid is applied to the predictions before computing MSE so that the
    model learns to output raw logits in (−∞, +∞) while the loss is computed
    in the AUC space [0, 1] matching the target scale.

    Input
    -----
    pred    : Tensor (N, n_drugs) — raw logits (no activation yet)
    targets : Tensor (N, n_drugs) — AUC values ∈ [0, 1], NaN = not tested
    """

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_auc = torch.sigmoid(pred)         # map logits → [0, 1]
        valid    = ~torch.isnan(targets)       # True where drug was tested

        if valid.sum() == 0:
            # Degenerate batch (all NaN) — return zero to avoid NaN gradients.
            return pred.sum() * 0.0

        return F.mse_loss(pred_auc[valid], targets[valid])


@TaskRegistry.register
class DrugSensitivityTask(DownstreamTask):
    """
    Drug sensitivity (AUC) prediction on the BeatAML dataset.

    The frozen encoder embeds bulk AML RNA-seq samples; a 2-layer regression
    head predicts AUC values for all drugs that pass a minimum-sample threshold.
    NaN entries (drug not tested) are excluded from both training loss and
    evaluation metrics.

    Evaluation: per-drug Pearson R (averaged over drugs with ≥ min_drug_samples
    test specimens) and RMSE on all non-NaN test entries.

    Config key: finetune.drug_sensitivity
    Required config fields:
        expression_path     Path to expression.parquet
        drug_response_path  Path to drug_response.parquet
        metadata_path       Path to metadata.csv
    """

    @property
    def task_name(self) -> str:
        return "drug_sensitivity"

    @property
    def config_key(self) -> str:
        return "finetune.drug_sensitivity"

    def get_head_class(self) -> type[nn.Module]:
        # output_dim = n_drugs (set by num_classes returned from load_data)
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return DrugSensitivityEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        # MaskedMSELoss applies sigmoid to predictions before computing MSE,
        # masking NaN positions in targets.
        return MaskedMSELoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)
        required = ["expression_path", "drug_response_path", "metadata_path"]
        missing = [k for k in required if getattr(task_cfg, k, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """
        Load BeatAML expression and drug response data.

        Steps:
          1. Load expression, drug_response, and metadata DataFrames.
          2. Inner-join expression and drug response on specimen ID.
          3. Filter drugs: keep only drugs tested on ≥ min_drug_samples specimens.
          4. Patient-level split: load patient_id from metadata and ensure all
             specimens from the same patient land in the same partition.
          5. Return AnnData (expression) + sparse drug target matrix (with NaN).

        Returns
        -------
        (n_drugs, train_adata, test_adata, train_drug, test_drug)
            n_drugs → runner sets output_dim = n_drugs.
            drug targets shape: (N, n_drugs), float32 with NaN.
        """
        expr_path = Path(hydra.utils.to_absolute_path(str(task_cfg.expression_path)))
        drug_path = Path(hydra.utils.to_absolute_path(str(task_cfg.drug_response_path)))
        meta_path = Path(hydra.utils.to_absolute_path(str(task_cfg.metadata_path)))

        df_expr = pd.read_parquet(expr_path)
        df_drug = pd.read_parquet(drug_path)
        df_meta = pd.read_csv(meta_path, index_col=0)

        # Inner join: keep specimens with both expression and drug response data
        shared = df_expr.index.intersection(df_drug.index)
        if len(shared) == 0:
            raise ValueError(
                "No shared specimen IDs between expression and drug_response data. "
                "Check that both parquets use the same specimen ID as the row index."
            )
        df_expr = df_expr.loc[shared]
        df_drug = df_drug.loc[shared]
        log.info(f"BeatAML: {len(shared)} specimens with paired expression + drug data")

        # Filter drugs by minimum number of tested specimens.
        # Drugs tested on very few specimens produce noisy gradients and
        # unreliable correlations — we discard them early.
        min_samples = int(getattr(task_cfg, "min_drug_samples", 20))
        n_tested    = df_drug.notna().sum(axis=0)
        kept_drugs  = n_tested[n_tested >= min_samples].index.tolist()
        if not kept_drugs:
            raise ValueError(
                f"No drugs remain after filtering with min_drug_samples={min_samples}. "
                f"Reduce the threshold or check the drug_response data."
            )
        df_drug = df_drug[kept_drugs]
        log.info(
            f"BeatAML: {len(kept_drugs)}/{len(n_tested)} drugs kept "
            f"(≥{min_samples} tested specimens)"
        )

        # Store for metric computation (compute_metrics has no access to task_cfg)
        self._drug_names       = kept_drugs
        self._min_drug_samples = min_samples

        # Patient-level split: prevents leakage from multi-specimen patients.
        # metadata.csv is expected to contain a 'patient_id' column; if missing,
        # fall back to specimen-level split with a warning.
        df_meta_aligned = df_meta.reindex(shared)
        if "patient_id" in df_meta_aligned.columns:
            patient_ids = (
                df_meta_aligned["patient_id"]
                .fillna(df_meta_aligned.index.to_series())
                .to_numpy()
            )
            train_idx, test_idx = self._patient_level_split(
                patient_ids,
                float(getattr(task_cfg, "test_size", 0.2)),
                self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1)),
            )
        else:
            log.warning(
                "metadata.csv has no 'patient_id' column — falling back to specimen-level "
                "random split. This may cause data leakage for multi-specimen patients."
            )
            test_size  = float(getattr(task_cfg, "test_size", 0.2))
            split_seed = self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1))
            train_idx, test_idx = train_test_split(
                np.arange(len(shared)), test_size=test_size, random_state=split_seed
            )

        adata   = parquet_to_adata(df_expr, df_expr.columns.tolist())
        targets = df_drug.to_numpy(dtype=np.float32)   # NaN preserved

        n_drugs = len(kept_drugs)
        avg_cov = float(df_drug.notna().mean().mean()) * 100
        log.info(
            f"BeatAML split: train={len(train_idx)}, test={len(test_idx)}, "
            f"n_drugs={n_drugs}, avg_coverage={avg_cov:.1f}%"
        )

        return (
            n_drugs,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    @staticmethod
    def _patient_level_split(
        patient_ids: np.ndarray,
        test_size: float,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split specimens by patient so that every specimen from a given patient
        is in the same partition. Avoids correlation leakage across specimens.

        Patients are shuffled deterministically with random_seed, then the last
        (test_size × 100)% of patients form the test set.
        """
        unique_patients = pd.unique(patient_ids)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(unique_patients)

        n_test   = max(1, int(len(unique_patients) * test_size))
        test_set = set(unique_patients[-n_test:])

        mask      = pd.Series(patient_ids).isin(test_set).to_numpy()
        train_idx = np.where(~mask)[0]
        test_idx  = np.where( mask)[0]
        return train_idx, test_idx

    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Generate embeddings and create DrugSensitivityEmbeddingDataset instances."""
        train_emb = self._embed_adata(embedder, train_adata)
        test_emb  = self._embed_adata(embedder, test_adata)
        embedding_dim = train_emb.shape[1]

        train_dataset = DrugSensitivityEmbeddingDataset(train_emb, train_targets)
        test_dataset  = DrugSensitivityEmbeddingDataset(test_emb,  test_targets)

        avg_cov = float(np.mean(~np.isnan(train_targets))) * 100
        log.info(
            f"Drug sensitivity datasets ready. "
            f"embedding_dim={embedding_dim}, n_drugs={train_targets.shape[1]}, "
            f"train_coverage={avg_cov:.1f}%"
        )
        return train_dataset, test_dataset, embedding_dim

    def _embed_adata(self, embedder: Any, adata: ad.AnnData, batch_size: int = 64) -> np.ndarray:
        embedder.eval()
        embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, flavor="seurat_v3")
        return df_emb.to_numpy()

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute per-drug Pearson R and overall RMSE on non-NaN entries.

        Parameters
        ----------
        predictions : np.ndarray, shape (N, n_drugs)
            Raw head logits (before sigmoid).
        targets : np.ndarray, shape (N, n_drugs)
            AUC values ∈ [0, 1]; NaN = not tested.

        Notes
        -----
        Sigmoid is applied to predictions before comparison — this mirrors the
        MaskedMSELoss that also applies sigmoid during training, keeping the
        prediction space consistent between loss and metrics.

        Per-drug correlation is only computed for drugs with ≥ min_drug_samples
        non-NaN test specimens. Drugs below this threshold are excluded from the
        mean/median (too few points for a reliable correlation estimate).
        """
        from scipy.stats import pearsonr

        # Bring predictions to AUC scale [0, 1] — must match MaskedMSELoss
        pred_auc = torch.sigmoid(torch.from_numpy(predictions)).numpy()

        min_samples = getattr(self, "_min_drug_samples", 20)
        per_drug_r: list[float] = []

        for d in range(targets.shape[1]):
            valid = ~np.isnan(targets[:, d])
            if valid.sum() < min_samples:
                continue
            r, _ = pearsonr(targets[valid, d], pred_auc[valid, d])
            if not np.isnan(r):
                per_drug_r.append(float(r))

        n_eval = len(per_drug_r)
        mean_r = float(np.mean(per_drug_r))   if n_eval > 0 else 0.0
        med_r  = float(np.median(per_drug_r)) if n_eval > 0 else 0.0

        # RMSE across all non-NaN test entries
        valid_all = ~np.isnan(targets)
        rmse = float(
            np.sqrt(np.mean((pred_auc[valid_all] - targets[valid_all]) ** 2))
        )

        log.info(
            f"Drug sensitivity: mean_r={mean_r:.4f}, median_r={med_r:.4f}, "
            f"rmse={rmse:.4f}, drugs_evaluated={n_eval}/{targets.shape[1]}"
        )

        return {
            "mean_pearson_r":     mean_r,
            "median_pearson_r":   med_r,
            "rmse":               rmse,
            "n_drugs_evaluated":  float(n_eval),
        }
