from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry
from evaluate.finetune.tasks.components import EmbeddingPredHead
from evaluate.finetune.utils import (
    deduplicate_var_names,
    strip_ensembl_versions,
    translate_gene_symbols,
)

log = logging.getLogger(__name__)


METADATA_ROWS = {
    "gene_symbol",
    "ensembl_gene_id",
    "nsembl_gene_id",
    "gene_id",
    "model_name",
}
ENSEMBL_ROWS = ("ensembl_gene_id", "nsembl_gene_id", "gene_id")


class DrugSensitivityV2EmbeddingDataset(Dataset):
    """Dataset for single-drug binary sensitivity prediction."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[index]).float(),
            torch.from_numpy(self.labels[index]).float(),
        )


def compute_prism_response(
    ic50_log2: np.ndarray | pd.Series | float,
    slope: np.ndarray | pd.Series | float,
    dose: float,
) -> np.ndarray:
    """Compute PRISM-style response from log2 IC50, Hill slope, and linear dose."""
    if dose <= 0:
        raise ValueError(f"dose must be positive, got {dose}.")
    ic50_arr = np.asarray(ic50_log2, dtype=np.float64)
    slope_arr = np.asarray(slope, dtype=np.float64)
    return 1.0 / (
        1.0 + np.power(2.0, -(slope_arr * (np.log2(float(dose)) - ic50_arr)))
    )


def binarize_response(
    response: np.ndarray | pd.Series,
    response_threshold: float = 0.5,
) -> np.ndarray:
    """Convert continuous response values to binary sensitivity labels."""
    return (np.asarray(response, dtype=np.float64) >= float(response_threshold)).astype(np.float32)


def cp10k_log1p_normalize(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each sample to CP10K and apply log1p."""
    counts = expr_df.clip(lower=0.0)
    library_sizes = counts.sum(axis=1)
    zero_library_mask = library_sizes <= 0
    if zero_library_mask.any():
        log.warning(
            "Expression CSV has %d rows with non-positive library size; leaving them as zeros.",
            int(zero_library_mask.sum()),
        )

    normalized = counts.div(library_sizes.where(~zero_library_mask, np.nan), axis=0) * 1e4
    normalized = normalized.fillna(0.0)
    return np.log1p(normalized).astype(np.float32)


def _resolve_dose(task_cfg: DictConfig, drug: str) -> float:
    doses = getattr(task_cfg, "drug_doses", None)
    if doses is None:
        raise ValueError("finetune.drug_sensitivity_v2.drug_doses must be configured.")

    doses_container = (
        OmegaConf.to_container(doses, resolve=True)
        if OmegaConf.is_config(doses)
        else doses
    )
    if drug not in doses_container:
        available = ", ".join(map(str, doses_container.keys()))
        raise ValueError(f"No dose configured for drug '{drug}'. Available drug_doses: {available}")

    return float(doses_container[drug])


def load_expression_csv(x_path: str | Path) -> ad.AnnData:
    """
    Load the PRISM-style transposed expression CSV.

    The first column contains row labels. Metadata rows such as gene_symbol,
    ensembl_gene_id, nsembl_gene_id, gene_id, and model_name are removed before
    training. Remaining rows are cell lines indexed by model_name.
    """
    df = pd.read_csv(x_path)
    if df.empty or df.shape[1] < 2:
        raise ValueError(
            f"Expression CSV must contain a row-label column and gene columns: {x_path}"
        )

    row_label_col = df.columns[0]
    df[row_label_col] = df[row_label_col].astype(str)
    df = df.set_index(row_label_col)
    normalized_index = pd.Index(df.index.astype(str).str.strip().str.lower())

    metadata_mask = normalized_index.isin(METADATA_ROWS)
    if metadata_mask.all():
        raise ValueError("Expression CSV contains only metadata rows and no cell-line rows.")

    var_names: list[str]
    ensembl_row = next((row for row in ENSEMBL_ROWS if row in normalized_index), None)
    if ensembl_row is not None:
        raw_var_names = df.iloc[normalized_index.get_loc(ensembl_row)].astype(str).tolist()
        var_names = strip_ensembl_versions(raw_var_names)
    elif "gene_symbol" in normalized_index:
        raw_var_names = df.iloc[normalized_index.get_loc("gene_symbol")].astype(str).tolist()
        var_names = translate_gene_symbols(raw_var_names, mapping_file="symbol_to_ensembl.json")
    else:
        raw_var_names = [str(col) for col in df.columns]
        var_names = translate_gene_symbols(raw_var_names, mapping_file="symbol_to_ensembl.json")

    expr_df = df.loc[~metadata_mask].copy()
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    if expr_df.isna().any().any():
        n_nan = int(expr_df.isna().sum().sum())
        log.warning("Expression CSV has %d non-finite values after parsing; filling with 0.", n_nan)
        expr_df = expr_df.fillna(0.0)

    expr_df = cp10k_log1p_normalize(expr_df)

    if len(var_names) != expr_df.shape[1]:
        raise ValueError(
            f"Expression CSV has {expr_df.shape[1]} gene columns but {len(var_names)} parsed gene names."
        )

    adata = ad.AnnData(X=expr_df.to_numpy(dtype=np.float32))
    adata.obs_names = expr_df.index.astype(str).tolist()
    adata.obs["model_name"] = adata.obs_names
    adata.var_names = pd.Index([str(v) for v in var_names])
    adata = deduplicate_var_names(adata)
    adata.obs_names_make_unique()
    return adata


def load_drug_response_labels(
    y_path: str | Path,
    drug: str,
    dose: float,
    response_threshold: float = 0.5,
) -> pd.DataFrame:
    """Load model_name-indexed binary labels for one configured drug."""
    df = pd.read_csv(y_path)
    if "model_name" not in df.columns:
        raise ValueError("Drug response CSV must contain a 'model_name' column.")

    ic50_col = f"{drug}_IC50" if f"{drug}_IC50" in df.columns else drug
    slope_col = f"{drug}_slope"
    missing = [col for col in (ic50_col, slope_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Drug response CSV is missing required columns for '{drug}': {missing}")

    labels = df[["model_name", ic50_col, slope_col]].copy()
    labels[ic50_col] = pd.to_numeric(labels[ic50_col], errors="coerce")
    labels[slope_col] = pd.to_numeric(labels[slope_col], errors="coerce")
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=[ic50_col, slope_col])
    if labels.empty:
        raise ValueError(f"No finite IC50/slope rows remain for drug '{drug}'.")

    response = compute_prism_response(labels[ic50_col], labels[slope_col], dose)
    labels["response"] = response
    labels["label"] = binarize_response(response, response_threshold)
    return labels[["model_name", "response", "label"]].set_index("model_name")


@TaskRegistry.register
class DrugSensitivityV2Task(DownstreamTask):
    """Single-drug binary classification using PRISM IC50+slope response labels."""

    @property
    def task_name(self) -> str:
        return "drug_sensitivity_v2"

    @property
    def config_key(self) -> str:
        return "finetune.drug_sensitivity_v2"

    def get_head_class(self) -> type[nn.Module]:
        return EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return DrugSensitivityV2EmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        return nn.BCEWithLogitsLoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)
        required = ["x_path", "y_path", "drug", "drug_doses"]
        missing = [key for key in required if getattr(task_cfg, key, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )
        _resolve_dose(task_cfg, str(task_cfg.drug))

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        x_path = Path(hydra.utils.to_absolute_path(str(task_cfg.x_path)))
        y_path = Path(hydra.utils.to_absolute_path(str(task_cfg.y_path)))
        drug = str(task_cfg.drug)
        dose = _resolve_dose(task_cfg, drug)
        response_threshold = float(getattr(task_cfg, "response_threshold", 0.5))

        adata = load_expression_csv(x_path)
        labels = load_drug_response_labels(y_path, drug, dose, response_threshold)

        shared = adata.obs_names.intersection(pd.Index(labels.index.astype(str)))
        if len(shared) == 0:
            raise ValueError(
                "No shared model_name values between expression and drug response CSVs."
            )

        adata = adata[shared].copy()
        label_df = labels.loc[shared]
        targets = label_df["label"].to_numpy(dtype=np.float32)

        test_size = float(getattr(task_cfg, "test_size", 0.2))
        split_seed = self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1))
        stratify = (
            targets
            if np.unique(targets).size == 2
            and min(np.bincount(targets.astype(int))) >= 2
            else None
        )
        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            random_state=split_seed,
            stratify=stratify,
        )

        self._drug = drug
        self._dose = dose
        self._response_threshold = response_threshold
        self._prediction_threshold = float(getattr(task_cfg, "prediction_threshold", 0.5))
        self._n_train = int(len(train_idx))
        self._n_test = int(len(test_idx))
        self._positive_rate_train = float(targets[train_idx].mean()) if len(train_idx) else 0.0
        self._positive_rate_test = float(targets[test_idx].mean()) if len(test_idx) else 0.0

        log.info(
            "Drug sensitivity v2: drug=%s dose=%s paired=%d train=%d test=%d positives train/test=%.3f/%.3f",
            drug,
            dose,
            adata.n_obs,
            len(train_idx),
            len(test_idx),
            self._positive_rate_train,
            self._positive_rate_test,
        )

        return (
            1,
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
        task_cfg: Any,
    ) -> tuple[Dataset, Dataset, int]:
        train_emb = self._embed_adata(embedder, train_adata, task_cfg)
        test_emb = self._embed_adata(embedder, test_adata, task_cfg)
        if train_emb.ndim != 2 or test_emb.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_samples, embedding_dim].")
        if train_emb.shape[1] != test_emb.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        return (
            DrugSensitivityV2EmbeddingDataset(train_emb, train_targets),
            DrugSensitivityV2EmbeddingDataset(test_emb, test_targets),
            int(train_emb.shape[1]),
        )

    def _embed_adata(self, embedder: Any, adata: ad.AnnData, task_cfg: Any) -> np.ndarray:
        batch_size = int(getattr(task_cfg, "embed_batch_size", 64))
        normalized = bool(getattr(task_cfg, "normalized", True))
        embedder.eval()
        if torch.cuda.is_available() and hasattr(embedder, "cuda"):
            embedder.cuda()
        df_emb = embedder.embed(adata, batch_size=batch_size, normalized=normalized)
        return df_emb.to_numpy(dtype=np.float32)

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        logits = np.asarray(predictions, dtype=np.float64).reshape(-1)
        y_true = np.asarray(targets, dtype=np.float64).reshape(-1).astype(int)
        y_prob = 1.0 / (1.0 + np.exp(-logits))
        threshold = float(getattr(self, "_prediction_threshold", 0.5))
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "n_train": float(getattr(self, "_n_train", 0)),
            "n_test": float(getattr(self, "_n_test", len(y_true))),
            "positive_rate_train": float(getattr(self, "_positive_rate_train", 0.0)),
            "positive_rate_test": float(
                getattr(self, "_positive_rate_test", y_true.mean() if len(y_true) else 0.0)
            ),
            "drug": str(getattr(self, "_drug", "")),
            "dose": float(getattr(self, "_dose", np.nan)),
            "response_threshold": float(getattr(self, "_response_threshold", 0.5)),
            "prediction_threshold": threshold,
        }

        if np.unique(y_true).size == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))

        return metrics
