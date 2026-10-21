from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry
from evaluate.finetune.normalization import resolve_policy
from evaluate.finetune.tasks.components import EmbeddingPredHead
from evaluate.finetune.utils import deduplicate_var_names, strip_ensembl_versions

log = logging.getLogger(__name__)

# ============================================================================
# Drug Sensitivity task 2 (implemented by Michael)
# ============================================================================
#
# Purpose:
#   The task fits 40 monotherapy (single-drug) models for both
#   IC50 concentration regression and Cmax viability classification
#   (sensitive/resistant at Cmax) to evaluate the foundation models.
#
# Data sources:
#   - Gene expression data from Cell Model Passports (CMP). CMP aggregates
#     1300+ cell-line expression profiles and metadata from two sequencing databases
#     (Sanger and Broad Institute). Genes are mapped to Ensembl IDs / the model
#     vocabulary and CP10K log1p normalization are applied to make inputs
#     comparable to the embedder's expected distribution.
#
#   - Drug response labels: CREAMMIST estimates (Yingtaweesittikul et al., 2022)
#     which harmonize dose–response measurements across multiple pharmacogenomic databases.
#     Those fitted parameters are more robust than typical raw IC50s because they
#     account for cross-study heterogeneity via probabilistic modelling.
#
#   - The ~40 selected drugs aim to span diverse mechanisms of action
#     while minimising missingness across cell models so training has
#     sufficient positive/negative examples per drug.
#
# ============================================================================

DEFAULT_X_PATH = "/cluster/work/boeva/bulkFM/data/processed/drug_sens_prediction/gene_expression.csv"
DEFAULT_Y_PATH = "/cluster/work/boeva/bulkFM/data/processed/drug_sens_prediction/drug_response.csv"

ENDPOINTS = ("cmax_classification", "ic50_regression")

METADATA_ROWS = {
    "gene_symbol",
    "ensembl_gene_id",
    "nsembl_gene_id",
    "gene_id",
    "model_name",
}
ENSEMBL_ROWS = ("ensembl_gene_id", "gene_id")


class PrecomputedEmbedder:
    """Drop-in embedder backed by a pre-computed embedding matrix.

    Satisfies the ``embed(adata, ...)`` interface used by ``_embed_adata`` so
    that the runner and task code require no changes per job.  The embeddings
    are looked up by ``adata.obs_names`` instead of running a transformer
    forward pass.
    """

    def __init__(self, embeddings: pd.DataFrame) -> None:
        # index = obs_name (cell-line ID), columns = embedding dimensions
        self._embeddings = embeddings
        self.embsize = embeddings.shape[1]

    def eval(self) -> "PrecomputedEmbedder":
        return self

    def cuda(self) -> "PrecomputedEmbedder":
        return self

    def embed(self, adata: ad.AnnData, **kwargs) -> pd.DataFrame:
        return self._embeddings.loc[adata.obs_names]


def precompute_embeddings(embedder: Any, task_cfg: DictConfig) -> pd.DataFrame:
    """Embed every cell line in the expression CSV once.

    Returns a DataFrame indexed by obs_name with one column per embedding
    dimension.  Pass the result to ``PrecomputedEmbedder`` so downstream
    jobs can look up pre-computed vectors instead of re-running the model.
    """
    x_path = hydra.utils.to_absolute_path(str(getattr(task_cfg, "x_path", DEFAULT_X_PATH)))
    policy = resolve_policy(task_cfg, None)
    full_adata = load_expression_csv(x_path, policy)
    batch_size = int(getattr(task_cfg, "embed_batch_size", 64))

    embedder.eval()
    if torch.cuda.is_available() and hasattr(embedder, "cuda"):
        embedder.cuda()

    kwargs = dict(batch_size=batch_size, **policy.embed_kwargs())
    if not getattr(embedder, "fittable", False):
        kwargs["modality"] = "bulk"  # cell-line bulk → log1p + MAD gene selection
    result = embedder.embed(full_adata, **kwargs)
    df = result[0] if isinstance(result, tuple) else result
    return df


class DrugSensitivityV2EmbeddingDataset(Dataset):
    """Embedding dataset for one drug and one endpoint."""

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[index]).float(),
            torch.from_numpy(self.targets[index]).float(),
        )


def binarize_response(response: np.ndarray | pd.Series, threshold: float = 0.5) -> np.ndarray:
    """Convert response = 1 - Cmax viability into sensitive/resistant labels."""
    return (np.asarray(response, dtype=np.float64) >= float(threshold)).astype(np.float32)


def cp10k_log1p_normalize(expr_df: pd.DataFrame) -> pd.DataFrame:
    counts = expr_df.clip(lower=0.0)
    library_sizes = counts.sum(axis=1)
    zero_library_mask = library_sizes <= 0

    if zero_library_mask.any():
        log.warning(
            "Expression CSV has %d rows with non-positive library size.",
            int(zero_library_mask.sum()),
        )

    normalized = counts.div(library_sizes.where(~zero_library_mask, np.nan), axis=0) * 1e4
    return np.log1p(normalized.fillna(0.0)).astype(np.float32)


def load_expression_csv(x_path: str | Path, policy: Any) -> ad.AnnData:
    """Load the cell-line expression CSV and normalize it per *policy*.

    ``policy`` is required rather than defaulted: this loader used to CP10K+log1p
    unconditionally, so a caller that forgot to pass one would silently reintroduce
    exactly the hidden normalization this switch exists to remove.
    """
    df = pd.read_csv(x_path, low_memory=False)
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Expression CSV must contain a row-label column and gene columns: {x_path}")

    row_label_col = df.columns[0]
    df[row_label_col] = df[row_label_col].astype(str)
    df = df.set_index(row_label_col)
    normalized_index = pd.Index(df.index.astype(str).str.strip().str.lower())

    metadata_mask = normalized_index.isin(METADATA_ROWS)
    ensembl_row = next((row for row in ENSEMBL_ROWS if row in normalized_index), None)

    var_names = strip_ensembl_versions(df.iloc[normalized_index.get_loc(ensembl_row)].astype(str).tolist())

    expr_df = df.loc[~metadata_mask].apply(pd.to_numeric, errors="coerce")
    if expr_df.isna().any().any():
        log.warning(
            "Expression CSV has %d non-finite values after parsing; filling with 0.",
            int(expr_df.isna().sum().sum()),
        )
        expr_df = expr_df.fillna(0.0)

    if len(var_names) != expr_df.shape[1]:
        raise ValueError(
            f"Expression CSV has {expr_df.shape[1]} gene columns but "
            f"{len(var_names)} parsed gene names."
        )

    adata = ad.AnnData(X=expr_df.to_numpy(dtype=np.float32))
    adata.obs_names = expr_df.index.astype(str).tolist()
    adata.obs["model_name"] = adata.obs_names
    adata.var_names = pd.Index([str(v) for v in var_names])
    # Before deduplicate_var_names, matching the order the unconditional
    # cp10k_log1p_normalize() used to run in: dropping duplicate genes changes each
    # row's library size, so normalizing after it would not reproduce past numbers.
    adata, _ = policy.apply(adata, task="drug_sensitivity_v2")
    adata = deduplicate_var_names(adata)
    adata.obs_names_make_unique()
    return adata

def discover_drugs(response_df: pd.DataFrame) -> list[str]:
    """Discover drug names from '<drug>_Cmax_viability' columns."""
    drugs = sorted(
        col.removesuffix("_Cmax_viability")
        for col in response_df.columns
        if col.endswith("_Cmax_viability")
    )
    return drugs


def find_ic50_col(response_df: pd.DataFrame, drug: str) -> str:
    """Find the IC50 column for a drug without hard-coding one exact suffix."""
    candidates = [
        col
        for col in response_df.columns
        if col.startswith(f"{drug}_") and "ic50" in col.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one IC50 column for '{drug}', found {candidates}.")
    return candidates[0]


def make_drug_endpoint_configs(task_cfg: DictConfig) -> list[dict[str, str]]:
    """Build independent jobs: drugs x endpoints."""
    y_path = getattr(task_cfg, "y_path", DEFAULT_Y_PATH)
    response_df = pd.read_csv(y_path)

    jobs = []
    for drug in discover_drugs(response_df):
        find_ic50_col(response_df, drug)
        jobs.extend({"drug": drug, "endpoint": endpoint} for endpoint in ENDPOINTS)

    return jobs


def load_drug_endpoint_targets(
    response_df: pd.DataFrame,
    drug: str,
    endpoint: str,
    response_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return model_name-indexed targets for one drug and one endpoint."""
    if endpoint == "cmax_classification":
        cmax_col = f"{drug}_Cmax_viability"
        labels = response_df[["model_name", cmax_col]].copy()
        labels[cmax_col] = pd.to_numeric(labels[cmax_col], errors="coerce")
        labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=[cmax_col])

        labels["response"] = 1.0 - labels[cmax_col]
        labels["target"] = binarize_response(labels["response"], response_threshold)
        return labels[["model_name", "target", "response"]].set_index("model_name")

    if endpoint == "ic50_regression":
        ic50_col = find_ic50_col(response_df, drug)
        labels = response_df[["model_name", ic50_col]].copy()
        labels[ic50_col] = pd.to_numeric(labels[ic50_col], errors="coerce")
        labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=[ic50_col])

        labels["target"] = labels[ic50_col].astype(np.float32)
        return labels[["model_name", "target"]].set_index("model_name")

    raise ValueError(f"Unknown endpoint '{endpoint}'. Expected one of {ENDPOINTS}.")


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Return Pearson/Spearman correlation, or NaN if undefined."""
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method=method))


@TaskRegistry.register
class DrugSensitivityV2Task(DownstreamTask):
    """One independent model for one drug and one endpoint."""

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
        if getattr(self, "_endpoint", "cmax_classification") == "ic50_regression":
            return nn.MSELoss().to(device)
        return nn.BCEWithLogitsLoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)

        endpoint = getattr(task_cfg, "endpoint", None)
        if endpoint not in (None, "") and str(endpoint) not in ENDPOINTS:
            raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {endpoint}.")

    def load_data(
        self,
        task_cfg: DictConfig,
        embedder: Any,
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        x_path = hydra.utils.to_absolute_path(str(getattr(task_cfg, "x_path", DEFAULT_X_PATH)))
        y_path = hydra.utils.to_absolute_path(str(getattr(task_cfg, "y_path", DEFAULT_Y_PATH)))

        missing = [
            key
            for key in ("drug", "endpoint")
            if getattr(task_cfg, key, None) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"Missing internal job keys for {self.task_name}: {missing}. "
                "Do not set endpoint manually in the base config; generate all drug/endpoint jobs "
                "with make_drug_endpoint_configs(), which always includes both endpoints."
            )

        drug = str(task_cfg.drug)
        endpoint = str(task_cfg.endpoint)
        response_threshold = float(getattr(task_cfg, "response_threshold", 0.5))

        adata = load_expression_csv(x_path, resolve_policy(task_cfg, None))
        response_df = pd.read_csv(y_path)
        labels = load_drug_endpoint_targets(response_df, drug, endpoint, response_threshold)

        shared = adata.obs_names.intersection(pd.Index(labels.index.astype(str)))
        if len(shared) == 0:
            raise ValueError("No shared model_name values between expression and drug response CSVs.")

        adata = adata[shared].copy()
        targets = labels.loc[shared, "target"].to_numpy(dtype=np.float32)

        test_size = float(getattr(task_cfg, "test_size", 0.2))
        split_seed = self.hash_split_version(getattr(task_cfg, "train_test_split_version", 1))
        stratify = self._get_stratification_targets(targets, endpoint)

        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            random_state=split_seed,
            stratify=stratify,
        )

        self._drug = drug
        self._endpoint = endpoint
        self._response_threshold = response_threshold
        self._prediction_threshold = float(getattr(task_cfg, "prediction_threshold", 0.5))
        self._n_train = int(len(train_idx))
        self._n_test = int(len(test_idx))
        self._target_mean_train = float(targets[train_idx].mean()) if len(train_idx) else 0.0
        self._target_mean_test = float(targets[test_idx].mean()) if len(test_idx) else 0.0

        if endpoint == "cmax_classification":
            self._positive_rate_train = self._target_mean_train
            self._positive_rate_test = self._target_mean_test
            log.info(
                "Drug sensitivity v2: drug=%s endpoint=%s paired=%d train=%d test=%d "
                "positives train/test=%.3f/%.3f",
                drug,
                endpoint,
                adata.n_obs,
                len(train_idx),
                len(test_idx),
                self._positive_rate_train,
                self._positive_rate_test,
            )
        else:
            log.info(
                "Drug sensitivity v2: drug=%s endpoint=%s paired=%d train=%d test=%d "
                "target mean train/test=%.3f/%.3f",
                drug,
                endpoint,
                adata.n_obs,
                len(train_idx),
                len(test_idx),
                self._target_mean_train,
                self._target_mean_test,
            )

        return (
            1,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    @staticmethod
    def _get_stratification_targets(targets: np.ndarray, endpoint: str) -> np.ndarray | None:
        """Stratify classification labels directly; bin regression targets into quantiles."""
        if endpoint == "cmax_classification":
            y = targets.astype(int)
            return y if np.unique(y).size == 2 and min(np.bincount(y)) >= 2 else None

        try:
            y = pd.qcut(targets, q=5, labels=False, duplicates="drop")
        except ValueError:
            return None

        if pd.isna(y).any():
            return None

        y = np.asarray(y, dtype=int)
        counts = np.bincount(y)
        return y if len(counts) > 1 and counts.min() >= 2 else None

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
        # adata here already came through load_expression_csv, which applied the
        # policy; embed_kwargs() keeps this call a pass-through.
        policy = resolve_policy(task_cfg, None)

        embedder.eval()
        if torch.cuda.is_available() and hasattr(embedder, "cuda"):
            embedder.cuda()

        kwargs = dict(batch_size=batch_size, **policy.embed_kwargs())
        if not getattr(embedder, "fittable", False):
            kwargs["modality"] = "bulk"  # cell-line bulk → log1p + MAD gene selection
        result = embedder.embed(adata, **kwargs)
        df = result[0] if isinstance(result, tuple) else result
        return df.to_numpy(dtype=np.float32)

    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float | str]:
        endpoint = str(getattr(self, "_endpoint", "cmax_classification"))
        y_true = np.asarray(targets, dtype=np.float64).reshape(-1)
        y_hat = np.asarray(predictions, dtype=np.float64).reshape(-1)

        metrics: dict[str, float | str] = {
            "drug": str(getattr(self, "_drug", "")),
            "endpoint": endpoint,
            "n_train": float(getattr(self, "_n_train", 0)),
            "n_test": float(getattr(self, "_n_test", len(y_true))),
            "target_mean_train": float(getattr(self, "_target_mean_train", 0.0)),
            "target_mean_test": float(
                getattr(self, "_target_mean_test", y_true.mean() if len(y_true) else 0.0)
            ),
        }

        if endpoint == "ic50_regression":
            mse = mean_squared_error(y_true, y_hat)
            metrics.update(
                mse=float(mse),
                rmse=float(np.sqrt(mse)),
                mae=float(mean_absolute_error(y_true, y_hat)),
                r2=float(r2_score(y_true, y_hat)) if len(y_true) >= 2 else np.nan,
                pearson_rho=safe_corr(y_true, y_hat, method="pearson"),
                spearman_rho=safe_corr(y_true, y_hat, method="spearman"),
            )
            return metrics

        y_true_int = y_true.astype(int)
        y_prob = 1.0 / (1.0 + np.exp(-y_hat))
        threshold = float(getattr(self, "_prediction_threshold", 0.5))
        y_pred = (y_prob >= threshold).astype(int)

        metrics.update(
            accuracy=float(accuracy_score(y_true_int, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true_int, y_pred)),
            f1=float(f1_score(y_true_int, y_pred, zero_division=0)),
            precision=float(precision_score(y_true_int, y_pred, zero_division=0)),
            recall=float(recall_score(y_true_int, y_pred, zero_division=0)),
            mcc=float(matthews_corrcoef(y_true_int, y_pred)),
            positive_rate_train=float(getattr(self, "_positive_rate_train", 0.0)),
            positive_rate_test=float(
                getattr(
                    self,
                    "_positive_rate_test",
                    y_true_int.mean() if len(y_true_int) else 0.0,
                )
            ),
            response_threshold=float(getattr(self, "_response_threshold", 0.5)),
            prediction_threshold=threshold,
        )

        if np.unique(y_true_int).size == 2:
            metrics["auroc"] = float(roc_auc_score(y_true_int, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true_int, y_prob))
        else:
            metrics["auroc"] = np.nan
            metrics["auprc"] = np.nan

        return metrics


def aggregate_drug_sensitivity_results(
    metrics: Iterable[dict[str, float | str]],
) -> dict[str, float]:
    """Aggregate per-drug/per-endpoint metrics separately."""
    df = pd.DataFrame(metrics)
    out: dict[str, float] = {}

    for endpoint, endpoint_df in df.groupby("endpoint"):
        numeric = endpoint_df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            if not col.startswith("n_"):
                out[f"{endpoint}_mean_{col}"] = float(numeric[col].mean())

    # Cross-endpoint summary aliases for plot_ablation_benchmark.py compatibility
    rho_vals = [v for k, v in out.items() if k.endswith("_mean_pearson_rho")]
    if rho_vals:
        out["mean_pearson_r"] = float(np.mean(rho_vals))
    auroc_vals = [v for k, v in out.items() if k.endswith("_mean_auroc")]
    if auroc_vals:
        out["mean_auroc"] = float(np.mean(auroc_vals))

    return out