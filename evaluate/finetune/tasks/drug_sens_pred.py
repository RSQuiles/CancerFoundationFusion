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
        self, task_cfg: DictConfig, embedder: Any, patient_id_column: str = "dbgap_subject_id",
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
        df_meta_aligned = df_meta[df_meta["dbgap_dnaseq_sample"].isin(shared)]
        df_meta_aligned = df_meta_aligned.set_index("dbgap_dnaseq_sample").reindex(shared)

        if patient_id_column in df_meta_aligned.columns:
            patient_ids = (
                df_meta_aligned[patient_id_column]
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

        expr_gene_names = df_expr.columns.tolist()
       # Fill NaNs in expression matrix
        n_nan_expr = df_expr[expr_gene_names].isna().sum().sum()
        if n_nan_expr > 0:
            log.warning(f"BeatAML: {n_nan_expr} NaN values in expression matrix — filling with 0")
            df_expr[expr_gene_names] = df_expr[expr_gene_names].fillna(0.0)

        # Translate gene names to model's vocabulary
        expr_gene_names_translated = translate_gene_symbols(expr_gene_names, "symbol_to_ensembl.json")
        adata   = parquet_to_adata(df_expr, expr_gene_names)
        adata.var_names = pd.Index(expr_gene_names_translated)
        adata = deduplicate_var_names(adata)

        targets = df_drug.to_numpy(dtype=np.float32)   # NaN preserved, handled in the masked loss

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
        df_emb = embedder.embed(adata, batch_size=batch_size, normalized=True)
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