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
        order = torch.argsort(time, descending=True)
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
                    # log.warning(f"SurvBoard parquet not found, skipping: {parquet_path}")
                    continue
                log.info(f"Loaded {parquet_path}")
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
        # Fill NaNs from cohort gene-set mismatches
        if hasattr(adata.X, "toarray"):  # sparse
            adata.X = np.nan_to_num(adata.X.toarray(), nan=0.0)
        else:
            adata.X = np.nan_to_num(adata.X, nan=0.0)

        # Make gene names compatible with vocabulary
        # stripped_genes = strip_ensembl_versions(adata.var_names.tolist())
        # adata.var_names = translate_gene_symbols(stripped_genes)

        # Manage the generated duplicated column names
        adata = deduplicate_var_names(adata)

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
        fold_index = int(getattr(task_cfg, "fold_index", 0)) # Use first split by default

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
        df_emb = embedder.embed(adata, batch_size=batch_size, normalized=True)
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