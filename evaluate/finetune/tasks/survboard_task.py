"""
SurvBoard survival prediction task.

Integrates the CancerFoundation embedder with the SurvBoard benchmark. For each
configured (cohort, cancer, fold) the task:

  1. Loads expression + survival labels from SurvBoard parquet files.
  2. Applies SurvBoard's pre-made train/test split indices.
  3. Embeds all samples with the frozen CFF encoder.
  4. Trains an EmbeddingPredHead (output_dim=1) with Cox PH loss.
  5. Fits a Breslow estimator (pure NumPy) to produce per-patient survival functions.
  6. Saves survival-function CSVs in SurvBoard's consolidated format for external
     evaluation with SurvBoard's evaluate_metrics.py.

CSV format (matches driver_unimodal.py):
    Rows     = test samples
    Columns  = event-time grid (float) + metadata cols
    Metadata = model_type, modality, project, cancer, split

Single-fold usage (fold_index = 0 ... 24):
    The runner trains one model; compute_metrics() returns C-index + saves CSV.

Multi-fold usage (fold_index = "all"):
    prepare_datasets() embeds the full dataset once, then trains a separate Cox
    head for each fold, saves per-fold CSVs, and stores the mean C-index.
    compute_metrics() then simply returns the pre-computed aggregated metrics.

Config key: finetune.survival
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry
from evaluate.finetune.tasks.components import EmbeddingPredHead, LinearPredHead
from evaluate.finetune.utils import parquet_to_adata, translate_gene_symbols, strip_ensembl_versions, deduplicate_var_names


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class SurvivalEmbeddingDataset(Dataset):
    """Wraps (embedding, [OS_days, OS_event]) pairs for CoxPHLoss."""

    def __init__(
        self,
        embeddings: np.ndarray,
        times:      np.ndarray,
        events:     np.ndarray,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.times      = np.asarray(times,      dtype=np.float32)
        self.events     = np.asarray(events,      dtype=np.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = torch.from_numpy(self.embeddings[idx]).float()
        target = torch.tensor(
            [self.times[idx], self.events[idx]], dtype=torch.float32
        )
        return emb, target


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

class CoxPHLoss(nn.Module):
    """
    Negative Cox partial log-likelihood.

    Inputs
    ------
    risk    : (N,) or (N, 1) — scalar log-risk scores; higher = shorter survival
    targets : (N, 2)         — column 0 = OS_days, column 1 = OS_event (0/1)
    """

    def forward(self, risk: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        risk  = risk.squeeze(-1)
        time  = targets[:, 0]
        event = targets[:, 1]

        if event.sum() == 0:
            return risk.sum() * 0.0

        order       = torch.argsort(time, descending=True)
        risk_s      = risk[order]
        event_s     = event[order]
        log_cumrisk = torch.logcumsumexp(risk_s, dim=0)
        pll         = (event_s * (risk_s - log_cumrisk)).sum()

        return -pll / event_s.sum()


# --------------------------------------------------------------------------- #
# Breslow estimator (pure NumPy)
# --------------------------------------------------------------------------- #

def _breslow_survival(
    train_times:  np.ndarray,
    train_events: np.ndarray,
    test_risk:    np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-patient survival functions via the Breslow estimator.

    Uses zero training risk scores (equivalent to a KM-based baseline hazard).

    Returns
    -------
    surv_mat  : (N_test, T)  survival probabilities at each unique event time
    time_grid : (T,)          unique event times (sorted ascending)
    """
    train_events = train_events.astype(bool)
    order        = np.argsort(train_times)
    t_sorted     = train_times[order]
    e_sorted     = train_events[order]

    time_grid = np.unique(t_sorted[e_sorted])
    if len(time_grid) == 0:
        return np.ones((len(test_risk), 1)), np.array([train_times.max()])

    dH0 = np.zeros(len(time_grid))
    for i, t_i in enumerate(time_grid):
        n_at_risk  = (t_sorted >= t_i).sum()
        n_events_i = ((t_sorted == t_i) & e_sorted).sum()
        dH0[i]     = n_events_i / max(n_at_risk, 1)

    H0 = np.cumsum(dH0)
    S0 = np.exp(-H0)

    exp_risk = np.exp(np.clip(test_risk, -75.0, 75.0))
    surv_mat = np.power(S0[None, :], exp_risk[:, None])   # (N_test, T)

    return surv_mat, time_grid


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

@TaskRegistry.register
class SurvBoardTask(DownstreamTask):
    """
    SurvBoard survival prediction as a DownstreamTask.

    Trains a scalar risk-score head with Cox PH loss, then converts risk scores
    to per-patient survival functions via a Breslow estimator and saves them in
    SurvBoard's consolidated CSV format for downstream evaluation.

    Config key: finetune.survival
    Required config fields:
        survboard_data_dir  : Path to the SurvBoard data directory.
        cancer_types        : List of cancer type codes, e.g. ["BRCA", "LUAD"].
        cohorts             : List of cohort names, e.g. ["TCGA", "METABRIC"].
        splits_dir          : Path to SurvBoard split index CSVs.
        fold_index          : Which fold to use (0-indexed) or "all" for all folds.
    Optional fields:
        survboard_results_dir : Where to write survival-function CSVs.
    """

    # ---- DownstreamTask interface ----------------------------------------- #

    @property
    def task_name(self) -> str:
        return "survival"

    @property
    def config_key(self) -> str:
        return "finetune.survival"

    def get_head_class(self) -> type[nn.Module]:
        return LinearPredHead

    def get_dataset_class(self) -> type[Dataset]:
        return SurvivalEmbeddingDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        return CoxPHLoss().to(device)

    def validate_config(self, task_cfg: DictConfig) -> None:
        super().validate_config(task_cfg)
        required = ["survboard_data_dir", "cancer_types", "cohorts"]
        missing  = [k for k in required if getattr(task_cfg, k, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )
        # Save configuration
        self.task_cfg = task_cfg

    # ---- Data loading ------------------------------------------------------- #

    def load_data(
        self,
        task_cfg: DictConfig,
        embedder: Any,
    ) -> tuple[int, ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        """
        Load SurvBoard parquet files and apply pre-made CV splits.

        When fold_index == "all", all fold splits are loaded and stored; the
        per-fold model training happens inside prepare_datasets().

        Returns (1, train_adata, test_adata, train_targets, test_targets)
        where targets shape = (N, 2) = [OS_days, OS_event].
        """
        data_dir     = Path(hydra.utils.to_absolute_path(str(task_cfg.survboard_data_dir)))
        cancer_types = list(task_cfg.cancer_types)
        cohorts      = list(task_cfg.cohorts)

        dfs:        list[pd.DataFrame]         = []
        block_info: list[tuple[str, str, int]] = []

        for cohort in cohorts:
            for ct in cancer_types:
                path = data_dir / f"{cohort}_{ct}.parquet"
                if not path.exists():
                    continue
                df                 = pd.read_parquet(path)
                df["_cohort"]      = cohort
                df["_cancer_type"] = ct
                block_info.append((cohort, ct, len(df)))
                dfs.append(df)
                log.info(f"Loaded {path.name}  ({len(df)} samples)")

        if not dfs:
            raise ValueError(
                f"No SurvBoard parquet files found for cohorts={cohorts}, "
                f"cancer_types={cancer_types} in {data_dir}."
            )

        df_all = pd.concat(dfs, axis=0, ignore_index=True)

        meta_cols = {"OS_days", "OS_event", "OS", "_cancer_type", "_cohort"}
        gene_cols = [c for c in df_all.columns if c not in meta_cols]

        # Translate gene names to model's vocabulary
        gene_cols_stripped = strip_ensembl_versions(gene_cols)

        event_col = "OS" if "OS" in df_all.columns else "OS_event"
        times     = df_all["OS_days"].to_numpy(dtype=np.float32)
        events    = df_all[event_col].to_numpy(dtype=np.float32)
        targets   = np.stack([times, events], axis=1)

        adata = parquet_to_adata(df_all, gene_cols)
        # print(adata.var_names)
        adata.var_names = pd.Index(gene_cols_stripped)
        # print(adata.var_names)
        adata = deduplicate_var_names(adata)

        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        adata.X = np.nan_to_num(np.asarray(adata.X, dtype=np.float32), nan=0.0)
        adata   = deduplicate_var_names(adata)

        adata.obs["cancer_type"] = df_all["_cancer_type"].values
        adata.obs["cohort"]      = df_all["_cohort"].values
        adata.obs["OS_days"]     = times
        adata.obs["OS_event"]    = events

        # Persist common state
        self._block_info = block_info
        self._task_cfg   = task_cfg

        # Parse fold_index: single int or "all"
        fold_cfg           = getattr(task_cfg, "fold_index", 0)
        self._is_all_folds = str(fold_cfg).lower() == "all"

        if self._is_all_folds:
            self._full_adata      = adata
            self._full_targets    = targets
            self._all_fold_splits = self._make_all_splits(adata, task_cfg, block_info)
            n_folds               = len(self._all_fold_splits)
            log.info(f"Multi-fold mode: {n_folds} folds, {adata.n_obs} total samples")
            train_idx, test_idx = self._all_fold_splits[0]
            self._fold_index = 0
        else:
            self._fold_index = int(fold_cfg)
            train_idx, test_idx = self._make_split(adata, task_cfg, block_info, self._fold_index)

        self._test_idx = test_idx

        log.info(
            f"SurvBoard: {adata.n_obs} total, {int(events.sum())} events. "
            f"Train={len(train_idx)}, Test={len(test_idx)}"
        )

        return (
            1,
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _make_split(
        self,
        adata:      ad.AnnData,
        task_cfg:   DictConfig,
        block_info: list[tuple[str, str, int]],
        fold_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (train_indices, test_indices) into the concatenated adata for one fold.

        Loads from:
            {splits_dir}/{COHORT}/{CANCER}_train_splits.csv
            {splits_dir}/{COHORT}/{CANCER}_test_splits.csv
        CSV rows = folds; values = NaN-padded integer row indices into the parquet.
        """
        splits_dir = getattr(task_cfg, "splits_dir", None)
        if splits_dir is None:
            raise ValueError(
                f"splits_dir must be configured for {self.task_name}. "
                f"Set it at {self.config_key}.splits_dir."
            )

        splits_base = Path(hydra.utils.to_absolute_path(str(splits_dir)))
        all_train: list[np.ndarray] = []
        all_test:  list[np.ndarray] = []
        global_off = 0

        for cohort, ct, block_size in block_info:
            tr_path = splits_base / cohort / f"{ct}_train_splits.csv"
            te_path = splits_base / cohort / f"{ct}_test_splits.csv"

            if not tr_path.exists():
                raise FileNotFoundError(f"Train splits not found: {tr_path}")
            if not te_path.exists():
                raise FileNotFoundError(f"Test splits not found: {te_path}")

            try:
                local_train = (
                    pd.read_csv(tr_path)
                    .iloc[fold_index, :]
                    .dropna()
                    .astype(int)
                    .to_numpy()
                )
                local_test = (
                    pd.read_csv(te_path)
                    .iloc[fold_index, :]
                    .dropna()
                    .astype(int)
                    .to_numpy()
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not load splits for {cohort}/{ct} fold {fold_index}: {exc}"
                ) from exc

            all_train.append(local_train + global_off)
            all_test.append(local_test  + global_off)
            global_off += block_size

        log.info(f"Loaded SurvBoard pre-made splits: fold={fold_index}")
        return np.concatenate(all_train), np.concatenate(all_test)

    def _make_all_splits(
        self,
        adata:      ad.AnnData,
        task_cfg:   DictConfig,
        block_info: list[tuple[str, str, int]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load every fold row from the split CSVs; return list of (train_idx, test_idx)."""
        splits_dir = getattr(task_cfg, "splits_dir", None)
        if splits_dir is None:
            raise ValueError(
                f"splits_dir must be configured for {self.task_name}."
            )

        splits_base = Path(hydra.utils.to_absolute_path(str(splits_dir)))

        # Determine n_folds from the first available split CSV
        n_folds: int | None = None
        for cohort, ct, _ in block_info:
            tr_path = splits_base / cohort / f"{ct}_train_splits.csv"
            if tr_path.exists():
                n_folds = len(pd.read_csv(tr_path))
                break

        if n_folds is None:
            raise FileNotFoundError(
                f"No split CSV files found under {splits_base} for the configured cancer types."
            )

        log.info(f"Detected {n_folds} folds in split CSVs.")
        return [
            self._make_split(adata, task_cfg, block_info, f)
            for f in range(n_folds)
        ]

    # ---- Dataset preparation ------------------------------------------------ #

    def prepare_datasets(
        self,
        train_adata:   ad.AnnData,
        test_adata:    ad.AnnData,
        train_targets: np.ndarray,
        test_targets:  np.ndarray,
        embedder:      Any,
    ) -> tuple[Dataset, Dataset, int]:
        """
        Embed data and create datasets.

        In all-fold mode, also trains one Cox head per fold, saves per-fold
        survival CSVs, and stores the averaged C-index for compute_metrics().
        Returns fold-0 datasets so the runner's training loop is valid.
        """
        if self._is_all_folds:
            return self._prepare_all_folds(embedder)

        # Single-fold path
        train_emb     = self._embed_adata(embedder, train_adata)
        test_emb      = self._embed_adata(embedder, test_adata)
        embedding_dim = train_emb.shape[1]

        self._train_times  = train_targets[:, 0]
        self._train_events = train_targets[:, 1]

        train_dataset = SurvivalEmbeddingDataset(
            train_emb, train_targets[:, 0], train_targets[:, 1]
        )
        test_dataset = SurvivalEmbeddingDataset(
            test_emb, test_targets[:, 0], test_targets[:, 1]
        )

        log.info(f"Survival datasets ready — embedding_dim={embedding_dim}")
        return train_dataset, test_dataset, embedding_dim

    def _prepare_all_folds(self, embedder: Any) -> tuple[Dataset, Dataset, int]:
        """
        Embed the full dataset once, then iterate over every fold:
          - Train a Cox head on the fold's training split.
          - Compute Breslow survival functions on the test split.
          - Save the per-fold survival CSV.
          - Record the Harrell C-index.

        Aggregated metrics (mean C-index) are stored in self._multi_fold_metrics
        for retrieval by compute_metrics().

        Returns fold-0 datasets so the outer runner loop has valid data.
        """
        log.info("Embedding full dataset for multi-fold evaluation …")
        full_emb      = self._embed_adata(embedder, self._full_adata)
        embedding_dim = full_emb.shape[1]
        full_times    = self._full_targets[:, 0]
        full_events   = self._full_targets[:, 1]

        device = next(embedder.parameters()).device

        cfg     = self._task_cfg
        epochs  = int(getattr(cfg, "epochs", 30))
        bs      = int(getattr(cfg, "batch_size", 64))

        n_folds        = len(self._all_fold_splits)
        fold_c_indices: list[float] = []

        for fold_i, (train_idx, test_idx) in enumerate(self._all_fold_splits):
            log.info(f"Fold {fold_i + 1}/{n_folds}: train={len(train_idx)}, test={len(test_idx)}")

            train_emb    = full_emb[train_idx]
            test_emb     = full_emb[test_idx]
            train_times  = full_times[train_idx]
            train_events = full_events[train_idx]
            test_times   = full_times[test_idx]
            test_events  = full_events[test_idx]

            head = self._train_cox_head(
                train_emb, train_times, train_events, device, epochs, bs
            )

            head.eval()
            with torch.no_grad():
                test_risk = (
                    head(torch.from_numpy(test_emb).float().to(device))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

            surv_mat, time_grid = _breslow_survival(train_times, train_events, test_risk)

            # Update instance state so _save_survival_csvs uses the correct fold/indices
            self._test_idx   = test_idx
            self._fold_index = fold_i
            self._save_survival_csvs(surv_mat, time_grid, test_times, test_events, self.get_model_name())

            c_idx = _c_index(test_times, test_risk, test_events.astype(bool))
            fold_c_indices.append(c_idx)
            log.info(
                f"  Fold {fold_i}: C-index={c_idx:.4f}, "
                f"n_events={int(test_events.sum())}"
            )

        mean_c = float(np.mean(fold_c_indices))
        log.info(
            f"Multi-fold C-index: mean={mean_c:.4f} (±{float(np.std(fold_c_indices)):.4f}) "
            f"across {n_folds} folds"
        )

        self._multi_fold_metrics = {
            "c_index":    mean_c,
            "n_events":   int(full_events.astype(bool).sum()),
            "event_rate": float(full_events.mean()),
        }

        # Restore fold-0 state so the runner's evaluate() call works normally
        train_idx_0, test_idx_0 = self._all_fold_splits[0]
        self._test_idx     = test_idx_0
        self._fold_index   = 0
        self._train_times  = full_times[train_idx_0]
        self._train_events = full_events[train_idx_0]

        train_dataset = SurvivalEmbeddingDataset(
            full_emb[train_idx_0], full_times[train_idx_0], full_events[train_idx_0]
        )
        test_dataset = SurvivalEmbeddingDataset(
            full_emb[test_idx_0], full_times[test_idx_0], full_events[test_idx_0]
        )
        return train_dataset, test_dataset, embedding_dim

    def _train_cox_head(
        self,
        train_emb:    np.ndarray,
        train_times:  np.ndarray,
        train_events: np.ndarray,
        device:       torch.device,
        epochs:       int,
        batch_size:   int,
    ):
        """Train an prediction head (output_dim=1) with CoxPHLoss for one fold."""
        cfg     = self._task_cfg
        lr      = float(getattr(cfg, "head_learning_rate", 1e-3))
        hidden  = int(getattr(cfg, "hidden_dim", 128))
        dropout = float(getattr(cfg, "dropout", 0.1))

        head_class = self.get_head_class()
        head = head_class(
            embedding_dim=train_emb.shape[1],
            output_dim=1,
            hidden_dim=hidden,
            dropout=dropout,
        ).to(device)

        optimizer = Adam(head.parameters(), lr=lr)
        loss_fn   = CoxPHLoss().to(device)

        dataset = SurvivalEmbeddingDataset(train_emb, train_times, train_events)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        head.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for emb_b, tgt_b in loader:
                emb_b = emb_b.to(device)
                tgt_b = tgt_b.to(device)
                optimizer.zero_grad()
                loss = loss_fn(head(emb_b), tgt_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                log.debug(
                    f"    Epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(loader):.4f}"
                )

        return head

    def _embed_adata(
        self, embedder: Any, adata: ad.AnnData, batch_size: int = 64
    ) -> np.ndarray:
        embedder.eval()
        embedder.cuda()
        return embedder.embed(adata, batch_size=batch_size, normalized=True).to_numpy()

    # ---- Evaluation --------------------------------------------------------- #

    def get_model_name(self):
        model_name = Path(getattr(self.task_cfg, "pretrained_model_path", "unknown")).parent.name
        log.info(f"Model name: {model_name}")
        return model_name

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets:     np.ndarray,
    ) -> dict[str, float]:
        """
        Compute Harrell's C-index and write per-cancer survival-function CSVs.

        In all-fold mode, returns the pre-computed aggregated metrics stored by
        _prepare_all_folds() and skips redundant computation.

        Returns
        -------
        dict with keys: c_index, n_events, event_rate
        """
        if self._is_all_folds:
            return self._multi_fold_metrics

        risk   = predictions[:, 0] if predictions.ndim == 2 else predictions
        times  = targets[:, 0]
        events = targets[:, 1].astype(bool)

        c_idx = _c_index(times, risk, events)

        surv_mat, time_grid = _breslow_survival(
            self._train_times, self._train_events, risk
        )
        self._save_survival_csvs(surv_mat, time_grid, times, events, self.get_model_name())

        n_events = int(events.sum())
        log.info(
            f"Survival: C-index={c_idx:.4f}, n_events={n_events}, "
            f"event_rate={events.mean():.3f}"
        )

        return {
            "c_index":    float(c_idx),
            "n_events":   n_events,
            "event_rate": float(events.mean()),
        }

    def _save_survival_csvs(
        self,
        surv_mat:  np.ndarray,
        time_grid: np.ndarray,
        times:     np.ndarray,
        events:    np.ndarray,
        model_name: str,
    ) -> None:
        """
        Write one consolidated CSV per (cohort, cancer) in SurvBoard format.

        Output path:
            {survboard_results_dir}/{cohort}/{cancer}/{model_name}/split_{fold_index}.csv

        Uses self._test_idx and self._fold_index, which are updated per-fold in
        multi-fold mode before each call.
        """
        results_dir = self._resolve_results_dir()
        fold        = self._fold_index
        offset      = 0

        for cohort, cancer, block_size in self._block_info:
            block_start = offset
            block_end   = offset + block_size

            mask        = (self._test_idx >= block_start) & (self._test_idx < block_end)
            pos_in_test = np.where(mask)[0]

            if len(pos_in_test) == 0:
                offset += block_size
                continue

            sf_df = pd.DataFrame(surv_mat[pos_in_test], columns=time_grid)
            sf_df["model_type"] = model_name
            sf_df["modality"]   = f"{model_name}_embeddings"
            sf_df["project"]    = cohort
            sf_df["cancer"]     = cancer
            sf_df["split"]      = fold

            out_dir = results_dir / cohort / cancer / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"split_{fold}.csv"
            sf_df.to_csv(out_path, index=False)
            log.info(
                f"Saved survival functions: {out_path} "
                f"({len(pos_in_test)} test samples, {len(time_grid)} time points)"
            )

            offset += block_size

    def _resolve_results_dir(self) -> Path:
        """Determine output directory for survival function CSVs."""
        cfg    = self._task_cfg
        custom = getattr(cfg, "survboard_results_dir", None)
        if custom:
            return Path(hydra.utils.to_absolute_path(str(custom)))
        save_dir = getattr(cfg, "save_dir", "./checkpoints/survival")
        return Path(hydra.utils.to_absolute_path(str(save_dir))) / "survival_functions_cff"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _c_index(
    times:  np.ndarray,
    risk:   np.ndarray,
    events: np.ndarray,
) -> float:
    """Harrell's concordance index (pure NumPy, O(n²) in event count)."""
    event_mask = events.astype(bool)
    et  = times[event_mask]
    rs  = risk[event_mask]

    at_risk   = times[None, :] - et[:, None] > 0          # (n_ev, n_all)
    risk_diff = rs[:, None] - risk[None, :]                # (n_ev, n_all)

    concordant  = float(((risk_diff > 0)  & at_risk).sum())
    tied        = float(((risk_diff == 0) & at_risk).sum())
    permissible = float(at_risk.sum())

    return (concordant + 0.5 * tied) / permissible if permissible > 0 else 0.5
