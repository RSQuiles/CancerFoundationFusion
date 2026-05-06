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
        rna_genes_translated = translate_gene_symbols(rna_genes, "symbol_to_ensembl.json")
        adata = parquet_to_adata(df_rna, rna_genes)
        adata.var_names = pd.Index(rna_genes_translated)
        adata = deduplicate_var_names(adata)

        targets = df_prot[prot_genes].to_numpy(dtype=np.float32)
        targets = np.nan_to_num(targets, nan=0.0)  # Impute nan values in targets

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
        df_emb = embedder.embed(adata, batch_size=batch_size, log1p_only=True)
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