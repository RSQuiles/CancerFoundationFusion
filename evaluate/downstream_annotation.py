from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any

import anndata as ad
import hydra
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COHORTS = ["BRCA", "BLCA", "GBM", "LGG", "LUAD", "UCEC"]
TASK_NAME = "canc_type_class"


class EmbeddingPredHead(nn.Module):
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


class GroupedCosineAnnealingWarmupRestarts:
    """Cosine warmup scheduler that preserves per-parameter-group max LRs."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        max_lrs: list[float],
        min_lr_ratio: float,
        cycle_mult: float = 1.0,
        warmup_steps: int = 0,
        gamma: float = 1.0,
    ) -> None:
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be smaller than first_cycle_steps.")
        if len(max_lrs) != len(optimizer.param_groups):
            raise ValueError("max_lrs must match optimizer.param_groups.")

        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lrs = [float(lr) for lr in max_lrs]
        self.max_lrs = list(self.base_max_lrs)
        self.min_lrs = [float(lr) * float(min_lr_ratio) for lr in self.base_max_lrs]
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = -1
        self.last_epoch = -1
        self._set_lrs(self.min_lrs)

    def _set_lrs(self, lrs: list[float]) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def get_lr(self) -> list[float]:
        if self.step_in_cycle == -1:
            return self.min_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                min_lr + (max_lr - min_lr) * self.step_in_cycle / self.warmup_steps
                for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
            ]
        return [
            min_lr
            + (max_lr - min_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.step_in_cycle - self.warmup_steps)
                    / (self.cur_cycle_steps - self.warmup_steps)
                )
            )
            / 2
            for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
        ]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    self.cycle = int(
                        math.log(
                            epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1,
                            self.cycle_mult,
                        )
                    )
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**self.cycle - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult**self.cycle)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [lr * (self.gamma**self.cycle) for lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)
        self._set_lrs(self.get_lr())

    def state_dict(self) -> dict[str, object]:
        return {
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "base_max_lrs": self.base_max_lrs,
            "max_lrs": self.max_lrs,
            "min_lrs": self.min_lrs,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
            "cur_cycle_steps": self.cur_cycle_steps,
            "cycle": self.cycle,
            "step_in_cycle": self.step_in_cycle,
            "last_epoch": self.last_epoch,
        }


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return emb, label


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.to_out = EmbeddingPredHead(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out(x)


class CancTypeClassEmbeddingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = self._resolve_task_cfg(cfg)

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.label_dict: np.ndarray | None = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.loss_fn: nn.Module | None = None
        self.pretrained_model_stem: str | None = None
        self.embedding_dim: int | None = None

        self.embedder = None

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "canc_type_class" in cfg.finetune:
            return cfg.finetune.canc_type_class
        raise ValueError(
            "Could not find a cancer type classification config. "
            "Expected cfg.finetune.canc_type_class."
        )

    def _setup_runtime(self) -> None:
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        seed_all(int(getattr(self.task_cfg, "random_seed", 42)) + self.rank)

    @staticmethod
    def _hash_split_version(version) -> int:
        version_str = str(version)
        digest = hashlib.sha256(version_str.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    def _load_tcga(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError("finetune.canc_type_class.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = sc.read_h5ad(data_path)
        if "project_id" not in adata.obs:
            raise ValueError("TCGA AnnData must contain obs['project_id'].")

        cohorts = list(getattr(self.task_cfg, "cohorts", DEFAULT_COHORTS))
        selected_projects = {f"TCGA-{cohort}" for cohort in cohorts}
        keep_mask = adata.obs["project_id"].astype(str).isin(selected_projects).to_numpy()
        adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(
                f"No TCGA samples matched cohorts {cohorts} in obs['project_id']."
            )

        adata.obs["cancer_type"] = adata.obs["project_id"].astype(str).str.removeprefix("TCGA-")
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    def _load_input_adata(self) -> ad.AnnData:
        log.info("Loading TCGA cohorts for cancer type classification")
        return self._load_tcga()

    def _load_gene_order(self) -> list[str] | None:
        gene_list_path = getattr(self.task_cfg, "gene_list_path", None)
        if not gene_list_path:
            return None
        resolved_path = Path(hydra.utils.to_absolute_path(str(gene_list_path)))
        with resolved_path.open() as handle:
            gene_order = [line.strip() for line in handle if line.strip()]
        if not gene_order:
            raise ValueError(f"Gene list is empty: {resolved_path}")
        return gene_order

    def _preprocess_adata(self, adata: ad.AnnData) -> ad.AnnData:
        gene_order = self._load_gene_order()
        if gene_order:
            counts = sparse.lil_matrix((adata.n_obs, len(gene_order)), dtype=np.float32)
            gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var_names.tolist())}
            matched_genes = [
                (target_idx, gene_to_idx[gene])
                for target_idx, gene in enumerate(gene_order)
                if gene in gene_to_idx
            ]
            if not matched_genes:
                raise ValueError("No genes matched the provided gene_list_path.")
            for target_idx, input_idx in matched_genes:
                counts[:, target_idx] = adata.X[:, input_idx]

            new_adata = ad.AnnData(X=counts.tocsr())
            new_adata.var_names = gene_order
            new_adata.obs_names = adata.obs_names.copy()
            new_adata.obs = adata.obs.copy()
            adata = new_adata

            log.info(
                "Matched %d/%d genes against the target gene list of size %d",
                len(matched_genes),
                len(gene_to_idx),
                len(gene_order),
            )

        normalized = bool(getattr(self.task_cfg, "normalized", True))
        if not normalized:
            min_genes = int(getattr(self.task_cfg, "min_genes", 200))
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        return adata

    def _prepare_train_test_data(self) -> tuple[ad.AnnData, ad.AnnData]:
        adata = self._load_input_adata()
        adata.obs["cancer_type"] = adata.obs["cancer_type"].astype(str)

        if bool(getattr(self.task_cfg, "merge_gbm_lgg", True)):
            adata.obs["cancer_type"] = adata.obs["cancer_type"].replace(
                {"GBM": "GBMLGG", "LGG": "GBMLGG"}
            )

        adata = self._preprocess_adata(adata)
        labels = adata.obs["cancer_type"]
        test_size = float(getattr(self.task_cfg, "test_size", 0.2))
        split_version = getattr(
            self.task_cfg,
            "train_test_split_version",
            getattr(self.task_cfg, "random_seed", 42),
        )
        split_seed = self._hash_split_version(split_version)

        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size,
            stratify=labels,
            random_state=split_seed,
        )
        return adata[train_idx].copy(), adata[test_idx].copy()

    def _load_embedder(self):
        checkpoint_path = getattr(self.task_cfg, "pretrained_model_path", None)
        if not checkpoint_path:
            raise ValueError("finetune.canc_type_class.pretrained_model_path must be set.")
        resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
        self.pretrained_model_stem = Path(resolved_path).stem

        # TODO: Replace this with your actual checkpoint loader.
        # It should return an object usable by `self._embed_adata`.
        embedder = load_embedding_model(resolved_path, self.device)
        log.info("Loaded pretrained embedding model from %s", resolved_path)
        return embedder

    def _embed_adata(self, adata: ad.AnnData) -> np.ndarray:
        # TODO: Replace this with your actual embedding call.
        # Expected output shape: [n_cells, embedding_dim]
        return embed_adata(self.embedder, adata)

    def _build_loaders(self, train_adata: ad.AnnData, test_adata: ad.AnnData) -> None:
        self.label_dict, train_labels = np.unique(
            np.asarray(train_adata.obs["cancer_type"]).astype(str),
            return_inverse=True,
        )
        label_to_idx = {label: idx for idx, label in enumerate(self.label_dict.tolist())}
        test_labels = np.array(
            [label_to_idx[label] for label in np.asarray(test_adata.obs["cancer_type"]).astype(str)],
            dtype=np.int64,
        )

        train_embeddings = np.asarray(self._embed_adata(train_adata), dtype=np.float32)
        test_embeddings = np.asarray(self._embed_adata(test_adata), dtype=np.float32)

        if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_cells, embedding_dim].")
        if train_embeddings.shape[1] != test_embeddings.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        self.embedding_dim = int(train_embeddings.shape[1])

        batch_size = int(getattr(self.task_cfg, "batch_size", 32))
        train_dataset = EmbeddingDataset(train_embeddings, train_labels)
        test_dataset = EmbeddingDataset(test_embeddings, test_labels)
        self.test_dataset_size = len(test_dataset)

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            test_sampler = SequentialDistributedSampler(
                test_dataset,
                batch_size=batch_size,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                shuffle=False,
            )
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def _build_model(self) -> None:
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dimension is not set. Build loaders first.")

        model = EmbeddingClassifier(
            embedding_dim=self.embedding_dim,
            output_dim=len(self.label_dict),
            hidden_dim=int(getattr(self.task_cfg, "hidden_dim", 128)),
            dropout=float(getattr(self.task_cfg, "dropout", 0.0)),
        )

        for param in model.parameters():
            param.requires_grad = True

        model = model.to(self.device)
        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

        self.model = model

    def _build_optimization(self) -> None:
        if not hasattr(self.task_cfg, "head_learning_rate"):
            raise ValueError("finetune.canc_type_class.head_learning_rate must be set.")
        head_learning_rate = float(self.task_cfg.head_learning_rate)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [param for param in model.parameters() if param.requires_grad]

        max_lrs = [head_learning_rate]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(head_learning_rate, 1e-12)
        )

        self.optimizer = Adam([{"params": head_params, "lr": head_learning_rate, "name": "head"}])
        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 15)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 2)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 5)),
            gamma=float(getattr(self.task_cfg, "gamma", 0.9)),
        )
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss = 0.0
        running_acc = 0.0

        for step_idx, (data, labels) in enumerate(self.train_loader, start=1):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(data)
            loss = self.loss_fn(logits, labels)
            (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            running_acc += (predictions == labels).float().mean().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * running_acc / len(self.train_loader)

        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.local_rank, 0, self.world_size)

        self.scheduler.step()
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def _evaluate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        predictions = []
        truths = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(data)
                loss = self.loss_fn(logits, labels)
                running_loss += loss.item()
                predictions.append(logits.argmax(dim=-1))
                truths.append(labels)

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        truths_np = truths.cpu().numpy()
        precision, recall, fscore, support = precision_recall_fscore_support(
            truths_np,
            predictions_np,
            labels=np.arange(len(self.label_dict)),
            zero_division=0,
        )

        test_loss = running_loss / len(self.test_loader)
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

        return {
            "loss": float(test_loss),
            "accuracy": float(accuracy_score(truths_np, predictions_np)),
            "f1_macro": float(f1_score(truths_np, predictions_np, average="macro")),
            "f1_weighted": float(f1_score(truths_np, predictions_np, average="weighted")),
            "confusion_matrix": confusion_matrix(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
            ),
            "classification_report": classification_report(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
                target_names=self.label_dict.tolist(),
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": self.label_dict.tolist(),
            "n_test_samples": int(len(truths_np)),
        }

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        test_metrics: dict[str, Any],
    ) -> Path | None:
        if not self.is_master:
            return None

        if not self.pretrained_model_stem:
            raise RuntimeError("Pretrained model stem is unavailable.")

        split_version = getattr(
            self.task_cfg,
            "train_test_split_version",
            getattr(self.task_cfg, "random_seed", 42),
        )
        model_name = f"{self.pretrained_model_stem}_{TASK_NAME}_head_only_v{split_version}"
        checkpoint_dir = ROOT / "output" / model_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{model_name}.pth"

        model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "label_dict": self.label_dict.tolist(),
                "embedding_dim": self.embedding_dim,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def run(self) -> dict[str, Any]:
        try:
            self._setup_runtime()
            self.embedder = self._load_embedder()

            train_adata, test_adata = self._prepare_train_test_data()
            if self.is_master:
                log.info(
                    "Prepared cancer type classification data: train=%d, test=%d, genes=%d",
                    train_adata.n_obs,
                    test_adata.n_obs,
                    train_adata.n_vars,
                )

            self._build_loaders(train_adata, test_adata)
            self._build_model()
            self._build_optimization()

            epochs = int(getattr(self.task_cfg, "epochs", 10))
            last_train_metrics = {"loss": float("nan"), "accuracy": float("nan")}

            for epoch in range(1, epochs + 1):
                last_train_metrics = self._train_one_epoch(epoch)
                if self.is_master:
                    log.info(
                        "Epoch %d | Training Loss: %.6f | Accuracy: %.4f%%",
                        epoch,
                        last_train_metrics["loss"],
                        last_train_metrics["accuracy"],
                    )

            test_metrics = self._evaluate()
            checkpoint_path = self._save_checkpoint(
                epoch=epochs,
                train_metrics={
                    "loss": float(last_train_metrics["loss"]),
                    "accuracy": float(last_train_metrics["accuracy"]),
                },
                test_metrics=test_metrics,
            )
            if checkpoint_path is not None:
                test_metrics["checkpoint_path"] = str(checkpoint_path)

            return {
                "train_loss": float(last_train_metrics["loss"]),
                "train_accuracy": float(last_train_metrics["accuracy"]),
                **test_metrics,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()


def load_runner_config(config_path: str | Path) -> DictConfig:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    if "finetune" not in cfg or cfg.finetune is None:
        raise ValueError("Config must contain a top-level `finetune` section.")

    if "canc_type_class" not in cfg.finetune or cfg.finetune.canc_type_class is None:
        raise ValueError("Config must contain `finetune.canc_type_class`.")

    task_cfg = cfg.finetune.canc_type_class
    required_keys = ["pretrained_model_path", "tcga_data_dir", "head_learning_rate"]
    missing = [key for key in required_keys if getattr(task_cfg, key, None) in (None, "")]
    if missing:
        raise ValueError(
            f"Missing required keys in `finetune.canc_type_class`: {missing}"
        )

    return cfg

def load_embedding_model(checkpoint_path: str, device: torch.device):
    """
    Load your frozen embedding model from checkpoint.
    Return any object needed by `embed_adata`.
    """
    raise NotImplementedError("Implement load_embedding_model(checkpoint_path, device)")


def embed_adata(model: Any, adata: ad.AnnData) -> np.ndarray:
    """
    Return a numpy array of shape [n_cells, embedding_dim].
    """
    raise NotImplementedError("Implement embed_adata(model, adata)")

def main(config_path: str) -> None:
    cfg = load_runner_config(config_path)
    runner = CancTypeClassEmbeddingRunner(cfg)
    results = runner.run()

    if runner.is_master:
        log.info("Final results: %s", results)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run cancer type classification from embeddings.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file for the runner.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config)