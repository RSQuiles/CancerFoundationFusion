"""
Base runner for downstream fine-tuning tasks.

Implements all common logic for fine-tuning a frozen pretrained embedder on
downstream objectives. Task-specific logic is delegated to DownstreamTask subclasses.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from evaluate.finetune.downstream_task import DownstreamTask
from evaluate.finetune.utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

from evaluate.finetune.pca_baseline import PCAEmbedder

log = logging.getLogger(__name__)


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning mode: holds preprocessed expression data instead of pre-computed embeddings.

    ``gene_ids`` stores a common gene set for every cell.
    The ``__getitem__`` tuple format ``(expr_row, target)`` resembled embedding-based datsets to keep 
    training loop unchanged.
    """

    def __init__(
        self,
        expr: np.ndarray,
        gene_ids: torch.Tensor,
        targets: np.ndarray,
    ) -> None:
        self.expr = torch.from_numpy(np.asarray(expr, dtype=np.float32))
        self.gene_ids = gene_ids
        self.targets = torch.from_numpy(np.asarray(targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expr[idx], self.targets[idx]


class BaseDownstreamRunner:
    """
    Base runner implementing common downstream fine-tuning logic.

    Tasks should provide task-specific implementations via the DownstreamTask interface.
    """

    def __init__(
        self,
        cfg: DictConfig,
        task: DownstreamTask,
        embedder: Any = None,
    ) -> None:
        """
        Initialize the runner.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config containing finetune section with task config.
        task : DownstreamTask
            Task object defining heads, datasets, metrics, etc.
        embedder : Any, optional
            Pre-built embedder to use instead of loading from a checkpoint.
            When provided, ``_load_embedder()`` is skipped and
            ``pretrained_model_path`` is not required to point to a real file.
        """
        self.cfg = cfg
        self.task = task
        self.task_cfg = self._resolve_task_cfg()

        # Distributed setup
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        # Training state
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.loss_fn: nn.Module | None = None
        self.embedding_dim: int | None = None
        self.pretrained_model_stem: str | None = None

        # Externally provided embedder
        self.embedder = embedder

        # Task-specific state (subclasses can add more)
        self.task_state: dict[str, Any] = {}

        # Gene names selected during finetune-mode HVG preprocessing (set in _build_loaders)
        self.finetune_gene_names: list[str] | None = None

        # Detrmine finetuning status (end-to-end or just head)
        self.finetune = bool(getattr(self.task_cfg, "finetune_embedder", False)) and not isinstance(self.embedder, PCAEmbedder)


    def _resolve_task_cfg(self) -> DictConfig:
        """Extract task config from full config."""
        config_key = self.task.config_key
        keys = config_key.split(".")
        current = self.cfg
        for key in keys: 
            # Go down one level in the dictonary
            if key not in current or current[key] is None:
                raise ValueError(
                    f"Could not find config at {config_key}. "
                    f"Expected structure: {config_key}"
                )
            current = current[key]
        return current

    def _setup_runtime(self) -> None:
        """Initialize distributed training, CUDA, and random seeds."""
        # Initialize distributed backend if needed
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)

        # Set device
        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Set random seeds
        seed = int(getattr(self.task_cfg, "random_seed", 42)) + self.rank
        seed_all(seed)

        if self.is_master:
            log.info(f"Initialized runtime on device: {self.device}")

    def _load_embedder(self):
        """
        Load pretrained embedder from checkpoint.
        It can be kept frozen or not, for end-to-end finetuning
        """
        checkpoint_path = getattr(self.task_cfg, "pretrained_model_path", None)
        if not checkpoint_path:
            raise ValueError(
                f"finetune.{self.task.config_key}.pretrained_model_path must be set."
            )

        resolved_path = str(checkpoint_path)  # Can be expanded with hydra if needed
        self.pretrained_model_stem = Path(resolved_path).stem

        # Import model class dynamically to avoid circular imports
        sys.path.insert(0, "../")
        from cancerfoundation.model.model import CancerFoundation

        embedder = CancerFoundation.load_for_inference(resolved_path)
        for param in embedder.parameters():
            param.requires_grad = self.finetune

        if self.is_master:
            mode = "trainable (fine-tuning)" if self.finetune else "frozen"
            log.info(f"Loaded pretrained embedder from {resolved_path} [{mode}]")

        return embedder

    def _build_loaders(self) -> None:
        """
        Load data and build data loaders.

        Delegates to task.load_data() for task-specific data loading.
        In frozen-embedder mode, delegates to task.prepare_datasets() for embedding.
        In fine-tuning mode, preprocesses the adata directly and builds FineTuneDatasets
        so that raw expression data flows through the
        training loop, thus enabling gradient propagation through the embedder.
        """
        num_classes, train_adata, test_adata, train_targets, test_targets = self.task.load_data(
            self.task_cfg, self.embedder
        )

        if num_classes is not None:
            self.task_state["output_dim"] = num_classes

        # is_multi_fold = getattr(self.task, "_is_multi_fold", False)

        # FINETUNE BRANCH (single-fold only — multi-fold delegates to prepare_datasets)
        if self.finetune and hasattr(self.embedder, "preprocess_for_embedding"):
            normalized = bool(getattr(self.task_cfg, "normalized", False))
            # Modality drives gene selection (sc → seurat HVG, bulk/pseudobulk → log1p+MAD)
            modality = str(getattr(self.task_cfg, "modality", "sc"))

            # Preprocess train adata — gene selection runs on training cells only
            if self.is_master:
                log.info("Fine-tuning mode: preprocessing train split for embedding...")
            processed_train = self.embedder.preprocess_for_embedding(
                train_adata, normalized=normalized, modality=modality
            )
            kept_genes = processed_train.var.index.tolist()

            # Apply same gene set to test (no re-fit on test cells)
            if self.is_master:
                log.info("Fine-tuning mode: preprocessing test split with train gene set...")
            processed_test = self.embedder.preprocess_for_embedding(
                test_adata, normalized=normalized, gene_subset=kept_genes, modality=modality
            )

            gene_ids = torch.LongTensor([self.embedder.vocab[g] for g in kept_genes])
            self.finetune_gene_names = kept_genes

            train_expr = processed_train.X if isinstance(processed_train.X, np.ndarray) else processed_train.X.toarray()
            test_expr = processed_test.X if isinstance(processed_test.X, np.ndarray) else processed_test.X.toarray()

            # Encode string labels (classification) or pass numeric targets through
            if np.asarray(train_targets).dtype.kind in ("U", "S", "O"):
                classes, train_encoded = np.unique(train_targets, return_inverse=True)
                label_to_idx = {c: i for i, c in enumerate(classes)}
                test_encoded = np.array([label_to_idx[t] for t in test_targets], dtype=np.int64)
                train_encoded = train_encoded.astype(np.int64)
            else:
                train_encoded = np.asarray(train_targets)
                test_encoded = np.asarray(test_targets)

            train_dataset = FineTuneDataset(train_expr, gene_ids, train_encoded)
            test_dataset = FineTuneDataset(test_expr, gene_ids, test_encoded)
            self.embedding_dim = self.embedder.embsize

            if self.is_master:
                log.info(
                    f"Fine-tuning datasets built: train={len(train_dataset)}, "
                    f"test={len(test_dataset)}, genes={len(kept_genes)}, "
                    f"embedding_dim={self.embedding_dim}"
                )

        # FROZEN EMBEDDING BRANCH
        else:
            # Frozen-embedder path: task pre-computes embeddings
            train_dataset, test_dataset, embedding_dim = self.task.prepare_datasets(
                train_adata,
                test_adata,
                train_targets,
                test_targets,
                self.embedder,
                self.task_cfg,
            )
            self.embedding_dim = embedding_dim

        self.test_dataset_size = len(test_dataset)

        batch_size = int(getattr(self.task_cfg, "batch_size", 32))

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

        if self.is_master:
            log.info(f"Built loaders. embedding_dim={self.embedding_dim}")

    def _build_model(self) -> None:
        """Build task-specific head"""
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dimension not set. Build loaders first.")

        # Get head class from task
        head_class = self.task.get_head_class()

        # Instantiate with task-specific output dimension
        output_dim = self._get_output_dim()
        model = head_class(
            embedding_dim=self.embedding_dim,
            output_dim=output_dim,
            hidden_dim=int(getattr(self.task_cfg, "hidden_dim", 128)),
            dropout=float(getattr(self.task_cfg, "dropout", 0.0)),
        )

        # Ensure all parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        model = model.to(self.device)

        # Wrap in DDP if distributed
        if self.is_distributed:
            if self.device.type == "cuda":
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                model = DDP(model)

        self.model = model
        if self.is_master:
            log.info(f"Built model with output_dim={output_dim}")

    def _get_output_dim(self) -> int:
        """Get output dimension for the head. Override in subclass if needed."""
        # Default: assume task_state has 'num_outputs' or similar
        return self.task_state.get("output_dim", 1)

    def _build_optimization(self) -> None:
        """Build optimizer and scheduler."""
        if not hasattr(self.task_cfg, "head_learning_rate"):
            raise ValueError(
                f"finetune.{self.task.config_key}.head_learning_rate must be set."
            )

        head_learning_rate = float(self.task_cfg.head_learning_rate)

        # Get trainable parameters
        model = self.model.module if isinstance(self.model, DDP) else self.model
        head_params = [param for param in model.parameters() if param.requires_grad]

        param_groups = [{"params": head_params, "lr": head_learning_rate, "name": "head"}]

        # Ensure embedder parameters are trainable in FINETUNE BRANCH
        if self.finetune and self.embedder is not None:
            embedder_lr = float(getattr(self.task_cfg, "embedder_learning_rate", 1e-5))
            embedder_params = [p for p in self.embedder.parameters() if p.requires_grad]
            if embedder_params:
                param_groups.append({"params": embedder_params, "lr": embedder_lr, "name": "embedder"})
                if self.is_master:
                    log.info(
                        f"Added embedder param group: {len(embedder_params)} params, lr={embedder_lr}"
                    )

        # Optimizer
        weight_decay = float(getattr(self.task_cfg, "weight_decay", 0.0))
        self.optimizer = Adam(param_groups, weight_decay=weight_decay)

        # Scheduler
        max_lrs = [head_learning_rate]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(head_learning_rate, 1e-12)
        )

        # Loss function
        self.loss_fn = self.task.get_loss_fn(self.device)

        if self.is_master:
            log.info("Built optimizer and scheduler")

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.

        Implements standard training loop: forward, backward, optimizer step.
        Returns metrics.
        """
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        if self.finetune and self.embedder is not None:
            self.embedder.train()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        all_metrics: dict[str, list] = {}

        for step_idx, batch in enumerate(self.train_loader, start=1):
            # Parse batch (assumes batch is tuple of tensors or dict)
            if isinstance(batch, (tuple, list)):
                data = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
            elif isinstance(batch, dict):
                data = batch["data"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            # Forward — in fine-tuning mode, run raw expression through the embedder first
            if self.finetune and self.embedder is not None:
                gene_ids = self.train_loader.dataset.gene_ids
                embeddings = self.embedder.embed_for_finetune(gene_ids, data)
            else:
                embeddings = data

            logits = self.model(embeddings)
            loss = self.loss_fn(logits, targets)

            # Backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().item()

            # Task-specific metric computation during training
            step_metrics = self._compute_train_metrics(logits, targets)
            for key, val in step_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = {"loss": epoch_loss}
        for key, values in all_metrics.items():
            epoch_metrics[key] = np.mean(values)

        if self.is_distributed:
            for key in epoch_metrics:
                epoch_metrics[key] = get_reduced(epoch_metrics[key], self.local_rank, 0, self.world_size)

        return epoch_metrics

    def _compute_train_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """
        Compute metrics during training.

        Override in subclass or implement via task for custom metrics.
        """
        return {}

    def _evaluate(self) -> dict[str, float]:
        """
        Evaluate on test set.

        Returns metrics computed via task.compute_metrics().
        """
        if self.finetune and self.embedder is not None:
            self.embedder.eval()

        self.model.eval()

        if self.is_distributed:
            dist.barrier()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, (tuple, list)):
                    embeddings = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)
                elif isinstance(batch, dict):
                    embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                    targets = batch["targets"].to(self.device, non_blocking=True)
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                if self.finetune and self.embedder is not None:
                    gene_ids = self.train_loader.dataset.gene_ids
                    embeddings = self.embedder.embed_for_finetune(gene_ids, embeddings)

                logits = self.model(embeddings)
                all_predictions.append(logits.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        if self.is_distributed:
            # Gather predictions and targets across all ranks
            predictions = distributed_concat(torch.from_numpy(predictions), self.world_size).numpy()
            targets = distributed_concat(torch.from_numpy(targets), self.world_size).numpy()

        # Compute metrics via task
        metrics = self.task.compute_metrics(predictions, targets)

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> Path | None:
        """Save final model checkpoint (head + optionally finetuned embedder)."""
        if not self.is_master:
            return None

        save_dir = Path(getattr(self.task_cfg, "save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        # Strip DDP wrapper from head state dict
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        head_state = raw_model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state": head_state,
            "optimizer_state": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.task_cfg),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "task": self.task.task_name,
            "output_dim": self._get_output_dim(),
            "embedding_dim": self.embedding_dim,
            "finetuned_embedder": self.finetune,
        }

        if self.finetune and self.embedder is not None:
            checkpoint["embedder_state"] = self.embedder.state_dict()
            checkpoint["gene_names"] = self.finetune_gene_names

        checkpoint_path = save_dir / "final.ckpt"
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def run(self) -> dict[str, Any]:
        """Main training loop."""
        # Validate task config
        self.task.validate_config(self.task_cfg)

        # Setup
        self._setup_runtime()
        if self.embedder is None:
            self.embedder = self._load_embedder()
        else:
            self.pretrained_model_stem = type(self.embedder).__name__
            if self.is_master:
                log.info("Using externally provided embedder: %s", self.pretrained_model_stem)

        # Load data and build components
        self._build_loaders()

        # Multi-fold tasks complete all work inside prepare_datasets(); skip the loop.
        if getattr(self.task, "_multi_fold_metrics", None) is not None:
            log.info("Multi-fold evaluation complete. Skipping runner training loop.")
            return self.task._multi_fold_metrics

        self._build_model()
        self._build_optimization()

        # Training loop
        epochs = int(getattr(self.task_cfg, "epochs", 10))
        best_metrics = {}

        train_metrics: dict[str, float] = {}
        test_metrics: dict[str, float] = {}

        for epoch in range(epochs):
            if self.is_master:
                log.info(f"Starting epoch {epoch + 1}/{epochs}")

            train_metrics = self._train_one_epoch(epoch)
            if self.is_master:
                log.info(f"Train metrics: {train_metrics}")

            test_metrics = self._evaluate()
            if self.is_master:
                log.info(f"Evaluation metrics: {test_metrics}")

        self._save_checkpoint(epochs - 1, train_metrics, test_metrics)

        return test_metrics