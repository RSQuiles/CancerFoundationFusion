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

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from evaluate.finetune.downstream_task import DownstreamTask
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)


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
                        self.first_cycle_steps * (self.cycle_mult ** self.cycle - 1) / (self.cycle_mult - 1)
                    )
            else:
                self.cycle = 0
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


class BaseDownstreamRunner:
    """
    Base runner implementing common downstream fine-tuning logic.

    Subclasses or tasks should inherit this and provide task-specific implementations
    via the DownstreamTask interface.
    """

    def __init__(self, cfg: DictConfig, task: DownstreamTask) -> None:
        """
        Initialize the runner.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config containing finetune section with task config.
        task : DownstreamTask
            Task object defining heads, datasets, metrics, etc.
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
        self.scheduler: GroupedCosineAnnealingWarmupRestarts | None = None
        self.loss_fn: nn.Module | None = None
        self.embedding_dim: int | None = None
        self.pretrained_model_stem: str | None = None

        # Embedder (frozen pretrained model)
        self.embedder = None

        # Task-specific state (subclasses can add more)
        self.task_state: dict[str, Any] = {}

    def _resolve_task_cfg(self) -> DictConfig:
        """Extract task config from full config."""
        config_key = self.task.config_key
        keys = config_key.split(".")
        current = self.cfg
        for key in keys:
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
        """Load frozen pretrained embedder from checkpoint."""
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

        embedder = CancerFoundation.load_from_checkpoint(resolved_path, strict=False)
        embedder.eval()
        for param in embedder.parameters():
            param.requires_grad = False

        if self.is_master:
            log.info(f"Loaded pretrained embedder from {resolved_path}")

        return embedder

    def _build_loaders(self) -> None:
        """
        Load data and build data loaders.

        Delegates to task.load_data() for task-specific data loading,
        then to task.prepare_datasets() for dataset creation.
        """
        # Load task-specific data
        train_adata, test_adata, train_targets, test_targets = self.task.load_data(
            self.task_cfg, self.embedder
        )

        # Get datasets (task creates embeddings internally)
        train_dataset, test_dataset, embedding_dim = self.task.prepare_datasets(
            train_adata,
            test_adata,
            train_targets,
            test_targets,
            self.embedder,
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
            log.info(f"Built loaders. embedding_dim={embedding_dim}")

    def _build_model(self) -> None:
        """Build task-specific head/model."""
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

        # Optimizer
        self.optimizer = Adam([{"params": head_params, "lr": head_learning_rate, "name": "head"}])

        # Scheduler
        max_lrs = [head_learning_rate]
        min_lr = float(getattr(self.task_cfg, "min_lr", 1e-6))
        configured_min_lr_ratio = getattr(self.task_cfg, "min_lr_ratio", None)
        min_lr_ratio = (
            float(configured_min_lr_ratio)
            if configured_min_lr_ratio is not None
            else min_lr / max(head_learning_rate, 1e-12)
        )

        self.scheduler = GroupedCosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(getattr(self.task_cfg, "first_cycle_steps", 15)),
            cycle_mult=float(getattr(self.task_cfg, "cycle_mult", 2)),
            max_lrs=max_lrs,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=int(getattr(self.task_cfg, "warmup_steps", 5)),
            gamma=float(getattr(self.task_cfg, "gamma", 0.9)),
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

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))

        running_loss = 0.0
        all_metrics: dict[str, list] = {}

        for step_idx, batch in enumerate(self.train_loader, start=1):
            # Parse batch (assumes batch is tuple of tensors or dict)
            if isinstance(batch, (tuple, list)):
                embeddings = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
            elif isinstance(batch, dict):
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            # Forward
            logits = self.model(embeddings)
            loss = self.loss_fn(logits, targets)

            # Backward with accumulation
            (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

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

        self.scheduler.step()
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

        if self.is_master:
            log.info(f"Evaluation metrics: {metrics}")

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> Path | None:
        """Save model checkpoint."""
        if not self.is_master:
            return None

        save_dir = Path(getattr(self.task_cfg, "save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": OmegaConf.to_container(self.task_cfg),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "task": self.task.task_name,
        }

        checkpoint_path = save_dir / f"epoch_epoch={epoch:02d}.ckpt"
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def run(self) -> dict[str, Any]:
        """Main training loop."""
        # Validate task config
        self.task.validate_config(self.task_cfg)

        # Setup
        self._setup_runtime()
        self.embedder = self._load_embedder()

        # Load data and build components
        self._build_loaders()
        self._build_model()
        self._build_optimization()

        # Training loop
        epochs = int(getattr(self.task_cfg, "epochs", 10))
        best_metrics = {}

        for epoch in range(epochs):
            if self.is_master:
                log.info(f"Starting epoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self._train_one_epoch(epoch)
            if self.is_master:
                log.info(f"Train metrics: {train_metrics}")

            # Evaluate
            test_metrics = self._evaluate()

            # Save checkpoint
            self._save_checkpoint(epoch, train_metrics, test_metrics)

            best_metrics = test_metrics

        return best_metrics


import sys
