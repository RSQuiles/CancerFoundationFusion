"""
Abstract base class and registry for downstream fine-tuning tasks.

This module provides a standardized interface for implementing downstream tasks
that fine-tune a frozen pretrained embedder on domain-specific objectives.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

import anndata as ad


class DownstreamTask(ABC):
    """
    Abstract base class defining the interface for downstream fine-tuning tasks.

    Each downstream task must specify:
    - Head/model architecture
    - Dataset class for loading data
    - Data loading logic
    - Loss function
    - Evaluation metrics
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Unique identifier for the task (e.g., 'canc_type_class', 'deconv')."""
        pass

    @property
    @abstractmethod
    def config_key(self) -> str:
        """Configuration key path in YAML (e.g., 'finetune.canc_type_class')."""
        pass

    @abstractmethod
    def get_head_class(self) -> type[nn.Module]:
        """Return the head/model class to fine-tune."""
        pass

    @abstractmethod
    def get_dataset_class(self) -> type[Dataset]:
        """Return the dataset class for this task."""
        pass

    @abstractmethod
    def get_loss_fn(self, device: torch.device) -> nn.Module:
        """Instantiate and return the loss function."""
        pass

    @abstractmethod
    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[ad.AnnData | None, ad.AnnData | None, np.ndarray | Any, np.ndarray | Any]:
        """
        Load training and test data.

        Returns
        -------
        tuple
            (train_adata, test_adata, train_targets, test_targets)
            where targets can be labels (1D) or values (2D) depending on task type.
        """
        pass

    @abstractmethod
    def prepare_datasets(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray | Any,
        test_targets: np.ndarray | Any,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """
        Create dataset instances and return embedding dimension.

        Returns
        -------
        tuple
            (train_dataset, test_dataset, embedding_dim)
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute task-specific evaluation metrics.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (logits for classification, values for regression).
        targets : np.ndarray
            Ground truth targets.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names to values.
        """
        pass

    def validate_config(self, task_cfg: DictConfig) -> None:
        """
        Validate task-specific configuration.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        required_keys = ["pretrained_model_path", "head_learning_rate"]
        missing = [key for key in required_keys if getattr(task_cfg, key, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at finetune.{self.config_key}."
            )

    @staticmethod
    def hash_split_version(version: Any) -> int:
        """Convert a version string/number to a deterministic random seed."""
        version_str = str(version)
        digest = hashlib.sha256(version_str.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)


class TaskRegistry:
    """Registry for downstream task implementations."""

    _registry: dict[str, type[DownstreamTask]] = {}

    @classmethod
    def register(cls, task_class: type[DownstreamTask]) -> type[DownstreamTask]:
        """
        Register a downstream task class.

        Parameters
        ----------
        task_class : type[DownstreamTask]
            Task class to register (must have task_name property).

        Returns
        -------
        type[DownstreamTask]
            The registered class (for use as decorator).
        """
        # Instantiate to get task_name
        temp_instance = task_class()
        task_name = temp_instance.task_name
        cls._registry[task_name] = task_class
        return task_class

    @classmethod
    def get_task(cls, task_name: str) -> DownstreamTask:
        """
        Get a task instance by name.

        Parameters
        ----------
        task_name : str
            Name of the task (e.g., 'canc_type_class').

        Returns
        -------
        DownstreamTask
            Instantiated task object.

        Raises
        ------
        KeyError
            If task is not registered.
        """
        if task_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {available}. "
                f"Register new tasks using @TaskRegistry.register decorator."
            )
        return cls._registry[task_name]()

    @classmethod
    def get_config_key_for_task(cls, task_name: str) -> str:
        """Get the YAML config key for a task."""
        task = cls.get_task(task_name)
        return task.config_key

    @classmethod
    def list_tasks(cls) -> list[str]:
        """List all registered task names."""
        return sorted(cls._registry.keys())
