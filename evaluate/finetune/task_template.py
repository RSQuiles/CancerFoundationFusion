"""
Template for creating a new downstream task.

Copy this file and replace the placeholders to implement your own downstream task.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from evaluate.finetune.downstream_task import DownstreamTask, TaskRegistry

log = logging.getLogger(__name__)


# ============================================================================
# Step 1: Define Dataset Class (if needed)
# ============================================================================


class MyTaskDataset(Dataset):
    """Dataset for MyTask.
    
    Customize this class to handle your task-specific data format.
    """

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        """
        Parameters
        ----------
        embeddings : np.ndarray
            Shape [n_samples, embedding_dim]
        targets : np.ndarray
            Shape [n_samples] or [n_samples, n_outputs] depending on task type
        """
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)  # Adjust dtype as needed
        
        # Optional: Normalize targets (e.g., for regression or classification)
        # self.targets = self.targets / self.targets.sum(axis=1, keepdims=True)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        target = torch.from_numpy(self.targets[index]).float()
        return emb, target


# ============================================================================
# Step 2: Define Head/Model Class (if needed)
# ============================================================================


class MyTaskHead(nn.Module):
    """Custom head/model for MyTask.
    
    Override this if you need a different architecture than EmbeddingPredHead.
    The base runner expects this signature:
        __init__(self, embedding_dim, output_dim, hidden_dim, dropout)
        forward(self, x: Tensor) -> Tensor
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Step 3: Implement Task Class
# ============================================================================


@TaskRegistry.register
class MyTask(DownstreamTask):
    """My custom downstream task.
    
    This task is registered and can be used with the framework automatically.
    """

    @property
    def task_name(self) -> str:
        """Unique task identifier. Used in CLI and registry."""
        return "my_task"

    @property
    def config_key(self) -> str:
        """YAML config path. Must match your config structure."""
        return "finetune.my_task"

    # ========== ARCHITECTURE ==========

    def get_head_class(self) -> type[nn.Module]:
        """Return the head/model class."""
        return MyTaskHead  # or EmbeddingPredHead

    def get_dataset_class(self) -> type[Dataset]:
        """Return the dataset class."""
        return MyTaskDataset

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        """Return the loss function.
        
        Parameters
        ----------
        device : torch.device
            Target device for the loss function.
        
        Returns
        -------
        nn.Module
            Loss function instance (will be moved to device automatically).
        """
        # Examples:
        # - Classification: nn.CrossEntropyLoss()
        # - Regression: nn.MSELoss() or nn.L1Loss()
        # - Regression normalized: nn.KLDivLoss(reduction="batchmean")
        return nn.MSELoss()

    # ========== CONFIGURATION VALIDATION ==========

    def validate_config(self, task_cfg: DictConfig) -> None:
        """Validate task-specific configuration.
        
        Override to add custom validation. Remember to call super().validate_config()
        to validate required keys like 'pretrained_model_path' and 'head_learning_rate'.
        
        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        # Check parent requirements
        super().validate_config(task_cfg)
        
        # Check task-specific requirements
        required = [
            "my_data_path",  # Example: your task-specific data path
            # "other_required_param",
        ]
        missing = [key for key in required if getattr(task_cfg, key, None) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required config keys for {self.task_name}: {missing}. "
                f"Expected at {self.config_key}."
            )

    # ========== DATA LOADING ==========

    def load_data(
        self, task_cfg: DictConfig, embedder: Any
    ) -> tuple[ad.AnnData | None, ad.AnnData | None, np.ndarray, np.ndarray]:
        """Load and split data for the task.
        
        This method should:
        1. Load raw data from paths in task_cfg
        2. Preprocess as needed
        3. Split into train/test
        4. Return train/test data and targets
        
        Parameters
        ----------
        task_cfg : DictConfig
            Task configuration containing data paths and parameters.
        embedder : Any
            Frozen pretrained embedder (for reference if needed).
        
        Returns
        -------
        tuple
            (train_adata, test_adata, train_targets, test_targets)
            - train/test_adata: AnnData objects or None if data is not in AnnData format
            - train/test_targets: np.ndarray of shape [n_samples] or [n_samples, n_outputs]
        
        Examples
        --------
        >>> def load_data(self, task_cfg, embedder):
        ...     # Load data
        ...     data_path = Path(task_cfg.my_data_path)
        ...     adata = ad.read_h5ad(data_path)
        ...     
        ...     # Preprocess
        ...     adata = self._preprocess(adata)
        ...     
        ...     # Extract targets
        ...     targets = adata.obs["label"].to_numpy()
        ...     
        ...     # Train/test split
        ...     train_idx, test_idx = train_test_split(...)
        ...     
        ...     return (
        ...         adata[train_idx].copy(),
        ...         adata[test_idx].copy(),
        ...         targets[train_idx],
        ...         targets[test_idx],
        ...     )
        """
        raise NotImplementedError("Must implement load_data()")

    def _preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        """Preprocess AnnData object. Override as needed."""
        # Example: normalize, filter genes, etc.
        return adata

    # ========== DATASET PREPARATION ==========

    def prepare_datasets(
        self,
        train_adata: ad.AnnData | None,
        test_adata: ad.AnnData | None,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
        embedder: Any,
    ) -> tuple[Dataset, Dataset, int]:
        """Create datasets and compute embedding dimension.
        
        This method should:
        1. Generate embeddings for train/test data using the embedder
        2. Create dataset instances
        3. Return datasets and embedding dimension
        
        Parameters
        ----------
        train_adata, test_adata : ad.AnnData or None
            Training and test data from load_data().
        train_targets, test_targets : np.ndarray
            Training and test targets from load_data().
        embedder : Any
            Frozen pretrained embedder.
        
        Returns
        -------
        tuple
            (train_dataset, test_dataset, embedding_dim)
            - Datasets: instances of your dataset class
            - embedding_dim: dimension of the embeddings (int)
        
        Examples
        --------
        >>> def prepare_datasets(self, train_adata, test_adata, train_targets, test_targets, embedder):
        ...     # Generate embeddings
        ...     train_emb = self._embed_adata(embedder, train_adata)
        ...     test_emb = self._embed_adata(embedder, test_adata)
        ...     
        ...     # Create datasets
        ...     train_ds = self.get_dataset_class()(train_emb, train_targets)
        ...     test_ds = self.get_dataset_class()(test_emb, test_targets)
        ...     
        ...     embedding_dim = train_emb.shape[1]
        ...     return train_ds, test_ds, embedding_dim
        """
        raise NotImplementedError("Must implement prepare_datasets()")

    def _embed_adata(self, embedder: Any, adata: ad.AnnData, batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for AnnData.
        
        This commonly involves:
        1. Set embedder to eval mode
        2. Move to CUDA if available
        3. Call forward pass in batches
        4. Concatenate results
        
        Parameters
        ----------
        embedder : Any
            Frozen pretrained embedder.
        adata : ad.AnnData
            Data to embed.
        batch_size : int, optional
            Batch size for embedding generation.
        
        Returns
        -------
        np.ndarray
            Shape [n_samples, embedding_dim]
        
        Examples
        --------
        >>> def _embed_adata(self, embedder, adata, batch_size=64):
        ...     embedder.eval()
        ...     embedder.cuda()
        ...     
        ...     embeddings = []
        ...     for i in range(0, adata.n_obs, batch_size):
        ...         batch = adata[i:i+batch_size]
        ...         with torch.no_grad():
        ...             # Your embedding logic
        ...             emb = embedder.embed(batch)
        ...         embeddings.append(emb)
        ...     
        ...     return np.concatenate(embeddings, axis=0)
        """
        raise NotImplementedError("Must implement _embed_adata()")

    # ========== EVALUATION ==========

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Compute task-specific evaluation metrics.
        
        This method is called at the end of evaluation to compute metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions. Shape [n_samples] or [n_samples, n_outputs].
            For classification: logits (before softmax/argmax).
            For regression: continuous predictions.
        targets : np.ndarray
            Ground truth targets. Shape [n_samples] or [n_samples, n_outputs].
            For classification: class indices (0, 1, 2, ...).
            For regression: continuous values.
        
        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to values.
            Example: {"accuracy": 0.95, "f1": 0.93, "precision": 0.94, "recall": 0.92}
        
        Examples
        --------
        >>> def compute_metrics(self, predictions, targets):
        ...     # Classification
        ...     pred_labels = predictions.argmax(axis=-1)
        ...     acc = (pred_labels == targets).mean()
        ...     f1 = f1_score(targets, pred_labels, average="weighted")
        ...     
        ...     return {"accuracy": float(acc), "f1": float(f1)}
        ...
        >>> def compute_metrics(self, predictions, targets):
        ...     # Regression
        ...     mae = np.mean(np.abs(predictions - targets))
        ...     mse = np.mean((predictions - targets) ** 2)
        ...     
        ...     return {"mae": float(mae), "mse": float(mse)}
        """
        raise NotImplementedError("Must implement compute_metrics()")


# ============================================================================
# Step 4: Create Config YAML
# ============================================================================

"""
Create a YAML config file (my_task_config.yaml):

finetune:
  my_task:
    # === REQUIRED ===
    pretrained_model_path: /path/to/pretrained_model.ckpt
    my_data_path: /path/to/mydata.h5ad  # Task-specific
    head_learning_rate: 0.001
    
    # === OPTIONAL (defaults shown) ===
    # Data
    test_size: 0.2
    random_seed: 42
    batch_size: 32
    
    # Model
    hidden_dim: 128
    dropout: 0.0
    
    # Training
    epochs: 10
    min_lr: 1e-6
    first_cycle_steps: 15
    cycle_mult: 2.0
    warmup_steps: 5
    gamma: 0.9
    
    # Output
    save_dir: ./checkpoints/my_task
"""

# ============================================================================
# Step 5: Usage
# ============================================================================

"""
To use your new task:

# List available tasks
python run_downstream_task.py --list-tasks

# Run your task
python run_downstream_task.py --config my_task_config.yaml --task my_task

# Or programmatically
from downstream_task import TaskRegistry
from base_downstream_runner import BaseDownstreamRunner
from omegaconf import OmegaConf

# Import to register
from downstream_tasks_impl import MyTask

cfg = OmegaConf.load("my_task_config.yaml")
task = TaskRegistry.get_task("my_task")
runner = BaseDownstreamRunner(cfg, task)
results = runner.run()
"""
