from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import anndata as ad
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from finetune.canc_type_class.runner import GroupedCosineAnnealingWarmupRestarts
from utils import (
    SequentialDistributedSampler,
    distributed_concat,
    get_reduced,
    seed_all,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
TASK_NAME = "deconv"


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


class EmbeddingRegressor(nn.Module):
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


class DeconvEmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        targets = np.asarray(targets, dtype=np.float32)
        target_sums = targets.sum(axis=1, keepdims=True)
        if np.any(target_sums <= 0):
            raise ValueError("Every deconvolution target row must have a positive sum.")
        self.targets = targets / target_sums

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.from_numpy(self.embeddings[index]).float()
        target = torch.from_numpy(self.targets[index]).float()
        return emb, target


class DeconvEmbeddingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = self._resolve_task_cfg(cfg)

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.cell_types: list[str] = []
        self.target_columns: list[str] = []
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.test_dataset_size = 0
        self.model: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler = None
        self.pretrained_model_stem: str | None = None
        self.embedding_dim: int | None = None

        self.embedder = None

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "deconv" in cfg.finetune:
            return cfg.finetune.deconv
        raise ValueError("Could not find config at cfg.finetune.deconv.")

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

    def _load_input_adata(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "pseudo_bulk_data_path", None)
        if not configured_path:
            raise ValueError("finetune.deconv.pseudo_bulk_data_path must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"Pseudo-bulk h5ad file not found: {data_path}")
        return ad.read_h5ad(data_path)

    @staticmethod
    def _as_string(value) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.str_):
            return str(value)
        return None

    def _as_string_list(self, values) -> list[str] | None:
        if values is None:
            return None
        array = np.asarray(values).reshape(-1)
        strings: list[str] = []
        for value in array:
            value_string = self._as_string(value)
            if value_string is None:
                return None
            strings.append(value_string)
        return strings

    def _normalize_proportion_mapping(self, raw_mapping, obs_columns) -> dict[str, str]:
        obs_columns = set(map(str, obs_columns))
        normalized: dict[str, str] = {}

        def visit(cell_type_parts: list[str], value) -> None:
            value_string = self._as_string(value)
            if value_string is not None:
                normalized["/".join(cell_type_parts)] = value_string
                return

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    visit(cell_type_parts, value.item())
                    return
                if value.size == 1:
                    visit(cell_type_parts, value.reshape(-1)[0])
                    return

            if isinstance(value, (list, tuple)) and len(value) == 1:
                visit(cell_type_parts, value[0])
                return

            if isinstance(value, dict):
                direct_value = None
                for key in ("column", "obs_column", "value"):
                    if key in value:
                        direct_value = self._as_string(value[key])
                        if direct_value is not None:
                            break
                if direct_value is not None:
                    normalized["/".join(cell_type_parts)] = direct_value
                    return

                for key, child in value.items():
                    key_string = self._as_string(key)
                    if key_string is None:
                        key_string = str(key)
                    child_string = self._as_string(child)
                    if child_string is None and key_string in obs_columns:
                        normalized["/".join(cell_type_parts)] = key_string
                    else:
                        visit([*cell_type_parts, key_string], child)
                return

            raise ValueError(
                "Unsupported cell_type_proportion_columns entry for "
                f"{'/'.join(cell_type_parts)!r}: expected a column-name string, got {value!r}."
            )

        try:
            mapping = dict(raw_mapping)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "uns['cell_type_proportion_columns'] must be a mapping from cell type "
                "to proportion-column name."
            ) from exc

        for cell_type, value in mapping.items():
            cell_type_string = self._as_string(cell_type)
            if cell_type_string is None:
                cell_type_string = str(cell_type)
            visit([cell_type_string], value)

        invalid_columns = [col for col in normalized.values() if col not in obs_columns]
        if invalid_columns:
            inferred = self._infer_proportion_mapping_from_obs(obs_columns)
            if inferred:
                log.warning(
                    "Ignoring unusable uns['cell_type_proportion_columns'] entries and "
                    "inferring targets from prop__* obs columns instead. Invalid columns: %s",
                    invalid_columns[:10],
                )
                return inferred

        return normalized

    @staticmethod
    def _infer_proportion_mapping_from_obs(obs_columns) -> dict[str, str]:
        prop_columns = sorted(str(col) for col in obs_columns if str(col).startswith("prop__"))
        return {col.removeprefix("prop__"): col for col in prop_columns}

    def _load_targets(self, adata: ad.AnnData) -> np.ndarray:
        cell_types = self._as_string_list(adata.uns.get("cell_type_proportion_cell_types"))
        obs_columns = self._as_string_list(adata.uns.get("cell_type_proportion_obs_columns"))
        if cell_types is not None and obs_columns is not None:
            if len(cell_types) != len(obs_columns):
                raise ValueError(
                    "uns['cell_type_proportion_cell_types'] and "
                    "uns['cell_type_proportion_obs_columns'] must have the same length."
                )
            mapping = dict(zip(cell_types, obs_columns))
        else:
            mapping = adata.uns.get("cell_type_proportion_columns")

        if mapping is None:
            mapping = self._infer_proportion_mapping_from_obs(adata.obs.columns)
            if not mapping:
                raise ValueError(
                    "Pseudo-bulk AnnData must contain uns['cell_type_proportion_columns'] "
                    "or obs columns prefixed with 'prop__'."
                )
        else:
            mapping = self._normalize_proportion_mapping(mapping, adata.obs.columns)

        self.cell_types = sorted(mapping)
        self.target_columns = [mapping[cell_type] for cell_type in self.cell_types]
        missing = [col for col in self.target_columns if col not in adata.obs]
        if missing:
            raise ValueError(f"Missing proportion columns in adata.obs: {missing}")

        targets = adata.obs[self.target_columns].to_numpy(dtype=np.float32)
        if np.any(targets < 0):
            raise ValueError("Cell type proportions must be non-negative.")
        return targets

    def _prepare_train_test_data(self) -> tuple[ad.AnnData, ad.AnnData, np.ndarray, np.ndarray]:
        adata = self._load_input_adata()
        adata.var_names_make_unique()
        targets = self._load_targets(adata)

        test_size = float(getattr(self.task_cfg, "test_size", 0.2))
        split_version = getattr(
            self.task_cfg,
            "train_test_split_version",
            getattr(self.task_cfg, "random_seed", 42),
        )
        split_seed = self._hash_split_version(split_version)

        if bool(getattr(self.task_cfg, "split_by_context", True)):
            context_columns = list(
                getattr(self.task_cfg, "context_columns", ["dataset_id", "donor_id", "tissue_general"])
            )
            missing_context = [col for col in context_columns if col not in adata.obs]
            if missing_context:
                raise ValueError(
                    f"split_by_context=true but these context columns are missing: {missing_context}"
                )
            context = adata.obs[context_columns].astype(str).agg("||".join, axis=1).to_numpy()
            unique_contexts = np.unique(context)
            train_contexts, _ = train_test_split(
                unique_contexts,
                test_size=test_size,
                random_state=split_seed,
            )
            train_contexts = set(train_contexts.tolist())
            train_idx = np.flatnonzero(np.isin(context, list(train_contexts)))
            test_idx = np.flatnonzero(~np.isin(context, list(train_contexts)))
        else:
            train_idx, test_idx = train_test_split(
                np.arange(adata.n_obs),
                test_size=test_size,
                random_state=split_seed,
            )

        return (
            adata[train_idx].copy(),
            adata[test_idx].copy(),
            targets[train_idx],
            targets[test_idx],
        )

    def _load_embedder(self):
        checkpoint_path = getattr(self.task_cfg, "pretrained_model_path", None)
        if not checkpoint_path:
            raise ValueError("finetune.deconv.pretrained_model_path must be set.")
        resolved_path = hydra.utils.to_absolute_path(str(checkpoint_path))
        self.pretrained_model_stem = Path(resolved_path).stem

        # TODO: Replace this with your actual checkpoint loader.
        # It should return an object usable by `self._embed_adata`.
        embedder = load_embedding_model(resolved_path, self.device)
        log.info("Loaded pretrained embedding model from %s", resolved_path)
        return embedder

    def _embed_adata(self, adata: ad.AnnData) -> np.ndarray:
        # TODO: Replace this with your actual embedding call.
        # Expected output shape: [n_samples, embedding_dim]
        return embed_adata(self.embedder, adata)

    def _build_loaders(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
        train_targets: np.ndarray,
        test_targets: np.ndarray,
    ) -> None:
        train_embeddings = np.asarray(self._embed_adata(train_adata), dtype=np.float32)
        test_embeddings = np.asarray(self._embed_adata(test_adata), dtype=np.float32)

        if train_embeddings.ndim != 2 or test_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays: [n_samples, embedding_dim].")
        if train_embeddings.shape[1] != test_embeddings.shape[1]:
            raise ValueError("Train/test embedding dimensions do not match.")

        self.embedding_dim = int(train_embeddings.shape[1])

        batch_size = int(getattr(self.task_cfg, "batch_size", 32))
        train_dataset = DeconvEmbeddingDataset(train_embeddings, train_targets)
        test_dataset = DeconvEmbeddingDataset(test_embeddings, test_targets)
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

        model = EmbeddingRegressor(
            embedding_dim=self.embedding_dim,
            output_dim=len(self.cell_types),
            hidden_dim=int(getattr(self.task_cfg, "head_hidden_dim", 128)),
            dropout=float(getattr(self.task_cfg, "head_dropout", 0.0)),
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
            raise ValueError("finetune.deconv.head_learning_rate must be set.")
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

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_name = str(getattr(self.task_cfg, "loss", "kl")).lower()
        if loss_name == "kl":
            return F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="batchmean")
        pred_props = F.softmax(logits, dim=-1)
        if loss_name == "mse":
            return F.mse_loss(pred_props, targets)
        if loss_name in {"mae", "l1"}:
            return F.l1_loss(pred_props, targets)
        raise ValueError("Unsupported deconvolution loss. Expected one of: kl, mse, mae.")

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            dist.barrier()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        grad_acc_steps = max(1, int(getattr(self.task_cfg, "grad_accumulation_steps", 1)))
        max_grad_norm = float(getattr(self.task_cfg, "max_grad_norm", 1e6))
        running_loss = 0.0

        for step_idx, (data, targets) in enumerate(self.train_loader, start=1):
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(data)
            loss = self._compute_loss(logits, targets)
            (loss / grad_acc_steps).backward()

            if step_idx % grad_acc_steps == 0 or step_idx == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        if self.is_distributed:
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)

        self.scheduler.step()
        return {"loss": epoch_loss}

    @staticmethod
    def _mean_pearson(predictions: np.ndarray, truths: np.ndarray) -> float:
        corrs = []
        for idx in range(truths.shape[1]):
            pred = predictions[:, idx]
            truth = truths[:, idx]
            if np.std(pred) == 0 or np.std(truth) == 0:
                continue
            corrs.append(float(np.corrcoef(pred, truth)[0, 1]))
        return float(np.mean(corrs)) if corrs else float("nan")

    def _evaluate(self) -> dict[str, Any]:
        self.model.eval()
        running_loss = 0.0
        predictions = []
        truths = []

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(data)
                loss = self._compute_loss(logits, targets)
                running_loss += loss.item()
                predictions.append(F.softmax(logits, dim=-1))
                truths.append(targets)

        predictions = torch.cat(predictions, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        truths_np = truths.cpu().numpy()
        test_loss = running_loss / len(self.test_loader)
        if self.is_distributed:
            test_loss = get_reduced(test_loss, self.local_rank, 0, self.world_size)

        per_type_mae = np.mean(np.abs(predictions_np - truths_np), axis=0)
        per_type_rmse = np.sqrt(np.mean((predictions_np - truths_np) ** 2, axis=0))
        return {
            "loss": float(test_loss),
            "mae": float(mean_absolute_error(truths_np, predictions_np)),
            "rmse": float(np.sqrt(mean_squared_error(truths_np, predictions_np))),
            "mean_pearson": self._mean_pearson(predictions_np, truths_np),
            "per_cell_type_mae": {
                cell_type: float(value)
                for cell_type, value in zip(self.cell_types, per_type_mae.tolist())
            },
            "per_cell_type_rmse": {
                cell_type: float(value)
                for cell_type, value in zip(self.cell_types, per_type_rmse.tolist())
            },
            "cell_types": self.cell_types,
            "target_columns": self.target_columns,
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
        model_name = f"{self.pretrained_model_stem}_{TASK_NAME}_embedding_head_only_v{split_version}"
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
                "cell_types": self.cell_types,
                "target_columns": self.target_columns,
                "embedding_dim": self.embedding_dim,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def run(self) -> dict[str, Any]:
        try:
            self._setup_runtime()
            self.embedder = self._load_embedder()

            train_adata, test_adata, train_targets, test_targets = self._prepare_train_test_data()
            if self.is_master:
                log.info(
                    "Prepared deconvolution data: train=%d, test=%d, genes=%d, cell_types=%d",
                    train_adata.n_obs,
                    test_adata.n_obs,
                    train_adata.n_vars,
                    len(self.cell_types),
                )

            self._build_loaders(train_adata, test_adata, train_targets, test_targets)
            self._build_model()
            self._build_optimization()

            epochs = int(getattr(self.task_cfg, "epochs", 10))
            last_train_metrics = {"loss": float("nan")}

            for epoch in range(1, epochs + 1):
                last_train_metrics = self._train_one_epoch(epoch)
                if self.is_master:
                    log.info("Epoch %d | Training Loss: %.6f", epoch, last_train_metrics["loss"])

            test_metrics = self._evaluate()
            checkpoint_path = self._save_checkpoint(
                epoch=epochs,
                train_metrics={"loss": float(last_train_metrics["loss"])} ,
                test_metrics=test_metrics,
            )
            if checkpoint_path is not None:
                test_metrics["checkpoint_path"] = str(checkpoint_path)

            return {
                "train_loss": float(last_train_metrics["loss"]),
                **test_metrics,
            }
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()


def load_runner_config(config_path: str | Path) -> DictConfig:
    """
    Load a YAML config file into an OmegaConf DictConfig and validate that it
    contains the structure expected by DeconvEmbeddingRunner.

    Expected shape:
        finetune:
          deconv:
            pretrained_model_path: ...
            pseudo_bulk_data_path: ...
            head_learning_rate: ...

    Returns
    -------
    DictConfig
        The loaded config object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    TypeError
        If the loaded object is not a DictConfig.
    ValueError
        If required sections or required keys are missing.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = hydra.compose  # type: ignore[attr-defined]  # keep linters quiet if hydra is partially stubbed
    del cfg  # avoid accidental shadowing; actual load is via OmegaConf below

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)!r}")

    if "finetune" not in cfg or cfg.finetune is None:
        raise ValueError("Config must contain a top-level `finetune` section.")

    if "deconv" not in cfg.finetune or cfg.finetune.deconv is None:
        raise ValueError("Config must contain `finetune.deconv`.")

    task_cfg = cfg.finetune.deconv
    required_keys = [
        "pretrained_model_path",
        "pseudo_bulk_data_path",
        "head_learning_rate",
    ]
    missing = [key for key in required_keys if getattr(task_cfg, key, None) in (None, "")]
    if missing:
        raise ValueError(
            f"Missing required keys in `finetune.deconv`: {missing}"
        )

    return cfg


def load_embedding_model(
    checkpoint_path: str,
    # device: torch.device,
    ):
    """
    Load your frozen embedding model from checkpoint.
    Return any object needed by `embed_adata`.
    """
    # Load model
    model = CancerFoundation.load_from_checkpoint(
    checkpoint_path,  
    strict=True
    )
    return model


def embed_adata(model: Any, adata: ad.AnnData, batch_size=64) -> np.ndarray:
    """
    Return a numpy array of shape [n_cells, embedding_dim].
    """
    model.eval().cuda()
    df_emb = model.embed(adata, batch_size=batch_size, flavor="seurat_v3")
    emb = df.to_numpy()
    return emb

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