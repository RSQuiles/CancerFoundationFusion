from pathlib import Path
import pytorch_lightning as pl
import os
from typing import Any, List, Optional, Union
import transformers
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from cancerfoundation.model.perturbation_model import PerturbationTransformer
from cancerfoundation.utils import load_pretrained
from cancerfoundation.model.module import TransformerModule
from cancerfoundation.data.preprocess import binning
from cancerfoundation.loss import get_loss
from safetensors import safe_open
from cancerfoundation.loss import LossType
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import torch.nn.functional as F
from tqdm import tqdm


class CancerFoundation(pl.LightningModule):
    """The main PyTorch Lightning module for the Cancer Foundation model.

    This class encapsulates the entire model training, validation, and optimization pipeline.
    It wraps the `TransformerModule` and handles hyperparameter configuration, loss calculation,
    optimizer and scheduler setup, and the training/validation loops required by PyTorch Lightning.
    """

    def __init__(
        self,
        n_bins: int,
        input_emb_style: str,
        max_seq_len: int,
        input_style: str,
        mask_ratio: float,
        TRUNC_BY_SAMPLE: bool,
        training_tasks: str,
        embsize: int,
        nheads: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        lr: float,
        epochs: int,
        vocab,
        warmup_ratio_or_step: float,
        scheduler_interval: int,
        scheduler_factor: float,
        compile_model: bool,
        data_path: Union[str, os.PathLike],
        loss_type: LossType,
        conditions: Optional[List[str]],
        conditions_nums: Optional[Any],
        mvc_decoder_style: str,
        scale_zero_expression: Optional[float],
        do_dat: bool,
        explicit_zero_prob: Optional[bool],
        balance_primary: Optional[str],
        balance_secondary: Optional[str],
        zero_percentages: Optional[List[float]],
        do_mvc: bool,
        no_invert_dat: bool,
        activation: str,
        norm_first: bool,
        cell_emb_style: str,
        batchnorm: bool,
        dat_scale: float,
        normalise_bins: bool,
        where_condition: str,
        gen_method: str,
        their_init_weights: bool,
        perturbation: bool = False,
        n_top_genes: int = 1200,
        # Unified FM parameters
        contrastive: bool = False,
        aggregation: bool = False,
        agg_fn: Optional[str] = None,
        noise: Optional[List[int]] = None,
        esm_emb: bool = False,
        esm_emb_path: Optional[Union[str, os.PathLike]] = None,
        esm_emb_finetune: bool = False,
        dat_columns: Optional[List[str]] = [],
    ):
        """Initializes the CancerFoundation LightningModule.

        Args:
            n_bins (int): The number of bins for discretizing expression values.
            input_emb_style (str): The style of input embedding ('category' or 'continuous').
            max_seq_len (int): The maximum sequence length.
            input_style (str): Style of input data processing.
            mask_ratio (float): The ratio of tokens to mask for the MLM task.
            TRUNC_BY_SAMPLE (bool): Whether to truncate sequences by sample.
            training_tasks (str): The training tasks to perform ('mlm', 'gen', 'both').
            embsize (int): The embedding size (d_model).
            nheads (int): The number of attention heads.
            d_hid (int): The dimension of the feed-forward hidden layer.
            nlayers (int): The number of transformer layers.
            dropout (float): The dropout rate.
            lr (float): The learning rate.
            epochs (int): The total number of training epochs.
            vocab: The vocabulary mapping gene names to token IDs.
            warmup_ratio_or_step (float): The ratio or number of steps for learning rate warmup.
            scheduler_interval (int): The interval for the StepLR scheduler.
            scheduler_factor (float): The factor for the StepLR scheduler.
            compile_model (bool): If True, compile the model using `torch.compile`.
            data_path (Union[str, os.PathLike]): The path to the data.
            loss_type (LossType, optional): The type of loss function to use. Defaults to LossType.MSE.
            conditions (Optional[List[str]], optional): A list of conditional variables. Defaults to None.
            conditions_nums (Optional[Any], optional): A dictionary mapping condition names to their number of categories. Defaults to None.
            mvc_decoder_style (str, optional): The architecture style for the MVC decoder. Defaults to "inner product".
            scale_zero_expression (Optional[float], optional): A factor to scale the loss for zero-expression values. Defaults to None.
            do_dat (bool, optional): If True, enable Domain Adversarial Training. Defaults to False.
            explicit_zero_prob (Optional[bool], optional): If True, explicitly model zero probability. Defaults to False.
            balance_primary (Optional[str], optional): The primary variable for balanced sampling. Defaults to None.
            balance_secondary (Optional[str], optional): The secondary variable for balanced sampling. Defaults to None.
            zero_percentages (Optional[List[float]], optional): Percentages for balancing zero expression. Defaults to None.
            contrastive (bool, optional): If True, enable contrastive learning. It brings the pseudobulk and real bulk samples closer together in the embedding space. Defaults to False.
            aggregation (bool, optional): If True, enable aggregation consistency losses. Defaults to False.
            agg_fn (Optional[str], optional): The function to use for aggregating single-cell embeddings into pseudobulk embeddings. Defaults to "mean".
            noise (Optional[List[int]], optional): If present, enable denoising task. For it, binning must have been disabled
            esm_emb (bool, optional): If True, load pretrained ESM gene embeddings instead of a learned lookup table.
            esm_emb_path (Optional[str | os.PathLike], optional): Path to the parquet file containing pretrained gene embeddings.
            esm_emb_finetune (bool, optional): If True, allow the pretrained gene embeddings to be fine-tuned.
        """
        # Checks
        if input_style == "binned":
            assert n_bins is not None, "When performing binning, n_bins must be provided"

        super().__init__()
        self.save_hyperparameters()
        self.vocab = vocab
        # Store all parameters
        self.n_bins = n_bins
        self.input_emb_style = input_emb_style
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = (
            [0.25, 0.50, 0.75] if training_tasks in ["gen", "both"] else mask_ratio
        )
        self.TRUNC_BY_SAMPLE = TRUNC_BY_SAMPLE
        self.training_tasks = training_tasks
        self.embsize = embsize
        self.nheads = nheads
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.lr = lr
        self.warmup_ratio_or_step = warmup_ratio_or_step
        self.scheduler_interval = scheduler_interval
        self.scheduler_factor = scheduler_factor
        self.loss_type = loss_type
        self.data_path = data_path
        self.epochs = epochs
        self.compile_model = compile_model
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm_first = norm_first
        self.batchnorm = batchnorm
        self.cell_emb_style = cell_emb_style
        self.perturbation = perturbation
        self.their_init_weights = their_init_weights
        self.n_top_genes = n_top_genes

        # Unified FM parameters
        self.contrastive = contrastive
        self.aggregation = aggregation
        self.agg_fn = agg_fn
        self.noise = noise or []
        self.denoise = len(self.noise) > 0
        self.esm_emb = esm_emb
        self.esm_emb_path = esm_emb_path
        self.esm_emb_finetune = esm_emb_finetune

        # Training configuration
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        self.do_mvc = do_mvc
        self.USE_GENERATIVE_TRAINING = (
            True if self.training_tasks in ["gen", "both"] else False
        )
        self.use_cell_embedding = False
        self.domain_nums = None
        self.explicit_zero_prob = explicit_zero_prob
        self.do_dat = do_dat
        self.dat_columns = dat_columns
        self.no_invert_dat = no_invert_dat
        self.conditions = conditions
        self.conditions_nums = conditions_nums
        self.where_condition = where_condition
        self.gen_method = gen_method

        self.normalise_bins = normalise_bins
        self.dat_scale = dat_scale

        # Balance sampling parameters
        if balance_primary is None and balance_secondary is not None:
            raise ValueError(
                "balance_secondary is not allowed to be set (not None) if balance_primary is None."
            )
        self.balance_primary = balance_primary
        self.balance_secondary = balance_secondary
        self.zero_percentages = zero_percentages
        self.scale_zero_expression = scale_zero_expression

        # Setup token values based on embedding style
        if self.input_emb_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins

        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]

        if self.esm_emb and self.esm_emb_path is None:
            raise ValueError(
                "esm_emb=True requires esm_emb_path to point to a parquet file with pretrained gene embeddings."
            )

        # Initialize dataset and model
        self._setup_model(mvc_decoder_style)

    def _setup_model(self, mvc_decoder_style: str):
        """Initializes the model and its loss function."""
        self.criterion = get_loss(
            loss_type=self.loss_type,
            num_classes=self.n_input_bins if self.n_input_bins else None,
            scale_zero_expression=self.scale_zero_expression,
        )

        if self.perturbation:
            self.model = PerturbationTransformer(
                ntoken=len(self.vocab.keys()),
                d_model=self.embsize,
                out_dim=self.criterion.get_in_dim(),
                mvc_decoder_style=mvc_decoder_style,
                nhead=self.nheads,
                d_hid=self.d_hid,
                nlayers=self.nlayers,
                dropout=self.dropout,
                pad_token_id=self.pad_token_id,
                criterion=self.criterion,
                pad_value=self.pad_value,
                do_mvc=self.do_mvc,
                conditions=self.conditions_nums,
                input_emb_style=self.input_emb_style,
                n_input_bins=self.n_input_bins,
                use_generative_training=self.USE_GENERATIVE_TRAINING,
                do_dat=self.do_dat,
                no_invert_dat=self.no_invert_dat,
                explicit_zero_prob=self.explicit_zero_prob,
                activation=self.activation,
                norm_first=self.norm_first,
                batchnorm=self.batchnorm,
                cell_emb_style=self.cell_emb_style,
                dat_scale=self.dat_scale,
                normalise_bins=self.normalise_bins,
                where_condition=self.where_condition,
                max_seq_len=self.max_seq_len,
                gen_method=self.gen_method,
                pert_pad_id=2,
                their_init_weights=self.their_init_weights,
                vocab=self.vocab,
                gene_embeddings_path=self.esm_emb_path if self.esm_emb else None,
                gene_embeddings_freeze=not self.esm_emb_finetune,
            )

        else:
            self.model = TransformerModule(
                ntoken=len(self.vocab.keys()),
                d_model=self.embsize,
                out_dim=self.criterion.get_in_dim(),
                mvc_decoder_style=mvc_decoder_style,
                nhead=self.nheads,
                d_hid=self.d_hid,
                nlayers=self.nlayers,
                dropout=self.dropout,
                pad_token_id=self.pad_token_id,
                criterion=self.criterion,
                pad_value=self.pad_value,
                do_mvc=self.do_mvc,
                conditions=self.conditions_nums,
                input_emb_style=self.input_emb_style,
                n_input_bins=self.n_input_bins,
                use_generative_training=self.USE_GENERATIVE_TRAINING,
                do_dat=self.do_dat,
                dat_columns=self.dat_columns,
                no_invert_dat=self.no_invert_dat,
                explicit_zero_prob=self.explicit_zero_prob,
                activation=self.activation,
                norm_first=self.norm_first,
                batchnorm=self.batchnorm,
                cell_emb_style=self.cell_emb_style,
                dat_scale=self.dat_scale,
                normalise_bins=self.normalise_bins,
                where_condition=self.where_condition,
                max_seq_len=self.max_seq_len,
                gen_method=self.gen_method,
                their_init_weights=self.their_init_weights,
                # Unified FM parameters
                contrastive=self.contrastive,
                aggregation=self.aggregation,
                agg_fn=self.agg_fn,
                vocab=self.vocab,
                gene_embeddings_path=self.esm_emb_path if self.esm_emb else None,
                gene_embeddings_freeze=not self.esm_emb_finetune,
            )
        if self.compile_model:
            self.model = torch.compile(self.model)

    def forward(self, data_dict, use_cell_embedding=None):
        """Performs a forward pass through the underlying `TransformerModule`.

        Args:
            data_dict (dict): A dictionary of input tensors.
            use_cell_embedding (Optional[bool], optional): A flag to control a specific training behavior.
                If None, uses the module's default. Defaults to None.

        Returns:
            dict: The output dictionary from the model, typically containing losses.
        """
        if use_cell_embedding is None:
            use_cell_embedding = self.use_cell_embedding

        # First pass without noising
        loss_dict = self.model(data_dict, use_cell_embedding=use_cell_embedding)

        # DENOISING TASK
        if self.denoise:
            loss_dict["loss_noise"] = 0
            for i in self.noise:
                # print(f"Running with noise level {i}")
                loss_noise = self.model(data_dict, use_cell_embedding=use_cell_embedding, noise=i)
                loss_dict["loss_noise"] += loss_noise
                loss_dict["total_loss"] += loss_noise
        
        return loss_dict

    def training_step(self, batch, batch_idx):  # batch = data_dict from collator
        """Performs a single training step.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        # Update use_cell_embedding based on global step
        self.use_cell_embedding = (
            self.USE_GENERATIVE_TRAINING and self.global_step > 1000
        )

        loss_dict = self.forward(batch, use_cell_embedding=self.use_cell_embedding)

        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)

        # Print loss dict
        # print(loss_dict)

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            dict: The dictionary of losses for the validation batch.
        """

        if batch_idx == 0:
            print(
                f"Rank {self.trainer.global_rank}: Starting validation with {len(self.trainer.val_dataloaders)} batches"
            )
        loss_dict = self.forward(batch, use_cell_embedding=True)

        # Log validation metrics
        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss_dict

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer and learning rate scheduler.

        Uses an Adam optimizer. If warmup is specified, it uses a cosine learning rate schedule with warmup.
            Otherwise, it uses a step-based decay scheduler.

        Returns:
            dict: The optimizer and scheduler configuration for PyTorch Lightning.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            # Calculate total training steps
            total_num_batches = self.trainer.estimated_stepping_batches
            warmup_steps = (
                int(total_num_batches * self.warmup_ratio_or_step)
                if self.warmup_ratio_or_step < 1
                else int(self.warmup_ratio_or_step)
            )

            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_num_batches,
                last_epoch=-1,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.scheduler_interval, gamma=self.scheduler_factor
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

    def load_pretrained_weights(
        self,
        pretrained_model_path: Path,
        gene_mapping: Optional[dict],
        verbose: bool = True,
    ):
        """Loads pretrained weights from a checkpoint file into the current model.

        This method supports both `.safetensors` and PyTorch `.pth`/`.pt` formats.
        It can handle vocab mismatches by re-mapping gene embeddings if a
        `gene_mapping` dictionary is provided.

        Args:
            pretrained_model_path (Path): Path to the pretrained model file.
            gene_mapping (Optional[dict]): A dictionary to map gene names from the
                pretrained vocab to the current vocab.
            verbose (bool, optional): If True, prints information about matched
                and unmatched weights. Defaults to True.
        """
        if pretrained_model_path.name.endswith(".safetensors"):
            tensors = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
        elif pretrained_model_path.name.endswith(
            ".pth"
        ) or pretrained_model_path.name.endswith(".pt"):
            tensors = torch.load(pretrained_model_path, map_location="cpu")
        else:
            raise ValueError("Unsupported file format. Use .safetensors, .pth, or .pt")

        return load_pretrained(self.model, tensors, gene_mapping, verbose=verbose)


    def _maybe_fix_negative(self, data):
        """Shift z-scored or centered data to non-negative range."""
        X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
        min_val = np.nanmin(X)
        if min_val < 0:
            print(
                f"Data contains negative values (min={min_val:.3f}), "
                "likely z-scored. Shifting to non-negative range."
            )
            data.X = X - min_val
        return data

    @torch.no_grad()
    def embed(self, adata, batch_size: int = 64, normalized=False, log1p_only=False, flavor="seurat"):
        """Embeds an AnnData object into cell embeddings.

        Handles all preprocessing: gene intersection with vocab, HVG selection,
        binning, and batched forward passes through the transformer.

        Args:
            adata: AnnData object (h5ad). X should contain expression values
                (dense or sparse).
            batch_size: Batch size for inference.

        Returns:
            pd.DataFrame with cell IDs as index and columns dim_0, dim_1, ...
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Work on a copy to avoid mutating the input
        data = adata.copy()
        if hasattr(data.X, "toarray"):
            data.X = data.X.toarray()

        data = self._maybe_fix_negative(data)

        # Normalize to CP10K + log1p if not already normalized
        if not normalized:
            print("Normalizing before embedding!")
            if not log1p_only:
                # Raw counts
                sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.log1p(data)

        # Intersect genes with vocab
        common_genes = list(set(self.vocab.keys()).intersection(set(data.var.index)))
        if len(common_genes) == 0:
            raise ValueError(f"No common genes between vocab and data. Check gene name format.")
        print(f"Common genes: {len(common_genes)} / {data.n_vars}")
        data = data[:, common_genes].copy()

        # Select highly variable genes (RAFA removed, HVG libraries gave many problems)
        # sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, layer=None, flavor=flavor)
        # data = data[:, data.var["highly_variable"]].copy()

        # RAFA: adapt HVG selection wihout scanpy to avoid library issues
        if data.n_vars > self.n_top_genes:
            X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
            
            mean = X.mean(axis=0)
            var = X.var(axis=0)
            
            # Clip means to avoid log(0), same as scanpy
            mean[mean == 0] = 1e-12
            dispersion = var / mean
            
            # Log-transform dispersion, same as scanpy seurat flavor
            dispersion[dispersion == 0] = np.nan
            dispersion = np.log(dispersion)
            
            # Bin genes by mean expression and normalize dispersion within bins
            n_bins = 20  # scanpy default
            mean_log = np.log1p(mean)
            bins = np.percentile(mean_log, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)  # remove duplicate bin edges (common in sparse data)
            bin_indices = np.digitize(mean_log, bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
            
            disp_norm = np.zeros_like(dispersion)
            for b in range(len(bins) - 1):
                mask = bin_indices == b
                if mask.sum() < 2:
                    # Not enough genes in bin to normalize, keep raw dispersion
                    disp_norm[mask] = dispersion[mask]
                    continue
                bin_disp = dispersion[mask]
                # Ignore NaNs when computing bin stats, same as scanpy
                bin_mean = np.nanmean(bin_disp)
                bin_std = np.nanstd(bin_disp)
                if bin_std == 0:
                    disp_norm[mask] = 0.0
                else:
                    disp_norm[mask] = (bin_disp - bin_mean) / bin_std
            
            # NaN genes (zero dispersion) get lowest score
            disp_norm = np.nan_to_num(disp_norm, nan=-np.inf)
            
            top_idx = np.argsort(disp_norm)[-self.n_top_genes:]
            data = data[:, top_idx].copy()

        # Bin expression values if required
        if self.input_style == "binned":
            normalise = self.model.decoder.normalise_bins
            for idx in range(data.n_obs):
                data.X[idx] = binning(data.X[idx], self.n_bins)
                if normalise:
                    data.X[idx] = data.X[idx] / self.n_bins

        # Build gene ID tensor
        gene_ids = torch.LongTensor([self.vocab[g] for g in data.var.index])
        count_matrix = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()

        # Embed in batches
        n_batches = (len(data) + batch_size - 1) // batch_size
        embeddings = []
        for i in tqdm(
            range(0, len(data), batch_size), total=n_batches, desc="Embedding cells"
        ):
            batch_expr = torch.FloatTensor(count_matrix[i : i + batch_size]).to(device)
            batch_genes = (
                gene_ids.unsqueeze(0).expand(batch_expr.shape[0], -1).to(device)
            )

            # Prepend CLS token
            batch_genes = torch.cat(
                [
                    torch.full(
                        (batch_expr.shape[0], 1),
                        self.cls_token_id,
                        dtype=torch.long,
                        device=device,
                    ),
                    batch_genes,
                ],
                dim=1,
            )
            batch_expr = torch.cat(
                [
                    torch.full(
                        (batch_expr.shape[0], 1),
                        self.pad_value,
                        dtype=batch_expr.dtype,
                        device=device,
                    ),
                    batch_expr,
                ],
                dim=1,
            )

            padding_mask = torch.zeros(
                batch_genes.shape, dtype=torch.bool, device=device
            )

            if self.model.use_generative_training:
                # Returns (pcpt_output, gen_output) tuple
                output = self.model.embed(
                    src=batch_genes,
                    values=batch_expr,
                    src_key_padding_mask=padding_mask,
                )
                transformer_output = output[0]
            else:
                # Returns a single tensor
                transformer_output = self.model.encode(
                    src=batch_genes,
                    values=batch_expr,
                    src_key_padding_mask=padding_mask,
                    check_conditions=False
                )

            cell_emb = transformer_output[:, 0, :]  # CLS token
            embeddings.append(cell_emb.cpu())

        emb = torch.cat(embeddings, dim=0).numpy()
        return pd.DataFrame(
            emb,
            index=adata.obs_names,
            columns=[f"dim_{i}" for i in range(emb.shape[1])],
        )
