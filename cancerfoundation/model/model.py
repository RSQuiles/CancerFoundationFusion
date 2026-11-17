from pathlib import Path
import pytorch_lightning as pl
import os
from typing import Any, List, Optional, Union
import transformers
import torch
import numpy as np
import pandas as pd
from cancerfoundation.model.perturbation_model import PerturbationTransformer
from cancerfoundation.utils import load_pretrained
from cancerfoundation.model.module import TransformerModule
from cancerfoundation.data.preprocess import binning
# Modality-aware gene-panel selection lives in cancerfoundation/data/gene_panel.py,
# kept free of torch/lightning imports so it can be exercised without a checkpoint.
from cancerfoundation.data.gene_panel import (  # noqa: F401  (re-exported)
    LIBRARY_SIZE_COL,
    _classify_modality,
    _top_mad_genes,
    gene_ranking,
    select_panel,
    select_shared_panel,
    shared_gene_ranking,
)
from cancerfoundation.data.preprocess import (
    NORMALIZE_METHODS,
    cell_seed,
    library_size,
    normalize_expression,
    train_matched_input,
    value_scale,
)
from cancerfoundation.loss import get_loss

# How CancerFoundation.embed builds the model input (see its docstring).
#   legacy_panel:  one fixed panel of n_top_genes for every cell, binned within the panel,
#                  every zero kept (the behaviour before input modes existed).
#   train_matched: as training feeds the model: each row binned over all its vocabulary
#                  genes first, then its top-ranked zero genes (zero_fraction of the
#                  sequence budget) and top-ranked non-zero genes, up to max_seq_len.
#   nonzero:       train_matched without zeros (scGPT-style).
EMBED_INPUT_MODES = ("legacy_panel", "train_matched", "nonzero")
from safetensors import safe_open
from cancerfoundation.loss import LossType
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
import torch.nn.functional as F
from tqdm import tqdm
import time

class CancerFoundation(pl.LightningModule):
    """The main PyTorch Lightning module for the Cancer Foundation model.

    This class encapsulates the entire model training, validation, and optimization pipeline.
    It wraps the `TransformerModule` and handles hyperparameter configuration, loss calculation,
    optimizer and scheduler setup, and the training/validation loops required by PyTorch Lightning.
    """

    @classmethod
    def load_for_inference(cls, path, map_location: str = "cpu", **overrides):
        """Load a checkpoint for evaluation or embedding, tolerating code drift.

        Prefer this over ``load_from_checkpoint`` anywhere the model is only used
        to produce embeddings or reconstructions: it repairs hyperparameters that
        predate current constructor arguments, strips the ``torch.compile`` key
        prefix, and drops the training-only DAT discriminators. See
        ``cancerfoundation/checkpoint.py``. Not suitable for resuming training.
        """
        from cancerfoundation.checkpoint import load_for_inference

        return load_for_inference(path, map_location=map_location, **overrides)

    # NOTE: new arguments below must be added *with a default*, chosen to
    # reproduce the previous behaviour. `save_hyperparameters()` stores this
    # signature into every checkpoint, and Lightning replays it on load, so a new
    # required argument breaks every checkpoint saved before it (see
    # cancerfoundation/checkpoint.py).
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
        norm_scheme: str,
        norm_type: str,
        cell_emb_style: str,
        batchnorm: bool,
        dat_scale: float,
        normalise_bins: bool,
        where_condition: str,
        gen_method: str,
        their_init_weights: bool,
        grad_checkpoint: bool = False,
        attn_impl: str = "mha",
        dat_start_step: int = 0,
        dat_interval_steps: int = 1,
        perturbation: bool = False,
        n_top_genes: int = 1200,
        # Unified FM parameters
        contrastive: bool = False,
        contrastive_temperature: float = 0.1,
        mmd: bool = False,
        aggregation: bool = False,
        agg_fn: Optional[str] = None,
        noise: Optional[List[int]] = None,
        esm_emb: bool = False,
        esm_emb_path: Optional[Union[str, os.PathLike]] = None,
        esm_emb_finetune: bool = False,
        dat_columns: Optional[List[str]] = [],
        encoded_conditions: Optional[List[str]] = None,
        paired_alignment: bool = False,
        verbose: bool = False,
        weight_mvc: float = 1.0,
        weight_contrastive: float = 1.0,
        weight_mmd: float = 1.0,
        weight_paired: float = 1.0,
        weight_agg: float = 1.0,
        weight_dat: float = 1.0,
        weight_reconstruction: float = 1.0,
        monitor_losses: Optional[List[str]] = None,
        n_sc_per_pseudobulk: int = 10,
        # Contrastive Domain Discrepancy (CAN) parameters
        cdd: bool = False,
        weight_cdd: float = 0.3,
        # Class labels and class-aware sampling, shared by CDD and contrastive
        class_column: str = "tissue_general",
        class_min_count: int = 2,
        class_exclude_labels: Optional[List[str]] = None,
        class_aware: bool = False,
        cdd_warmup_steps: int = 2000,
        cdd_ramp_steps: int = 1000,
        cdd_mmd_final_weight: float = 0.0,
        # Training-only parameter groups (see utils/warm_start.py for the groups)
        freeze_groups: Optional[List[str]] = None,
        backbone_lr_scale: float = 1.0,
        # Default reproduces every checkpoint saved before the option existed.
        binning_method: str = "quantile",
        # Deprecated names of class_column, class_min_count, class_exclude_labels and
        # class_aware, kept so checkpoints saved under them still load. None = unset.
        cdd_class_column: Optional[str] = None,
        cdd_min_class_count: Optional[int] = None,
        cdd_exclude_labels: Optional[List[str]] = None,
        cdd_class_aware: Optional[bool] = None,
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
            contrastive (bool, optional): If True, enable the tissue-aware cross-modal InfoNCE: bulk and pseudobulk of the same class (class_column) are positives, other classes negatives. Defaults to False.
            contrastive_temperature (float, optional): InfoNCE temperature. Defaults to 0.1.
            aggregation (bool, optional): If True, enable aggregation consistency losses. Defaults to False.
            agg_fn (Optional[str], optional): The function to use for aggregating single-cell embeddings into pseudobulk embeddings. Defaults to "mean".
            noise (Optional[List[int]], optional): If present, enable denoising task. For it, binning must have been disabled
            esm_emb (bool, optional): If True, load pretrained ESM gene embeddings instead of a learned lookup table.
            esm_emb_path (Optional[str | os.PathLike], optional): Path to the parquet file containing pretrained gene embeddings.
            esm_emb_finetune (bool, optional): If True, allow the pretrained gene embeddings to be fine-tuned.
            freeze_groups (Optional[List[str]], optional): Weight groups ("backbone",
                "decoders", "conditions", "heads") left out of the optimizer and frozen.
            backbone_lr_scale (float, optional): Learning-rate multiplier for the backbone group.
            binning_method (str, optional): How non-zero values are binned, "quantile"
                (default) or "midrank" (see cancerfoundation.data.preprocess). Used by
                embed() and the evaluation targets; training gets it via the data module.
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
        # LR segment: the schedule built by configure_optimizers spans the steps from
        # lr_segment_start to the end of training.
        # reset_lr_schedule discards the checkpoint's scheduler state so a fresh
        # warmup + cosine starts at the restored step.
        self.lr_segment_start = 0
        self.reset_lr_schedule = False
        self.expected_lr_segment: dict = {}  # exact resume: the checkpoint's segment
        self._lr_segment_info: dict = {}
        # Warm start (utils/warm_start.py): set by pretrain.py, kept in every checkpoint.
        self.warm_start_record: Optional[dict] = None
        self.freeze_groups = list(freeze_groups or [])
        self.backbone_lr_scale = float(backbone_lr_scale)
        from cancerfoundation.data.preprocess import check_binning_method
        self.binning_method = check_binning_method(binning_method)
        self.compile_model = compile_model
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm_scheme = norm_scheme
        self.norm_type = norm_type
        self.batchnorm = batchnorm
        self.cell_emb_style = cell_emb_style
        self.perturbation = perturbation
        self.their_init_weights = their_init_weights
        self.n_top_genes = n_top_genes

        # Unified FM parameters
        self.contrastive = contrastive
        self.contrastive_temperature = contrastive_temperature
        self.mmd = mmd
        self.aggregation = aggregation
        self.agg_fn = agg_fn
        self.paired_alignment = paired_alignment
        self.noise = noise or []
        self.denoise = len(self.noise) > 0
        self.esm_emb = esm_emb
        self.esm_emb_path = esm_emb_path
        self.esm_emb_finetune = esm_emb_finetune
        self.verbose = verbose
        self.weight_mvc = weight_mvc
        self.weight_contrastive = weight_contrastive
        self.weight_mmd = weight_mmd
        self.weight_paired = weight_paired
        self.weight_agg = weight_agg
        self.weight_dat = weight_dat
        self.weight_reconstruction = weight_reconstruction
        self.monitor_losses = list(monitor_losses or [])
        self.n_sc_per_pseudobulk = n_sc_per_pseudobulk

        # Contrastive Domain Discrepancy (CAN) parameters
        self.cdd = cdd
        self.weight_cdd = weight_cdd
        # A checkpoint saved under the old cdd_* names passes them instead.
        if cdd_class_column is not None:
            class_column = cdd_class_column
        if cdd_min_class_count is not None:
            class_min_count = cdd_min_class_count
        if cdd_exclude_labels is not None:
            class_exclude_labels = cdd_exclude_labels
        if cdd_class_aware is not None:
            class_aware = cdd_class_aware
        self.class_column = class_column
        self.class_min_count = class_min_count
        self.class_exclude_labels = class_exclude_labels or ["unknown"]
        self.class_aware = class_aware
        self.cdd_warmup_steps = cdd_warmup_steps
        self.cdd_ramp_steps = cdd_ramp_steps
        self.cdd_mmd_final_weight = cdd_mmd_final_weight

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
        self.encoded_conditions = encoded_conditions
        self.where_condition = where_condition
        self.gen_method = gen_method
        self.grad_checkpoint = grad_checkpoint
        self.attn_impl = attn_impl

        self.normalise_bins = normalise_bins
        self.dat_scale = dat_scale
        self.dat_start_step = dat_start_step
        self.dat_interval_steps = dat_interval_steps

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
                mask_value=self.mask_value,
                do_mvc=self.do_mvc,
                conditions=self.conditions_nums,
                input_emb_style=self.input_emb_style,
                n_input_bins=self.n_input_bins,
                use_generative_training=self.USE_GENERATIVE_TRAINING,
                do_dat=self.do_dat,
                no_invert_dat=self.no_invert_dat,
                explicit_zero_prob=self.explicit_zero_prob,
                activation=self.activation,
                norm_scheme=self.norm_scheme,
                norm_type=self.norm_type,
                batchnorm=self.batchnorm,
                cell_emb_style=self.cell_emb_style,
                dat_scale=self.dat_scale,
                normalise_bins=self.normalise_bins,
                where_condition=self.where_condition,
                max_seq_len=self.max_seq_len,
                gen_method=self.gen_method,
                grad_checkpoint=self.grad_checkpoint,
                attn_impl=self.attn_impl,
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
                mask_value=self.mask_value,
                do_mvc=self.do_mvc,
                conditions=self.conditions_nums,
                input_emb_style=self.input_emb_style,
                n_input_bins=self.n_input_bins,
                use_generative_training=self.USE_GENERATIVE_TRAINING,
                do_dat=self.do_dat,
                dat_columns=self.dat_columns,
                encoded_conditions=self.encoded_conditions,
                no_invert_dat=self.no_invert_dat,
                explicit_zero_prob=self.explicit_zero_prob,
                activation=self.activation,
                norm_scheme=self.norm_scheme,
                norm_type=self.norm_type,
                batchnorm=self.batchnorm,
                cell_emb_style=self.cell_emb_style,
                dat_scale=self.dat_scale,
                normalise_bins=self.normalise_bins,
                where_condition=self.where_condition,
                max_seq_len=self.max_seq_len,
                gen_method=self.gen_method,
                grad_checkpoint=self.grad_checkpoint,
                attn_impl=self.attn_impl,
                their_init_weights=self.their_init_weights,
                # Unified FM parameters
                contrastive=self.contrastive,
                contrastive_temperature=self.contrastive_temperature,
                mmd=self.mmd,
                aggregation=self.aggregation,
                agg_fn=self.agg_fn,
                paired_alignment=self.paired_alignment,
                vocab=self.vocab,
                gene_embeddings_path=self.esm_emb_path if self.esm_emb else None,
                gene_embeddings_freeze=not self.esm_emb_finetune,
                verbose=self.verbose,
                weight_mvc=self.weight_mvc,
                weight_contrastive=self.weight_contrastive,
                weight_mmd=self.weight_mmd,
                weight_paired=self.weight_paired,
                weight_agg=self.weight_agg,
                weight_dat=self.weight_dat,
                weight_reconstruction=self.weight_reconstruction,
                monitor_losses=self.monitor_losses,
                # Contrastive Domain Discrepancy (CAN) and the class labels it shares
                # with contrastive. Exclude-label codes are resolved from the category
                # mapping in setup() (needs the datamodule).
                cdd=self.cdd,
                weight_cdd=self.weight_cdd,
                class_column=self.class_column,
                class_min_count=self.class_min_count,
            )
        if self.compile_model:
            self.model = torch.compile(self.model)

    def forward(
            self, 
            data_dict, 
            use_cell_embedding=None, 
            apply_dat: bool = True, 
            skip_unified_losses: bool = False
            ) -> dict:
        """Performs a forward pass through the underlying `TransformerModule`.

        Args:
            data_dict (dict): A dictionary of input tensors.
            use_cell_embedding (Optional[bool], optional): A flag to control a specific training behavior.
                If None, uses the module's default. Defaults to None.
            apply_dat (bool): Whether to apply the DAT loss this step. Defaults to True.

        Returns:
            dict: The output dictionary from the model, typically containing losses.
        """
        if use_cell_embedding is None:
            use_cell_embedding = self.use_cell_embedding

        # First pass without noising
        loss_dict = self.model(data_dict, use_cell_embedding=use_cell_embedding, apply_dat=apply_dat, skip_unified_losses=skip_unified_losses)

        # DENOISING TASK
        if self.denoise:
            loss_dict["loss_noise"] = 0
            for i in self.noise:
                # print(f"Running with noise level {i}")
                loss_noise = self.model(data_dict, use_cell_embedding=use_cell_embedding, noise=i, apply_dat=False)
                loss_dict["loss_noise"] += loss_noise
                loss_dict["total_loss"] += loss_noise

        return loss_dict

    # ------------------------------------------------------------------
    # Contrastive Domain Discrepancy (CAN): exclude codes + MMD->CDD schedule
    # ------------------------------------------------------------------
    def setup(self, stage=None):
        """Resolve the excluded class codes (CDD, contrastive) from the category mapping."""
        if not (self.cdd or self.contrastive) or stage not in (None, "fit"):
            return
        core = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
        dm = getattr(self.trainer, "datamodule", None)
        dataset = getattr(dm, "dataset", None) if dm is not None else None
        if dataset is None:
            return

        class_map = (getattr(dataset, "mapping", {}) or {}).get(self.class_column, {})
        exclude_codes = [int(code) for name, code in class_map.items() if name in self.class_exclude_labels]
        core.class_exclude_codes = tuple(exclude_codes)

    def _update_cdd_schedule(self):
        """Linear MMD->CDD handover.

        During warmup (``cdd_warmup_steps``) the marginal MMD pulls the bulk/pseudobulk
        clouds together; CDD then ramps in over ``cdd_ramp_steps`` as MMD decays out. Stacking both at full weight works against CDD: marginal
        alignment matches P(f(bulk)) to P(f(pseudobulk)) ignoring class, which distorts
        class structure whenever the two domains' tissue proportions differ.
        """
        core = getattr(self.model, "_orig_mod", self.model)
        if not self.cdd:
            return
        R = max(1, self.cdd_ramp_steps)
        t = min(max((self.global_step - self.cdd_warmup_steps) / R, 0.0), 1.0)
        core._cdd_w = self.weight_cdd * t
        core._mmd_w = self.weight_mmd + (self.cdd_mmd_final_weight - self.weight_mmd) * t

    def on_train_batch_start(self, batch, batch_idx):
        self._update_cdd_schedule()

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

        # Assess time speedup
        if not hasattr(self, "_step_timer_start"):
            self._step_timer_start = None
        if self.global_step == 10:
            self._step_timer_start = time.perf_counter()
        if self.global_step == 20 and self._step_timer_start is not None and self.verbose:
            elapsed = time.perf_counter() - self._step_timer_start
            print(f"\n\n[timing] 10 steps took {elapsed:.2f}s \n\n"
                  f"({elapsed / 10:.3f}s/step, {10 / elapsed:.2f} it/s)")

        apply_dat = (
            self.do_dat
            and self.global_step >= self.dat_start_step
            and (self.dat_interval_steps <= 1 or self.global_step % self.dat_interval_steps == 0)
        )
        loss_dict = self.forward(batch, use_cell_embedding=self.use_cell_embedding, apply_dat=apply_dat)

        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Args:
            batch (dict): The batch of data from the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            dict: The dictionary of losses for the validation batch.
        """

        if batch_idx == 0 and self.verbose:
            print(
                f"Rank {self.trainer.global_rank}: Starting validation with {len(self.trainer.val_dataloaders)} batches"
            )
        loss_dict = self.forward(batch, use_cell_embedding=True, skip_unified_losses=True)

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
        optimizer = torch.optim.Adam(self._optimizer_param_groups(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            # Steps of this LR segment: from lr_segment_start (0 unless resuming) to the
            # end of training. LambdaLR counts its own steps, so a fresh scheduler starts
            # at warmup step 0 even when global_step does not.
            total_num_batches = max(
                1, int(self.trainer.estimated_stepping_batches) - int(self.lr_segment_start)
            )
            warmup_steps = (
                int(total_num_batches * self.warmup_ratio_or_step)
                if self.warmup_ratio_or_step < 1
                else int(self.warmup_ratio_or_step)
            )
            self._lr_segment_info = {
                "start_step": int(self.lr_segment_start),
                "total_steps": total_num_batches,
                "warmup_steps": warmup_steps,
                "lr": float(self.lr),
            }

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
            self._lr_segment_info = {
                "start_step": int(self.lr_segment_start),
                "step_lr_interval": self.scheduler_interval,
                "lr": float(self.lr),
            }

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

    def _optimizer_param_groups(self):
        """All parameters, or per-group lists when freezing or scaling the backbone LR."""
        if not self.freeze_groups and self.backbone_lr_scale == 1.0:
            return self.model.parameters()
        # Repo-root helper; only needed when training, where the root is on sys.path.
        from utils.warm_start import classify, normalize_key

        frozen = set(self.freeze_groups)
        base, backbone, n_frozen = [], [], 0
        for name, param in self.model.named_parameters():
            group = classify(normalize_key(f"model.{name}") or "")
            if group in frozen:
                param.requires_grad_(False)
                n_frozen += param.numel()
                continue
            (backbone if group == "backbone" else base).append(param)
        if frozen:
            print(f"Frozen groups {sorted(frozen)}: {n_frozen:,} parameters left out of the optimizer.")
        groups = []
        if base:
            groups.append({"params": base, "lr": self.lr})
        if backbone:
            groups.append({"params": backbone, "lr": self.lr * self.backbone_lr_scale})
        if not groups:
            raise ValueError(f"freeze_groups={sorted(frozen)} leaves nothing to train.")
        return groups

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Lets an exact resume rebuild the same schedule (see pretrain.py).
        checkpoint["lr_segment"] = dict(self._lr_segment_info)
        # Category names of the condition codes, so a warm start from this checkpoint
        # can remap condition encoders by name (utils/warm_start.py).
        mapping = self._condition_mapping()
        if mapping:
            checkpoint["condition_mapping"] = mapping
        if self.warm_start_record:
            checkpoint["warm_start"] = dict(self.warm_start_record)

    def _condition_mapping(self) -> Optional[dict]:
        try:
            dm = self.trainer.datamodule
        except RuntimeError:  # not attached to a trainer
            return None
        dataset = getattr(dm, "dataset", None)
        mapping = getattr(dataset, "mapping", None) or {}
        return {
            col: {str(cat): int(code) for cat, code in mapping[col].items()}
            for col in (self.conditions or [])
            if col in mapping
        } or None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # A resumed warm-started model keeps its lineage in its later checkpoints.
        if checkpoint.get("warm_start") and not self.warm_start_record:
            self.warm_start_record = dict(checkpoint["warm_start"])
        # Runs trained with the removed --cdd-infer-labels saved its label buffers;
        # drop them so the strict state_dict load of a resume does not fail.
        from cancerfoundation.checkpoint import drop_removed_cdd_state

        if drop_removed_cdd_state(checkpoint.get("state_dict", {})) and self.verbose:
            print("[CDD] dropped the removed label-inference buffers from the checkpoint")
        # Called before Lightning restores optimizer and scheduler states. An empty list
        # (not a missing key, which Lightning rejects) restores no scheduler, so the
        # fresh one from configure_optimizers is kept.
        if self.reset_lr_schedule:
            checkpoint["lr_schedulers"] = []

    def on_train_start(self) -> None:
        if self.expected_lr_segment:
            # An exact resume restores the scheduler's step count, but the schedule's
            # shape comes from the current config: say so if the two disagree.
            differs = {
                k: (self.expected_lr_segment[k], v)
                for k, v in self._lr_segment_info.items()
                if k in self.expected_lr_segment and self.expected_lr_segment[k] != v
            }
            if differs:
                print(
                    "[WARNING] exact resume, but the LR schedule differs from the "
                    f"checkpoint's (checkpoint, now): {differs}. Use the original "
                    "epochs/warmup/lr, or resume_mode 'continue' for a new schedule."
                )
        # The optimizer restore brings back the old run's last lr (about 0 at the end of
        # a cosine); start the new segment at the fresh scheduler's value instead.
        if not self.reset_lr_schedule:
            return
        for config in self.trainer.lr_scheduler_configs:
            for group, lr in zip(config.scheduler.optimizer.param_groups, config.scheduler.get_last_lr()):
                group["lr"] = lr

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

    def _run_dense_embed(self, data, batch_size: int, device) -> torch.Tensor:
        """Embed a fully-preprocessed (post-HVG) AnnData using the dense path.

        Applies binning if required, then runs batched forward passes and returns
        a (n_obs, d_model) CPU tensor of CLS-token embeddings.
        """
        if self.input_style == "binned":
            normalise = self.model.decoder.normalise_bins
            for idx in range(data.n_obs):
                data.X[idx] = binning(data.X[idx], self.n_bins, method=self.binning_method)
                if normalise:
                    data.X[idx] = data.X[idx] / self.n_bins

        gene_ids = torch.LongTensor([self.vocab[g] for g in data.var.index])
        count_matrix = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
        n_batches = (len(data) + batch_size - 1) // batch_size
        embeddings = []

        for i in tqdm(range(0, len(data), batch_size), total=n_batches, desc="Embedding cells"):
            batch_expr = torch.FloatTensor(count_matrix[i : i + batch_size]).to(device)
            batch_genes = gene_ids.unsqueeze(0).expand(batch_expr.shape[0], -1).to(device)
            batch_genes = torch.cat(
                [torch.full((batch_expr.shape[0], 1), self.cls_token_id, dtype=torch.long, device=device), batch_genes],
                dim=1,
            )
            batch_expr = torch.cat(
                [torch.full((batch_expr.shape[0], 1), self.pad_value, dtype=batch_expr.dtype, device=device), batch_expr],
                dim=1,
            )
            padding_mask = torch.zeros(batch_genes.shape, dtype=torch.bool, device=device)

            if self.model.use_generative_training:
                output = self.model.embed(src=batch_genes, values=batch_expr, src_key_padding_mask=padding_mask)
                transformer_output = output[0]
            else:
                transformer_output = self.model.encode(src=batch_genes, values=batch_expr, src_key_padding_mask=padding_mask, check_conditions=False)

            embeddings.append(transformer_output[:, 0, :].cpu())

        return torch.cat(embeddings, dim=0)

    def _embed_train_matched(
        self, data, groups, modality, hvg_select, gene_subset, shared_panel, panel_strategy,
        panel_reference, flavor, normalize, batch_size, device, zero_fraction, panel_size,
    ):
        """``input_mode="train_matched"`` / ``"nonzero"``: rank genes, then embed per cell.

        ``data`` is vocabulary-intersected and not normalized (with the library size in
        obs). Returns ``(embeddings, ranking)``, the ranking a list (best first) or, for
        per-modality rankings, a dict of them.
        """
        if not 0.0 <= zero_fraction <= 1.0:
            raise ValueError(f"zero_fraction must be in [0, 1], got {zero_fraction}")
        if gene_subset is not None:
            if isinstance(gene_subset, dict):
                ranking = {str(k): list(v) for k, v in gene_subset.items()}
            else:
                ranking = list(gene_subset)
            print("[INFO] Gene ranking given; not re-fitted.")
        elif not hvg_select:
            ranking = data.var_names.tolist()
            print("[INFO] hvg_select=False: genes are taken in input order, not ranked.")
        elif groups is not None and shared_panel:
            print(f"[INFO] Shared gene ranking across {len(set(groups))} modality groups "
                  f"(strategy='{panel_strategy}').")
            ranking = shared_gene_ranking(data, groups, flavor=flavor, strategy=panel_strategy,
                                          reference=panel_reference)
        elif groups is not None:
            print("Ranking genes per data modality!")
            ranking = {g: gene_ranking(data[groups == g], _classify_modality(g), flavor=flavor)
                       for g in dict.fromkeys(groups.tolist())}
        else:
            kind = _classify_modality(str(modality))
            print(f"  Ranking genes for modality '{kind}'")
            ranking = gene_ranking(data, kind, flavor=flavor)
        if panel_size:
            ranking = ({k: v[:panel_size] for k, v in ranking.items()} if isinstance(ranking, dict)
                       else ranking[:panel_size])

        data = self._apply_normalize(data, normalize)
        if not isinstance(ranking, dict):
            return self._run_sequence_embed(data, ranking, batch_size, device, zero_fraction), ranking
        if groups is None:
            raise ValueError("A per-modality gene ranking (dict) needs modality_col with several values")
        emb = np.zeros((data.n_obs, self.embsize), dtype=np.float32)
        for g, genes in ranking.items():
            mask = groups == g
            if mask.any():
                print(f"Modality '{g}': {int(mask.sum())} cells")
                emb[mask] = self._run_sequence_embed(data[mask], genes, batch_size, device, zero_fraction)
        missing = sorted(set(groups.tolist()) - set(ranking))
        if missing:
            raise ValueError(f"No gene ranking for modality group(s) {missing}")
        return emb, ranking

    def _run_sequence_embed(self, data, ranking, batch_size: int, device, zero_fraction: float) -> np.ndarray:
        """CLS embeddings of per-cell sequences built as in training (``train_matched``).

        Each row of ``data`` (every vocabulary gene, values as the model reads them) is
        binned whole; then its top-ranked zero genes, at most
        ``round(zero_fraction * (max_seq_len - 1))``, and its top-ranked non-zero genes
        fill the ``max_seq_len - 1`` slots after CLS. Rows are padded to the longest
        sequence of their batch.
        """
        pos = data.var_names.get_indexer(list(ranking))
        pos = pos[pos >= 0]
        if len(pos) == 0:
            raise ValueError("None of the ranked genes are in the data")
        budget = self.max_seq_len - 1
        zero_cap = int(round(zero_fraction * budget))
        normalise = self.model.decoder.normalise_bins
        token_ids = np.array([self.vocab[g] for g in data.var_names], dtype=np.int64)
        X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
        names = data.obs_names
        n = data.n_obs
        n_nonzero = np.zeros(n, dtype=np.int64)
        n_zero = np.zeros(n, dtype=np.int64)
        embeddings = []
        print(f"  [train_matched] {len(pos)} ranked genes; per cell up to {zero_cap} zero and "
              f"{budget} total genes after CLS, binned over all {data.n_vars} vocabulary genes "
              f"('{self.binning_method}')")

        for start in tqdm(range(0, n, batch_size), total=(n + batch_size - 1) // batch_size,
                          desc="Embedding cells"):
            stop = min(start + batch_size, n)
            seqs = []
            for i in range(start, stop):
                # Binned whole, then top-ranked zeros + non-zeros; seeded per cell, so a
                # cell's input does not depend on which rows were embedded before it.
                cols, bins, n_nonzero[i], n_zero[i] = train_matched_input(
                    X[i], pos, self.n_bins, budget, zero_cap,
                    method=self.binning_method, seed=cell_seed(names[i]))
                values = bins / self.n_bins if normalise else bins
                seqs.append((token_ids[cols], values))

            length = 1 + max(len(c) for c, _ in seqs)
            b = len(seqs)
            src = torch.full((b, length), self.pad_token_id, dtype=torch.long)
            vals = torch.full((b, length), float(self.pad_value), dtype=torch.float32)
            src[:, 0] = self.cls_token_id
            for j, (ids, values) in enumerate(seqs):
                src[j, 1:1 + len(ids)] = torch.as_tensor(ids)
                vals[j, 1:1 + len(values)] = torch.as_tensor(values, dtype=torch.float32)
            padding_mask = src.eq(self.pad_token_id)
            src, vals, padding_mask = src.to(device), vals.to(device), padding_mask.to(device)
            with torch.no_grad():
                if self.model.use_generative_training:
                    out = self.model.embed(src=src, values=vals, src_key_padding_mask=padding_mask)[0]
                else:
                    out = self.model.encode(src=src, values=vals, src_key_padding_mask=padding_mask,
                                            check_conditions=False)
            embeddings.append(out[:, 0, :].float().cpu())

        total = n_nonzero + n_zero
        print(f"  [train_matched] tokens per cell: mean {total.mean():.0f} "
              f"(non-zero {n_nonzero.mean():.0f}, zero {n_zero.mean():.0f}, "
              f"zero share {n_zero.sum() / max(total.sum(), 1):.0%}); "
              f"min {total.min()}, max {total.max()}")
        return torch.cat(embeddings, dim=0).numpy()

    def _select_genes(self, data, kind: str, seurat_flavor: str = "seurat"):
        """Reduce an AnnData to ``n_top_genes`` using a modality-appropriate scheme.

        Deterministic, and independent of how the model's input is normalized:
        scoring runs on a view of ``data`` that meets the scorer's assumption (see
        ``gene_panel.scoring_view``).
          - ``kind == "sc"``: scanpy HVG (``seurat_flavor``), on CP10K + log1p
            (``seurat_v3``: on raw counts).
          - bulk / pseudobulk: highest MAD on CPM + log1p, robust to the
            cell-type-composition variance that dominates (pseudo)bulk expression.

        Returns the gene-subset AnnData (a copy, values untouched), or ``data``
        unchanged when it already has at most ``n_top_genes`` genes.
        """
        if data.n_vars <= self.n_top_genes:
            return data
        label = f"HVGs (flavor='{seurat_flavor}')" if kind == "sc" else "top-MAD genes"
        print(f"  Selecting {self.n_top_genes} {label} for modality '{kind}'")
        genes = select_panel(data, kind, self.n_top_genes, flavor=seurat_flavor)
        return data[:, genes].copy()

    @staticmethod
    def _resolve_normalize(normalize, normalized, log1p_only) -> Optional[str]:
        """``normalize`` from the new argument or the deprecated ``normalized``/``log1p_only``.

        The old pair meant: ``normalized=True`` -> nothing; otherwise log1p, preceded
        by CP10K unless ``log1p_only`` (whose old default was True).
        """
        if normalized is None and log1p_only is None:
            if normalize is not None and normalize not in NORMALIZE_METHODS:
                raise ValueError(f"normalize must be None or one of {NORMALIZE_METHODS}, got {normalize!r}")
            return normalize
        if normalize is not None:
            raise ValueError("Pass either normalize=... or the deprecated normalized/log1p_only, not both")
        if normalized:
            resolved = None
        elif log1p_only is False:
            resolved = "cp10k_log1p"
        else:
            resolved = "log1p"
        print(f"[DEPRECATED] embed(normalized={normalized}, log1p_only={log1p_only}) is "
              f"normalize={resolved!r}; pass normalize= instead.")
        return resolved

    @staticmethod
    def _apply_normalize(data, normalize: Optional[str]):
        """Normalize the model input in place, by each row's total over all input genes."""
        if normalize is None:
            return data
        lib = data.obs[LIBRARY_SIZE_COL].to_numpy(dtype=np.float64)
        data.X = normalize_expression(data.X, normalize, lib)
        return data

    def _embed_by_modality(
        self,
        data,
        modality_col: str,
        hvg_select: bool,
        batch_size: int,
        device,
        gene_subsets: Optional[dict] = None,
        normalize: Optional[str] = None,
    ) -> tuple:
        """Embed each modality group independently with its own gene set.

        Gene selection per group follows _select_genes: scanpy 'seurat' HVG for SC,
        MAD on CPM + log1p for bulk/pseudobulk, each on its own scoring view. The
        model input is then normalized per ``normalize``.

        Args:
            data: AnnData already vocab-intersected, NOT normalized, with
                ``obs[LIBRARY_SIZE_COL]``.
            modality_col: obs column holding modality labels.
            hvg_select: whether to run gene selection per modality (ignored when
                gene_subsets given).
            batch_size: forward-pass batch size.
            device: torch device.
            gene_subsets: if provided, a dict mapping modality label → gene list; skips
                gene selection.
            normalize: model-input normalization (see ``embed``).

        Returns:
            (emb_array: np.ndarray of shape (n_obs, d_model),
             gene_set_used: dict[str, list[str]])
        """
        mod_vals = data.obs[modality_col].astype(str)
        unique_mods = mod_vals.unique()
        print(f"[INFO] Available modalities: {unique_mods}")

        emb_array = np.zeros((len(data), self.embsize), dtype=np.float32)
        gene_set_used: dict = {}
        obs_pos = {name: i for i, name in enumerate(data.obs_names)}

        for mod_val in unique_mods:
            mod_mask = (mod_vals == mod_val).values
            mod_data = data[mod_mask].copy()
            kind = _classify_modality(mod_val)
            print(f"Modality '{mod_val}' ({kind}): {mod_data.n_obs} cells, {mod_data.n_vars} genes")

            if gene_subsets is not None and mod_val in gene_subsets:
                avail = [g for g in gene_subsets[mod_val] if g in mod_data.var.index]
                mod_data = mod_data[:, avail].copy()
            elif hvg_select:
                mod_data = self._select_genes(mod_data, kind)

            gene_set_used[mod_val] = mod_data.var_names.tolist()
            mod_data = self._apply_normalize(mod_data, normalize)
            mod_emb = self._run_dense_embed(mod_data, batch_size, device)

            positions = [obs_pos[name] for name in mod_data.obs_names]
            emb_array[positions] = mod_emb.numpy()

        return emb_array, gene_set_used

    @torch.no_grad()
    def embed(
        self,
        adata,
        batch_size: int = 64,
        normalize: Optional[str] = None,
        hvg_select=True,
        gene_subset=None,
        flavor="seurat",
        modality: str = "sc",
        modality_col: Optional[str] = None,
        shared_panel: bool = True,
        panel_strategy: str = "consensus",
        panel_reference: Optional[str] = None,
        return_preprocessed: bool = False,
        normalized: Optional[bool] = None,
        log1p_only: Optional[bool] = None,
        input_mode: str = "legacy_panel",
        zero_fraction: float = 0.4,
        panel_size: Optional[int] = None):
        """Embeds an AnnData object into cell embeddings.

        **Input mode.** ``input_mode`` decides how each cell's input sequence is built
        (``EMBED_INPUT_MODES``):

        - ``"legacy_panel"`` (default here, for existing callers): one fixed panel of
          ``n_top_genes`` genes for every cell, binned *within the panel*, every zero
          kept. Unlike training: a single cell's HVG panel is mostly zeros, and its bins
          depend on which genes the panel kept.
        - ``"train_matched"``: what the collator does in training. Each row is binned
          over ALL its vocabulary genes first. The panel becomes a gene *ranking* (the
          same scorers, joint or per modality, no cut unless ``panel_size``). Each cell
          then takes its ``min(#zeros, round(zero_fraction * (max_seq_len - 1)))``
          top-ranked zero genes, as training takes that many zeros (an absolute count,
          ``zero_percentages`` of the budget), and fills the rest of the
          ``max_seq_len - 1`` slots with its top-ranked non-zero genes. Deterministic:
          no sampling, and quantile ties are seeded per cell from its obs name, so a
          cell's embedding does not depend on the batch or the other rows. Works the
          same for sc, bulk and pseudobulk; bulk rows have few zeros and get mostly
          expressed genes.
        - ``"nonzero"``: ``train_matched`` with ``zero_fraction=0`` (scGPT-style).

        With ``train_matched`` / ``nonzero``, ``gene_set_used`` is the ranking (ordered,
        best first; a dict per modality without a shared panel), and ``gene_subset``
        takes such a ranking back, so a chunked caller ranks once.

        Handles all preprocessing: gene intersection with vocab, deterministic
        modality-aware gene selection, optional normalization, binning, and batched
        forward passes through the transformer.

        **Model input.** By default (``normalize=None``) the values are embedded as
        they are, which matches training: the training store holds the source ``X``
        unnormalized, and per-row quantile binning is scale-invariant. ``normalize``
        transforms them first: ``"log1p"``, ``"cpm_log1p"`` or ``"cp10k_log1p"``. The
        library size is each row's total over ALL input genes, taken before the
        vocabulary intersection and the panel subset.

        **Gene panel.** Selection never sees the model input. It scores genes on a view
        of the input values transformed to meet each scorer's assumption (see
        ``gene_panel.scoring_view``): scanpy ``seurat`` HVG on CP10K + log1p for
        single-cell (``modality="sc"``), MAD on CPM + log1p for bulk/pseudobulk. Values
        that are already normalized or log-scale are detected and not transformed twice.

        Args:
            adata: AnnData object (h5ad). X should contain expression values
                (dense or sparse).
            batch_size: Batch size for inference.
            normalize: None (default), "log1p", "cpm_log1p" or "cp10k_log1p".
            modality: Modality of the whole AnnData when ``modality_col`` is not set —
                one of "sc", "bulk", "pseudobulk". Drives single-modality gene selection.
            modality_col: If provided, name of an obs column holding per-cell modality
                labels (e.g. "modality"). A column with a single value sets
                ``modality``. With several values and ``shared_panel=False``, gene
                selection and dense embedding are performed independently per modality
                group, and ``gene_set_used`` comes back as a ``dict[str, list[str]]``.
                ``gene_subset`` can also be passed as such a dict to skip re-fitting
                selection.
            shared_panel: When True (default) and ``modality_col`` resolves to two or
                more groups, fit ONE gene panel across all of them (see
                ``select_shared_panel``) and embed every cell through it, returning a
                flat ``gene_set_used``. Cross-modality metrics are only interpretable
                this way: a per-modality panel makes the input distribution differ
                *with* modality, which no batch-integration metric can tell apart from
                a real batch effect. No-op for single-group data, so single-modality
                callers are unaffected. Ignored when ``gene_subset`` is given.
            panel_strategy: How to fit the shared panel — "consensus" (default),
                "reference", or "pooled". See ``select_shared_panel``.
            panel_reference: Group name to fit on when ``panel_strategy="reference"``.
            return_preprocessed: If True, return the normalized, vocab-intersected,
                gene-selected AnnData (un-binned) instead of running the forward pass.
                Used by ``preprocess_for_embedding`` so both paths share identical
                preprocessing. Not compatible with the per-modality ``modality_col`` path.
            normalized, log1p_only: Deprecated; mapped onto ``normalize``
                (``normalized=True`` -> None, ``log1p_only=False`` -> "cp10k_log1p",
                otherwise "log1p").

        Returns:
            Tuple of (pd.DataFrame with cell IDs as index and columns dim_0, dim_1, ...,
            gene_set_used as list[str] or dict[str, list[str]] when modality_col is set),
            or the preprocessed AnnData when ``return_preprocessed`` is True.
        """
        normalize = self._resolve_normalize(normalize, normalized, log1p_only)
        self.model.eval()
        device = next(self.model.parameters()).device

        # Work on a copy to avoid mutating the input
        data = adata.copy()
        if hasattr(data.X, "toarray"):
            data.X = data.X.toarray()
        # Float even without normalization: binning writes bin indices back into X and
        # normalise_bins divides them, which an integer counts matrix would truncate.
        data.X = np.asarray(data.X, dtype=np.float32)

        data = self._maybe_fix_negative(data)

        # Library size over every input gene, before any gene is dropped, so CPM/CP10K
        # (for the model input and for panel scoring) means the row's real total.
        data.obs[LIBRARY_SIZE_COL] = library_size(data.X)
        scale = value_scale(data.X)
        if normalize is None:
            print(f"Model input: values as stored ({scale}), as in training.")
        else:
            print(f"Model input: {normalize} of the input values ({scale}).")
            if scale == "log":
                print(
                    f"[WARNING] normalize={normalize!r} but the values already look "
                    "log-scale; they will be transformed again."
                )

        # Intersect genes with vocab. Sorted so the column order is reproducible
        # across processes (set iteration order is not); the encoder is permutation
        # invariant over gene tokens, so this only affects reproducibility.
        common_genes = sorted(set(self.vocab.keys()).intersection(set(data.var.index)))
        if len(common_genes) == 0:
            raise ValueError("No common genes between vocab and data. Check gene name format.")
        print(f"Common genes: {len(common_genes)} / {data.n_vars}")
        data = data[:, common_genes].copy()

        # A modality column with a single value names the modality of the whole file.
        n_groups = 0
        if modality_col is not None and modality_col in data.obs.columns:
            group_vals = data.obs[modality_col].astype(str).to_numpy()
            n_groups = len(set(group_vals.tolist()))
            if n_groups == 1:
                modality = group_vals[0]

        # ── Shared panel: fit one gene set across modality groups ──────────────
        # Resolved into a flat `gene_subset` so the single-panel branch below does the
        # rest (subset once, one dense forward pass for every cell).
        if (
            gene_subset is None
            and shared_panel
            and hvg_select
            and n_groups > 1
            and not return_preprocessed
            and input_mode == "legacy_panel"
        ):
            if data.n_vars > self.n_top_genes:
                print(
                    f"[INFO] Shared gene panel across {n_groups} modality groups "
                    f"(strategy='{panel_strategy}'): one panel for all cells."
                )
                gene_subset = select_shared_panel(
                    data,
                    groups=group_vals,
                    n_top=self.n_top_genes,
                    flavor=flavor,
                    strategy=panel_strategy,
                    reference=panel_reference,
                )
            else:
                print(
                    f"[INFO] Shared panel requested but data has {data.n_vars} genes "
                    f"<= n_top_genes={self.n_top_genes}; keeping all genes."
                )
                gene_subset = data.var_names.tolist()

        if input_mode not in EMBED_INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; expected one of {EMBED_INPUT_MODES}")
        if input_mode != "legacy_panel":
            if return_preprocessed:
                raise ValueError("return_preprocessed needs input_mode='legacy_panel'")
            groups = (data.obs[modality_col].astype(str).to_numpy() if n_groups > 1 else None)
            emb, ranking = self._embed_train_matched(
                data, groups, modality, hvg_select, gene_subset, shared_panel, panel_strategy,
                panel_reference, flavor, normalize, batch_size, device,
                zero_fraction=0.0 if input_mode == "nonzero" else zero_fraction,
                panel_size=panel_size,
            )
            return pd.DataFrame(
                emb,
                index=adata.obs_names,
                columns=[f"dim_{i}" for i in range(emb.shape[1])],
            ), ranking

        if gene_subset is not None:
            if isinstance(gene_subset, dict):
                # Per-modality gene subset — requires modality_col
                if modality_col is None or modality_col not in data.obs.columns:
                    raise ValueError(
                        "gene_subset as dict requires modality_col to be provided and present in adata.obs."
                    )
                emb_array, gs_used = self._embed_by_modality(
                    data, modality_col, hvg_select=False,
                    batch_size=batch_size, device=device, gene_subsets=gene_subset,
                    normalize=normalize,
                )
                return pd.DataFrame(
                    emb_array,
                    index=adata.obs_names,
                    columns=[f"dim_{i}" for i in range(emb_array.shape[1])],
                ), gs_used
            else:
                # Apply supplied gene set (e.g. from train preprocessing) — no re-fit
                available_set = set(data.var.index)
                available = [g for g in gene_subset if g in available_set]
                data = data[:, available].copy()

        elif n_groups > 1 and hvg_select:
            # Per-modality selection: fit and embed each modality group independently
            print("Performing gene selection per data modality!")
            emb_array, gs_used = self._embed_by_modality(
                data, modality_col, hvg_select=True,
                batch_size=batch_size, device=device, normalize=normalize,
            )
            return pd.DataFrame(
                emb_array,
                index=adata.obs_names,
                columns=[f"dim_{i}" for i in range(emb_array.shape[1])],
            ), gs_used

        elif hvg_select:
            data = self._select_genes(data, kind=_classify_modality(str(modality)), seurat_flavor=flavor)

        data = self._apply_normalize(data, normalize)

        # Early return for preprocess_for_embedding: selected, un-binned data.
        if return_preprocessed:
            return data

        # Retrieve gene set
        gene_set_used = data.var_names.tolist()

        # Dense path: all samples in a batch share the same gene columns.
        emb = self._run_dense_embed(data, batch_size, device).numpy()
        return pd.DataFrame(
            emb,
            index=adata.obs_names,
            columns=[f"dim_{i}" for i in range(emb.shape[1])],
        ), gene_set_used

    def preprocess_for_embedding(
        self,
        adata,
        normalize: Optional[str] = None,
        gene_subset: list | None = None,
        return_edges: bool = False,
        modality: str = "sc",
        normalized: Optional[bool] = None,
    ):
        """Normalize, intersect vocab, select genes, and optionally bin — without running the forward pass.

        Delegates normalization, vocab intersection, and gene selection to
        ``embed(..., return_preprocessed=True)`` so the frozen and fine-tuned paths
        select genes identically (scanpy ``seurat`` for SC, MAD for bulk/pseudobulk),
        then applies binning here.

        Calling this on the full dataset before a train/test split ensures every cell
        uses the same gene set.  Pass the returned ``var.index`` as ``gene_subset`` to
        a second call on held-out data so gene selection is never re-fit on test samples.

        Args:
            adata: Input AnnData object.
            normalize: Model-input normalization, as in ``embed`` (default None).
            gene_subset: If provided, skip gene selection and subset directly to these gene
                names (must be a subset of the vocab-intersected genes).
            modality: Modality of the data ("sc", "bulk", "pseudobulk") — drives gene
                selection, matching the ``modality`` argument of ``embed``.
            normalized: Deprecated. ``True`` -> no normalization, ``False`` -> log1p
                (what it meant before ``normalize`` existed).

        Returns:
            Preprocessed AnnData ready for ``embed_for_finetune()`` or ``embed(hvg_select=False)``.
        """
        # Reuse embed()'s preprocessing (intersect → select → normalize) so the two
        # paths never diverge; return_preprocessed short-circuits before the forward pass.
        data = self.embed(
            adata,
            normalize=normalize,
            normalized=normalized,
            hvg_select=(gene_subset is None),
            gene_subset=gene_subset,
            modality=modality,
            return_preprocessed=True,
        )

        if self.input_style == "binned":
            from cancerfoundation.data.preprocess import binning_with_edges
            normalise = self.model.decoder.normalise_bins
            X = data.X if isinstance(data.X, np.ndarray) else data.X.toarray()
            if return_edges:
                orig_X = X.copy()
                all_edges = np.zeros((data.n_obs, self.n_bins - 1), dtype=np.float32)
                for idx in range(data.n_obs):
                    binned, edges = binning_with_edges(X[idx], self.n_bins, method=self.binning_method)
                    all_edges[idx] = edges
                    X[idx] = binned / self.n_bins if normalise else binned
            else:
                for idx in range(data.n_obs):
                    X[idx] = binning(X[idx], self.n_bins, method=self.binning_method)
                    if normalise:
                        X[idx] = X[idx] / self.n_bins
            data.X = X

        if return_edges and self.input_style == "binned":
            return data, orig_X, all_edges
        return data

    def embed_for_finetune(
        self,
        gene_ids: torch.Tensor,
        expr: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient-enabled cell embedding for end-to-end fine-tuning.

        Identical to the per-batch forward pass inside ``embed()``, but without the
        ``@torch.no_grad()`` decorator so gradients flow through the transformer.
        Expects inputs already preprocessed by ``preprocess_for_embedding()``.

        Args:
            gene_ids: ``(n_genes,)`` LongTensor of vocabulary IDs for the fixed gene set.
            expr: ``(batch_size, n_genes)`` FloatTensor of preprocessed expression values.

        Returns:
            ``(batch_size, d_model)`` FloatTensor — CLS token cell embeddings with gradient history.
        """
        device = next(self.model.parameters()).device
        bs = expr.shape[0]

        batch_genes = gene_ids.unsqueeze(0).expand(bs, -1).to(device)
        batch_expr = expr.to(device)

        # Prepend CLS token (pad expression)
        batch_genes = torch.cat(
            [torch.full((bs, 1), self.cls_token_id, dtype=torch.long, device=device), batch_genes],
            dim=1,
        )
        batch_expr = torch.cat(
            [torch.full((bs, 1), self.pad_value, dtype=batch_expr.dtype, device=device), batch_expr],
            dim=1,
        )

        # Nothing is masked
        padding_mask = torch.zeros(batch_genes.shape, dtype=torch.bool, device=device)

        if self.model.use_generative_training:
            output, _ = self.model.embed(
                src=batch_genes,
                values=batch_expr,
                src_key_padding_mask=padding_mask,
            )
        else:
            output = self.model.encode(
                src=batch_genes,
                values=batch_expr,
                src_key_padding_mask=padding_mask,
                check_conditions=False,
            )

        return output[:, 0, :]  # CLS token embedding
