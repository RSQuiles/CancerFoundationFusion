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
import time

# ---------------------------------------------------------------------------
# Module-level helpers for modality-aware HVG selection
# ---------------------------------------------------------------------------
_SC_MODALITY_NORM: frozenset = frozenset({
    "sc", "singlecell", "scrna", "scrnaseq", "subsampled", "pairedsc",
})
_PB_MODALITY_NORM: frozenset = frozenset({
    "pseudobulk", "synthpb", "pairedpb", "pseudo",
})

def _classify_modality(val: str) -> str:
    """Map a raw modality label to 'sc', 'pseudobulk', or 'bulk'."""
    norm = val.lower().replace(" ", "").replace("-", "").replace("_", "")
    if norm in _SC_MODALITY_NORM or norm.startswith("sc"):
        return "sc"
    if "pseudo" in norm or norm in _PB_MODALITY_NORM:
        return "pseudobulk"
    return "bulk"


def _top_mad_genes(X, n_top: int) -> np.ndarray:
    """Return indices of the ``n_top`` genes (columns) with highest MAD.

    MAD (median absolute deviation) is computed on the values as passed — in this
    pipeline the data is already log1p-transformed by the time gene selection runs,
    so this is log1p + MAD. Robust to the composition-driven variance that dominates
    bulk/pseudobulk expression.
    """
    X = X if isinstance(X, np.ndarray) else X.toarray()

    # Detect if data is likely not log1p-transformed.
    # Heuristic: log1p data has a max value typically well below 20;.
    if X.max() > 20:
        print("  [INFO] Non-log1p data detected. Normalizing!") 
        X = np.log1p(X)

    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    return np.argsort(mad)[-n_top:]


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
        norm_scheme: str,
        norm_type: str,
        cell_emb_style: str,
        batchnorm: bool,
        dat_scale: float,
        normalise_bins: bool,
        where_condition: str,
        gen_method: str,
        their_init_weights: bool,
        dat_start_step: int = 0,
        dat_interval_steps: int = 1,
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
        paired_alignment: bool = False,
        verbose: bool = False,
        weight_mvc: float = 1.0,
        weight_contrastive: float = 1.0,
        weight_paired: float = 1.0,
        weight_agg: float = 1.0,
        weight_dat: float = 1.0,
        weight_reconstruction: float = 1.0,
        n_sc_per_pseudobulk: int = 10,
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
        self.norm_scheme = norm_scheme
        self.norm_type = norm_type
        self.batchnorm = batchnorm
        self.cell_emb_style = cell_emb_style
        self.perturbation = perturbation
        self.their_init_weights = their_init_weights
        self.n_top_genes = n_top_genes

        # Unified FM parameters
        self.contrastive = contrastive
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
        self.weight_paired = weight_paired
        self.weight_agg = weight_agg
        self.weight_dat = weight_dat
        self.weight_reconstruction = weight_reconstruction
        self.n_sc_per_pseudobulk = n_sc_per_pseudobulk

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
                norm_scheme=self.norm_scheme,
                norm_type=self.norm_type,
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
                paired_alignment=self.paired_alignment,
                vocab=self.vocab,
                gene_embeddings_path=self.esm_emb_path if self.esm_emb else None,
                gene_embeddings_freeze=not self.esm_emb_finetune,
                verbose=self.verbose,
                weight_mvc=self.weight_mvc,
                weight_contrastive=self.weight_contrastive,
                weight_paired=self.weight_paired,
                weight_agg=self.weight_agg,
                weight_dat=self.weight_dat,
                weight_reconstruction=self.weight_reconstruction,
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
        if self.global_step == 5:
            self._step_timer_start = time.perf_counter()
        if self.global_step == 15 and self._step_timer_start is not None and self.verbose:
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

    def _run_dense_embed(self, data, batch_size: int, device) -> torch.Tensor:
        """Embed a fully-preprocessed (post-HVG) AnnData using the dense path.

        Applies binning if required, then runs batched forward passes and returns
        a (n_obs, d_model) CPU tensor of CLS-token embeddings.
        """
        if self.input_style == "binned":
            normalise = self.model.decoder.normalise_bins
            for idx in range(data.n_obs):
                data.X[idx] = binning(data.X[idx], self.n_bins)
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

    def _select_genes(self, data, kind: str, seurat_flavor: str = "seurat"):
        """Reduce an AnnData to ``n_top_genes`` using a modality-appropriate scheme.

        Deterministic. Assumes ``data`` is already log1p-normalised.
          - ``kind == "sc"``: scanpy HVG (``seurat_flavor``, bin-normalised dispersion),
            appropriate for sparse log1p single-cell data.
          - bulk / pseudobulk: genes with highest MAD on the log1p values, robust to
            the cell-type-composition variance that dominates (pseudo)bulk expression.

        Returns the gene-subset AnnData (a copy), or ``data`` unchanged when it already
        has at most ``n_top_genes`` genes.
        """
        if data.n_vars <= self.n_top_genes:
            return data
        if kind == "sc":
            print(f"  Selecting {self.n_top_genes} HVGs (flavor='{seurat_flavor}') for modality 'sc'")
            sc.pp.highly_variable_genes(data, n_top_genes=self.n_top_genes, flavor=seurat_flavor)
            return data[:, data.var["highly_variable"]].copy()
        # bulk / pseudobulk → log1p + MAD
        print(f"  Selecting {self.n_top_genes} top-MAD genes for modality '{kind}'")
        top_idx = _top_mad_genes(data.X, self.n_top_genes)
        return data[:, top_idx].copy()

    def _embed_by_modality(
        self,
        data,
        modality_col: str,
        hvg_select: bool,
        batch_size: int,
        device,
        gene_subsets: Optional[dict] = None,
    ) -> tuple:
        """Embed each modality group independently with its own gene set.

        Gene selection per group follows _select_genes: scanpy 'seurat' HVG for SC,
        log1p + MAD for bulk/pseudobulk. Data is assumed already log1p-normalised.

        Args:
            data: AnnData already vocab-intersected and normalised.
            modality_col: obs column holding modality labels.
            hvg_select: whether to run gene selection per modality (ignored when
                gene_subsets given).
            batch_size: forward-pass batch size.
            device: torch device.
            gene_subsets: if provided, a dict mapping modality label → gene list; skips
                gene selection.

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
            mod_emb = self._run_dense_embed(mod_data, batch_size, device)

            positions = [obs_pos[name] for name in mod_data.obs_names]
            emb_array[positions] = mod_emb.numpy()

        return emb_array, gene_set_used

    @torch.no_grad()
    def embed(
        self,
        adata,
        batch_size: int = 64,
        normalized=False,
        log1p_only=True,
        hvg_select=True,
        gene_subset=None,
        flavor="seurat",
        modality: str = "sc",
        modality_col: Optional[str] = None,
        return_preprocessed: bool = False):
        """Embeds an AnnData object into cell embeddings.

        Handles all preprocessing: gene intersection with vocab, deterministic
        modality-aware gene selection, binning, and batched forward passes through the
        transformer.

        Gene selection (see ``_select_genes``) depends on modality: scanpy ``seurat``
        HVG for single-cell (``modality="sc"``), log1p + MAD for bulk/pseudobulk
        (``modality="bulk"`` / ``"pseudobulk"``). Data is log1p by the time selection
        runs (either already ``normalized`` or normalised here).

        Args:
            adata: AnnData object (h5ad). X should contain expression values
                (dense or sparse).
            batch_size: Batch size for inference.
            modality: Modality of the whole AnnData when ``modality_col`` is not set —
                one of "sc", "bulk", "pseudobulk". Drives single-modality gene selection.
            modality_col: If provided, name of an obs column holding per-cell modality
                labels (e.g. "modality" or "_eval_modality"). When set, gene selection
                and dense embedding are performed independently per modality group so
                each modality's variance structure drives its own gene set.
                ``gene_set_used`` is then returned as a ``dict[str, list[str]]`` instead
                of a flat list. ``gene_subset`` can also be passed as such a dict to skip
                re-fitting selection.
            return_preprocessed: If True, return the normalised, vocab-intersected,
                gene-selected AnnData (un-binned) instead of running the forward pass.
                Used by ``preprocess_for_embedding`` so both paths share identical
                preprocessing. Not compatible with the per-modality ``modality_col`` path.

        Returns:
            Tuple of (pd.DataFrame with cell IDs as index and columns dim_0, dim_1, ...,
            gene_set_used as list[str] or dict[str, list[str]] when modality_col is set),
            or the preprocessed AnnData when ``return_preprocessed`` is True.
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
            sc.pp.log1p(data)

        # Intersect genes with vocab
        common_genes = list(set(self.vocab.keys()).intersection(set(data.var.index)))
        if len(common_genes) == 0:
            raise ValueError(f"No common genes between vocab and data. Check gene name format.")
        print(f"Common genes: {len(common_genes)} / {data.n_vars}")
        data = data[:, common_genes].copy()

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

        elif modality_col is not None and modality_col in data.obs.columns and hvg_select:
            # Per-modality selection: fit and embed each modality group independently
            print("Performing gene selection per data modality!")
            emb_array, gs_used = self._embed_by_modality(
                data, modality_col, hvg_select=True,
                batch_size=batch_size, device=device,
            )
            return pd.DataFrame(
                emb_array,
                index=adata.obs_names,
                columns=[f"dim_{i}" for i in range(emb_array.shape[1])],
            ), gs_used

        elif hvg_select:
            data = self._select_genes(data, kind=modality, seurat_flavor=flavor)

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
        normalized: bool = False,
        gene_subset: list | None = None,
        return_edges: bool = False,
        modality: str = "sc",
    ):
        """Normalize, intersect vocab, select genes, and optionally bin — without running the forward pass.

        Delegates normalization, vocab intersection, and gene selection to
        ``embed(..., return_preprocessed=True)`` so the frozen and fine-tuned paths
        select genes identically (scanpy ``seurat`` for SC, log1p + MAD for
        bulk/pseudobulk), then applies binning here.

        Calling this on the full dataset before a train/test split ensures every cell
        uses the same gene set.  Pass the returned ``var.index`` as ``gene_subset`` to
        a second call on held-out data so gene selection is never re-fit on test samples.

        Args:
            adata: Input AnnData object.
            normalized: If True, skip CP10K + log1p normalization (data already normalized).
            gene_subset: If provided, skip gene selection and subset directly to these gene
                names (must be a subset of the vocab-intersected genes).
            modality: Modality of the data ("sc", "bulk", "pseudobulk") — drives gene
                selection, matching the ``modality`` argument of ``embed``.

        Returns:
            Preprocessed AnnData ready for ``embed_for_finetune()`` or ``embed(hvg_select=False)``.
        """
        # Reuse embed()'s preprocessing (normalize → intersect → select) so the two
        # paths never diverge; return_preprocessed short-circuits before the forward pass.
        data = self.embed(
            adata,
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
                    binned, edges = binning_with_edges(X[idx], self.n_bins)
                    all_edges[idx] = edges
                    X[idx] = binned / self.n_bins if normalise else binned
            else:
                for idx in range(data.n_obs):
                    X[idx] = binning(X[idx], self.n_bins)
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
