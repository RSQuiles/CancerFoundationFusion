from pathlib import Path
import os
from typing import Dict, Mapping, Optional, Union, Type, Callable, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pandas as pd
import math
from .layers import RefactoredCFGenerator, QuickCFGenerator, CFLayer, CFGenerator

from .grad_reverse import grad_reverse

from . import utils

class TransformerModule(nn.Module):
    """The main Transformer model for gene expression modeling.

    This model can be configured for both perceptual (masked language model-style) and generative tasks.
    It handles gene and expression value encoding, optional conditional information,
    and can be extended with modules for Masked Value Prediction for Cell-embeddings (MVC) and Domain Adversarial Training (DAT).
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        out_dim: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        pad_value: int,
        pad_token_id: int,
        criterion,
        activation: Callable[[Tensor], Tensor],
        do_mvc: bool,
        dropout: float,
        conditions: Optional[Dict],
        input_emb_style: str,
        n_input_bins: Optional[int],
        cell_emb_style: str,
        mvc_decoder_style: str,
        explicit_zero_prob: Optional[bool],
        use_generative_training: bool,
        norm_scheme: str,
        norm_type: str,
        do_dat: bool,
        batchnorm: bool,
        dat_scale: float,
        normalise_bins: bool,
        no_invert_dat: bool,
        where_condition: str,
        max_seq_len: int,
        gen_method: str,
        their_init_weights: bool,
        grad_checkpoint: bool = False,
        attn_impl: str = "mha",
        # Unified FM parameters
        contrastive: bool = False,
        mmd: bool = False,
        aggregation: bool = False,
        agg_fn: Optional[str] = None,
        paired_alignment: bool = False,
        vocab: Optional[Dict[str, int]] = None,
        gene_embeddings_path: Optional[Union[str, os.PathLike, Path]] = None,
        gene_embeddings_freeze: bool = True,
        dat_columns: Optional[list[str]] = [],
        encoded_conditions: Optional[list[str]] = None,
        verbose: bool = False,
        weight_mvc: float = 1.0,
        weight_contrastive: float = 1.0,
        weight_mmd: float = 1.0,
        weight_paired: float = 1.0,
        weight_agg: float = 1.0,
        weight_dat: float = 1.0,
        weight_reconstruction: float = 1.0,
        monitor_losses: Optional[list[str]] = None,
        # Contrastive Domain Discrepancy (CAN) parameters
        cdd: bool = False,
        weight_cdd: float = 0.3,
        cdd_class_column: str = "tissue_general",
        cdd_min_class_count: int = 2,
        cdd_exclude_class_codes: Optional[list[int]] = None,
        cdd_infer_labels: bool = False,
        cdd_cluster_iters: int = 10,
        cdd_cluster_ambiguity: float = 0.05,
        cdd_cluster_min_size: int = 3,
        cdd_relabel_known: bool = False,
        cdd_cluster_source_fallback: bool = True,
    ):
        """Initializes the TransformerModule.

        Args:
            ntoken (int): The number of unique gene tokens.
            d_model (int): The dimensionality of the model embeddings.
            out_dim (int): The output dimension of the decoders.
            nhead (int): The number of attention heads in the transformer.
            d_hid (int): The dimension of the feedforward network model in the transformer.
            nlayers (int): The number of transformer encoder layers.
            pad_value (int): The value used for padding in the input expression values.
            pad_token_id (int): The token ID used for padding.
            criterion: The loss function for expression prediction.
            activation (Callable[[Tensor], Tensor]): The activation function for the transformer encoder layers.
            do_mvc (bool): Whether to include the MVC decoder.
            dropout (float): The dropout rate.
            conditions (Optional[Dict]): A dictionary defining conditional variables, mapping condition names to the number of categories.
            input_emb_style (str): The style of input value embedding ("continuous", "category", "scaling").
            n_input_bins (Optional[int]): The number of bins for categorical value embedding. Required if `input_emb_style` is "category".
            cell_emb_style (str): The method to obtain cell embeddings ("cls", "avg-pool", "w-pool").
            mvc_decoder_style (str): The architecture for the MVC decoder.
            explicit_zero_prob (Optional[bool]): Whether to explicitly predict zero-expression probability.
            use_generative_training (bool): Whether to use the generative training setup.
            norm_scheme (str): Normalisation scheme: "pre" or "post".
            norm_type (str): Normalisation layer type: "layer" (LayerNorm) or "rms" (RMSNorm).
            do_dat (bool): Whether to include Domain Adversarial Training.
            batchnorm (bool): Whether to use batch normalization on the input embeddings.
            weight_conditionloss (float): Weight for the condition prediction loss in DAT.
            dat_scale (float): Scale factor for the gradient reversal layer in DAT.
            normalise_bins (bool): Whether to apply a sigmoid to the output of the decoders.
            contrastive (bool): If True, enable contrastive learning. It brings the pseudobulk and real bulk samples closer together in the embedding space. Defaults to False.
            aggregation (bool): If True, enable aggregation consistency losses. Defaults to False.
            agg_fn (Optional[str]): The function to use for aggregating single-cell embeddings into pseudobulk embeddings. Defaults to "mean".
        """
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.conditions = conditions

        # Which conditions actually get a ConditionEncoder + where_condition injection.
        # The full `conditions` set still drives label provisioning for DAT/CDD and the
        # collator; `encoded_conditions` (a subset, default = all) decouples "encode &
        # inject this condition" from "expose this condition's label". Order follows the
        # `conditions` dict for deterministic decoder concatenation / checkpoint layout.
        if conditions is None:
            self.encoded_conditions = []
        elif encoded_conditions is None:
            self.encoded_conditions = list(conditions.keys())
        else:
            encoded_set = set(encoded_conditions)
            unknown = encoded_set - set(conditions.keys())
            if unknown:
                raise ValueError(
                    f"encoded_conditions {sorted(unknown)} are not in conditions "
                    f"{list(conditions.keys())}."
                )
            self.encoded_conditions = [c for c in conditions.keys() if c in encoded_set]
        self._use_condition_encoders = len(self.encoded_conditions) > 0
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.pad_token_id = pad_token_id
        self.norm_scheme = norm_scheme
        self.use_generative_training = use_generative_training
        self.where_condition = where_condition
        self.max_seq_len = max_seq_len
        self.gen_method = gen_method
        self.grad_checkpoint = grad_checkpoint
        self.attn_impl = attn_impl
        self.contrastive = contrastive
        self.mmd = mmd
        self.aggregation = aggregation
        self.agg_fn = agg_fn
        # Latch so the precomputed-pseudobulk reduction warning fires once, not per step.
        self._warned_agg_fn_precomputed = False
        self.paired_alignment = paired_alignment
        self.vocab = vocab
        self.gene_embeddings_path = gene_embeddings_path
        self.gene_embeddings_freeze = gene_embeddings_freeze
        self.verbose = verbose
        self.weight_mvc = weight_mvc
        self.weight_contrastive = weight_contrastive
        self.weight_mmd = weight_mmd
        self.weight_paired = weight_paired
        self.weight_agg = weight_agg
        self.weight_reconstruction = weight_reconstruction
        self.weight_dat = weight_dat
        # Auxiliary losses to compute + log for monitoring only (no gradient). Logged
        # under train/loss_<name>_monitor when the corresponding loss is disabled.
        self.monitor_losses = set(monitor_losses or ())

        # Contrastive Domain Discrepancy (CAN) configuration
        self.cdd = cdd
        self.weight_cdd = weight_cdd
        self.cdd_class_column = cdd_class_column
        self.cdd_min_class_count = cdd_min_class_count
        self.cdd_exclude_class_codes = tuple(cdd_exclude_class_codes or ())
        self.cdd_infer_labels = cdd_infer_labels
        self.cdd_cluster_iters = cdd_cluster_iters
        self.cdd_cluster_ambiguity = cdd_cluster_ambiguity
        self.cdd_cluster_min_size = cdd_cluster_min_size
        self.cdd_relabel_known = cdd_relabel_known
        self.cdd_cluster_source_fallback = cdd_cluster_source_fallback
        # Target-label bank (allocated lazily via init_target_bank when
        # cdd_infer_labels is enabled and the datamodule is available).
        self._target_bank_ready = False
        # Handover schedule, refreshed each step by the LightningModule. Defaults are
        # the post-ramp values so a module driven without the schedule (tests, direct
        # use) behaves as if warmup is already over.
        self._cdd_w = weight_cdd
        self._mmd_w = weight_mmd

        self.n_input_bins = n_input_bins
        # if self.input_emb_style not in ["category", "continuous", "scaling"]:
        #     raise ValueError(
        #         f"input_emb_style should be one of category, continuous, scaling, "
        #         f"got {input_emb_style}"
        #     )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "mine":
            self.value_encoder = MyContinuousValueEncoder(
                d_model=d_model, pcpt=not use_generative_training, dropout=dropout
            )
        elif input_emb_style == "theirs":
            self.value_encoder = TheirContinuousValueEncoder(d_model, dropout)

        elif input_emb_style == "continuous":
            self.value_encoder = TheirContinuousValueEncoder(d_model, dropout)

        self.do_dat = do_dat
        self.dat_columns = dat_columns,
        self.no_invert_dat = no_invert_dat
        self.do_mvc = do_mvc
        self.criterion_conditions = nn.CrossEntropyLoss()
        self.criterion = criterion

        mvc_decoder_d_in = d_model
        expr_decoder_d_in = d_model
        # Conditions are taken into account only in the decoder. Only the encoded
        # subset contributes to the decoder input width.
        if self._use_condition_encoders and self.where_condition != "none":
            mvc_decoder_d_in = d_model * (len(self.encoded_conditions) + 1)
            if where_condition == "end":
                expr_decoder_d_in = d_model * (len(self.encoded_conditions) + 1)

        # Conditions are encoded as separate embeddings. Build encoders only for the
        # encoded subset; label-only conditions (used by DAT/CDD) get no encoder.
        if conditions:
            self.condition_encoders = nn.ModuleDict({})
            for cond_name in self.encoded_conditions:
                self.condition_encoders[cond_name] = ConditionEncoder(
                    self.conditions[cond_name], d_model
                )
            # Check condition encoders
            # if self.verbose:
            #     print("Condition Encoders Inspection:")
            #     print(f"Conditions: {self.conditions}")
            #     print(f"Encoders: {self.condition_encoders}")

            if do_dat:
                self.grad_reverse_discriminators = nn.ModuleDict({})
                for cond_name, cond_num in self.conditions.items():
                    if dat_columns and cond_name not in dat_columns:
                        continue
                    print(f"Using {cond_name} for DAT!")
                    # For modality DAT, only distinguish bulk vs pseudobulk (binary)
                    n_cls = 2 if cond_name == "modality" else cond_num
                    self.grad_reverse_discriminators[cond_name] = (
                        AdversarialDiscriminator(
                            d_model,
                            n_cls=n_cls,
                            scale=dat_scale,
                            no_invert_dat=no_invert_dat,
                        )
                    )
                if len(self.grad_reverse_discriminators) == 0:
                    raise ValueError(
                        f"do_dat=True but no discriminators were created. "
                        f"dat_columns={dat_columns}, available conditions={list(self.conditions.keys())}"
                    )
        if use_generative_training:
            if gen_method == "orig":
                encoder_layers = CFLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                    norm_type=norm_type,
                )
                self.transformer_encoder = CFGenerator(
                    encoder_layer=encoder_layers,
                    num_layers=nlayers,
                    grad_checkpoint=grad_checkpoint,
                    attn_impl=attn_impl,
                )
                self.flag_encoder = nn.Embedding(2, d_model)
                self.encoder = self._build_gene_encoder(
                    ntoken=ntoken,
                    d_model=d_model,
                    padding_idx=pad_token_id,
                )
            elif gen_method == "theirs":
                encoder_layers = CFLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                    norm_type=norm_type,
                )
                self.transformer_encoder = CFGenerator(
                    encoder_layer=encoder_layers,
                    num_layers=nlayers,
                    grad_checkpoint=grad_checkpoint,
                    attn_impl=attn_impl,
                )
                self.generative_flag = nn.Parameter(torch.randn(d_model))
                self.gene_encoder = self._build_gene_encoder(
                    ntoken=ntoken,
                    d_model=d_model,
                    padding_idx=pad_token_id,
                )
            elif gen_method == "mine":
                self.transformer_encoder = RefactoredCFGenerator(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_hid,
                    dropout=dropout,
                    norm_scheme=self.norm_scheme,
                    num_layers=nlayers,
                    grad_checkpoint=grad_checkpoint,
                    attn_impl=attn_impl,
                )
                self.generative_flag = nn.Parameter(torch.randn(d_model))
                self.gene_encoder = self._build_gene_encoder(
                    ntoken=ntoken,
                    d_model=d_model,
                    padding_idx=pad_token_id,
                )

            elif gen_method == "quick":
                self.transformer_encoder = QuickCFGenerator(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_hid,
                    dropout=dropout,
                    norm_scheme=self.norm_scheme,
                    num_layers=nlayers,
                    grad_checkpoint=grad_checkpoint,
                )
                self.generative_flag = nn.Parameter(torch.randn(d_model))
                self.gene_encoder = self._build_gene_encoder(
                    ntoken=ntoken,
                    d_model=d_model,
                    padding_idx=pad_token_id,
                )
        else:
            # norm_first maps "pre" → True, everything else → False
            _norm_first = self.norm_scheme == "pre"
            encoder_layers = TransformerEncoderLayer(
                d_model,
                nhead,
                d_hid,
                dropout,
                batch_first=True,
                norm_first=_norm_first,
                activation=activation,
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.gene_encoder = self._build_gene_encoder(
                ntoken=ntoken,
                d_model=d_model,
                padding_idx=pad_token_id,
            )

        self.decoder = ExprDecoder(
            d_in=expr_decoder_d_in,
            d_model=d_model,
            out_dim=out_dim,
            normalise_bins=normalise_bins,
        )

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_in=mvc_decoder_d_in,
                d_model=d_model,
                arch_style=mvc_decoder_style,
                out_dim=out_dim,
                normalise_bins=normalise_bins,
            )

        # if their_init_weights:
        #     self.init_weights()

    @staticmethod
    def reset_all_weights(model):
        """Recursively reset all parameters in the model"""

        @torch.no_grad()
        def init_weights(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        model.apply(init_weights)

    def _build_gene_encoder(
        self,
        ntoken: int,
        d_model: int,
        padding_idx: Optional[int] = None,
    ) -> nn.Module:
        return GeneEncoder(
            num_embeddings=ntoken,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            vocab=self.vocab,
            weights_file=self.gene_embeddings_path,
            freeze=self.gene_embeddings_freeze,
            verbose=self.verbose,
        )

    # def init_weights(self) -> None:
    #     """Initializes the weights of the gene embedding layer."""
    #     initrange = 0.1
    #     self.gene_encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def embed(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Tensor] = None,
    ) -> Tensor:
        """Embeds gene IDs into dense vectors. Used if self.use_generative_training

        Args:
            src (Tensor): Input gene token IDs of shape (batch, seq_len).
            values (Tensor): Input expression values of shape (batch, seq_len).
            src_key_padding_mask (Tensor): Padding mask of shape (batch, seq_len).
            conditions (Optional[Tensor], optional): Dictionary of condition tensors. Defaults to None.

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, embsize).
        """
        if hasattr(self, "gene_encoder"):
            gene_embs = self.gene_encoder(src)
        elif hasattr(self, "encoder"):
            gene_embs = self.encoder(src)
        value_embs = self.value_encoder(values)
        # RAFA: modified handling of the where_condition == "begin"
        total_embs = gene_embs + value_embs

        if self.where_condition == "begin" and self._use_condition_encoders and conditions is not None:
            # We sum the condition embeddings along the first dimension to generate a unified condition token
            condition_emb = torch.stack(
                [
                    self.condition_encoders[cond_name](conditions[cond_name])
                    for cond_name in self.condition_encoders
                ],
                dim=1,
            ).sum(dim=1)
            # Adapt padding mask
            src_key_padding_mask = torch.cat(
                [
                    src_key_padding_mask[:, :1],
                    torch.zeros(
                        src_key_padding_mask.size(0),
                        1,
                        dtype=src_key_padding_mask.dtype,
                        device=src_key_padding_mask.device,
                    ),
                    src_key_padding_mask[:, 1:],
                ],
                dim=1,
            )
            # Insert condition embedding after CLS token (assumed at position 0) and before the rest of the gene embeddings
            total_embs = torch.cat(
                [
                    total_embs[:, 0, :].unsqueeze(1),
                    condition_emb.unsqueeze(1),
                    total_embs[:, 1:, :],
                ],
                dim=1,
            )

        output = self.transformer_encoder(
            pcpt_total_embs=total_embs,
            gen_total_embs=None,
            src_key_padding_mask=src_key_padding_mask,
            attn_mask=torch.zeros((total_embs.shape[0], total_embs.shape[0])),
        )

        return output

    def encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        check_conditions: bool = True
    ) -> Tensor:
        """Encodes gene IDs and expression values into contextual embeddings. This method is used during perceptual (non-generative) training.

        Args:
            src (Tensor): Input gene token IDs of shape (batch, seq_len).
            values (Tensor): Input expression values of shape (batch, seq_len).
            src_key_padding_mask (Tensor): Padding mask of shape (batch, seq_len).
            conditions (Optional[Dict], optional): Dictionary of condition tensors. Defaults to None.
            check_conditions (bool): Whether to use function _check_condition_labels

        Returns:
            Tensor: The output of the Transformer encoder, of shape (batch, seq_len, embsize).
        """
        if check_conditions:
            self._check_condition_labels(conditions)

        if hasattr(self, "gene_encoder"):
            src_embs = self.gene_encoder(src)
        elif hasattr(self, "encoder"):
            src_embs = self.encoder(src)
        self.cur_gene_token_embs = src_embs

        values = self.value_encoder(values)

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src_embs * values
        else:
            total_embs = src_embs + values

        # RAFA: Match generative begin-conditioning behavior
        if self.where_condition == "begin" and self._use_condition_encoders and conditions is not None:
            # We sum the condition embeddings along the first dimension to generate a unified condition token
            condition_emb = torch.stack(
                [
                    self.condition_encoders[cond_name](conditions[cond_name])
                    for cond_name in self.condition_encoders
                ],
                dim=1,
            ).sum(dim=1)
            # Adapt padding mask (pads one position in the left, as CLS is assumed non-padding)
            src_key_padding_mask = torch.cat(
                [
                    src_key_padding_mask[:, :1],
                    torch.zeros(
                        src_key_padding_mask.size(0),
                        1,
                        dtype=src_key_padding_mask.dtype,
                        device=src_key_padding_mask.device,
                    ),
                    src_key_padding_mask[:, 1:],
                ],
                dim=1,
            )
            total_embs = torch.cat(
                [
                    total_embs[:, 0, :].unsqueeze(1),
                    condition_emb.unsqueeze(1),
                    total_embs[:, 1:, :],
                ],
                dim=1,
            )

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        return output

    def encode_cls(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
    ) -> Tensor:
        """CLS embedding for a batch of rows, dispatching on the training mode.

        Returns the same tensor the CDD/MMD blocks read in ``forward``
        (``embeddings[:, 0, :]``), so embeddings produced here are directly
        comparable with the ones the losses see. Used by the clustering refresh
        passes, which need every bulk row encoded under a single model state.

        Unlike ``embed_for_finetune``, conditions are passed through — the modality
        token is what distinguishes bulk from pseudobulk, so dropping it would make
        the two domains encode identically.
        """
        if self.use_generative_training:
            output, _ = self.embed(
                src, values, src_key_padding_mask, conditions=conditions
            )
        else:
            output = self.encode(
                src, values, src_key_padding_mask, conditions=conditions
            )
        return output[:, 0, :]

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        src_key_padding_mask: Tensor,
        attn_mask: Tensor,
        conditions: Optional[Dict] = None,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Processes inputs through the generative transformer model, adding conditions as input tokens.

        Args:
            pcpt_genes (Tensor): Gene tokens for the perceptual (context) part.
            pcpt_values (Tensor): Expression values for the perceptual part.
            pcpt_key_padding_mask (Tensor): Padding mask for the perceptual part.
            gen_genes (Tensor): Gene tokens for the generative (target) part.
            gen_key_padding_mask (Tensor): Padding mask for the generative part.
            conditions (Optional[Dict], optional): Conditional labels. Defaults to None.
            input_cell_emb (Optional[Tensor], optional): Pre-computed cell embeddings to inject. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformer output for the perceptual and generative parts, respectively.
        """
        self._check_condition_labels(conditions)

        if hasattr(self, "gene_encoder"):
            pcpt_token_embs = self.gene_encoder(pcpt_genes)
        elif hasattr(self, "encoder"):
            pcpt_token_embs = self.encoder(pcpt_genes)
        pcpt_values_embs = self.value_encoder(pcpt_values)
        pcpt_total_embs = pcpt_token_embs + pcpt_values_embs

        if self.where_condition == "begin" and self._use_condition_encoders:
            # We sum the condition embeddings along the first dimension to generate a unified condition token
            condition_emb = torch.stack(
                [
                    self.condition_encoders[cond_name](conditions[cond_name])
                    for cond_name in self.condition_encoders
                ],
                dim=1,
            ).sum(dim=1)
            # Adapt padding mask (pads one position in the left, as CLS is assumed non-padding)
            pcpt_key_padding_mask = torch.cat(
                [
                    pcpt_key_padding_mask[:, :1],
                    torch.zeros(
                        pcpt_key_padding_mask.size(0),
                        1,
                        dtype=pcpt_key_padding_mask.dtype,
                        device=pcpt_key_padding_mask.device,
                    ),
                    pcpt_key_padding_mask[:, 1:],
                ],
                dim=1,
            )
            # Insert condition embedding after CLS token (assumed at position 0) and before the rest of the gene embeddings
            pcpt_total_embs = torch.cat(
                [
                    pcpt_total_embs[:, 0, :].unsqueeze(1),
                    condition_emb.unsqueeze(1),
                    pcpt_total_embs[:, 1:, :],
                ],
                dim=1,
            )

        assert self.input_emb_style != "scaling"

        if gen_genes is not None:
            if hasattr(self, "gene_encoder"):
                gen_token_embs = self.gene_encoder(gen_genes)
            elif hasattr(self, "encoder"):
                gen_token_embs = self.encoder(gen_genes)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs], dim=1
            )
            if hasattr(self, "generative_flag"):
                gen_flags = self.generative_flag
            elif hasattr(self, "flag_encoder"):
                gen_flags = self.flag_encoder(
                    torch.tensor(1).to(pcpt_values.device)
                ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)
            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        # Modify masks in case condition token is being fed into the model
        if self.where_condition == "begin" and self._use_condition_encoders:
            assert gen_total_embs is not None
            src_key_padding_mask = torch.cat(
                [
                    pcpt_key_padding_mask,
                    gen_key_padding_mask,
                ],
                dim=1,
            )

            if attn_mask is not None:
                total_len = pcpt_total_embs.shape[1] + gen_key_padding_mask.shape[1]
                gen_len = gen_key_padding_mask.shape[1]
                if attn_mask.shape != (total_len, total_len):
                    attn_mask = torch.zeros(
                        (total_len, total_len),
                        dtype=torch.bool,
                        device=pcpt_total_embs.device,
                    )
                    attn_mask[:, -gen_len:] = True
                    attn_mask.diagonal().fill_(False)

        # Transformer Encoder forward pass
        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs=pcpt_total_embs,
            gen_total_embs=gen_total_embs,
            src_key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
        )
        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Optional[Tensor] = None
    ) -> Tensor:
        """Extracts cell embeddings from the transformer's output layer.

        Args:
            layer_output (Tensor): The transformer output tensor of shape (batch, seq_len, embsize).
            weights (Optional[Tensor], optional): A tensor of weights of shape (batch, seq_len) used only when `self.cell_emb_style` is "w-pool".

        Returns:
            Tensor: The extracted cell embeddings of shape (batch, embsize).
        """
        # Remove condition token if present (assumed at position 1)
        if self.where_condition == "begin" and self._use_condition_encoders:
            layer_output = torch.cat(
                [
                    layer_output[:, :1, :],   # CLS
                    layer_output[:, 2:, :],   # skip condition token
                ],
                dim=1,
            )
            if weights is not None:
                weights = torch.cat(
                    [
                        weights[:, :1],
                        weights[:, 2:],
                    ],
                    dim=1,
                )
        else:
            layer_output = layer_output

        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:  # self.cell_emb_style == "w-pool"
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_condition_labels(
        self, condition_labels: Optional[Tensor] = None
    ) -> None:
        """Validates that condition labels are provided if and only if conditions are defined for the model."""
        # Use 'is not None' instead of bool() to avoid graph breaks in torch.compile
        assert (self.conditions is not None) == (condition_labels is not None)

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        conditions: Optional[Tensor] = None,
        condition_emb: Optional[Tensor] = None,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """Extends the output dictionary with cell embeddings and optional predictions.

        Args:
            output (Mapping[str, Tensor]): The dictionary of current outputs.
            transformer_output (Tensor): The raw output from the transformer encoder.
            condition_emb (Optional[Tensor], optional): The embedding for conditional variables. Defaults to None.
            do_sample (bool, optional): If True, samples from the Bernoulli distribution for zero-inflation. Defaults to False.

        Returns:
            Mapping[str, Tensor]: The extended output dictionary.
        """
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        # Also append with embeddings
        output["embeddings"] = transformer_output

        if self.do_mvc:
            if self._use_condition_encoders and self.where_condition != "none":
                mvc_input_emb = torch.cat(
                    [cell_emb, condition_emb.view(condition_emb.shape[0], -1)], dim=1
                )
            else:
                mvc_input_emb = cell_emb

            mvc_output = self.mvc_decoder(mvc_input_emb, self.cur_gene_token_embs)
            output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)

        if self.do_dat:
            if self.conditions:
                modality = conditions.get("modality")
                output["condition_output"] = {}
                # print(f"Performing DAT on: {list(self.grad_reverse_discriminators.keys())}")
                for cond_name, discriminator in self.grad_reverse_discriminators.items():
                    if cond_name == "modality" and modality is not None:
                        # Only apply DAT between real bulk (0) and pseudobulk (2)
                        # print("Masking modality DAT...")
                        bulk_pb_mask = (modality == 0) | (modality == 2)
                        if bulk_pb_mask.sum() == 0:
                            continue
                        filtered_emb = cell_emb[bulk_pb_mask]
                        output["condition_output"][cond_name] = discriminator(filtered_emb)
                    else:
                        # All samples for other conditions
                        output["condition_output"][cond_name] = discriminator(cell_emb)

        return output

    def _prepare_generative_input(self, tensors: dict[str, torch.Tensor], noise:float=None):
        """Prepares tensors for the generative forward pass."""
        # Apply noising schedule
        if noise is not None:
            # print("Noising generative...")
            tensors = utils.apply_log1p_noise_to_branch_inputs(tensors, "gen", noise)

        pcpt_gene = tensors["pcpt_gene"]
        pcpt_expr = tensors["pcpt_expr"]
        pcpt_key_padding_mask = tensors["pcpt_key_padding_mask"]
        gen_gene = tensors["gen_gene"]
        gen_expr_target = tensors["gen_expr_target"]
        gen_key_padding_mask = tensors["gen_key_padding_mask"]
        attn_mask = tensors["attn_mask"]

        src_key_padding_mask = torch.cat(
            [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
        )

        return (
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_expr_target,
            gen_key_padding_mask,
            src_key_padding_mask,
            attn_mask,
        )

    def _prepare_perceptual_input(self, tensors: dict[str, torch.Tensor], noise:float=None):
        """Prepares tensors for the perceptual forward pass."""
        # Apply noising schedule
        if noise is not None:
            # print("Noising perceptual...")
            tensors = utils.apply_log1p_noise_to_branch_inputs(tensors, "pcpt", noise)

        input_gene_ids = tensors["gene"]
        input_values = tensors["masked_expr"]
        src_key_padding_mask = tensors["gene_key_padding_mask"]
        target_values = tensors["expr"]

        return input_gene_ids, input_values, src_key_padding_mask, target_values

    def forward(  # tensors is the data_dict from collator
        self,
        tensors: dict[str, torch.Tensor],
        use_cell_embedding: bool = False,
        noise: float = None,
        apply_dat: bool = True,
        skip_unified_losses: bool = False, # Necessary for validation loss
    ) -> Mapping[str, Tensor]:
        """Main forward pass that dispatches to generative or perceptual mode.

        This wrapper determines the training mode based on the `use_generative_training` attribute,
        computes the primary predictions and losses, and adds auxiliary losses from MVC, DAT, and a generative consistency loss.

        Args:
            tensors (dict[str, torch.Tensor]): A dictionary of input tensors from the dataloader.
            use_cell_embedding (bool, optional): If True, a consistency loss is added by feeding the cell embedding back into the generative forward pass. Defaults to False.

        Returns:
            Mapping[str, Tensor]: A dictionary of losses for training.
        """
        loss_dict = {}
        conditions_batch = tensors["conditions"] if self.conditions else None
        if self.use_generative_training:
            (
                pcpt_gene,
                pcpt_expr,
                pcpt_key_padding_mask,
                gen_gene,
                gen_expr_target,
                gen_key_padding_mask,
                src_key_padding_mask,
                attn_mask,
            ) = self._prepare_generative_input(tensors, noise=noise)
            output_dict = self.generative_forward(
                pcpt_gene,
                pcpt_expr,
                pcpt_key_padding_mask,
                gen_gene,
                gen_key_padding_mask,
                src_key_padding_mask,
                attn_mask,
                conditions=conditions_batch,
            )

            gen_expr_preds = output_dict["gen_preds"]
            # Do not take into account sc_for_pb for the reconstruction loss
            keep_samples = (
                tensors["is_sc_for_pb"] == 0
                if "is_sc_for_pb" in tensors
                else torch.ones(
                    gen_expr_target.shape[0],
                    dtype=torch.bool,
                    device=gen_expr_target.device,
                )
            )
            positions_to_match = (~gen_key_padding_mask) & keep_samples.unsqueeze(1)

            # print(f"Gene expression output dimensions: {gen_expr_preds.shape}")
            loss_expr = self.criterion(
                gen_expr_preds, gen_expr_target, positions_to_match
            )
            loss = self.weight_reconstruction * loss_expr
            loss_dict["loss_expr"] = loss_expr * self.weight_reconstruction

            if self.do_mvc:
                mvc_preds_for_gen = output_dict["mvc_output"][:, pcpt_gene.shape[1] :]
                loss_mvc = self.criterion(
                    mvc_preds_for_gen, gen_expr_target, positions_to_match
                )
                loss = loss + self.weight_reconstruction * self.weight_mvc * loss_mvc
                loss_dict["loss_mvc"] = loss_mvc * self.weight_mvc * self.weight_reconstruction

            # The cell-embedding-conditioned pass contributes to `loss` only through
            # the `use_cell_embedding` factor below, which CancerFoundation keeps at
            # False for the first 1000 steps. Running it anyway costs a full extra
            # transformer forward whose activations are retained until backward, so
            # skip it outright when its weight is zero. Numerically identical.
            if use_cell_embedding:
                previous_cell_embs = output_dict["cell_emb"].detach()
                preds = self.generative_forward(
                    pcpt_gene,
                    pcpt_expr,
                    pcpt_key_padding_mask,
                    gen_gene,
                    gen_key_padding_mask,
                    src_key_padding_mask,
                    attn_mask,
                    input_cell_emb=previous_cell_embs,
                    conditions=conditions_batch,
                )["gen_preds"]
                loss_gen = self.criterion(preds, gen_expr_target, positions_to_match)
                loss = loss + use_cell_embedding * self.weight_reconstruction * loss_gen
                loss_dict["loss_gen"] = loss_gen * self.weight_reconstruction
            # else: no `loss_gen` key. Logging a 0 here would draw a flat line in
            # W&B that looks like a converged loss rather than an absent one.

        # Perceptual training
        else:
            input_gene_ids, input_values, src_key_padding_mask, target_values = (
                self._prepare_perceptual_input(tensors, noise=noise)
            )
            output_dict = self.perceptual_forward(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                conditions=conditions_batch,
            )

            output_values = output_dict["mlm_output"]
            # Do not take into account sc_for_pb for the reconstruction loss
            keep_samples = (
                tensors["is_sc_for_pb"] == 0
                if "is_sc_for_pb" in tensors
                else torch.ones(
                    target_values.shape[0],
                    dtype=torch.bool,
                    device=target_values.device,
                )
            )
            positions_to_match = (
                ~src_key_padding_mask
                & (target_values != -2)
                & keep_samples.unsqueeze(1)
            )
            loss_expr = self.criterion(
                output_values, target_values, positions_to_match
            )
            loss = self.weight_reconstruction * loss_expr
            loss_dict["loss_expr"] = loss_expr * self.weight_reconstruction

            if self.do_mvc:
                loss_mvc = self.criterion(
                    output_dict["mvc_output"], target_values, positions_to_match
                )
                loss = loss + self.weight_reconstruction * self.weight_mvc * loss_mvc
                loss_dict["loss_mvc"] = loss_mvc * self.weight_reconstruction* self.weight_mvc

        # Resolve paired-batch flag before any loss blocks that branch on it
        is_paired_batch = tensors.get("is_paired_batch", False)
        if isinstance(is_paired_batch, torch.Tensor):
            is_paired_batch = bool(is_paired_batch.item())

        # Return directly if denoising task
        if noise is not None:
            return loss_dict["loss_expr"]

        # Domain adversarial training
        if self.do_dat and apply_dat and not skip_unified_losses:
            if self.conditions:
                modality = conditions_batch.get("modality")
                for condition in self.grad_reverse_discriminators:
                    cond_preds = output_dict["condition_output"].get(condition)
                    if cond_preds is None:
                        continue

                    if condition == "modality" and modality is not None:
                        # Only bulk (0) and pseudobulk (2)
                        bulk_pb_mask = (modality == 0) | (modality == 2)
                        cond_labels = modality[bulk_pb_mask].squeeze()
                        # Remap 0→0, 2→1 for binary classification
                        cond_labels = (cond_labels == 2).long()
                    else:
                        cond_labels = conditions_batch[condition].squeeze()

                    condition_loss = self.criterion_conditions(cond_preds, cond_labels)

                    # Confidence: mean probability assigned to the correct class
                    loss_dict[condition + "_confidence"] = (
                        cond_preds
                        .softmax(dim=-1)
                        .gather(dim=1, index=cond_labels.unsqueeze(1))
                        .squeeze(1)
                        .float()
                        .mean()
                    )

                    loss += self.weight_dat * condition_loss / len(self.grad_reverse_discriminators)
                    loss_dict["condition_" + condition] = condition_loss.detach() / len(self.grad_reverse_discriminators) * self.weight_dat

        # Contrastive loss: if enabled, it brings the pseudobulk and real bulk samples closer together in the embedding space.
        # Skipped for paired batches: the all-positive InfoNCE treats every (bulk[i], pb[j])
        # pair as a positive, which contradicts the 1-to-1 pairing enforced by the paired
        # alignment loss. VICReg collapse-prevention is still applied via modality_contrastive_loss
        # in non-paired batches; paired batches rely solely on the paired alignment loss for signal.
        want_contrastive = self.contrastive or "contrastive" in self.monitor_losses
        if want_contrastive and not is_paired_batch and not skip_unified_losses:
            embeddings = output_dict["embeddings"]
            modalities = tensors["conditions"]["modality"]
            assert len(embeddings) == len(
                modalities
            ), "Embeddings and modalities tensors must have the same batch size"

            # sc_for_pb cells have modality=1 but are pseudobulk constituents, not free
            # SC cells. Including them as hard negatives opposes the aggregation consistency
            # loss, which pulls pseudobulk toward their mean. Mask them out with sentinel -1
            # so they fall outside every == k selector inside modality_contrastive_loss.
            if "is_sc_for_pb" in tensors and tensors["is_sc_for_pb"].any():
                modalities = modalities.clone()
                modalities[tensors["is_sc_for_pb"] == 1] = -1

            loss_contrastive = self.modality_contrastive_loss(embeddings, modalities)
            if self.contrastive:
                loss = loss + self.weight_contrastive * loss_contrastive
                loss_dict["loss_contrastive"] = loss_contrastive.detach() * self.weight_contrastive
            else:
                loss_dict["loss_contrastive_monitor"] = loss_contrastive.detach() * self.weight_contrastive

        # MMD alignment loss: distribution-level matching of the real-bulk and
        # pseudobulk embedding marginals via a mixture-of-RBF-kernels MMD. A
        # non-adversarial complement/alternative to DAT that matches the whole
        # distribution rather than just the means. Marginal-only, so — unlike the
        # contrastive loss — it does not conflict with 1-to-1 pairing and runs on
        # paired and unpaired batches alike.
        # When CDD is enabled, the MMD weight follows the handover schedule set by the
        # LightningModule (full during CDD warmup to pull the clouds together, then
        # decaying to cdd_mmd_final_weight as CDD ramps in). Without CDD it is the
        # plain configured weight.
        want_mmd = self.mmd or "mmd" in self.monitor_losses
        if want_mmd and not skip_unified_losses:
            embeddings = output_dict["embeddings"]
            modalities = tensors["conditions"]["modality"]
            assert len(embeddings) == len(
                modalities
            ), "Embeddings and modalities tensors must have the same batch size"

            loss_mmd = self.modality_mmd_loss(embeddings, modalities)
            if self.mmd:
                w_mmd = self._mmd_w if self.cdd else self.weight_mmd
                loss = loss + w_mmd * loss_mmd
                loss_dict["loss_mmd"] = loss_mmd.detach() * w_mmd
            else:
                loss_dict["loss_mmd_monitor"] = loss_mmd.detach() * self.weight_mmd

        # Contrastive Domain Discrepancy (CAN, Kang et al. CVPR 2019): a
        # class-conditional MMD that pulls same-tissue source(pseudobulk)/target(bulk)
        # embeddings together and pushes different-tissue apart. Unlike the marginal
        # MMD above, it preserves tissue structure. Supervised on the tissue label;
        # when label inference is enabled the bulk labels are the K-means pseudo-labels.
        # Skipped on paired batches (those use the 1-to-1 paired alignment loss instead)
        # and during the warmup window, where _cdd_w is 0 and MMD is doing the marginal
        # alignment that makes the first clustering meaningful.
        # want_cdd fires either when CDD is actively training (post-warmup) or when it is
        # requested for monitoring only. Monitoring additionally requires the tissue class
        # column to be present in the batch conditions; without it no label is available and
        # the block is skipped cleanly (no crash).
        want_cdd = (self.cdd and self._cdd_w > 0) or "cdd" in self.monitor_losses
        if (
            want_cdd
            and not is_paired_batch
            and not skip_unified_losses
            and self.cdd_class_column in tensors["conditions"]
        ):
            embeddings = output_dict["embeddings"]
            modalities = tensors["conditions"]["modality"]
            emb = embeddings[:, 0, :]  # CLS token, matches modality_mmd_loss

            # Keep the bank warm between clustering events. The authoritative fill is
            # refresh_bulk_bank() (one model state, full coverage); this streaming write
            # is a cheap top-up from embeddings we already have. Only relevant when CDD is
            # actually enabled — the bank is never allocated in monitor-only mode.
            if self.cdd and self.cdd_infer_labels and self._target_bank_ready and self.training:
                self.update_target_bank(emb, modalities, tensors.get("row_index"))

            tissue = self._cdd_target_labels(tensors)
            src_m = modalities == 2  # pseudobulk = source
            tgt_m = modalities == 0  # real bulk   = target
            loss_cdd = self._cdd_loss(
                emb[src_m], tissue[src_m], emb[tgt_m], tissue[tgt_m],
                min_class_count=self.cdd_min_class_count,
                exclude_class_codes=self.cdd_exclude_class_codes,
            )
            if self.cdd and self._cdd_w > 0:
                loss = loss + self._cdd_w * loss_cdd
                loss_dict["loss_cdd"] = loss_cdd.detach() * self._cdd_w
            else:
                loss_dict["loss_cdd_monitor"] = loss_cdd.detach() * self.weight_cdd

        # Paired alignment loss: MSE between matched bulk–pseudobulk–SC_mean CLS embeddings
        want_paired = self.paired_alignment or "paired" in self.monitor_losses
        if want_paired and is_paired_batch and not skip_unified_losses:
            if self.verbose:
                print("Applying paired alignment loss!")
            modality = tensors["conditions"]["modality"]
            cell_emb = output_dict["cell_emb"]
            sc_pb_idx = tensors["sample_pseudobulk_index"]
            bulk_mask = modality == 0
            pb_mask   = modality == 2
            sc_mask   = (modality == 1) & (sc_pb_idx >= 0)
            if bulk_mask.any() and pb_mask.any():
                pb_bulk_local_idx = sc_pb_idx[pb_mask]
                bulk_embs = cell_emb[bulk_mask][pb_bulk_local_idx]
                pb_embs   = cell_emb[pb_mask]
                n_pb, d_model = pb_embs.shape

                loss_paired = self._paired_alignment_loss(bulk_embs, pb_embs)

                # Per-pair mean SC embedding via scatter_add (fully differentiable)
                if sc_mask.any():
                    sc_embs = cell_emb[sc_mask]
                    sc_local = sc_pb_idx[sc_mask]
                    idx_exp = sc_local.unsqueeze(1).expand(-1, d_model)
                    sc_sums = torch.zeros(n_pb, d_model, device=cell_emb.device, dtype=cell_emb.dtype)
                    sc_sums = sc_sums.scatter_add(0, idx_exp, sc_embs)
                    sc_counts = torch.zeros(n_pb, device=cell_emb.device, dtype=cell_emb.dtype)
                    sc_counts = sc_counts.scatter_add(
                        0, sc_local,
                        torch.ones(sc_embs.shape[0], device=cell_emb.device, dtype=cell_emb.dtype),
                    )
                    has_sc = sc_counts > 0
                    if has_sc.any():
                        if self.verbose:
                            print("Using paired SC samples!")
                        sc_means = sc_sums[has_sc] / sc_counts[has_sc].unsqueeze(1)
                        loss_paired_sc = (
                            self._paired_alignment_loss(sc_means, bulk_embs[has_sc])
                            + self._paired_alignment_loss(sc_means, pb_embs[has_sc])
                        )
                        loss_paired = loss_paired + loss_paired_sc
                        # loss_dict["paired_alignment_loss_sc"] = loss_paired_sc.detach() * self.weight_paired

                if self.paired_alignment:
                    loss = loss + self.weight_paired * loss_paired
                    loss_dict["paired_alignment_loss"] = loss_paired.detach() * self.weight_paired
                else:
                    loss_dict["paired_alignment_loss_monitor"] = loss_paired.detach() * self.weight_paired

        # Aggregation consistency loss: skip for paired batches (SC cells are unrelated to the PBs)
        want_agg = self.aggregation or "aggregation" in self.monitor_losses
        if want_agg and not is_paired_batch and not skip_unified_losses:
            # CLS embeddings, not the full (B, L, D) transformer output: token position
            # l is a different gene in every row, because _sample_or_truncate permutes
            # each sample's genes independently and only keep_first_n_tokens=1 (the CLS)
            # is held fixed. Comparing positions 1..L-1 compares embeddings of unrelated
            # genes and of padding. The paired-alignment loss above uses cell_emb for
            # the same reason.
            embeddings = output_dict["cell_emb"]
            assert len(embeddings) == len(
                tensors["is_sc_for_pb"]
            ), "Embeddings and input dictionaries must have the same batch size"

            # Map each sc_for_pb cell to its pseudobulk local index (0…n_pb-1).
            sc_assignment: dict = {}
            for idx in range(len(embeddings)):
                if tensors["is_sc_for_pb"][idx] == 1:
                    pb_local_idx = tensors["sample_pseudobulk_index"][idx].item()
                    sc_assignment.setdefault(pb_local_idx, []).append(idx)

            # Resolve local PB indices to global positions in the embeddings tensor.
            # sample_pseudobulk_index stores 0-based local PB indices, but PBs occupy
            # positions n_bulk+n_sc … n_bulk+n_sc+n_pb-1 in the unified batch — they
            # cannot be used directly to index into embeddings without this lookup.
            pb_global_pos = (tensors["conditions"]["modality"] == 2).nonzero(as_tuple=True)[0]

            # A precomputed pseudobulk aggregates all N of its cells, but only
            # n_sc_per_pseudobulk of them are in the batch, so a sum would scale with
            # the sample size rather than with the pseudobulk. Mean is the only
            # reduction that means the same thing on both sides.
            agg_fn = self.agg_fn
            if bool(tensors.get("is_precomputed_pb_batch", False)) and agg_fn != "mean":
                if not self._warned_agg_fn_precomputed:
                    self._warned_agg_fn_precomputed = True
                    print(
                        f"[WARNING] agg_fn={self.agg_fn!r} with precomputed pseudobulks: "
                        f"using 'mean' instead. A sum over the sampled cells scales with "
                        f"n_sc_per_pseudobulk, not with the pseudobulk it is matched to."
                    )
                agg_fn = "mean"

            loss_agg = torch.tensor(0.0, device=loss.device)
            for pb_local_idx, sc_indices in sc_assignment.items():
                pb_embedding = embeddings[pb_global_pos[pb_local_idx]]
                sc_embeddings = embeddings[sc_indices]
                if agg_fn == "mean":
                    sc_embedding_agg = sc_embeddings.mean(dim=0)
                elif agg_fn == "sum":
                    sc_embedding_agg = sc_embeddings.sum(dim=0)
                else:
                    raise ValueError(f"Unknown agg_fn: {agg_fn}")
                loss_agg = loss_agg + F.mse_loss(pb_embedding, sc_embedding_agg)

            if sc_assignment:
                if self.aggregation:
                    loss = loss + self.weight_agg * loss_agg
                    loss_dict["loss_agg"] = loss_agg.detach() * self.weight_agg
                else:
                    loss_dict["loss_agg_monitor"] = loss_agg.detach() * self.weight_agg

        loss_dict["total_loss"] = loss
        return loss_dict

    def modality_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        modalities: torch.Tensor,
        temperature: float = 0.1,
        use_cls_token: bool = True,
        lambda_: float = 25.0, # invariance
        mu: float = 25.0, # variance
        nu: float = 1.0, # covariance
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Wrapper for bulk-pseudobulk alignment.

        Modality convention (from BulkSCCollator):
            0 = real bulk      → anchors in fwd, positives in bwd
            1 = single-cell    → hard negatives in both directions
            2 = pseudobulk     → positives in fwd, anchors in bwd

        embeddings : (B, L, D) or (B, D)
        modalities  : (B,)
        """
        # print("Performing VICReg with Symmetric Multi-Positive InfoNCE")
        assert embeddings.size(0) == modalities.size(0), \
            "Embeddings and modalities tensors must have the same batch size"

        # Reduce sequence dim → (B, D)
        if embeddings.dim() == 3:
            embeddings = embeddings[:, 0, :] if use_cls_token else embeddings.mean(dim=1)

        assert embeddings.dim() == 2, f"Expected (B, D), got {embeddings.shape}"
        assert modalities.dim() == 1, f"Expected (B,), got {modalities.shape}"

        # Split by modality
        bulk_emb = embeddings[modalities == 0]   # (N0, D)
        sc_emb   = embeddings[modalities == 1]   # (N1, D)
        pb_emb   = embeddings[modalities == 2]   # (N2, D)

        if bulk_emb.size(0) == 0:
            raise ValueError("No modality-0 (bulk) samples in batch.")
        if pb_emb.size(0) == 0:
            raise ValueError("No modality-2 (pseudobulk) samples in batch.")

        sc_emb = sc_emb if sc_emb.size(0) > 0 else None

        inv_loss = self.symmetric_multipositive_infonce(
            bulk_emb, pb_emb, sc_emb, temperature=temperature
        )
        var_loss, cov_loss = self.vicreg_loss(bulk_emb, pb_emb, eps=eps)

        return lambda_ * inv_loss + mu * var_loss + nu * cov_loss


    def symmetric_multipositive_infonce(
        self,
        bulk_emb: torch.Tensor,
        pb_emb: torch.Tensor,
        sc_emb: torch.Tensor | None = None,
        temperature: float = 0.1,
    ) -> torch.Tensor:

        has_sc = sc_emb is not None and sc_emb.size(0) > 0
        if not has_sc:
            raise ValueError(
                "SC samples are required as hard negatives for the symmetric "
                "multi-positive InfoNCE loss. Ensure sc_ratio > 0 in your sampler."
            )

        bulk_norm = F.normalize(bulk_emb, dim=1)
        pb_norm   = F.normalize(pb_emb,   dim=1)
        sc_norm   = F.normalize(sc_emb,   dim=1)

        def _infonce(anchors, positives, negatives):
            pos_sim   = (anchors @ positives.T) / temperature # (Na, Np)
            neg_sim   = (anchors @ negatives.T) / temperature # (Na, Nn)
            log_denom = torch.logsumexp(
                torch.cat([pos_sim, neg_sim], dim=1), dim=1 # (Na,)
            )
            log_prob_pos = pos_sim - log_denom.unsqueeze(1) # (Na, Np)
            return -log_prob_pos.mean()

        loss_fwd = _infonce(bulk_norm, pb_norm,   sc_norm)   # bulk → pb, SC as neg
        loss_bwd = _infonce(pb_norm,   bulk_norm, sc_norm)   # pb   → bulk, SC as neg

        return (loss_fwd + loss_bwd) / 2


    def vicreg_loss(
        self,
        bulk_emb: torch.Tensor,
        pb_emb: torch.Tensor,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        VICReg variance and covariance terms for collapse prevention.
        Operates on unnormalized embeddings independently per modality.

        Returns
        -------
        var_loss : variance term (scalar)
        cov_loss : covariance term (scalar)
        """
        # Hinge function on the standard deviation of the embeddings along the batch dimension
        def _variance(z: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(z.var(dim=0) + eps)
            return torch.mean(F.relu(1 - std))

        # Covariance to enforce decorrelation between embedidng positions
        def _covariance(z: torch.Tensor) -> torch.Tensor:
            n, d = z.shape
            if n < 2:
                return torch.tensor(0.0, device=z.device)
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (n - 1)
            off_diag = cov ** 2
            off_diag.fill_diagonal_(0)
            return off_diag.sum() / d

        var_loss = _variance(bulk_emb) + _variance(pb_emb)
        cov_loss = _covariance(bulk_emb) + _covariance(pb_emb)

        return var_loss, cov_loss

    def _paired_alignment_loss(
        self,
        bulk_embs: torch.Tensor,
        pb_embs:   torch.Tensor,
    ) -> torch.Tensor:
        """MSE between CLS embeddings of matched bulk–pseudobulk pairs."""
        return F.mse_loss(pb_embs, bulk_embs)

    def modality_mmd_loss(
        self,
        embeddings: torch.Tensor,
        modalities: torch.Tensor,
        use_cls_token: bool = True,
    ) -> torch.Tensor:
        """Maximum Mean Discrepancy between real-bulk and pseudobulk embeddings.

        Distribution-level alignment of the pseudobulk (modality 2) and real-bulk
        (modality 0) marginals in cell-embedding space — a stable, non-adversarial
        alternative/complement to the DAT discriminator. Single-cell cells
        (modality 1, incl. pseudobulk constituents) are ignored: MMD aligns only
        the two bulk-level clouds.

        References
        ----------
        Gretton et al., "A Kernel Two-Sample Test", JMLR 2012 (MMD estimator).
        Long et al., "Learning Transferable Features with Deep Adaptation
        Networks", ICML 2015 (multi-kernel MMD for domain adaptation).
        Shaham et al., "Removal of batch effects using distribution-matching
        residual networks", Bioinformatics 2017 (MMD for batch-effect removal).

        Modality convention (from BulkSCCollator): 0 = real bulk, 2 = pseudobulk.

        embeddings : (B, L, D) or (B, D)
        modalities  : (B,)
        """
        # Reduce sequence dim → (B, D)
        if embeddings.dim() == 3:
            embeddings = embeddings[:, 0, :] if use_cls_token else embeddings.mean(dim=1)

        assert embeddings.dim() == 2, f"Expected (B, D), got {embeddings.shape}"
        assert modalities.dim() == 1, f"Expected (B,), got {modalities.shape}"

        bulk_emb = embeddings[modalities == 0]   # (N0, D)
        pb_emb   = embeddings[modalities == 2]   # (N2, D)

        # MMD is undefined without both clouds; skip cleanly (keeps the graph intact).
        if bulk_emb.size(0) < 1 or pb_emb.size(0) < 1:
            return embeddings.new_zeros(())

        return self._mmd_rbf(bulk_emb, pb_emb)

    @staticmethod
    def _pairwise_sq_dists(xy: torch.Tensor) -> torch.Tensor:
        """Pairwise squared Euclidean distances via the Gram identity.

        Avoids ``torch.cdist``, whose p=2 backward divides by the distance and so
        returns NaN gradients at zero distance — and the diagonal of
        ``cdist(xy, xy)`` is always exactly zero (as is any pair of coincident
        embeddings). ``‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·bᵀ`` never takes a square root,
        keeping gradients finite; clamp at 0 to absorb float round-off.
        """
        sq = (xy * xy).sum(dim=1, keepdim=True)          # (N, 1)
        dist = sq + sq.t() - 2.0 * (xy @ xy.t())
        return dist.clamp_min(0.0)

    @staticmethod
    def _mmd_rbf(
        x: torch.Tensor,
        y: torch.Tensor,
        scales: tuple = (0.25, 0.5, 1.0, 2.0, 4.0),
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Biased squared-MMD with a mixture of RBF kernels.

        Bandwidths are set from the median pairwise squared distance of the pooled
        sample (the median heuristic) times a range of ``scales``. Averaging over
        several bandwidths makes the estimate far less sensitive to any single
        bandwidth than a fixed-sigma kernel — the standard choice for MMD-based
        domain adaptation. The biased V-statistic (diagonal included) is used: it
        is non-negative for a valid kernel and smoother than the unbiased form,
        which matters at the small per-modality batch sizes here.
        """
        xy = torch.cat([x, y], dim=0)
        dist = TransformerModule._pairwise_sq_dists(xy)  # (N+M, N+M) squared distances
        n = x.size(0)

        # Median heuristic, detached: the bandwidth is a kernel hyperparameter,
        # not a quantity to backpropagate through.
        with torch.no_grad():
            pos = dist[dist > 0]
            median = pos.median() if pos.numel() > 0 else dist.new_tensor(1.0)

        xx = dist[:n, :n]
        yy = dist[n:, n:]
        xy_block = dist[:n, n:]

        mmd = x.new_zeros(())
        for s in scales:
            gamma = 1.0 / (2.0 * (s * median).clamp_min(eps))
            k_xx = torch.exp(-gamma * xx).mean()
            k_yy = torch.exp(-gamma * yy).mean()
            k_xy = torch.exp(-gamma * xy_block).mean()
            mmd = mmd + (k_xx + k_yy - 2.0 * k_xy)

        return mmd / len(scales)

    # ------------------------------------------------------------------
    # Contrastive Domain Discrepancy (CAN, Kang et al. CVPR 2019)
    # ------------------------------------------------------------------
    def _cdd_target_labels(self, tensors) -> torch.Tensor:
        """Tissue label per batch sample used as the CDD class.

        Normally the raw ``cdd_class_column`` condition. When target-label
        inference is active, the bulk (target) rows are overridden with the current
        spherical-K-means pseudo-labels, which recover both the ``"unknown"`` bulk
        and the bulk-only tissues that have no single-cell counterpart. Rows still
        unassigned carry ``-1`` and are dropped by ``_cdd_loss``.
        """
        labels = tensors["conditions"][self.cdd_class_column].long().reshape(-1)
        if not (self.cdd_infer_labels and self._target_bank_ready):
            return labels
        row_index = tensors.get("row_index")
        if row_index is None:
            return labels
        modalities = tensors["conditions"]["modality"]
        labels = labels.clone()
        bulk_pos = (modalities == 0).nonzero(as_tuple=True)[0]
        if bulk_pos.numel() == 0:
            return labels
        rows = row_index[bulk_pos].long()
        local = self.row_to_bulk_local[rows]
        pseudo = torch.full_like(rows, -1)
        valid = local >= 0
        pseudo[valid] = self.pseudo_label[local[valid]]
        labels[bulk_pos] = pseudo
        return labels

    def _cdd_loss(
        self,
        source_emb: torch.Tensor,
        source_labels: torch.Tensor,
        target_emb: torch.Tensor,
        target_labels: torch.Tensor,
        scales: tuple = (0.25, 0.5, 1.0, 2.0, 4.0),
        min_class_count: int = 2,
        exclude_class_codes: tuple = (),
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Contrastive Domain Discrepancy: intra-class MMD minus inter-class MMD.

        Class-conditional, multi-bandwidth-RBF MMD between source (pseudobulk)
        and target (bulk) embeddings, keyed on the tissue class. Only classes
        present in BOTH domains with >= ``min_class_count`` samples and not in
        ``exclude_class_codes`` (and non-negative) are used. Biased kernel means
        (diagonal included) with a single shared bandwidth (median heuristic over
        the pooled sample), matching :meth:`_mmd_rbf` conventions.
        """
        ref = source_emb if source_emb.numel() else target_emb
        zero = ref.new_zeros(())
        if source_emb.size(0) == 0 or target_emb.size(0) == 0:
            return zero

        exclude = {int(c) for c in exclude_class_codes}
        s_lab = source_labels.long()
        t_lab = target_labels.long()
        s_cls, s_cnt = torch.unique(s_lab, return_counts=True)
        t_cls, t_cnt = torch.unique(t_lab, return_counts=True)
        s_ok = {int(c): int(n) for c, n in zip(s_cls.tolist(), s_cnt.tolist())}
        t_ok = {int(c): int(n) for c, n in zip(t_cls.tolist(), t_cnt.tolist())}
        valid = sorted(
            c for c in s_ok
            if c >= 0 and c not in exclude and c in t_ok
            and s_ok[c] >= min_class_count and t_ok[c] >= min_class_count
        )
        M = len(valid)
        if M == 0:
            return zero

        # One-hot class indicators over the valid classes only.
        Us = source_emb.new_zeros(source_emb.size(0), M)
        Ut = target_emb.new_zeros(target_emb.size(0), M)
        for i, c in enumerate(valid):
            Us[:, i] = (s_lab == c).to(Us.dtype)
            Ut[:, i] = (t_lab == c).to(Ut.dtype)

        # Shared multi-RBF kernel over the pooled source+target sample.
        xy = torch.cat([source_emb, target_emb], dim=0)
        dist_all = self._pairwise_sq_dists(xy)
        with torch.no_grad():
            pos = dist_all[dist_all > 0]
            median = pos.median() if pos.numel() > 0 else dist_all.new_tensor(1.0)
        ns_ = source_emb.size(0)
        Dss, Dtt, Dst = dist_all[:ns_, :ns_], dist_all[ns_:, ns_:], dist_all[:ns_, ns_:]
        Kss = source_emb.new_zeros(Dss.shape)
        Ktt = target_emb.new_zeros(Dtt.shape)
        Kst = source_emb.new_zeros(Dst.shape)
        for s in scales:
            gamma = 1.0 / (2.0 * (s * median).clamp_min(eps))
            Kss = Kss + torch.exp(-gamma * Dss)
            Ktt = Ktt + torch.exp(-gamma * Dtt)
            Kst = Kst + torch.exp(-gamma * Dst)
        Kss, Ktt, Kst = Kss / len(scales), Ktt / len(scales), Kst / len(scales)

        ns = Us.sum(0)  # (M,) per-class source counts
        nt = Ut.sum(0)  # (M,) per-class target counts
        E1 = (Us * (Kss @ Us)).sum(0) / (ns * ns).clamp_min(eps)          # (M,)
        E2 = (Ut * (Ktt @ Ut)).sum(0) / (nt * nt).clamp_min(eps)          # (M,)
        E3 = (Us.t() @ Kst @ Ut) / (ns.unsqueeze(1).clamp_min(eps) * nt.unsqueeze(0).clamp_min(eps))  # (M,M)
        Dmat = E1.unsqueeze(1) + E2.unsqueeze(0) - 2.0 * E3               # (M,M)

        intra = torch.diagonal(Dmat).mean()
        if M >= 2:
            inter = (Dmat.sum() - torch.diagonal(Dmat).sum()) / (M * (M - 1))
            return intra - inter
        return intra

    # --- Target-label inference (semi-supervised spherical K-means) ---
    def init_target_bank(self, n_bulk, known_label, is_unknown, row_to_bulk_local, class_ids):
        """Allocate the bulk-embedding memory bank and label buffers.

        Indexed by compact bulk-local id (0..n_bulk-1). ``known_label`` holds the
        ground-truth tissue code (or -1 for ``"unknown"``/excluded); ``pseudo_label``
        starts equal to it and is refreshed by :meth:`recluster`.
        """
        dev = next(self.parameters()).device
        n_cls = len(list(class_ids))
        self.register_buffer("bank_emb", torch.zeros(n_bulk, self.d_model, device=dev), persistent=False)
        self.register_buffer("bank_filled", torch.zeros(n_bulk, dtype=torch.bool, device=dev), persistent=False)
        self.register_buffer("known_label", torch.as_tensor(known_label, dtype=torch.long, device=dev))
        self.register_buffer("is_unknown", torch.as_tensor(is_unknown, dtype=torch.bool, device=dev))
        self.register_buffer("pseudo_label", torch.as_tensor(known_label, dtype=torch.long, device=dev).clone())
        self.register_buffer("row_to_bulk_local", torch.as_tensor(row_to_bulk_local, dtype=torch.long, device=dev))
        self.register_buffer("cdd_class_ids", torch.as_tensor(list(class_ids), dtype=torch.long, device=dev))
        # Source (pseudobulk) class means, indexed like cdd_class_ids. Recomputed from
        # scratch by refresh_source_means() at each clustering event rather than
        # accumulated: pseudobulks are re-aggregated from randomly drawn cells every
        # step, so an EMA would smear over both a changing PB composition and a
        # changing model state — and the offset correction in recluster() is only
        # sound if src_mean and bank_emb come from the same model state.
        self.register_buffer("src_mean", torch.zeros(n_cls, self.d_model, device=dev), persistent=False)
        self.register_buffer("src_count", torch.zeros(n_cls, dtype=torch.long, device=dev), persistent=False)
        self._target_bank_ready = True

    @torch.no_grad()
    def set_source_means(self, means: Tensor, counts: Tensor):
        """Overwrite the source class means (see :meth:`init_target_bank`).

        ``means``/``counts`` are indexed like ``cdd_class_ids``; a zero count marks a
        class the source could not supply this event.
        """
        if not self._target_bank_ready:
            return
        self.src_mean.copy_(means.to(self.src_mean.dtype))
        self.src_count.copy_(counts.to(self.src_count.dtype))

    @torch.no_grad()
    def update_target_bank(self, emb, modalities, row_index):
        """Store the latest (normalized, detached) bulk embeddings in the bank."""
        if row_index is None or not self._target_bank_ready:
            return
        bulk_pos = (modalities == 0).nonzero(as_tuple=True)[0]
        if bulk_pos.numel() == 0:
            return
        rows = row_index[bulk_pos].long()
        local = self.row_to_bulk_local[rows]
        valid = local >= 0
        if not valid.any():
            return
        feats = F.normalize(emb[bulk_pos][valid].detach().float(), dim=1)
        self.bank_emb[local[valid]] = feats.to(self.bank_emb.dtype)
        self.bank_filled[local[valid]] = True

    @torch.no_grad()
    def recluster(self) -> dict:
        """Refresh bulk pseudo-labels via bulk-anchored spherical K-means.

        Centroids are initialized and anchored by the known-labeled bulk (same
        domain as the points being clustered), then refined over the unknown
        bulk. Ambiguous (cosine-dist > D0) and small (< N0) clusters are dropped.
        Known labels are kept fixed unless ``cdd_relabel_known``.

        Classes with no known bulk anchors — tissues only the single-cell data knows
        about — are unreachable from the bulk alone. When ``cdd_cluster_source_fallback``
        is set they are seeded from the source (pseudobulk) class mean instead, see
        below.
        """
        if not self._target_bank_ready:
            return {}
        filled = self.bank_filled
        if int(filled.sum()) == 0:
            return {}
        N0 = self.cdd_cluster_min_size

        # 1. Init centroids, preferring known-labeled bulk anchors (bulk-structure
        #    seeding: same domain as the points being clustered, so most reliable).
        #
        #    Source-seeded fallback: a class with no bulk anchors can only be reached
        #    from the pseudobulk side. But assignment below is a single argmax over all
        #    centroids, and modality dominates tissue — a raw pseudobulk centroid sits
        #    in the source cloud, so every bulk row would be nearer every bulk-anchored
        #    centroid and the fallback would win nothing. Translating it by the
        #    domain-mean offset removes that first-order modality shift, keeping the
        #    class's tissue structure while placing it where the bulk cloud lives.
        #    Sound only because refresh_source_means() recomputes src_mean in the same
        #    pass as bank_emb, so both frames come from one model state.
        use_fallback = self.cdd_cluster_source_fallback and int(self.src_count.sum()) > 0
        if use_fallback:
            src_valid = self.src_count > 0
            src_global = F.normalize(self.src_mean[src_valid], dim=1).mean(0)
            bulk_global = self.bank_emb[filled].mean(0)

        centroids, used_classes, n_source_seeded = [], [], 0
        for k, c in enumerate(self.cdd_class_ids.tolist()):
            anchor_mask = filled & (self.known_label == c)
            if int(anchor_mask.sum()) >= N0:
                centroids.append(F.normalize(self.bank_emb[anchor_mask].mean(0), dim=0))
                used_classes.append(c)
            elif use_fallback and int(self.src_count[k]) > 0:
                shifted = F.normalize(self.src_mean[k], dim=0) - src_global + bulk_global
                centroids.append(F.normalize(shifted, dim=0))
                used_classes.append(c)
                n_source_seeded += 1
        if len(used_classes) == 0:
            return {"cdd_classes_used": 0}
        C = torch.stack(centroids, 0)  # (K, D)

        pool_mask = filled.clone() if self.cdd_relabel_known else (filled & self.is_unknown)
        pool_idx = pool_mask.nonzero(as_tuple=True)[0]

        # 2. Spherical K-means (cosine), anchored by the known bulk each iter.
        #    A source-seeded class has no bulk anchors, so its centroid is defined
        #    purely by whatever it wins; once it wins any bulk rows it becomes
        #    bulk-anchored and self-corrects. If it wins nothing, its parts are empty
        #    and the mean would be NaN — hold the previous centroid instead.
        for _ in range(self.cdd_cluster_iters):
            assign = None
            if pool_idx.numel() > 0:
                pool_feats = F.normalize(self.bank_emb[pool_idx], dim=1)
                assign = (pool_feats @ C.t()).argmax(1)
            new_C = []
            for k, c in enumerate(used_classes):
                anchor_mask = filled & (self.known_label == c)
                parts = []
                if int(anchor_mask.sum()) > 0:
                    parts.append(F.normalize(self.bank_emb[anchor_mask], dim=1))
                if assign is not None:
                    sel = pool_idx[assign == k]
                    if sel.numel() > 0:
                        parts.append(F.normalize(self.bank_emb[sel], dim=1))
                if parts:
                    new_C.append(F.normalize(torch.cat(parts, 0).mean(0), dim=0))
                else:
                    new_C.append(C[k])
            new_C = torch.stack(new_C, 0)
            if torch.allclose(new_C, C, atol=1e-4):
                C = new_C
                break
            C = new_C

        # 3. Final assignment of the target rows + purity filters.
        used_t = torch.tensor(used_classes, device=C.device)
        target_mask = filled.clone() if self.cdd_relabel_known else (filled & self.is_unknown)
        new_pseudo = self.pseudo_label.clone()
        n_assigned = 0
        n_orphans = 0
        if target_mask.any():
            idx = target_mask.nonzero(as_tuple=True)[0]
            feats = F.normalize(self.bank_emb[idx], dim=1)
            best_sim, best_k = (feats @ C.t()).max(1)
            # Orphans: rows too far from EVERY centroid to be claimed. A high count
            # means the target rows do not sit near any class the source knows about
            # — typically because the domains have not been pulled together yet.
            ambiguous = (1.0 - best_sim) > self.cdd_cluster_ambiguity
            n_orphans = int(ambiguous.sum())
            assigned = torch.where(
                ambiguous,
                torch.full_like(best_k, -1),
                used_t[best_k],
            )
            for c in used_classes:  # min-size filter
                if int((assigned == c).sum()) < N0:
                    assigned[assigned == c] = -1
            new_pseudo[idx] = assigned
            n_assigned = int((assigned >= 0).sum())

        # Known labels stay fixed unless explicitly relabeling.
        if not self.cdd_relabel_known:
            known_mask = self.known_label >= 0
            new_pseudo[known_mask] = self.known_label[known_mask]
        self.pseudo_label.copy_(new_pseudo)
        return {
            "cdd_classes_used": len(used_classes),
            "cdd_source_seeded": n_source_seeded,
            "cdd_unknown_assigned": n_assigned,
            "cdd_orphans": n_orphans,
            "cdd_bank_filled": int(filled.sum()),
        }

    def training_step(self, batch, batch_idx):
        """Performs a single training step (for PyTorch Lightning).

        Args:
            batch: The batch of data from the DataLoader.
            batch_idx: The index of the batch.

        Returns:
            The total loss for the batch.
        """
        loss_dict = self(batch, use_cell_embedding=False)
        return loss_dict["total_loss"]

    def generative_forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        src_key_padding_mask: Tensor,
        attn_mask: Tensor,
        conditions: Optional[Dict] = None,
        do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """Forward pass for the generative training mode.

        Args:
            pcpt_genes (Tensor): Token IDs of the perceptual part.
            pcpt_values (Tensor): Token values of the perceptual part.
            pcpt_key_padding_mask (Tensor): Mask for pcpt_genes.
            gen_genes (Tensor): Token IDs of the generative part.
            gen_key_padding_mask (Tensor): Mask for gen_genes.
            conditions (Optional[Dict]): Dictionary of condition tensors.
            do_sample (bool): If True, samples from Bernoulli for zero predictions.
            input_cell_emb (Optional[Tensor]): Pre-computed cell embeddings to inject.

        Returns:
            Mapping[str, Tensor]: A dictionary containing predictions and other outputs.
        """
        pcpt_output, gen_output = self.transformer_generate(
            pcpt_genes,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_genes,
            gen_key_padding_mask,
            src_key_padding_mask,
            attn_mask,
            conditions,
            input_cell_emb=input_cell_emb,
        )

        if self.where_condition == "begin" and self._use_condition_encoders:
            # Condition token was inserted after CLS in the perceptual stream.
            pcpt_output_for_decoder = torch.cat(
                [
                    pcpt_output[:, :1, :],   # CLS
                    pcpt_output[:, 2:, :],   # drop condition token
                ],
                dim=1,
            )
        else:
            pcpt_output_for_decoder = pcpt_output
            
        transformer_output = (
            pcpt_output_for_decoder
            if gen_output is None
            else torch.cat([pcpt_output, gen_output], dim=1)
        )

        condition_emb = None
        decoder_input = transformer_output
        if self._use_condition_encoders:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](conditions[cond_name])
                    for cond_name in self.condition_encoders
                ],
                dim=1,
            ).view(transformer_output.shape[0], -1)

            if self.where_condition == "end":
                decoder_input = torch.cat(
                    [
                        transformer_output,
                        condition_emb.unsqueeze(1).repeat(
                            1, transformer_output.shape[1], 1
                        ),
                    ],
                    dim=2,
                )
        output = {}
        decoder_output = self.decoder(decoder_input)

        full_preds = decoder_output["pred"]

        pcpt_out_len = pcpt_output.shape[1]

        output["pcpt_preds"] = full_preds[:, :pcpt_out_len]
        output["gen_preds"] = full_preds[:, pcpt_out_len:]

        output = self._extend_output(
            output,
            transformer_output,
            condition_emb=condition_emb,
            do_sample=do_sample,
            conditions=conditions,
        )

        return output

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        conditions: Optional[Dict] = None,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """Forward pass for the perceptual (MLM-style) training mode.

        Args:
            src (Tensor): Input token IDs, shape [batch_size, seq_len].
            values (Tensor): Input expression values (with masking), shape [batch_size, seq_len].
            src_key_padding_mask (Tensor): Mask for src, shape [batch_size, seq_len].
            conditions (Optional[Dict], optional): Dictionary of condition tensors. Defaults to None.
            do_sample (bool, optional): If True, samples from Bernoulli for zero predictions. Defaults to False.

        Returns:
            Mapping[str, Tensor]: A dictionary containing MLM predictions ('mlm_output'), cell embeddings ('cell_emb'), and other optional outputs.
        """
        condition_emb = None
        if self._use_condition_encoders:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](conditions[cond_name]).unsqueeze(1)
                    for cond_name in self.condition_encoders
                ],
                dim=1,
            )
        transformer_output = self.encode(src, values, src_key_padding_mask, conditions)

        if self.where_condition == "begin":
            # RAFA: Get rid of the condition token once the it has been encoded through self-attention
            if self._use_condition_encoders and condition_emb is not None:
                # num_conditions = condition_emb.shape[1]
                decoder_input = torch.cat(
                    [
                        transformer_output[:, :1, :],                    # keep CLS
                        transformer_output[:, 2 :, :], # skip condition tokens
                    ],
                    dim=1,
                )
            else:
                decoder_input = transformer_output

        elif self.where_condition == "end":
            if self._use_condition_encoders:
                decoder_input = torch.cat(
                    [
                        condition_emb.view(condition_emb.shape[0], -1)
                        .unsqueeze(1)
                        .repeat(1, transformer_output.shape[1], 1),
                        transformer_output,
                    ],
                    dim=2,
                )
            else:
                decoder_input = transformer_output

        output = {}
        mlm_output = self.decoder(decoder_input)
        output["mlm_output"] = mlm_output["pred"]

        output = self._extend_output(
            output,
            transformer_output,
            condition_emb=(condition_emb if self._use_condition_encoders else None),
            do_sample=do_sample,
            conditions=conditions
        )

        return output


class GeneEncoder(nn.Module):
    """Embeds integer gene IDs, optionally using pretrained ESM-based weights.

    When ``weights_file`` or ``weights`` is provided, the encoder loads a gene
    embedding table, aligns it to the model vocabulary if available, and
    projects it to ``embedding_dim`` when the imported width differs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        vocab: Optional[Dict[str, int]] = None,
        weights: Optional[Tensor] = None,
        weights_file: Optional[Union[str, os.PathLike, Path]] = None,
        freeze: bool = False,
        verbose: bool = False,
    ):
        """Initializes the gene encoder.

        Args:
            num_embeddings (int): The total number of unique genes.
            embedding_dim (int): The dimensionality of the gene embeddings.
            padding_idx (Optional[int], optional): The index of the padding token. Defaults to None.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.verbose = verbose
        self.enc_norm = nn.LayerNorm(embedding_dim)

        pretrained_weights = self._load_pretrained_weights(
            num_embeddings=num_embeddings,
            vocab=vocab,
            weights=weights,
            weights_file=weights_file,
        )

        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights,
                freeze=freeze,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim, padding_idx=padding_idx
            )

        source_dim = self.embedding.weight.shape[1]

        # Project down to embedding_dim required by the model
        self.projection = (
            nn.Identity()
            if source_dim == embedding_dim
            else nn.Linear(source_dim, embedding_dim, bias=False)
        )

    def _load_pretrained_weights(
        self,
        num_embeddings: int,
        vocab: Optional[Dict[str, int]],
        weights: Optional[Tensor],
        weights_file: Optional[Union[str, os.PathLike, Path]],
    ) -> Optional[Tensor]:
        if weights is None and weights_file is None:
            return None

        print("Using pretrained embeddings!")
        if weights is not None:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        else:
            if weights_file is None:
                raise ValueError("weights_file cannot be None when no weights tensor is provided.")
            print(f"Weight file: {weights_file}")

            weights_path = Path(weights_file)
            weight_frame = pd.read_parquet(weights_path)
            if vocab is not None and len(vocab) > 0:
                ordered_tokens: list[Optional[str]] = [None] * num_embeddings
                # Example vocab: {"<cls>": 0, "<pad>": 1, "geneA": 2}
                for token, idx in vocab.items():
                    if 0 <= idx < num_embeddings:
                        ordered_tokens[idx] = token

                source_dim = weight_frame.shape[1]
                weight_tensor = torch.zeros(
                    (num_embeddings, source_dim), dtype=torch.float32
                )
                # Compute mean vector for tokens in the weight frame to use for <cls> token
                mean_vector = torch.tensor(
                    weight_frame.to_numpy(dtype=np.float32).mean(axis=0),
                    dtype=torch.float32,
                )
                # Keep track of genes not present in pretrained (non-protein-coding genes)
                non_encoded_counter = 0
                scale = 1.0 / math.sqrt(source_dim)
                for idx, token in enumerate(ordered_tokens):
                    if token is None:
                        continue
                    if token in weight_frame.index:
                        # if self.verbose:
                        #     print(f"{token} in pretrained weights!")
                        row = np.asarray(weight_frame.loc[token], dtype=np.float32)
                        weight_tensor[idx] = torch.tensor(row, dtype=torch.float32)
                    elif token == "<cls>":
                        weight_tensor[idx] = mean_vector
                    # The <pad> token embedding is hardcoded to zero
                    elif token == "<pad>":
                        weight_tensor[idx] = torch.zeros(source_dim, dtype=torch.float32)
                    else:
                        # random init for unsuported genes
                        weight_tensor[idx] = torch.empty(source_dim).uniform_(-scale, scale)
                        non_encoded_counter += 1
                print(f"Missing tokens initialized randomly: {non_encoded_counter} / {num_embeddings}\n\n")
                    
            else:
                weight_tensor = torch.tensor(
                    weight_frame.to_numpy(dtype=np.float32), dtype=torch.float32
                )

        if weight_tensor.shape[0] != num_embeddings:
            raise ValueError(
                f"Pretrained gene embedding table has {weight_tensor.shape[0]} rows, "
                f"but model vocabulary expects {num_embeddings}."
            )

        return weight_tensor

    def forward(self, x: Tensor) -> Tensor:
        """Encodes a batch of gene IDs.

        Args:
            x (Tensor): A tensor of gene IDs of shape (batch, seq_len).

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, embsize).
        """
        x = self.embedding(x)
        x = self.projection(x)
        x = self.enc_norm(x)
        return x


class TheirContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class MyContinuousValueEncoder(nn.Module):
    """
    Embeds continuous gene expression values using a small feed-forward network.
    This version is rewritten to be compatible with `torch.compile`.
    """

    def __init__(
        self, d_model: int, pcpt: bool, dropout: float = 0.1, max_value: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        if pcpt:
            # This embedding is used for values of -1, indicating a masked gene.
            self.masked_expression_embedding = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Embeds a batch of continuous expression values using a compiler-friendly,
        functional approach.

        Args:
            x (Tensor): A tensor of expression values of shape (batch, seq_len).
                        Values >= 0 are gene expressions.
                        Value == -1 indicates a masked gene.

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, d_model).
        """
        # --- 1. Perform the main computation on the entire tensor ---
        # Instead of filtering with a mask, we compute embeddings for all positions.
        # This creates a static computation graph. Negative values are clamped to 0
        # so they produce a consistent "zero embedding" that can be discarded later.
        x_processed = x.unsqueeze(-1).clamp(max=self.max_value)
        expression_embs_full = self.dropout(
            self.norm(self.linear2(self.activation(self.linear1(x_processed))))
        )
        # --- 2. Create boolean masks for selection ---
        # The masks must be unsqueezed to broadcast correctly with the embeddings tensor.
        is_expression = (x >= 0).unsqueeze(-1)

        # --- 3. Select the final output using torch.where ---
        # This replaces the conditional logic and in-place assignments. It's a single,
        # functional operation that the compiler can heavily optimize.

        # Start with a default tensor of zeros for any values that don't meet other conditions.
        output_embeddings = torch.zeros_like(expression_embs_full)

        # If the masked embedding is defined, use it for positions where x is -1.
        if hasattr(self, "masked_expression_embedding"):
            is_masked = (x == -1).unsqueeze(-1)
            output_embeddings = torch.where(
                is_masked, self.masked_expression_embedding, output_embeddings
            )

        # Finally, for all positions where x represents a valid gene expression,
        # select the values we computed in step 1. Otherwise, keep the existing value
        # (which will be either the masked embedding or zero).
        output_embeddings = torch.where(
            is_expression, expression_embs_full, output_embeddings
        )
        return output_embeddings


class CategoricalValueEncoder(nn.Module):
    """Embeds discretized (binned) gene expression values using an embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """Initializes the categorical value encoder.

        Args:
            num_embeddings (int): The number of discrete bins for expression values.
            embedding_dim (int): The dimensionality of the value embeddings.
            padding_idx (Optional[int], optional): The index of the padding value. Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Embeds a batch of binned expression values.

        Args:
            x (Tensor): A tensor of binned values of shape (batch, seq_len).

        Returns:
            Tensor: The resulting embeddings of shape (batch, seq_len, embsize).
        """
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ConditionEncoder(nn.Module):
    """Embeds integer condition IDs, allowing the model to be conditioned on categorical metadata, such as cell type or sequencing technology."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """Initializes the condition encoder.

        Args:
            num_embeddings (int): Number of unique conditions.
            embedding_dim (int): Dimension of the condition embeddings.
            padding_idx (Optional[int], optional): Padding index for the embeddings. Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the input conditions.

        Args:
            x (Tensor): Input tensor of shape (batch).

        Returns:
            Tensor: Encoded tensor of shape (batch, embedding_dim).
        """
        x = self.embedding(x)  # (batch, embedding_dim)
        x = self.enc_norm(x)
        return x

# Could also include dropout and layer normalization?
class ExprDecoder(nn.Module):
    """Decodes contextual gene embeddings to predict gene expression values.

    Takes the output of the transformer encoder for a gene as input and passes it through a feed-forward network to predict its expression value.
    If configured, it can also predict the probability of the gene's expression being zero.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        out_dim: int,
        normalise_bins: bool,
        # Added for ZINB prediction
        zinb: bool = False,
    ):
        """Initialises the gene expression value decoder.

        Args:
            d_model (int): Dimension of the input embeddings.
            out_dim (int): Dimension of the output gene expression values.
            explicit_zero_prob (bool, optional): Whether to predict the probability of zero expression. Defaults to False.
            conditions (Optional[Dict], optional): Configuration for additional conditions, used to adjust the input dimension. Defaults to None.
        """
        super().__init__()
        self.normalise_bins = normalise_bins
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass for the gene expression value decoder.

        Args:
            x (Tensor): Input tensor from the transformer encoder of shape (batch, seq_len, d_in).

        Returns:
            Dict[str, Tensor]: A dictionary containing the predicted expression values ('pred') and,
            if applicable, the zero expression probabilities ('zero_probs').
        """
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)
        if self.normalise_bins:
            pred_value = torch.sigmoid(pred_value)
        return dict(pred=pred_value)

class MVCDecoder(nn.Module):
    """Decoder for the Masked Value Prediction for Cell embeddings (MVC) task."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        out_dim: int,
        normalise_bins: bool,
        arch_style: str = "inner product",
        query_activation: Type[nn.Module] = nn.Sigmoid,
        hidden_activation: Type[nn.Module] = nn.PReLU,
    ) -> None:
        """Initialises the MVC decoder.

        Args:
            d_model (int): Dimension of the model embeddings.
            out_dim (int): Dimension of the output gene expression values.
            arch_style (str, optional): Architecture style of the decoder ('inner product', 'concat query', 'sum query'). Defaults to "inner product".
            query_activation (Type[nn.Module], optional): Activation function for the query vectors. Defaults to nn.Sigmoid.
            hidden_activation (Type[nn.Module], optional): Activation function for hidden layers. Defaults to nn.PReLU.
            explicit_zero_prob (bool, optional): Whether to predict the probability of zero expression. Defaults to False.
            conditions (Optional[Dict], optional): Configuration for additional conditions. Defaults to None.

        Raises:
            ValueError: If an unknown architecture style is provided.
        """
        super().__init__()
        # Inner products don't work with output dimension > 1
        self.out_dim = out_dim
        self.normalise_bins = normalise_bins
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if self.out_dim > 1:
                self.fc1 = nn.Linear(1, out_dim)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_in + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, out_dim)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, out_dim)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass for the MVC decoder.

        Args:
            cell_emb (Tensor): Cell embedding tensor of shape (batch, d_in).
            gene_embs (Tensor): Gene embedding tensor of shape (batch, seq_len, d_model).

        Raises:
            NotImplementedError: If explicit zero probability is enabled with 'concat query' or 'sum query' architecture.

        Returns:
            Dict[str, Tensor]: A dictionary of predicted gene expression values ('pred') and optional zero probabilities ('zero_probs').
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if self.out_dim > 1:
                pred_value = pred_value.unsqueeze(2)
                pred_value = self.fc1(pred_value)
            if self.normalise_bins:
                pred_value = torch.sigmoid(pred_value)
            return dict(pred=pred_value)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            return dict(pred=self.fc2(h).squeeze(2))  # (batch, seq_len)
        else:  # self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """A discriminator for Domain Adversarial Training (DAT).

    This network takes cell embeddings as input and tries to predict their domain (e.g., batch of origin). It is used with a gradient reversal layer to encourage the main model to learn domain-invariant representations.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        scale: float,
        no_invert_dat: bool,
        nlayers: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """Initializes the AdversarialDiscriminator.

        Args:
            d_model (int): Dimension of the input embeddings (cell embeddings).
            n_cls (int): Number of domain classes to predict.
            nlayers (int, optional): Number of layers in the discriminator network. Defaults to 3.
            activation (Type[nn.Module], optional): Activation function for hidden layers. Defaults to nn.GELU.
        """
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.scale = scale
        self.no_invert_dat = no_invert_dat

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the discriminator.

        Args:
            x (Tensor): Input tensor (cell embeddings) of shape [batch_size, d_model].

        Returns:
            Tensor: Output logits of shape [batch_size, n_cls].
        """
        if not self.no_invert_dat:
            x = grad_reverse(x, scale=self.scale)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class WassersteinDiscriminator(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, scale=1.0):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            # spectral norm for Lipschitz constraint
            self._decoder.append(nn.utils.spectral_norm(nn.Linear(d_model, d_model)))
            self._decoder.append(nn.LeakyReLU())
            self._decoder.append(nn.LayerNorm(d_model))
        # outputs raw scores per class, no softmax
        self.out_layer = nn.utils.spectral_norm(nn.Linear(d_model, n_cls))
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)  # (batch, n_cls) — raw scores
