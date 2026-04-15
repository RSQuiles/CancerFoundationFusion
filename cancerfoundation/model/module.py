from pathlib import Path
import os
from typing import Dict, Mapping, Optional, Union, Type, Callable, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pandas as pd
from .layers import RefactoredCFGenerator, QuickCFGenerator, CFLayer, CFGenerator

from .grad_reverse import grad_reverse


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
        norm_first: bool,
        do_dat: bool,
        batchnorm: bool,
        dat_scale: float,
        normalise_bins: bool,
        no_invert_dat: bool,
        where_condition: str,
        max_seq_len: int,
        gen_method: str,
        their_init_weights: bool,
        # Unified FM parameters
        contrastive: bool = False,
        aggregation: bool = False,
        agg_fn: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        gene_embeddings_path: Optional[Union[str, os.PathLike, Path]] = None,
        gene_embeddings_freeze: bool = True,
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
            norm_first (bool): Whether to use pre-layer normalization in the transformer.
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
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.pad_token_id = pad_token_id
        self.norm_scheme = "pre" if norm_first else "post"
        self.use_generative_training = use_generative_training
        self.where_condition = where_condition
        self.max_seq_len = max_seq_len
        self.gen_method = gen_method
        self.contrastive = contrastive
        self.aggregation = aggregation
        self.agg_fn = agg_fn
        self.vocab = vocab
        self.gene_embeddings_path = gene_embeddings_path
        self.gene_embeddings_freeze = gene_embeddings_freeze

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
        self.no_invert_dat = no_invert_dat
        self.do_mvc = do_mvc
        self.criterion_conditions = nn.CrossEntropyLoss()
        self.criterion = criterion

        mvc_decoder_d_in = d_model
        expr_decoder_d_in = d_model
        # Conditions are taken into account only in the decoder
        if self.conditions:
            mvc_decoder_d_in = d_model * (len(self.conditions) + 1)
            if where_condition == "end":
                expr_decoder_d_in = d_model * (len(self.conditions) + 1)

        # Conditions are encoded as separate embeddings
        if conditions:
            self.condition_encoders = nn.ModuleDict({})
            for cond_name, cond_num in self.conditions.items():
                self.condition_encoders[cond_name] = ConditionEncoder(cond_num, d_model)
            # Check condition encoders
            print("Condition Encoders Inspection:")
            print(f"Conditions: {self.conditions}")
            print(f"Encoders: {self.condition_encoders}")

            if do_dat:
                self.grad_reverse_discriminators = nn.ModuleDict({})
                for cond_name, cond_num in self.conditions.items():
                    self.grad_reverse_discriminators[cond_name] = (
                        AdversarialDiscriminator(
                            d_model,
                            n_cls=cond_num,
                            scale=dat_scale,
                            no_invert_dat=no_invert_dat,
                        )
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
                )
                self.transformer_encoder = CFGenerator(
                    encoder_layer=encoder_layers, num_layers=nlayers
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
                )
                self.transformer_encoder = CFGenerator(
                    encoder_layer=encoder_layers, num_layers=nlayers
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
                )
                self.generative_flag = nn.Parameter(torch.randn(d_model))
                self.gene_encoder = self._build_gene_encoder(
                    ntoken=ntoken,
                    d_model=d_model,
                    padding_idx=pad_token_id,
                )
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model,
                nhead,
                d_hid,
                dropout,
                batch_first=True,
                norm_first=norm_first,
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
        """Embeds gene IDs into dense vectors.

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

        if conditions is not None and self.where_condition == "begin":
            cond_embs = self.condition_encoders["technology"](conditions)
            cond_embs = cond_embs.unsqueeze(1)  # (batch, 1, embsize)
            total_embs = torch.cat([cond_embs, gene_embs], dim=1)
            padded_value_embs = torch.nn.functional.pad(
                value_embs, (0, 0, 0, 1), "constant", 0
            )  # (batch, seq_len+1, embsize)
            total_embs = total_embs + padded_value_embs  # (batch, seq_len+1, embsize)
            src_key_padding_mask = torch.nn.functional.pad(
                src_key_padding_mask, (0, 1), "constant", False
            )  # (batch, seq_len+1)
        else:
            total_embs = gene_embs + value_embs

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
    ) -> Tensor:
        """Encodes gene IDs and expression values into contextual embeddings. This method is used during perceptual (non-generative) training.

        Args:
            src (Tensor): Input gene token IDs of shape (batch, seq_len).
            values (Tensor): Input expression values of shape (batch, seq_len).
            src_key_padding_mask (Tensor): Padding mask of shape (batch, seq_len).
            conditions (Optional[Dict], optional): Dictionary of condition tensors. Defaults to None.

        Returns:
            Tensor: The output of the Transformer encoder, of shape (batch, seq_len, embsize).
        """
        self._check_condition_labels(conditions)

        src_embs = self.encoder(src)
        self.cur_gene_token_embs = src_embs

        values = self.value_encoder(values)
        if conditions:
            cond_emb = self.condition_encoders["technology"](conditions["technology"])
            values[:, 1, :] = cond_emb

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src_embs * values
        else:
            total_embs = src_embs + values

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        return output

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

        if self.where_condition == "begin" and self.conditions:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](cond_values)
                    for cond_name, cond_values in conditions.items()
                ],
                dim=0,
            )
            pcpt_key_padding_mask = torch.nn.functional.pad(
                pcpt_key_padding_mask, (1, 0), "constant", False
            )
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
            if self.conditions:
                mvc_input_emb = torch.cat(
                    [cell_emb, condition_emb.view(condition_emb.shape[0], -1)], dim=1
                )
            else:
                mvc_input_emb = cell_emb

            mvc_output = self.mvc_decoder(mvc_input_emb, self.cur_gene_token_embs)
            output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)

        if self.do_dat:
            if self.conditions:
                output["condition_output"] = {}
                for (
                    cond_name,
                    discriminator,
                ) in self.grad_reverse_discriminators.items():
                    output["condition_output"][cond_name] = discriminator(cell_emb)
                # Inform about DAT conditions
                print(
                    f"Performing DAT on: {list(self.grad_reverse_discriminators.keys())}"
                )

        return output

    def _prepare_generative_input(self, tensors: dict[str, torch.Tensor], noise:float=None):
        """Prepares tensors for the generative forward pass."""
        pcpt_gene = tensors["pcpt_gene"]
        pcpt_expr = tensors["pcpt_expr"]
        pcpt_key_padding_mask = tensors["pcpt_key_padding_mask"]
        gen_gene = tensors["gen_gene"]
        gen_expr_target = tensors["gen_expr_target"]
        gen_key_padding_mask = tensors["gen_key_padding_mask"]
        attn_mask = tensors["attn_mask"]

        # Apply noising schedule
        if noise is not None:
            pcpt_expr = utils.downsample_profile(pcpt_expr)

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

    def _prepare_perceptual_input(self, tensors: dict[str, torch.Tensor]):
        """Prepares tensors for the perceptual forward pass."""
        input_gene_ids = tensors["gene"]
        input_values = tensors["masked_expr"]

        # Apply noising schedule
        if noise is not None:
            input_values = utils.downsample_profile(input_values)

        src_key_padding_mask = tensors["gene_key_padding_mask"]
        target_values = tensors["expr"]

        return input_gene_ids, input_values, src_key_padding_mask, target_values

    def forward(  # tensors is the data_dict from collator
        self, 
        tensors: dict[str, torch.Tensor], 
        use_cell_embedding: bool = False,
        noise: float = None
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
            ) = self._prepare_generative_input(tensors)
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
            loss = loss_expr = self.criterion(
                gen_expr_preds, gen_expr_target, positions_to_match
            )
            loss_dict["loss_expr"] = loss_expr

            if self.do_mvc:
                mvc_preds_for_gen = output_dict["mvc_output"][:, pcpt_gene.shape[1] :]
                loss_mvc = self.criterion(
                    mvc_preds_for_gen, gen_expr_target, positions_to_match
                )
                loss = loss + loss_mvc
                loss_dict["loss_mvc"] = loss_mvc

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
            loss = loss + use_cell_embedding * loss_gen
            loss_dict["loss_gen"] = loss_gen

        # Perceptual training
        else:
            input_gene_ids, input_values, src_key_padding_mask, target_values = (
                self._prepare_perceptual_input(tensors)
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
            loss = loss_expr = self.criterion(
                output_values, target_values, positions_to_match
            )
            loss_dict["loss_expr"] = loss_expr

            if self.do_mvc:
                loss_mvc = self.criterion(
                    output_dict["mvc_output"], target_values, positions_to_match
                )
                loss = loss + loss_mvc
                loss_dict["loss_mvc"] = loss_mvc

        # Return directly if denoising task
        if noise is not None:
            return loss_dict["loss_expr"]

        # Domain adversarial training
        if self.do_dat:
            if self.conditions:
                for condition in self.conditions:
                    condition_loss = self.criterion_conditions(
                        output_dict["condition_output"][condition],
                        conditions_batch[condition].squeeze(),
                    )
                    loss_dict[condition + "_confidence"] = (
                        output_dict["condition_output"][condition]
                        .softmax(dim=-1)
                        .gather(dim=1, index=conditions_batch[condition].unsqueeze(1))
                        .squeeze(1)
                        .float()
                        .mean()
                    )
                    loss += condition_loss / len(self.conditions)
                    loss_dict["condition_" + condition] = condition_loss.detach() / len(
                        self.conditions
                    )

        # Contrastive loss: if enabled, it brings the pseudobulk and real bulk samples closer together in the embedding space.
        if self.contrastive:
            # print("Using contrastive loss...")
            embeddings = output_dict[
                "embeddings"
            ]  # Embeddings are transformer_output regardless of the training mode
            modalities = tensors["conditions"]["modality"]
            assert len(embeddings) == len(
                modalities
            ), "Embeddings and modalities tensors must have the same batch size"

            loss_contrastive = self.modality_contrastive_loss(embeddings, modalities)
            loss = loss + loss_contrastive
            loss_dict["loss_contrastive"] = loss_contrastive.detach()

        # Aggregation consistency loss: if enabled, it encourages the model to produce consistent predictions for single-cell and pseudobulk samples
        if self.aggregation:
            # print("Using aggregation consistency loss...")
            embeddings = output_dict["embeddings"]
            # Reconstruct assignment
            sc_assignment = {}
            assert len(embeddings) == len(
                tensors["is_sc_for_pb"]
            ), "Embeddings and input dictionaries must have the same batch size"
            for idx in range(len(embeddings)):
                if tensors["is_sc_for_pb"][idx] == 1:
                    if (
                        tensors["sample_pseudobulk_index"][idx].item()
                        not in sc_assignment
                    ):
                        sc_assignment[
                            tensors["sample_pseudobulk_index"][idx].item()
                        ] = []
                    sc_assignment[
                        tensors["sample_pseudobulk_index"][idx].item()
                    ].append(idx)

            # Enforce consistency loss for each pseudobulk and its assigned single cells
            loss_agg = 0
            for pb_idx, sc_indices in sc_assignment.items():
                if len(sc_indices) > 0:
                    pb_embedding = embeddings[pb_idx]
                    sc_embeddings = embeddings[sc_indices]
                    if self.agg_fn == "mean":
                        sc_embedding_agg = sc_embeddings.mean(dim=0)
                    elif self.agg_fn == "sum":
                        sc_embedding_agg = sc_embeddings.sum(dim=0)
                    else:
                        raise ValueError(f"Unknown agg_fn: {self.agg_fn}")
                    loss_agg += F.mse_loss(pb_embedding, sc_embedding_agg)
            loss = loss + loss_agg
            loss_dict["loss_agg"] = loss_agg.detach()

        loss_dict["total_loss"] = loss
        return loss_dict

    def modality_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        modalities: torch.Tensor,
        temperature: float = 0.1,
        use_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Multi-positive InfoNCE:
        - anchors: modality == 0
        - positives: modality == 2
        - negatives: modality == 1

        embeddings:
            shape (B, D) or (B, L, D)
        modalities:
            shape (B,)
        """
        assert embeddings.size(0) == modalities.size(
            0
        ), "Embeddings and modalities tensors must have the same batch size"

        # As transformer_output is sequence-shaped, reduce to one embedding per sample.
        if embeddings.dim() == 3:
            if use_cls_token:
                embeddings = embeddings[:, 0, :]  # CLS token
            else:
                embeddings = embeddings.mean(dim=1)  # mean pooling

        assert embeddings.dim() == 2, f"Expected (B, D), got {embeddings.shape}"
        assert modalities.dim() == 1, f"Expected (B,), got {modalities.shape}"

        # Masks
        anchor_mask = modalities == 0
        candidate_mask = (modalities == 1) | (modalities == 2)

        # Select anchor and candidate embeddings
        anchor_emb = embeddings[anchor_mask]  # (N0, D)
        candidate_emb = embeddings[candidate_mask]  # (N1+N2, D)
        candidate_mods = modalities[candidate_mask]  # (N1+N2,)

        if anchor_emb.size(0) == 0:
            raise ValueError("No modality-0 samples in batch.")
        if not (candidate_mods == 2).any():
            raise ValueError("No modality-2 samples in batch for positives.")
        if not (candidate_mods == 1).any():
            raise ValueError("No modality-1 samples in batch for negatives.")

        # Normalize for cosine similarity
        anchor_emb = F.normalize(anchor_emb, dim=1)
        candidate_emb = F.normalize(candidate_emb, dim=1)

        # Similarity matrix: (N0, N1+N2)
        logits = anchor_emb @ candidate_emb.T
        logits = logits / temperature

        # Positive mask over candidate columns
        pos_cols = (candidate_mods == 2).unsqueeze(0).expand(anchor_emb.size(0), -1)

        # Stable log-softmax form:
        log_denom = torch.logsumexp(logits, dim=1)  # (N0,)
        pos_logits = logits.masked_fill(~pos_cols, float("-inf"))
        log_num = torch.logsumexp(pos_logits, dim=1)  # (N0,)

        loss = -(log_num - log_denom).mean()
        return loss

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
        transformer_output = (
            pcpt_output
            if gen_output is None
            else torch.cat([pcpt_output, gen_output], dim=1)
        )

        condition_emb = None
        decoder_input = transformer_output
        if self.conditions:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](cond_values)
                    for cond_name, cond_values in conditions.items()
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
        if self.conditions:
            condition_emb = torch.cat(
                [
                    self.condition_encoders[cond_name](cond_values).unsqueeze(1)
                    for cond_name, cond_values in conditions.items()
                ],
                dim=1,
            )
        transformer_output = self.encode(src, values, src_key_padding_mask, conditions)

        if self.where_condition == "begin":
            decoder_input = transformer_output

        elif self.where_condition == "end":
            if self.conditions:
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
            condition_emb=(condition_emb if self.conditions else None),
            do_sample=do_sample,
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

        if weights is not None:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        else:
            if weights_file is None:
                raise ValueError("weights_file cannot be None when no weights tensor is provided.")
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
                for idx, token in enumerate(ordered_tokens):
                    if token is None:
                        continue
                    if token in weight_frame.index:
                        row = np.asarray(weight_frame.loc[token], dtype=np.float32)
                        weight_tensor[idx] = torch.tensor(row, dtype=torch.float32)
                    elif token == "<cls>":
                        weight_tensor[idx] = mean_vector
                    elif token == "<pad>":
                        weight_tensor[idx] = torch.zeros(source_dim, dtype=torch.float32)
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
