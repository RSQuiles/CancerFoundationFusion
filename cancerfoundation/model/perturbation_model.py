# in perturbation_model.py
import torch
from torch import nn, Tensor
from typing import Dict

# Make sure to import your base model correctly
from .module import TransformerModule


class PerturbationTransformer(TransformerModule):
    """
    A specialized Transformer model for gene perturbation prediction.
    """

    def __init__(self, pert_pad_id: int, **kwargs):
        # Initialize the parent class to get all the shared layers
        super().__init__(**kwargs)

        # Add the new, perturbation-specific layer
        self.pert_encoder = nn.Embedding(3, self.d_model, padding_idx=pert_pad_id)

        if self.conditions is not None:
            print(
                "Support for conditions is not yet implemented, but this model has been pretrained with conditions. Performance may suffer!"
            )

    def _encode_perturbation(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        pert_flags: Tensor,
    ) -> Tensor:
        if self.gen_method == "orig":
            src_embs = self.encoder(src)
        else:
            src_embs = self.gene_encoder(src)
        self.cur_gene_token_embs = src_embs  # For MVC if you use it
        values_embs = self.value_encoder(values)
        pert_embs = self.pert_encoder(pert_flags)

        total_embs = src_embs + values_embs + pert_embs

        output = self.transformer_encoder(
            pcpt_total_embs=total_embs,
            gen_total_embs=None,
            pcpt_key_padding_mask=src_key_padding_mask,
            gen_key_padding_mask=None,
        )

        return output

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        use_cell_embedding: bool = False,  # Keep all args for compatibility
    ) -> Dict[str, torch.Tensor]:
        condition = None
        # 1. Unpack the dictionary at the beginning of the method
        src = tensors["gene"]
        values = tensors["masked_expr"]
        pert_flags = tensors["pert_flags"]
        src_key_padding_mask = tensors["src_key_padding_mask"]

        # 2. Call your perturbation-specific encoding method
        transformer_output, _ = self._encode_perturbation(
            src, values, src_key_padding_mask, pert_flags
        )

        if condition is None and self.conditions is not None:
            conditions = torch.zeros_like(transformer_output)
            decoder_input = torch.cat([transformer_output, conditions], dim=-1)
        else:
            decoder_input = transformer_output

        # 3. Decode the output to get predictions
        decoder_output = self.decoder(decoder_input)

        # 4. Return the result
        return decoder_output

    # The inference method stays the same
    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="all",
        gene_ids=None,
        amp=True,
    ) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)
        pert_flags = x[:, 1].long().view(batch_size, -1)
        n_genes = ori_gene_values.size(1)

        # This logic is from the fine-tuning script to select genes
        if include_zero_gene == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:  # "batch-wise"
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        # You need a function like this, maybe in your utils
        # from scgpt.utils import map_raw_id_to_vocab_id
        # mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)

        # For now, let's assume `gene_ids` is a mapping from raw index to vocab id
        vocab_ids = torch.tensor(gene_ids, device=device)
        mapped_input_gene_ids = vocab_ids[input_gene_ids]
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )

        with torch.cuda.amp.autocast(enabled=amp):
            # IMPORTANT: Call your model's forward method here
            output_dict = self.perceptual_forward(
                src=mapped_input_gene_ids,
                values=input_values,
                src_key_padding_mask=src_key_padding_mask,
                pert_flags=input_pert_flags,
                conditions=None,  # Assuming no conditions for this task
            )

        output_values = output_dict["mlm_output"]
        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values
