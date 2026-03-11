from typing import Optional, Tuple
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
from functools import partial, lru_cache


class MHA(nn.Module):
    """
    Custom MHA layer. This takes two separate forward passes on the pect
    genes, and on the gen genes.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
    ):
        """
        pcpt_total_embs: (batch, pcpt_len, hidden_dim) (where hidden_dim = num heads * head dim)
        gen_total_embs: (batch, gen_len, hidden_dim)
        pcpt_key_padding_mask: bool tensor of shape (batch, pcpt_len), 1 means valid and 0 means not valid.
        gen_key_padding_mask: bool tensor of shape (batch, gen_len), 1 means valid and 0 means not valid.
        """

        assert not need_weights

        if gen_total_embs is None:
            return self._forward_perceptual(pcpt_total_embs, pcpt_key_padding_mask)
        pcpt_seq_len, gen_seq_len = (
            pcpt_total_embs.shape[1],
            0 if gen_total_embs is None else gen_total_embs.shape[1],
        )
        total_seq_len = pcpt_seq_len + gen_seq_len

        if pcpt_key_padding_mask is None:
            pcpt_key_padding_mask = (
                torch.zeros(pcpt_total_embs.shape[0], pcpt_seq_len)
                .bool()
                .to(pcpt_total_embs.device)
            )
        if gen_key_padding_mask is None:
            gen_key_padding_mask = (
                torch.zeros(gen_total_embs.shape[0], gen_seq_len)
                .bool()
                .to(pcpt_total_embs.device)
            )

        key_padding_mask = torch.cat(
            [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
        ).to(pcpt_total_embs.device)

        @lru_cache(maxsize=1)
        def make_mask(len, gen_len, device):
            attn_mask = torch.zeros(len, len).bool()
            attn_mask[:, -gen_len:] = True
            attn_mask.diagonal().fill_(False)
            return attn_mask.to(device)

        attn_mask = make_mask(total_seq_len, gen_seq_len, pcpt_total_embs.device)

        total_embs = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)

        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(pcpt_total_embs.device),
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        return (out[:, :pcpt_seq_len], out[:, pcpt_seq_len:]), (None, None)

    def _forward_perceptual(self, total_embs, key_padding_mask):
        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(key_padding_mask.device),
        )
        return (out, None), (None, None)


class CFLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def _reverse_key_padding_mask(self, src_key_padding_mask):
        """
        Reverse the true false values of the key padding mask. This is because
        we follow pytorch rule that the mask is True for padded tokens, but
        in the inner flash MHA, it assumes the mask is False for padded tokens.
        """
        if src_key_padding_mask is None:
            return None

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            return None
        return ~src_key_padding_mask

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        """pcpt_key_padding_mask_ = self._reverse_key_padding_mask(
            pcpt_key_padding_mask)
        gen_key_padding_mask_ = self._reverse_key_padding_mask(
            gen_key_padding_mask)"""  # Used this when using Flash-Attention package
        pcpt_key_padding_mask_ = pcpt_key_padding_mask
        gen_key_padding_mask_ = gen_key_padding_mask

        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
        else:
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm1(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)

        return pcpt_total_embs, gen_total_embs


class CFGenerator(nn.Module):
    # takes in the set of different inputs in an mapping
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        pcpt_key_padding_mask = src_key_padding_mask[:, : pcpt_total_embs.shape[1]]
        gen_key_padding_mask = src_key_padding_mask[:, pcpt_total_embs.shape[1] :]

        if pcpt_key_padding_mask is not None:
            _skpm_dtype = pcpt_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                pcpt_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        for mod in self.layers:
            pcpt_total_embs, gen_total_embs = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
            )

        if self.norm is not None:
            pcpt_total_embs = self.norm(pcpt_total_embs)
            gen_total_embs = self.norm(gen_total_embs)

        return pcpt_total_embs, gen_total_embs


class RefactoredCFGenerator(nn.Module):
    """
    A refactored Transformer Encoder that uses standard PyTorch components.

    This module replicates the behavior of the custom CFGenerator by handling
    the concatenation of perceptual/generative sequences, creating the specific
    attention mask, and splitting the outputs. This logic is wrapped around
    a standard `torch.nn.TransformerEncoder`.

    Performance optimizations:
    - Uses standard PyTorch TransformerEncoder for optimized kernels
    - Caches attention masks to avoid recreation on every forward pass
    - Single concatenation/split operation vs per-layer operations
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_scheme: str = "post",
        norm: Optional[nn.Module] = None,
    ):
        """
        Initializes the refactored Transformer Encoder.

        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multi-head attention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feed-forward network.
            dropout (float): The dropout value.
            activation (str): The activation function ('relu' or 'gelu').
            layer_norm_eps (float): The epsilon value for layer normalization.
            batch_first (bool): If True, inputs are (batch, seq, feature).
            norm_scheme (str): The normalization scheme, "pre" or "post".
            norm (Optional[nn.Module]): An optional final layer normalization.
        """
        super().__init__()
        assert batch_first, "This implementation requires batch_first=True"
        assert norm_scheme in [
            "pre",
            "post",
        ], "norm_scheme must be either 'pre' or 'post'"

        # Map norm_scheme to the 'norm_first' parameter in the standard layer
        # norm_first = True if norm_scheme == "pre" else False
        norm_first = False

        # Create a standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        # Create the standard Transformer Encoder by stacking the layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=norm,
        )

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        src_key_padding_mask: Tensor,
        attn_mask,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Passes the perceptual and generative sequences through the encoder.

        Args:
            pcpt_total_embs (Tensor): The perceptual sequence embeddings.
            gen_total_embs (Optional[Tensor]): The generative sequence embeddings.
            pcpt_key_padding_mask (Optional[Tensor]): Mask for the perceptual sequence.
            gen_key_padding_mask (Optional[Tensor]): Mask for the generative sequence.

        Returns:
            A tuple of the final (perceptual_output, generative_output) tensors.
        """
        # --- Pre-processing Step ---
        pcpt_seq_len = pcpt_total_embs.shape[1]

        # 1. Concatenate inputs
        src = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)

        # --- Call the standard Transformer Encoder ---
        output = self.transformer_encoder(
            src=src,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
        )

        # --- Post-processing Step ---
        # Split the output back into perceptual and generative parts
        pcpt_output = output[:, :pcpt_seq_len]
        gen_output = output[:, pcpt_seq_len:]

        return pcpt_output, gen_output


def biological_mask_mod(b, h, q_idx, kv_idx, pcpt_len):
    """
    Topology:
    - P can attend to P
    - G can attend to P and itself
    - P cannot attend to G
    - G cannot attend to other G
    """
    # 1. Valid if Key is in Perceptual region
    # This covers (P -> P) and (G -> P)
    key_is_pcpt = kv_idx < pcpt_len

    # 2. Valid if Diagonal (Self Attention)
    # This covers (G -> Self) and (P -> Self, which is redundant with #1 but harmless)
    is_diagonal = q_idx == kv_idx

    # Allowed if either condition is true
    return key_is_pcpt | is_diagonal


class FlexTransformerLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, activation, norm_first
    ):
        super().__init__()
        self.norm_first = norm_first
        self.nhead = nhead

        # Attention components
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward components
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, block_mask):
        # src shape: (Batch, Seq_Len, Dim)

        # 1. Self Attention Block
        residual = src
        if self.norm_first:
            x = self.norm1(src)
        else:
            x = src

        # Projection for FlexAttention
        # shape: (Batch, Seq_Len, 3 * Dim) -> (Batch, Seq_Len, 3, Heads, Head_Dim)
        B, L, _ = x.shape
        qkv = self.in_proj(x).view(B, L, 3, self.nhead, -1)
        query, key, value = qkv.unbind(2)

        # Transpose for FlexAttention: (Batch, Heads, Seq_Len, Head_Dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_out = flex_attention(query, key, value, block_mask=block_mask)

        # Reshape back: (Batch, Seq_Len, Dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        attn_out = self.out_proj(attn_out)

        x = residual + self.dropout(attn_out)
        if not self.norm_first:
            x = self.norm1(x)

        # 2. Feed Forward Block (Standard)
        residual = x
        if self.norm_first:
            x = self.norm2(x)

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)

        if not self.norm_first:
            x = self.norm2(x)

        return x


class QuickCFGenerator(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_scheme: str = "post",
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FlexTransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="relu",  # or pass from config
                    norm_first=(norm_scheme == "pre"),
                )
                for _ in range(num_layers)
            ]
        )
        # self.pcpt_len = pcpt_len

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        src_key_padding_mask: Tensor,  # kept for API compatibility, handle inside if needed
        attn_mask=None,  # IGNORED now
    ) -> Tuple[Tensor, Optional[Tensor]]:
        pcpt_len = pcpt_total_embs.shape[1]

        # 1. Concatenate inputs
        src = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)
        B, total_len, _ = src.shape

        # 2. Create the Block Mask (The Efficient "Virtual" Mask)
        # We bind the 'pcpt_len' variable to the function using partial

        # This creation is very fast and low memory
        block_mask = create_block_mask(
            partial(biological_mask_mod, pcpt_len=pcpt_len),
            B=None,
            H=None,  # None lets it broadcast across batch/heads
            Q_LEN=total_len,
            KV_LEN=total_len,
            device=src.device,
        )

        # 3. Pass through layers
        x = src
        for layer in self.layers:
            x = layer(x, block_mask)

        # 4. Split output
        pcpt_output = x[:, :pcpt_len]
        gen_output = x[:, pcpt_len:]

        return pcpt_output, gen_output
