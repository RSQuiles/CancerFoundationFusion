from typing import Optional, Tuple
import warnings
import torch
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
from functools import partial, lru_cache


# flex_attention is a slow reference implementation in eager; torch.compile lowers it
# into the fused kernel. Done here rather than via --compile, which wraps the whole
# module (graph breaks) and defaults to dynamic shapes (unsupported by flex).
# dynamic=False => one compile per distinct sequence length.
_flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
_create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)


@lru_cache(maxsize=8)
def _pcpt_gen_attn_mask(total_len: int, gen_len: int, device: torch.device) -> Tensor:
    """Bool attention mask for the concatenated ``[pcpt | gen]`` sequence.

    ``True`` means *masked out*: every query may attend to the perceptual block,
    the generative block is invisible to everyone except itself (the diagonal).

    Module-level so the cache actually survives between calls — an ``lru_cache``
    defined inside ``forward`` is rebuilt (on CPU, then copied H2D) on every layer
    of every step.
    """
    attn_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    attn_mask[:, total_len - gen_len :] = True
    attn_mask.diagonal().fill_(False)
    return attn_mask


def _flex_mask_mod(b, h, q_idx, kv_idx, pcpt_len, valid):
    """FlexAttention ``mask_mod`` equivalent to ``_pcpt_gen_attn_mask`` + key padding.

    ``True`` means *keep* (the inverse convention of the dense mask above).
    ``valid`` is ``~key_padding_mask`` of shape ``(B, L)``, or ``None`` when the
    batch has no padding at all.

    The diagonal is always kept so that no query row is fully masked (which would
    make softmax produce NaNs). For a valid query this is a no-op — the topology
    already keeps the diagonal — and for a *padded* query the row is discarded
    downstream by ``positions_to_match`` in ``TransformerModule.forward``.
    """
    allowed = (kv_idx < pcpt_len) | (q_idx == kv_idx)
    if valid is not None:
        allowed = (allowed & valid[b, kv_idx]) | (q_idx == kv_idx)
    return allowed


@lru_cache(maxsize=8)
def _cached_block_mask(total_len: int, pcpt_len: int, device: torch.device):
    """Padding-free block mask — identical across steps, so worth caching."""
    return _create_block_mask_compiled(
        partial(_flex_mask_mod, pcpt_len=pcpt_len, valid=None),
        B=None,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )


def build_block_mask(pcpt_len: int, total_len: int, key_padding_mask: Optional[Tensor],
                     device: torch.device):
    """Build the FlexAttention block mask once per encoder forward.

    Callers must hoist this above the layer loop: ``create_block_mask`` is not
    free, and the mask is the same for all layers.
    """
    if key_padding_mask is None or not bool(key_padding_mask.any()):
        return _cached_block_mask(total_len, pcpt_len, device)

    valid = ~key_padding_mask
    return _create_block_mask_compiled(
        partial(_flex_mask_mod, pcpt_len=pcpt_len, valid=valid),
        B=key_padding_mask.shape[0],
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )


def _make_norm(norm_type: str, d_model: int, eps: float, **factory_kwargs):
    """Create a normalisation layer.

    Args:
        norm_type: "layer" for LayerNorm, "rms" for RMSNorm.
        d_model: Feature dimension.
        eps: Epsilon for numerical stability.
        **factory_kwargs: Extra kwargs (device, dtype) forwarded to the layer.
    """
    if norm_type == "rms":
        return nn.RMSNorm(d_model, eps=eps, **factory_kwargs)
    return nn.LayerNorm(d_model, eps=eps, **factory_kwargs)


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
        self.attention_dropout = attention_dropout

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
        block_mask=None,
    ):
        """
        pcpt_total_embs: (batch, pcpt_len, hidden_dim) (where hidden_dim = num heads * head dim)
        gen_total_embs: (batch, gen_len, hidden_dim)
        pcpt_key_padding_mask: bool tensor of shape (batch, pcpt_len), 1 means valid and 0 means not valid.
        gen_key_padding_mask: bool tensor of shape (batch, gen_len), 1 means valid and 0 means not valid.
        block_mask: a FlexAttention ``BlockMask`` built once per encoder forward by
            ``build_block_mask``. When given, attention runs through FlexAttention,
            which never materialises the ``[B*H, L, L]`` mask or score matrix.
        """

        assert not need_weights

        if gen_total_embs is None:
            return self._forward_perceptual(pcpt_total_embs, pcpt_key_padding_mask)
        pcpt_seq_len, gen_seq_len = (
            pcpt_total_embs.shape[1],
            0 if gen_total_embs is None else gen_total_embs.shape[1],
        )
        total_seq_len = pcpt_seq_len + gen_seq_len

        total_embs = torch.cat((pcpt_total_embs, gen_total_embs), dim=1)

        if block_mask is not None:
            out = self._flex_attend(total_embs, block_mask)
            return (out[:, :pcpt_seq_len], out[:, pcpt_seq_len:]), (None, None)

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

        attn_mask = _pcpt_gen_attn_mask(
            total_seq_len, gen_seq_len, pcpt_total_embs.device
        )

        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(pcpt_total_embs.device),
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        return (out[:, :pcpt_seq_len], out[:, pcpt_seq_len:]), (None, None)

    def _flex_attend(self, total_embs: Tensor, block_mask) -> Tensor:
        """Same weights as ``self.self_attn``, but O(B*H*L*d) memory.

        ``nn.MultiheadAttention`` is kept purely as the parameter container, so a
        checkpoint trained under ``--attn-impl mha`` loads unchanged here and vice
        versa. The projections below are exactly what
        ``F.multi_head_attention_forward`` does for the self-attention,
        equal-embed-dim case.
        """
        B, L, E = total_embs.shape
        qkv = F.linear(
            total_embs, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        )
        # (B, L, 3E) -> 3 x (B, H, L, head_dim)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        query, key, value = (t.permute(0, 2, 1, 3) for t in qkv.unbind(2))

        # flex_attention's default scale is 1/sqrt(head_dim), matching nn.MHA.
        attn_out = _flex_attention_compiled(query, key, value, block_mask=block_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        return self.self_attn.out_proj(attn_out)

    def _forward_perceptual(self, total_embs, key_padding_mask):
        # need_weights=False keeps this on the SDPA path; the default (True) falls
        # back to an explicit baddbmm+softmax and allocates a [B, L, L] weight
        # tensor that is thrown away. This path is hit by TransformerModule.embed(),
        # and therefore by the CDD bulk-bank refresh in on_train_batch_start.
        out, _ = self.self_attn(
            total_embs,
            total_embs,
            total_embs,
            key_padding_mask=key_padding_mask.to(key_padding_mask.device),
            need_weights=False,
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
        norm_type="layer",   # "layer" or "rms"
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
        self.activation_name = activation
        if activation == "swiglu":
            # SwiGLU: two parallel projections replace linear1
            self.linear_gate = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
            self.linear_val = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = _make_norm(norm_type, d_model, layer_norm_eps, **factory_kwargs)
        self.norm2 = _make_norm(norm_type, d_model, layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be 'pre' or 'post'")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swiglu":
            # SwiGLU is handled directly in _ff; no scalar activation needed here
            return None

        raise RuntimeError(
            "activation should be relu/gelu/swiglu, not {}".format(activation)
        )

    def _ff(self, x: Tensor) -> Tensor:
        """Feed-forward sub-layer, dispatching between standard and SwiGLU."""
        if self.activation_name == "swiglu":
            # SwiGLU: gate * value, then project out
            gate = F.silu(self.linear_gate(x))
            val = self.linear_val(x)
            return self.linear2(self.dropout(gate * val))
        else:
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

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
        block_mask=None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            block_mask: optional FlexAttention ``BlockMask``, built once per encoder
                forward and shared by every layer.
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
            # Pre-norm: normalise BEFORE each sublayer
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                self.norm1(pcpt_total_embs),
                self.norm1(gen_total_embs) if gen_total_embs is not None else None,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
                block_mask=block_mask,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = pcpt_total_embs + self.dropout2(self._ff(self.norm2(pcpt_total_embs)))

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = gen_total_embs + self.dropout2(self._ff(self.norm2(gen_total_embs)))

        else:
            # Post-norm: normalise AFTER residual addition
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
                block_mask=block_mask,
            )[0]
            pcpt_total_embs = self.norm1(pcpt_total_embs + self.dropout1(pcpt_total_embs2))
            pcpt_total_embs = self.norm2(pcpt_total_embs + self.dropout2(self._ff(pcpt_total_embs)))

            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs + self.dropout1(gen_total_embs2))
                gen_total_embs = self.norm2(gen_total_embs + self.dropout2(self._ff(gen_total_embs)))

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
        grad_checkpoint: bool = False,
        attn_impl: str = "mha",
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check
        self.grad_checkpoint = grad_checkpoint
        self.attn_impl = attn_impl

        # flex_attention has no attention-weight dropout. The residual and FFN
        # dropouts in CFLayer still apply, but this one is silently lost, so say so
        # rather than let it look like an unexplained regularisation change.
        attn_dropout = getattr(encoder_layer.self_attn, "attention_dropout", 0.0)
        if attn_impl == "flex" and attn_dropout:
            warnings.warn(
                f"--attn-impl flex drops attention-weight dropout (currently "
                f"{attn_dropout}); FFN and residual dropout are unaffected. Pass "
                f"--dropout 0 for an exact match against --attn-impl mha, or accept "
                f"the slightly weaker regularisation.",
                RuntimeWarning,
                stacklevel=2,
            )

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

        # Build the FlexAttention block mask once for the whole stack — it is the
        # same for every layer, and create_block_mask is not free.
        block_mask = None
        if self.attn_impl == "flex" and gen_total_embs is not None:
            block_mask = build_block_mask(
                pcpt_len=pcpt_total_embs.shape[1],
                total_len=pcpt_total_embs.shape[1] + gen_total_embs.shape[1],
                key_padding_mask=src_key_padding_mask,
                device=pcpt_total_embs.device,
            )

        for mod in self.layers:
            if self.grad_checkpoint and self.training:
                # use_reentrant=False is required, not preferred: gen_total_embs is
                # None on the perceptual-only path and block_mask is not a tensor,
                # neither of which the reentrant implementation accepts. It also
                # preserves RNG state, so dropout is reproduced exactly on the
                # recompute and the result is numerically identical.
                pcpt_total_embs, gen_total_embs = torch.utils.checkpoint.checkpoint(
                    mod,
                    pcpt_total_embs,
                    gen_total_embs,
                    pcpt_key_padding_mask,
                    gen_key_padding_mask,
                    block_mask,
                    use_reentrant=False,
                )
            else:
                pcpt_total_embs, gen_total_embs = mod(
                    pcpt_total_embs,
                    gen_total_embs,
                    pcpt_key_padding_mask,
                    gen_key_padding_mask,
                    block_mask,
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
        grad_checkpoint: bool = False,
        attn_impl: str = "mha",
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
        ], "norm_scheme must be 'pre' or 'post'"

        norm_first = norm_scheme == "pre"

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
        self.grad_checkpoint = grad_checkpoint
        if attn_impl == "flex":
            # This variant delegates to nn.TransformerEncoder, whose layers own their
            # attention internally; there is no seam to swap in FlexAttention without
            # reimplementing the layer. Use --gen-method theirs/orig for --attn-impl flex.
            raise ValueError(
                "--attn-impl flex is not supported with --gen-method mine "
                "(RefactoredCFGenerator). Use --gen-method theirs or orig."
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
        if self.grad_checkpoint and self.training:
            # nn.TransformerEncoder has no checkpointing hook, so drive its layers
            # directly. Same layers, same order, same final norm.
            output = src
            for layer in self.transformer_encoder.layers:
                output = torch.utils.checkpoint.checkpoint(
                    layer,
                    output,
                    attn_mask,
                    src_key_padding_mask,
                    False,  # is_causal
                    use_reentrant=False,
                )
            if self.transformer_encoder.norm is not None:
                output = self.transformer_encoder.norm(output)
        else:
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
        grad_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
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
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, block_mask, use_reentrant=False
                )
            else:
                x = layer(x, block_mask)

        # 4. Split output
        pcpt_output = x[:, :pcpt_len]
        gen_output = x[:, pcpt_len:]

        return pcpt_output, gen_output
