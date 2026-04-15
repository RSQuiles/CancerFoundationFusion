import gc
import math
import os.path
from collections import Counter
from typing import Dict, List, Optional, Union

# import bionty as bt
# from scdataloader.utils import translate
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.distributions import Gamma, Poisson

def noise_log1p_profile(
    mat: Tensor,
    noise_level: float,
    mode: str = "mask",
    clamp_min: float = 0.0,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Corrupt a log1p-normalized profile with one of two noise options.

    Supported modes:
    - "mask": multiplies values by a Bernoulli 0-1 mask with keep prob 1-noise_level
    - "gaussian": adds isotropic Gaussian noise N(0, noise_level^2 I)

    Args:
        mat: Input log1p-normalized tensor.
        noise_level: Scalar in [0, 1]. For "mask" it is drop probability,
            for "gaussian" it is the standard deviation.
        mode: "mask" or "gaussian".
        clamp_min: Lower bound after corruption. Keep at 0.0 for log1p values.
        valid_mask: Optional boolean mask selecting positions to corrupt. If None,
            all positions are eligible.

    Returns:
        Tensor with same shape and dtype as ``mat``.
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")

    x = mat.float()
    vm: Optional[Tensor] = valid_mask
    if vm is not None:
        vm_bool = vm.bool()
        if vm_bool.shape != x.shape:
            raise ValueError(
                f"valid_mask shape {vm_bool.shape} must match input shape {x.shape}"
            )
        vm = vm_bool

    if mode == "mask":
        keep_prob = 1.0 - noise_level
        keep = (torch.rand_like(x) < keep_prob).to(x.dtype)
        out = x * keep
    elif mode == "gaussian":
        out = x + torch.randn_like(x) * noise_level
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: 'mask', 'gaussian'."
        )

    if vm is not None:
        out = torch.where(vm, out, x)

    out = torch.clamp(out, min=clamp_min)
    return out.to(dtype=mat.dtype)


def apply_log1p_noise_to_branch_inputs(
    tensors: dict[str, Tensor],
    branch: str,
    noise_level: float,
    mode: str = "mask",
    keep_first_n_tokens: int = 1,
    clamp_min: float = 0.0,
    clone: bool = True,
) -> dict[str, Tensor]:
    """Apply log1p noise to collator outputs with branch-aware semantics.

    Branch behavior:
    - "pcpt": noise only the model input `masked_expr` using `gene_key_padding_mask`.
    - "both" or "gen": noise only context input `pcpt_expr` using
      `pcpt_key_padding_mask`; do not touch `gen_expr_target`.

    Protected tokens:
    - The first ``keep_first_n_tokens`` tokens are never noised.

    Args:
        tensors: Data dictionary produced by the collator.
        branch: One of "pcpt", "both", or "gen".
        noise_level: Scalar in [0, 1].
        mode: "mask" or "gaussian".
        keep_first_n_tokens: Number of leading tokens to keep unchanged.
        clamp_min: Lower clamp passed to ``noise_log1p_profile``.
        clone: If True, returns a shallow-copied dict with a cloned noised tensor.

    Returns:
        Updated dict with noised input tensor for the selected branch.
    """
    if branch not in {"pcpt", "both", "gen"}:
        raise ValueError("branch must be one of: 'pcpt', 'both', 'gen'")

    out = dict(tensors) if clone else tensors

    if branch == "pcpt":
        if "masked_expr" not in out or "gene_key_padding_mask" not in out:
            raise KeyError(
                "Perceptual branch expects keys 'masked_expr' and 'gene_key_padding_mask'."
            )
        input_key = "masked_expr"
        pad_key = "gene_key_padding_mask"
    else:
        if "pcpt_expr" not in out or "pcpt_key_padding_mask" not in out:
            raise KeyError(
                "Both/gen branch expects keys 'pcpt_expr' and 'pcpt_key_padding_mask'."
            )
        input_key = "pcpt_expr"
        pad_key = "pcpt_key_padding_mask"

    x = out[input_key]
    key_padding_mask = out[pad_key]
    valid_mask = ~key_padding_mask

    if keep_first_n_tokens > 0:
        valid_mask = valid_mask.clone()
        valid_mask[:, :keep_first_n_tokens] = False

    noised = noise_log1p_profile(
        x,
        noise_level=noise_level,
        mode=mode,
        clamp_min=clamp_min,
        valid_mask=valid_mask,
    )
    out[input_key] = noised
    return out


def noise_count_profile(mat: Tensor, dropout: float, method="new", randsamp=False) -> Tensor:
    """
    adopted from scPRINT2 downsample_profile
    
    This function downsamples the expression profile of a given single cell RNA matrix.

    The noise is applied based on the renoise parameter,
    the total counts of the matrix, and the number of genes. The function first calculates the noise
    threshold (scaler) based on the renoise parameter. It then generates an initial matrix count by
    applying a Poisson distribution to a random tensor scaled by the total counts and the number of genes.
    The function then models the sampling zeros by applying a Poisson distribution to a random tensor
    scaled by the noise threshold, the total counts, and the number of genes. The function also models
    the technical zeros by generating a random tensor and comparing it to the noise threshold. The final
    matrix count is calculated by subtracting the sampling zeros from the initial matrix count and
    multiplying by the technical zeros. The function ensures that the final matrix count is not less
    than zero by taking the maximum of the final matrix count and a tensor of zeros. The function
    returns the final matrix count.

    Args:
        mat (torch.Tensor): The input matrix.
        dropout (float): The renoise parameter.

    Returns:
        torch.Tensor: The matrix count after applying noise.
    """
    # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
    # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
    # https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02601-5#:~:text=Zero%20measurements%20in%20scRNA%2Dseq,generation%20of%20scRNA%2Dseq%20data.
    if randsamp:
        dropout = torch.rand(mat.shape[0], device=mat.device) * dropout
        dropout = (
            dropout.unsqueeze(1)
            if len(mat.shape) == 2
            else dropout.unsqueeze(1).unsqueeze(1)
        )
    if method == "old":
        totcounts = mat.sum(-1)
        ngenes = mat.shape[-1]
        tnoise = 1 - (1 - dropout) ** (1 / 2)
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson(
            torch.rand(mat.shape, device=mat.device)
            * ((tnoise * totcounts.unsqueeze(-1)) / (0.5 * ngenes))
        ).int()
        # we model the technical zeros (dropping 50% of the genes)
        drop = (torch.rand(mat.shape, device=mat.device) > tnoise).int()

        mat = (mat - res) * drop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    elif method == "jules":
        scaler = (1 - dropout) ** (1 / 2)
        notdrop = (
            torch.rand(
                mat.shape,
                device=mat.device,
            )
            < scaler
        ).int()
        notdrop[mat == 0] = 0
        # apply the dropout after the poisson, right?
        return notdrop * torch.poisson(mat * scaler)
    elif method == "new":
        dropout = dropout * 1.1
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson((mat * (dropout / 2))).int()
        # we model the technical zeros (dropping 50% of the genes)
        notdrop = (torch.rand(mat.shape, device=mat.device) >= (dropout / 2)).int()
        mat = (mat - res) * notdrop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    else:
        raise ValueError(f"method {method} not recognized")