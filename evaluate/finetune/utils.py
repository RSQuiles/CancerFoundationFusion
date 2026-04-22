from __future__ import annotations
from __future__ import print_function

from pathlib import Path
import random
import anndata as ad
from anndata import AnnData
import numpy as np
import ftplib
import logging
import os
from typing import List, Optional, Union
import gc

import json
import os
import struct
import sys
import platform
import re
import time
import traceback
import requests
import socket
import random
import math
import numpy as np
import torch
import logging
import datetime
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

## From Enrico's implementation:

def seed_all(seed_value, cuda_deterministic=False):
    """
    set all random seeds
    """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, world_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = world_size
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
def distributed_concat(tensor, num_total_examples, world_size):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def get_reduced(tensor, current_device, dest_device, world_size):
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / world_size
    return tensor_mean

# Implemented by RAFA

# ============================================================================
# Shared Utilities for Bulk RNA-seq Downstream Tasks
# (used by SurvivalTask, ProteomePredTask, DrugSensitivityTask)
# ============================================================================


def parquet_to_adata(df: pd.DataFrame, gene_cols: list[str]) -> ad.AnnData:
    """
    Wrap a bulk RNA-seq Parquet DataFrame (samples × genes) into an AnnData
    object so it can be passed directly to CancerFoundation.embed().

    The three bulk downstream tasks (SurvBoard, CPTAC, BeatAML) all store
    expression data as Parquet files with sample IDs as the DataFrame index and
    HGNC gene symbols as column names. This utility creates a minimal AnnData
    without copying the data matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Rows = samples (index = sample IDs), columns include gene symbols.
        Non-gene columns (e.g. OS_days, OS_event, cancer_type) must be excluded
        from gene_cols before calling; they are ignored here.
    gene_cols : list[str]
        Subset of df.columns to use as the expression feature matrix.

    Returns
    -------
    ad.AnnData
        X = float32 expression matrix, obs_names = sample IDs, var_names = genes.
    """
    X = df[gene_cols].to_numpy(dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.obs_names = df.index.astype(str).tolist()
    adata.var_names = [str(g) for g in gene_cols]
    return adata


def _translate_gene_symbols(gene_symbols: list[str]) -> list[str]:
    """
    Translate HGNC gene symbols to the naming convention used by the
    CancerFoundation vocabulary.

    The model vocab is mostly HGNC but contains older Ensembl-style lncRNA
    names (e.g. RP11-554J4.1, AC020910.4) that have since been retired or
    renamed. CPTAC uses current HGNC symbols throughout.

    Currently a pass-through — genes not in the vocab are silently dropped
    by CancerFoundation.embed() during its internal gene intersection, so
    the model still runs correctly at the cost of reduced gene coverage.
    Replace this with a mygene.info lookup or a pre-built alias table to
    recover renamed symbols.
    """
    return gene_symbols