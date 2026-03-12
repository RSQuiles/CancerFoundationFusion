from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np
from cancerfoundation.data.preprocess import binning


@dataclass
class BulkSCCollator:
    """Collator for mixed bulk + single-cell pretraining.

    Works with ``BulkSCBatchSampler``, which yields flat lists of indices
    laid out as repeating blocks of
    ``[sc_0, ..., sc_{n-1}, bulk_0, ..., bulk_{m-1}]``.

    The collator:
      1. Splits the flat sample list back into per-pair SC / bulk groups.
      2. Aggregates each SC group into a **pseudobulk** profile.
      3. Processes the real bulk sample(s) with the same pipeline.
      4. Returns paired (pseudobulk, bulk) tensors ready for the model.

    Parameters
    ----------
    n_sc_samples : int
        Number of SC rows per pseudobulk (must match ``BulkSCBatchSampler``).
    n_bulk_samples : int
        Number of bulk rows per pair (must match ``BulkSCBatchSampler``).
    pad_token_id : int
        Token id used for gene padding.
    pad_value : float
        Value used for expression padding.
    max_length : int
        Maximum sequence length after sampling/truncation + padding.
    n_bins : int | None
        Number of quantile bins.  Set together with ``do_binning=True``.
    do_binning : bool
        Whether to bin expression values.
    normalise_bins : bool
        Scale binned values to [0, 1].
    sampling : bool
        Random-sample genes when length > max_length (else truncate).
    keep_first_n_tokens : int
        Leading tokens (e.g. CLS) preserved during sampling.
    mask_ratio : float
        MLM mask probability.
    mask_value : int
        Value written into masked positions.
    aggregation : str
        ``"sum"`` or ``"mean"`` for SC → pseudobulk.
    conditions : list[str] | None
        Condition columns to propagate.
    match_fn : callable | None
        ``(pseudobulk_dict, list[bulk_dict]) -> bulk_dict``
        Picks the best-matching bulk sample for a given pseudobulk.
        If ``None``, the first bulk sample (already random) is used.
    """

    n_sc_samples: int
    n_bulk_samples: int
    pad_token_id: int
    pad_value: float = -2
    max_length: int = 2048
    n_bins: Optional[int] = None
    do_binning: bool = True
    normalise_bins: bool = False
    sampling: bool = True
    keep_first_n_tokens: int = 1
    mask_ratio: float = 0.15
    mask_value: int = -1
    aggregation: str = "sum"
    conditions: Optional[List[str]] = None
    match_fn: Optional[Callable] = None

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a flat list of samples from ``BulkSCBatchSampler``.

        The list is ordered as repeating blocks of size
        ``n_sc_samples + n_bulk_samples``.

        Returns
        -------
        dict[str, Tensor]:
            pseudobulk_gene             (B, L)
            pseudobulk_expr             (B, L)
            pseudobulk_masked_expr      (B, L)
            pseudobulk_key_padding_mask (B, L)
            bulk_gene                   (B, L)
            bulk_expr                   (B, L)
            bulk_masked_expr            (B, L)
            bulk_key_padding_mask       (B, L)
            conditions                  dict[str, (B,)]  (if configured)

        where B = batch_size (number of pairs).
        """
        pair_size = self.n_sc_samples + self.n_bulk_samples
        n_pairs = len(examples) // pair_size

        pseudobulk_genes_list = []
        pseudobulk_expr_list = []
        bulk_genes_list = []
        bulk_expr_list = []
        bulk_samples_for_cond = []

        for i in range(n_pairs):
            offset = i * pair_size
            sc_samples = examples[offset : offset + self.n_sc_samples]
            bulk_samples = examples[offset + self.n_sc_samples : offset + pair_size]

            # --- 1. aggregate SC → pseudobulk ---
            pb_genes, pb_expr = self._aggregate_sc(sc_samples)

            # --- 2. pick the matching bulk sample ---
            if self.match_fn is not None:
                bulk = self.match_fn(
                    {"genes": pb_genes, "expressions": pb_expr},
                    bulk_samples,
                )
            else:
                bulk = bulk_samples[0]

            pseudobulk_genes_list.append(pb_genes)
            pseudobulk_expr_list.append(pb_expr)
            bulk_genes_list.append(bulk["genes"])
            bulk_expr_list.append(bulk["expressions"])
            bulk_samples_for_cond.append(bulk)

        # --- 3. bin, sample/truncate, pad, mask  ---
        pb_batch = self._collate_branch(pseudobulk_genes_list, pseudobulk_expr_list)
        bulk_batch = self._collate_branch(bulk_genes_list, bulk_expr_list)

        data_dict = {
            "pseudobulk_gene": pb_batch["gene"],
            "pseudobulk_expr": pb_batch["expr"],
            "pseudobulk_masked_expr": pb_batch["masked_expr"],
            "pseudobulk_key_padding_mask": pb_batch["key_padding_mask"],
            "bulk_gene": bulk_batch["gene"],
            "bulk_expr": bulk_batch["expr"],
            "bulk_masked_expr": bulk_batch["masked_expr"],
            "bulk_key_padding_mask": bulk_batch["key_padding_mask"],
        }

        # --- 4. propagate conditions ---
        if self.conditions:
            cond_dict = {}
            for cond in self.conditions:
                cond_dict[cond] = torch.LongTensor(
                    [s.get(cond, 0) for s in bulk_samples_for_cond]
                )
            data_dict["conditions"] = cond_dict

        return data_dict

    # ------------------------------------------------------------------
    # SC → pseudobulk aggregation
    # ------------------------------------------------------------------

    def _aggregate_sc(
        self,
        sc_samples: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate single-cell samples into one pseudobulk profile.

        Gene sets across cells may differ (variable-length sparse rows), so we
        build the union of gene ids, sum (or average) expression values, and
        return aligned tensors.

        Returns (genes, expressions) tensors with CLS token at position 0.
        """
        # Collect all gene→expr mappings, skipping the CLS position
        k = self.keep_first_n_tokens
        gene_expr: dict[int, float] = {}

        for sample in sc_samples:
            genes = sample["genes"][k:].numpy()
            exprs = sample["expressions"][k:].numpy()
            for g, e in zip(genes, exprs):
                g = int(g)
                if g == self.pad_token_id:
                    continue
                gene_expr[g] = gene_expr.get(g, 0.0) + float(e)

        if self.aggregation == "mean":
            # Second pass: count occurrences
            gene_count: dict[int, int] = {}
            for sample in sc_samples:
                genes = sample["genes"][k:].numpy()
                for g in genes:
                    g = int(g)
                    if g == self.pad_token_id:
                        continue
                    gene_count[g] = gene_count.get(g, 0) + 1
            for g in gene_expr:
                gene_expr[g] /= gene_count[g]

        gene_ids = np.array(list(gene_expr.keys()), dtype=np.int64)
        expr_vals = np.array(list(gene_expr.values()), dtype=np.float32)

        # Prepend CLS token (mirrors SingleCellDataset)
        cls_id = sc_samples[0]["genes"][0].item()
        gene_ids = np.insert(gene_ids, 0, cls_id)
        expr_vals = np.insert(expr_vals, 0, self.pad_value)

        return (
            torch.from_numpy(gene_ids),
            torch.tensor(expr_vals, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # shared collation logic (bin → sample/truncate → pad → mask)
    # ------------------------------------------------------------------

    def _collate_branch(
        self,
        genes_list: List[torch.Tensor],
        expr_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Process one branch (pseudobulk or bulk) through the standard
        bin → sample → pad → mask pipeline."""

        max_ori_len = max(len(g) for g in genes_list)
        _max_length = min(self.max_length, max_ori_len)

        padded_genes = []
        padded_expr = []

        for genes, expr in zip(genes_list, expr_list):
            if self.do_binning and self.n_bins is not None:
                if self.normalise_bins:
                    expr[self.keep_first_n_tokens :] = (
                        binning(expr[self.keep_first_n_tokens :], self.n_bins)
                        / self.n_bins
                    )
                else:
                    expr[self.keep_first_n_tokens :] = binning(
                        expr[self.keep_first_n_tokens :], self.n_bins
                    )

            genes, expr = self._sample_or_truncate_plus_pad(genes, expr, _max_length)
            padded_genes.append(genes)
            padded_expr.append(expr)

        padded_genes = torch.stack(padded_genes, dim=0)
        padded_expr = torch.stack(padded_expr, dim=0)

        key_padding_mask = padded_genes.eq(self.pad_token_id)
        masked_expr = self._mask(padded_expr, self.keep_first_n_tokens)

        return {
            "gene": padded_genes,
            "expr": padded_expr,
            "masked_expr": masked_expr,
            "key_padding_mask": key_padding_mask,
        }

    # ------------------------------------------------------------------
    # sampling / truncation / padding / masking — mirrors AnnDataCollator
    # ------------------------------------------------------------------

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions

        if len(genes) > max_length:
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            return genes[:max_length], expressions[:max_length]

        return self._pad(genes, expressions, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        _n = self.keep_first_n_tokens
        if _n == 0:
            idx = torch.randperm(len(genes))[:max_length]
            return genes[idx], expressions[idx]
        idx = torch.randperm(len(genes) - _n)[: max_length - _n]
        idx = torch.cat([torch.arange(_n), idx + _n], dim=0)
        return genes[idx], expressions[idx]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        pad_len = max_length - len(genes)
        genes = torch.cat(
            [
                genes,
                torch.full((pad_len,), self.pad_token_id, dtype=genes.dtype),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full((pad_len,), self.pad_value, dtype=expressions.dtype),
            ]
        )
        return genes, expressions

    def _mask(
        self, expressions: torch.Tensor, keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        if keep_first_n_tokens > 0:
            masked_tail = self._mask(
                expressions[:, keep_first_n_tokens:], keep_first_n_tokens=0
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], masked_tail], dim=1)

        prob = torch.full(expressions.shape, self.mask_ratio)
        prob[expressions.eq(self.pad_value)] = 0
        mask = torch.bernoulli(prob).bool()
        return expressions.masked_fill(mask, self.mask_value)
