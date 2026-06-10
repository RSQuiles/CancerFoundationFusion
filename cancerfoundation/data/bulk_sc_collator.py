from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from cancerfoundation.data.data_collator import AnnDataCollator


@dataclass
class BulkSCCollator(AnnDataCollator):
    """Mixed-modality collator built on top of ``AnnDataCollator``.

    It first assembles a unified sample list from:
    - single-cell samples
    - pseudobulk samples aggregated from SC subsets
    - real bulk samples

    Then it delegates objective-specific formatting (``pcpt``, ``gen``, ``both``)
    to ``AnnDataCollator.__call__``.
    """

    # Parent parameters
    normalise_bins: bool
    condition_token: bool
    do_padding: bool = True
    gene_key: str = "var_gene_token"
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    n_bins: Optional[int] = None
    do_binning: bool = False
    probabilistic_augmentation: bool = False
    mask_ratio: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    data_style: str = "pcpt"
    # Must be defined to account for data modality
    conditions: List[str] = None
    cls_predictions: List[str] = None
    zero_percentages: Optional[List[float]] = None

    # New parameters for bulk/SC collation
    batch_size: int = 128
    bulk_ratio: float = 0.3
    pb_ratio: float = 0.3
    n_sc_per_pseudobulk: int = 10
    aggregation: str = "sum"
    match_fn: Optional[Callable] = None
    agg_consistency: bool = False # Determines whether to include the sc_for_pb samples in the batch
    paired_column: Optional[str] = None  # obs column carrying pair IDs; enables is_paired_batch detection
    verbose: bool = False

    def __post_init__(self):
        """
        We must take into account the structure of the batch, with the following structure:
            1. Single-Cell samples
            2. Single-Cell samples to generate pseudobulk
            3. Real Bulk samples
        """
        # Determine binning
        self.do_binning = self.n_bins is not None

        super().__post_init__()
        self.n_bulk = round(self.batch_size * self.bulk_ratio)
        self.n_pb = round(self.batch_size * self.pb_ratio)
        self.n_sc = self.batch_size - self.n_bulk - self.n_pb
        self.raw_batch_size = (
            self.n_bulk + self.n_sc + self.n_pb * self.n_sc_per_pseudobulk
        )

        # Confirm batch composition
        print("\nBatch composition at the collator level")
        print("batch_size:", self.batch_size)
        print("n_bulk:", self.n_bulk)
        print("n_pb:", self.n_pb)
        print("n_sc:", self.n_sc)
        print("raw_batch_size:", self.raw_batch_size)
        print("sum_logical:", self.n_bulk + self.n_pb + self.n_sc)

        if self.n_bulk <= 0:
            raise ValueError(f"n_bulk_samples must be positive, got {self.n_bulk}.")
        if self.n_sc_per_pseudobulk <= 0:
            raise ValueError(
                "n_sc_per_pseudobulk must be positive, got "
                f"{self.n_sc_per_pseudobulk}."
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(examples) == self.raw_batch_size:
            n_sc_per_pb = self.n_sc_per_pseudobulk
        elif len(examples) == self.batch_size:
            n_sc_per_pb = 1  # paired batch: one precomputed PB row per slot
        else:
            raise ValueError(
                f"Expected {self.raw_batch_size} or {self.batch_size} samples, "
                f"got {len(examples)}."
            )

        sc_samples = [dict(sample) for sample in examples[: self.n_sc]]
        sc_for_pb_samples = [
            dict(sample)
            for sample in examples[
                self.n_sc : self.n_sc + self.n_pb * n_sc_per_pb
            ]
        ]
        bulk_samples = [
            dict(sample)
            for sample in examples[self.n_sc + self.n_pb * n_sc_per_pb :]
        ]

        # Detect paired batch: in paired sampling, sc_for_pb_samples holds precomputed
        # PB rows (not SC cells), each matched to its corresponding bulk row.
        # ORDERING INVARIANT: sample_paired_batch indexes both paired_pb_indices and
        # paired_bulk_indices with the same pair_positions array, so element i of
        # sc_for_pb_samples is always paired with element i of bulk_samples.
        # The set-equality check below catches any future reordering before it silently
        # corrupts the loss; the element-wise check then confirms the invariant holds.
        is_paired = False
        if self.paired_column is not None and n_sc_per_pb == 1:
            pb_pair_ids   = [int(s.get(self.paired_column, 0)) for s in sc_for_pb_samples]
            bulk_pair_ids = [int(s.get(self.paired_column, 0)) for s in bulk_samples]
            nonzero_pb   = [p for p in pb_pair_ids   if p != 0]
            nonzero_bulk = [b for b in bulk_pair_ids if b != 0]
            if (nonzero_pb
                and set(nonzero_pb) == set(nonzero_bulk)
                and all(p == b for p, b in zip(pb_pair_ids, bulk_pair_ids))):
                is_paired = True
                if self.verbose:
                    print(f"Sampled paired indexes:\n- PB: {pb_pair_ids}\n- Bulk: {bulk_pair_ids}")

        pseudobulk_samples: List[Dict[str, Any]] = []
        sc_pseudobulk_index: List[int] = []
        pseudobulk_sizes: List[int] = []

        if is_paired:
            # Precomputed PB rows pass through unchanged; no aggregation needed.
            # Order matches bulk_samples because sample_paired_batch indexes both
            # paired_pb_indices and paired_bulk_indices with the same pair_positions,
            # so pseudobulk_samples[i] is guaranteed to be paired with bulk_samples[i].
            pseudobulk_samples = list(sc_for_pb_samples)
            pseudobulk_sizes = [1] * len(sc_for_pb_samples)
        else:
            for pb_idx, start in enumerate(
                range(0, len(sc_for_pb_samples), n_sc_per_pb)
            ):
                chunk = sc_for_pb_samples[start : start + n_sc_per_pb]
                pb_genes, pb_expr = self._aggregate_sc(chunk)
                pb_sample = {"genes": pb_genes, "expressions": pb_expr}
                self._fill_missing_conditions(pb_sample, chunk)
                pseudobulk_samples.append(pb_sample)
                pseudobulk_sizes.append(len(chunk))
                sc_pseudobulk_index.extend([pb_idx] * len(chunk))

        unified_samples: List[Dict[str, Any]] = []
        unified_modalities: List[int] = []
        unified_is_real: List[int] = []
        unified_pseudobulk_index: List[int] = []
        unified_is_sc_for_pb: List[int] = []  # mask for sc_for_pb samples

        # 0 -> real bulk
        for sample in bulk_samples:
            unified_samples.append(sample)
            unified_modalities.append(0)
            unified_is_real.append(1)
            unified_pseudobulk_index.append(-1)
            unified_is_sc_for_pb.append(0)

        # 1 -> sc
        for sc_idx, sample in enumerate(sc_samples):
            unified_samples.append(sample)
            unified_modalities.append(1)
            unified_is_real.append(1)
            unified_pseudobulk_index.append(-1)
            unified_is_sc_for_pb.append(0)

        # 2 -> pseudobulk (aggregated SC or precomputed; real only when paired)
        for pb_idx, sample in enumerate(pseudobulk_samples):
            unified_samples.append(sample)
            unified_modalities.append(2)
            unified_is_real.append(1 if is_paired else 0)
            unified_pseudobulk_index.append(pb_idx)
            unified_is_sc_for_pb.append(0)

        # 3 (1) -> sc for pb
        if self.agg_consistency and not is_paired:
            for sc_idx, sample in enumerate(sc_for_pb_samples):
                unified_samples.append(sample)
                unified_modalities.append(1)
                unified_is_real.append(1)
                unified_pseudobulk_index.append(sc_pseudobulk_index[sc_idx])
                unified_is_sc_for_pb.append(1)

        """
        # 4 -> matched bulk (optional)
        if self.match_fn is not None:
            for pb_idx, pseudobulk in enumerate(pseudobulk_samples):
                matched = dict(self.match_fn(pseudobulk, bulk_samples))
                unified_samples.append(matched)
                unified_modalities.append(4)
                unified_is_real.append(1)
                unified_pseudobulk_index.append(pb_idx)
        """

        # Delegate objective-specific collation to AnnDataCollator
        data_dict: Dict[str, Any] = super().__call__(unified_samples)

        existing_conditions = data_dict.get("conditions", {})
        if not isinstance(existing_conditions, dict):
            existing_conditions = {}
        data_dict["conditions"] = {
            **existing_conditions,
            "modality": torch.LongTensor(unified_modalities),
        }

        # Additional structural metadata for mixed losses
        data_dict["is_real_sample"] = torch.LongTensor(unified_is_real)
        data_dict["is_sc_for_pb"] = torch.LongTensor(unified_is_sc_for_pb)
        data_dict["sc_pseudobulk_index"] = torch.LongTensor(
            sc_pseudobulk_index
        )  # for aggregation consistency losses, local for sc_for_pb_samples
        data_dict["sample_pseudobulk_index"] = torch.LongTensor(
            unified_pseudobulk_index
        )  # similar to above, but for all samples in the batch
        data_dict["pseudobulk_sizes"] = torch.LongTensor(pseudobulk_sizes)
        data_dict["is_paired_batch"] = torch.tensor(is_paired, dtype=torch.bool)

        return data_dict

    def _fill_missing_conditions(
        self, pb_sample: Dict[str, Any], sc_samples: List[Dict[str, Any]]
    ) -> None:
        """
        Fill missing conditions from generated pseudobulk samples, borrowing from
        underlying single-cell samples
        """
        if not self.conditions:
            return
        for cond in self.conditions:
            if cond not in pb_sample:
                pb_sample[cond] = self._average_condition_value(sc_samples, cond)

    def _average_condition_value(self, samples: List[Dict[str, Any]], condition: str):
        """Return the most frequent value of *condition* across the given samples."""
        counts: dict = {}
        max_count = 0
        max_value = None
        for sample in samples:
            assert condition in sample, f"{condition} not present in some of the samples"
            val = sample[condition]
            counts[val] = counts.get(val, 0) + 1
            if counts[val] > max_count:
                max_value = val
                max_count = counts[val]
        return max_value

    def _aggregate_sc(
        self,
        sc_samples: List[Dict[str, Any]],
        counts: bool = False,
        rank_normalise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.keep_first_n_tokens
        """
        Aggregate a list of single-cell token sequences into a single pseudobulk
        in the same sparse (gene_ids, expressions) format used by individual cells.
        The log1p values are first mapped back to count space via expm1, then aggregated by gene ID.

        Each cell can be rank-normalised to unit sum (via expm1 → proportion) before
        aggregation, so all cells contribute equally regardless of sequencing depth.
        The result is re-normalised to log1p(CPM) to match the model's input
        format. Zero entries are dropped and the CLS token is prepended.
        """

        # Per-cell mapping to count space and normalization
        genes_list = []
        exprs_list = []

        for s in sc_samples:
            genes = s["genes"][k:].detach().cpu().numpy().astype(np.int64)
            exprs = s["expressions"][k:].detach().cpu().numpy().astype(np.float64)

            # Mask padding
            valid = genes != self.pad_token_id
            genes = genes[valid]
            exprs = exprs[valid]

            if len(genes) == 0:
                continue

            # Map to count space: log1p → expm1
            if not counts:
                exprs = np.expm1(exprs)

            # Rank normalize: expm1 → normalize to proportions
            # This way each cell contributes equally to the pseudobulk, regardless of sequencing depth or previous normalization
            if rank_normalise:
                cell_sum = exprs.sum()
                if cell_sum > 0:
                    exprs /= cell_sum  # each cell now sums to 1

            genes_list.append(genes)
            exprs_list.append(exprs)

        if not genes_list:
            gene_ids  = np.array([], dtype=np.int64)
            expr_vals = np.array([], dtype=np.float32)
        else:
            # Concatenate normalized cells and use bincount to scatter-add
            all_genes = np.concatenate(genes_list)
            all_exprs = np.concatenate(exprs_list)

            n_bins   = int(all_genes.max()) + 1
            # Sums by gene index and stores sum at corresponding position in expr_sum
            expr_sum = np.bincount(all_genes, weights=all_exprs, minlength=n_bins)

            # Re-normalize to CP10K → log1p
            total = expr_sum.sum()
            if total > 0:
                expr_sum = expr_sum / total * 1e4
            expr_sum = np.log1p(expr_sum)

            expressed = expr_sum != 0
            gene_ids  = np.where(expressed)[0].astype(np.int64)
            expr_vals = expr_sum[expressed].astype(np.float32)

        # Prepend CLS token
        cls_id    = int(sc_samples[0]["genes"][0].item())
        gene_ids  = np.insert(gene_ids,  0, cls_id)
        expr_vals = np.insert(expr_vals, 0, self.pad_value)

        return torch.from_numpy(gene_ids), torch.tensor(expr_vals, dtype=torch.float32)
