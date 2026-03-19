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
    do_binning: bool = True
    probabilistic_augmentation: bool = False
    mask_ratio: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    data_style: str = "pcpt"
    n_bins: int = None
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

    def __post_init__(self):
        """
        We must take into account the structure of the batch, with the following structure:
            1. Single-Cell samples
            2. Single-Cell samples to generate pseudobulk
            3. Real Bulk samples
        """
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
        print("sum logical:", self.n_bulk + self.n_pb + self.n_sc)

        if self.n_bulk <= 0:
            raise ValueError(f"n_bulk_samples must be positive, got {self.n_bulk}.")
        if self.n_sc_per_pseudobulk <= 0:
            raise ValueError(
                "n_sc_per_pseudobulk must be positive, got "
                f"{self.n_sc_per_pseudobulk}."
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(examples) != self.raw_batch_size:
            raise ValueError(
                f"Expected {self.raw_batch_size} samples, got {len(examples)}."
            )

        sc_samples = [dict(sample) for sample in examples[: self.n_sc]]
        sc_for_pb_samples = [
            dict(sample)
            for sample in examples[
                self.n_sc : self.n_sc + self.n_pb * self.n_sc_per_pseudobulk
            ]
        ]
        bulk_samples = [
            dict(sample)
            for sample in examples[self.n_sc + self.n_pb * self.n_sc_per_pseudobulk :]
        ]

        pseudobulk_samples: List[Dict[str, Any]] = []
        sc_pseudobulk_index: List[int] = []
        pseudobulk_sizes: List[int] = []

        for pb_idx, start in enumerate(
            range(0, len(sc_for_pb_samples), self.n_sc_per_pseudobulk)
        ):
            chunk = sc_for_pb_samples[start : start + self.n_sc_per_pseudobulk]
            pb_genes, pb_expr = self._aggregate_sc(chunk)
            pb_sample = {"genes": pb_genes, "expressions": pb_expr}
            self._fill_missing_conditions(pb_sample, chunk)
            pseudobulk_samples.append(pb_sample)
            pseudobulk_sizes.append(len(chunk))
            # To which pb does each sc sample belong
            sc_pseudobulk_index.extend([pb_idx] * len(chunk))

        unified_samples: List[Dict[str, Any]] = []
        unified_modalities: List[int] = []
        unified_is_real: List[int] = []
        unified_pseudobulk_index: List[int] = []

        # 0 -> real bulk
        for sample in bulk_samples:
            unified_samples.append(sample)
            unified_modalities.append(0)
            unified_is_real.append(1)
            unified_pseudobulk_index.append(-1)

        # 1 -> sc
        for sc_idx, sample in enumerate(sc_samples):
            unified_samples.append(sample)
            unified_modalities.append(1)
            unified_is_real.append(1)
            unified_pseudobulk_index.append(-1)

        # 2 -> pseudobulk
        for pb_idx, sample in enumerate(pseudobulk_samples):
            unified_samples.append(sample)
            unified_modalities.append(2)
            unified_is_real.append(0)
            unified_pseudobulk_index.append(pb_idx)

        # 3 (1) -> sc for pb
        # Avoid passing this to the model for initial runs
        """
        for sc_idx, sample in enumerate(sc_for_pb_samples):
            unified_samples.append(sample)
            unified_modalities.append(1)
            unified_is_real.append(1)
            unified_pseudobulk_index.append(sc_pseudobulk_index[sc_idx])

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
        data_dict["sc_pseudobulk_index"] = torch.LongTensor(
            sc_pseudobulk_index
        )  # for aggregation consistency losses, local for sc_for_pb_samples
        data_dict["sample_pseudobulk_index"] = torch.LongTensor(
            unified_pseudobulk_index
        )  # similar to above, but for all samples in the batch
        data_dict["pseudobulk_sizes"] = torch.LongTensor(pseudobulk_sizes)

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
        """
        Return the "average" value for a given condition and a set of samples, preferrably related
        """
        values = {}
        max_count = 0
        max_value = 0
        # Determine the cardinalities and choose the most prevalent value
        for sample in samples:
            assert (
                condition in sample
            ), f"{condition} not present in some of the samples"
            if sample[condition] not in values:
                values[sample[condition]] = 1
            else:
                ++values[sample[condition]]
            # Retrieve max
            if values[sample[condition]] > max_count and sample[condition] != max_value:
                max_value = sample[condition]
                max_count = values[sample[condition]]
        return max_value

    def _aggregate_sc(
        self,
        sc_samples: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.keep_first_n_tokens
        gene_expr: Dict[int, float] = {}
        gene_count: Dict[int, int] = {}

        for sample in sc_samples:
            genes = sample["genes"][k:].detach().cpu().numpy()
            exprs = sample["expressions"][k:].detach().cpu().numpy()
            for g, e in zip(genes, exprs):
                g = int(g)
                if g == self.pad_token_id:
                    continue
                # Sum over all samples
                gene_expr[g] = gene_expr.get(g, 0.0) + float(e)
                gene_count[g] = gene_count.get(g, 0) + 1

        if self.aggregation == "mean":
            for g in gene_expr:
                gene_expr[g] /= gene_count[g]

        gene_ids = np.array(list(gene_expr.keys()), dtype=np.int64)
        expr_vals = np.array(list(gene_expr.values()), dtype=np.float32)

        cls_id = int(sc_samples[0]["genes"][0].item())
        gene_ids = np.insert(gene_ids, 0, cls_id)
        expr_vals = np.insert(expr_vals, 0, self.pad_value)

        return torch.from_numpy(gene_ids), torch.tensor(expr_vals, dtype=torch.float32)
