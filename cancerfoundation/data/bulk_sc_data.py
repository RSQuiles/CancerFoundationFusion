from typing import Optional

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler, Subset
from typing import Union, List, Dict

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

from .dataset import DatasetDir
from .utils import concat_categorical_codes


class BulkSCDataset(Dataset):
    """Dataset that serves individual bulk and single-cell samples from a
    shared memory-mapped store — exactly like ``SingleCellDataset`` but with
    per-row modality and group metadata exposed.

    Each ``__getitem__`` returns **one** sample (bulk *or* SC).  Batch
    composition (grouping SC cells into pseudobulk and pairing with bulk)
    is handled by ``BulkSCBatchSampler`` + ``BulkSCCollator``.

    The ``obs.parquet`` must contain a ``modality_column`` (values
    ``bulk_label`` / ``sc_label``).  An optional ``pb_group_column``
    (e.g. ``"tissue_general"``) groups SC cells for tissue-aware pseudobulk sampling.

    Parameters
    ----------
    data_dir : str | Path
        Root ``DatasetDir`` (``vocab.json``, ``obs.parquet``,
        ``mapping.json``, ``mem.map/``).
    modality_column : str
        Column distinguishing bulk from SC rows.
    bulk_label, sc_label : str
        Values in ``modality_column``.
    pb_group_column : str | None
        obs column used to group SC cells for tissue-aware pseudobulk sampling.
        Cells aggregated into a single pseudobulk are drawn exclusively from
        one group (e.g. one tissue). Bulk and SC tissue labels need not match.
    pad_value : float
        Value placed at the CLS position.
    obs_columns : list[str] | None
        Extra metadata columns to include per sample.
    balance : bool
        Whether to prepare for balanced sampling based on dataset labels.
    balance_labels : str | list[str] | None
        Column(s) to use for balanced sampling labels. If None, defaults to all obs columns
    """

    GENE_ID = "_cf_gene_id"
    CLS_TOKEN = "<cls>"
    PAD_TOKEN = "<pad>"

    def __init__(
        self,
        data_dir: str | Path,
        modality_column: str = "modality",
        bulk_label: str = "bulk",
        sc_label: str = "sc",
        pb_label: Optional[str] = "pseudobulk",
        pb_group_column: Optional[str] = None,
        paired_column: Optional[str] = "paired",
        pb_id_column: Optional[str] = "pseudobulk_id",
        pad_value: float = -1.0,
        obs_columns: Optional[list[str]] = None,
        balance: Optional[bool] = False,
        balance_labels: Optional[Union[str, List[str]]] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.data_dir = DatasetDir(data_dir)
        self.vocab = self._load_vocab()
        self.pad_value = pad_value
        self.memmap = SingleCellMemMapDataset(str(self.data_dir.memmap_path))
        self.obs = pd.read_parquet(self.data_dir.obs_path)
        self.mapping = self._load_mapping()
        self.obs_columns = obs_columns

        self.verbose = verbose
        self.modality_column = modality_column
        # Kept so consumers can resolve the pseudobulk modality code from the mapping
        # (e.g. the CDD refresh pass, which synthesizes pseudobulk rows outside the
        # collator and must tag them with the right modality).
        self.pb_label = pb_label
        self.pb_group_column = pb_group_column
        self.paired_column = paired_column if (
            paired_column is not None and paired_column in self.obs.columns
        ) else None
        # Links a precomputed pseudobulk row to the single cells it was aggregated from.
        # Written by data_preprocess/reconstruct_pseudobulk_cell_map.py --rebuild onto
        # both the pseudobulk and the source-cell h5ads; absent for datasets built
        # without it, in which case aggregation consistency over precomputed PBs is off.
        self.pb_id_column = pb_id_column if (
            pb_id_column is not None and pb_id_column in self.obs.columns
        ) else None

        if self.verbose:
            print(f"MemMap ({str(self.data_dir.memmap_path)}) rows: {self.memmap.number_of_rows()}")
        if self.verbose:
            print(f"OBS parquet ({self.data_dir.obs_path}) rows: {self.obs.shape[0]}")
        assert self.memmap.number_of_rows() == self.obs.shape[0]
        assert modality_column in self.obs.columns

        # Pre-extract obs columns as plain numpy arrays for O(1) random access in __getitem__
        # (pandas .iloc on a 100M-row DataFrame is expensive — numpy array indexing is not)
        self._obs_arrays = {col: self.obs[col].to_numpy() for col in (obs_columns or [])}
        # Also pre-extract the pairing column (sampler use only — not served to the model)
        if self.paired_column is not None:
            self._obs_arrays[self.paired_column] = self.obs[self.paired_column].to_numpy()
        if self.pb_id_column is not None:
            self._obs_arrays[self.pb_id_column] = self.obs[self.pb_id_column].to_numpy()

        # Pre-compute index arrays per modality
        modality_vals = self.obs[modality_column].values
        bulk_code = self.mapping[modality_column][bulk_label]
        sc_code = self.mapping[modality_column][sc_label]

        self.bulk_indices = np.where(modality_vals == bulk_code)[0]
        self.sc_indices = np.where(modality_vals == sc_code)[0]

        assert len(self.bulk_indices) > 0, "No bulk samples found"
        assert len(self.sc_indices) > 0, "No SC samples found"

        # Precomputed pseudobulk rows (optional — always present when paired data exists)
        if pb_label is not None and pb_label in self.mapping.get(modality_column, {}):
            pb_code = self.mapping[modality_column][pb_label]
            self.pb_indices = np.where(modality_vals == pb_code)[0]
        else:
            self.pb_indices = np.empty(0, dtype=np.int64)

        # Build mapping: pair_id → SC indices of constituent cells.
        # Enables matched-SC sampling in paired batches.
        self.sc_pair_to_indices: dict[int, np.ndarray] = {}
        if self.paired_column is not None and self.paired_column in self._obs_arrays:
            paired_arr = self._obs_arrays[self.paired_column]
            sc_pair_ids = paired_arr[self.sc_indices]
            for pid in np.unique(sc_pair_ids[sc_pair_ids != 0]):
                self.sc_pair_to_indices[int(pid)] = self.sc_indices[sc_pair_ids == pid]
        # Build mapping: pseudobulk_id → SC indices of the cells it was aggregated from.
        # This is the precomputed-pseudobulk analogue of sc_pair_to_indices above, but a
        # strictly finer link: `paired` matches a PB to a *bulk* row at cell-line
        # granularity, while `pseudobulk_id` names the exact cells that were summed.
        self.sc_pb_to_indices: dict[int, np.ndarray] = {}
        self.pb_id_fill_code: Optional[int] = None
        if self.pb_id_column is not None:
            self._build_pb_id_index()
            if self.verbose and self.sc_pair_to_indices:
                print(f"Found {len(self.sc_pair_to_indices)} pair IDs with matched SC cells.")

        # SC-only group index pools for tissue-aware pseudobulk sampling
        if pb_group_column is not None:
            assert pb_group_column in self.obs.columns, (
                f"pb_group_column '{pb_group_column}' not found in obs"
            )
            group_vals = np.asarray(self.obs[pb_group_column].values)
            sc_group_vals = group_vals[self.sc_indices]
            self.sc_group_to_indices: Optional[dict] = {
                g: self.sc_indices[sc_group_vals == g]
                for g in np.unique(sc_group_vals)
            }
            assert len(self.sc_group_to_indices) > 0, "No SC groups found"
            # Bulk analog: group column value → bulk row indices. Enables
            # class-aware CDD batches (same tissue in both bulk and pseudobulk).
            bulk_group_vals = group_vals[self.bulk_indices]
            self.bulk_group_to_indices: Optional[dict] = {
                g: self.bulk_indices[bulk_group_vals == g]
                for g in np.unique(bulk_group_vals)
            }
            # Precomputed-pseudobulk analog: group column value → precomputed PB row
            # indices. This is the tissue-pure SOURCE pool used by --precomputed-pb mode
            # (the standard/class-aware batches draw PB rows directly from here instead
            # of aggregating SC on the fly). None when there are no precomputed PB rows.
            if len(self.pb_indices) > 0:
                pb_group_vals = group_vals[self.pb_indices]
                self.pb_group_to_indices: Optional[dict] = {
                    g: self.pb_indices[pb_group_vals == g]
                    for g in np.unique(pb_group_vals)
                }
            else:
                self.pb_group_to_indices = None
        else:
            self.sc_group_to_indices = None
            self.bulk_group_to_indices = None
            self.pb_group_to_indices = None

        # Label categories for balanced sampling
        self.balance = balance
        self.labels = None
        if balance:
            self.balance_labels = (
                balance_labels if balance_labels is not None else obs_columns
            )
            # Dictionary for the different modalities
            print(f"Generating label arrays for: {self.balance_labels}")
            self.labels = self.get_label_cats(self.balance_labels)

    # ------------------------------------------------------------------
    # Dataset interface — one row per call
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.memmap.number_of_rows()

    def __getitem__(self, index: int) -> dict:
        """Return a single sample with ``genes``, ``expressions``, and a
        tags for the different conditions accounted for."""
        try: 
            exp, genes = self.memmap.get_row_padded(
                index, return_features=True, feature_vars=[self.GENE_ID]
            )
        except IndexError as e:
            obs_row = self.obs.iloc[index] if hasattr(self, "obs") else "N/A"
            raise IndexError(
                f"IndexError at index={index}: {e}\n"
                f"obs row: {obs_row}"
            ) from e

        genes = np.insert(genes[0], 0, self.vocab[self.CLS_TOKEN])
        exp = np.insert(exp, 0, self.pad_value)

        data = {
            "expressions": torch.tensor(exp, dtype=torch.float32),
            "genes": torch.from_numpy(genes),
            # Original dataset row index (Subset forwards it unchanged); used by the
            # CDD target-label bank to look up per-bulk pseudo-labels.
            "_row_index": int(index),
        }

        # Additional conditions input to model (e.g. tissue type)
        for col in self.obs_columns:
            data[col] = self._obs_arrays[col][index]

        # Expose the pair ID so the collator can detect paired batches
        if self.paired_column is not None:
            data[self.paired_column] = int(self._obs_arrays[self.paired_column][index])

        # Expose the pseudobulk ID so the collator can verify that the SC cells it was
        # handed really do belong to the pseudobulk they are matched against
        if self.pb_id_column is not None:
            data[self.pb_id_column] = int(self._obs_arrays[self.pb_id_column][index])

        return data

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_pb_id_index(self) -> None:
        """Index the pseudobulk-id column: which SC rows belong to which pseudobulk.

        The values in ``obs.parquet`` are *category codes*, not the original ids —
        preprocessing re-encodes every column through
        ``convert_columns_to_categorical_with_mapping``. Codes are 1:1 with ids and
        shared across modalities (the encoder runs on the concatenated obs), so
        grouping by code is equivalent to grouping by id and nothing needs inverting.

        Rows coming from h5ads that never carried the column are filled with ``0`` by
        ``obs.reindex(..., fill_value=0)``, so the category ``"0"`` means "no
        pseudobulk" and is excluded here.
        """
        pb_id_arr = self._obs_arrays[self.pb_id_column]
        categories = self.mapping.get(self.pb_id_column, {})
        self.pb_id_fill_code = categories.get("0")

        # A purely numeric id space cannot express "no pseudobulk": the reindex fill of
        # 0 is indistinguishable from a genuine id 0, which would bind every unrelated
        # cell to that one pseudobulk. Preprocessing should write a string id (the
        # pseudobulk's sample_id) so the fill category is unambiguous.
        other = [c for c in categories if c != "0"]
        if self.pb_id_fill_code is not None and other and all(
            c.lstrip("-").isdigit() for c in other
        ):
            print(
                f"[WARNING] '{self.pb_id_column}' holds plain integer ids, so category "
                f"'0' is ambiguous: it is both the reindex fill for rows without a "
                f"pseudobulk and a valid id. Cells with id 0 will be dropped. Write the "
                f"column as a string id (e.g. the pseudobulk's sample_id) to fix this."
            )

        sc_pb_ids = pb_id_arr[self.sc_indices]
        if self.pb_id_fill_code is not None:
            valid = sc_pb_ids != self.pb_id_fill_code
        else:
            valid = np.ones(len(sc_pb_ids), dtype=bool)

        for code in np.unique(sc_pb_ids[valid]):
            self.sc_pb_to_indices[int(code)] = self.sc_indices[sc_pb_ids == code]

        if self.verbose and self.sc_pb_to_indices:
            print(
                f"Found {len(self.sc_pb_to_indices)} pseudobulk ids with constituent "
                f"SC cells."
            )

    def _load_mapping(self) -> dict:
        with self.data_dir.mapping_path.open("r") as f:
            return json.load(f)

    def _load_vocab(self) -> dict[str, int]:
        with open(self.data_dir.vocab_path, "r") as f:
            return json.load(f)

    def get_label_cats(
        self,
        obs_keys: Union[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """
        Get combined categorical codes for one or more label columns.

        Retrieves labels from the mapped dataset and combines them into a single
        categorical encoding. Useful for creating compound class labels for
        stratified sampling.

        Given the virtual separation between bulk and SC samples in relation to
        sampling, this method deals with these two different set indices independently

        Args:
            obs_keys (str | List[str]): Column name(s) to retrieve and combine.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping modality groups to arrays of integer codes representing the combined categories.
                Each array has shape (n_samples,) and contains integer codes corresponding to the unique combinations of the specified label columns for that modality group.
        """
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        labels = None
        for label_key in obs_keys:
            labels_to_str = self.get_merged_labels(label_key)
            if labels is None:
                labels = labels_to_str
            else:
                labels = {
                    key: concat_categorical_codes([labels[key], labels_to_str[key]])
                    for key in labels
                }
        return {key: np.array(label.codes) for key, label in labels.items()}

    def get_merged_labels(self, label_key: str) -> Dict[str, pd.Categorical]:
        """
        Get categorical labels for a given key as integer-coded Categoricals.

        Returns "sc" and "bulk" groups.
        """
        if label_key not in self.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in obs columns.")
        if label_key not in self.mapping:
            raise ValueError(f"Label key '{label_key}' not found in mapping.")
        all_codes = self.obs[label_key].to_numpy()
        n_cats = len(self.mapping[label_key])
        categories = np.arange(n_cats)
        return {
            "sc":   pd.Categorical.from_codes(all_codes[self.sc_indices],   categories=categories),
            "bulk": pd.Categorical.from_codes(all_codes[self.bulk_indices], categories=categories),
        }


# ======================================================================
# Subset reindexer
# ======================================================================

class SubsetReindexer:
    """Translates base-dataset indices to Subset-local indices via a dense LUT.

    Build once from the Subset's index array, then call ``remap`` for each
    index pool that needs to be translated.  The LUT is O(max_base_index) in
    memory (~800 MB at 100 M rows); call ``del reindexer`` when done.

    Parameters
    ----------
    subset_base_indices : np.ndarray
        The ``dataset.indices`` array of the ``torch.utils.data.Subset``.
    """

    def __init__(self, subset_base_indices: np.ndarray):
        max_idx = int(subset_base_indices.max()) + 1
        self._lut = np.full(max_idx, -1, dtype=np.int64)
        self._lut[subset_base_indices] = np.arange(len(subset_base_indices), dtype=np.int64)

    def remap(self, base_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(subset_local_indices, survival_mask)``.

        ``survival_mask`` is a boolean array aligned with ``base_indices``.
        Use it to filter any array that is positionally aligned with
        ``base_indices`` (e.g. a per-sample label array).
        """
        base_indices = np.asarray(base_indices)
        in_bounds = base_indices < len(self._lut)
        mapped = np.full(len(base_indices), -1, dtype=np.int64)
        mapped[in_bounds] = self._lut[base_indices[in_bounds]]
        mask = mapped >= 0
        return mapped[mask], mask

    def remap_dict(self, group_dict: dict) -> dict:
        """Remap each value array in ``group_dict``, dropping empty groups."""
        return {
            g: mapped
            for g, idxs in group_dict.items()
            if len(mapped := self.remap(np.asarray(idxs))[0]) > 0
        }


# ======================================================================
# Batch sampler
# ======================================================================


class BulkSCSampler(Sampler[list[int]]):
    """Yields batches made of bulk, pseudobulk and single-cell samples. The
    pseudobulk samples are generated from single-cells different from the ones in the batch,
    which can also be saved for later use in consistency losses.

    ``batch_size`` is interpreted as the total number of samples that get fed into the model
    to perform the autorregressive task.

    Parameters
    ----------
    dataset : BulkSCDataset
        Dataset providing global ``sc_indices`` and ``bulk_indices``.
        Group membership, if present, is ignored by this sampler.
    batch_size : int
        Total number of samples per batch.
    bulk_ratio : float
        The ratio of bulk samples in each batch.
    pb_ratio : float
        The ratio of pseudobulk samples in each batch (relative to the total batch size).
    n_sc_per_pb : int
        Number of single-cell samples to aggregate into each pseudobulk. These are drawn from the same pool as the single-cell samples in the batch, but are guaranteed to be different samples.
    drop_last : bool
        Drop the last incomplete batch.
    shuffle : bool
        Shuffle group order each epoch.
    balance : bool
        Whether to perform balanced sampling based on the dataset's labels.
    weight_scaler : float
        Scaling factor for label weights in balanced sampling. Higher values increase the relative weight of more common classes.
    num_workers : int
        Number of parallel workers to use for building class indices in balanced sampling.
    chunk_size : int
        Number of samples to process per chunk when building class indices in balanced sampling. Adjust based on available memory and dataset size.
    curiculum (int, optional): Curriculum learning parameter. If > 0, gradually
        increases sampling weight balance over epochs. Defaults to 0.
    replacement (bool, optional): Whether to sample with replacement when balanced=True. Defaults to True.
    epoch_size : int | None
        Absolute number of batches per epoch. Overrides ``epoch_coverage``.
    epoch_coverage : float
        How many passes over the bulk pool make up one epoch. The batch count is
        ``ceil(coverage * n_bulk_rows / n_bulk)``, so it scales with ``batch_size``:
        doubling the batch size halves the step count and leaves samples-per-epoch
        unchanged. Bulk rows are drawn independently per batch, so coverage 1.0
        means "as many draws as there are rows" (~63% of distinct rows in
        expectation), not "every row exactly once".

    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        bulk_ratio: float = 0.3,
        pb_ratio: float = 0.3,
        n_sc_per_pb: int = 5,
        drop_last: bool = True,
        shuffle: bool = True,
        balance: Optional[bool] = False,
        weight_scaler: float = 1.0,
        num_workers: int = 1,
        chunk_size: int = 1000,
        curiculum: int = 0,
        replacement: bool = True,
        epoch_size: Optional[int] = None,
        epoch_coverage: float = 1.0,
        precomputed_pb: bool = False,
        agg_consistency: bool = False,
        paired_sampling: bool = False,
        paired_every_n: int = 10,
        verbose: bool = False,
        seed: Optional[int] = None,
        world_size: int = 1,
        class_aware_cdd: bool = False,
        cdd_exclude_group_codes: Optional[list] = None,
        cdd_min_class_count: int = 2,
        n_cdd_classes: int = 8,
        cdd_bulk_class_frac: float = 0.6,
    ):
        # Account for the Subset resulting from random_split.
        # SubsetReindexer builds a LUT once; each remap() call also returns a
        # survival_mask aligned with the base pool — reused for label filtering.
        sc_mask = bulk_mask = None
        if isinstance(dataset, Subset):
            self.dataset = dataset
            self.subset_base_indices = np.asarray(dataset.indices)
            self.base_dataset = dataset.dataset

            reindexer = SubsetReindexer(self.subset_base_indices)
            self.bulk_indices, bulk_mask = reindexer.remap(self.base_dataset.bulk_indices)
            self.sc_indices,   sc_mask   = reindexer.remap(self.base_dataset.sc_indices)
            self.pb_indices,   _         = reindexer.remap(self.base_dataset.pb_indices)
            self.sc_group_to_indices = (
                reindexer.remap_dict(self.base_dataset.sc_group_to_indices)
                if self.base_dataset.sc_group_to_indices is not None
                else None
            )
            self.bulk_group_to_indices = (
                reindexer.remap_dict(self.base_dataset.bulk_group_to_indices)
                if getattr(self.base_dataset, "bulk_group_to_indices", None) is not None
                else None
            )
            self.pb_group_to_indices = (
                reindexer.remap_dict(self.base_dataset.pb_group_to_indices)
                if getattr(self.base_dataset, "pb_group_to_indices", None) is not None
                else None
            )
            self.sc_pair_to_indices = (
                reindexer.remap_dict(self.base_dataset.sc_pair_to_indices)
                if getattr(self.base_dataset, "sc_pair_to_indices", None)
                else {}
            )
            # Remapped, not filtered afterwards: a pseudobulk whose cells all landed in
            # the other split drops out here, which is exactly what should happen.
            self.sc_pb_to_indices = (
                reindexer.remap_dict(self.base_dataset.sc_pb_to_indices)
                if getattr(self.base_dataset, "sc_pb_to_indices", None)
                else {}
            )
            del reindexer

        else:
            self.dataset = dataset
            self.base_dataset = dataset
            self.subset_base_indices = np.arange(len(dataset))
            self.subset_indices = None
            self.bulk_indices = self.dataset.bulk_indices
            self.sc_indices = self.dataset.sc_indices
            self.pb_indices = self.dataset.pb_indices
            self.sc_group_to_indices = self.base_dataset.sc_group_to_indices
            self.bulk_group_to_indices = getattr(self.base_dataset, "bulk_group_to_indices", None)
            self.pb_group_to_indices = getattr(self.base_dataset, "pb_group_to_indices", None)
            self.sc_pair_to_indices = getattr(self.base_dataset, "sc_pair_to_indices", {})
            self.sc_pb_to_indices = getattr(self.base_dataset, "sc_pb_to_indices", {})

        # Pre-compute sorted group keys (drop any group that became empty after Subset)
        if self.sc_group_to_indices is not None:
            self.sc_groups = sorted(
                g for g, idxs in self.sc_group_to_indices.items() if len(idxs) > 0
            )
        else:
            self.sc_groups = None

        # Source-pool alias: in --precomputed-pb mode the pseudobulk SOURCE is the set of
        # precomputed PB rows (pb_group_to_indices), not cells aggregated on the fly from
        # the SC pool. Every place that reasons about "what the source can supply per
        # tissue" (class-aware shared tissues, CDD label refresh) goes through this alias,
        # so the two modes share one code path. With precomputed_pb=False the alias is the
        # SC pool, preserving the original behaviour exactly.
        # Assigned here (not below) because the class-aware CDD block further down reads
        # self.verbose before that later assignment would run.
        self.verbose = verbose
        self.precomputed_pb = precomputed_pb
        self.source_group_to_indices = (
            self.pb_group_to_indices if precomputed_pb else self.sc_group_to_indices
        )
        if self.source_group_to_indices is not None:
            self.source_groups = sorted(
                g for g, idxs in self.source_group_to_indices.items() if len(idxs) > 0
            )
        else:
            self.source_groups = None

        # Class-aware CDD sampling: tissue groups present in BOTH the source (SC cells, or
        # precomputed PB rows in --precomputed-pb mode) and bulk (target) pools with enough
        # samples, excluding non-tissue codes (e.g. "unknown"). When enabled, batches draw
        # the same tissues in both domains so the CDD loss is non-trivial.
        self.class_aware_cdd = class_aware_cdd
        self.n_cdd_classes = n_cdd_classes
        self.cdd_bulk_class_frac = min(max(cdd_bulk_class_frac, 0.0), 1.0)
        self.cdd_min_class_count = cdd_min_class_count
        self.cdd_exclude_group_codes = set(cdd_exclude_group_codes or ())
        exclude_codes = self.cdd_exclude_group_codes
        self.shared_groups = None
        if class_aware_cdd:
            if self.source_group_to_indices is None or self.bulk_group_to_indices is None:
                raise ValueError(
                    "class_aware_cdd requires pb_group_column (tissue) so both the source "
                    "and bulk group pools exist"
                    + (" (precomputed_pb: check that precomputed PB rows exist)."
                       if precomputed_pb else ".")
                )
            min_c = max(1, cdd_min_class_count)
            self.shared_groups = sorted(
                g for g in self.source_groups
                if g not in exclude_codes
                and len(self.source_group_to_indices.get(g, [])) >= min_c
                and len(self.bulk_group_to_indices.get(g, [])) >= min_c
            )
            if len(self.shared_groups) == 0:
                raise ValueError(
                    "class_aware_cdd: no tissue is present in both bulk and source "
                    f"({'precomputed PB' if precomputed_pb else 'SC'}) pools with enough "
                    "samples. Check the tissue labels / exclude list."
                )
            if self.verbose:
                print(f"[CDD] class-aware sampling over {len(self.shared_groups)} shared tissues.")

        self.world_size = max(1, world_size)
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epoch_coverage = epoch_coverage
        self.bulk_ratio = bulk_ratio
        self.pb_ratio = pb_ratio
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.n_bulk = round(self.batch_size * self.bulk_ratio)
        self.n_pb = round(self.batch_size * self.pb_ratio)
        self.n_sc = self.batch_size - self.n_bulk - self.n_pb
        self.n_sc_per_pb = n_sc_per_pb
        # In --precomputed-pb mode each pseudobulk is a single precomputed row (no
        # n_sc_per_pb blow-up), so the raw batch is just n_sc + n_pb + n_bulk — unless
        # aggregation consistency is on, which additionally draws the n_sc_per_pb
        # constituent cells of every drawn PB row so the loss has something to compare
        # against. That makes the batch the same size as the on-the-fly case plus the
        # n_pb precomputed rows themselves.
        self.agg_consistency = agg_consistency
        self.precomputed_agg = precomputed_pb and agg_consistency
        if self.precomputed_pb:
            if self.verbose:
                print("[PSEUDOBULK] Using precomputed pseudobulks!")
            if self.precomputed_agg:
                self.raw_batch_size = (
                    self.n_bulk + self.n_sc + self.n_pb * (1 + self.n_sc_per_pb)
                )
            else:
                self.raw_batch_size = self.n_bulk + self.n_sc + self.n_pb
        else:
            self.raw_batch_size = self.n_bulk + self.n_sc + self.n_pb * self.n_sc_per_pb

        # Restrict the PB draw pool to pseudobulks whose constituent cells are actually
        # in this split. A PB without them cannot supply an aggregation target, and
        # falling back to unrelated cells (as sample_paired_batch does for pairs) would
        # train the model to match a pseudobulk against cells it never contained.
        self.pb_indices_with_sc = self.pb_indices
        self.pb_group_to_indices_agg = self.pb_group_to_indices
        if self.precomputed_agg:
            self._build_agg_pb_pools()

        if precomputed_pb and len(self.pb_indices) == 0:
            raise ValueError(
                "precomputed_pb=True but no precomputed pseudobulk rows were found. "
                "Check that --pb-label matches the modality label of the precomputed PB "
                "rows in the dataset."
            )

        # Define RNG — seeded for reproducibility and correct DDP sharding
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Confirm batch composition
        # if self.verbose:
        #     print("Batch composition at the sampler level:")
        #     print("batch_size:", self.batch_size)
        #     print("n_bulk:", self.n_bulk)
        #     print("n_pb:", self.n_pb)
        #     print("n_sc:", self.n_sc)
        #     print("raw_batch_size:", self.raw_batch_size)
        #     print("sum logical:", self.n_bulk + self.n_pb + self.n_sc)

        if self.n_bulk <= 0:
            raise ValueError(f"n_bulk_samples must be positive, got {self.n_bulk}.")
        if paired_sampling and self.n_bulk != self.n_pb:
            raise ValueError(
                f"paired_sampling requires bulk_ratio == pb_ratio "
                f"(got n_bulk={self.n_bulk}, n_pb={self.n_pb})."
            )

        # One epoch = `epoch_coverage` passes over the bulk pool. Each batch consumes
        # n_bulk bulk rows, so the batch count scales with batch_size: doubling
        # --batch-size halves the step count and keeps samples-per-epoch constant.
        # (Previously this was len(bulk_indices) — a fixed step count independent of
        # batch_size, which drew every bulk row ~n_bulk times per "epoch".)
        # Must stay below the n_bulk > 0 check above: it divides by n_bulk.
        if self.epoch_size is not None:
            self._n_batches = int(self.epoch_size)
        else:
            self._n_batches = max(
                1,
                int(np.ceil(self.epoch_coverage * len(self.bulk_indices) / self.n_bulk)),
            )

        # DistributedBatchSamplerWrapper shards these batches across ranks, so an epoch
        # shorter than world_size would leave a rank with nothing to do and hang the
        # collective.
        if self._n_batches < self.world_size:
            raise ValueError(
                f"Epoch is {self._n_batches} batches but world_size={self.world_size}; "
                f"at least one rank would get zero steps. Raise --epoch-coverage "
                f"(currently {self.epoch_coverage}) or lower --batch-size / --bulk-ratio "
                f"(n_bulk={self.n_bulk} over {len(self.bulk_indices)} bulk rows)."
            )

        # Printed unconditionally: this determines how much data a run actually sees,
        # and it used to be an invisible len(bulk_indices).
        print(self._describe_epoch())

        self.count = 0

        # Balanced sampling setup
        self.balance = balance
        self.sample_balanced = bool(balance)
        if balance:
            print("Setting up balanced sampler...")
            self.curiculum = curiculum
            self.element_weights = None
            self.replacement = replacement

            if self.base_dataset.labels is None:
                raise ValueError("Dataset does not have labels for balanced sampling.")

            # survival_masks come from the SubsetReindexer above (None when not a Subset)
            labels = {
                "sc":   self.base_dataset.labels["sc"]   if sc_mask   is None else self.base_dataset.labels["sc"][sc_mask],
                "bulk": self.base_dataset.labels["bulk"] if bulk_mask is None else self.base_dataset.labels["bulk"][bulk_mask],
            }

            if self.verbose:
                print("Computing label weights...")
            counts = {key: np.bincount(labels[key]) for key in labels}
            label_weights = {
                key: (weight_scaler * counts[key]) / (counts[key] + weight_scaler)
                for key in counts
            }
            self.label_weights = {
                key: torch.as_tensor(label_weights[key], dtype=torch.float32).share_memory_()
                for key in label_weights
            }

            if self.verbose:
                print("Building class indices...")
            self.klass_indices = {}
            self.klass_offsets = {}
            # Key corresponds to modality in this case
            for key in labels:
                idx_t, off_t = self._build_klass_tensors(labels[key])
                self.klass_indices[key] = idx_t
                self.klass_offsets[key] = off_t
            n_classes = {key: int(len(self.klass_offsets[key]) - 1) for key in self.klass_offsets}
            if self.verbose:
                print(f"Done: {len(self.klass_offsets)} modalities, max class label per modality: {n_classes}")

        # Paired sampling — match precomputed PB rows to bulk rows via the "paired" obs column.
        # paired == 0  → unpaired;  paired == k (k > 0)  → belongs to pair k.
        # paired_pb_indices[i] and paired_bulk_indices[i] are the subset-local indices for pair i.
        self.paired_sampling = paired_sampling
        self.paired_every_n = paired_every_n
        self.paired_pb_indices: Optional[np.ndarray] = None
        self.paired_bulk_indices: Optional[np.ndarray] = None
        self.paired_common_ids: Optional[np.ndarray] = None

        if paired_sampling:
            # paired_sampling was explicitly requested, so a dataset that cannot form
            # paired batches is a misconfiguration — fail loudly rather than silently
            # training without any paired batch.
            paired_col_name = self.base_dataset.paired_column
            if len(self.pb_indices) == 0:
                raise ValueError(
                    "paired_sampling=True but the dataset has no precomputed pseudobulk "
                    "rows (check --pb-label and that the memory-mapped store actually "
                    "contains rows with that modality code). Paired batches pair a "
                    "precomputed PB row to a bulk row, so none can be formed."
                )
            if paired_col_name is None:
                raise ValueError(
                    "paired_sampling=True but the 'paired' column is missing from obs "
                    "(expected column: "
                    f"{self.base_dataset.paired_column!r}). Paired batches are matched by "
                    "shared pair id, which this column supplies."
                )
            # Access via pre-extracted _obs_arrays; index by base (obs-row) positions
            paired_arr = self.base_dataset._obs_arrays[paired_col_name]
            pb_pair_ids   = paired_arr[self.subset_base_indices[self.pb_indices]]
            bulk_pair_ids = paired_arr[self.subset_base_indices[self.bulk_indices]]

            # Build pair-id → subset-local index lookup (1-to-1; last writer wins for duplicates)
            pb_by_id = {
                int(pid): int(lidx)
                for lidx, pid in zip(self.pb_indices, pb_pair_ids)
                if pid != 0
            }
            bulk_by_id = {
                int(pid): int(lidx)
                for lidx, pid in zip(self.bulk_indices, bulk_pair_ids)
                if pid != 0
            }

            common_ids = sorted(set(pb_by_id) & set(bulk_by_id))
            if not common_ids:
                raise ValueError(
                    "paired_sampling=True but no pair id is shared between the "
                    "precomputed pseudobulk rows and the bulk rows, so no PB–bulk pair "
                    "can be formed. Check that the 'paired' column is populated "
                    "consistently across both modalities."
                )
            self.paired_pb_indices = np.array(
                [pb_by_id[k] for k in common_ids], dtype=np.int64
            )
            self.paired_bulk_indices = np.array(
                [bulk_by_id[k] for k in common_ids], dtype=np.int64
            )
            self.paired_common_ids = np.array(common_ids, dtype=np.int64)
            print(f"Paired sampling: {len(common_ids)} PB–bulk pairs found.")

    def _build_agg_pb_pools(self) -> None:
        """Restrict the precomputed-PB pools to pseudobulks that have SC cells here.

        Also builds ``pb_local_to_pb_id``, the pseudobulk code of every entry of
        ``self.pb_indices``, so a drawn PB row can be resolved back to its cells.
        Indices are subset-local after a random_split, while ``_obs_arrays`` is keyed by
        base obs rows, so the lookup goes through ``subset_base_indices`` — the same
        translation the paired-sampling block does.
        """
        pb_id_column = getattr(self.base_dataset, "pb_id_column", None)
        if pb_id_column is None:
            raise ValueError(
                "precomputed_pb=True and agg_consistency=True, but the dataset has no "
                "pseudobulk-id column. Aggregation consistency over precomputed "
                "pseudobulks needs to know which single cells each one was aggregated "
                "from. Re-run data_preprocess/bulk_sc_data_preprocessing.py so that "
                "'pseudobulk_id' is present in obs.parquet (it is written by "
                "data_preprocess/reconstruct_pseudobulk_cell_map.py --rebuild), or drop "
                "--agg-consistency."
            )

        pb_id_arr = self.base_dataset._obs_arrays[pb_id_column]
        # subset_base_indices is set on both construction paths — the Subset's own
        # indices, or arange(len(dataset)) when there is no Subset — so this translation
        # is the identity in the latter case and needs no branch. (subset_indices is
        # vestigial: it only exists on the non-Subset path.)
        base_pb = self.subset_base_indices[self.pb_indices]
        self.pb_local_to_pb_id = pb_id_arr[base_pb].astype(np.int64)
        # Built once: _draw_agg_sc runs per batch and must not rescan pb_indices.
        self.pb_row_to_pb_id = {
            int(row): int(code)
            for row, code in zip(self.pb_indices, self.pb_local_to_pb_id)
        }

        has_sc = np.fromiter(
            (int(code) in self.sc_pb_to_indices for code in self.pb_local_to_pb_id),
            dtype=bool,
            count=len(self.pb_local_to_pb_id),
        )
        self.pb_indices_with_sc = self.pb_indices[has_sc]

        n_dropped = len(self.pb_indices) - len(self.pb_indices_with_sc)
        if n_dropped:
            print(
                f"[PSEUDOBULK] {n_dropped} of {len(self.pb_indices)} precomputed "
                f"pseudobulk rows have no constituent SC cells in this split and are "
                f"excluded from the aggregation-consistency pool."
            )
        if len(self.pb_indices_with_sc) == 0:
            raise ValueError(
                f"precomputed_pb=True and agg_consistency=True, but not one of the "
                f"{len(self.pb_indices)} precomputed pseudobulk rows has constituent SC "
                f"cells in this split. Check that '{pb_id_column}' is populated "
                f"consistently on both the pseudobulk and the single-cell rows — the "
                f"values must come from the same preprocessing pass, since obs.parquet "
                f"stores category codes rather than the ids themselves."
            )

        # Same restriction applied to the tissue-pure pools, so group-aware draws also
        # only ever pick a usable pseudobulk.
        if self.pb_group_to_indices is not None:
            usable = set(int(i) for i in self.pb_indices_with_sc)
            self.pb_group_to_indices_agg = {
                g: np.array([i for i in idxs if int(i) in usable], dtype=np.int64)
                for g, idxs in self.pb_group_to_indices.items()
            }
            self.pb_group_to_indices_agg = {
                g: idxs for g, idxs in self.pb_group_to_indices_agg.items() if len(idxs)
            }
            if not self.pb_group_to_indices_agg:
                raise ValueError(
                    "No tissue group retains a precomputed pseudobulk with constituent "
                    "SC cells; cannot form aggregation-consistency batches."
                )

    def _draw_agg_sc(self, pb_row_indices) -> list[int]:
        """Draw ``n_sc_per_pb`` constituent cells for each of *pb_row_indices*.

        Returns them grouped by pseudobulk and in the same order as the input, so the
        collator can slice the block into contiguous per-pseudobulk chunks exactly as it
        does for the on-the-fly path.
        """
        out: list[int] = []
        for row in pb_row_indices:
            pool = self.sc_pb_to_indices[self.pb_row_to_pb_id[int(row)]]
            chosen = self.rng.choice(
                pool,
                size=self.n_sc_per_pb,
                replace=len(pool) < self.n_sc_per_pb,
            )
            out.extend(int(i) for i in chosen)
        return out

    def _build_klass_tensors(
        self, labels: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a flat sorted-index tensor and a dense offset array for O(1) class lookup.

        Uses a single O(N log N) numpy pass with np.argsort.

        Returns
        -------
        klass_indices : int64 tensor of shape (N,) — sample indices sorted by class.
        klass_offsets : int64 tensor of shape (max_label+2,) — the slice
                        klass_indices[offsets[c] : offsets[c+1]] gives all sample
                        indices belonging to class c.
        """
        order = np.argsort(labels, kind="stable")
        unique_labels, counts = np.unique(labels[order], return_counts=True)
        max_label = int(unique_labels[-1]) if len(unique_labels) else 0
        offsets = np.zeros(max_label + 2, dtype=np.int64)
        offsets[unique_labels.astype(np.int64) + 1] = counts
        np.cumsum(offsets, out=offsets)
        return (
            torch.from_numpy(order.astype(np.int64)).share_memory_(),
            torch.from_numpy(offsets).share_memory_(),
        )

    def _describe_epoch(self) -> str:
        """Human-readable summary of what one epoch actually consumes.

        Worth printing: the pseudobulk constituents (n_pb * n_sc_per_pb) dominate SC
        throughput and are drawn every batch whether or not --agg-consistency forwards
        them to the model, so SC coverage is easy to misjudge from batch_size alone.
        """
        sc_per_batch = self.n_sc + (
            0 if self.precomputed_pb else self.n_pb * self.n_sc_per_pb
        )
        n_bulk_rows = max(1, len(self.bulk_indices))
        n_sc_rows = max(1, len(self.sc_indices))
        bulk_draws = self._n_batches * self.n_bulk
        sc_draws = self._n_batches * sc_per_batch
        source = (
            f"epoch_size={self.epoch_size} (absolute)"
            if self.epoch_size is not None
            else f"epoch_coverage={self.epoch_coverage}"
        )
        lines = [
            f"[EPOCH] {self._n_batches} batches/epoch from {source}",
            f"[EPOCH] batch_size={self.batch_size} -> n_bulk={self.n_bulk}, "
            f"n_pb={self.n_pb}, n_sc={self.n_sc}, rows/batch={self.raw_batch_size}",
            f"[EPOCH] bulk: {bulk_draws} draws over {len(self.bulk_indices)} rows "
            f"({bulk_draws / n_bulk_rows:.2f} passes/epoch)",
            f"[EPOCH] sc:   {sc_draws} draws over {len(self.sc_indices)} rows "
            f"({sc_draws / n_sc_rows:.3f} passes/epoch, "
            f"{n_sc_rows / max(1, sc_draws):.1f} epochs to cover)",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return self._n_batches

    def set_epoch(self, epoch: int) -> None:
        """Reseed the RNG so each epoch produces a different shuffle while
        remaining identical across all DDP ranks (given the same base seed)."""
        self.rng = np.random.default_rng(
            self._seed + epoch if self._seed is not None else None
        )

    def __iter__(self):
        """
        Yield a list of indices for each batch.
        The number of batches per epoch is set by `epoch_coverage` passes over the bulk
        pool (or by `epoch_size` when given) — see __init__.
        In paired batches there is a one-to-one correspondence between bulk and pseudobulk samples
        """
        for batch_i in range(self._n_batches):
            self.count += 1
            # Group world_size consecutive global positions together so that
            # every rank sees a paired batch at the same local step frequency.
            is_paired = ((batch_i // self.world_size) % self.paired_every_n == 0) \
                        and self.paired_sampling \
                        and self.paired_pb_indices is not None

            # Batch constructor by priority: paired sampling (when scheduled) wins;
            # otherwise class-aware sampling (if enabled) matches tissues across the
            # bulk/pseudobulk halves for CDD; otherwise the standard batch. The CDD
            # loss itself runs only on the non-paired batches (see module.forward).
            if is_paired:
                if self.verbose:
                    print("Sampling paired batch!")
                yield self.sample_paired_batch()
            elif self.class_aware_cdd:
                yield self.sample_class_aware_batch()
            else:
                yield self.sample_standard_batch()
                

    def sample_paired_batch(self):
        """
        Sample a paired batch: each precomputed PB row is matched to its bulk sample.
        Pairing is preserved by sampling the same positions from both index arrays.
        """
        assert self.paired_pb_indices is not None and self.paired_bulk_indices is not None, \
            "Paired batches require precomputed paired indices"

        indices= []
        # Sample pair positions (not indices directly)
        n_pairs = self.n_bulk
        pair_positions = self.rng.choice(
            len(self.paired_pb_indices),
            size=n_pairs,
            replace=len(self.paired_pb_indices) < n_pairs,
        )

        # Both paired arrays are indexed by same positions to preserve pairing
        pb_idx = self.paired_pb_indices[pair_positions].tolist()
        bulk_idx = self.paired_bulk_indices[pair_positions].tolist()

        # Sample exactly n_sc_per_pb SC cells per selected pair (stratified).
        # Falls back to the global SC pool for pairs with no matched SC cells.
        sc_idx = []
        for pos in pair_positions:
            pid = int(self.paired_common_ids[pos])
            pair_pool = self.sc_pair_to_indices.get(pid, np.empty(0, dtype=np.int64))
            if len(pair_pool) > 0:
                chosen = self.rng.choice(
                    pair_pool,
                    size=self.n_sc_per_pb,
                    replace=len(pair_pool) < self.n_sc_per_pb,
                )
            else:
                chosen = self.sample(
                    self.sc_indices,
                    size=self.n_sc_per_pb,
                    modality="sc",
                    balanced=self.sample_balanced,
                )
            sc_idx.extend(chosen.tolist())

        indices.extend(sc_idx)
        indices.extend(pb_idx)
        indices.extend(bulk_idx)
        return indices


    def sample_standard_batch(self):
        # print(f"Sampling a new batch ({self.count}) of size {self.batch_size}: ", end="")
        indices: list[int] = []
        # Order matters for the collator: [sc_0, ..., sc_{n-1}, pseudobulk_0, ...].
        sc_idx = self.sample(
            self.sc_indices,
            size=self.n_sc,
            modality="sc",
            balanced=self.sample_balanced,
        )

        # --precomputed-pb: draw n_pb precomputed PB rows directly (one row per PB), tissue
        # -pure when a PB group column is set, else from the global precomputed-PB pool.
        # The collator passes these rows through unchanged (no on-the-fly aggregation).
        if self.precomputed_pb:
            # With aggregation consistency the pools are pre-filtered to pseudobulks
            # whose constituent cells are present, so every drawn row is usable.
            pb_pool = self.pb_indices_with_sc if self.precomputed_agg else self.pb_indices
            group_pool = (
                self.pb_group_to_indices_agg
                if self.precomputed_agg
                else self.pb_group_to_indices
            )
            if group_pool is not None:
                groups = sorted(group_pool) if self.precomputed_agg else self.source_groups
                pb_groups = self.rng.choice(groups, size=self.n_pb, replace=True)
                pb_sel = [
                    int(self.sample(group_pool[g], size=1, balanced=False)[0])
                    for g in pb_groups
                ]
                pb_idx = np.array(pb_sel)
            else:
                pb_idx = self.sample(pb_pool, size=self.n_pb, balanced=False)
            bulk_idx = self.sample(
                self.bulk_indices,
                size=self.n_bulk,
                modality="bulk",
                balanced=self.sample_balanced,
            )
            indices.extend(sc_idx)
            indices.extend(pb_idx)
            # Block order is a hard contract with the collator:
            # [sc, pb, agg_sc, bulk], with n_sc_per_pb consecutive cells per pseudobulk.
            if self.precomputed_agg:
                indices.extend(self._draw_agg_sc(pb_idx))
            indices.extend(bulk_idx)
            return indices

        # If a PB group column is set, each pseudobulk is built from cells
        # of a single randomly chosen tissue group
        if self.sc_group_to_indices is not None:
            pb_groups = self.rng.choice(self.sc_groups, size=self.n_pb, replace=True)
            pb_sc_indices = []
            for g in pb_groups:
                sc_pool = self.sc_group_to_indices[g]
                # Tissue-group selection already defines the sampling strategy
                # for PB; balanced class sampling would ignore the pool, so
                # we always sample uniformly within the group here.
                pb_sc_indices.extend(
                    list(self.sample(
                        sc_pool,
                        size=self.n_sc_per_pb,
                        modality="pb",
                        balanced=False,
                        ))
                )
            pb_idx = np.array(pb_sc_indices)
        else:
            pb_idx = self.sample(
                self.sc_indices,
                size=self.n_pb * self.n_sc_per_pb,
                modality="pb",
                balanced=self.sample_balanced,
            )

        bulk_idx = self.sample(
            self.bulk_indices,
            size=self.n_bulk,
            modality="bulk",
            balanced=self.sample_balanced,
        )

        indices.extend(sc_idx)
        indices.extend(pb_idx)
        indices.extend(bulk_idx)
        return indices

    def refresh_cdd_labels(self, pseudo_label, row_to_bulk_local):
        """Re-key the bulk tissue pools on the clustering's pseudo-labels.

        Called after each clustering event. Without it, rows the clustering recovered
        (formerly "unknown", or a tissue with no single-cell counterpart) stay
        undrawable in the matched half below, because these pools would still be built
        from the raw labels that marked them unusable in the first place.

        ``pseudo_label`` is indexed by bank-local id and ``row_to_bulk_local`` by base
        dataset row, so sampler-space indices are mapped through
        ``subset_base_indices`` first. Rows still unassigned (-1) drop out of every
        pool; they remain reachable through the free part of the bulk half.
        """
        if not self.class_aware_cdd or self.bulk_group_to_indices is None:
            return
        base = self.subset_base_indices[self.bulk_indices]
        local = np.asarray(row_to_bulk_local)[base]
        labels = np.where(local >= 0, np.asarray(pseudo_label)[np.clip(local, 0, None)], -1)

        bulk_arr = np.asarray(self.bulk_indices)
        new_pools = {
            int(g): bulk_arr[labels == g] for g in np.unique(labels) if g >= 0
        }
        min_c = max(1, self.cdd_min_class_count)
        new_shared = sorted(
            g for g in self.source_groups
            if g not in self.cdd_exclude_group_codes
            and len(self.source_group_to_indices.get(g, [])) >= min_c
            and len(new_pools.get(g, [])) >= min_c
        )
        # A degenerate clustering pass (e.g. everything filtered as ambiguous) must not
        # leave the sampler with no tissue to draw; keep the previous pools instead.
        if not new_shared:
            if self.verbose:
                print("[CDD] sampler pools NOT refreshed: no shared tissue survived.")
            return
        self.bulk_group_to_indices = new_pools
        self.shared_groups = new_shared
        if self.verbose:
            print(
                f"[CDD] sampler pools refreshed: {len(self.shared_groups)} shared "
                f"tissues, {int((labels >= 0).sum())}/{len(labels)} bulk rows labelled."
            )

    def sample_class_aware_batch(self):
        """Class-aware batch for CDD: the same K tissues are drawn in both the
        pseudobulk (source) and bulk (target) halves, so the class-conditional
        MMD has matched classes in both domains. Batch structure/order matches
        sample_standard_batch: [sc..., sc_for_pb..., (agg_sc...,) bulk...].

        Only a fraction (cdd_bulk_class_frac) of the bulk slots is reserved for the
        chosen tissues; the rest are drawn freely from ALL bulk. Reserving every slot
        would mean bulk rows outside the shared tissues — "unknown" rows above all —
        never appear in any batch, so they would receive no gradient from any loss and
        never be embedded for the clustering that exists to recover them.
        """
        indices: list[int] = []

        # Free SC filler (unchanged from the standard batch).
        sc_idx = self.sample(
            self.sc_indices, size=self.n_sc, modality="sc", balanced=self.sample_balanced,
        )

        # Pick K shared tissues for this batch and spread PBs/bulk across them.
        K = min(self.n_cdd_classes, len(self.shared_groups))
        chosen = self.rng.choice(self.shared_groups, size=K, replace=False)

        # Pseudobulks: n_pb tissue-pure PBs distributed round-robin over the K tissues.
        # --precomputed-pb draws one precomputed PB row per slot from the tissue's PB pool;
        # otherwise it draws n_sc_per_pb SC cells per slot for on-the-fly aggregation.
        pb_groups = [chosen[i % K] for i in range(self.n_pb)]
        pb_sc_indices: list[int] = []
        pb_group_pool = (
            self.pb_group_to_indices_agg
            if self.precomputed_agg
            else self.pb_group_to_indices
        )
        for g in pb_groups:
            if self.precomputed_pb:
                pb_sc_indices.extend(list(self.sample(
                    pb_group_pool[g], size=1, balanced=False,
                )))
            else:
                pb_sc_indices.extend(list(self.sample(
                    self.sc_group_to_indices[g], size=self.n_sc_per_pb, modality="pb", balanced=False,
                )))
        pb_idx = np.array(pb_sc_indices)

        # Bulk: a matched part over the SAME K tissues (so CDD sees paired classes),
        # plus a free part over all bulk (so every bulk row still trains).
        n_matched = int(round(self.n_bulk * self.cdd_bulk_class_frac))
        n_free = self.n_bulk - n_matched
        bulk_sel: list[int] = []
        if n_matched > 0:
            bulk_groups = [chosen[i % K] for i in range(n_matched)]
            bulk_counts: dict = {}
            for g in bulk_groups:
                bulk_counts[g] = bulk_counts.get(g, 0) + 1
            for g, cnt in bulk_counts.items():
                bulk_sel.extend(list(self.sample(
                    self.bulk_group_to_indices[g], size=cnt, modality="bulk", balanced=False,
                )))
        if n_free > 0:
            bulk_sel.extend(list(self.sample(
                self.bulk_indices, size=n_free, modality="bulk", balanced=self.sample_balanced,
            )))
        bulk_idx = np.array(bulk_sel)

        indices.extend(sc_idx)
        indices.extend(pb_idx)
        # Same [sc, pb, agg_sc, bulk] contract as sample_standard_batch.
        if self.precomputed_agg:
            indices.extend(self._draw_agg_sc(pb_idx))
        indices.extend(bulk_idx)
        return indices

    def sample(
        self,
        indices: Union[List[int], List[int]],
        size: int,
        modality: Optional[str] = None,
        balanced: bool = False,
    ):
        """Sample a batch of single-cell or real bulk indices with optional balancing."""

        if size == 0:
            return []

        if not balanced:
            return self.rng.choice(
                indices,
                size=size,
                replace=len(indices) < size,
            )

        # Balanced sampling logic
        assert modality in [
            "sc",
            "bulk",
            "pb",
        ], "Modality must be one of 'sc', 'bulk', or 'pb' for balanced sampling"

        # print(f"{modality} samples, ", end="") if modality != "bulk" else print(f"{modality} samples.")
        sample_modality = "sc" if modality in ["sc", "pb"] else "bulk"

        sample_labels = torch.multinomial(
            (
                self.label_weights[sample_modality]
                ** min(1, ((self.count + 5) / self.curiculum))
                if self.curiculum
                else self.label_weights[sample_modality]
            ),
            num_samples=size,
            replacement=True,
        )
        # Get counts of each class in sample_labels
        unique_samples, sample_counts = torch.unique(sample_labels, return_counts=True)

        # Initialize result tensor
        result_indices_list = []  # Changed name to avoid conflict if you had result_indices elsewhere

        # Process only the classes that were actually sampled
        for i, (label, count) in enumerate(
            zip(unique_samples.tolist(), sample_counts.tolist())
        ):
            klass_index = self.klass_indices[sample_modality][
                self.klass_offsets[sample_modality][label] : self.klass_offsets[
                    sample_modality
                ][label + 1]
            ]

            if klass_index.numel() == 0:
                continue

            # Sample elements from this class
            if self.element_weights is not None:
                # This is a critical point for memory
                current_element_weights_slice = self.element_weights[klass_index]

                if current_element_weights_slice.shape[0] >= (2**24) - 1:
                    ind = torch.randperm(len(klass_index))[: (2**24) - 10]
                    klass_index = klass_index[ind]
                    current_element_weights_slice = current_element_weights_slice[ind]

                if self.replacement:
                    right_inds = torch.multinomial(
                        current_element_weights_slice,
                        num_samples=count,
                        replacement=True,
                    )
                else:
                    num_to_sample = min(count, len(klass_index))
                    right_inds = torch.multinomial(
                        current_element_weights_slice,
                        num_samples=num_to_sample,
                        replacement=False,
                    )
            elif self.replacement:
                right_inds = torch.randint(len(klass_index), size=(count,))
            else:
                num_to_sample = min(count, len(klass_index))
                right_inds = torch.randperm(len(klass_index))[:num_to_sample]

            # Get actual indices
            sampled_indices = klass_index[right_inds]
            result_indices_list.append(sampled_indices)

        # Combine all indices
        if result_indices_list:  # Check if the list is not empty
            final_result_indices = torch.cat(
                result_indices_list
            )  # Use the list with the appended new name

            # Shuffle the combined indices
            shuffled_indices = final_result_indices[
                torch.randperm(len(final_result_indices))
            ]

            # Map back to original indices (klass_indices conains positions within sc_indices or bulk_indices)
            if sample_modality == "sc":
                true_indices = self.sc_indices[shuffled_indices]
            else:
                true_indices = self.bulk_indices[shuffled_indices]

            return true_indices.tolist()
