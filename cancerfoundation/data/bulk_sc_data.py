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
        pb_group_column: Optional[str] = None,
        pad_value: float = -1.0,
        obs_columns: Optional[list[str]] = None,
        balance: Optional[bool] = False,
        balance_labels: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()
        self.data_dir = DatasetDir(data_dir)
        self.vocab = self._load_vocab()
        self.pad_value = pad_value
        self.memmap = SingleCellMemMapDataset(str(self.data_dir.memmap_path))
        self.obs = pd.read_parquet(self.data_dir.obs_path)
        self.mapping = self._load_mapping()
        self.obs_columns = obs_columns

        self.modality_column = modality_column
        self.pb_group_column = pb_group_column

        assert self.memmap.number_of_rows() == self.obs.shape[0]
        assert modality_column in self.obs.columns

        # Pre-extract obs columns as plain numpy arrays for O(1) random access in __getitem__
        # (pandas .iloc on a 100M-row DataFrame is expensive — numpy array indexing is not)
        self._obs_arrays = {col: self.obs[col].to_numpy() for col in (obs_columns or [])}

        # Pre-compute index arrays per modality
        modality_vals = self.obs[modality_column].values
        bulk_code = self.mapping[modality_column][bulk_label]
        sc_code = self.mapping[modality_column][sc_label]

        self.bulk_indices = np.where(modality_vals == bulk_code)[0]
        self.sc_indices = np.where(modality_vals == sc_code)[0]

        assert len(self.bulk_indices) > 0, "No bulk samples found"
        assert len(self.sc_indices) > 0, "No SC samples found"

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
        else:
            self.sc_group_to_indices = None

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
        exp, genes = self.memmap.get_row_padded(
            index, return_features=True, feature_vars=[self.GENE_ID]
        )
        genes = np.insert(genes[0], 0, self.vocab[self.CLS_TOKEN])
        exp = np.insert(exp, 0, self.pad_value)

        data = {
            "expressions": torch.tensor(exp, dtype=torch.float32),
            "genes": torch.from_numpy(genes),
        }

        # Additional conditions input to model (e.g. tissue type)
        for col in self.obs_columns:
            data[col] = self._obs_arrays[col][index]

        return data

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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

        Returns only "sc" and "bulk" groups — the "all" group is never used by
        the sampler and would wastefully process 100M+ entries at scale.
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

    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        bulk_ratio: float = 0.3,
        pb_ratio: float = 0.3,
        n_sc_per_pb: int = 10,
        drop_last: bool = True,
        shuffle: bool = True,
        balance: Optional[bool] = False,
        weight_scaler: float = 1.0,
        num_workers: int = 1,
        chunk_size: int = 1000,
        curiculum: int = 0,
        replacement: bool = True,
        epoch_size: Optional[int] = None,
    ):
        # Account for the Subset resulting from random_split
        # Since the Subset uses a local set of indices, different from the original dataset
        if isinstance(dataset, Subset):
            self.dataset = dataset
            self.subset_base_indices = np.asarray(dataset.indices)
            self.base_dataset = dataset.dataset

            # Vectorized base->subset mapping using a lookup array
            max_idx = int(self.subset_base_indices.max()) + 1
            base_to_subset = np.full(max_idx, -1, dtype=np.int64)
            base_to_subset[self.subset_base_indices] = np.arange(len(self.subset_base_indices), dtype=np.int64)

            # Vectorized bulk remapping
            bulk_mapped = base_to_subset[self.base_dataset.bulk_indices]
            self.bulk_indices = bulk_mapped[bulk_mapped >= 0]

            # Vectorized SC remapping
            sc_mapped = base_to_subset[self.base_dataset.sc_indices]
            self.sc_indices = sc_mapped[sc_mapped >= 0]

            # Vectorized SC group remapping
            if self.base_dataset.sc_group_to_indices is not None:
                self.sc_group_to_indices = {}
                for g, idxs in self.base_dataset.sc_group_to_indices.items():
                    idxs = np.asarray(idxs)
                    # Only remap indices that fall within the lookup array bounds
                    valid_mask = idxs < max_idx
                    mapped = np.full(len(idxs), -1, dtype=np.int64)
                    mapped[valid_mask] = base_to_subset[idxs[valid_mask]]
                    self.sc_group_to_indices[g] = mapped[mapped >= 0]
            else:
                self.sc_group_to_indices = None
            del base_to_subset  # free the large lookup table (800MB at 100M rows)

        else:
            self.dataset = dataset
            self.base_dataset = dataset
            self.subset_base_indices = np.arange(len(dataset))
            self.subset_indices = None
            self.bulk_indices = self.dataset.bulk_indices
            self.sc_indices = self.dataset.sc_indices
            self.sc_group_to_indices = self.base_dataset.sc_group_to_indices

        # Pre-compute sorted group keys (drop any group that became empty after Subset)
        if self.sc_group_to_indices is not None:
            self.sc_groups = sorted(
                g for g, idxs in self.sc_group_to_indices.items() if len(idxs) > 0
            )
        else:
            self.sc_groups = None

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.bulk_ratio = bulk_ratio
        self.pb_ratio = pb_ratio
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.n_bulk = round(self.batch_size * self.bulk_ratio)
        self.n_pb = round(self.batch_size * self.pb_ratio)
        self.n_sc = self.batch_size - self.n_bulk - self.n_pb
        self.n_sc_per_pb = n_sc_per_pb
        self.raw_batch_size = self.n_bulk + self.n_sc + self.n_pb * self.n_sc_per_pb

        # Define RNG
        self.rng = np.random.default_rng()

        # Confirm batch composition
        """
        print("Batch composition at the sampler level:")
        print("batch_size:", self.batch_size)
        print("n_bulk:", self.n_bulk)
        print("n_pb:", self.n_pb)
        print("n_sc:", self.n_sc)
        print("raw_batch_size:", self.raw_batch_size)
        print("sum logical:", self.n_bulk + self.n_pb + self.n_sc)
        """

        if self.n_bulk <= 0:
            raise ValueError(f"n_bulk_samples must be positive, got {self.n_bulk}.")

        self._n_batches = (
            len(self.bulk_indices) // 5
            if self.epoch_size is None
            else self.epoch_size
        )
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

            # Map labels to the current Subset if necessary; skip the "all" modality
            # which is never used by sample() and wastes 800MB+ at 100M-row scale.
            if isinstance(dataset, Subset):
                subset_base_indices = np.asarray(self.subset_base_indices)
                sc_mask   = np.isin(self.base_dataset.sc_indices,   subset_base_indices)
                bulk_mask = np.isin(self.base_dataset.bulk_indices, subset_base_indices)
                labels = {
                    "sc":   self.base_dataset.labels["sc"][sc_mask],
                    "bulk": self.base_dataset.labels["bulk"][bulk_mask],
                }
            else:
                labels = {
                    "sc":   self.base_dataset.labels["sc"],
                    "bulk": self.base_dataset.labels["bulk"],
                }

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

            print("Building class indices...")
            self.klass_indices = {}
            self.klass_offsets = {}
            for key in labels:
                idx_t, off_t = self._build_klass_tensors(labels[key])
                self.klass_indices[key] = idx_t
                self.klass_offsets[key] = off_t
            n_classes = {key: int(len(self.klass_offsets[key]) - 1) for key in self.klass_offsets}
            print(f"Done: {len(self.klass_offsets)} modalities, max class label per modality: {n_classes}")

    def _build_klass_tensors(
        self, labels: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a flat sorted-index tensor and a dense offset array for O(1) class lookup.

        Replaces the old chunk+executor approach with a single O(N log N) numpy pass.

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

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = self.rng
        batch_order = np.arange(self._n_batches)

        for _ in batch_order:
            self.count += 1
            # print(f"Sampling a new batch ({self.count}) of size {self.batch_size}: ", end="")
            indices: list[int] = []
            # Order matters for the collator: [sc_0, ..., sc_{n-1}, pseudobulk_0, ...].
            sc_idx = self.sample(
                self.sc_indices,
                size=self.n_sc,
                modality="sc",
                balanced=self.sample_balanced,
            )

            # If a PB group column is set, each pseudobulk is built from cells
            # of a single randomly chosen tissue group
            if self.sc_group_to_indices is not None:
                pb_groups = rng.choice(self.sc_groups, size=self.n_pb, replace=True)
                pb_sc_indices = []
                for g in pb_groups:
                    sc_pool = self.sc_group_to_indices[g]
                    # Tissue-group selection already defines the sampling strategy
                    # for PB; balanced class sampling would ignore the pool, so
                    # we always sample uniformly within the group here.
                    pb_sc_indices.extend(
                        self.sample(
                            sc_pool,
                            size=self.n_sc_per_pb,
                            modality="pb",
                            balanced=False,
                        )
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
            yield indices

    def sample(
        self,
        indices: Union[List[int], List[int]],
        size: int,
        modality: Optional[str] = None,
        balanced: bool = False,
    ):
        """Sample a batch of single-cell or real bulkindices with optional balancing."""

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
        if modality == "sc":
            sample_modality = "sc"
        if modality == "bulk":
            sample_modality = "bulk"
        if modality == "pb":
            sample_modality = "sc"

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

            # Map back to original indices
            if sample_modality == "sc":
                true_indices = self.sc_indices[shuffled_indices]
            elif sample_modality == "bulk":
                true_indices = self.bulk_indices[shuffled_indices]

            return true_indices.tolist()
