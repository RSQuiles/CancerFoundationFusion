from typing import Optional

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler, Subset
from typing import Union, List, Dict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
    ``bulk_label`` / ``sc_label``).  An optional ``group_column``
    (e.g. ``"tissue"``) defines matching pools.

    Parameters
    ----------
    data_dir : str | Path
        Root ``DatasetDir`` (``vocab.json``, ``obs.parquet``,
        ``mapping.json``, ``mem.map/``).
    modality_column : str
        Column distinguishing bulk from SC rows.
    bulk_label, sc_label : str
        Values in ``modality_column``.
    group_column : str | None
        Column for matching bulk ↔ SC (e.g. tissue).
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
        group_column: Optional[str] = None,
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
        self.group_column = group_column
        self.group_to_bulk: Optional[dict[int, np.ndarray]] = None
        self.group_to_sc: Optional[dict[int, np.ndarray]] = None

        assert self.memmap.number_of_rows() == self.obs.shape[0]
        assert modality_column in self.obs.columns
        if group_column is not None:
            assert (
                group_column in self.obs_columns
            ), f"The grouping feature {group_column} is not part of the selected {self.obs_columns}"

        # Pre-compute index arrays per modality
        modality_vals = self.obs[modality_column].values
        bulk_code = self.mapping[modality_column][bulk_label]
        sc_code = self.mapping[modality_column][sc_label]

        self.bulk_indices = np.where(modality_vals == bulk_code)[0]
        self.sc_indices = np.where(modality_vals == sc_code)[0]

        assert len(self.bulk_indices) > 0, "No bulk samples found"
        assert len(self.sc_indices) > 0, "No SC samples found"

        # Per-group index pools
        if group_column is not None:
            assert group_column in self.obs.columns
            group_vals = np.asarray(self.obs[group_column].values)

            self.group_to_bulk = {}
            self.group_to_sc = {}
            for g in np.unique(group_vals):
                g_mask = group_vals == g
                b = np.where(g_mask & (modality_vals == bulk_code))[0]
                s = np.where(g_mask & (modality_vals == sc_code))[0]
                if len(b) > 0 and len(s) > 0:
                    self.group_to_bulk[g] = b
                    self.group_to_sc[g] = s

            self.groups = sorted(self.group_to_bulk.keys())
            assert len(self.groups) > 0, "No groups with both bulk and SC"
        else:
            self.groups = None

        # Label categories for balanced sampling
        self.balance = balance
        self.labels = None
        if balance is not None:
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
        row = self.obs.iloc[index]
        for col in self.obs_columns:
            data[col] = row[col]

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
        Get categorical labels for a given key, with categories mapped to their string values.

        Given the virtual separation between bulk and SC samples in relation to
        sampling, this method deals with these two different set indices independently

        Args:
            label_key (str): Column name in the obs DataFrame to retrieve.
        Returns:
            Dict[str, pd.Categorical]: Categorical labels with string categories for each group.
        """
        if label_key not in self.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in obs columns.")
        if label_key not in self.mapping:
            raise ValueError(f"Label key '{label_key}' not found in mapping.")
        code_to_str = {v: k for k, v in self.mapping[label_key].items()}
        all_codes = self.obs[label_key].values
        sc_codes = self.obs[label_key].values[self.sc_indices]
        bulk_codes = self.obs[label_key].values[self.bulk_indices]
        all_labels = pd.Categorical([code_to_str[code] for code in all_codes])
        sc_labels = pd.Categorical([code_to_str[code] for code in sc_codes])
        bulk_labels = pd.Categorical([code_to_str[code] for code in bulk_codes])
        str_labels = {"all": all_labels, "sc": sc_labels, "bulk": bulk_labels}
        return str_labels


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
    ):
        # Account for the Subset resulting from random_split
        # Since the Subset uses a local set of indices, different from the original dataset
        if isinstance(dataset, Subset):
            self.dataset = dataset
            self.subset_base_indices = dataset.indices
            base_to_subset = {
                base_idx: sub_idx
                for sub_idx, base_idx in enumerate(self.subset_base_indices)
            }

            self.base_dataset = dataset.dataset

            # Map previously computed indices to indices in the current dataset (Subset)
            self.bulk_indices = np.array(
                [
                    base_to_subset[i]
                    for i in self.base_dataset.bulk_indices
                    if i in base_to_subset
                ]
            )
            self.sc_indices = np.array(
                [
                    base_to_subset[i]
                    for i in self.base_dataset.sc_indices
                    if i in base_to_subset
                ]
            )
        else:
            self.dataset = dataset
            self.base_dataset = dataset
            self.subset_indices = None
            self.bulk_indices = self.dataset.bulk_indices
            self.sc_indices = self.dataset.sc_indices

        self.batch_size = batch_size
        self.bulk_ratio = bulk_ratio
        self.pb_ratio = pb_ratio
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.n_bulk = round(self.batch_size * self.bulk_ratio)
        self.n_pb = round(self.batch_size * self.pb_ratio)
        self.n_sc = self.batch_size - self.n_bulk - self.n_pb
        self.n_sc_per_pb = n_sc_per_pb
        self.raw_batch_size = self.n_bulk + self.n_sc + self.n_pb * self.n_sc_per_pb

        # Confirm batch composition
        print("Batch composition at the sampler level:")
        print("batch_size:", self.batch_size)
        print("n_bulk:", self.n_bulk)
        print("n_pb:", self.n_pb)
        print("n_sc:", self.n_sc)
        print("raw_batch_size:", self.raw_batch_size)
        print("sum logical:", self.n_bulk + self.n_pb + self.n_sc)

        if self.n_bulk <= 0:
            raise ValueError(f"n_bulk_samples must be positive, got {self.n_bulk}.")

        self._n_batches = len(self.bulk_indices) // 5
        self.count = 0

        # Balanced sampling setup
        self.balance = balance
        self.sample_balanced = balance is not None
        if balance is not None:
            print("Setting up balanced sampler...")
            self.curiculum = curiculum
            self.element_weights = (
                None  # Placeholder for potential future use of per-sample weights
            )
            self.replacement = (
                replacement  # Balanced sampling typically requires replacement
            )

            if self.base_dataset.labels is None:
                raise ValueError("Dataset does not have labels for balanced sampling.")
            # Map labels to the current Subset if necessary
            if isinstance(dataset, Subset):
                print("1. Mapping labels back to current Subset")
                labels = {}
                subset_base_indices = np.asarray(self.subset_base_indices)

                labels["all"] = self.base_dataset.labels["all"][subset_base_indices]

                sc_mask = np.isin(self.base_dataset.sc_indices, subset_base_indices)
                labels["sc"] = self.base_dataset.labels["sc"][sc_mask]

                bulk_mask = np.isin(self.base_dataset.bulk_indices, subset_base_indices)
                labels["bulk"] = self.base_dataset.labels["bulk"][bulk_mask]
            else:
                labels = self.base_dataset.labels

            print("2. Computing label weights...")
            counts = {key: np.bincount(labels[key]) for key in labels}
            label_weights = {
                key: (weight_scaler * counts[key]) / (counts[key] + weight_scaler)
                for key in counts
            }
            self.label_weights = {
                key: torch.as_tensor(
                    label_weights[key], dtype=torch.float32
                ).share_memory_()
                for key in label_weights
            }
            # Build class indices for weighted sampling
            print("3. Building class indices")
            print(f"Building class indices with {num_workers} workers...")
            klass_indices = {
                key: self._build_class_indices(
                    labels[key], key, chunk_size, num_workers
                )
                for key in labels
            }

            # Convert klass_indices to a single tensor and offset vector
            all_indices = {key: [] for key in klass_indices}
            offsets = {key: [] for key in klass_indices}
            current_offset = {key: 0 for key in klass_indices}

            # Build concatenated tensor and track offsets
            print("Concatenating...")
            for modality in klass_indices.keys():
                print(modality)
                # Sort keys to ensure consistent ordering
                keys = klass_indices[modality].keys()
                print(f"{modality}: {keys}")
                for i in range(max(keys) + 2):
                    try:
                        offsets[modality].append(current_offset[modality])
                        if i in keys:
                            indices = klass_indices[modality][i]
                            all_indices[modality].append(indices)
                            current_offset[modality] += len(indices)
                    except Exception as e:
                        print(e)
                        print(
                            f"Encountered problem with key {i} in modality {modality}"
                        )

            # Convert to tensors
            print("4. Converting to tensors")
            self.klass_indices = {
                modality: torch.cat(all_indices[modality])
                .to(torch.int32)
                .share_memory_()
                for modality in all_indices
            }
            self.klass_offsets = {
                modality: torch.tensor(
                    offsets[modality], dtype=torch.long
                ).share_memory_()
                for modality in offsets
            }
            print(
                f"Done initializing balanced sampler for {len(self.klass_offsets)} modalities with {len(self.klass_offsets["all"])} classes"
            )

    def _build_class_indices(
        self, labels: np.ndarray, modality: str, chunk_size: int, n_workers: int
    ):
        """Build class indices in parallel across multiple workers.

        Args:
            labels: array of class labels
            modality: the modality group to which the labels belong ("sc", "bulk", or "all")
            n_workers: number of parallel workers
            chunk_size: size of chunks to process

        Returns:
            dictionary mapping class labels to tensors of indices
        """
        n = len(labels)
        assert modality in ["sc", "bulk", "all"], f"Invalid modality: {modality}"
        if modality == "sc":
            assert n == len(
                self.sc_indices
            ), "Label length does not match number of SC samples"
        if modality == "bulk":
            assert n == len(
                self.bulk_indices
            ), "Label length does not match number of bulk samples"
        if modality == "all":
            assert n == len(
                self.subset_base_indices
            ), "Label length does not match number of total samples"

        print(f"For {modality}:")

        results = []
        # Create chunks of the labels array with proper sizing
        n_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
        print(f"Processing {n:,} elements in {n_chunks} chunks...")

        # Process in chunks to limit memory usage
        if n_workers == 1:
            # Process sequentially without multiprocessing
            for i in tqdm(
                range(n_chunks), total=n_chunks, desc="Processing chunks sequentially"
            ):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n)
                results.append(
                    self._process_chunk_with_slice((start_idx, end_idx, labels))
                )
            print("Merging results from all chunks...")
            return self._merge_chunk_results(results)

        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            # Submit chunks for processing
            futures = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n)
                # We pass only chunk boundaries, not the data itself
                # This avoids unnecessary copies during process creation
                futures.append(
                    executor.submit(
                        self._process_chunk_with_slice,
                        (start_idx, end_idx, labels),
                    )
                )

            # Collect results as they complete with progress reporting
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chunks"
            ):
                results.append(future.result())

        # Merge results from all chunks
        print("Merging results from all chunks...")
        merged_results = self._merge_chunk_results(results)

        return merged_results

    def _process_chunk_with_slice(self, slice_info):
        """Process a slice of the labels array by indices.

        Args:
            slice_info: tuple of (start_idx, end_idx, labels_array) where
                       start_idx and end_idx define the slice to process

        Returns:
            dict mapping class labels to arrays of indices
        """
        start_idx, end_idx, labels_array = slice_info

        # We're processing a slice of the original array
        labels_slice = labels_array[start_idx:end_idx]
        chunk_indices = {}

        # Create a direct map of indices
        indices = np.arange(start_idx, end_idx)

        # Get unique labels in this slice for more efficient processing
        unique_labels = np.unique(labels_slice)
        # For each valid label, find its indices
        for label in unique_labels:
            # Find positions where this label appears (using direct boolean indexing)
            label_mask = labels_slice == label
            chunk_indices[int(label)] = indices[label_mask]

        return chunk_indices

    def _merge_chunk_results(self, results_list):
        """Merge results from multiple chunks into a single dictionary.

        Args:
            results_list: list of dictionaries mapping class labels to index arrays

        Returns:
            merged dictionary with PyTorch tensors
        """
        merged = {}

        # Collect all labels across all chunks
        all_labels = set()
        for chunk_result in results_list:
            all_labels.update(chunk_result.keys())

        # For each unique label
        for label in all_labels:
            # Collect indices from all chunks where this label appears
            indices_lists = [
                chunk_result[label]
                for chunk_result in results_list
                if label in chunk_result
            ]

            if indices_lists:
                # Concatenate all indices for this label
                merged[label] = torch.tensor(
                    np.concatenate(indices_lists), dtype=torch.long
                )
            else:
                merged[label] = torch.tensor([], dtype=torch.long)

        return merged

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = np.random.default_rng()
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

            # If group information is available, we sample pseudobulk from the same group to ensure feasibility of the synthetic samples
            if self.base_dataset.group_column is not None:
                # Sample groups for the pseudobulk samples
                pb_groups = rng.choice(
                    self.base_dataset.groups,
                    size=self.n_pb,
                    replace=True,
                )
                pb_sc_indices = []
                for g in pb_groups:
                    sc_pool = self.base_dataset.group_to_sc[g]
                    pb_sc_indices.extend(
                        self.sample(
                            sc_pool,
                            size=self.n_sc_per_pb,
                            modality="pb",
                            balanced=self.sample_balanced,
                        )
                    )
                pb_idx = np.array(pb_sc_indices)
            # Else, we sample pseudobulk from the same pool as the single-cell samplesº
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
            rng = np.random.default_rng()
            return rng.choice(
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
            sample_modality = "sc"  # Must be later improved to account for sensible pseudobulk generation

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
