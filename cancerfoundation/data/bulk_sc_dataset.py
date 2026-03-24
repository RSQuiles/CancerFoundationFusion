from typing import Optional

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler, Subset

from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

from .dataset import DatasetDir


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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load_mapping(self) -> dict:
        with self.data_dir.mapping_path.open("r") as f:
            return json.load(f)

    def _load_vocab(self) -> dict[str, int]:
        with open(self.data_dir.vocab_path, "r") as f:
            return json.load(f)

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
    ):
        # Account for the Subset resulting from random_split
        # Since the Subset uses a local set of indices, different from the original dataset
        if isinstance(dataset, Subset):
            self.dataset = dataset
            subset_base_indices = dataset.indices
            base_to_subset = {
                base_idx: sub_idx
                for sub_idx, base_idx in enumerate(subset_base_indices)
            }

            self.base_dataset = dataset.dataset

            self.bulk_indices = [
                base_to_subset[i]
                for i in self.base_dataset.bulk_indices
                if i in base_to_subset
            ]
            self.sc_indices = [
                base_to_subset[i]
                for i in self.base_dataset.sc_indices
                if i in base_to_subset
            ]
        else:
            self.dataset = dataset
            self.subset_indices = None
            self.bulk_indices = self.dataset.bulk_indices
            self.sc_indices = self.dataset.sc_indices

        self.batch_size = batch_size
        self.bulk_ratio = bulk_ratio
        self.pb_ratio = pb_ratio
        self.drop_last = drop_last
        self.shuffle = shuffle
        """
        The sampling follows the following structure:
            1. Single-Cell samples
            2. Single-Cell samples to generate pseudobulk
            3. Real Bulk samples
        """
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

        self._n_batches = len(self.bulk_indices)

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        rng = np.random.default_rng()
        batch_order = np.arange(self._n_batches)

        if self.shuffle:
            rng.shuffle(batch_order)

        for _ in batch_order:
            indices: list[int] = []
            # Order matters for the collator: [sc_0, ..., sc_{n-1}, pseudobulk_0, ...].
            sc_idx = rng.choice(
                self.sc_indices,
                size=self.n_sc,
                replace=len(self.sc_indices) < self.n_sc,
            )

            # If group information is available, we sample pseudobulk from the same group to ensure feasibility of the synthetic samples
            if self.base_dataset.group_column is not None:
                # Sample groups for the pseudobulk samples
                pb_groups = rng.choice(
                    self.base_dataset.groups,
                    size=self.n_pb,
                    replace=len(self.base_dataset.groups) < self.n_pb,
                )
                pb_sc_indices = []
                for g in pb_groups:
                    sc_pool = self.base_dataset.group_to_sc[g]
                    pb_sc_indices.extend(
                        rng.choice(
                            sc_pool,
                            size=self.n_sc_per_pb,
                            replace=len(sc_pool) < self.n_sc_per_pb,
                        ).tolist()
                    )
                pb_idx = np.array(pb_sc_indices)
            # Else, we sample pseudobulk from the same pool as the single-cell samplesº
            else:
                pb_idx = rng.choice(
                    self.sc_indices,
                    size=self.n_pb * self.n_sc_per_pb,
                    replace=len(self.sc_indices) < self.n_pb * self.n_sc_per_pb,
                )

            bulk_idx = rng.choice(
                self.bulk_indices,
                size=self.n_bulk,
                replace=len(self.bulk_indices) < self.n_bulk,
            )
            indices.extend(sc_idx.tolist())
            indices.extend(pb_idx.tolist())
            indices.extend(bulk_idx.tolist())
            yield indices
