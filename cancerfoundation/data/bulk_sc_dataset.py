from typing import Optional

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler

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
        self.memmap = SingleCellMemMapDataset(self.data_dir.memmap_path)
        self.obs = pd.read_parquet(self.data_dir.obs_path)
        self.mapping = self._load_mapping()
        self.obs_columns = obs_columns

        self.modality_column = modality_column
        self.group_column = group_column

        assert self.memmap.number_of_rows() == self.obs.shape[0]
        assert modality_column in self.obs.columns

        # Pre-compute index arrays per modality
        modality_vals = self.obs[modality_column].values
        bulk_code = self.mapping[modality_column][bulk_label]
        sc_code = self.mapping[modality_column][sc_label]

        self.bulk_indices = np.where(modality_vals == bulk_code)[0]
        self.sc_indices = np.where(modality_vals == sc_code)[0]

        assert len(self.bulk_indices) > 0, "No bulk samples found"
        assert len(self.sc_indices) > 0, "No SC samples found"

        # Per-group index pools (used by BulkSCBatchSampler)
        if group_column is not None:
            assert group_column in self.obs.columns
            group_vals = self.obs[group_column].values

            self.group_to_bulk: dict[int, np.ndarray] = {}
            self.group_to_sc: dict[int, np.ndarray] = {}
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
            self.group_to_bulk = None
            self.group_to_sc = None
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
        ``modality`` tag (``0`` = bulk, ``1`` = SC)."""
        exp, genes = self.memmap.get_row_padded(
            index, return_features=True, feature_vars=[self.GENE_ID]
        )
        genes = np.insert(genes[0], 0, self.vocab[self.CLS_TOKEN])
        exp = np.insert(exp, 0, self.pad_value)

        data = {
            "expressions": torch.tensor(exp, dtype=torch.float32),
            "genes": torch.from_numpy(genes),
            "modality": int(self.obs.iloc[index][self.modality_column]),
        }

        if self.obs_columns is not None:
            row = self.obs.iloc[index]
            for col in self.obs_columns:
                data[col] = row[col]

        return data


# ======================================================================
# Batch sampler
# ======================================================================


class BulkSCBatchSampler(Sampler[list[int]]):
    """Yields batches where each *pair* = ``n_sc_samples`` SC indices +
    ``n_bulk_samples`` bulk indices, all from the same group.

    ``batch_size`` controls the number of pseudobulk↔bulk **pairs** per
    batch, so the effective number of memmap rows loaded is
    ``batch_size * (n_sc_samples + n_bulk_samples)``.

    Parameters
    ----------
    dataset : BulkSCDataset
        Must have ``groups``, ``group_to_sc``, ``group_to_bulk`` populated
        (i.e. a ``group_column`` was set), *or* ``groups is None`` for the
        un-grouped case where any SC can pair with any bulk.
    batch_size : int
        Number of pseudobulk↔bulk pairs per batch.
    n_sc_samples : int
        SC rows per pseudobulk aggregate.
    n_bulk_samples : int
        Bulk rows per pair.
    drop_last : bool
        Drop the last incomplete batch.
    shuffle : bool
        Shuffle group order each epoch.
    """

    def __init__(
        self,
        dataset: "BulkSCDataset",
        batch_size: int,
        n_sc_samples: int = 32,
        n_bulk_samples: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_sc = n_sc_samples
        self.n_bulk = n_bulk_samples
        self.drop_last = drop_last
        self.shuffle = shuffle

        if dataset.groups is not None:
            self._n_pairs = len(dataset.groups)
        else:
            self._n_pairs = len(dataset.bulk_indices)

    def __len__(self) -> int:
        if self.drop_last:
            return self._n_pairs // self.batch_size
        return (self._n_pairs + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        ds = self.dataset

        if ds.groups is not None:
            group_order = np.array(ds.groups)
        else:
            group_order = np.arange(self._n_pairs)

        if self.shuffle:
            rng.shuffle(group_order)

        for start in range(0, len(group_order), self.batch_size):
            batch_groups = group_order[start : start + self.batch_size]
            if self.drop_last and len(batch_groups) < self.batch_size:
                break

            indices: list[int] = []
            # We also need to tell the collator which indices belong to
            # which pair and which modality.  We encode this in the *order*:
            # for each pair we emit [sc_0, ..., sc_{n-1}, bulk_0, ..., bulk_{m-1}]
            for g in batch_groups:
                if ds.groups is not None:
                    sc_pool = ds.group_to_sc[g]
                    bulk_pool = ds.group_to_bulk[g]
                else:
                    sc_pool = ds.sc_indices
                    bulk_pool = ds.bulk_indices

                sc_idx = rng.choice(
                    sc_pool,
                    size=self.n_sc,
                    replace=len(sc_pool) < self.n_sc,
                )
                bulk_idx = rng.choice(
                    bulk_pool,
                    size=self.n_bulk,
                    replace=len(bulk_pool) < self.n_bulk,
                )
                indices.extend(sc_idx.tolist())
                indices.extend(bulk_idx.tolist())

            yield indices
