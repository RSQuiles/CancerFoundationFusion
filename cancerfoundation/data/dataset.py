from typing import Optional
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from torch.utils.data import Dataset
import torch


import json

from pathlib import Path
import pandas as pd

import numpy as np


# TODO: Move from_h5ads to its own script!


class DatasetDir:
    VOCAB_PATH = "vocab.json"
    MEMMAP_PATH = "mem.map"
    MAPPING_PATH = "mapping.json"
    OBS_PATH = "obs.parquet"

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def validate(self):
        return all(
            [
                self.vocab_path.is_file(),
                self.obs_path.is_file(),
                self.mapping_path.is_file(),
                self.memmap_path.is_file(),
            ]
        )

    def mkdir(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)

    @property
    def memmap_path(self):
        return self.data_dir / self.MEMMAP_PATH

    @property
    def mapping_path(self):
        return self.data_dir / self.MAPPING_PATH

    @property
    def vocab_path(self):
        return self.data_dir / self.VOCAB_PATH

    @property
    def obs_path(self):
        return self.data_dir / self.OBS_PATH


class SingleCellDataset(Dataset):
    GENE_ID = "_cf_gene_id"  # column in the obs dataframe that contains the gene ids corresponding to the expression values
    CLS_TOKEN = "<cls>"
    PAD_TOKEN = "<pad>"

    def __init__(
        self,
        data_dir: str | Path,
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

        assert self.memmap.number_of_rows() == self.obs.shape[0]

    def _load_mapping(self) -> dict[str, int]:
        with self.data_dir.mapping_path.open("r") as f:
            mapping = json.load(f)
        return mapping

    def _load_vocab(self) -> dict[str, int]:
        with open(self.data_dir.vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_metadata(self, key: str) -> np.array:
        return self.obs[key].values

    def __len__(self):
        return self.memmap.number_of_rows()

    def __getitem__(self, index: int) -> dict[str, int | torch.Tensor]:
        # Return expression values and gene ids from _cf_gene_id
        exp, genes = self.memmap.get_row_padded(
            index, return_features=True, feature_vars=[self.GENE_ID]
        )

        # Add CLS token at the beginning of the sequence and pad gene ids with pad_value
        genes = np.insert(genes[0], 0, self.vocab["<cls>"])
        exp = np.insert(exp, 0, self.pad_value)

        data = {
            "expressions": torch.Tensor(exp).type(torch.float32),
            "genes": torch.from_numpy(genes),
        }

        if self.obs_columns is None:
            return data

        row = self.obs.iloc[index]
        for column in self.obs_columns:
            data[column] = row[column]

        return data
