"""Read-only access to h5ad files: obs, var, obsm and selected rows of X, via h5py.

Nothing here writes to a source file, and ``uns`` is never read, so files whose
``uns`` anndata cannot parse (e.g. the CellxGene ``log1p/base`` entry) open as they
are. Needs h5py, numpy, pandas, scipy and anndata; no torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_elem(group):
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(group)


def _sparse_dataset(group):
    try:
        from anndata.io import sparse_dataset
    except ImportError:
        try:
            from anndata.experimental import sparse_dataset
        except ImportError:  # anndata < 0.10
            from anndata._core.sparse_dataset import SparseDataset as sparse_dataset
    return sparse_dataset(group)


class SourceReader:
    """obs, var and rows of one h5ad, opened read-only. Use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        import h5py

        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        self.obs: pd.DataFrame = _read_elem(self._file["obs"])
        self.var: pd.DataFrame = _read_elem(self._file["var"])

    def __enter__(self) -> "SourceReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def n_obs(self) -> int:
        return len(self.obs)

    def gene_names(self, gene_column: str | None = None) -> pd.Index:
        """The var index, or a var column holding the gene names."""
        if gene_column is None:
            return pd.Index(self.var.index.astype(str))
        if gene_column not in self.var.columns:
            raise KeyError(f"{self.path.name}: var has no column {gene_column!r}; "
                           f"available: {list(self.var.columns)}")
        return pd.Index(self.var[gene_column].astype(str))

    def obsm(self, key: str) -> np.ndarray:
        if "obsm" not in self._file or key not in self._file["obsm"]:
            available = list(self._file["obsm"].keys()) if "obsm" in self._file else []
            raise KeyError(f"{self.path.name}: no obsm[{key!r}]; available: {available}")
        return np.asarray(_read_elem(self._file["obsm"][key]))

    def rows(self, positions: np.ndarray, chunk_rows: int = 100_000):
        """CSR matrix of the rows at ``positions`` (sorted ascending), read in chunks."""
        import h5py
        import scipy.sparse as sp

        positions = np.asarray(positions, dtype=np.int64)
        X = self._file["X"]
        blocks = []
        for start in range(0, len(positions), chunk_rows):
            idx = positions[start:start + chunk_rows]
            if isinstance(X, h5py.Group):
                block = _sparse_dataset(X)[idx]
                block = block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)
            else:
                block = sp.csr_matrix(X[idx])
            blocks.append(block)
        if not blocks:
            return sp.csr_matrix((0, len(self.var)))
        return sp.vstack(blocks, format="csr")

    def subset(self, positions: np.ndarray):
        """In-memory AnnData of the rows at ``positions`` (X, obs and var only)."""
        import anndata as ad

        return ad.AnnData(X=self.rows(positions), obs=self.obs.iloc[positions].copy(), var=self.var.copy())

    def close(self) -> None:
        self._file.close()


def expand_h5ad_paths(paths: list[str | Path]) -> list[Path]:
    """Files as given, directories expanded to their ``*.h5ad`` (sorted)."""
    out: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            out.extend(sorted(path.glob("*.h5ad")))
        elif path.exists():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise FileNotFoundError(f"No .h5ad files in {paths}")
    return out
