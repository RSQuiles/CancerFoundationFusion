from __future__ import annotations

import sys
from typing import Union

import numpy as np


def _is_tensor(x) -> bool:
    # torch is imported lazily: if it was never loaded, x cannot be a tensor, and the
    # numpy-only helpers here (normalization, value_scale) stay usable without it.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(x, torch.Tensor)


def looks_like_counts(X, max_rows: int = 512, seed: int = 0) -> bool:
    """Return True when ``X`` looks like raw counts rather than log1p values.

    Two conditions must hold, because either alone gives false positives:
      - the maximum exceeds 20 (log1p of a plausible CP10K value stays well below
        that), and
      - the values are integral (counts are whole numbers; log1p values are not).

    Accepts dense arrays, scipy sparse matrices, and torch tensors. Only the
    non-zero entries of at most ``max_rows`` randomly drawn rows are inspected, so
    this is cheap enough to call on a full expression matrix.

    Note this is a heuristic, not a guarantee: a CPM matrix that happens to be
    integral would be reported as counts. Prefer an explicit flag (the training
    ``--input-data`` argument, or ``embed(normalized=...)``) where one exists, and
    use this to *check* that flag rather than to replace it.
    """
    import scipy.sparse as sp

    if _is_tensor(X):
        X = X.detach().cpu().numpy()

    n_rows = X.shape[0] if X.ndim == 2 else 1
    if n_rows > max_rows:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
        X = X[rows]

    if sp.issparse(X):
        vals = np.asarray(X.tocsr().data)
    else:
        vals = np.asarray(X).ravel()
        vals = vals[vals != 0]

    if vals.size == 0:
        return False

    vals = vals[np.isfinite(vals)].astype(np.float64)
    if vals.size == 0:
        return False

    return bool(vals.max() > 20) and bool(np.allclose(vals, np.round(vals)))


# Above this, values cannot be log-scale expression (log1p of 1e6 CPM is ~13.8).
LINEAR_SCALE_MIN_MAX = 30.0


def value_scale(X, max_rows: int = 512, seed: int = 0) -> str:
    """Classify a matrix as ``"counts"``, ``"linear"`` or ``"log"``.

    ``counts``: raw integer counts (``looks_like_counts``). ``linear``: not integral but
    far beyond any log-scale value, e.g. TPM, CPM or CP10K. ``log``: everything else,
    i.e. already log-transformed. A heuristic like ``looks_like_counts``.
    """
    import scipy.sparse as sp

    if looks_like_counts(X, max_rows=max_rows, seed=seed):
        return "counts"
    n_rows = X.shape[0] if X.ndim == 2 else 1
    if n_rows > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(n_rows, size=max_rows, replace=False))
        X = X[rows]
    vals = np.asarray(X.tocsr().data) if sp.issparse(X) else np.asarray(X).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size and float(vals.max()) > LINEAR_SCALE_MIN_MAX:
        return "linear"
    return "log"


NORMALIZE_METHODS = ("log1p", "cpm_log1p", "cp10k_log1p")
_NORMALIZE_TARGET = {"cpm_log1p": 1e6, "cp10k_log1p": 1e4}


def library_size(X) -> np.ndarray:
    """Per-row total, as float64 (dense or sparse input)."""
    return np.asarray(X.sum(axis=1), dtype=np.float64).ravel()


def normalize_expression(X, method: str | None, lib_size: np.ndarray | None = None) -> np.ndarray:
    """Apply one of ``NORMALIZE_METHODS`` (or nothing for ``None``) to a dense matrix.

    ``cpm_log1p`` / ``cp10k_log1p`` scale each row to 1e6 / 1e4 by ``lib_size`` and then
    log1p. Pass ``lib_size`` computed over ALL of a row's genes when ``X`` holds only
    some of them (after a vocabulary intersection or a panel subset); by default it is
    the row sum of ``X``. Rows with a zero library size stay zero.
    """
    if method is None:
        return X
    if method not in NORMALIZE_METHODS:
        raise ValueError(f"normalize must be None or one of {NORMALIZE_METHODS}, got {method!r}")
    X = np.asarray(X, dtype=np.float32)
    if method in _NORMALIZE_TARGET:
        lib = library_size(X) if lib_size is None else np.asarray(lib_size, dtype=np.float64)
        scale = np.divide(_NORMALIZE_TARGET[method], lib, out=np.zeros_like(lib), where=lib > 0)
        X = X * scale[:, None].astype(np.float32)
    return np.log1p(X)


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning_with_edges(
    row: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Same as binning() but also returns the (n_bins-1,) quantile edge array.

    Returns ``(binned_row, edges)`` where ``edges`` has shape ``(n_bins-1,)``.
    ``edges`` is all-zeros when the row is all-zero.  The edges are the
    expression-space quantile breakpoints used to assign bin indices, so two
    bin indices that share the same edge value represent the same expression
    level (repeated-quantile case).
    """
    if row.size == 0 or row.max() == 0:
        return np.zeros_like(row, dtype=np.int64), np.zeros(n_bins - 1, dtype=np.float32)

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        edges = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1)).astype(np.float32)
        non_zero_digits = _digitize(non_zero_row, edges)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        edges = np.quantile(row, np.linspace(0, 1, n_bins - 1)).astype(np.float32)
        binned_row = _digitize(row, edges).astype(np.int64)

    return binned_row, edges


def binning(
    row: Union[np.ndarray, "torch.Tensor"], n_bins: int  # noqa: F821
) -> Union[np.ndarray, "torch.Tensor"]:  # noqa: F821
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = not _is_tensor(row)
    if not return_np:
        import torch
    row = row.cpu().numpy() if not return_np else row
    # TODO: use torch.quantile and torch.bucketize

    if row.size == 0 or row.max() == 0:
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)
