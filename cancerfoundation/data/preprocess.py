from typing import Union

import numpy as np
import torch


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

    if isinstance(X, torch.Tensor):
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
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
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
