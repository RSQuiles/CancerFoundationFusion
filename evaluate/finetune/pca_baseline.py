"""PCA baseline embedder for downstream evaluation.

Provides a drop-in replacement for the CancerFoundation embedder that uses
sklearn PCA on (optionally normalised) raw gene expression instead of a
pretrained transformer.  Fitting happens lazily on the first embed() call
(assumed to be training data); all subsequent calls only transform.  This
matches the calling convention used by prepare_datasets() in every task
implementation, where training data is always embedded before test data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class PCAEmbedder:
    """Embed cells with PCA for use as a downstream-evaluation baseline.

    Parameters
    ----------
    n_components:
        Number of PCA components to retain (upper-bounded by the number of
        genes and the number of training samples minus one).
    random_state:
        Random seed passed to sklearn's PCA for reproducibility.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._pca = None

    # ------------------------------------------------------------------
    # Public API (mirrors CancerFoundation.embed)
    # ------------------------------------------------------------------

    def embed(
        self,
        adata: Any,
        batch_size: int = 64,
        normalized: bool = True,
        log1p_only: bool = False,
    ) -> pd.DataFrame:
        """Embed *adata* with PCA and return a cell × component DataFrame.

        The first call fits the PCA on the provided data; subsequent calls
        only transform.

        Parameters
        ----------
        adata:
            AnnData object whose ``.X`` attribute holds the expression matrix.
        batch_size:
            Ignored; kept for API compatibility with the model's embed().
        normalized:
            If True, library-size normalise to 10k counts per cell then apply
            log1p — the same pre-processing the model uses by default.
        log1p_only:
            If True (and *normalized* is False), apply log1p without library-
            size normalisation.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None

        from sklearn.decomposition import PCA

        X = adata.X
        if sp is not None and sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        if normalized:
            row_sums = X.sum(axis=1, keepdims=True)
            safe_sums = np.where(row_sums == 0, 1.0, row_sums)
            X = X / safe_sums * 1e4
            X = np.log1p(X)
        elif log1p_only:
            X = np.log1p(X)

        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])

        if self._pca is None:
            self._pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_emb = self._pca.fit_transform(X)
            log.info(
                "PCAEmbedder: fitted on %d samples × %d genes → %d components "
                "(explained variance: %.3f)",
                X.shape[0],
                X.shape[1],
                n_comp,
                float(self._pca.explained_variance_ratio_.sum()),
            )
        else:
            X_emb = self._pca.transform(X)
            log.info(
                "PCAEmbedder: transformed %d samples → %d components",
                X.shape[0],
                n_comp,
            )

        return pd.DataFrame(X_emb, index=adata.obs_names)
