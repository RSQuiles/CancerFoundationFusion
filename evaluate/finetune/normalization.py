"""Single source of truth for input normalization in the downstream tasks.

Every downstream task used to decide normalization for itself, through two config
keys with *opposite* meanings and a model default that reads backwards
(``embed(normalized=True)`` means "already normalized, so skip"). The result was
that ``canc_type_class`` embedded log1p values through the model but CP10K+log1p
values through the PCA baseline, and ``survival`` silently changed preprocessing
when ``finetune_embedder`` was flipped.

This module replaces all of that with one boolean in "do it?" space:

    normalize = True   ->  the data MUST be normalized (CP10K + log1p)
    normalize = False  ->  nothing is applied, anywhere

Resolution order is CLI flag -> task config ``normalize:`` -> ``False``.

The design rule that makes this safe: normalization happens **once, here**, and
every embedder is then called as a pass-through via :meth:`NormalizationPolicy.embed_kwargs`.
Both ``CancerFoundation.embed`` and ``PCAEmbedder.embed`` are no-ops under those
kwargs, so the model and its baseline see identical input by construction rather
than by coincidence.

``looks_like_counts`` is imported lazily inside :meth:`NormalizationPolicy.apply`,
because reaching it pulls in the whole ``cancerfoundation`` package (and therefore
torch). That keeps :func:`resolve_policy` and
:meth:`NormalizationPolicy.embed_kwargs` — the parts a planner or a config check
needs — usable without the training stack installed.
``evaluate/check/check_normalization_policy.py`` stubs the heavy deps to exercise
the rest, the same way ``evaluate/check/check_embed_panel_flow.py`` does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# The one key downstream configs should carry, in "do it?" space.
CANONICAL_KEY = "normalize"

# Written alongside CANONICAL_KEY by run_downstream_task when the value came from
# the CLI rather than the YAML, so the reported source stays honest after the
# override has been folded into the config object the tasks actually see.
SOURCE_KEY = "normalize_source"

# Retired keys. They are NOT synonyms of each other, which is why they are gone:
#   normalize_input: true   meant "please normalize"
#   normalized:      true   meant "already normalized, so do nothing"
# Reading them as a fallback would require silently inverting one of them, so
# instead they are reported and ignored.
_OBSOLETE_KEYS: tuple[str, ...] = ("normalized", "normalize_input")

# Library size each cell is scaled to before log1p. 1e4 (CP10K) matches what
# CancerFoundation.embed and PCAEmbedder use internally.
_TARGET_SUM = 1e4

# Every apply() appends its provenance here so run_downstream_task can write the
# full record into results_<task>.json without threading an object through five
# task files. Per-process, which is what we want: under DDP each rank has its own,
# and only rank 0 writes results.
_PROVENANCE: list[dict[str, Any]] = []


def drain_provenance() -> list[dict[str, Any]]:
    """Return every provenance record collected so far and reset the collector."""
    out = list(_PROVENANCE)
    _PROVENANCE.clear()
    return out


def _densify(X: Any) -> np.ndarray:
    """Return X as a dense float32 array, accepting scipy sparse or ndarray."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _matrix_stats(X: Any) -> dict[str, float]:
    """Cheap descriptive stats used in the log lines and the provenance record."""
    if hasattr(X, "toarray"):
        vals = np.asarray(X.tocsr().data, dtype=np.float64)
        size = float(X.shape[0]) * float(X.shape[1])
        nnz = float(vals.size)
    else:
        arr = np.asarray(X, dtype=np.float64)
        vals = arr[arr != 0]
        size = float(arr.size)
        nnz = float(vals.size)

    vals = vals[np.isfinite(vals)] if vals.size else vals
    return {
        "max": float(vals.max()) if vals.size else 0.0,
        "mean_nonzero": float(vals.mean()) if vals.size else 0.0,
        "frac_nonzero": (nnz / size) if size else 0.0,
    }


@dataclass(frozen=True)
class NormalizationPolicy:
    """What to do to a downstream task's expression matrix before embedding.

    Attributes:
        normalize: True -> apply CP10K + log1p. False -> apply nothing.
        source: Where the decision came from -- "cli", "config", or "default".
            Reported in the logs and in the results JSON so a benchmark bar can be
            traced back to the preprocessing that produced it.
    """

    normalize: bool
    source: str

    # -- embedder contract ---------------------------------------------------

    def embed_kwargs(self) -> dict[str, Any]:
        """Kwargs that make any embedder's own normalization a pass-through.

        Always the same pair, whatever ``normalize`` is, because :meth:`apply` has
        already done whatever was asked. Both members matter:

          * ``normalized=True``  -> CancerFoundation.embed skips its log1p, and
            PCAEmbedder.embed skips its CP10K+log1p.
          * ``log1p_only=False`` -> without it, ``PCAEmbedder.embed`` takes its
            ``elif log1p_only:`` branch and applies log1p *despite* normalized=True.

        Pass with ``**policy.embed_kwargs()`` rather than spelling the values out,
        so the two never drift apart at a call site.
        """
        return {"normalized": True, "log1p_only": False}

    # -- the transform -------------------------------------------------------

    def apply(self, adata: Any, task: str) -> tuple[Any, dict[str, Any]]:
        """Normalize ``adata.X`` in place on a copy, per this policy.

        Returns ``(adata, provenance)``. The input is never mutated: when the
        transform runs, a copy is taken first.

        With ``normalize=True`` the matrix is checked with ``looks_like_counts``
        first. If it does not look like raw counts it is left alone and a loud
        warning is emitted -- log1p'ing an already-log1p matrix is silent and
        destroys the scale, so skipping is the safer failure.

        CP10K + log1p is per-row and carries no fitted state, so applying it to a
        train and a test split separately is identical to applying it before the
        split. There is no leakage concern.
        """
        before = _matrix_stats(adata.X)

        if not self.normalize:
            log.info(
                "[norm] %s: normalization DISABLED (source=%s) -- embedding X as-is "
                "(max=%.3g, mean_nonzero=%.3g, nonzero=%.1f%%)",
                task, self.source, before["max"], before["mean_nonzero"],
                100.0 * before["frac_nonzero"],
            )
            return adata, self._provenance(
                applied=False,
                reason="normalize=False",
                is_counts=None,
                before=before,
                after=before,
                task=task,
            )

        # Lazy: reaching this pulls in the cancerfoundation package, hence torch.
        from cancerfoundation.data.preprocess import looks_like_counts

        is_counts = bool(looks_like_counts(adata.X))

        if not is_counts:
            log.warning(
                "[norm] %s: normalize=True (source=%s) but X does NOT look like raw "
                "counts (max=%.3g, mean_nonzero=%.3g) -- it is probably already "
                "log1p. SKIPPING CP10K+log1p to avoid double normalization. "
                "Pass --no-normalize to silence this, or check the input path.",
                task, self.source, before["max"], before["mean_nonzero"],
            )
            return adata, self._provenance(
                applied=False,
                reason="safeguard: X does not look like raw counts",
                is_counts=False,
                before=before,
                after=before,
                task=task,
            )

        adata = adata.copy()
        X = _densify(adata.X)
        row_sums = X.sum(axis=1, keepdims=True)
        # A zero-library row would divide by zero; leave it as the all-zero row it is.
        np.divide(X, np.where(row_sums == 0.0, 1.0, row_sums), out=X)
        X *= _TARGET_SUM
        np.log1p(X, out=X)
        adata.X = X

        after = _matrix_stats(X)
        log.info(
            "[norm] %s: applied CP10K+log1p (source=%s) -- max %.3g -> %.3g, "
            "mean_nonzero %.3g -> %.3g, nonzero=%.1f%%",
            task, self.source, before["max"], after["max"],
            before["mean_nonzero"], after["mean_nonzero"],
            100.0 * after["frac_nonzero"],
        )
        return adata, self._provenance(
            applied=True,
            reason="normalize=True and X looks like raw counts",
            is_counts=True,
            before=before,
            after=after,
            task=task,
        )

    # -- provenance ----------------------------------------------------------

    def _provenance(
        self,
        applied: bool,
        reason: str,
        is_counts: bool | None,
        before: dict[str, float],
        after: dict[str, float],
        task: str,
    ) -> dict[str, Any]:
        record = {
            "task": task,
            "normalize": self.normalize,
            "source": self.source,
            "applied": applied,
            "reason": reason,
            "looks_like_counts": is_counts,
            "x_max_before": round(before["max"], 6),
            "x_max_after": round(after["max"], 6),
        }
        _PROVENANCE.append(record)
        return record

    def describe(self) -> str:
        """One-line human summary for banners and summary tables."""
        what = "CP10K+log1p" if self.normalize else "none"
        return f"normalize={self.normalize} ({what}), source={self.source}"


def resolve_policy(task_cfg: Any, cli_normalize: bool | None) -> NormalizationPolicy:
    """Resolve the normalization policy for one task.

    Precedence: ``cli_normalize`` (when not None) -> ``task_cfg.normalize`` ->
    ``False``. The default is off: absent from both the CLI and the config means
    nothing is applied anywhere.

    A config still carrying a retired key (``normalized`` / ``normalize_input``) is
    reported at WARNING and the key ignored -- see :data:`_OBSOLETE_KEYS` for why
    they cannot be honoured as fallbacks.
    """
    for key in _OBSOLETE_KEYS:
        if task_cfg is not None and getattr(task_cfg, key, None) is not None:
            log.warning(
                "[norm] config key '%s: %s' is obsolete and IGNORED. Use "
                "'%s: <bool>' instead, which is in 'do it?' space: true means the "
                "data must be normalized. Note '%s' meant the opposite of that.",
                key, getattr(task_cfg, key), CANONICAL_KEY,
                "normalized" if key == "normalized" else key,
            )

    if cli_normalize is not None:
        return NormalizationPolicy(normalize=bool(cli_normalize), source="cli")

    if task_cfg is not None:
        value = getattr(task_cfg, CANONICAL_KEY, None)
        if value is not None:
            # run_downstream_task folds a CLI override into the config object the
            # tasks see, and stamps SOURCE_KEY so this still reports "cli".
            source = str(getattr(task_cfg, SOURCE_KEY, None) or "config")
            return NormalizationPolicy(normalize=bool(value), source=source)

    return NormalizationPolicy(normalize=False, source="default")
