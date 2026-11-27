"""Embed an h5ad with ANY model (baselines, other foundation models) into a directory.

Every model is an ``Embedder`` registered under a name. The output is the same as
``embed.py``'s, so evaluation treats a baseline exactly like a CancerFoundation
model: ``<out-dir>/<name>.h5ad`` (obs + ``obsm["X_emb"]``, no X) plus
``<out-dir>/embed.json``.

    python embed_any.py --list
    python embed_any.py --embedder pca --input data.h5ad --out-dir out/ --option n_pcs=50
    python embed_any.py --embedder bulkformer --input bulk.h5ad --out-dir out/ \\
        --option repo=/path/to/bulkformer-latest --option gene_column=gene_id

``scvi`` and ``harmony`` are integration references: they use the batch label
(``batch_key``, default ``modality``); see ``SCVIEmbedder`` and ``HarmonyEmbedder``.

Adding a model
--------------
Subclass ``Embedder``, decorate it with ``@register("<name>")`` and implement
``embed`` (and ``fit`` if the model learns anything from the data, as PCA does).
``--option key=value`` pairs are passed to ``__init__`` as keyword arguments, with
values parsed as YAML scalars (``50`` -> int, ``true`` -> bool). ``PCAEmbedder`` is the
simplest complete example; ``BulkFormerEmbedder`` wraps an external model and runs in
that model's own environment.

``embed_splits.py --embedder <name>`` runs any registered embedder over a split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed.embed import EMB_KEY, NORMALIZE_CHOICES, write_embeddings  # noqa: E402

EMBEDDERS: dict[str, type[Embedder]] = {}


def register(name: str):
    """Class decorator adding an ``Embedder`` to the registry under ``name``."""
    def wrap(cls):
        if name in EMBEDDERS:
            raise ValueError(f"Embedder {name!r} registered twice")
        cls.name = name
        EMBEDDERS[name] = cls
        return cls
    return wrap


class Embedder:
    """Interface every model implements.

    ``fit`` sees the rows the model may learn from (a train subset, never the test
    rows), ``embed`` returns one row per obs of its input. ``genes`` is the input gene
    panel when the model selects one (``None`` otherwise); it becomes the
    ``panel_hash`` that alignment plots compare across files.
    """

    name: str = "base"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.genes: list[str] | None = None

    def fit(self, adata) -> None:  # noqa: B027 - optional hook, no-op by default
        """Learn whatever the model needs from ``adata``; default: nothing."""

    def embed(self, adata) -> np.ndarray:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        """What goes into ``embed.json`` beside the file (method and options)."""
        return {"method": self.name, "options": {k: str(v) for k, v in self.options.items()}}


# --------------------------------------------------------------------------- #
# Implemented embedders
# --------------------------------------------------------------------------- #


@register("pca")
class PCAEmbedder(Embedder):
    """Normalization, seurat HVGs and PCA, all fitted in ``fit``.

    Options: ``n_pcs`` (50), ``n_hvg`` (2000), ``seed`` (0) and ``normalize``
    (``cpm_log1p``): ``log1p``, ``cpm_log1p``, ``cp10k_log1p``, or ``none`` for the
    values as they are, the same choices as ``embed.py --normalize``. CPM / CP10K use
    each row's total over all its genes. The HVGs are chosen on their own normalized
    view of the data whatever ``normalize`` is (``gene_panel.scoring_view``), as for
    CancerFoundation.
    """

    def __init__(self, n_pcs: int = 50, n_hvg: int = 2000, normalize: str | None = "cpm_log1p",
                 seed: int = 0, **options: Any) -> None:
        legacy = sorted({"normalized", "log1p_only"} & set(options))
        if legacy:
            raise ValueError(f"PCA option(s) {legacy} were replaced by normalize=log1p|cpm_log1p|cp10k_log1p|none")
        if normalize in (None, False) or str(normalize).lower() == "none":
            normalize = None
        elif normalize not in NORMALIZE_CHOICES:
            raise ValueError(f"normalize must be one of {NORMALIZE_CHOICES} or none, got {normalize!r}")
        super().__init__(n_pcs=n_pcs, n_hvg=n_hvg, normalize=normalize, seed=seed, **options)
        self.n_pcs, self.n_hvg, self.seed = int(n_pcs), int(n_hvg), int(seed)
        self.normalize = normalize
        self._pca = None

    def _prepare(self, adata) -> np.ndarray:
        """The panel genes of ``adata`` as a dense matrix, normalized per ``normalize``.

        The library size is each row's total over all of its genes, taken before the
        panel subset, so only the panel genes are ever densified.
        """
        import scipy.sparse as sp

        from cancerfoundation.data.preprocess import library_size, normalize_expression

        lib = library_size(adata.X) if self.normalize else None
        X = adata[:, self.genes].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)
        return normalize_expression(X, self.normalize, lib)

    def fit(self, adata) -> None:
        from sklearn.decomposition import PCA

        from cancerfoundation.data.gene_panel import select_panel

        self.genes = select_panel(adata, "sc", min(self.n_hvg, adata.n_vars), flavor="seurat")
        X = self._prepare(adata)
        n_comps = min(self.n_pcs, len(self.genes), adata.n_obs - 1)
        self._pca = PCA(n_components=n_comps, random_state=self.seed).fit(X)
        print(f"PCA: {n_comps} components on {len(self.genes)} HVGs, fitted on {adata.n_obs} rows, "
              f"input {self.normalize or 'as stored'} "
              f"({self._pca.explained_variance_ratio_.sum():.1%} variance)")

    def embed(self, adata) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAEmbedder.embed called before fit")
        missing = sorted(set(self.genes) - set(adata.var_names))
        if missing:
            raise ValueError(f"{len(missing)} fitted HVGs are missing from the input, e.g. {missing[:5]}")
        return self._pca.transform(self._prepare(adata)).astype(np.float32)


# --------------------------------------------------------------------------- #
# BulkFormer (bulkformer-latest)
# --------------------------------------------------------------------------- #

BULKFORMER_REPO_ENV = "BULKFORMER_REPO"
BULKFORMER_INPUTS = ("auto", "counts", "cpm", "log1p_cpm", "tpm", "log_tpm", "log2_tpm")


def _load_bulkformer_api(repo: Path):
    """``utils.latest`` from a bulkformer-latest checkout, imported as ``bulkformer_latest``.

    Imported under its own name because this project has a top-level ``utils``
    package too; the two would shadow each other on ``sys.path``. The package only
    uses relative imports, so loading it by file location is enough.
    """
    import importlib.util

    name = "bulkformer_latest"
    if name in sys.modules:
        return sys.modules[name]
    package_dir = Path(repo) / "utils" / "latest"
    init = package_dir / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(
            f"{init} not found: repo= must be a clone of "
            "https://github.com/KangBoming/bulkformer-latest"
        )
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(package_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _count_like_rows(X) -> np.ndarray:
    """Per row: can BulkFormer's ``normalize_data`` treat it as raw counts?

    True when every value is finite, non-negative and integral, and the row has a
    positive total (the TPM denominator). ``X`` holds the vocabulary-matched genes.
    """
    import scipy.sparse as sp

    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64, copy=False)
    finite = np.isfinite(X).all(axis=1)
    safe = np.where(np.isfinite(X), X, 0.0)
    nonneg = (safe >= 0).all(axis=1)
    integral = np.isclose(safe, np.round(safe)).all(axis=1)
    return finite & nonneg & integral & (safe.sum(axis=1) > 0)


LOG_SCALE_MAX = 30.0  # a non-integral row with a larger value is linear (CPM, CP10K, TPM)
INTEGRAL_SHARE = 0.99  # share of non-zeros that must land on integers for an exact recovery
CONVERTED_NAMES_KEPT = 1000  # converted row names recorded per label in the provenance


def _integral_share(values: np.ndarray) -> float:
    """Share of ``values`` within float32 round-off of an integer (relative 1e-5, absolute 0.05)."""
    if not values.size:
        return 0.0
    return float(np.mean(np.isclose(values, np.round(values), rtol=1e-5, atol=0.05)))


def recover_counts(X) -> tuple[np.ndarray, np.ndarray]:
    """Bring each row back to raw counts, guessing its normalization. Numpy only.

    Returns ``(counts, labels)``, one label per row:
      ``counts``           integral already, unchanged;
      ``log1p->counts``    natural ln(x + 1) of a rescaled count row (ln(CP10K+1), ...);
      ``log2p->counts``    log2(x + 1) of one;
      ``linear->counts``   a rescaled count row (CPM, CP10K);
      ``...~``             the same transform undone, but the result is not integral
                           (e.g. TPM, or another log base): approximate, left unrounded;
      ``invalid``          negative, non-finite or zero total: not usable.

    The transform is undone, then the row is divided by its smallest non-zero value,
    which is one count in UMI data; integral results (``INTEGRAL_SHARE`` of the
    non-zeros) confirm the guess and are rounded. BulkFormer's TPM divides by the row's
    own total, so ln(TPM+1) does not depend on that scale: only the log base matters.
    """
    import scipy.sparse as sp

    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64, copy=True)
    labels = np.empty(len(X), dtype=object)
    for i, row in enumerate(X):
        if not np.isfinite(row).all() or (row < 0).any() or row.sum() <= 0:
            labels[i] = "invalid"
            continue
        nz = row[row > 0]
        if np.isclose(nz, np.round(nz)).all():
            labels[i] = "counts"
            continue
        if nz.max() <= LOG_SCALE_MAX:
            candidates = [("log1p", np.expm1(row)), ("log2p", np.exp2(row) - 1.0)]
        else:
            candidates = [("linear", row)]
        fallback = None
        for name, linear in candidates:
            scaled = linear / linear[linear > 0].min()
            if _integral_share(scaled[scaled > 0]) >= INTEGRAL_SHARE:
                X[i], labels[i] = np.round(scaled), f"{name}->counts"
                break
            fallback = fallback or (name, scaled)
        else:
            X[i], labels[i] = fallback[1], f"{fallback[0]}->counts~"
    return X, labels


def conversion_summary(labels, names) -> dict[str, Any]:
    """``{"n_rows", "n_converted", "n_invalid", "by_label": {label: n}, "rows": {label: [names]}}``.

    Rows labelled ``counts`` (or an explicit ``input=`` scale) are not conversions.
    """
    labels = np.asarray(labels, dtype=object)
    names = np.asarray(names).astype(str)
    by_label = {str(k): int(v) for k, v in zip(*np.unique(labels.astype(str), return_counts=True))}
    converted = np.array(["->" in str(label) for label in labels], dtype=bool)
    rows = {label: names[labels == label][:CONVERTED_NAMES_KEPT].tolist()
            for label in by_label if "->" in label}
    return {"n_rows": int(len(labels)), "n_converted": int(converted.sum()),
            "n_invalid": int((labels == "invalid").sum()), "by_label": by_label, "rows": rows}


def report_conversion(summary: dict[str, Any], who: str = "bulkformer") -> None:
    """The warning line for rows that were not raw counts and were converted."""
    if not summary["n_converted"]:
        return
    converted = {k: v for k, v in summary["by_label"].items() if "->" in k}
    examples = [n for names in summary["rows"].values() for n in names[:2]][:5]
    print(f"  [{who}] [WARNING] {summary['n_converted']} of {summary['n_rows']} rows were not raw "
          f"counts and were converted to counts: {converted}, e.g. {examples}"
          + ("; '~' = approximate (not integral after conversion)"
             if any(k.endswith("~") for k in converted) else ""))


@register("bulkformer")
class BulkFormerEmbedder(Embedder):
    """bulkformer-latest: mean of the contextual vectors of the observed genes (768-d).

    Follows ``notebooks/bulkformer-latest_feature_extraction.ipynb`` of
    https://github.com/KangBoming/bulkformer-latest, reading an AnnData instead of a
    CSV, and calls that repository's own ``normalize_data``, ``main_gene_selection``
    and ``extract_feature``, so the preprocessing is exactly theirs:

    1. Genes are matched to the release vocabulary (19,973 Ensembl IDs, version
       suffixes stripped). Unmatched input genes are dropped; vocabulary genes the
       input lacks are fed as missing (-10), never as zero.
    2. Raw counts become natural ln(TPM + 1) with the release gene lengths, the TPM
       denominator summing over vocabulary-matched genes only.
    3. The full gene axis goes through the encoder; the sample vector is the mean of
       the contextual vectors of the observed genes (measured zeros included).

    Options:
      ``repo``        bulkformer-latest checkout (default: ``$BULKFORMER_REPO``).
      ``model_dir``   the six release files (default: ``<repo>/model/bulkformer-latest``).
      ``input``       ``auto`` (default): per row, raw counts are used as they are and
                      normalized rows are converted back to counts (``recover_counts``:
                      ln(x+1), log2(x+1) or linear, rescaled by the smallest non-zero
                      value), with a warning and ``input_conversion`` in the provenance.
                      Or one scale for every row: ``counts`` (non-count rows skipped),
                      ``cpm``, ``log1p_cpm``, ``tpm``, ``log_tpm`` (natural ln(TPM+1)) or
                      ``log2_tpm``.
      ``gene_column`` var column holding the gene IDs (default: ``var_names``).
      ``match_on``    ``auto`` (Ensembl when most IDs start with ENSG), ``ensembl`` or
                      ``symbol`` (mapped through the vocabulary's ``gene_symbol``;
                      ambiguous symbols are dropped).
      ``batch_size``  samples per forward pass (8). Every sample is 19,973 tokens, so
                      lower it if the GPU runs out of memory; BulkFormer starts at 1.
      ``chunk_rows``  rows densified and normalized at a time (1024).
      ``device``      default ``cuda`` when available, else ``cpu``.
      ``amp_dtype``   ``auto`` (bf16, else fp16 on GPU; fp32 on CPU), ``bf16``, ``fp16``
                      or ``fp32``.

    Runs in BulkFormer's environment (``configs/bulkformer-latest.yaml``) plus
    ``anndata`` and ``pyyaml``: it needs torch_geometric and performer_pytorch, and
    never imports cancerfoundation.
    """

    def __init__(self, repo: str | None = None, model_dir: str | None = None, input: str = "auto",
                 gene_column: str | None = None, match_on: str = "auto", batch_size: int = 8,
                 chunk_rows: int = 1024, device: str | None = None, amp_dtype: str = "auto",
                 **options: Any) -> None:
        import os

        repo = repo or os.environ.get(BULKFORMER_REPO_ENV)
        if not repo:
            raise ValueError(f"bulkformer needs repo=<bulkformer-latest checkout> (or ${BULKFORMER_REPO_ENV})")
        model_dir = model_dir or str(Path(repo) / "model" / "bulkformer-latest")
        if input not in BULKFORMER_INPUTS:
            raise ValueError(f"input must be one of {BULKFORMER_INPUTS}, got {input!r}")
        if match_on not in ("auto", "ensembl", "symbol"):
            raise ValueError(f"match_on must be auto, ensembl or symbol, got {match_on!r}")
        if amp_dtype not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError(f"amp_dtype must be auto, bf16, fp16 or fp32, got {amp_dtype!r}")
        super().__init__(repo=repo, model_dir=model_dir, input=input, gene_column=gene_column,
                         match_on=match_on, batch_size=batch_size, chunk_rows=chunk_rows,
                         device=device, amp_dtype=amp_dtype, **options)
        self.repo, self.model_dir = Path(repo), Path(model_dir)
        self.input, self.gene_column, self.match_on = input, gene_column, match_on
        self.batch_size, self.chunk_rows = int(batch_size), int(chunk_rows)
        self.device, self.amp_dtype = device, amp_dtype
        self._api = self._model = self._vocab = self._manifest = None
        self.dropped_rows: list[str] = []  # rows the last embed() skipped (not usable)
        self.input_conversion: dict[str, Any] | None = None  # per-row scales of the last call

    # -- loading ---------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._api = _load_bulkformer_api(self.repo)
        device = self.device
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading bulkformer-latest from {self.model_dir} on {device}")
        # load_latest_model verifies every release file against its manifest checksums.
        self._model, self._vocab, self._manifest = self._api.load_latest_model(self.model_dir, device=device)
        if self.amp_dtype == "auto":
            if device.startswith("cuda"):
                import torch

                self.amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            else:
                self.amp_dtype = "fp32"
        print(f"  {len(self._vocab)} genes, dim {self._model.config.dim}, amp {self.amp_dtype}")

    # -- input handling --------------------------------------------------------

    def _match_genes(self, adata) -> tuple[np.ndarray, list[str]]:
        """``(var positions, Ensembl IDs)`` of the input genes found in the vocabulary."""
        if self.gene_column is not None:
            if self.gene_column not in adata.var.columns:
                raise KeyError(f"gene_column {self.gene_column!r} not in var: {list(adata.var.columns)}")
            names = adata.var[self.gene_column].astype(str).to_numpy()
        else:
            names = adata.var_names.astype(str).to_numpy()
        match_on = self.match_on
        if match_on == "auto":
            match_on = "ensembl" if np.mean([n.startswith("ENSG") for n in names]) > 0.5 else "symbol"

        vocab_ids = self._vocab["canonical_ensg"].astype(str)
        if match_on == "ensembl":
            ids = np.array([n.split(".", 1)[0] for n in names], dtype=object)
        else:
            symbols = self._vocab["gene_symbol"].astype(str)
            counts = symbols.value_counts()
            unique = symbols[symbols.map(counts) == 1]
            to_ensg = dict(zip(unique, vocab_ids[unique.index]))
            ids = np.array([to_ensg.get(n) for n in names], dtype=object)
            n_ambiguous = int(sum(n in counts.index and counts[n] > 1 for n in set(names)))
            print("  [bulkformer] matching gene symbols through the vocabulary; Ensembl IDs are "
                  "the reference keys, so prefer gene_column=<Ensembl column> when there is one"
                  + (f" ({n_ambiguous} ambiguous symbols dropped)" if n_ambiguous else ""))

        in_vocab = set(vocab_ids)
        positions, seen, duplicates = [], set(), 0
        for pos, gene in enumerate(ids):
            if gene is None or gene not in in_vocab:
                continue
            if gene in seen:
                duplicates += 1
                continue
            seen.add(gene)
            positions.append(pos)
        if not positions:
            raise ValueError(f"No input gene matches the bulkformer vocabulary (matching on {match_on}); "
                             f"first names: {list(names[:5])}")
        missing = 1 - len(positions) / len(vocab_ids)
        print(f"  [bulkformer] {len(positions)} of {adata.n_vars} input genes in the vocabulary; "
              f"{missing:.1%} of the {len(vocab_ids)} vocabulary genes are missing (fed as -10)"
              + (f"; {duplicates} duplicate IDs kept once" if duplicates else ""))
        if missing > 0.10:
            print("  [bulkformer] [WARNING] BulkFormer recommends at most 10% missing genes per sample")
        positions = np.asarray(positions)
        return positions, [ids[p] for p in positions]

    def _resolve_input(self, X=None) -> str:
        """The input scale: ``auto`` (decided per row by ``_prepare``) or the given one."""
        return self.input

    def _prepare(self, X, input_type: str) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        """``(values, scale, ok, labels)`` of a dense chunk of vocabulary-matched genes.

        ``auto``: ``recover_counts`` per row, so every usable row goes in as counts and
        ``labels`` says what was converted. ``counts``: rows that are not count-like are
        not usable. Any other scale applies to every row as given.
        """
        if input_type == "auto":
            values, labels = recover_counts(X)
            return values, "counts", labels != "invalid", labels
        ok = _count_like_rows(X) if input_type == "counts" else np.ones(len(X), dtype=bool)
        labels = np.where(ok, input_type, "invalid").astype(object)
        return np.asarray(X), input_type, ok, labels

    def _to_log_tpm(self, frame, input_type: str):
        """Natural ln(TPM + 1), as BulkFormer's usage guide prescribes per input scale."""
        if input_type in {"counts", "cpm"}:
            return self._api.normalize_data(X_df=frame, gene_vocab=self._vocab)
        if input_type == "log1p_cpm":
            return self._api.normalize_data(X_df=np.expm1(frame), gene_vocab=self._vocab)
        if input_type == "tpm":
            return np.log1p(frame)
        if input_type == "log2_tpm":
            return frame * np.log(2.0)
        return frame

    # -- Embedder interface ----------------------------------------------------

    def embed(self, adata) -> np.ndarray:
        import pandas as pd
        import scipy.sparse as sp

        self._ensure_loaded()
        positions, ensg = self._match_genes(adata)
        self.genes = ensg
        vocab_genes = self._vocab["canonical_ensg"].astype(str).tolist()
        input_type = self._resolve_input(adata.X[:512][:, positions])
        print(f"  [bulkformer] input scale: {'per row (auto)' if input_type == 'auto' else input_type}; "
              f"embedding {adata.n_obs} samples (batch {self.batch_size})")

        # A row BulkFormer's normalize_data would reject (negative, non-finite, all-zero
        # over the vocabulary genes; with input=counts also non-integral) is skipped and
        # comes back as a NaN row; write_embeddings leaves NaN rows out of the file.
        dim = int(self._model.config.dim)
        dropped = np.zeros(adata.n_obs, dtype=bool)
        labels = np.empty(adata.n_obs, dtype=object)
        blocks = []
        for start in range(0, adata.n_obs, self.chunk_rows):
            stop = min(start + self.chunk_rows, adata.n_obs)
            X = adata.X[start:stop][:, positions]
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            X, scale, ok, labels[start:stop] = self._prepare(X, input_type)
            dropped[start:stop] = ~ok
            out = np.full((len(X), dim), np.nan, dtype=np.float32)
            if ok.any():
                frame = pd.DataFrame(X[ok].astype(np.float32), index=adata.obs_names[start:stop][ok], columns=ensg)
                log_tpm = self._to_log_tpm(frame, scale)
                # Full release gene axis in vocabulary order; absent genes and NaN become -10.
                aligned, _, _ = self._api.main_gene_selection(X_df=log_tpm, gene_list=vocab_genes)
                out[ok] = self._api.extract_feature(
                    model=self._model,
                    expr_array=aligned.to_numpy(),
                    output_feature_type="sample_level",
                    aggregate_type="mean",
                    batch_size=self.batch_size,
                    amp_dtype=self.amp_dtype,
                )
            blocks.append(out)
            if adata.n_obs > self.chunk_rows:
                print(f"  [bulkformer] rows {start}-{stop} of {adata.n_obs}")
        self.input_conversion = conversion_summary(labels, adata.obs_names)
        report_conversion(self.input_conversion)
        if dropped.any():
            names = list(adata.obs_names[dropped][:5])
            print(f"  [bulkformer] [WARNING] {int(dropped.sum())} of {adata.n_obs} rows are not usable "
                  f"(negative, non-finite or all-zero over the vocabulary genes"
                  f"{', or not raw counts with input=counts' if input_type == 'counts' else ''}) and "
                  f"were not embedded, e.g. {names}")
        self.dropped_rows = list(map(str, adata.obs_names[dropped]))
        return np.concatenate(blocks, axis=0)

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        if self._manifest is not None:
            out["checkpoint_metadata"] = self._manifest.get("checkpoint_metadata")
            out["options"]["amp_dtype"] = self.amp_dtype
        if self.input_conversion is not None:
            out["input_conversion"] = self.input_conversion
        return out


# --------------------------------------------------------------------------- #
# Integration references: scVI and Harmony
# --------------------------------------------------------------------------- #
#
# Methods built to remove a batch effect, given the batch label (``batch_key``, default
# ``modality``). They are reference rows for the scIB tables: CancerFoundation never sees
# the modality label when it embeds, so they are not competitors on equal terms.
# Neither imports cancerfoundation (its package init loads torch, lightning and SCDL),
# so both run in a light env: scvi-tools, scanpy, harmonypy, anndata, pyyaml (plus
# scikit-misc for seurat_v3 HVGs).


def counts_matrix(adata, genes=None, chunk_rows: int = 2048):
    """``(counts, labels, library)`` of ``adata`` as raw counts, ``chunk_rows`` at a time.

    Every row goes through ``recover_counts`` over ALL its genes (part of the
    single-cell source is normalized), then ``genes`` (var names, ``None`` = all) are
    kept, as a float32 CSR matrix. ``library`` is each row's count total over all
    genes; ``labels`` says per row what was converted (``invalid`` = not usable).
    """
    import scipy.sparse as sp

    cols = None if genes is None else adata.var_names.get_indexer(list(genes))
    if cols is not None and (cols < 0).any():
        missing = [g for g, c in zip(genes, cols) if c < 0]
        raise ValueError(f"{len(missing)} fitted genes are missing from the input, e.g. {missing[:5]}")
    blocks, labels, library = [], [], []
    for start in range(0, adata.n_obs, chunk_rows):
        counts, lab = recover_counts(adata.X[start:start + chunk_rows])
        library.append(np.where(lab == "invalid", 0.0, counts.sum(axis=1)))
        counts[lab == "invalid"] = 0.0
        blocks.append(sp.csr_matrix((counts if cols is None else counts[:, cols]).astype(np.float32)))
        labels.append(lab)
    return sp.vstack(blocks).tocsr(), np.concatenate(labels), np.concatenate(library)


def select_hvgs(counts, var_names, batches, n_top: int, verbose_name: str) -> tuple[list[str], str]:
    """``(genes in var order, flavor)``: scanpy HVGs on raw counts, ranked across batches.

    ``seurat_v3`` (the scVI recipe; needs scikit-misc) on the counts, else ``seurat`` on
    log1p CP10K. With more than one batch, ``batch_key`` ranks genes within each batch
    first, so a large modality does not pick the genes alone.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    a = ad.AnnData(X=counts.copy(), obs=pd.DataFrame({"batch": np.asarray(batches).astype(str)}),
                   var=pd.DataFrame(index=list(map(str, var_names))))
    batch_key = "batch" if a.obs["batch"].nunique() > 1 else None
    n_top = min(int(n_top), a.n_vars)
    try:
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top, batch_key=batch_key)
        flavor = "seurat_v3"
    except ImportError:
        print(f"  [{verbose_name}] [WARNING] seurat_v3 needs scikit-misc; using seurat on log1p CP10K")
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_top, batch_key=batch_key)
        flavor = "seurat"
    return [g for g, keep in zip(a.var_names, a.var["highly_variable"].to_numpy()) if keep], flavor


def _batch_labels(adata, batch_key: str, who: str) -> np.ndarray:
    if batch_key not in adata.obs.columns:
        raise KeyError(f"{who}: batch_key {batch_key!r} not in obs: {list(adata.obs.columns)}")
    return adata.obs[batch_key].astype(str).to_numpy()


def _report_invalid(labels, names, who: str) -> None:
    bad = labels == "invalid"
    if bad.any():
        print(f"  [{who}] [WARNING] {int(bad.sum())} of {len(labels)} rows are not usable (negative, "
              f"non-finite or zero total) and get no embedding, e.g. {list(np.asarray(names)[bad][:5])}")


@register("scvi")
class SCVIEmbedder(Embedder):
    """scVI (scvi-tools) latent, trained with the modality as the batch covariate.

    ``fit`` (the train-subset rows): every row back to raw counts (``recover_counts``),
    ``n_hvg`` HVGs (``select_hvgs``), then ``scvi.model.SCVI`` with ``batch_key`` as the
    batch. ``embed``: the latent mean of each row (``get_latent_representation``); its
    batch must be one seen in ``fit``. Rows that cannot be read as counts are NaN rows,
    which ``write_embeddings`` leaves out. Counts recovered only approximately (``~``)
    are rounded for the NB likelihood.

    Options: ``batch_key`` (``modality``), ``n_hvg`` (4000), ``n_latent`` (30),
    ``n_hidden`` (128), ``n_layers`` (2), ``gene_likelihood`` (``nb``), ``max_epochs``
    (scvi-tools' default for the data size when unset), ``train_batch_size`` (512),
    ``early_stopping`` (true), ``accelerator`` (``auto``), ``seed`` (0), ``chunk_rows``.
    """

    def __init__(self, batch_key: str = "modality", n_hvg: int = 4000, n_latent: int = 30,
                 n_hidden: int = 128, n_layers: int = 2, gene_likelihood: str = "nb",
                 max_epochs: int | None = None, train_batch_size: int = 512, early_stopping: bool = True,
                 accelerator: str = "auto", seed: int = 0, chunk_rows: int = 2048, **options: Any) -> None:
        if gene_likelihood not in ("nb", "zinb", "poisson"):
            raise ValueError(f"gene_likelihood must be nb, zinb or poisson, got {gene_likelihood!r}")
        super().__init__(batch_key=batch_key, n_hvg=n_hvg, n_latent=n_latent, n_hidden=n_hidden,
                         n_layers=n_layers, gene_likelihood=gene_likelihood, max_epochs=max_epochs,
                         train_batch_size=train_batch_size, early_stopping=early_stopping,
                         accelerator=accelerator, seed=seed, chunk_rows=chunk_rows, **options)
        self.batch_key, self.n_hvg, self.seed = batch_key, int(n_hvg), int(seed)
        self.chunk_rows = int(chunk_rows)
        self._model = None
        self._batches: list[str] = []
        self._fit_info: dict[str, Any] = {}
        self.input_conversion: dict[str, Any] | None = None

    def _anndata(self, counts, batches, names):
        import anndata as ad
        import pandas as pd

        counts = counts.copy()
        counts.data = np.rint(counts.data)
        return ad.AnnData(X=counts, obs=pd.DataFrame({self.batch_key: pd.Categorical(batches)},
                                                     index=list(map(str, names))),
                          var=pd.DataFrame(index=self.genes))

    def fit(self, adata) -> None:
        import scvi

        batches = _batch_labels(adata, self.batch_key, "scvi")
        counts, labels, _ = counts_matrix(adata, chunk_rows=self.chunk_rows)
        _report_invalid(labels, adata.obs_names, "scvi")
        ok = labels != "invalid"
        self.genes, flavor = select_hvgs(counts[ok], adata.var_names, batches[ok], self.n_hvg, "scvi")
        cols = adata.var_names.get_indexer(self.genes)
        train = self._anndata(counts[ok][:, cols], batches[ok], adata.obs_names[ok])
        train = train[np.asarray(train.X.sum(axis=1)).ravel() > 0].copy()  # no zero library in training
        self._batches = sorted(set(train.obs[self.batch_key]))

        scvi.settings.seed = self.seed
        scvi.model.SCVI.setup_anndata(train, batch_key=self.batch_key)
        o = self.options
        self._model = scvi.model.SCVI(train, n_latent=int(o["n_latent"]), n_hidden=int(o["n_hidden"]),
                                      n_layers=int(o["n_layers"]), gene_likelihood=o["gene_likelihood"])
        print(f"scVI: {train.n_obs} rows x {len(self.genes)} HVGs ({flavor}), batch '{self.batch_key}' "
              f"{ {str(k): int(v) for k, v in train.obs[self.batch_key].value_counts().items()} }")
        self._model.train(max_epochs=o["max_epochs"], batch_size=int(o["train_batch_size"]),
                          early_stopping=bool(o["early_stopping"]), accelerator=o["accelerator"])
        history = getattr(self._model, "history", None) or {}
        elbo = history.get("elbo_train")
        self._fit_info = {"fit_rows": int(train.n_obs), "hvg_flavor": flavor, "batches": self._batches,
                          "epochs": int(len(elbo)) if elbo is not None else None,
                          "fit_input_conversion": conversion_summary(labels, adata.obs_names)}

    def embed(self, adata) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SCVIEmbedder.embed called before fit")
        batches = _batch_labels(adata, self.batch_key, "scvi")
        unseen = sorted(set(batches) - set(self._batches))
        if unseen:
            raise ValueError(f"scvi: batches {unseen} were not in the fit rows {self._batches}; "
                             "fit on rows of every modality you embed (parts: [test, train_subset])")
        counts, labels, _ = counts_matrix(adata, self.genes, chunk_rows=self.chunk_rows)
        self.input_conversion = conversion_summary(labels, adata.obs_names)
        report_conversion(self.input_conversion, "scvi")
        _report_invalid(labels, adata.obs_names, "scvi")
        ok = labels != "invalid"
        out = np.full((adata.n_obs, int(self.options["n_latent"])), np.nan, dtype=np.float32)
        if ok.any():
            query = self._anndata(counts[ok], batches[ok], adata.obs_names[ok])
            out[ok] = np.asarray(self._model.get_latent_representation(
                adata=query, batch_size=int(self.options["train_batch_size"])), dtype=np.float32)
        return out

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        out.update({"uses_batch_labels": self.batch_key, **self._fit_info})
        if self.input_conversion is not None:
            out["input_conversion"] = self.input_conversion
        return out


def harmony_correct(Z: np.ndarray, batches, seed: int = 0, max_iter: int = 10) -> np.ndarray:
    """``harmonypy.run_harmony`` on rows ``Z`` with one batch label per row, rows out."""
    import harmonypy
    import pandas as pd

    meta = pd.DataFrame({"batch": np.asarray(batches).astype(str)})
    ho = harmonypy.run_harmony(Z, meta, ["batch"], random_state=seed, max_iter_harmony=max_iter)
    out = ho.Z_corr
    if hasattr(out, "detach"):  # torch versions of harmonypy
        out = out.detach().cpu().numpy()
    out = np.asarray(out)
    if out.shape == Z.shape:  # rows already (newer versions); a square input stays as is
        return out.astype(np.float32)
    if out.shape == Z.shape[::-1]:  # (dims, rows), harmonypy 0.0.x
        return out.T.astype(np.float32)
    raise ValueError(f"harmonypy returned shape {out.shape} for an input of {Z.shape}")


@register("harmony")
class HarmonyEmbedder(Embedder):
    """PCA corrected by Harmony (harmonypy) on the batch label.

    ``fit`` (the train-subset rows): rows back to raw counts (``recover_counts``),
    ``n_hvg`` HVGs (``select_hvgs``), CPM + log1p (library over all genes) and a PCA.
    ``embed``: the PCA of the rows, then Harmony over ALL the rows of the call with
    ``batch_key``. Harmony is transductive: the correction depends on every row it
    sees, so ``embed_splits`` corrects the test and train-subset rows of all modalities
    together in its one call.

    Options: ``batch_key`` (``modality``), ``n_hvg`` (2000), ``n_pcs`` (50),
    ``max_iter`` (10), ``seed`` (0), ``chunk_rows``.
    """

    def __init__(self, batch_key: str = "modality", n_hvg: int = 2000, n_pcs: int = 50,
                 max_iter: int = 10, seed: int = 0, chunk_rows: int = 2048, **options: Any) -> None:
        super().__init__(batch_key=batch_key, n_hvg=n_hvg, n_pcs=n_pcs, max_iter=max_iter, seed=seed,
                         chunk_rows=chunk_rows, **options)
        self.batch_key, self.n_hvg, self.n_pcs = batch_key, int(n_hvg), int(n_pcs)
        self.max_iter, self.seed, self.chunk_rows = int(max_iter), int(seed), int(chunk_rows)
        self._pca = None
        self._fit_info: dict[str, Any] = {}
        self.input_conversion: dict[str, Any] | None = None

    @staticmethod
    def _log_cpm(counts, library) -> np.ndarray:
        X = counts.toarray().astype(np.float64)
        scale = np.divide(1e6, library, out=np.zeros_like(library, dtype=np.float64), where=library > 0)
        return np.log1p(X * scale[:, None]).astype(np.float32)

    def fit(self, adata) -> None:
        from sklearn.decomposition import PCA

        batches = _batch_labels(adata, self.batch_key, "harmony")
        counts, labels, library = counts_matrix(adata, chunk_rows=self.chunk_rows)
        _report_invalid(labels, adata.obs_names, "harmony")
        ok = labels != "invalid"
        self.genes, flavor = select_hvgs(counts[ok], adata.var_names, batches[ok], self.n_hvg, "harmony")
        cols = adata.var_names.get_indexer(self.genes)
        X = self._log_cpm(counts[ok][:, cols], library[ok])
        n_comps = min(self.n_pcs, len(self.genes), int(ok.sum()) - 1)
        self._pca = PCA(n_components=n_comps, random_state=self.seed).fit(X)
        print(f"Harmony: PCA of {n_comps} components on {len(self.genes)} HVGs ({flavor}), fitted on "
              f"{int(ok.sum())} rows ({self._pca.explained_variance_ratio_.sum():.1%} variance)")
        self._fit_info = {"fit_rows": int(ok.sum()), "hvg_flavor": flavor, "n_pcs": int(n_comps),
                          "fit_input_conversion": conversion_summary(labels, adata.obs_names)}

    def embed(self, adata) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("HarmonyEmbedder.embed called before fit")
        batches = _batch_labels(adata, self.batch_key, "harmony")
        counts, labels, library = counts_matrix(adata, self.genes, chunk_rows=self.chunk_rows)
        self.input_conversion = conversion_summary(labels, adata.obs_names)
        report_conversion(self.input_conversion, "harmony")
        _report_invalid(labels, adata.obs_names, "harmony")
        ok = labels != "invalid"
        out = np.full((adata.n_obs, self._pca.n_components_), np.nan, dtype=np.float32)
        if not ok.any():
            return out
        Z = self._pca.transform(self._log_cpm(counts[ok], library[ok])).astype(np.float64)
        print(f"  [harmony] correcting {int(ok.sum())} rows on '{self.batch_key}' "
              f"{ {str(k): int(v) for k, v in zip(*np.unique(batches[ok], return_counts=True))} }")
        out[ok] = harmony_correct(Z, batches[ok], seed=self.seed, max_iter=self.max_iter)
        return out

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        out.update({"uses_batch_labels": self.batch_key, "transductive": True, **self._fit_info})
        if self.input_conversion is not None:
            out["input_conversion"] = self.input_conversion
        return out


# --------------------------------------------------------------------------- #
# Running an embedder
# --------------------------------------------------------------------------- #


def build_embedder(name: str, options: dict[str, Any] | None = None) -> Embedder:
    if name not in EMBEDDERS:
        raise KeyError(f"Unknown embedder {name!r}; registered: {sorted(EMBEDDERS)}")
    return EMBEDDERS[name](**(options or {}))


def parse_options(specs: list[str] | None) -> dict[str, Any]:
    """``["n_pcs=50", "normalized=true"]`` -> ``{"n_pcs": 50, "normalized": True}``."""
    import yaml

    options: dict[str, Any] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--option {spec!r} must look like KEY=VALUE")
        key, value = spec.split("=", 1)
        options[key.strip().replace("-", "_")] = yaml.safe_load(value)
    return options


def run_embedder(embedder: Embedder, adata, fit_adata=None) -> np.ndarray:
    """Fit on ``fit_adata`` (``adata`` itself when None), then embed ``adata``."""
    embedder.fit(adata if fit_adata is None else fit_adata)
    emb = np.asarray(embedder.embed(adata), dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != adata.n_obs:
        raise ValueError(f"{embedder.name}.embed returned shape {emb.shape} for {adata.n_obs} rows")
    return emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed an h5ad with any registered model into a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List the registered embedders and exit.")
    parser.add_argument("--embedder", default=None, help="Registered embedder name.")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Keyword argument for the embedder (repeatable).")
    parser.add_argument("--input", default=None, help="Input .h5ad file.")
    parser.add_argument("--fit-input", default=None,
                        help="h5ad the embedder is fitted on (default: the input itself).")
    parser.add_argument("--out-dir", default=None, help="Directory the embeddings are written to.")
    parser.add_argument("--name", default=None, help="Output file stem (default: the input's stem).")
    parser.add_argument("--obsm-key", default=EMB_KEY, help="obsm key of the saved embedding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name, cls in sorted(EMBEDDERS.items()):
            doc = (cls.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:12s} {doc}")
        return
    if not (args.embedder and args.input and args.out_dir):
        sys.exit("Error: --embedder, --input and --out-dir are required (or --list)")

    import anndata as ad  # not scanpy: external models' environments may lack it

    embedder = build_embedder(args.embedder, parse_options(args.option))
    input_path = Path(args.input)
    print(f"Loading data: {input_path}")
    adata = ad.read_h5ad(str(input_path))
    fit_adata = ad.read_h5ad(args.fit_input) if args.fit_input else None
    emb = run_embedder(embedder, adata, fit_adata)
    write_embeddings(
        Path(args.out_dir), args.name or input_path.stem, adata.obs, emb, embedder.genes,
        provenance={**embedder.provenance(), "input": str(input_path),
                    "fit_input": args.fit_input or str(input_path)},
        obsm_key=args.obsm_key,
    )
    print("Done.")


if __name__ == "__main__":
    main()
