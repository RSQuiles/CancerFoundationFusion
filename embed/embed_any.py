"""Embed an h5ad with ANY model (baselines, other foundation models) into a directory.

Every model is an ``Embedder`` registered under a name. The output is the same as
``embed.py``'s, so evaluation treats a baseline exactly like a CancerFoundation
model: ``<out-dir>/<name>.h5ad`` (obs + ``obsm["X_emb"]``, no X) plus
``<out-dir>/embed.json``.

    python embed_any.py --list
    python embed_any.py --embedder pca --input data.h5ad --out-dir out/ --option n_pcs=50
    python embed_any.py --embedder bulkformer --input bulk.h5ad --out-dir out/ \\
        --option repo=/path/to/bulkformer-latest --option gene_column=gene_id

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
BULKFORMER_INPUTS = ("auto", "counts", "tpm", "log_tpm", "log2_tpm")


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


def _looks_like_counts(X, max_rows: int = 512) -> bool:
    """Same rule as ``cancerfoundation.data.preprocess.looks_like_counts`` (max > 20 and
    integral), restated here so this embedder runs in BulkFormer's own environment,
    which cannot import the cancerfoundation package."""
    import scipy.sparse as sp

    X = X[: min(max_rows, X.shape[0])]
    vals = np.asarray(X.data if sp.issparse(X) else X).ravel()
    vals = vals[np.isfinite(vals) & (vals != 0)]
    return bool(vals.size) and float(vals.max()) > 20 and bool(np.allclose(vals, np.round(vals)))


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
      ``input``       ``auto`` (default: raw counts if they look like counts, otherwise an
                      error asking for the scale), ``counts``, ``tpm``, ``log_tpm``
                      (natural ln(TPM+1)) or ``log2_tpm``.
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
        self.dropped_rows: list[str] = []  # rows the last embed() skipped (not count-like)

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

    def _resolve_input(self, X) -> str:
        """The input scale; ``auto`` means counts when most sampled rows look like counts.

        Rows that do not look like counts are then dropped by ``embed``, so a few stray
        normalized rows no longer stop the run; a matrix that is mostly not counts still
        does, because then the scale is genuinely unknown.
        """
        if self.input != "auto":
            return self.input
        ok = _count_like_rows(X)
        if ok.mean() >= 0.5 and _looks_like_counts(X[np.flatnonzero(ok)]):
            return "counts"
        raise ValueError(
            f"bulkformer input=auto only recognizes raw counts, and only {int(ok.sum())} of {len(ok)} "
            "sampled rows look like counts. Set input=tpm, log_tpm (natural ln(TPM+1)) or log2_tpm."
        )

    def _to_log_tpm(self, frame, input_type: str):
        """Natural ln(TPM + 1), as BulkFormer's usage guide prescribes per input scale."""
        if input_type == "counts":
            return self._api.normalize_data(X_df=frame, gene_vocab=self._vocab)
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
        print(f"  [bulkformer] input treated as {input_type}; embedding {adata.n_obs} samples "
              f"(batch {self.batch_size})")

        # With count input, a row BulkFormer's normalize_data would reject (negative,
        # non-integral, non-finite or all-zero over the vocabulary genes) is skipped and
        # comes back as a NaN row; write_embeddings leaves NaN rows out of the file.
        dim = int(self._model.config.dim)
        dropped = np.zeros(adata.n_obs, dtype=bool)
        blocks = []
        for start in range(0, adata.n_obs, self.chunk_rows):
            stop = min(start + self.chunk_rows, adata.n_obs)
            X = adata.X[start:stop][:, positions]
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            ok = _count_like_rows(X) if input_type == "counts" else np.ones(len(X), dtype=bool)
            dropped[start:stop] = ~ok
            out = np.full((len(X), dim), np.nan, dtype=np.float32)
            if ok.any():
                frame = pd.DataFrame(X[ok].astype(np.float32), index=adata.obs_names[start:stop][ok], columns=ensg)
                log_tpm = self._to_log_tpm(frame, input_type)
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
        if dropped.any():
            names = list(adata.obs_names[dropped][:5])
            print(f"  [bulkformer] [WARNING] {int(dropped.sum())} of {adata.n_obs} rows do not look like raw "
                  f"counts (negative, non-integral, non-finite or all-zero over the vocabulary genes) and "
                  f"were not embedded, e.g. {names}")
        self.dropped_rows = list(map(str, adata.obs_names[dropped]))
        return np.concatenate(blocks, axis=0)

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        if self._manifest is not None:
            out["checkpoint_metadata"] = self._manifest.get("checkpoint_metadata")
            out["options"]["amp_dtype"] = self.amp_dtype
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
