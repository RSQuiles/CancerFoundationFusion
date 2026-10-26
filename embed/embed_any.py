"""Embed an h5ad with ANY model (baselines, other foundation models) into a directory.

Every model is an ``Embedder`` registered under a name. The output is the same as
``embed.py``'s, so evaluation treats a baseline exactly like a CancerFoundation
model: ``<out-dir>/<name>.h5ad`` (obs + ``obsm["X_emb"]``, no X) plus
``<out-dir>/embed.json``.

    python embed_any.py --list
    python embed_any.py --embedder pca --input data.h5ad --out-dir out/ --option n_pcs=50
    python embed_any.py --embedder bulkformer --input bulk.h5ad --out-dir out/ \\
        --option checkpoint=/path/to/bulkformer.pt

Adding a model
--------------
Subclass ``Embedder``, decorate it with ``@register("<name>")`` and implement
``embed`` (and ``fit`` if the model learns anything from the data, as PCA does).
``--option key=value`` pairs are passed to ``__init__`` as keyword arguments, with
values parsed as YAML scalars (``50`` -> int, ``true`` -> bool). ``BulkFormerEmbedder``
below is the stub to fill in; ``PCAEmbedder`` is a complete example.

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

from embed.embed import EMB_KEY, write_embeddings  # noqa: E402

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
    """CP10K + log1p, seurat HVGs and PCA, all fitted in ``fit``.

    Options: ``n_pcs`` (50), ``n_hvg`` (2000), ``normalized`` / ``log1p_only``
    (False; same meaning as in embed.py), ``seed`` (0).
    """

    def __init__(self, n_pcs: int = 50, n_hvg: int = 2000, normalized: bool = False,
                 log1p_only: bool = False, seed: int = 0, **options: Any) -> None:
        super().__init__(n_pcs=n_pcs, n_hvg=n_hvg, normalized=normalized,
                         log1p_only=log1p_only, seed=seed, **options)
        self.n_pcs, self.n_hvg, self.seed = int(n_pcs), int(n_hvg), int(seed)
        self.normalized, self.log1p_only = bool(normalized), bool(log1p_only)
        self._pca = None

    def _prepare(self, adata):
        import scanpy as sc

        work = adata.copy()
        if not self.normalized:
            if not self.log1p_only:
                sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
        return work

    def _dense(self, adata) -> np.ndarray:
        import scipy.sparse as sp

        X = adata[:, self.genes].X
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    def fit(self, adata) -> None:
        import scanpy as sc
        from sklearn.decomposition import PCA

        work = self._prepare(adata)
        sc.pp.highly_variable_genes(work, n_top_genes=min(self.n_hvg, work.n_vars), flavor="seurat")
        self.genes = work.var_names[work.var["highly_variable"].to_numpy()].tolist()
        n_comps = min(self.n_pcs, len(self.genes), work.n_obs - 1)
        self._pca = PCA(n_components=n_comps, random_state=self.seed).fit(self._dense(work))
        print(f"PCA: {n_comps} components on {len(self.genes)} HVGs, fitted on {work.n_obs} rows "
              f"({self._pca.explained_variance_ratio_.sum():.1%} variance)")

    def embed(self, adata) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAEmbedder.embed called before fit")
        missing = sorted(set(self.genes) - set(adata.var_names))
        if missing:
            raise ValueError(f"{len(missing)} fitted HVGs are missing from the input, e.g. {missing[:5]}")
        return self._pca.transform(self._dense(self._prepare(adata))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stubs: yet to be implemented
# --------------------------------------------------------------------------- #


@register("bulkformer")
class BulkFormerEmbedder(Embedder):
    """BulkFormer (bulk RNA-seq foundation model). TO BE IMPLEMENTED.

    Expected options: ``checkpoint`` (path to the weights), ``batch_size``,
    ``device``. Things ``embed`` has to settle: the gene identifiers and gene order
    the model expects (map ``adata.var_names`` onto them, filling absent genes the
    way the model was trained), the input scale (raw counts, TPM, log), and which
    output is the sample embedding (pooled token states or a CLS token). Set
    ``self.genes`` to the genes actually fed in so ``embed.json`` records the panel.
    """

    def __init__(self, checkpoint: str | None = None, batch_size: int = 64,
                 device: str = "cuda", **options: Any) -> None:
        super().__init__(checkpoint=checkpoint, batch_size=batch_size, device=device, **options)
        self.checkpoint, self.batch_size, self.device = checkpoint, int(batch_size), device

    def embed(self, adata) -> np.ndarray:
        raise NotImplementedError(
            "BulkFormerEmbedder.embed is not implemented yet: load the model from "
            f"{self.checkpoint!r}, align adata.var_names to its gene order and return "
            "an (n_obs, dim) array"
        )


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

    import scanpy as sc

    embedder = build_embedder(args.embedder, parse_options(args.option))
    input_path = Path(args.input)
    print(f"Loading data: {input_path}")
    adata = sc.read_h5ad(str(input_path))
    fit_adata = sc.read_h5ad(args.fit_input) if args.fit_input else None
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
