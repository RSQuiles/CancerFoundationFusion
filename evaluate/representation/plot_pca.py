"""PCA of one modality's embeddings, per model.

The PCA counterpart of ``plot_umap.py``, with the same inputs, filtering, subsampling,
colouring, titles and labels. The PCA is fitted on the rows that are plotted (after
``--filter`` and the subsample), and two of its components are drawn, PC1 against PC2
unless ``--components`` says otherwise. Writes, per model::

    results/representation/<model>/pca_<modality>_<part>[_<filter tag>][_pc<a>-<b>].png
    results/representation/<model>/pca_<modality>_<part>[...].parquet   coords + obs
    results/representation/<model>/pca_<modality>_<part>[...].json      variance, settings

    python -m evaluate.representation.plot_pca --models m1 m2 --modality bulk --part test \\
        --filter tissue_general=lung,breast --subsample 20000 --stratify-by tissue_general \\
        --color tissue_general assay --skip-unknown \\
        --title "{model}: {modality_name} ({part}, {pcs})" --labels tissue_general=Tissue

``--already-pca`` says the embedding already holds principal components (the ``pca``
baseline model, or an h5ad's ``obsm['X_pca']``), so nothing is fitted and the chosen
columns are plotted as stored. A bare ``--already-pca`` applies to every input;
``--already-pca pca`` only to the named models. For a precomputed ``X_pca`` the axis
labels show the explained variance when the file has ``uns['pca']['variance_ratio']``.

Any h5ad outside the layout works too (``--input file.h5ad --obsm-key X_pca``);
the figure then goes next to it unless ``--out-dir`` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import format_title, parse_mapping, parse_with_config  # noqa: E402
from experiments.layout import ExperimentLayout, canonical_modality  # noqa: E402
from utils.sampling import filter_tag, parse_filter_specs  # noqa: E402
from evaluate.io import EMB_KEY, load_embedding, load_external  # noqa: E402
from evaluate.modalities import DISPLAY_NAMES  # noqa: E402
from evaluate.representation.plot_umap import prepare  # noqa: E402
from evaluate.representation.render import draw_umap_panels, save_coords, save_figure  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-model PCA plots of one modality's embeddings.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", default=None, help="Model names (embeddings/<model>/)")
    parser.add_argument("--modality", default="sc", help="sc | bulk | pseudobulk")
    parser.add_argument("--part", default="test", choices=["test", "train_subset"])
    parser.add_argument("--input", type=Path, default=None, help="Any h5ad instead of --models")
    parser.add_argument("--obsm-key", default=EMB_KEY, help="obsm key holding the embedding (with --input)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results/representation/<model>/)")
    parser.add_argument("--already-pca", nargs="*", default=None, metavar="MODEL",
                        help="The embedding already holds principal components: plot them, fit nothing. "
                             "Bare = every input; with names = only those models")
    parser.add_argument("--components", nargs=2, type=int, default=[1, 2], metavar=("X", "Y"),
                        help="Components to plot, 1-based (default: 1 2)")
    parser.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2",
                        help="Keep rows whose COL is one of the values (repeatable, AND across columns)")
    parser.add_argument("--subsample", type=int, default=20_000, help="Rows to plot (0 or less = all)")
    parser.add_argument("--stratify-by", nargs="*", default=["tissue_general"],
                        help="obs columns the subsample is stratified on (missing ones are skipped)")
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--color", nargs="+", default=["tissue_general"], help="obs columns to colour by")
    parser.add_argument("--skip-unknown", action="store_true", help="Grey out unknown/NaN labels")
    parser.add_argument("--title", default=None,
                        help="Figure title; may use {model}, {modality}, {modality_name}, {part}, {filter}, "
                             "{pcs} (default: '<model>: <modality> <part> [<filter>]')")
    parser.add_argument("--labels", nargs="*", default=None, metavar="COL=LABEL",
                        help="Readable names for obs columns in panel and legend titles "
                             "(e.g. tissue_general=Tissue); a mapping in --config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def is_precomputed(value, name: str | None) -> bool:
    """Whether ``--already-pca`` covers ``name`` (None = the ``--input`` file).

    ``None``/``False`` = no input; a bare flag (``[]``) or ``true`` = every input;
    a list of names = those models. With ``--input`` any use of the flag counts.
    """
    if value is None or value is False:
        return False
    if value is True or (isinstance(value, (list, tuple)) and not value) or name is None:
        return True
    names = [value] if isinstance(value, str) else [str(v) for v in value]
    return name in names


def fit_pca(emb: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores of the first ``n_components`` PCs and every PC's explained-variance ratio.

    Centred, not scaled, as scanpy and sklearn do. Each component's sign is fixed so
    its largest loading is positive, which makes the figure reproducible.
    """
    X = np.asarray(emb, dtype=np.float64)
    X = X - X.mean(axis=0)
    n, d = X.shape
    if n_components > min(n, d):
        raise ValueError(f"cannot take {n_components} components from {n} rows x {d} dimensions")
    if n >= d:
        # Covariance route: memory is d x d, not n x d, for large row counts.
        eigval, eigvec = np.linalg.eigh(X.T @ X)
        order = np.argsort(eigval)[::-1]
        eigval, loadings = np.clip(eigval[order], 0, None), eigvec[:, order]
    else:
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        eigval, loadings = s ** 2, vt.T
    loadings = loadings[:, :n_components]
    signs = np.sign(loadings[np.argmax(np.abs(loadings), axis=0), np.arange(n_components)])
    signs[signs == 0] = 1
    loadings = loadings * signs
    total = eigval.sum()
    ratio = eigval / total if total > 0 else np.zeros_like(eigval)
    return X @ loadings, ratio


def stored_variance_ratio(path: Path, obsm_key: str) -> np.ndarray | None:
    """``uns['pca']['variance_ratio']`` of an h5ad, read only for an ``X_pca`` embedding."""
    if obsm_key != "X_pca":
        return None
    import h5py

    with h5py.File(path, "r") as f:
        node = f.get("uns/pca/variance_ratio")
        return None if node is None else np.asarray(node[()], dtype=float)


def pca_coords(adata, components: list[int], precomputed: bool, variance_ratio=None):
    """(coords, per-axis variance ratio or None) for the two ``components`` (1-based)."""
    emb = np.asarray(adata.obsm[EMB_KEY])
    idx = [c - 1 for c in components]
    if precomputed:
        if max(components) > emb.shape[1]:
            raise ValueError(f"--components {components}: the embedding has only {emb.shape[1]} columns")
        ratios = None
        if variance_ratio is not None and len(variance_ratio) >= max(components):
            ratios = [float(variance_ratio[i]) for i in idx]
        return emb[:, idx], ratios
    scores, ratio = fit_pca(emb, max(components))
    return scores[:, idx], [float(ratio[i]) for i in idx]


def plot_one(adata, out_stem: Path, title: str, args, labels: dict[str, str],
             precomputed: bool, variance_ratio=None) -> None:
    if adata.n_obs < 3:
        print(f"  [skip] {title}: only {adata.n_obs} rows")
        return
    finite = np.isfinite(np.asarray(adata.obsm[EMB_KEY])).all(axis=1)
    if not finite.all():
        print(f"  [warn] {int((~finite).sum())} rows with non-finite values left out")
        adata = adata[finite].copy()
    coords, ratios = pca_coords(adata, args.components, precomputed, variance_ratio)
    axis_labels = tuple(f"PC{c}" + (f" ({r:.1%})" if ratios else "")
                        for c, r in zip(args.components, ratios or [None, None]))
    modality_col = "modality" if "modality" in adata.obs.columns else None
    fig = draw_umap_panels(coords, adata.obs, args.color, modality_col=modality_col,
                           skip_unknown=args.skip_unknown, title=title, labels=labels,
                           axis_labels=axis_labels)
    save_figure(fig, out_stem.with_suffix(".png"), dpi=args.dpi)
    save_coords(adata.obs, coords, out_stem.with_suffix(".parquet"),
                columns=tuple(f"pc{c}" for c in args.components))
    meta = {
        "components": list(args.components),
        "precomputed": precomputed,
        "variance_ratio": ratios,
        "n_rows": int(adata.n_obs),
        "n_dims": int(adata.obsm[EMB_KEY].shape[1]),
    }
    out_stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), "pca", argv)
    if min(args.components) < 1 or args.components[0] == args.components[1]:
        sys.exit(f"Error: --components takes two different 1-based indices, got {args.components}")
    filters = parse_filter_specs(args.filter)
    tag = filter_tag(filters)
    pcs = f"PC{args.components[0]}/PC{args.components[1]}"
    suffix = (f"_{tag}" if tag else "") + ("" if args.components == [1, 2] else
                                           f"_pc{args.components[0]}-{args.components[1]}")
    labels = parse_mapping(args.labels)

    if args.input is not None:
        precomputed = is_precomputed(args.already_pca, None)
        adata = load_external(args.input, args.obsm_key)
        ratio = stored_variance_ratio(args.input, args.obsm_key) if precomputed else None
        out_dir = args.out_dir or args.input.parent
        stem = out_dir / f"pca_{args.input.stem}{suffix}"
        title = format_title(args.title, args.input.stem + (f" [{tag}]" if tag else ""),
                             model=args.input.stem, modality="", modality_name="", part="",
                             filter=tag, pcs=pcs)
        print(f"[{args.input.name}] " + ("precomputed PCs" if precomputed else "fitting PCA"))
        plot_one(prepare(adata, filters, args, args.input.name), stem, title, args, labels, precomputed, ratio)
        return

    if not args.models:
        sys.exit("Error: give --models (or --input)")
    layout = ExperimentLayout.from_root(args.root)
    label = canonical_modality(args.modality)
    for model in args.models:
        precomputed = is_precomputed(args.already_pca, model)
        print(f"[{model}] {label} {args.part}: " + ("precomputed PCs" if precomputed else "fitting PCA"))
        adata = load_embedding(layout, model, label, args.part)
        out_dir = args.out_dir or layout.representation_dir(model)
        stem = out_dir / f"pca_{label}_{args.part}{suffix}"
        title = format_title(args.title, f"{model}: {label} {args.part}" + (f" [{tag}]" if tag else ""),
                             model=model, modality=label, modality_name=DISPLAY_NAMES.get(label, label),
                             part=args.part, filter=tag, pcs=pcs)
        plot_one(prepare(adata, filters, args, f"{model}/{label}_{args.part}"), stem, title, args, labels,
                 precomputed)


if __name__ == "__main__":
    main()
