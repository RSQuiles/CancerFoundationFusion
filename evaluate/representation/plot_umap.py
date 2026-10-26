"""UMAP of one modality's embeddings, per model.

Reads ``embeddings/<model>/<modality>_<part>.h5ad`` (written by ``embed/embed_splits.py``),
filters rows by obs fields, draws a stratified subsample, fits a UMAP and colours
it by each ``--color`` column. Filtering comes first, so the sample budget is
spent inside the selection. Writes, per model::

    results/representation/<model>/umap_<modality>_<part>[_<filter tag>].png
    results/representation/<model>/umap_<modality>_<part>[_<filter tag>].parquet   coords + obs

    python -m evaluate.representation.plot_umap --models m1 m2 pca --modality sc --part test \\
        --filter tissue_general=lung,breast --subsample 20000 --stratify-by tissue_general \\
        --color tissue_general cancer_type assay --skip-unknown

Any h5ad outside the layout works too (``--input file.h5ad --obsm-key X_pca``);
the figure then goes next to it unless ``--out-dir`` says otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.representation._cli import parse_with_config  # noqa: E402
from experiments.layout import ExperimentLayout, canonical_modality  # noqa: E402
from utils.sampling import filter_mask, filter_tag, parse_filter_specs, stratified_subsample  # noqa: E402
from evaluate.io import EMB_KEY, load_embedding, load_external  # noqa: E402
from evaluate.representation.render import compute_umap, draw_umap_panels, save_coords, save_figure  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-model UMAPs of one modality's embeddings.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", default=None, help="Model names (embeddings/<model>/)")
    parser.add_argument("--modality", default="sc", help="sc | bulk | pseudobulk")
    parser.add_argument("--part", default="test", choices=["test", "train_subset"])
    parser.add_argument("--input", type=Path, default=None, help="Any h5ad instead of --models")
    parser.add_argument("--obsm-key", default=EMB_KEY, help="obsm key holding the embedding (with --input)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results/representation/<model>/)")
    parser.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2",
                        help="Keep rows whose COL is one of the values (repeatable, AND across columns)")
    parser.add_argument("--subsample", type=int, default=20_000, help="Rows to plot (0 or less = all)")
    parser.add_argument("--stratify-by", nargs="*", default=["tissue_general"],
                        help="obs columns the subsample is stratified on (missing ones are skipped)")
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--color", nargs="+", default=["tissue_general"], help="obs columns to colour by")
    parser.add_argument("--skip-unknown", action="store_true", help="Grey out unknown/NaN labels")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def prepare(adata, filters, args, label: str):
    """Filter, then stratified subsample; returns the selected AnnData view copy."""
    mask = filter_mask(adata.obs, filters, label=label)
    sub = adata[mask].copy()
    strata = [c for c in args.stratify_by if c in sub.obs.columns]
    n = args.subsample if args.subsample and args.subsample > 0 else None
    keep = stratified_subsample(sub.obs, n, strata, args.min_per_stratum, args.seed)
    sub = sub[sub.obs_names.isin(keep)].copy()
    print(f"  {label}: plotting {sub.n_obs} rows" + (f" (stratified on {strata})" if strata and n else ""))
    return sub


def plot_one(adata, out_stem: Path, title: str, args) -> None:
    if adata.n_obs < 3:
        print(f"  [skip] {title}: only {adata.n_obs} rows")
        return
    coords = compute_umap(adata.obsm[EMB_KEY], args.n_neighbors, args.min_dist, args.seed)
    modality_col = "modality" if "modality" in adata.obs.columns else None
    fig = draw_umap_panels(coords, adata.obs, args.color, modality_col=modality_col,
                           skip_unknown=args.skip_unknown, title=title)
    save_figure(fig, out_stem.with_suffix(".png"), dpi=args.dpi)
    save_coords(adata.obs, coords, out_stem.with_suffix(".parquet"))


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), "umap", argv)
    filters = parse_filter_specs(args.filter)
    tag = filter_tag(filters)
    suffix = f"_{tag}" if tag else ""

    if args.input is not None:
        adata = load_external(args.input, args.obsm_key)
        out_dir = args.out_dir or args.input.parent
        stem = out_dir / f"umap_{args.input.stem}{suffix}"
        plot_one(prepare(adata, filters, args, args.input.name), stem, args.input.stem + (f" [{tag}]" if tag else ""), args)
        return

    if not args.models:
        sys.exit("Error: give --models (or --input)")
    layout = ExperimentLayout.from_root(args.root)
    label = canonical_modality(args.modality)
    for model in args.models:
        print(f"[{model}] {label} {args.part}")
        adata = load_embedding(layout, model, label, args.part)
        out_dir = args.out_dir or layout.representation_dir(model)
        stem = out_dir / f"umap_{label}_{args.part}{suffix}"
        title = f"{model}: {label} {args.part}" + (f" [{tag}]" if tag else "")
        plot_one(prepare(adata, filters, args, f"{model}/{label}_{args.part}"), stem, title, args)


if __name__ == "__main__":
    main()
