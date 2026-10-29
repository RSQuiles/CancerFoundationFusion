"""Alignment UMAP: single-cell, pseudobulk and bulk embeddings of one model in one map.

Each modality's ``embeddings/<model>/<modality>_<part>.h5ad`` is filtered and
subsampled on its own (``--per-modality sc=5000 bulk=all``), the rows are pooled
and ONE UMAP is fitted, so distances between modalities mean something. Panels:
modality (fixed colours), then every other ``--color`` column with marker shape
still encoding modality. The pair-id column only colours paired rows; the rest are
grey. ``--connect-pairs`` draws a faint segment from each bulk point to the
centroid of its pair's pseudobulk and single-cell points.

The three files must come from one gene panel (``embed_splits.py`` embeds them together);
mismatched ``panel_hash`` values in ``embed.json`` are refused, since a per-modality
panel separates modalities by construction.

    python -m evaluate.representation.plot_alignment --models m1 --part test \\
        --per-modality sc=5000 bulk=all pseudobulk=all --stratify-by tissue_general \\
        --filter tissue_general=lung --connect-pairs --color modality tissue_general paired

Writes ``results/representation/_alignment/<model>/alignment_<part>[_<tag>].{png,parquet}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import format_title, parse_mapping, parse_quotas, parse_with_config  # noqa: E402
from experiments.layout import ExperimentLayout, canonical_modality  # noqa: E402
from utils.sampling import filter_mask, filter_tag, parse_filter_specs, stratified_subsample  # noqa: E402
from evaluate.io import EMB_KEY, load_embedding, panel_hashes  # noqa: E402
from evaluate.modalities import MOD_BULK, MODALITIES, PAIRED_COL  # noqa: E402
from evaluate.representation.render import compute_umap, draw_umap_panels, save_coords, save_figure  # noqa: E402

MODALITY_COL = "modality"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint UMAP of the three modalities' embeddings, per model.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", required=False, default=None, help="Model names")
    parser.add_argument("--part", default="test", choices=["test", "train_subset"])
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--per-modality", nargs="*", default=["sc=5000"], metavar="MOD=N|all",
                        help="Rows per modality (default: sc=5000, the others all)")
    parser.add_argument("--stratify-by", nargs="*", default=["tissue_general"],
                        help="obs columns each modality's subsample is stratified on")
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2",
                        help="Applied to every modality (a column missing from one modality is an error)")
    parser.add_argument("--paired-col", default=PAIRED_COL, help="obs column with pair ids (>0 = paired)")
    parser.add_argument("--paired-only", action="store_true", help="Keep only paired rows")
    parser.add_argument("--connect-pairs", action="store_true",
                        help="Segments from each bulk point to its pair's pseudobulk / SC centroids")
    parser.add_argument("--color", nargs="+", default=["modality", "tissue_general", PAIRED_COL])
    parser.add_argument("--skip-unknown", action="store_true")
    parser.add_argument("--title", default=None,
                        help="Figure title; may use {model}, {part}, {modalities}, {counts}, {filter} "
                             "(default: '<model>: alignment (<part>) <counts> [<filter>]')")
    parser.add_argument("--labels", nargs="*", default=None, metavar="COL=LABEL",
                        help="Readable names for obs columns in panel and legend titles "
                             "(e.g. tissue_general=Tissue); a mapping in --config")
    parser.add_argument("--allow-mixed-panels", action="store_true",
                        help="Plot even when the files were embedded with different gene panels")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def pair_ids(obs: pd.DataFrame, col: str) -> pd.Series:
    """Pair id as a string, NaN for unpaired rows (id 0, missing or non-numeric)."""
    if col not in obs.columns:
        return pd.Series(np.nan, index=obs.index, dtype=object)
    ids = pd.to_numeric(obs[col], errors="coerce")
    return ids.where(ids > 0).map(lambda v: str(int(v)) if pd.notna(v) else np.nan)


def pair_segments(coords: np.ndarray, obs: pd.DataFrame, pairs: pd.Series) -> list[np.ndarray]:
    """One (2, 2) segment per bulk point and per other modality present in its pair."""
    frame = pd.DataFrame({"pair": pairs.to_numpy(), "mod": obs[MODALITY_COL].astype(str).to_numpy(),
                          "x": coords[:, 0], "y": coords[:, 1]}).dropna(subset=["pair"])
    centroids = frame[frame["mod"] != MOD_BULK].groupby(["pair", "mod"])[["x", "y"]].mean()
    segments = []
    for row in frame[frame["mod"] == MOD_BULK].itertuples():
        if row.pair not in centroids.index.get_level_values(0):
            continue
        for _, target in centroids.loc[row.pair].iterrows():
            segments.append(np.array([[row.x, row.y], [target["x"], target["y"]]]))
    return segments


def collect(layout: ExperimentLayout, model: str, args, filters) -> tuple:
    import anndata as ad

    labels = list(dict.fromkeys(canonical_modality(m) for m in args.modalities))
    quotas = parse_quotas(args.per_modality, labels)
    files = [layout.embedding_file(model, m, args.part) for m in labels]
    hashes = panel_hashes(layout, model, [f for f in files if f.exists()])
    distinct = {h for h in hashes.values() if h is not None}
    if len(distinct) > 1 and not args.allow_mixed_panels:
        raise SystemExit(
            f"[{model}] the modalities were embedded with different gene panels {hashes}; "
            "re-run embed/embed_splits.py with all modalities at once, or pass --allow-mixed-panels"
        )
    if None in hashes.values():
        print(f"  [warn] no panel hash recorded for some files: {hashes}")

    pieces = []
    for label in labels:
        try:
            adata = load_embedding(layout, model, label, args.part)
        except FileNotFoundError as exc:
            print(f"  [warn] {exc}")
            continue
        adata.obs[MODALITY_COL] = label
        mask = filter_mask(adata.obs, filters, label=f"{label}_{args.part}")
        if args.paired_only:
            mask &= pair_ids(adata.obs, args.paired_col).notna().to_numpy()
        adata = adata[mask].copy()
        strata = [c for c in args.stratify_by if c in adata.obs.columns]
        keep = stratified_subsample(adata.obs, quotas[label], strata, args.min_per_stratum, args.seed)
        adata = adata[adata.obs_names.isin(keep)].copy()
        print(f"  {label}: {adata.n_obs} rows")
        if adata.n_obs:
            pieces.append(adata)
    if len(pieces) < 2:
        raise SystemExit(f"[{model}] fewer than two modalities left to align")
    combined = ad.concat(pieces, join="outer", merge="same", label=None)
    combined.obs_names_make_unique()
    return combined, hashes


def main(argv: list[str] | None = None) -> None:
    args = parse_with_config(build_parser(), "alignment", argv)
    if not args.models:
        sys.exit("Error: give --models")
    filters = parse_filter_specs(args.filter)
    tag = filter_tag(filters)
    suffix = ("_paired" if args.paired_only else "") + (f"_{tag}" if tag else "")
    layout = ExperimentLayout.from_root(args.root)
    labels = parse_mapping(args.labels)

    for model in args.models:
        print(f"[{model}] alignment of {args.part}")
        combined, _ = collect(layout, model, args, filters)
        coords = compute_umap(combined.obsm[EMB_KEY], args.n_neighbors, args.min_dist, args.seed)

        obs = combined.obs.copy()
        pairs = pair_ids(obs, args.paired_col)
        highlight = {}
        if args.paired_col in obs.columns:
            obs[args.paired_col] = pairs.fillna("unpaired")
            highlight[args.paired_col] = pairs.notna().to_numpy()
            print(f"  {int(pairs.notna().sum())} paired rows across {pairs.nunique()} pairs")
        segments = pair_segments(coords, obs, pairs) if args.connect_pairs else None

        counts = obs[MODALITY_COL].value_counts().to_dict()
        count_text = ", ".join(f"{k}={v}" for k, v in counts.items())
        default = f"{model}: alignment ({args.part}) {count_text}" + (f" [{tag}]" if tag else "")
        title = format_title(args.title, default, model=model, part=args.part, counts=count_text,
                             modalities=", ".join(str(k) for k in counts), filter=tag)
        fig = draw_umap_panels(coords, obs, args.color, modality_col=MODALITY_COL,
                               skip_unknown=args.skip_unknown, title=title,
                               segments=segments, highlight=highlight, labels=labels)
        out_dir = args.out_dir or layout.alignment_dir(model)
        stem = out_dir / f"alignment_{args.part}{suffix}"
        save_figure(fig, stem.with_suffix(".png"), dpi=args.dpi)
        save_coords(obs, coords, stem.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
