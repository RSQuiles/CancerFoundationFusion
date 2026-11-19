"""Alignment UMAP: single-cell, pseudobulk and bulk embeddings of one model in one map.

Each modality's ``embeddings/<model>/<modality>_<part>.h5ad`` is filtered and
subsampled on its own (``--per-modality sc=5000 bulk=all``), the rows are pooled
and ONE UMAP is fitted, so distances between modalities mean something. Panels:
modality (fixed colours), then every other ``--color`` column with marker shape
still encoding modality. The pair-id column only colours paired rows; the rest are
grey. ``--connect-pairs`` draws a faint segment from each bulk point to the
centroid of its pair's pseudobulk and single-cell points.

``--cell-means --link-col COL`` adds one star per link key (a pseudobulk with
``pseudobulk_id``, a pair with ``paired``): the mean embedding of its single cells, taken
over every loaded cell before the subsample and fitted with the UMAP, so it sits where
``evaluate/integration/integration.py`` measures it. A segment joins each star to its
pseudobulk; with ``--connect-pairs``, each bulk point is joined to its pair's pseudobulk,
which draws the chain cells -> pseudobulk -> bulk when the link is ``paired``.

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
from evaluate.modalities import MOD_BULK, MOD_CELL_MEAN, MOD_PB, MOD_SC, MODALITIES, PAIRED_COL  # noqa: E402
from utils.sampling import link_keys  # noqa: E402
from evaluate.representation.render import compute_umap, draw_umap_panels, save_coords, save_figure  # noqa: E402

MODALITY_COL = "modality"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint UMAP of the three modalities' embeddings, per model.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", required=False, default=None, help="Model names")
    parser.add_argument("--part", default="test", choices=["test", "train_subset"])
    parser.add_argument("--variant", default=None,
                        help="Embedding variant: read embeddings/<model>/<variant>/ (e.g. own_panel) "
                             "and write the figures under <variant>/ (default: the model's own files)")
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
                        help="Segments from each bulk point to its pair's pseudobulk / SC centroids "
                             "(with --cell-means: to its pair's pseudobulk only)")
    parser.add_argument("--cell-means", action="store_true",
                        help="Add the mean embedding of the cells of each --link-col key as a star, "
                             "joined to that key's pseudobulk")
    parser.add_argument("--link-col", default="pseudobulk_id",
                        help="obs column linking cells to their pseudobulk (pseudobulk_id) or pair (paired)")
    parser.add_argument("--min-cells", type=int, default=5, help="Cells a key needs for its star")
    parser.add_argument("--color", nargs="+", default=["modality", "tissue_general", PAIRED_COL])
    parser.add_argument("--skip-unknown", action="store_true")
    parser.add_argument("--max-legend-entries", type=int, default=None,
                        help="Cap a category legend to its N most frequent values plus a \"+K more\" line "
                             "(default: every category is listed)")
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


def cell_mean_rows(adata, link_col: str, min_cells: int, columns: list[str]):
    """One row per ``link_col`` key with >= ``min_cells`` cells: their mean embedding, and
    per obs column in ``columns`` the most common value among them. None when there is
    nothing to add."""
    import anndata as ad
    from evaluate.integration.integration import link_centroids

    if link_col not in adata.obs.columns:
        print(f"  [warn] sc has no '{link_col}' column; no cell means")
        return None
    keys = link_keys(adata.obs[link_col])
    frame, vectors = link_centroids(adata.obsm[EMB_KEY], keys, pd.Series(np.nan, index=adata.obs.index),
                                    min_cells)
    if frame.empty:
        print(f"  [warn] no '{link_col}' key has {min_cells}+ cells; no cell means")
        return None
    cols = [c for c in dict.fromkeys(columns) if c in adata.obs.columns and c != link_col]
    obs = adata.obs[cols].astype(object).assign(_key=keys.to_numpy()).dropna(subset=["_key"])
    modes = obs.groupby("_key").agg(lambda c: c.mode().iloc[0] if c.notna().any() else np.nan)
    means = modes.reindex(frame.index.astype(str))
    means[link_col] = frame.index.astype(str)
    means[MODALITY_COL] = MOD_CELL_MEAN
    means.index = [f"cell_mean:{k}" for k in frame.index]
    print(f"  cell means: {len(frame)} '{link_col}' keys (median {int(frame['n'].median())} cells)")
    return ad.AnnData(obs=means.astype(object), obsm={EMB_KEY: vectors.astype(np.float32)})


def link_segments(coords: np.ndarray, obs: pd.DataFrame, key: pd.Series, pairs: pd.Series,
                  connect_pairs: bool) -> list[np.ndarray]:
    """Cell mean -> pseudobulk of the same key; with ``connect_pairs`` also bulk -> its
    pair's pseudobulk centroid (cells -> pb -> bulk when ``key`` is the pair id). ``key``
    and ``pairs`` hold the link key per row, NaN for none."""
    mod = obs[MODALITY_COL].astype(str).to_numpy()
    xy = pd.DataFrame(coords, columns=["x", "y"], index=obs.index)
    segments = []

    def targets(key: pd.Series, where: np.ndarray) -> pd.DataFrame:
        return xy[where].groupby(key[where].to_numpy()).mean()

    if key.notna().any():
        pb = targets(key, (mod == MOD_PB) & key.notna().to_numpy())
        for name in obs.index[(mod == MOD_CELL_MEAN) & key.notna().to_numpy()]:
            if key[name] in pb.index:
                segments.append(np.array([xy.loc[name].to_numpy(), pb.loc[key[name]].to_numpy()]))
    if connect_pairs:
        pb = targets(pairs, (mod == MOD_PB) & pairs.notna().to_numpy())
        for name in obs.index[(mod == MOD_BULK) & pairs.notna().to_numpy()]:
            if pairs[name] in pb.index:
                segments.append(np.array([xy.loc[name].to_numpy(), pb.loc[pairs[name]].to_numpy()]))
    return segments


def collect(layout: ExperimentLayout, model: str, args, filters) -> tuple:
    import anndata as ad

    labels = list(dict.fromkeys(canonical_modality(m) for m in args.modalities))
    quotas = parse_quotas(args.per_modality, labels)
    files = [layout.embedding_file(model, m, args.part, args.variant) for m in labels]
    hashes = panel_hashes(layout, model, [f for f in files if f.exists()], args.variant)
    distinct = {h for h in hashes.values() if h is not None}
    if len(distinct) > 1 and not args.allow_mixed_panels:
        raise SystemExit(
            f"[{model}] the modalities were embedded with different gene panels {hashes}; "
            "re-run embed/embed_splits.py with all modalities at once, or pass --allow-mixed-panels"
        )
    if None in hashes.values():
        print(f"  [warn] no panel hash recorded for some files: {hashes}")

    pieces, extra = [], []
    for label in labels:
        try:
            adata = load_embedding(layout, model, label, args.part, args.variant)
        except FileNotFoundError as exc:
            print(f"  [warn] {exc}")
            continue
        adata.obs[MODALITY_COL] = label
        mask = filter_mask(adata.obs, filters, label=f"{label}_{args.part}")
        if args.paired_only:
            mask &= pair_ids(adata.obs, args.paired_col).notna().to_numpy()
        adata = adata[mask].copy()
        if args.cell_means and label == MOD_SC:
            means = cell_mean_rows(adata, args.link_col, args.min_cells,
                                   [*args.color, args.paired_col, *args.stratify_by])
            if means is not None:
                extra.append(means)
        strata = [c for c in args.stratify_by if c in adata.obs.columns]
        keep = stratified_subsample(adata.obs, quotas[label], strata, args.min_per_stratum, args.seed)
        adata = adata[adata.obs_names.isin(keep)].copy()
        print(f"  {label}: {adata.n_obs} rows")
        if adata.n_obs:
            pieces.append(adata)
    if len(pieces) < 2:
        raise SystemExit(f"[{model}] fewer than two modalities left to align")
    combined = ad.concat(pieces + extra, join="outer", merge="same", label=None)
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
        if args.link_col == args.paired_col:
            link = pairs
        elif args.link_col in obs.columns:
            link = link_keys(obs[args.link_col])
        else:
            link = pd.Series(np.nan, index=obs.index, dtype=object)
        highlight = {}
        if args.paired_col in obs.columns:
            obs[args.paired_col] = pairs.fillna("unpaired")
            highlight[args.paired_col] = pairs.notna().to_numpy()
            print(f"  {int(pairs.notna().sum())} paired rows across {pairs.nunique()} pairs")
        if args.cell_means:
            segments = link_segments(coords, obs, link, pairs, args.connect_pairs)
        else:
            segments = pair_segments(coords, obs, pairs) if args.connect_pairs else None

        counts = obs[MODALITY_COL].value_counts().to_dict()
        count_text = ", ".join(f"{k}={v}" for k, v in counts.items())
        default = f"{model}: alignment ({args.part}) {count_text}" + (f" [{tag}]" if tag else "")
        title = format_title(args.title, default, model=model, part=args.part, counts=count_text,
                             modalities=", ".join(str(k) for k in counts), filter=tag,
                             variant=args.variant or "")
        fig = draw_umap_panels(coords, obs, args.color, modality_col=MODALITY_COL,
                               skip_unknown=args.skip_unknown, title=title,
                               segments=segments, highlight=highlight, labels=labels,
                               max_legend_entries=args.max_legend_entries)
        out_dir = args.out_dir or layout.alignment_dir(model, args.variant)
        stem = out_dir / f"alignment_{args.part}{suffix}"
        save_figure(fig, stem.with_suffix(".png"), dpi=args.dpi)
        save_coords(obs, coords, stem.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
