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

``--pb-cells`` is a figure of its own: ``--pb-tissues`` tissues (default 6), each with
``--pb-per-tissue`` pseudobulks (4) that have >= ``--pb-min-cells`` cells (10), each with up to
``--cells-per-pb`` of its own cells (50) and its cell-mean star (the mean of ALL its cells,
as ``integration.py``'s pb_cells edge measures it), in one UMAP over those rows only,
coloured by ``--link-col`` and tissue. It shows whether a pseudobulk sits in the cloud of
its own cells rather than only among same-tissue samples. The selection
(``select_pb_cells``) depends only on obs and ``--seed``, so every model shows the same
pseudobulks and cells; it is written as ``<stem>.selection.json``.

The three files must come from one gene panel (``embed_splits.py`` embeds them together);
mismatched ``panel_hash`` values in ``embed.json`` are refused, since a per-modality
panel separates modalities by construction.

    python -m evaluate.representation.plot_alignment --models m1 --part test \\
        --per-modality sc=5000 bulk=all pseudobulk=all --stratify-by tissue_general \\
        --filter tissue_general=lung --connect-pairs --color modality tissue_general paired

Writes ``results/representation/_alignment/<model>/alignment_<part>[_paired|_pb_cells][_<tag>].{png,parquet}``
(plus ``.selection.json`` with ``--pb-cells``).
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
from utils.sampling import filter_mask, filter_tag, parse_filter_specs, parse_label_caps, STRATIFY_MODES, stratified_subsample  # noqa: E402
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
    parser.add_argument("--stratify-mode", default="proportional", choices=list(STRATIFY_MODES),
                        help="How the subsample is shared among strata: proportional to their sizes, "
                             "balanced (equal per stratum, capped at its size) or sqrt.")
    parser.add_argument("--unknown-frac", type=float, default=None,
                        help="At most this share of the subsample for unknown strata (0 = none).")
    parser.add_argument("--cap-labels", nargs="*", default=[], metavar="LABEL=FRACTION",
                        help="At most this share of the subset for strata carrying LABEL (e.g. "
                             "paired=0.05 \"cell line=0.05\"); YAML may give a mapping. The freed share "
                             "goes to the other strata.")
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
    pb = parser.add_argument_group(
        "pseudobulks with their cells (--pb-cells)",
        "A few pseudobulks per tissue, each with a subset of its own cells and their mean, in "
        "one UMAP fitted on those rows only. The selection depends only on obs and --seed, so "
        "every model shows the same pseudobulks and cells.")
    pb.add_argument("--pb-cells", action="store_true",
                    help="This mode (modalities sc + pseudobulk; no stratified subsample)")
    pb.add_argument("--pb-tissues", type=int, default=6, help="Tissues shown")
    pb.add_argument("--pb-per-tissue", type=int, default=4, help="Pseudobulks per tissue")
    pb.add_argument("--cells-per-pb", type=int, default=50,
                    help="Cells drawn per pseudobulk (all of them when it has fewer)")
    pb.add_argument("--pb-min-cells", type=int, default=10,
                    help="Cells (in the sc file, after --filter) a pseudobulk needs to be chosen")
    pb.add_argument("--pb-label-key", default="tissue_general",
                    help="obs column with the tissue the pseudobulks are grouped by")
    parser.add_argument("--color", nargs="+", default=None,
                        help="obs columns to colour by (default: modality tissue_general paired; "
                             "with --paired-only: paired)")
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


# Labels that name no tissue: such pseudobulks are never chosen in --pb-cells.
NON_TISSUE_LABELS = {"unknown", "nan", "none", "", "paired"}


def select_pb_cells(sc_obs: pd.DataFrame, pb_obs: pd.DataFrame, link_col: str, label_key: str,
                    n_tissues: int, per_tissue: int, cells_per_pb: int, min_cells: int,
                    seed: int) -> dict:
    """The pseudobulks and cells of a ``--pb-cells`` figure; pandas only.

    Eligible pseudobulks have a tissue in ``label_key`` (not unknown / paired / empty) and
    at least ``min_cells`` cells in ``sc_obs`` linked by ``link_col``. Tissues with at
    least ``per_tissue`` eligible pseudobulks are ranked by how many they have (ties by
    name); the first ``n_tissues`` are kept. Per tissue ``per_tissue`` pseudobulks, per
    pseudobulk ``min(cells_per_pb, available)`` cells, both drawn with ``seed``.

    Returns ``{"tissues", "keys" (key -> tissue), "pb_rows", "cell_rows",
    "cells_per_key" (drawn), "cells_available"}``; raises ValueError when no tissue
    qualifies or a file lacks ``link_col``.
    """
    for name, obs in (("sc", sc_obs), ("pseudobulk", pb_obs)):
        if link_col not in obs.columns:
            raise ValueError(f"the {name} embedding has no '{link_col}' column (joint-panel files made "
                             "before the obs fix: run python -m embed.restore_obs --model <model>)")
    if label_key not in pb_obs.columns:
        raise ValueError(f"the pseudobulk embedding has no '{label_key}' column")
    rng = np.random.default_rng(seed)
    cell_key = link_keys(sc_obs[link_col])
    n_cells = cell_key.dropna().value_counts()

    pb_key = link_keys(pb_obs[link_col])
    tissue = pb_obs[label_key].astype(object)
    ok_tissue = tissue.notna() & ~tissue.astype(str).str.strip().str.lower().isin(NON_TISSUE_LABELS)
    pb = pd.DataFrame({"key": pb_key, "tissue": tissue}, index=pb_obs.index)[pb_key.notna() & ok_tissue]
    pb = pb[~pb["key"].duplicated()]                       # one pseudobulk row per key
    pb = pb[pb["key"].map(n_cells).fillna(0) >= min_cells]
    per = pb.groupby("tissue")["key"].nunique()
    per = per[per >= per_tissue]
    if per.empty:
        raise ValueError(f"no tissue has {per_tissue} pseudobulks with >= {min_cells} cells "
                         "(lower --pb-per-tissue or --pb-min-cells)")
    ranked = sorted(per.index, key=lambda t: (-per[t], str(t)))
    tissues = ranked[:n_tissues]
    if len(tissues) < n_tissues:
        print(f"  [note] only {len(tissues)} tissues have {per_tissue} eligible pseudobulks")

    keys: dict[str, str] = {}
    pb_rows: list[str] = []
    for t in tissues:
        cand = pb[pb["tissue"] == t].sort_values("key")
        chosen = cand.iloc[np.sort(rng.choice(len(cand), size=per_tissue, replace=False))]
        for name, row in chosen.iterrows():
            keys[row["key"]] = t
            pb_rows.append(name)

    cell_rows: list[str] = []
    drawn: dict[str, int] = {}
    for key in keys:
        names = sc_obs.index[(cell_key == key).to_numpy()]
        take = min(cells_per_pb, len(names))
        pick = names[np.sort(rng.choice(len(names), size=take, replace=False))]
        cell_rows.extend(pick)
        drawn[key] = take
    return {"tissues": [str(t) for t in tissues], "keys": {k: str(v) for k, v in keys.items()},
            "pb_rows": [str(r) for r in pb_rows], "cell_rows": [str(r) for r in cell_rows],
            "cells_per_key": drawn, "cells_available": {k: int(n_cells[k]) for k in keys}}


def collect_pb_cells(layout: ExperimentLayout, model: str, args, filters) -> tuple:
    """Rows of a ``--pb-cells`` figure: the chosen pseudobulks, their drawn cells and one
    cell-mean star each (the mean over ALL the pseudobulk's cells after --filter, as in
    integration.py's pb_cells edge)."""
    import anndata as ad

    labels = [MOD_SC, MOD_PB]
    files = [layout.embedding_file(model, m, args.part, args.variant) for m in labels]
    hashes = panel_hashes(layout, model, [f for f in files if f.exists()], args.variant)
    if len({h for h in hashes.values() if h is not None}) > 1 and not args.allow_mixed_panels:
        raise SystemExit(f"[{model}] the modalities were embedded with different gene panels {hashes}; "
                         "pass --allow-mixed-panels to plot anyway")
    data = {}
    for label in labels:
        adata = load_embedding(layout, model, label, args.part, args.variant)
        adata.obs[MODALITY_COL] = label
        data[label] = adata[filter_mask(adata.obs, filters, label=f"{label}_{args.part}")].copy()
    try:
        sel = select_pb_cells(data[MOD_SC].obs, data[MOD_PB].obs, args.link_col, args.pb_label_key,
                              args.pb_tissues, args.pb_per_tissue, args.cells_per_pb, args.pb_min_cells,
                              args.seed)
    except ValueError as exc:
        raise SystemExit(f"[{model}] --pb-cells: {exc}") from exc
    means = cell_mean_rows(data[MOD_SC], args.link_col, args.pb_min_cells,
                           [*args.color, args.pb_label_key])
    if means is not None:
        means = means[means.obs[args.link_col].astype(str).isin(sel["keys"]).to_numpy()].copy()
        means.obs[args.pb_label_key] = means.obs[args.link_col].map(sel["keys"])
    pb = data[MOD_PB][sel["pb_rows"]].copy()
    cells = data[MOD_SC][sel["cell_rows"]].copy()
    cells.obs[args.pb_label_key] = link_keys(cells.obs[args.link_col]).map(sel["keys"]).to_numpy()
    print(f"  {len(sel['tissues'])} tissues x {args.pb_per_tissue} pseudobulks: {sel['tissues']}; "
          f"{cells.n_obs} cells (median {int(np.median(list(sel['cells_per_key'].values())))} per pseudobulk)")
    combined = ad.concat([pb, cells] + ([means] if means is not None else []),
                         join="outer", merge="same", label=None)
    combined.obs_names_make_unique()
    return combined, hashes, sel


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
        keep = stratified_subsample(adata.obs, quotas[label], strata, args.min_per_stratum, args.seed,
                                    mode=args.stratify_mode, unknown_frac=args.unknown_frac,
                                    caps=parse_label_caps(args.cap_labels))
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
    if args.pb_cells and args.paired_only:
        sys.exit("Error: --pb-cells and --paired-only are separate figures; give one")
    if args.color is None:
        args.color = ([args.link_col, args.pb_label_key] if args.pb_cells
                      else [args.paired_col] if args.paired_only
                      else ["modality", "tissue_general", args.paired_col])
    if not args.models:
        sys.exit("Error: give --models")
    filters = parse_filter_specs(args.filter)
    tag = filter_tag(filters)
    suffix = ("_paired" if args.paired_only else "_pb_cells" if args.pb_cells else "") + (f"_{tag}" if tag else "")
    layout = ExperimentLayout.from_root(args.root)
    labels = parse_mapping(args.labels)

    for model in args.models:
        print(f"[{model}] alignment of {args.part}" + (" (pseudobulks with their cells)" if args.pb_cells else ""))
        selection = None
        if args.pb_cells:
            combined, _, selection = collect_pb_cells(layout, model, args, filters)
        else:
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
        if args.pb_cells:                 # each star -> its pseudobulk
            segments = link_segments(coords, obs, link, pairs, False)
        elif args.cell_means:
            segments = link_segments(coords, obs, link, pairs, args.connect_pairs)
        else:
            segments = pair_segments(coords, obs, pairs) if args.connect_pairs else None

        counts = obs[MODALITY_COL].value_counts().to_dict()
        count_text = ", ".join(f"{k}={v}" for k, v in counts.items())
        default = f"{model}: alignment ({args.part}) {count_text}" + (f" [{tag}]" if tag else "")
        title = format_title(args.title, default, model=model, part=args.part, counts=count_text,
                             modalities=", ".join(str(k) for k in counts), filter=tag,
                             variant=args.variant or "")
        if args.pb_cells:
            # pair ids would only add a grey panel here; the link column is a category
            highlight = {}
            if args.link_col in obs.columns:
                obs[args.link_col] = link.astype(object).to_numpy()
        fig = draw_umap_panels(coords, obs, args.color, modality_col=MODALITY_COL,
                               skip_unknown=args.skip_unknown, title=title,
                               segments=segments, highlight=highlight, labels=labels,
                               max_legend_entries=args.max_legend_entries,
                               marker_style="paired" if (args.paired_only or args.pb_cells) else None)
        if args.pb_cells:
            fig.text(0.01, 0.005, "Triangles: pseudobulks; dots: a sample of their own cells "
                     f"(at most {args.cells_per_pb}); stars: the mean of ALL their cells in the file "
                     "(as integration.py's pb_cells edge), joined to their pseudobulk.",
                     fontsize=7, color="#898781", ha="left", va="bottom")
        out_dir = args.out_dir or layout.alignment_dir(model, args.variant)
        stem = out_dir / f"alignment_{args.part}{suffix}"
        save_figure(fig, stem.with_suffix(".png"), dpi=args.dpi)
        save_coords(obs, coords, stem.with_suffix(".parquet"))
        if selection is not None:
            import json

            stem.with_suffix(".selection.json").write_text(json.dumps({
                "link_col": args.link_col, "label_key": args.pb_label_key, "seed": args.seed,
                "pb_tissues": args.pb_tissues, "pb_per_tissue": args.pb_per_tissue,
                "cells_per_pb": args.cells_per_pb, "pb_min_cells": args.pb_min_cells,
                "filter": args.filter, **selection}, indent=2), encoding="utf-8")
            print(f"  selection -> {stem.with_suffix('.selection.json')}")


if __name__ == "__main__":
    main()
