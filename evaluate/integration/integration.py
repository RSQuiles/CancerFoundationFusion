"""Cross-level integration: do linked single cells, pseudobulks and bulk samples meet?

scIB (``scib_benchmark.py``) measures integration at the label level: within a tissue, do
the modalities mix? It never sees which rows belong together, so a model that maps every
sample of a tissue to one point scores perfectly there. This script measures the
sample-level correspondence the linked split provides, and puts it in one table next to
the scIB scores.

Links (``utils.sampling.link_keys``, so ``0``, ``-1`` and empty mean "no link"):

- ``pseudobulk_id``: a pseudobulk and the single cells it was aggregated from;
- ``paired``: one pair id shared by a bulk sample, its pseudobulk(s) and their cells.

Each level gives ONE vector per key: the mean ``X_emb`` of its rows. Cell means need at
least ``--min-cells`` cells; the cells come from the joint-panel ``sc_<part>.h5ad``, so a
subsampled file gives a mean over the cells it holds (the same cells for every model).

An edge A -> B ranks, for each key's A vector, every B vector by cosine similarity:

- ``top1``: the key's own B vector is the nearest;
- ``rank``: its normalized rank among all B (0 = nearest, 0.5 = chance);
- ``rank_tissue``: the same among the B vectors of the key's tissue only, i.e. how well
  the model tells samples apart beyond their tissue (chance 0.5 as well).

Both directions are computed (``links.csv`` keeps them) and averaged in the summary.
A key's tissue is the most common ``--label-key`` value among its rows that pass
``--label-filter`` (curated labels), looked up in the pseudobulk, then the cells, then
the bulk rows. A pair therefore keeps its tissue when its bulk row is ``unknown``. Keys
without a curated tissue are left out of ``rank_tissue`` only.

Edges: ``pb_cells`` (cells mean <-> pseudobulk, on ``pseudobulk_id``) and ``pair_cells_pb``,
``pair_pb_bulk``, ``pair_cells_bulk`` (on ``paired``).

    python -m evaluate.integration.integration --name leiden_s1 --models m1 m2 pca_leiden_s1 \\
        --scib-runs leiden_s1_sc_pb leiden_s1_sc_bulk leiden_s1_bulk_pb leiden_s1_all

Writes ``results/integration/<name>/``: ``links.csv`` (one row per model, edge and
direction), ``coverage.csv``, ``summary.csv`` (one row per model: link metrics plus the scIB
aggregates of each ``--scib-runs`` analysis), ``manifest.json`` and ``integration.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_mapping, parse_with_config  # noqa: E402
from evaluate.io import EMB_KEY, load_embedding, panel_hashes  # noqa: E402
from evaluate.modalities import MOD_BULK, MOD_PB, MOD_SC, PAIRED_COL  # noqa: E402
from experiments.layout import ExperimentLayout, validate_model_name  # noqa: E402
from utils.sampling import fold_label, link_keys, parse_filter_specs  # noqa: E402

PB_ID_COL = "pseudobulk_id"
UNKNOWN_LABELS = {"unknown", "nan", "none", ""}
# edge name -> (link column, level A, level B); "cells" is the mean of a key's sc rows
EDGES = {
    "pb_cells": (PB_ID_COL, MOD_SC, MOD_PB),
    "pair_cells_pb": (PAIRED_COL, MOD_SC, MOD_PB),
    "pair_pb_bulk": (PAIRED_COL, MOD_PB, MOD_BULK),
    "pair_cells_bulk": (PAIRED_COL, MOD_SC, MOD_BULK),
}
TISSUE_PRIORITY = (MOD_PB, MOD_SC, MOD_BULK)
LEVEL_NAMES = {MOD_SC: "cells", MOD_PB: "pb", MOD_BULK: "bulk"}
SCIB_AGGREGATES = {"batch": "Batch correction", "bio": "Bio conservation", "total": "Total"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Link metrics between single cells, pseudobulk and bulk, "
                                            "plus a summary with the scIB scores.")
    p.add_argument("--root", type=Path, default=None, help="Experiments root")
    p.add_argument("--name", default=None, help="Analysis name -> results/integration/<name>/")
    p.add_argument("--models", nargs="+", default=None, help="Model names (embeddings/<model>/)")
    p.add_argument("--part", default="test", choices=["test", "train_subset"])
    p.add_argument("--variant", default=None, help="Embedding variant (default: the joint-panel files)")
    p.add_argument("--min-cells", type=int, default=5, help="Cells a key needs for its cell mean")
    p.add_argument("--label-key", default="tissue_general", help="obs column with the tissue")
    p.add_argument("--label-filter", nargs="*", default=["tissue_label_source=cellxgene,archs4"],
                   metavar="COL=V1,V2",
                   help="Rows whose label counts as curated (others give no tissue). Never removes rows "
                        "from the link metrics; empty = every known label counts")
    p.add_argument("--scib-runs", nargs="*", default=[], metavar="NAME|GROUP=NAME1,NAME2",
                   help="scIB analyses (results/scib_benchmark/<name>/) whose aggregates join the summary. "
                        "GROUP=NAME1,NAME2 merges runs made over different models into one column group: "
                        "each model takes its values from the first run listing it")
    p.add_argument("--links-from", nargs="*", default=[], metavar="NAME",
                   help="Reuse the link metrics of earlier analyses (results/integration/<name>/links.csv) "
                        "instead of reading embeddings: each model from the first analysis that has it")
    p.add_argument("--row-labels", nargs="*", default=None, metavar="MODEL=LABEL",
                   help="Display names of the models in the figure (a mapping in --config)")
    p.add_argument("--title", default=None, help="Figure title (default: 'Integration: <name>')")
    p.add_argument("--highlight-top", type=int, default=2,
                   help="Metric tables: shade the top N values of each column, darker = better (0 = no shading; the best stays bold)")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--dpi", type=int, default=200)
    from evaluate._recompute import add_recompute_arg

    add_recompute_arg(p)
    return p


# ---------------------------------------------------------------------------- metrics


def curated_labels(obs: pd.DataFrame, label_key: str, filters: dict[str, list[str]], where: str) -> pd.Series:
    """The label of each row, NaN where it is unknown or fails the curation filter."""
    if label_key not in obs.columns:
        print(f"  [warn] {where}: no '{label_key}' column; no tissue from these rows")
        return pd.Series(np.nan, index=obs.index, dtype=object)
    labels = obs[label_key].astype(object).where(obs[label_key].notna())
    labels = labels.where(~labels.astype(str).str.strip().str.lower().isin(UNKNOWN_LABELS))
    for col, values in filters.items():
        if col not in obs.columns:
            print(f"  [warn] {where}: no '{col}' column for --label-filter (embedded before "
                  "map_archs4_tissue.py curate?); its labels are used unfiltered")
            continue
        wanted = {fold_label(v) for v in values}
        labels = labels.where(obs[col].astype(str).map(fold_label).isin(wanted))
    return labels


def link_centroids(emb: np.ndarray, keys: pd.Series, labels: pd.Series, min_count: int = 1):
    """Mean vector per link key with at least ``min_count`` rows.

    Returns ``(frame, vectors)``: ``frame`` indexed by key with ``n`` (rows) and ``tissue``
    (most common non-NaN label, NaN if none), ``vectors`` in the frame's order.
    """
    keys = pd.Series(np.asarray(keys, dtype=object))
    has = keys.notna().to_numpy()
    if not has.any():
        return pd.DataFrame(columns=["n", "tissue"]), np.zeros((0, emb.shape[1]))
    k = keys[has].astype(str).to_numpy()
    codes, uniq = pd.factorize(k)
    counts = np.bincount(codes, minlength=len(uniq))
    sums = np.zeros((len(uniq), emb.shape[1]))
    np.add.at(sums, codes, np.asarray(emb, dtype=np.float64)[has])
    lab = pd.Series(np.asarray(labels, dtype=object)[has])
    tissue = (pd.DataFrame({"k": k, "t": lab}).dropna(subset=["t"])
              .groupby("k")["t"].agg(lambda s: s.value_counts().index[0]))
    frame = pd.DataFrame({"n": counts, "tissue": tissue.reindex(uniq).to_numpy()}, index=pd.Index(uniq, name="key"))
    keep = counts >= min_count
    return frame[keep], sums[keep] / counts[keep, None]


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norm > 0, norm, 1.0)


def match_edge(keys_a: pd.Index, vec_a: np.ndarray, keys_b: pd.Index, vec_b: np.ndarray,
               tissue: pd.Series) -> dict:
    """Queries A, candidates B: top-1, normalized rank of the partner, and that rank among
    same-tissue candidates. ``tissue`` maps key -> tissue (NaN = none)."""
    pos_b = pd.Index(keys_b).get_indexer(keys_a)
    q = np.flatnonzero(pos_b >= 0)
    out = {"n_queries": int(len(q)), "n_candidates": int(len(keys_b)),
           "top1": np.nan, "rank": np.nan, "rank_tissue": np.nan, "n_tissue_queries": 0}
    if len(q) == 0 or len(keys_b) < 2:
        return out
    sims = _normalize(vec_a[q]) @ _normalize(vec_b).T
    partner = pos_b[q]
    own = sims[np.arange(len(q)), partner][:, None]
    others = np.ones_like(sims, dtype=bool)
    others[np.arange(len(q)), partner] = False
    # ties count half, so a constant embedding sits at chance (0.5), not at 0
    better = ((sims > own) & others).sum(1) + 0.5 * ((sims == own) & others).sum(1)
    out["top1"] = float(np.mean(better == 0))
    out["rank"] = float(np.mean(better / (len(keys_b) - 1)))

    t_b = tissue.reindex(keys_b).to_numpy(dtype=object)
    t_q = tissue.reindex(pd.Index(keys_a)[q]).to_numpy(dtype=object)
    ranks = []
    for i in range(len(q)):
        if pd.isna(t_q[i]):
            continue
        same = (t_b == t_q[i]) & others[i]
        if not same.any():
            continue
        r = ((sims[i] > own[i]) & same).sum() + 0.5 * ((sims[i] == own[i]) & same).sum()
        ranks.append(r / same.sum())
    if ranks:
        out["rank_tissue"] = float(np.mean(ranks))
        out["n_tissue_queries"] = len(ranks)
    return out


def key_tissues(levels: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> pd.Series:
    """key -> tissue, from the first level (pseudobulk, cells, bulk) that has one."""
    tissue = pd.Series(dtype=object)
    for mod in TISSUE_PRIORITY:
        if mod in levels:
            t = levels[mod][0]["tissue"].dropna()
            tissue = pd.concat([tissue, t[~t.index.isin(tissue.index)]])
    return tissue


def model_links(data: dict[str, tuple[np.ndarray, pd.DataFrame]], label_key: str,
                filters: dict[str, list[str]], min_cells: int,
                missing: dict[str, list[str]] | None = None) -> tuple[list[dict], list[dict]]:
    """``(link rows, coverage rows)`` of one model; ``data`` maps modality -> (emb, obs).

    A modality whose obs lacks a link column makes that column's edges n/a; it is printed
    and, when ``missing`` is given, recorded there (modality -> columns)."""
    links, coverage = [], []
    for col in dict.fromkeys(c for c, _, _ in EDGES.values()):
        levels = {}
        for mod, (emb, obs) in data.items():
            if col not in obs.columns:
                edges = [e for e, (c, a, b) in EDGES.items() if c == col and mod in (a, b)]
                if edges:
                    print(f"  [warn] {mod}: no '{col}' column in the embedding obs; {', '.join(edges)} n/a "
                          "(files from a joint embed_splits call before the obs fix: run "
                          "python -m embed.restore_obs --model <model>)")
                    if missing is not None:
                        missing.setdefault(mod, []).append(col)
                continue
            labels = curated_labels(obs, label_key, filters, f"{mod}")
            levels[mod] = link_centroids(emb, link_keys(obs[col]), labels, min_cells if mod == MOD_SC else 1)
        tissue = key_tissues(levels)
        for edge, (edge_col, a, b) in EDGES.items():
            if edge_col != col:
                continue
            if a not in levels or b not in levels:
                links.append({"edge": edge, "direction": "n/a", "n_queries": 0})
                continue
            (fa, va), (fb, vb) = levels[a], levels[b]
            shared = fa.index.intersection(fb.index)
            coverage.append({
                "edge": edge, "link": col, f"keys_{LEVEL_NAMES[a]}": len(fa), f"keys_{LEVEL_NAMES[b]}": len(fb),
                "keys_shared": len(shared),
                "keys_shared_with_tissue": int(tissue.reindex(shared).notna().sum()),
                "median_cells_per_key": float(levels[MOD_SC][0]["n"].reindex(shared).median())
                if MOD_SC in (a, b) and len(shared) else np.nan,
            })
            for direction, (ka, xa, kb, xb) in {
                f"{LEVEL_NAMES[a]}->{LEVEL_NAMES[b]}": (fa.index, va, fb.index, vb),
                f"{LEVEL_NAMES[b]}->{LEVEL_NAMES[a]}": (fb.index, vb, fa.index, va),
            }.items():
                links.append({"edge": edge, "direction": direction, **match_edge(ka, xa, kb, xb, tissue)})
    return links, coverage


# ---------------------------------------------------------------------------- scIB


def parse_scib_specs(specs: list[str]) -> dict[str, list[str]]:
    """``NAME`` -> {NAME: [NAME]}; ``GROUP=A,B`` -> {GROUP: [A, B]} (order kept)."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        group, eq, runs = spec.partition("=")
        names = [r.strip() for r in runs.split(",") if r.strip()] if eq else [group]
        if not group or not names:
            raise SystemExit(f"Error: --scib-runs entry {spec!r}: expected NAME or GROUP=NAME1,NAME2")
        if group in out:
            raise SystemExit(f"Error: --scib-runs group {group!r} given twice")
        out[group] = names
    return out


def _read_scib(layout: ExperimentLayout, name: str):
    """``(frame of the batch / bio / total aggregates, meta)`` of one scIB run, or None."""
    from evaluate.integration.scib_plot import METRIC_TYPE, read_results

    out_dir = layout.scib_dir(name)
    path = out_dir / "scib.csv"
    if not path.exists():
        print(f"  [warn] {path} not found; scIB '{name}' left out of the summary")
        return None
    df = read_results(path).drop(index=METRIC_TYPE)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    frame = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce")
                          for k, v in SCIB_AGGREGATES.items() if v in df.columns}, index=df.index)
    return frame, meta


def scib_aggregates(layout: ExperimentLayout, specs: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """``(frame indexed by model with scib_<group>_<batch|bio|total>, {group: meta})``.

    A group merges scIB runs made over different model sets (e.g. the baselines' and an
    ablation's run on the same modalities): each model takes its values from the first
    run listing it. Models in several runs are compared across them, and so are the runs'
    row counts: a difference means the runs kept different rows (scIB keeps only the rows
    every model of a run embedded), so scores from different runs are not strictly
    comparable; both cases are warned about and recorded in the meta.
    """
    frames, metas = [], {}
    for group, names in parse_scib_specs(specs).items():
        parts = []
        for name in names:
            got = _read_scib(layout, name)
            if got is not None:
                parts.append((name, got[0], got[1]))
        if not parts:
            continue
        merged = parts[0][1].copy()
        max_diff = 0.0
        for name, frame, _ in parts[1:]:
            shared = merged.index.intersection(frame.index)
            if len(shared):
                cols = merged.columns.intersection(frame.columns)
                diff = (merged.loc[shared, cols] - frame.loc[shared, cols]).abs().max().max()
                if pd.notna(diff):
                    max_diff = max(max_diff, float(diff))
                    if diff > 1e-4:
                        print(f"  [warn] scIB group '{group}': {list(shared)} differ between "
                              f"{parts[0][0]} and {name} (max |diff| {diff:.4f}); the first run's values are used")
            merged = pd.concat([merged, frame.loc[frame.index.difference(merged.index)]])
        rows = {name: meta.get("n_rows") for name, _, meta in parts}
        if len({r for r in rows.values() if r is not None}) > 1:
            print(f"  [warn] scIB group '{group}': the runs kept different numbers of rows {rows}; "
                  "scores of models from different runs are not strictly comparable")
        meta = dict(parts[0][2])
        meta.update(runs=[name for name, _, _ in parts], n_rows_per_run=rows, max_shared_model_diff=max_diff,
                    run_of_model={m: next(n for n, f, _ in parts if m in f.index) for m in merged.index})
        metas[group] = meta
        frames.append(merged.add_prefix(f"scib_{group}_"))
    return (pd.concat(frames, axis=1) if frames else pd.DataFrame()), metas


def links_from(layout: ExperimentLayout, names: list[str], models: list[str]):
    """Link metrics of ``models`` from earlier analyses; nothing is recomputed.

    Returns ``(links, coverage, sources, manifests)``: each model's rows come from the
    first analysis in ``names`` whose ``links.csv`` has it."""
    loaded, manifests = [], {}
    for name in names:
        out_dir = layout.integration_dir(name)
        path = out_dir / "links.csv"
        if not path.exists():
            print(f"  [warn] {path} not found; '{name}' not used")
            continue
        cov_path, man_path = out_dir / "coverage.csv", out_dir / "manifest.json"
        loaded.append((name, pd.read_csv(path),
                       pd.read_csv(cov_path) if cov_path.exists() else pd.DataFrame(columns=["model"])))
        manifests[name] = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    links, cov, sources = [], [], {}
    for model in models:
        for name, lk, cv in loaded:
            if (lk["model"] == model).any():
                links.append(lk[lk["model"] == model])
                cov.append(cv[cv["model"] == model])
                sources[model] = name
                break
        else:
            print(f"  [warn] {model}: in none of {names}; its link metrics are n/a")
    links = (pd.concat(links, ignore_index=True) if links
             else pd.DataFrame(columns=["model", "edge", "direction", "top1", "rank", "rank_tissue"]))
    cov = pd.concat(cov, ignore_index=True) if cov else pd.DataFrame()
    return links, cov, sources, manifests


def scib_header(name: str, meta: dict) -> str:
    """Short column header: the modalities of the run, e.g. ``sc-pb``."""
    mods = meta.get("modalities") or []
    short = {MOD_SC: "sc", MOD_PB: "pb", MOD_BULK: "bulk"}
    return "-".join(short.get(m, m) for m in mods) if mods else name


# ---------------------------------------------------------------------------- summary + figure


def summarize(links: pd.DataFrame) -> pd.DataFrame:
    """One row per model: each edge's metrics averaged over its two directions."""
    valid = links[links["direction"] != "n/a"]
    if valid.empty:
        return pd.DataFrame(index=pd.Index(links["model"].unique(), name="model"))
    mean = valid.groupby(["model", "edge"])[["top1", "rank", "rank_tissue"]].mean()
    wide = mean.unstack("edge")
    wide.columns = [f"{edge}_{metric}" for metric, edge in wide.columns]
    return wide.reindex(links["model"].unique())


def missing_edge_notes(links: pd.DataFrame) -> list[str]:
    """One note per edge that no model could compute (its columns vanish from the table)."""
    if links.empty or "edge" not in links:
        return []
    notes = []
    for edge, rows in links.groupby("edge", sort=False):
        if (rows["direction"] == "n/a").all():
            col, a, b = EDGES[edge]
            notes.append(f"{edge} ({LEVEL_NAMES[a]} vs {LEVEL_NAMES[b]}) n/a for every model: "
                         f"no '{col}' in the embedding obs.")
    return notes


def figure(summary: pd.DataFrame, labels: dict[str, str], scib_meta: dict[str, dict], args,
           notes: list[str] | None = None):
    from evaluate._plot_style import Col, draw_metric_table, highlight_note

    # Pairs carry no curated tissue (their cells and pseudobulks are labelled "paired"),
    # so their columns rank among all paired candidates; rank_tissue stays in the CSVs.
    cols = [
        Col("pb_cells_top1", "top-1", "Pseudobulk vs its cells", "max"),
        Col("pb_cells_rank", "rank", "Pseudobulk vs its cells", "min"),
        Col("pb_cells_rank_tissue", "rank in tissue", "Pseudobulk vs its cells", "min"),
        Col("pair_pb_bulk_top1", "pb→bulk top-1", "Pairs: rank", "max"),
        Col("pair_cells_pb_rank", "cells→pb", "Pairs: rank", "min"),
        Col("pair_pb_bulk_rank", "pb→bulk", "Pairs: rank", "min"),
        Col("pair_cells_bulk_rank", "cells→bulk", "Pairs: rank", "min"),
    ]
    names = list(scib_meta)
    for name in names:
        cols.append(Col(f"scib_{name}_batch", scib_header(name, scib_meta[name]), "scIB batch", "max"))
    if names:  # bio conservation of the widest run (the one with the most modalities)
        widest = max(names, key=lambda n: len(scib_meta[n].get("modalities") or []))
        cols.append(Col(f"scib_{widest}_bio", scib_header(widest, scib_meta[widest]), "scIB bio", "max"))
    row_names = [labels.get(m, m) for m in summary.index]
    footnote = (f"Link metrics: cosine similarity between per-key mean embeddings ({args.part} split; cell means "
                f"need >= {args.min_cells} cells), averaged over both directions. top-1: own partner is the "
                "nearest (chance = 1/candidates). rank: normalized rank of the own partner among all "
                "candidates (0 = nearest, 0.5 = chance); in tissue: among candidates of its curated tissue "
                "only, i.e. sample identity beyond tissue. scIB: aggregate scores of each run "
                "(batch = modality, label = tissue). n/a: the model lacks a modality. "
                + highlight_note(args.highlight_top)
                + "".join(" " + n for n in (notes or [])))
    title = args.title or f"Integration: {args.name}"
    return draw_metric_table(summary.reset_index(drop=True), row_names, cols, title, footnote,
                             highlight_top=args.highlight_top)


def main(argv: list[str] | None = None) -> Path:
    args = parse_with_config(build_parser(), "integration", argv)
    if not args.models or not args.name:
        sys.exit("Error: give --models and --name")
    layout = ExperimentLayout.from_root(args.root)
    out_dir = layout.integration_dir(args.name)
    labels = parse_mapping(args.row_labels, "--row-labels")
    filters = parse_filter_specs(args.label_filter)
    models = [validate_model_name(m) for m in args.models]

    all_links, all_cov, hashes = [], [], {}
    missing: dict[str, dict[str, list[str]]] = {}
    sources: dict[str, str] = {}
    # --no-recompute: a model already in this analysis's links.csv keeps its rows; only the
    # others are computed (the summary and the figure are always rebuilt).
    reused: list[str] = []
    own_links = out_dir / "links.csv"
    if not args.recompute and not args.links_from and own_links.exists():
        present = set(pd.read_csv(own_links, usecols=["model"])["model"].astype(str))
        reused = [m for m in models if m in present]
        for m in reused:
            print(f"[skip] {m}: link metrics present in {own_links} (recompute false)")
        for m in (m for m in models if m not in present):
            print(f"[compute] {m}: not in {own_links}")
    for model in ([] if args.links_from else [m for m in models if m not in reused]):
        print(f"[{model}]")
        data = {}
        for mod in (MOD_SC, MOD_PB, MOD_BULK):
            try:
                adata = load_embedding(layout, model, mod, args.part, args.variant)
            except FileNotFoundError:
                print(f"  no {mod} embeddings; its edges are n/a")
                continue
            data[mod] = (np.asarray(adata.obsm[EMB_KEY]), adata.obs)
            print(f"  {mod}: {adata.n_obs} rows")
        files = [layout.embedding_file(model, m, args.part, args.variant) for m in data]
        hashes[model] = panel_hashes(layout, model, files, args.variant)
        if len({h for h in hashes[model].values() if h is not None}) > 1:
            print(f"  [warn] {model}: modalities embedded with different gene panels {hashes[model]}")
        missing_cols: dict[str, list[str]] = {}
        links, cov = model_links(data, args.label_key, filters, args.min_cells, missing_cols)
        if missing_cols:
            missing[model] = missing_cols
        all_links += [{"model": model, **r} for r in links]
        all_cov += [{"model": model, **r} for r in cov]

    if args.links_from:
        print(f"link metrics reused from {args.links_from} (no embeddings read)")
        links, cov_frame, sources, manifests = links_from(layout, args.links_from, models)
        for model, name in sources.items():
            man = manifests.get(name, {})
            hashes[model] = (man.get("panel_hashes") or {}).get(model)
            if model in (man.get("missing_link_columns") or {}):
                missing[model] = man["missing_link_columns"][model]
            print(f"  [{model}] from {name}")
        all_cov = cov_frame.to_dict("records")
    else:
        links = pd.DataFrame(all_links)
        if reused:
            old_links, old_cov, old_src, old_man = links_from(layout, [args.name], reused)
            man = old_man.get(args.name, {})
            for model in reused:
                hashes[model] = (man.get("panel_hashes") or {}).get(model)
                if model in (man.get("missing_link_columns") or {}):
                    missing[model] = man["missing_link_columns"][model]
            links = pd.concat([old_links, links], ignore_index=True) if len(links) else old_links
            all_cov = old_cov.to_dict("records") + all_cov
            order = {m: i for i, m in enumerate(models)}
            links = links.sort_values("model", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    summary = summarize(links).reindex(models)
    scib, scib_meta = scib_aggregates(layout, args.scib_runs)
    if not scib.empty:
        summary = summary.join(scib, how="left")
    summary.index.name = "model"

    out_dir.mkdir(parents=True, exist_ok=True)
    links.to_csv(out_dir / "links.csv", index=False)
    pd.DataFrame(all_cov).to_csv(out_dir / "coverage.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(3).to_string())

    from evaluate._plot_style import save

    figures = save(figure(summary, labels, scib_meta, args, notes=missing_edge_notes(links)),
                   out_dir / "integration", args.formats, args.dpi)
    (out_dir / "manifest.json").write_text(json.dumps({
        "analysis": "integration", "models": models, "row_labels": labels, "scib": list(scib_meta),
        "part": args.part, "variant": args.variant, "min_cells": args.min_cells, "label_key": args.label_key,
        "label_filter": filters, "panel_hashes": hashes, "figures": figures,
        "highlight_top": args.highlight_top, "missing_link_columns": missing,
        "links_from": sources if args.links_from else None,
        "links_reused": reused or None,
        "scib_runs": {g: m.get("runs") for g, m in scib_meta.items()},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
