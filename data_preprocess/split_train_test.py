"""Split the rows of an h5ad into named parts (train/test, or any others).

Writes ``<out-dir>/<part>.txt`` (one obs name per line) and ``<out-dir>/split.json``.
The source file is only read. Rows are assigned so that:

- each part gets its fraction of every stratum (``--stratify-by`` columns, plus
  Leiden clusters with ``--cluster leiden``);
- no group straddles parts (``--group-by``; columns linked through shared rows, e.g.
  a pair spanning two donors, are merged into one unit);
- rows linked to an existing split (``--inherit-from``) land in their parent's part.

Any h5ad, explicit paths:

    python split_train_test.py --input data.h5ad --out-dir splits/s0 \\
        --stratify-by tissue_general --group-by donor_id paired:0 --fractions train=0.8 test=0.2

Experiments layout shorthand (input ``data/<modality>/<modality>.h5ad``, output
``data/<modality>/splits/<split>/``):

    python split_train_test.py --modality single_cell --split-name tissue_s0 \\
        --stratify-by tissue_general --group-by donor_id paired:0
    python split_train_test.py --modality single_cell --split-name leiden_s0 --cluster leiden
    python split_train_test.py --modality pseudobulk --split-name tissue_s0 \\
        --inherit-from single_cell:tissue_s0 --link-col pseudobulk_id --build-store

``--group-by COL:V1,V2`` names the values of COL that mean "no group" besides
empty/NaN, e.g. ``paired:0`` where 0 is "unpaired". ``--build-store`` converts one
part (default ``train``) straight from the source into a training store, see
``bulk_sc_data_preprocessing.build_pipeline_ready``. Expression values are never
normalized; ``--cluster-transform`` only shapes the features Leiden clusters on.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from data_preprocess/ like the other scripts; the project root is appended so
# the sibling modules (bulk_sc_data_preprocessing, h5ad_io) still resolve first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import hashlib
import json
import shutil
from argparse import ArgumentParser
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from h5ad_io import SourceReader
from utils.sampling import (
    filter_mask,
    parse_filter_specs,
    parse_fractions,
    parse_group_specs,
    split_summary,
    stratified_group_split,
    stratum_labels,
)

CLUSTER_COL = "leiden"
NO_LINK = {"", "0", "nan", "none", "unknown"}
SPLIT_CARD = "split.json"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def source_fingerprint(path: Path, obs_names) -> dict:
    """Size, mtime and a hash of the obs names: enough to notice the source changed."""
    stat = path.stat()
    digest = hashlib.sha1()
    for name in obs_names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "obs_names_hash": digest.hexdigest()[:16],
    }


def read_parts(split_dir: Path) -> dict[str, list[str]]:
    """``{part: obs names}`` of an existing split directory."""
    card = split_dir / SPLIT_CARD
    if card.exists():
        names = json.loads(card.read_text(encoding="utf-8")).get("parts") or []
    else:
        names = [p.stem for p in sorted(split_dir.glob("*.txt"))]
    if not names:
        raise FileNotFoundError(f"No split parts in {split_dir}")
    return {part: (split_dir / f"{part}.txt").read_text(encoding="utf-8").split() for part in names}


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def _transform(X, how: str):
    """Features for clustering only; the stored data is never transformed."""
    import scipy.sparse as sp

    X = X.astype(np.float32)
    if how == "none":
        return X
    if how == "cpm_log1p":
        totals = np.asarray(X.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        X = sp.diags(1e6 / totals) @ X if sp.issparse(X) else X * (1e6 / totals)[:, None]
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.log1p(X.data)
        return X
    return np.log1p(X)


def _leiden(rep: np.ndarray, n_neighbors: int, resolution: float, seed: int) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(rep))]),
                     obsm={"X_rep": np.asarray(rep, dtype=np.float32)})
    sc.pp.neighbors(tmp, use_rep="X_rep", n_neighbors=n_neighbors, random_state=seed)
    try:
        sc.tl.leiden(tmp, resolution=resolution, random_state=seed, key_added=CLUSTER_COL,
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:  # older scanpy without the igraph flavor
        sc.tl.leiden(tmp, resolution=resolution, random_state=seed, key_added=CLUSTER_COL)
    return tmp.obs[CLUSTER_COL].astype(str).to_numpy()


def _nearest_centroid(rep: np.ndarray, fit_rep: np.ndarray, fit_labels: np.ndarray) -> np.ndarray:
    clusters = np.unique(fit_labels)
    centroids = np.stack([fit_rep[fit_labels == c].mean(axis=0) for c in clusters])
    out = np.empty(len(rep), dtype=object)
    for start in range(0, len(rep), 50_000):
        block = rep[start:start + 50_000]
        dist = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        out[start:start + 50_000] = clusters[dist.argmin(axis=1)]
    return out


def cluster_rows(reader: SourceReader, positions: np.ndarray, args) -> tuple[np.ndarray, dict]:
    """Leiden label for each row at ``positions``.

    With ``--cluster-rep`` the clusters are computed on that obsm representation.
    Otherwise: HVGs, ``--cluster-transform``, PCA. Either way they are fitted on at
    most ``--cluster-max-rows`` rows and the rest get the nearest cluster centroid.
    """
    rng = np.random.default_rng(args.seed)
    n = len(positions)
    fit_local = np.arange(n)
    if args.cluster_max_rows is not None and n > args.cluster_max_rows:
        fit_local = np.sort(rng.choice(n, size=args.cluster_max_rows, replace=False))
    rest_local = np.setdiff1d(np.arange(n), fit_local)
    print(f"  [cluster] fitting on {len(fit_local)} of {n} rows", flush=True)
    info: dict = {"method": "leiden", "resolution": args.leiden_resolution,
                  "n_neighbors": args.n_neighbors, "fit_rows": int(len(fit_local))}

    if args.cluster_rep:
        rep = reader.obsm(args.cluster_rep)[positions]
        fit_labels = _leiden(rep[fit_local], args.n_neighbors, args.leiden_resolution, args.seed)
        labels = np.empty(n, dtype=object)
        labels[fit_local] = fit_labels
        if len(rest_local):
            labels[rest_local] = _nearest_centroid(rep[rest_local], rep[fit_local], fit_labels)
        info.update({"representation": f"obsm[{args.cluster_rep!r}]"})
    else:
        import scanpy as sc

        fit = reader.subset(positions[fit_local])
        n_top = min(args.n_hvg, fit.n_vars)

        # Use CPM_log1p normalization with Seurat flavor by default to account for depth variance
        tmp = fit.copy()
        tmp.X = _transform(tmp.X, args.cluster_transform)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")
        fit.var["highly_variable"] = tmp.var["highly_variable"].to_numpy()
        hvg_flavor = "seurat"
        hvg = fit.var["highly_variable"].to_numpy()
        fit.X = _transform(fit.X, args.cluster_transform)  # library size over all genes, then subset
        fit = fit[:, hvg].copy()

        # Compute PCs
        n_comps = min(args.n_pcs, fit.n_vars - 1, fit.n_obs - 1)
        sc.tl.pca(fit, n_comps=n_comps, zero_center=False)
        fit_rep = fit.obsm["X_pca"]

        # Leiden clustering and report
        fit_labels = _leiden(fit_rep, args.n_neighbors, args.leiden_resolution, args.seed)
        labels = np.empty(n, dtype=object)
        labels[fit_local] = fit_labels
        pcs = fit.varm["PCs"]
        for start in range(0, len(rest_local), 50_000):
            idx = rest_local[start:start + 50_000]
            proj = np.asarray(_transform(reader.rows(positions[idx]), args.cluster_transform)[:, hvg] @ pcs)
            labels[idx] = _nearest_centroid(proj, fit_rep, fit_labels)
        info.update({"representation": "pca", "transform": args.cluster_transform, "n_hvg": int(hvg.sum()),
                     "hvg_flavor": hvg_flavor, "n_pcs": int(n_comps)})
    info.update({"assigned_by_centroid": int(len(rest_local)), "n_clusters": int(len(np.unique(fit_labels)))})
    print(f"  [cluster] {info['n_clusters']} clusters", flush=True)
    return labels, info


# --------------------------------------------------------------------------- #
# Inheritance through a link column
# --------------------------------------------------------------------------- #


def _link_values(series: pd.Series) -> pd.Series:
    """Link keys as strings, with 'no link' values (0, NaN, '') as NaN."""
    values = series.astype(str).str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    values = values.where(~(numeric == 0), "")
    values = values.str.replace(r"\.0$", "", regex=True)
    return values.where(~values.str.lower().isin(NO_LINK))


def inherit_split(child_obs: pd.DataFrame, parent_obs: pd.DataFrame, parent_parts: dict[str, list[str]],
                  link_col: str, parent_link_col: str) -> pd.Series:
    """Part per child row from the parent's split; NaN when the row is not linked."""
    if link_col not in child_obs.columns:
        raise ValueError(f"--link-col {link_col!r} not in this file's obs")
    if parent_link_col not in parent_obs.columns:
        raise ValueError(f"--parent-link-col {parent_link_col!r} not in the parent's obs")
    side = pd.Series(np.nan, index=parent_obs.index, dtype=object)
    for part, names in parent_parts.items():
        side[side.index.isin(names)] = part
    parent = pd.DataFrame({"key": _link_values(parent_obs[parent_link_col]), "side": side}).dropna()
    per_key = parent.groupby("key")["side"].agg(lambda s: s.value_counts().index[0])
    n_mixed = int((parent.groupby("key")["side"].nunique() > 1).sum())
    if n_mixed:
        print(f"  [warn] {n_mixed} link keys sit in several parts of the parent split; using the majority part")
    return _link_values(child_obs[link_col]).map(per_key)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def resolve_paths(args):
    """``(input, out_dir, layout, modality)`` from explicit paths or the layout shorthand."""
    layout = modality = None
    if args.modality or args.split_name:
        from experiments.layout import ExperimentLayout, canonical_modality

        if not (args.modality and args.split_name):
            raise SystemExit("Error: the layout shorthand needs both --modality and --split-name")
        layout = ExperimentLayout.from_root(args.root)
        modality = canonical_modality(args.modality)
    source = args.input or (layout.raw_h5ad(modality) if layout else None)
    out_dir = args.out_dir or (layout.split_dir(modality, args.split_name) if layout else None)
    if source is None or out_dir is None:
        raise SystemExit("Error: give --input and --out-dir, or --modality and --split-name")
    return Path(source), Path(out_dir), layout, modality


def resolve_parent(args, layout):
    """``(parent h5ad, parent split dir)`` for ``--inherit-from``."""
    spec = args.inherit_from
    if Path(spec).is_dir():
        if not args.parent_input:
            raise SystemExit("Error: --inherit-from DIR needs --parent-input (the parent's h5ad)")
        return Path(args.parent_input), Path(spec)
    mod, sep, split = spec.partition(":")
    if not sep or not split:
        raise SystemExit("Error: --inherit-from must be a split directory or <modality>:<split>")
    if layout is None:
        from experiments.layout import ExperimentLayout

        layout = ExperimentLayout.from_root(args.root)
    return Path(args.parent_input or layout.raw_h5ad(mod)), layout.split_dir(mod, split)


def main(args):
    source, out_dir, layout, modality = resolve_paths(args)
    fractions = parse_fractions(args.fractions) if args.fractions else parse_fractions(args.test_size)
    parts = list(fractions)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite to replace it")
    if not source.exists():
        raise FileNotFoundError(f"Source h5ad not found: {source}")

    print(f"Splitting {source} into {fractions}")
    with SourceReader(source) as reader:
        all_names = reader.obs.index
        obs = reader.obs.copy()
        if not obs.index.is_unique:
            raise ValueError(f"{source} has duplicate obs names; the part lists would be ambiguous")
        print(f"  {reader.n_obs} rows x {len(reader.var)} genes")

        filters = parse_filter_specs(args.filter)
        selected = filter_mask(obs, filters, label=source.name) if filters else np.ones(len(obs), dtype=bool)
        positions = np.flatnonzero(selected)
        obs = obs.iloc[positions]

        strata = list(args.stratify_by or [])
        cluster_info: dict = {}
        if args.cluster == "leiden":
            labels, cluster_info = cluster_rows(reader, positions, args)
            obs[CLUSTER_COL] = labels
            strata = [CLUSTER_COL] + strata

    group_specs = parse_group_specs(args.group_by)
    assignment = pd.Series(np.nan, index=obs.index, dtype=object)
    inherited = 0
    if args.inherit_from:
        parent_input, parent_dir = resolve_parent(args, layout)
        link_col = args.link_col
        if not link_col:
            raise SystemExit("Error: --inherit-from needs --link-col")
        with SourceReader(parent_input) as parent:
            parent_obs = parent.obs
        parent_parts = read_parts(parent_dir)
        unknown = sorted(set(parent_parts) - set(parts))
        if unknown:
            raise SystemExit(f"Error: the parent split has parts {unknown} that this split does not ({parts})")
        side = inherit_split(obs, parent_obs, parent_parts, link_col, args.parent_link_col or link_col)
        assignment[side.notna()] = side[side.notna()]
        inherited = int(side.notna().sum())
        print(f"  {inherited} rows inherit their part from {parent_dir} via '{link_col}'")

    todo = assignment.isna()
    if todo.any():
        rest = obs.loc[todo]
        if len(rest) < len(parts):
            assignment[todo] = parts[0]
        else:
            assignment[todo] = stratified_group_split(rest, strata, fractions, seed=args.seed,
                                                      group_cols=group_specs)

    table = split_summary(assignment, stratum_labels(obs, strata), parts)
    print(table.to_string())
    counts = {part: int((assignment == part).sum()) for part in parts}
    print("  " + "  ".join(f"{p}={c}" for p, c in counts.items()))
    empty = [p for p, c in counts.items() if not c]
    if empty:
        raise SystemExit(
            f"Refusing to write the split: part(s) {empty} are empty. Usually one linked unit "
            "(see the warning above) or a parent split with nothing in those parts."
        )

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        names = obs.index[(assignment == part).to_numpy()]
        (out_dir / f"{part}.txt").write_text("\n".join(map(str, names)) + "\n", encoding="utf-8")
    if args.cluster == "leiden":
        obs[[CLUSTER_COL]].to_csv(out_dir / "clusters.csv")

    card = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_fingerprint(source, all_names),
        "parts": parts,
        "fractions": fractions,
        "counts": counts,
        "filters": filters,
        "rows_selected": int(len(obs)),
        "stratify_by": strata,
        "group_by": list(args.group_by or []),
        "cluster": cluster_info or None,
        "inherited_from": args.inherit_from,
        "link_col": args.link_col if args.inherit_from else None,
        "n_inherited": inherited,
        "seed": args.seed,
        "per_stratum": table.reset_index().to_dict(orient="records"),
    }
    if layout is not None:
        card.update({"modality": modality, "split_name": args.split_name})
    (out_dir / SPLIT_CARD).write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}")

    if args.write_h5ad:
        with SourceReader(source) as reader:
            for part in parts:
                keep = np.flatnonzero(reader.obs.index.isin(obs.index[(assignment == part).to_numpy()]))
                path = out_dir / f"{part}.h5ad"
                print(f"  writing {path}")
                reader.subset(keep).write_h5ad(path)

    if args.build_store:
        rows = obs.index[(assignment == args.store_part).to_numpy()]
        build_store_from_split(args, source, out_dir, layout, modality, rows)
    print("Done.")


def build_store_from_split(args, source, out_dir, layout, modality, rows) -> None:
    # Heavy (bionemo, torch): only imported when a store is actually built.
    from bulk_sc_data_preprocessing import Source, build_pipeline_ready

    store = args.store_dir or (layout.pipeline_ready(modality, args.split_name) if layout else None)
    if store is None:
        raise SystemExit("Error: --build-store needs --store-dir (or the layout shorthand)")
    store = Path(store)
    if store.exists() and any(store.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{store} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(store)
    print(f"Building the training store {store} from the '{args.store_part}' part")
    summary = build_pipeline_ready(
        [Source(path=source, rows=list(rows), modality=args.store_modality or modality)],
        store,
        obs_columns=args.obs_columns,
        vocab_path=args.vocab_path,
        gene_column=args.gene_column,
        pb_id_column=args.pb_id_column,
    )
    summary.update({"split": str(out_dir), "part": args.store_part})
    (store / "store.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def get_args():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    io = parser.add_argument_group("input and output")
    io.add_argument("--input", type=Path, default=None, help="Source h5ad")
    io.add_argument("--out-dir", type=Path, default=None, help="Directory for <part>.txt and split.json")
    io.add_argument("--modality", default=None, help="Layout shorthand: single_cell | bulk | pseudobulk")
    io.add_argument("--split-name", default=None, help="Layout shorthand: splits/<name>/")
    io.add_argument("--root", type=Path, default=None, help="Experiments root for the layout shorthand")
    io.add_argument("--overwrite", action="store_true", help="Replace an existing split (and store)")

    parts = parser.add_argument_group("parts")
    parts.add_argument("--fractions", nargs="+", default=None, metavar="PART=FRAC",
                       help="Named parts and their fractions, e.g. train=0.7 val=0.1 test=0.2")
    parts.add_argument("--test-size", type=float, default=0.2,
                       help="Shorthand for --fractions train=1-x test=x")
    parts.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2",
                       help="Only split these rows; the others are left out of every part")
    parts.add_argument("--seed", type=int, default=0)

    strata = parser.add_argument_group("stratification and groups")
    strata.add_argument("--stratify-by", nargs="*", default=[], help="obs columns to stratify on")
    strata.add_argument("--group-by", nargs="*", default=[], metavar="COL[:NOGROUP,...]",
                        help="obs columns whose groups never straddle parts; COL:0 marks 0 as 'no group'")
    strata.add_argument("--cluster", choices=["none", "leiden"], default="none",
                        help="Also stratify on Leiden clusters")
    strata.add_argument("--cluster-rep", default=None,
                        help="obsm key to cluster on (e.g. X_pca) instead of computing one from X")
    strata.add_argument("--cluster-transform", choices=["cpm_log1p", "log1p", "none"], default="cpm_log1p",
                        help="Transform of X for the clustering features only")
    strata.add_argument("--leiden-resolution", type=float, default=1.0)
    strata.add_argument("--n-hvg", type=int, default=2000)
    strata.add_argument("--n-pcs", type=int, default=50)
    strata.add_argument("--n-neighbors", type=int, default=15)
    strata.add_argument("--cluster-max-rows", type=int, default=200_000,
                        help="Fit clusters on at most this many rows; assign the rest by nearest centroid")

    inherit = parser.add_argument_group("inheriting another split")
    inherit.add_argument("--inherit-from", default=None, metavar="DIR|MODALITY:SPLIT",
                         help="Existing split whose parts linked rows inherit")
    inherit.add_argument("--parent-input", type=Path, default=None, help="The parent split's h5ad")
    inherit.add_argument("--link-col", default=None, help="obs column linking rows to the parent's rows")
    inherit.add_argument("--parent-link-col", default=None, help="Link column in the parent, if named differently")

    out = parser.add_argument_group("extra outputs")
    out.add_argument("--write-h5ad", action="store_true", help="Also write <part>.h5ad for every part")
    out.add_argument("--build-store", action="store_true",
                     help="Build a training store from one part, straight from the source")
    out.add_argument("--store-part", default="train", help="Part the store is built from")
    out.add_argument("--store-dir", type=Path, default=None,
                     help="Store directory (layout default: data/<modality>/pipeline_ready/<split>/)")
    out.add_argument("--store-modality", default=None,
                     help="Modality label for rows without a 'modality' obs column (layout: the modality)")
    out.add_argument("--obs-columns", nargs="+", default=None, help="obs columns kept in the store")
    out.add_argument("--vocab-path", type=Path, default=None, help="Existing vocab.json for the store")
    out.add_argument("--gene-column", default=None, help="var column holding the gene names")
    out.add_argument("--pb-id-column", default=None, help="Pseudobulk -> cell link column kept in the store")
    args = parser.parse_args()
    if args.build_store and args.fractions and args.store_part not in [f.split("=")[0] for f in args.fractions]:
        parser.error(f"--store-part {args.store_part!r} is not one of the --fractions parts")
    return args


if __name__ == "__main__":
    main(get_args())
