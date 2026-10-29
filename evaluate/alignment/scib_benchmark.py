"""scIB integration benchmark of several models' embeddings across modalities.

Reads ``embeddings/<model>/<modality>_<part>.h5ad`` (``embed/embed_splits.py``) for each
model, keeps the rows every model embedded, filters and subsamples them ONCE (so every
model is scored on the same rows), and runs ``scib_metrics``' Benchmarker with the
modality as batch and a biological label (e.g. tissue) as label.

    python -m evaluate.alignment.scib_benchmark --name tissue_s0_test \\
        --models unified_v1 bulkformer --pre-integrated pca \\
        --modalities bulk pseudobulk --per-modality bulk=all pseudobulk=all --label-key tissue_general

Writes ``results/assessment/scib_<name>/``: ``scib.csv`` (``get_results(min_max_scale=False,
clean_names=True)``, the trailing ``Metric Type`` row included), ``scib_minmax.csv``, the
results table figure, ``label_batch.csv`` (the pre-flight crosstab), ``meta.json`` and
``manifest.json`` listing the models.

Reading the numbers:

- **Batch correction** here means *modality mixing*: bras, iLISI and graph connectivity
  (kBET with ``--kbet``; slow). BRAS and kBET work inside each label and skip labels
  present in one modality only; the pre-flight report says how many labels qualify, and
  with fewer than 2 they measure almost nothing: rank on iLISI then.
- **PCR comparison** needs a non-integrated reference embedding: ``--pre-integrated
  pca`` (the reserved PCA pseudo-model) enables it, and that model is then the reference
  rather than a row. Without it PCR is off.
- BRAS uses cosine distance while iLISI and the UMAPs use Euclidean, so they can disagree.

``scib_metrics`` lives in the ``bulkFM`` conda env, not the bionemo container
(``scib_benchmark.sh`` activates it). ``--dry-run`` stops after the pre-flight report and
needs no scib_metrics.
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

from evaluate._cli import parse_quotas, parse_with_config  # noqa: E402
from evaluate.io import EMB_KEY, load_embedding, panel_hashes  # noqa: E402
from evaluate.modalities import MOD_BULK, MOD_PB  # noqa: E402
from experiments.layout import ExperimentLayout, canonical_modality, validate_model_name  # noqa: E402
from utils.sampling import filter_mask, parse_filter_specs, stratified_subsample  # noqa: E402

UNKNOWN_LABEL = "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="scIB integration benchmark across modalities.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--name", default=None, help="Analysis name -> results/assessment/scib_<name>/")
    parser.add_argument("--models", nargs="+", default=None, help="Models to benchmark (embeddings/<model>/)")
    parser.add_argument("--pre-integrated", default=None,
                        help="Model used as the non-integrated reference (e.g. pca); enables PCR comparison")
    parser.add_argument("--part", default="test", choices=["test", "train_subset"])
    parser.add_argument("--modalities", nargs="+", default=[MOD_BULK, MOD_PB])
    parser.add_argument("--per-modality", nargs="*", default=[], metavar="MOD=N|all",
                        help="Rows per modality, stratified by the label (default: all)")
    parser.add_argument("--batch-key", default="modality", help="obs column used as batch")
    parser.add_argument("--label-key", default="tissue_general", help="obs column used as biological label")
    parser.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2")
    parser.add_argument("--min-label-cells", type=int, default=10,
                        help="Cells a label needs (with >= 2 batches) to count as usable by BRAS/kBET")
    parser.add_argument("--kbet", action="store_true", help="Also run kBET per label (slow)")
    parser.add_argument("--leiden", action="store_true", help="Also run Leiden NMI/ARI (slow)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Pre-flight report only; no scib_metrics needed")
    return parser


def load_models(layout: ExperimentLayout, models: list[str], modalities: list[str], part: str):
    """``(obs, {model: embedding})`` on the rows every model has, row names ``<mod>:<obs name>``."""
    obs_parts, embs, hashes = [], {m: [] for m in models}, {}
    for mod in modalities:
        loaded = {m: load_embedding(layout, m, mod, part) for m in models}
        names = None
        for m, a in loaded.items():
            names = set(a.obs_names) if names is None else names & set(a.obs_names)
        for m, a in loaded.items():
            if a.n_obs != len(names):
                print(f"  [warn] {m}/{mod}_{part}: {a.n_obs - len(names)} rows not embedded by every model, dropped")
        if not names:
            raise ValueError(f"No {mod} row is embedded by every model")
        order = [n for n in loaded[models[0]].obs_names if n in names]
        obs = loaded[models[0]].obs.loc[order].copy()
        obs.index = [f"{mod}:{n}" for n in order]
        obs_parts.append(obs)
        for m, a in loaded.items():
            embs[m].append(np.asarray(a[order].obsm[EMB_KEY], dtype=np.float32))
        for m in models:
            hashes.setdefault(m, {}).update(panel_hashes(layout, m, [layout.embedding_file(m, mod, part)]))
    obs = pd.concat(obs_parts)
    return obs, {m: np.concatenate(v) for m, v in embs.items()}, hashes


def select_rows(obs: pd.DataFrame, args, modalities: list[str]) -> np.ndarray:
    """Filter, then a label-stratified subsample per batch value (modality); boolean mask."""
    filters = parse_filter_specs(args.filter)
    keep = filter_mask(obs, filters, label="embeddings") if filters else np.ones(len(obs), bool)
    quotas = parse_quotas(args.per_modality, modalities)
    chosen = set()
    for mod in modalities:
        sub = obs[keep & (obs[args.batch_key].astype(str) == mod).to_numpy()] if args.batch_key in obs else obs[keep]
        strata = [args.label_key] if args.label_key in sub.columns else []
        chosen |= set(stratified_subsample(sub, quotas.get(mod), strata, seed=args.seed))
    return obs.index.isin(list(chosen))


def label_batch_report(obs: pd.DataFrame, batch_key: str, label_key: str, min_cells: int) -> tuple[pd.DataFrame, dict]:
    """Label x batch crosstab and how many labels BRAS / kBET can use."""
    table = pd.crosstab(obs[label_key], obs[batch_key])
    n_batches = (table > 0).sum(axis=1)
    usable = (n_batches >= 2) & (table.sum(axis=1) >= min_cells)
    summary = {"n_labels": int(len(table)), "n_labels_usable": int(usable.sum()),
               "usable_labels": sorted(map(str, table.index[usable])),
               "batch_counts": {str(k): int(v) for k, v in obs[batch_key].value_counts().items()}}
    print(f"  {summary['n_labels']} labels; {summary['n_labels_usable']} present in >= 2 batches with "
          f">= {min_cells} cells (usable by BRAS / kBET)")
    if summary["n_labels_usable"] < 2:
        print("  [WARNING] fewer than 2 usable labels: BRAS and kBET measure almost nothing; rank on iLISI")
    return table, summary


def run_benchmark(adata, args, row_keys: list[str], pre_key: str | None):
    try:
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    except ImportError:
        sys.exit("Error: scib_metrics is not installed here; it lives in the bulkFM conda env "
                 "(scib_benchmark.sh activates it). --dry-run works without it.")
    bm = Benchmarker(
        adata, batch_key=args.batch_key, label_key=args.label_key, embedding_obsm_keys=row_keys,
        bio_conservation_metrics=BioConservation(
            isolated_labels=True, silhouette_label=True, clisi_knn=True,
            nmi_ari_cluster_labels_kmeans=True, nmi_ari_cluster_labels_leiden=args.leiden),
        batch_correction_metrics=BatchCorrection(
            bras=True, ilisi_knn=True, graph_connectivity=True, kbet_per_label=args.kbet,
            pcr_comparison=args.pre_integrated is not None),
        # Benchmarker computes a PCA of X when no reference is given; these files hold no
        # X, so without --pre-integrated the first row stands in (PCR is off then).
        pre_integrated_embedding_obsm_key=pre_key or row_keys[0],
        n_jobs=args.n_jobs,
    )
    bm.benchmark()
    return bm


def main(argv: list[str] | None = None) -> Path:
    import anndata as ad

    args = parse_with_config(build_parser(), "scib", argv)
    if not (args.models and args.name):
        sys.exit("Error: give --models and --name")
    layout = ExperimentLayout.from_root(args.root)
    rows = [validate_model_name(m) for m in args.models if m != args.pre_integrated]
    if args.pre_integrated in args.models:
        print(f"  '{args.pre_integrated}' is the pre-integrated reference, so it is not also a row")
    if not rows:
        sys.exit("Error: no model left to benchmark")
    models = rows + ([args.pre_integrated] if args.pre_integrated else [])
    modalities = [canonical_modality(m) for m in args.modalities]

    print(f"scIB '{args.name}': {rows} on {modalities} ({args.part})"
          + (f", reference {args.pre_integrated}" if args.pre_integrated else ""))
    obs, embs, hashes = load_models(layout, models, modalities, args.part)
    for key in (args.batch_key, args.label_key):
        if key not in obs.columns:
            sys.exit(f"Error: obs has no '{key}' column (columns: {list(obs.columns)[:15]})")
    mask = select_rows(obs, args, modalities)
    obs = obs[mask].copy()
    obs[args.label_key] = obs[args.label_key].astype(object).where(obs[args.label_key].notna(), UNKNOWN_LABEL).astype(str)
    obs[args.batch_key] = obs[args.batch_key].astype(str)
    print(f"  {len(obs)} rows: " + ", ".join(f"{k}={v}" for k, v in obs[args.batch_key].value_counts().items()))
    if obs[args.batch_key].nunique() < 2:
        sys.exit(f"Error: only one '{args.batch_key}' value left; nothing to integrate")
    distinct = {tuple(sorted(set(h.values()) - {None})) for h in hashes.values()}
    if len(distinct) > 1:
        print("  [note] the models used different gene panels (expected across models)")

    out_dir = layout.assessment_dir(f"scib_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table, summary = label_batch_report(obs, args.batch_key, args.label_key, args.min_label_cells)
    table.to_csv(out_dir / "label_batch.csv")
    meta = {
        "name": args.name, "models": rows, "pre_integrated": args.pre_integrated, "part": args.part,
        "modalities": modalities, "batch_key": args.batch_key, "label_key": args.label_key,
        "filters": parse_filter_specs(args.filter), "per_modality": args.per_modality, "seed": args.seed,
        "n_rows": int(len(obs)), "kbet": args.kbet, "leiden": args.leiden, "panel_hashes": hashes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **summary,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"analysis": "scib", "models": models, "rows": rows, "pre_integrated": args.pre_integrated},
        indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"  dry run: pre-flight report written to {out_dir}")
        return out_dir

    adata = ad.AnnData(obs=obs[[args.batch_key, args.label_key]].copy())
    keep = mask.nonzero()[0]
    for m in models:
        adata.obsm[m] = embs[m][keep]
    bm = run_benchmark(adata, args, rows, args.pre_integrated)
    bm.get_results(min_max_scale=False, clean_names=True).to_csv(out_dir / "scib.csv")
    bm.get_results(min_max_scale=True, clean_names=True).to_csv(out_dir / "scib_minmax.csv")
    try:
        tab = bm.plot_results_table(min_max_scale=False, show=False, save_dir=str(out_dir))
        tab.ax.get_figure().savefig(out_dir / "scib_results.png", dpi=200, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001 - the CSVs are the result; the figure is a convenience
        print(f"  [warn] results table figure failed ({type(exc).__name__}: {exc})")
    print(bm.get_results(min_max_scale=False, clean_names=True))
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
