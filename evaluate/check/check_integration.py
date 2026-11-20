"""Self-check for ``evaluate/integration/scib_benchmark.py`` on synthetic embeddings.

Covers loading several models on their shared rows, the filter and the label-stratified
per-modality quotas, the label x batch pre-flight report, the dry-run outputs and the
results table figure with custom row labels (``--plot-only`` on a synthetic results CSV;
skipped without ``plottable``). The Benchmarker itself runs only where ``scib_metrics``
imports (the bulkFM env); otherwise it is skipped.

Also covers ``integration.py``: the link metrics on planted, shuffled and constant
embeddings (numpy only), and an end-to-end run on fake sc / pseudobulk / bulk files with
a fake scIB analysis (needs anndata).

    python -m evaluate.check.check_integration
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises  # noqa: E402
from experiments.layout import ExperimentLayout  # noqa: E402
from evaluate.integration import integration, scib_benchmark  # noqa: E402

SIZES = {"bulk": 80, "pseudobulk": 120}


def _write_root(root: Path) -> ExperimentLayout:
    import anndata as ad

    layout = ExperimentLayout(root)
    rng = np.random.default_rng(0)
    for model, drop in (("m1", 0), ("m2", 5), ("pca", 0)):
        out = layout.embedding_dir(model)
        out.mkdir(parents=True)
        files = {}
        for mod, n in SIZES.items():
            # Tissue "skin" exists only in bulk, so it is not usable by per-label metrics.
            tissues = np.where(np.arange(n) % 4 == 0, "skin" if mod == "bulk" else "lung",
                               np.where(np.arange(n) % 2 == 0, "lung", "colon"))
            obs = pd.DataFrame({"modality": mod, "tissue_general": tissues}, index=[f"{mod}{i}" for i in range(n)])
            obs = obs.iloc[drop:]  # m2 did not embed the first rows of each modality
            emb = rng.normal(size=(len(obs), 8)).astype(np.float32)
            name = f"{mod}_test.h5ad"
            ad.AnnData(obs=obs, obsm={"X_emb": emb}).write_h5ad(out / name)
            files[name] = {"panel_hash": f"hash_{model}"}
        (out / "embed.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return layout


def check_link_metrics() -> None:
    print("\n-- link metrics (integration.py) --")
    rng = np.random.default_rng(1)
    keys = pd.Index([f"k{i}" for i in range(40)])
    tissue = pd.Series(np.array(["lung", "colon", "liver", "skin"])[np.arange(40) % 4], index=keys)
    b = rng.normal(size=(40, 16))
    good = integration.match_edge(keys, b + 0.01 * rng.normal(size=b.shape), keys, b, tissue)
    check("planted links: top-1 = 1, rank = 0, rank in tissue = 0",
          good["top1"] == 1.0 and good["rank"] == 0.0 and good["rank_tissue"] == 0.0
          and good["n_queries"] == 40 and good["n_tissue_queries"] == 40)
    shuffled = [integration.match_edge(keys, rng.normal(size=b.shape), keys, b, tissue)["rank"] for _ in range(20)]
    check("independent embeddings: rank near chance (0.5)", abs(np.mean(shuffled) - 0.5) < 0.05)
    flat = integration.match_edge(keys, np.ones((40, 4)), keys, np.ones((40, 4)), tissue)
    check("constant embedding sits at chance, not at the top", flat["rank"] == 0.5 and flat["rank_tissue"] == 0.5)
    half = integration.match_edge(keys[:10], b[:10], keys, b, tissue.where(np.arange(40) >= 5))
    check("queries without a tissue left out of rank in tissue only",
          half["n_queries"] == 10 and half["n_tissue_queries"] == 5)
    none = integration.match_edge(pd.Index(["x"]), b[:1], keys, b, tissue)
    check("no shared key: n/a", none["n_queries"] == 0 and np.isnan(none["rank"]))

    emb = np.arange(10, dtype=float)[:, None] * np.ones((1, 3))
    link = pd.Series(["a"] + ["b"] * 3 + ["c"] * 6, dtype=object)
    labels = pd.Series(["lung"] * 4 + [np.nan, "colon", "colon", "liver", np.nan, np.nan], dtype=object)
    frame, vec = integration.link_centroids(emb, link, labels, min_count=5)
    check("min count drops keys", list(frame.index) == ["c"] and int(frame.loc["c", "n"]) == 6)
    check("key vector is the mean of its rows", np.allclose(vec[0], emb[4:].mean(0)))
    check("key tissue is its most common known label", frame.loc["c", "tissue"] == "colon")
    frame, _ = integration.link_centroids(emb, pd.Series([np.nan] * 10, dtype=object), labels)
    check("no linked row: empty", frame.empty)

    obs = pd.DataFrame({"tissue_general": ["lung", "unknown", "colon", "cell line"],
                        "tissue_label_source": ["cellxgene", "none", "keyword", "archs4_culture"]})
    cur = integration.curated_labels(obs, "tissue_general", {"tissue_label_source": ["cellxgene", "archs4"]}, "x")
    check("only curated, known labels count", cur.tolist()[0] == "lung" and cur.iloc[1:].isna().all())
    cur = integration.curated_labels(obs.drop(columns="tissue_label_source"), "tissue_general",
                                     {"tissue_label_source": ["cellxgene"]}, "x")
    check("missing filter column: labels used unfiltered (unknown still dropped)",
          cur.notna().tolist() == [True, False, True, True])

    pb = pd.DataFrame({"n": [1, 1], "tissue": ["lung", np.nan]}, index=["p1", "p2"])
    sc = pd.DataFrame({"n": [9, 9], "tissue": ["colon", "liver"]}, index=["p1", "p2"])
    bulk = pd.DataFrame({"n": [1, 1, 1], "tissue": [np.nan, np.nan, "skin"]}, index=["p1", "p2", "p3"])
    t = integration.key_tissues({"bulk": (bulk, None), "sc": (sc, None), "pseudobulk": (pb, None)})
    check("key tissue: pseudobulk first, then cells, then bulk",
          t["p1"] == "lung" and t["p2"] == "liver" and t["p3"] == "skin")


def check_integration_run(tmp: str) -> None:
    import anndata as ad

    print("\n-- integration.py end to end --")
    layout = ExperimentLayout(Path(tmp))
    rng = np.random.default_rng(2)
    n_pb = 24
    tissues = np.array(["lung", "colon", "liver"])[np.arange(n_pb) % 3]
    pb_vec = rng.normal(size=(n_pb, 8))
    pair = np.where(np.arange(n_pb) < 12, np.arange(n_pb) + 1, 0)
    for model, linked in (("good", True), ("bad", False)):
        out = layout.embedding_dir(model)
        out.mkdir(parents=True)
        cell_pb = np.repeat(np.arange(n_pb), 6)
        noise = rng.normal(size=(len(cell_pb), 8))
        cell_vec = pb_vec[cell_pb] + 0.05 * noise if linked else noise
        bulk_vec = pb_vec[:12] + 0.05 * rng.normal(size=(12, 8)) if linked else rng.normal(size=(12, 8))
        tables = {
            "sc": (pd.DataFrame({"pseudobulk_id": [f"pb:{i}" for i in cell_pb], "paired": pair[cell_pb],
                                 "tissue_general": tissues[cell_pb], "tissue_label_source": "cellxgene"}),
                   cell_vec),
            "pseudobulk": (pd.DataFrame({"pseudobulk_id": [f"pb:{i}" for i in range(n_pb)], "paired": pair,
                                         "tissue_general": tissues, "tissue_label_source": "cellxgene"}),
                           pb_vec if linked else rng.normal(size=pb_vec.shape)),
            # paired bulk rows lost their tissue in curate: they take their pair's
            "bulk": (pd.DataFrame({"paired": np.arange(12) + 1, "tissue_general": "unknown",
                                   "tissue_label_source": "none"}), bulk_vec),
        }
        for mod, (obs, emb) in tables.items():
            obs.index = [f"{mod}{i}" for i in range(len(obs))]
            ad.AnnData(obs=obs, obsm={"X_emb": emb.astype(np.float32)}).write_h5ad(out / f"{mod}_test.h5ad")
    scib_dir = layout.scib_dir("fake_all")
    scib_dir.mkdir(parents=True)
    _fake_results(scib_dir)
    for f in ("scib.csv", "scib_minmax.csv"):  # rows named after this run's models
        text = (scib_dir / f).read_text(encoding="utf-8")
        (scib_dir / f).write_text(text.replace("m1,", "good,").replace("m2,", "bad,"), encoding="utf-8")
    (scib_dir / "meta.json").write_text(json.dumps({"modalities": ["sc", "pseudobulk", "bulk"]}), encoding="utf-8")

    out = integration.main(["--root", tmp, "--name", "links", "--models", "good", "bad", "--min-cells", "5",
                            "--scib-runs", "fake_all", "missing_run", "--row-labels", "good=Good", "--formats", "png"])
    summary = pd.read_csv(out / "summary.csv", index_col=0)
    check("one summary row per model", list(summary.index) == ["good", "bad"])
    check("linked model ranks its partners first",
          summary.loc["good", "pb_cells_rank_tissue"] == 0.0 and summary.loc["good", "pair_pb_bulk_rank_tissue"] == 0.0)
    check("random model near chance", summary.loc["bad", "pb_cells_rank"] > 0.2)
    check("pairs with an unknown bulk row keep their tissue", summary["pair_cells_bulk_rank_tissue"].notna().all())
    check("scIB aggregates joined, a missing analysis skipped",
          "scib_fake_all_batch" in summary.columns and "scib_fake_all_bio" in summary.columns
          and not any(c.startswith("scib_missing_run") for c in summary.columns))
    coverage = pd.read_csv(out / "coverage.csv")
    row = coverage[(coverage["model"] == "good") & (coverage["edge"] == "pair_pb_bulk")].iloc[0]
    check("coverage counts shared keys with a tissue", row["keys_shared"] == 12 and row["keys_shared_with_tissue"] == 12)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("manifest lists the models and row labels",
          manifest["models"] == ["good", "bad"] and manifest["row_labels"] == {"good": "Good"})
    check("written to results/integration/<name>/",
          out == layout.integration_dir("links") and (out / "integration.png").exists())


def main() -> None:
    check_link_metrics()
    try:
        import anndata  # noqa: F401
    except ImportError:
        print("  [skip] anndata not importable: no end-to-end runs")
        finish()
        return
    with tempfile.TemporaryDirectory() as tmp:
        check_integration_run(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_root(Path(tmp))
        print("\n-- loading and selection --")
        obs, embs, hashes = scib_benchmark.load_models(layout, ["m1", "m2"], ["bulk", "pseudobulk"], "test")
        check("rows restricted to those every model embedded", len(obs) == 200 - 10)
        check("one aligned embedding per model", all(e.shape == (190, 8) for e in embs.values()))
        check("row names carry the modality", obs.index[0].startswith("bulk:"))
        check("panel hashes collected", hashes["m1"]["bulk_test.h5ad"] == "hash_m1")

        args = scib_benchmark.build_parser().parse_args(
            ["--models", "m1", "--name", "x", "--per-modality", "bulk=40", "pseudobulk=all"])
        mask = scib_benchmark.select_rows(obs, args, ["bulk", "pseudobulk"])
        picked = obs[mask]
        check("per-modality quota applied", (picked["modality"] == "bulk").sum() == 40
              and (picked["modality"] == "pseudobulk").sum() == 115)
        check("quota stratified by label (all bulk tissues kept)",
              set(picked.loc[picked["modality"] == "bulk", "tissue_general"]) == {"skin", "lung", "colon"})

        _, summary = scib_benchmark.label_batch_report(picked, "modality", "tissue_general", 10)
        check("pre-flight: skin is bulk-only, so 2 of 3 labels usable",
              summary["n_labels"] == 3 and summary["n_labels_usable"] == 2)

        print("\n-- dry run --")
        out = scib_benchmark.main(["--root", tmp, "--name", "t", "--models", "m1", "m2", "pca",
                                   "--pre-integrated", "pca", "--dry-run"])
        check("written under results/scib_benchmark/<name>/", out == layout.scib_dir("t"))
        manifest = json.loads((out / "manifest.json").read_text())
        check("manifest lists every model; the listed reference is also a row (unintegrated)",
              manifest["models"] == ["m1", "m2", "pca"] and manifest["rows"] == ["m1", "m2", "pca"])
        ref_only = scib_benchmark.main(["--root", tmp, "--name", "t_ref", "--models", "m1", "m2",
                                        "--pre-integrated", "pca", "--dry-run"])
        manifest = json.loads((ref_only / "manifest.json").read_text())
        check("a reference not listed under --models is no row",
              manifest["models"] == ["m1", "m2", "pca"] and manifest["rows"] == ["m1", "m2"])
        check("label x batch table saved", (out / "label_batch.csv").exists())
        meta = json.loads((out / "meta.json").read_text())
        check("meta records keys and counts", meta["batch_key"] == "modality" and meta["n_rows"] == 190)
        check("a missing label column is refused",
              raises(lambda: scib_benchmark.main(["--root", tmp, "--name", "t2", "--models", "m1",
                                                  "--label-key", "nope", "--dry-run"]), SystemExit))
        check("a single modality is refused",
              raises(lambda: scib_benchmark.main(["--root", tmp, "--name", "t3", "--models", "m1",
                                                  "--modalities", "bulk", "--dry-run"]), SystemExit))

        print("\n-- bio conservation only --")
        bio = ["--root", tmp, "--name", "bio", "--models", "m1", "m2", "pca", "--modalities", "bulk", "--bio-only"]
        out_bio = scib_benchmark.main(bio + ["--dry-run"])
        check("one modality accepted with --bio-only",
              json.loads((out_bio / "manifest.json").read_text())["bio_only"] is True
              and json.loads((out_bio / "meta.json").read_text())["batch_counts"] == {"bulk": 75})
        check("pca is a plain row with --bio-only",
              json.loads((out_bio / "manifest.json").read_text())["rows"] == ["m1", "m2", "pca"])
        check("--bio-only with --kbet refused",
              raises(lambda: scib_benchmark.main(bio + ["--kbet", "--dry-run"]), SystemExit))
        check("--bio-only with --pre-integrated refused",
              raises(lambda: scib_benchmark.main(bio + ["--pre-integrated", "pca", "--dry-run"]), SystemExit))

        print("\n-- results table figure --")
        try:
            import plottable  # noqa: F401
        except ImportError:
            print("  SKIP  plottable not importable here (it comes with scib_metrics)")
        else:
            _fake_results(out)
            base = ["--root", tmp, "--name", "t", "--plot-only", "--plot-formats", "png"]
            scib_benchmark.main(base + ["--row-labels", "m1=Model one (long label)", "m2=Model two"])
            check("raw and min-max tables drawn",
                  (out / "scib_results.png").exists() and (out / "scib_results_minmax.png").exists())
            manifest = json.loads((out / "manifest.json").read_text())
            check("row labels and figures recorded, models kept",
                  manifest["row_labels"] == {"m1": "Model one (long label)", "m2": "Model two"}
                  and "scib_results.png" in manifest["figures"] and manifest["models"] == ["m1", "m2", "pca"])
            from evaluate.integration.scib_plot import read_results, relabel

            relabeled = relabel(read_results(out / "scib.csv"), {"m1": "A"})
            check("relabel renames rows, keeps the Metric Type row",
                  list(relabeled.index) == ["A", "m2", "Metric Type"])
            check("CSV keeps the model names", "m1" in read_results(out / "scib.csv").index)
            check("duplicate labels refused", raises(lambda: scib_benchmark.main(
                base + ["--row-labels", "m1=X", "m2=X"]), SystemExit))
            check("malformed label refused", raises(lambda: scib_benchmark.main(
                base + ["--row-labels", "m1"]), SystemExit))
            check("plot-only needs an earlier run", raises(lambda: scib_benchmark.main(
                ["--root", tmp, "--name", "none", "--plot-only"]), SystemExit))
            _fake_results(out_bio, bio_only=True)
            scib_benchmark.main(["--root", tmp, "--name", "bio", "--plot-only", "--plot-formats", "png",
                                 "--row-labels", "m1=Model one"])
            check("bio-only table (no Total) drawn", (out_bio / "scib_results.png").exists())

        print("\n-- Benchmarker --")
        try:
            import scib_metrics  # noqa: F401
        except ImportError:
            print("  SKIP  scib_metrics not importable here (it lives in the bulkFM env)")
        else:
            out = scib_benchmark.main(["--root", tmp, "--name", "full", "--models", "m1", "m2",
                                       "--pre-integrated", "pca"])
            results = pd.read_csv(out / "scib.csv", index_col=0)
            check("one row per model plus the Metric Type row", {"m1", "m2", "Metric Type"} <= set(results.index))
            check("results table figure drawn", (out / "scib_results.png").exists())
            out = scib_benchmark.main(["--root", tmp, "--name", "bio_full", "--models", "m1", "m2",
                                       "--modalities", "bulk", "--bio-only"])
            results = pd.read_csv(out / "scib.csv", index_col=0)
            check("--bio-only: no batch metrics, no Total",
                  "Total" not in results.columns and "Batch correction" not in results.columns)

    print("\n-- XLA autotuning (JAX compile stalls) --")
    env: dict[str, str] = {}
    check("no XLA_FLAGS: autotuning turned off",
          scib_benchmark.configure_xla(False, env) == "--xla_gpu_autotune_level=0" and env["XLA_FLAGS"] == "--xla_gpu_autotune_level=0")
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    check("other flags kept, level appended",
          scib_benchmark.configure_xla(False, env) == "--xla_force_host_platform_device_count=2 --xla_gpu_autotune_level=0")
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=4"}
    check("a level the user set is respected",
          scib_benchmark.configure_xla(False, env) == "--xla_gpu_autotune_level=4" and env["XLA_FLAGS"].endswith("=4"))
    env = {}
    check("--xla-autotune leaves XLA_FLAGS alone", scib_benchmark.configure_xla(True, env) == "" and "XLA_FLAGS" not in env)
    check("--xla-autotune parses", scib_benchmark.build_parser().parse_args(["--xla-autotune"]).xla_autotune)
    finish()


def _fake_results(out: Path, bio_only: bool = False) -> None:
    """``scib.csv`` / ``scib_minmax.csv`` shaped like ``Benchmarker.get_results``."""
    types = {"Isolated labels": "Bio conservation", "KMeans NMI": "Bio conservation",
             "Silhouette label": "Bio conservation", "iLISI": "Batch correction",
             "Graph connectivity": "Batch correction", "Batch correction": "Aggregate score",
             "Bio conservation": "Aggregate score", "Total": "Aggregate score"}
    values = {"m1": [0.6, 0.7, 0.55, 0.2, 0.9], "m2": [0.5, 0.6, 0.52, 0.4, 0.95]}
    drop = {"iLISI", "Graph connectivity", "Batch correction", "Total"} if bio_only else set()
    for name, scale in (("scib.csv", False), ("scib_minmax.csv", True)):
        rows = {}
        for model, v in values.items():
            v = np.array(v)
            if scale:
                other = np.array(values["m2" if model == "m1" else "m1"])
                v = (v >= other).astype(float)
            bio, batch = v[:3].mean(), v[3:].mean()
            rows[model] = [*v, batch, bio, 0.4 * batch + 0.6 * bio]
        df = pd.DataFrame(rows, index=list(types)).T.drop(columns=list(drop)).astype(object)
        df.loc["Metric Type"] = [types[c] for c in df.columns]
        df.index.name = "Embedding"
        df.to_csv(out / name)


if __name__ == "__main__":
    main()
