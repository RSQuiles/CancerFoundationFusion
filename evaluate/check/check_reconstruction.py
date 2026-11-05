"""Self-check for ``evaluate/reconstruction`` on synthetic counts (no torch or checkpoint).

Covers the mask plan (determinism, arithmetic, disjoint masked/context sets, layout
path), the bin decoders against known logits, the metrics against hand-built cases
(perfect and constant predictors, baselines), the tied-bin ranges against
``preprocess.binning`` and the tie-aware metrics, and ``run_reconstruction``,
``compare`` and ``plot_reconstruction`` end to end with fake reconstructors.

    python -m evaluate.check.check_reconstruction
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
from evaluate.reconstruction import compare, mask_plan, plot_reconstruction, run_reconstruction  # noqa: E402
from evaluate.reconstruction import metrics as M  # noqa: E402
from evaluate.reconstruction.reconstructors import (  # noqa: E402
    RECONSTRUCTORS, Reconstructor, bin_tie_range, decode_bins, register,
)

N_SAMPLES, N_GENES = 60, 80


def _write_root(root: Path) -> ExperimentLayout:
    import anndata as ad

    rng = np.random.default_rng(0)
    layout = ExperimentLayout(root)
    X = rng.negative_binomial(2, 0.1, size=(N_SAMPLES, N_GENES)).astype(np.float32)
    X[:, :10] = 0  # some all-zero genes
    obs = pd.DataFrame({"tissue_general": rng.choice(["lung", "colon", "skin"], N_SAMPLES),
                        "modality": "bulk"}, index=[f"s{i}" for i in range(N_SAMPLES)])
    path = layout.raw_h5ad("bulk")
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(N_GENES)])).write_h5ad(path)
    split = layout.split_dir("bulk", "s0")
    split.mkdir(parents=True)
    (split / "train.txt").write_text("\n".join(obs.index[:30]), encoding="utf-8")
    (split / "test.txt").write_text("\n".join(obs.index[30:]), encoding="utf-8")
    return layout


def check_plan(tmp: Path, layout: ExperimentLayout) -> None:
    print("\n-- mask plan --")
    args = ["--root", str(tmp), "--modality", "bulk", "--split", "s0", "--part", "test", "--name", "r25",
            "--mask-ratio", "0.25", "--seq-len", "41", "--n-samples", "0"]
    path = mask_plan.main(args)
    check("plan written beside the split", path == layout.mask_plan("bulk", "s0", "test", "r25"))
    plan = mask_plan.load_plan(path)
    check("all test rows, none of train", plan.n_samples == 30 and set(plan.obs_names) == {f"s{i}" for i in range(30, 60)})
    check("seq-len and mask-ratio arithmetic: 40 genes, 10 masked",
          all(len(plan.masked(i)) == 10 and len(plan.context(i)) == 30 for i in range(plan.n_samples)))
    check("masked and context disjoint",
          all(not set(plan.masked(i)) & set(plan.context(i)) for i in range(plan.n_samples)))
    check("plan id recorded", plan.plan_id == "bulk:s0:test:r25")
    check("existing plan not overwritten", raises(lambda: mask_plan.main(args), SystemExit))
    again = mask_plan.load_plan(mask_plan.main(args + ["--overwrite"]))
    check("deterministic under a seed", np.array_equal(again.masked_idx, plan.masked_idx))
    check("resolved from MODALITY:SPLIT:PART:NAME",
          mask_plan.resolve_plan_path("bulk:s0:test:r25", tmp) == path)

    rng = np.random.default_rng(1)
    values = np.r_[np.zeros(50), np.ones(50)]
    masked, context = mask_plan.sample_plan_row(values, 21, 0.5, rng, zero_fraction=0.4)
    chosen = np.r_[masked, context]
    check("zero_fraction respected (8 zeros of 20)", int((values[chosen] == 0).sum()) == 8)
    check("a ratio that masks nothing is refused",
          raises(lambda: mask_plan.sample_plan_row(values, 3, 0.1, rng), ValueError))

    universe = tmp / "genes.txt"
    universe.write_text("\n".join(f"G{i}" for i in range(20, 60)), encoding="utf-8")
    u = mask_plan.load_plan(mask_plan.main(args[:-6] + ["--name", "u", "--mask-ratio", "0.25", "--seq-len", "41",
                                                       "--n-samples", "10", "--gene-universe", str(universe)]))
    check("gene universe restricts the plan genes", set(u.genes) == {f"G{i}" for i in range(20, 60)})


def check_decoders() -> None:
    print("\n-- decoders --")
    n_bins = 5
    big = 30.0
    # ordinal CE: class 0 always on; target bin 3 -> logits on for k<=3
    ordinal = np.array([[big, big, big, big, -big]])
    check("ordinal CE decodes bin 3 (no off-by-one)", np.allclose(decode_bins(ordinal, "ordinal_cross_entropy", n_bins), 3))
    corn = np.array([[big, big, -big, -big]])
    check("CORN decodes bin 2", np.allclose(decode_bins(corn, "corn", n_bins), 2))
    check("mse with normalised bins scaled back", np.allclose(decode_bins(np.array([0.4]), "mse", 5, True), 2.0))
    check("mse (B, L, 1) squeezed", decode_bins(np.zeros((2, 3, 1)), "mse", 5).shape == (2, 3))
    check("zinb refused", raises(lambda: decode_bins(np.zeros((1, 3)), "zinb", 5), ValueError))


def check_metrics() -> None:
    print("\n-- metrics --")
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        t = rng.integers(0, 10, size=30).astype(float)
        rows.append(pd.DataFrame({"sample": f"s{s}", "target": t, "pred": t,
                                  "gene_mean": np.full(30, 4.5), "sample_mean": t + rng.normal(0, 3, 30)}))
    frame = pd.concat(rows, ignore_index=True)
    res = M.evaluate(frame, "bins")
    check("perfect predictor: mae 0, exact accuracy 1", res["pred"]["mae"] == 0 and res["pred"]["acc_exact"] == 1)
    check("perfect predictor: spearman 1", np.isclose(res["pred"]["spearman_mean"], 1))
    check("constant predictor: correlation undefined, not a crash", np.isnan(res["gene_mean"]["spearman_mean"]))
    check("noisy baseline between", 0 < res["sample_mean"]["spearman_mean"] < 1)
    check("AUROC of a perfect zero ranking is 1", M.auroc(-frame["pred"].to_numpy(), frame["target"].to_numpy() == 0) == 1)
    check("AUROC of a constant score is 0.5", M.auroc(np.zeros(10), np.r_[np.ones(5), np.zeros(5)].astype(bool)) == 0.5)
    check("MAE by true bin", set(M.mae_by_target(frame, "bins")["mae"]) == {0.0})
    check("MAE by decile in other spaces", len(M.mae_by_target(frame.assign(target=frame["target"] + 0.1), "ln_tpm1")) > 1)


def _binning():
    """``preprocess.binning`` loaded by path: the package's ``__init__`` imports torch."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_preprocess", PROJECT_ROOT / "cancerfoundation" / "data" / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.binning


def check_ties() -> None:
    print("\n-- tied bins --")
    binning = _binning()
    rng = np.random.default_rng(0)
    inside, tied, single = True, False, True
    for n_bins in (11, 51):
        for _ in range(20):
            # Few distinct values, so many quantile edges repeat.
            row = rng.choice([0, 1, 2, 3, 5, 8, 13], size=300,
                             p=[.3, .3, .15, .1, .07, .05, .03]).astype(np.float32)
            lo, hi = bin_tie_range(row, n_bins)
            for seed in range(5):
                np.random.seed(seed)
                b = binning(row.copy(), n_bins)
                inside &= bool(((b >= lo) & (b <= hi)).all())
            tied |= bool((lo < hi).any())
            single &= bool((lo[row == 0] == 0).all() and (hi[row == 0] == 0).all())
        cont = rng.gamma(2.0, 1.0, size=300).astype(np.float32)  # distinct values: no ties
        lo, hi = bin_tie_range(cont, n_bins)
        np.random.seed(0)
        single &= bool((lo == hi).all() and (binning(cont.copy(), n_bins) == lo).all())
    check("binning always lands in [lo, hi]", inside)
    check("repeated edges give a range of bins", tied)
    check("zeros and distinct values have one bin", single)
    check("all-zero row", all((x == 0).all() for x in bin_tie_range(np.zeros(5, np.float32), 11)))

    frame = pd.DataFrame({"sample": "s", "target": [0., 3., 3., 7.], "target_lo": [0., 2., 2., 7.],
                          "target_hi": [0., 5., 5., 7.], "pred": [0., 4.6, 1., 9.]})
    nat = M.native(frame, "bins")
    check("tie error: 0 inside the range, distance outside", np.allclose(M.tie_error(frame), [0, 0, 1, 2]))
    check("mae_tie <= mae", nat["mae_tie"] <= nat["mae"])
    check("tie-aware exact accuracy counts rint(pred) inside the range", nat["acc_exact_tie"] == 0.5)
    check("tie-aware ±1 accuracy", nat["acc_within1_tie"] == 0.75)
    check("raw metrics kept", "mae" in nat and "acc_exact" in nat)
    check("no tie columns: no tie metrics",
          "mae_tie" not in M.native(frame.drop(columns=["target_lo", "target_hi"]), "bins"))
    check("mae_by_target gains mae_tie", "mae_tie" in M.mae_by_target(frame, "bins").columns)


@register("_fake_perfect")
class _Perfect(Reconstructor):
    space = "ln_tpm1"

    def predict(self, adata, plan):
        X = np.asarray(adata.X)
        pos = adata.var_names.get_indexer(plan.genes)
        targets = np.log1p(X[:, pos])
        rows = np.concatenate([np.full(len(plan.masked(i)), i) for i in range(plan.n_samples)])
        genes = np.concatenate([plan.masked(i) for i in range(plan.n_samples)])
        # Drops the first masked gene of every sample, as if outside the vocabulary.
        keep = np.ones(len(rows), bool)
        keep[np.r_[0, np.flatnonzero(np.diff(rows)) + 1]] = False
        return {"targets": targets, "row": rows[keep], "gene": genes[keep],
                "pred": targets[rows[keep], genes[keep]], "n_dropped": int((~keep).sum())}


@register("_fake_noisy")
class _Noisy(_Perfect):
    space = "bins"

    def predict(self, adata, plan):
        out = super().predict(adata, plan)
        out["pred"] = out["pred"] + np.random.default_rng(0).normal(0, 1, len(out["pred"]))
        return out


def check_end_to_end(tmp: Path, layout: ExperimentLayout) -> None:
    print("\n-- run_reconstruction + compare --")
    base = ["--root", str(tmp), "--plan", "bulk:s0:test:r25"]
    report = run_reconstruction.main(base + ["--reconstructor", "_fake_perfect", "--name", "perfect"])
    out = layout.task_dir("reconstruction", "perfect") / "bulk_s0_test_r25"
    check("outputs written under results/tasks/reconstruction/<name>/<plan>/",
          all((out / f).exists() for f in ("metrics.json", "entries.parquet", "per_sample.csv", "mae_by_target.csv")))
    check("dropped entries counted", report["n_dropped"] == 30 and report["n_scored"] + 30 == report["n_masked"])
    check("perfect model scores spearman ~1", report["metrics"]["pred"]["spearman_mean"] > 0.99)
    check("breakdown by tissue and modality", {"tissue_general", "modality"} <= set(report["by_group"]))
    card = json.loads((out / "metrics.json").read_text())
    check("metrics.json is strict JSON (NaN -> null)", "NaN" not in (out / "metrics.json").read_text() and card["space"] == "ln_tpm1")
    entries = pd.read_parquet(out / "entries.parquet")
    check("baselines stored per entry", {"gene_mean", "sample_mean"} <= set(entries.columns))

    run_reconstruction.main(base + ["--reconstructor", "_fake_noisy", "--name", "noisy"])
    table = compare.main(["--root", str(tmp), "--plan", "bulk:s0:test:r25", "--models", "perfect", "noisy"])
    assess = layout.assessment_dir("reconstruction_bulk_s0_test_r25")
    check("comparison written with a manifest",
          json.loads((assess / "manifest.json").read_text())["models"] == ["perfect", "noisy"])
    check("perfect beats noisy on shared entries",
          table.set_index("model").loc["perfect", "spearman_mean"] > table.set_index("model").loc["noisy", "spearman_mean"])
    check("native metrics omitted across different spaces", "mae" not in table.columns)
    check("unknown model refused", raises(lambda: compare.main(["--root", str(tmp), "--plan", "bulk:s0:test:r25",
                                                                "--models", "perfect", "nope"]), SystemExit))


def _fake_run(layout: ExperimentLayout, model: str, slug: str, frame: pd.DataFrame, space: str,
              n_bins: int | None = None) -> None:
    out = layout.task_dir("reconstruction", model) / slug
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "entries.parquet", index=False)
    prov = {"method": model, "space": space, **({"n_bins": n_bins} if n_bins else {})}
    (out / "metrics.json").write_text(json.dumps({"model": model, "space": space, "provenance": prov}))


def _bins_frame(rng, noise: float, n_bins: int, ties: bool = True) -> pd.DataFrame:
    rows = []
    for s in range(30):
        t = rng.integers(0, n_bins, size=40).astype(float)
        f = pd.DataFrame({"row": s, "sample": f"s{s}", "gene": [f"G{g}" for g in range(40)], "target": t,
                          "pred": np.clip(t + rng.normal(0, noise, 40), 0, n_bins - 1),
                          "gene_mean": np.full(40, n_bins / 2), "sample_mean": np.full(40, t.mean())})
        if ties:
            # Non-zero targets tied with the bin below; zeros are never tied.
            f["target_lo"], f["target_hi"] = np.where(t > 1, t - 1, t), t
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def _group(space: str, tie_aware: bool):
    return plot_reconstruction.Group("g", space, [], {}, 0, tie_aware)


def check_plots(tmp: Path, layout: ExperimentLayout) -> None:
    print("\n-- plot_reconstruction --")
    slug = "bulk_s0_test_r25"
    rng = np.random.default_rng(0)
    good, bad = _bins_frame(rng, 0.5, 51), _bins_frame(rng, 4.0, 51)
    _fake_run(layout, "cf_good", slug, good, "bins", 51)
    _fake_run(layout, "cf_bad", slug, bad.iloc[:-40], "bins", 51)  # one sample fewer
    _fake_run(layout, "cf_small", slug, _bins_frame(rng, 1.0, 21, ties=False), "bins", 21)
    ln = good.drop(columns=["target_lo", "target_hi"]).assign(target=np.log1p(good["target"]),
                                                              pred=np.log1p(good["pred"]))
    _fake_run(layout, "bf", slug, ln, "ln_tpm1")

    base = ["--root", str(tmp), "--plan", "bulk:s0:test:r25", "--formats", "png"]
    models = ["cf_good", "cf_bad", "cf_small", "bf"]
    out = plot_reconstruction.main(base + ["--models", *models, "--baselines"])
    check("written under results/tasks/reconstruction/_plots/<selection>/<plan>/",
          out == layout.task_plot_dir("reconstruction", "_".join(models), slug))
    manifest = json.loads((out / "manifest.json").read_text())
    check("one group per space and n_bins, plus all",
          set(manifest["groups"]) == {"bins51", "bins21", "ln_tpm1", "all"})
    check("manifest lists the models", manifest["models"] == models)
    expected = {"bins51": ["mae_by_bin", "metrics_table"], "bins21": ["mae_by_bin", "metrics_table"],
                "ln_tpm1": ["mae_by_target", "metrics_table"],
                "all": ["space_free_table", "per_sample_spearman", "rank_mae_by_target"]}
    check("every plot drawn where it applies",
          all((out / g / f"{p}.png").exists() and (out / g / f"{p}.csv").exists()
              for g, ps in expected.items() for p in ps))
    check("bins-only plot not drawn for ln_tpm1", not (out / "ln_tpm1" / "mae_by_bin.png").exists())
    check("tie-aware where the runs have ranges", manifest["groups"]["bins51"]["tie_aware"]
          and not manifest["groups"]["bins21"]["tie_aware"] and manifest["groups"]["bins21"]["notes"])
    check("shared entries: the sample cf_bad lacks is left out",
          manifest["groups"]["bins51"]["n_shared_entries"] == 29 * 40)

    table = pd.read_csv(out / "bins51" / "metrics_table.csv").set_index("row")
    _, subs = compare.shared_entries({"cf_good": good, "cf_bad": bad.iloc[:-40]})
    ref = M.native(subs["cf_good"], "bins")
    check("table matches metrics.native on the shared entries",
          np.isclose(table.loc["cf_good", "mae_tie"], ref["mae_tie"]))
    check("baseline rows included", "cf_good:gene_mean" in table.index)
    from evaluate._plot_style import is_best

    check("best MAE is the better model",
          table.index[is_best(table["mae_tie"].to_numpy(), "min")].tolist() == ["cf_good"])
    small = pd.read_csv(out / "bins21" / "metrics_table.csv").set_index("row")
    check("no tie ranges: raw bin error, no tie columns", "mae_tie" not in small.columns
          and [c.key for c in plot_reconstruction.drawn_columns(_group("bins", False))][0] == "mae")
    by_bin = pd.read_csv(out / "bins51" / "mae_by_bin.csv")
    check("mae_by_bin per true bin", set(by_bin["row"]) >= {"cf_good", "cf_bad"} and by_bin["bin"].max() <= 50)
    rank = pd.read_csv(out / "all" / "rank_mae_by_target.csv")
    check("rank error in [0, 1] by decile", rank["mae_rank"].between(0, 1).all() and rank["quantile"].max() <= 9)
    by_target = pd.read_csv(out / "ln_tpm1" / "mae_by_target.csv")
    check("zero targets get their own group", "zero" in set(by_target["label"]))

    alone = plot_reconstruction.main(base + ["--models", "cf_good", "--plots", "mae_by_bin"])
    check("single model: no cross-space group",
          not (alone / "all").exists() and (alone / "bins51" / "mae_by_bin.png").exists())
    everything = plot_reconstruction.main(base + ["--plots", "metrics_table"])
    check("all models -> all_models", everything.parent.name == "all_models"
          and set(json.loads((everything / "manifest.json").read_text())["models"]) >= set(models))
    check("unknown plot refused", raises(lambda: plot_reconstruction.main(base + ["--plots", "nope"]), SystemExit))
    check("unknown model refused", raises(lambda: plot_reconstruction.main(base + ["--models", "nope"]), SystemExit))


def _cleanup() -> None:
    for name in ("_fake_perfect", "_fake_noisy"):
        RECONSTRUCTORS.pop(name, None)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        layout = _write_root(tmp)
        check_plan(tmp, layout)
        check_decoders()
        check_metrics()
        check_ties()
        check_end_to_end(tmp, layout)
        check_plots(tmp, layout)
    _cleanup()
    finish()


if __name__ == "__main__":
    main()
