"""Self-check for ``evaluate/reconstruction`` on synthetic counts (no torch or checkpoint).

Covers the mask plan (determinism, arithmetic, disjoint masked/context sets, layout
path), the bin decoders against known logits, the metrics against hand-built cases
(perfect and constant predictors, baselines), and ``run_reconstruction`` plus
``compare`` end to end with fake reconstructors.

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
from evaluate.reconstruction import compare, mask_plan, run_reconstruction  # noqa: E402
from evaluate.reconstruction import metrics as M  # noqa: E402
from evaluate.reconstruction.reconstructors import (  # noqa: E402
    RECONSTRUCTORS, Reconstructor, decode_bins, register,
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
    for name in ("_fake_perfect", "_fake_noisy"):
        RECONSTRUCTORS.pop(name)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        layout = _write_root(tmp)
        check_plan(tmp, layout)
        check_decoders()
        check_metrics()
        check_end_to_end(tmp, layout)
    finish()


if __name__ == "__main__":
    main()
