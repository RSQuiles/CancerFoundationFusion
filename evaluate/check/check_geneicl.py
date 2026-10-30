"""Self-check for ``evaluate/geneicl`` against a fake GeneICL checkout (no torch, no GeneICL env).

The fake checkout has two datasets, three tasks and a stub ``benchmark.py`` that,
like the real one, accepts only models in its ``EMB_MODELS`` set and writes
``<out>_<task_type>.csv`` plus the regression OOF companion. Covers checkout
discovery, gene mapping, ``embed_suite`` (format, resume, refusals), ``run_probe``
(override quoting, the shim, dataset subsets, ``run.json``) and ``compare`` (task
intersection, pooled out-of-fold R2, references, manifest).

    python -m evaluate.check.check_geneicl
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

from embed.embed_any import EMBEDDERS, Embedder, register  # noqa: E402
from evaluate.geneicl import compare, embed_suite, run_probe  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402
from experiments.check._harness import check, finish, raises  # noqa: E402
from experiments.layout import ExperimentLayout  # noqa: E402

GENES = ["A1BG", "TP53", "EGFR", "MYC", "KRAS", "GAPDH"]
MAPPING = {"A1BG": "ENSG01", "TP53": "ENSG02", "EGFR": "ENSG03", "MYC": "ENSG04", "KRAS": "ENSG02"}  # KRAS dup
DATASETS = {"DSA": 30, "DSB": 24}

STUB_BENCHMARK = '''
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

EMB_MODELS = {"scfoundation"}

def main():
    cfg = dict(a.split("=", 1) for a in sys.argv[1:])
    cfg = {k: v.strip("'") for k, v in cfg.items()}
    assert cfg["model"] in EMB_MODELS, cfg["model"]
    data, emb, out = Path(cfg["data_dir"]), Path(cfg["emb_dir"]), Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, oof = {"classification": [], "regression": []}, []
    for t in json.load(open(data / "index.json"))["tasks"]:
        if t["task_type"] not in ("classification", "regression"):
            continue
        z = np.load(data / t["file"], allow_pickle=True)
        e = np.load(emb / (t["dataset"] + ".npz"), allow_pickle=True)
        pos = {str(s): i for i, s in enumerate(e["obs_names"].tolist())}
        X = e["emb"][[pos[str(s)] for s in z["sample_ids"].tolist()]]
        for fold in range(2):
            m = {"model": cfg["model"], "dataset": t["dataset"], "task_id": t["task_id"],
                 "task_type": t["task_type"], "fold": fold, "n_train": len(X) // 2, "n_test": len(X) // 2}
            if t["task_type"] == "classification":
                m.update(auroc=0.5 + 0.1 * fold, auprc=0.5, balanced_accuracy=0.5, mcc=0.0, f1_macro=0.5)
            else:
                m.update(spearman=0.2 + 0.2 * fold, pearson=0.3, r2=0.1)
                y = np.asarray(z["y"], float)
                oof += [{"model": cfg["model"], "task_id": t["task_id"], "fold": fold,
                         "y_true": a, "pred": a + 1.0} for a in y[fold::2]]
            rows[t["task_type"]].append(m)
    for tt, r in rows.items():
        if r:
            pd.DataFrame(r).to_csv(f"{out}_{tt}.csv", index=False)
    if oof:
        pd.DataFrame(oof).to_csv(f"{out}_regression.oof.csv", index=False)
'''


@register("_geneicl_fake")
class FakeEmbedder(Embedder):
    """First three input genes as the embedding."""

    def embed(self, adata) -> np.ndarray:
        self.genes = list(adata.var_names[:3])
        return np.asarray(adata.X[:, :3], dtype=np.float32)


def _write_geneicl(repo: Path) -> Path:
    rng = np.random.default_rng(0)
    bench = repo / "eval" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "benchmark.py").write_text(STUB_BENCHMARK, encoding="utf-8")
    suite = repo / "eval" / "data" / "benchmark"
    (suite / "datasets").mkdir(parents=True)
    (suite / "tasks").mkdir()
    (suite / "reference_genes.txt").write_text("\n".join(GENES) + "\n", encoding="utf-8")
    tasks = []
    for ds, n in DATASETS.items():
        obs = np.array([f"{ds}_s{i}" for i in range(n)], dtype=object)
        X = np.log1p(rng.poisson(20, size=(n, len(GENES)))).astype(np.float32)
        np.savez(suite / "datasets" / f"{ds}.npz", X=X, obs_names=obs)
        specs = [("cls", "classification", rng.choice(["a", "b"], n))]
        if ds == "DSA":
            specs.append(("reg", "regression", rng.normal(size=n)))
        for label, tt, y in specs:
            tid, file = f"{ds}::{label}", f"tasks/{ds}__{label}.npz"   # GeneICL names files <task_id>.npz;
            np.savez(suite / file, sample_ids=obs[::-1], y=y, task_type=tt, dataset=ds)  # ':' is invalid on Windows
            tasks.append({"task_id": tid, "dataset": ds, "task_type": tt, "n": n, "file": file})
    tasks.append({"task_id": "DSB::os", "dataset": "DSB", "task_type": "survival", "n": 24,
                  "file": "tasks/DSB__os.npz"})
    (suite / "index.json").write_text(json.dumps({"n_genes": len(GENES), "datasets": DATASETS, "tasks": tasks}),
                                      encoding="utf-8")
    return suite


def check_suite(tmp: Path, repo: Path, suite: Path) -> None:
    print("\n-- GeneICL checkout --")
    check("explicit path resolved", S.resolve_geneicl(repo) == repo)
    check("missing checkout refused", raises(lambda: S.resolve_geneicl(tmp / "nope"), FileNotFoundError))
    (tmp / "empty").mkdir()
    check("checkout without benchmark.py refused (restore hint)",
          raises(lambda: S.resolve_geneicl(tmp / "empty"), FileNotFoundError))
    check("suite dir", S.suite_dir(repo, "benchmark") == suite)
    check("unknown suite refused", raises(lambda: S.suite_dir(repo, "train"), ValueError))
    check("datasets of scored tasks, sorted", S.list_datasets(suite) == ["DSA", "DSB"])
    check("survival task not scored", all(t["task_type"] != "survival" for t in S.scored_tasks(S.load_index(suite))))
    check("unknown dataset refused", raises(lambda: S.list_datasets(suite, ["GTEx"]), ValueError))
    X, obs = S.load_dataset(suite, "DSB")
    check("dataset loads as float32 with its obs names", X.dtype == np.float32 and X.shape == (24, 6)
          and obs[0] == "DSB_s0")

    print("\n-- gene mapping --")
    cols, names, space = S.map_genes(GENES, "symbol")
    check("symbols kept as they are", space == "symbol" and names == GENES and len(cols) == 6)
    cols, names, space = S.map_genes(GENES, "ensembl", mapping=MAPPING)
    check("ensembl: unmapped and duplicate targets dropped",
          names == ["ENSG01", "ENSG02", "ENSG03", "ENSG04"] and cols.tolist() == [0, 1, 2, 3])
    _, _, space = S.map_genes(GENES, "auto", vocab={"ENSG01": 0, "ENSG02": 1, "TP53": 2}, mapping=MAPPING)
    check("auto picks Ensembl when it overlaps the vocabulary more", space == "ensembl")
    _, _, space = S.map_genes(GENES, "auto", vocab={"TP53": 0, "MYC": 1, "ENSG01": 2}, mapping=MAPPING)
    check("auto picks symbols when they overlap more", space == "symbol")
    check("auto without a vocabulary refused", raises(lambda: S.map_genes(GENES, "auto"), ValueError))
    check("the shipped symbol map is readable", "TP53" in json.loads(S.SYMBOL_TO_ENSEMBL.read_text(encoding="utf-8")))


def check_embed(tmp: Path, repo: Path, suite: Path, layout: ExperimentLayout) -> None:
    print("\n-- embed_suite --")
    out = tmp / "emb_direct"
    calls = []

    def embed_fn(adata):
        calls.append(adata.n_obs)
        return adata.X[:, :2] * 2, list(adata.var_names[:2])

    cols, names, _ = S.map_genes(GENES, "ensembl", mapping=MAPPING)
    identity = {"method": "cancerfoundation", "checkpoint": "a.ckpt", "suite": "benchmark"}
    card = embed_suite.embed_suite(suite, ["DSA", "DSB"], out, embed_fn, cols, names, identity)
    z = np.load(out / "DSA.npz", allow_pickle=True)
    X, obs = S.load_dataset(suite, "DSA")
    check("npz holds emb (float32) and obs_names", set(z.files) == {"emb", "obs_names"} and z["emb"].dtype == np.float32)
    check("rows keep the dataset's order and values", z["obs_names"].tolist() == obs
          and np.allclose(z["emb"], X[:, :2] * 2))
    check("card per file with panel hash", set(card["files"]) == {"DSA.npz", "DSB.npz"}
          and card["files"]["DSA.npz"]["panel_hash"] and card["files"]["DSB.npz"]["dim"] == 2)
    check("panel saved", (out / f"panel_{card['files']['DSA.npz']['panel_hash']}.txt").exists())
    check("panel genes are the mapped names", (out / f"panel_{card['files']['DSA.npz']['panel_hash']}.txt")
          .read_text(encoding="utf-8").split() == ["ENSG01", "ENSG02"])
    embed_suite.embed_suite(suite, ["DSA", "DSB"], out, embed_fn, cols, names, identity)
    check("existing files skipped on rerun", calls == [30, 24])
    other = {**identity, "checkpoint": "b.ckpt"}
    check("another checkpoint refused without --overwrite",
          raises(lambda: embed_suite.embed_suite(suite, ["DSA"], out, embed_fn, cols, names, other), SystemExit))
    embed_suite.embed_suite(suite, ["DSA"], out, embed_fn, cols, names, other, overwrite=True)
    check("--overwrite re-embeds and replaces the card", calls == [30, 24, 30]
          and json.loads((out / "embed.json").read_text(encoding="utf-8"))["checkpoint"] == "b.ckpt")

    def nan_fn(adata):
        emb = np.ones((adata.n_obs, 2), np.float32)
        emb[3] = np.nan
        return emb, None

    bad = tmp / "emb_nan"
    check("non-finite rows refused", raises(
        lambda: embed_suite.embed_suite(suite, ["DSA"], bad, nan_fn, cols, names, identity), ValueError))
    check("nothing written for a refused dataset", not (bad / "DSA.npz").exists())

    card = embed_suite.main(["--root", str(tmp / "root"), "--geneicl-repo", str(repo),
                             "--embedder", "_geneicl_fake", "--name", "fake"])
    path = layout.geneicl_embedding_dir("fake", "benchmark")
    check("--embedder writes under embeddings/<name>/geneicl_<suite>/",
          (path / "DSA.npz").exists() and (path / "DSB.npz").exists())
    check("embedder card records method, suite and symbols",
          card["method"] == "_geneicl_fake" and card["suite"] == "benchmark" and card["gene_ids"] == "symbol"
          and card["normalize"] is None)
    check("--model with --name refused", raises(lambda: embed_suite.main(
        ["--root", str(tmp / "root"), "--geneicl-repo", str(repo), "--model", "m", "--name", "x"]), SystemExit))


def check_probe(tmp: Path, repo: Path, layout: ExperimentLayout) -> None:
    print("\n-- run_probe --")
    check("safe values bare", run_probe.hydra_value(Path("/a/b_c.d")) == "/a/b_c.d")
    check("unsafe values quoted", run_probe.hydra_value("C:/x y") == "'C:/x y'")
    check("values with a quote refused", raises(lambda: run_probe.hydra_value("a'b"), ValueError))
    check("probe key from the card", run_probe.default_probe_key({"method": "cancerfoundation"}) == "cancerfoundation"
          and run_probe.default_probe_key({"method": "pca"}) == "cf_pca")

    base = ["--root", str(tmp / "root"), "--geneicl-repo", str(repo), "--geneicl-python", sys.executable]
    dry = run_probe.main(base + ["--model", "fake", "--dry-run", "--resume"])
    cmd = dry["command"]
    check("command runs the shim with key and overrides",
          cmd[1].endswith("_probe_shim.py") and cmd[3] == "cf__geneicl_fake" and "overwrite=false" in cmd
          and "hydra.output_subdir=null" in cmd)
    check("missing embeddings refused", raises(lambda: run_probe.main(base + ["--model", "nobody"]), SystemExit))

    run = run_probe.main(base + ["--model", "fake"])
    task_dir = layout.task_dir("geneicl_benchmark", "fake")
    cls = pd.read_csv(task_dir / "probe_classification.csv")
    check("shim registered the key: CSVs written under results/tasks/geneicl_benchmark/<model>/",
          (task_dir / "probe_regression.csv").exists() and set(cls["model"]) == {"cf__geneicl_fake"})
    check("every scored task of both datasets", set(cls["task_id"]) == {"DSA::cls", "DSB::cls"})
    check("run.json records key, suite and panels", run["probe_key"] == "cf__geneicl_fake"
          and json.loads((task_dir / "run.json").read_text(encoding="utf-8"))["panel_hashes"].keys()
          == {"DSA.npz", "DSB.npz"})

    run_probe.main(base + ["--model", "fake", "--datasets", "DSB", "--probe-key", "subset_key"])
    cls = pd.read_csv(task_dir / "probe_classification.csv")
    subset = task_dir / run_probe.SUBSET_DIR
    check("--datasets scores only those tasks", set(cls["task_id"]) == {"DSB::cls"})
    check("subset suite holds only that dataset's task files",
          sorted(p.name for p in (subset / "tasks").iterdir()) == ["DSB__cls.npz"])


def check_compare(tmp: Path, repo: Path, suite: Path, layout: ExperimentLayout) -> None:
    print("\n-- compare --")
    base = ["--root", str(tmp / "root"), "--geneicl-repo", str(repo), "--geneicl-python", sys.executable]
    run_probe.main(base + ["--model", "fake"])  # full suite again
    results = repo / "eval" / "benchmark" / "results"
    results.mkdir()
    pd.DataFrame([{"model": "scfoundation", "dataset": "DSA", "task_id": "DSA::cls", "task_type": "classification",
                   "fold": f, "auroc": 0.9, "auprc": 0.9, "balanced_accuracy": 0.8, "mcc": 0.6, "f1_macro": 0.8}
                  for f in range(2)]).to_csv(results / "scfoundation_classification.csv", index=False)
    summary = compare.main(["--root", str(tmp / "root"), "--geneicl-repo", str(repo), "--models", "fake",
                            "--reference", "scfoundation", "missing_tag", "--name", "t"])
    out = layout.assessment_dir("geneicl_benchmark_t")
    per_task = pd.read_csv(out / "per_task.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("rows: the model and the found reference", summary["row"].tolist() == ["fake", "ref:scfoundation"])
    check("classification on the shared task only", set(per_task.loc[per_task.task_type == "classification",
                                                                      "task_id"]) == {"DSA::cls"})
    check("fold mean of AUROC", np.isclose(summary.loc[0, "cla_auroc"], 0.55))
    check("rank: reference first on AUROC", summary.loc[1, "cla_rank_auroc"] == 1.0
          and summary.loc[0, "cla_rank_auroc"] == 2.0)
    check("regression left out when a row has none", summary.loc[0, "n_regression_tasks"] == 0)
    check("manifest lists the model and, apart, the references",
          manifest["models"] == ["fake"] and manifest["references"] == ["scfoundation"])

    summary = compare.main(["--root", str(tmp / "root"), "--models", "fake"])
    with np.load(suite / "tasks" / "DSA__reg.npz", allow_pickle=True) as z:
        y = np.asarray(z["y"], float)
    expected = 1.0 - len(y) / float(((y - y.mean()) ** 2).sum())   # every prediction is y + 1
    check("pooled out-of-fold R2", np.isclose(summary.loc[0, "reg_r2_pooled_oof"], expected))
    check("fold mean of Spearman", np.isclose(summary.loc[0, "reg_spearman"], 0.3))
    check("unknown model refused", raises(lambda: compare.main(["--root", str(tmp / "root"),
                                                                 "--models", "nobody"]), SystemExit))


def main() -> None:
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            repo = tmp / "GeneICL"
            suite = _write_geneicl(repo)
            layout = ExperimentLayout(tmp / "root")
            check_suite(tmp, repo, suite)
            check_embed(tmp, repo, suite, layout)
            check_probe(tmp, repo, layout)
            check_compare(tmp, repo, suite, layout)
    finally:
        EMBEDDERS.pop("_geneicl_fake", None)
    finish()


if __name__ == "__main__":
    main()
