"""Self-check for ``evaluate/finetune/scripts/run_ablation_survboard_metrics.py``.

Runs anywhere — no cluster, GPU, checkpoint or SurvBoard environment. Everything
that needs pycox is reached through one lazy import inside ``run_job``, which is
stubbed here, so config loading, model discovery, storage naming, the orphan probe
and the status handling are all exercised offline against a synthetic tree.

    python evaluate/check/check_survboard_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evaluate" / "finetune" / "scripts" / "run_ablation_survboard_metrics.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPT.parent))

import run_ablation_survboard_metrics as S  # noqa: E402

FAILED = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILED.append(name)


def make_tree(root: Path) -> None:
    for abl, models in (
        ("abl_a", ["unified", "dat", "pca_baseline", "notamodel"]),
        ("abl_b", ["unified", "mmd"]),
    ):
        for m in models:
            d = root / abl / m
            d.mkdir(parents=True)
            if m == "pca_baseline":
                (d / "metrics").mkdir()
            elif m != "notamodel":
                (d / f"step_step=1_epoch_epoch=00.ckpt").write_text("x")
    (root / "abl_a" / "loose_file.txt").write_text("x")
    for p in ("data", "splits", "functions"):
        (root / p).mkdir()


def write_task_cfg(path: Path, root: Path) -> None:
    path.write_text(f"""
finetune:
  survival:
    pretrained_model_path: /nope/unified/x.ckpt
    ablation_dir: /nope
    fold_index: 5
    survboard_data_dir: {(root / 'data').as_posix()}
    splits_dir: {(root / 'splits').as_posix()}
    survboard_results_dir: {(root / 'functions').as_posix()}
    cohorts: [TCGA, ICGC]
    cancer_types: [BRCA, LUAD, PAAD, KIRC, LGG, LIHC, UCEC, BRCA]
    epochs: 30
""", encoding="utf-8")


def write_sweep_cfg(path: Path, root: Path, task_cfg: Path, extra: str = "") -> None:
    path.write_text(f"""
vars:
  work: {root.as_posix()}
task_config: {task_cfg.as_posix()}
defaults:
  ibs_grid_len: 50
{extra}
ablations:
  - name: a
    ablation_dir: ${{work}}/abl_a
  - name: b
    ablation_dir: ${{work}}/abl_b
  - name: missing
    ablation_dir: ${{work}}/does_not_exist
""", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_tree(root)
        task_cfg = root / "survival_pred_config.yaml"
        sweep = root / "sweep.yaml"
        write_task_cfg(task_cfg, root)
        write_sweep_cfg(sweep, root, task_cfg)

        print("\n-- config loading --")
        jobs = S.load_jobs(sweep)
        check("three jobs", len(jobs) == 3)
        a = jobs[0]
        check("name", a.name == "a")
        check("${work} interpolated", a.ablation_dir == root / "abl_a")
        check("inherits data_dir from task_config", a.data_dir == root / "data")
        check("inherits results_dir", a.results_dir == root / "functions")
        check("inherits cohorts", a.cohorts == ["TCGA", "ICGC"])
        check("dedupes cancer types (BRCA listed twice)",
              a.cancer_types == ["BRCA", "LUAD", "PAAD", "KIRC", "LGG", "LIHC", "UCEC"])
        check("defaults override task_config", a.ibs_grid_len == 50)
        check("models default None", a.models is None)
        check("skip_existing default False", a.skip_existing is False)

        print("\n-- per-ablation override --")
        sweep2 = root / "sweep2.yaml"
        sweep2.write_text(f"""
task_config: {task_cfg.as_posix()}
ablations:
  - name: a
    ablation_dir: {(root / 'abl_a').as_posix()}
    survboard_results_dir: {(root / 'functions').as_posix()}/a
    models: [unified]
    ibs_grid_len: 7
""", encoding="utf-8")
        j2 = S.load_jobs(sweep2)[0]
        check("results_dir overridden", j2.results_dir == root / "functions" / "a")
        check("models overridden", j2.models == ["unified"])
        check("ibs_grid_len overridden", j2.ibs_grid_len == 7)

        print("\n-- validation errors --")
        for body, why in (
            ("ablations: []", "empty ablations"),
            ("ablations:\n  - name: a", "missing ablation_dir"),
            ("ablations:\n  - {name: a, ablation_dir: /x}\n  - {name: a, ablation_dir: /y}",
             "duplicate name"),
        ):
            bad = root / "bad.yaml"
            bad.write_text(f"task_config: {task_cfg.as_posix()}\n{body}\n", encoding="utf-8")
            try:
                S.load_jobs(bad)
                check(f"rejects {why}", False)
            except (KeyError, ValueError):
                check(f"rejects {why}", True)

        bad = root / "bad2.yaml"
        bad.write_text("ablations:\n  - {name: a, ablation_dir: /x}\n", encoding="utf-8")
        try:
            S.load_jobs(bad)
            check("rejects missing required dirs", False)
        except KeyError as e:
            check("rejects missing required dirs",
                  "survboard_data_dir" in str(e))

        print("\n-- discovery --")
        models_a = S.discover_models(root / "abl_a")
        check("finds ckpt dirs", "unified" in models_a and "dat" in models_a)
        check("includes pca_baseline (metrics/, no ckpt)", "pca_baseline" in models_a)
        check("excludes empty dir", "notamodel" not in models_a)
        check("ignores loose files", len(models_a) == 3)
        check("missing dir -> []", S.discover_models(root / "nope") == [])

        print("\n-- skip-existing detection --")
        mdir = root / "abl_a" / "unified" / "metrics"
        mdir.mkdir(exist_ok=True)
        (mdir / "results_survival.json").write_text(json.dumps({"c_index": 0.6}))
        check("c_index only -> not done", not S._has_metrics(root / "abl_a", "unified"))
        (mdir / "results_survival.json").write_text(
            json.dumps({"c_index": 0.6, "antolini_concordance": 0.61}))
        check("survboard metrics -> done", S._has_metrics(root / "abl_a", "unified"))
        (mdir / "results_survival.json").write_text("{not json")
        check("corrupt json -> not done", not S._has_metrics(root / "abl_a", "unified"))
        (mdir / "results_survival.json").unlink()

        print("\n-- storage names --")
        check("prefixed with the ablation dir basename",
              S.job_storage_name(jobs[0], "unified") == "abl_a_unified")
        check("pca_baseline too",
              S.job_storage_name(jobs[0], "pca_baseline") == "abl_a_pca_baseline")
        check("same model, different ablation -> different directory",
              S.job_storage_name(jobs[0], "unified")
              != S.job_storage_name(jobs[1], "unified"))

        print("\n-- collisions --")
        plan = [(jobs[0], ["unified", "dat"]), (jobs[1], ["unified", "mmd"])]
        check("shared model name across ablations no longer collides",
              S.check_collisions(plan) == 0)
        # Same basename under two roots is the one case that still collides.
        (root / "other").mkdir()
        (root / "other" / "abl_a").mkdir()
        j_dup = S.replace(jobs[1], name="a_dup", ablation_dir=root / "other" / "abl_a")
        check("same ablation basename under two roots collides",
              S.check_collisions([(jobs[0], ["unified"]), (j_dup, ["unified"])]) == 1)
        j_alt = S.replace(j_dup, results_dir=root / "functions" / "b")
        check("...unless the results dirs differ",
              S.check_collisions([(jobs[0], ["unified"]), (j_alt, ["unified"])]) == 0)

        print("\n-- orphan probe --")
        check("nothing on a clean tree", S.find_orphans(jobs[0], ["unified", "dat"]) == [])
        legacy = root / "functions" / "TCGA" / "LUAD" / "unified"
        legacy.mkdir(parents=True)
        (legacy / "split_5.csv").write_text("x")
        check("finds pre-cutover bare-name CSVs",
              S.find_orphans(jobs[0], ["unified", "dat"]) == ["unified"])
        new_layout = root / "functions" / "TCGA" / "LUAD" / "abl_a_dat"
        new_layout.mkdir(parents=True)
        (new_layout / "split_5.csv").write_text("x")
        check("new-layout CSVs are not reported as orphans",
              S.find_orphans(jobs[0], ["dat"]) == [])
        empty = root / "functions" / "TCGA" / "PAAD" / "mmd"
        empty.mkdir(parents=True)
        check("a directory without split_*.csv is not an orphan",
              S.find_orphans(jobs[0], ["mmd"]) == [])

        print("\n-- missing paths --")
        check("reports missing ablation_dir", len(S._missing_paths(jobs[2])) == 1)
        check("all present -> none", S._missing_paths(jobs[0]) == [])

        print("\n-- run_job with a stubbed evaluator --")
        calls = []

        def fake_evaluate_all(**kw):
            calls.append(kw)
            if kw["model_name"] == "dat":
                sys.exit(1)              # no CSVs found
            if kw["model_name"] == "mmd":
                raise RuntimeError("boom")

        stub = types.ModuleType("evaluate.finetune.tasks.evaluate_survboard_metrics")
        stub.evaluate_all = fake_evaluate_all
        sys.modules["evaluate.finetune.tasks.evaluate_survboard_metrics"] = stub

        statuses = S.run_job(jobs[0], ["unified", "dat", "mmd"])
        check("ok recorded", statuses["unified"] == "ok")
        check("SystemExit -> no-data", statuses["dat"] == "no-data")
        check("exception -> fail", statuses["mmd"] == "fail")
        check("kwargs forwarded", calls[0]["ibs_grid_len"] == 50
              and calls[0]["cohorts"] == ["TCGA", "ICGC"]
              and calls[0]["ablation_dir"] == (root / "abl_a").resolve())
        check("storage_name passed, distinct from model_name",
              calls[0]["storage_name"] == "abl_a_unified"
              and calls[0]["model_name"] == "unified")

        (mdir).mkdir(exist_ok=True)
        (mdir / "results_survival.json").write_text(
            json.dumps({"antolini_concordance": 0.61}))
        st = S.run_job(S.replace(jobs[0], skip_existing=True), ["unified"])
        check("skip_existing honoured", st["unified"] == "skip")

        print("\n-- CLI --")
        for args, want in (
            (["--dry-run"], "MISSING PATHS"),
            (["--list"], "pca_baseline"),
            (["--dry-run", "--only", "b"], "abl_b"),
            (["--dry-run", "--only", "a"], "abl_a_unified"),      # resolved storage name
            (["--dry-run", "--only", "a"], "ORPHANED CSVs"),      # written just above
        ):
            r = subprocess.run([sys.executable, str(SCRIPT), "--config", str(sweep)] + args,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            check(f"CLI {' '.join(args)} rc=0", r.returncode == 0)
            check(f"CLI {' '.join(args)} mentions {want!r}", want in r.stdout + r.stderr)

        r = subprocess.run([sys.executable, str(SCRIPT), "--config", str(sweep),
                            "--dry-run", "--only", "nope"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        check("CLI unknown --only fails", r.returncode == 1)

        r = subprocess.run([sys.executable, str(SCRIPT), "--config", str(sweep), "--dry-run"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        check("no collision now that names carry the ablation",
              "COLLISION" not in r.stdout + r.stderr)

    print("\n" + ("FAILURES: " + ", ".join(FAILED) if FAILED else "All checks passed."))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
