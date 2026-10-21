"""Self-checks for the analysis orchestrator's config loading and planning.

Run directly; needs only PyYAML (no cluster, no GPU, no container, no checkpoints):

    python evaluate/check_analysis_plan.py

Exits non-zero on failure. Everything is exercised against temporary directories, so
it never touches a real ablation directory.
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate.analysis_config import (  # noqa: E402
    DEFAULT_STEPS,
    STEP_ORDER,
    deep_merge,
    interpolate,
    load_analysis_config,
)
from evaluate.run_analysis import (  # noqa: E402
    STEP_ENV,
    _explain_returncode,
    plan_experiment,
    resolve_sample_sizes,
    wrap_command,
)

FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}   {detail}")
        FAILED.append(label)


def write_cfg(tmp: Path, body: str) -> Path:
    path = tmp / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return path


BASE_CFG = """
vars:
  work: {work}
env:
  container_image: /images/bionemo.sif
  binds: [/cluster]
  conda_env: bulkFM
experiments:
  - name: exp_a
    ablation_dir: ${{work}}/abl_a
    adata_dir: ${{work}}/data_a
slurm:
  enabled: true
  mode: per-experiment
"""


def main() -> int:
    print("=== interpolation ===")
    variables = {"work": "/w", "nested": "/w/sub"}
    check("${var} expands", interpolate("${work}/x", variables) == "/w/x")
    check("nested expansion", interpolate("${nested}/y", variables) == "/w/sub/y")
    check("expands inside lists/dicts",
          interpolate({"a": ["${work}"]}, variables) == {"a": ["/w"]})
    try:
        interpolate("${nope}", {})
        check("unknown var raises", False)
    except KeyError:
        check("unknown var raises", True)

    print("\n=== deep_merge ===")
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    check("nested keys merge, not replace", merged == {"a": {"x": 1, "y": 9}, "b": 3},
          str(merged))
    check("lists replace wholesale",
          deep_merge({"s": [1, 2, 3]}, {"s": [9]})["s"] == [9])

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work = tmp / "work"
        (work / "abl_a").mkdir(parents=True)
        (work / "data_a").mkdir(parents=True)

        print("\n=== defaults + step ordering ===")
        cfg_path = write_cfg(tmp, BASE_CFG.format(work=work))
        cfg = load_analysis_config(cfg_path)
        exp = cfg.experiments[0]
        check("one experiment loaded", len(cfg.experiments) == 1)
        check("defaults applied", exp.build["modality_sample_size"] == 5000)
        check("ablation_dir interpolated", exp.ablation_dir == work / "abl_a")
        check("eval_adata derived", exp.eval_adata == work / "abl_a" / "eval.h5ad")
        check("analysis_dir under ablation_dir",
              exp.analysis_dir == work / "abl_a" / "analysis")

        steps, skipped = plan_experiment(exp, ROOT)
        names = [s.name for s in steps]
        check("default steps planned", names == list(DEFAULT_STEPS), str(names))
        check("paired_umap is opt-in, not default", "paired_umap" not in names)
        check("build runs (no eval.h5ad yet)", "build" in names)

        print("\n=== environment per step ===")
        by_name = {s.name: s for s in steps}
        expected = {"build": "container", "metrics": "container", "scib": "conda",
                    "diagnose": "plain", "umap": "container", "benchmark": "plain"}
        for name, want in expected.items():
            check(f"{name} -> {want}", by_name[name].env_kind == want,
                  by_name[name].env_kind)
        check("STEP_ENV covers every step", set(STEP_ENV) == set(STEP_ORDER))

        print("\n=== wrappers ===")
        c = wrap_command(by_name["build"], cfg.env, local=False)
        check("container step uses singularity", c[0] == "singularity", shlex.join(c[:3]))
        check("container step binds /cluster", "--bind" in c and "/cluster:/cluster" in c)
        check("container step passes --nv", "--nv" in c)
        s = wrap_command(by_name["scib"], cfg.env, local=False)
        check("conda step uses bash -lc", s[:2] == ["bash", "-lc"])
        check("conda step activates the env", "conda activate bulkFM" in s[2])
        pl = wrap_command(by_name["benchmark"], cfg.env, local=False)
        check("plain step runs python directly", pl[0] == "python")
        loc = wrap_command(by_name["build"], cfg.env, local=True)
        check("--local bypasses the container", loc[0] == "python", shlex.join(loc[:2]))

        print("\n=== step argv ===")
        # Inspect argv items, not the joined string: shlex.join quotes Windows paths.
        b_argv = by_name["build"].argv
        check("build writes eval.h5ad to the ablation dir",
              b_argv[b_argv.index("--out") + 1] == str(work / "abl_a" / "eval.h5ad"),
              b_argv[b_argv.index("--out") + 1])
        build_argv = shlex.join(b_argv)
        check("build passes --precomputed-pb", "--precomputed-pb" in build_argv)
        check("build passes the panel strategy",
              "--panel-strategy consensus" in build_argv)
        check("build omits --not-normalized when normalized: true",
              "--not-normalized" not in build_argv)
        metrics_argv = shlex.join(by_name["metrics"].argv)
        scib_argv = shlex.join(by_name["scib"].argv)
        check("metrics pass has no --scib", "--scib" not in metrics_argv)
        check("scib pass has --scib", "--scib " in scib_argv + " ")
        check("scib pass forwards n_max", "--scib-n-max 500" in scib_argv)
        check("both use the same script",
              by_name["metrics"].argv[0] == by_name["scib"].argv[0])
        umap_argv = shlex.join(by_name["umap"].argv)
        check("umap reads eval.h5ad", "--eval-adata" in umap_argv)
        check("umap drops --sample-size when reading eval.h5ad",
              "--sample-size" not in umap_argv, umap_argv)
        check("umap forwards --color", "--color tissue_general" in umap_argv)
        bench_argv = shlex.join(by_name["benchmark"].argv)
        check("benchmark defaults its output path", "benchmark.png" in bench_argv)
        check("benchmark passes --no-show", "--no-show" in bench_argv)

        print("\n=== paired_umap (opt-in step) ===")
        paired_h5ad = tmp / "paired_samples.h5ad"
        paired_h5ad.write_text("stub", encoding="utf-8")
        p_ok = tmp / "cfg_paired.yaml"
        p_ok.write_text(
            f"""
env: {{container_image: /i.sif, conda_env: e}}
experiments:
  - name: p
    ablation_dir: {(work / 'abl_a').as_posix()}
    steps: [paired_umap]
    paired_umap:
      input_h5ad: {paired_h5ad.as_posix()}
      n_cell_lines: 10
      no_sc: true
""",
            encoding="utf-8",
        )
        cfg_p = load_analysis_config(p_ok)
        exp_p = cfg_p.experiments[0]
        check("paired_umap needs no adata_dir", exp_p.adata_dir is None)
        steps_p, _ = plan_experiment(exp_p, ROOT)
        check("paired_umap is planned when requested",
              [s.name for s in steps_p] == ["paired_umap"], str([s.name for s in steps_p]))
        sp = steps_p[0]
        check("paired_umap runs in the container", sp.env_kind == "container")
        check("paired_umap does not depend on eval.h5ad", sp.needs_eval is False)
        pa = shlex.join(sp.argv)
        check("targets plot_paired_umap.py", "plot_paired_umap.py" in pa)
        check("passes --ablation-dir", "--ablation-dir" in pa)
        check("passes --input-h5ad", "--input-h5ad" in pa)
        check("defaults out-dir to {ablation_dir}/paired_umaps",
              sp.argv[sp.argv.index("--out-dir") + 1]
              == str(work / "abl_a" / "paired_umaps"),
              sp.argv[sp.argv.index("--out-dir") + 1])
        check("forwards --n-cell-lines", "--n-cell-lines 10" in pa)
        check("forwards --no-sc", "--no-sc" in pa)
        # Flags main() drops in --ablation-dir mode must not be sent.
        for dead in ("--out-prefix", "--domain-col", "--sc-label", "--bulk-label",
                     "--is-log1p", "--max-sc-per-line"):
            check(f"omits {dead} (ignored in ablation-dir mode)", dead not in pa)

        p_bad = tmp / "cfg_paired_bad.yaml"
        p_bad.write_text(
            f"experiments:\n  - name: p\n    ablation_dir: {(work / 'abl_a').as_posix()}\n"
            "    steps: [paired_umap]\n",
            encoding="utf-8",
        )
        try:
            load_analysis_config(p_bad)
            check("rejects paired_umap without input_h5ad", False)
        except KeyError:
            check("rejects paired_umap without input_h5ad", True)

        print("\n=== return-code explanation ===")
        # Runs during failure handling, so it must never raise -- including on
        # Windows, where signal.SIGKILL does not exist.
        name9, hint9 = _explain_returncode(-9, "build")
        check("SIGKILL is named", name9 in ("SIGKILL", "signal 9"), str(name9))
        check("SIGKILL hint mentions OOM", "OOM" in (hint9 or ""), str(hint9))
        check("SIGKILL hint is actionable for build",
              "modality_sample_size" in (hint9 or "")
              and "n_pca_components" in (hint9 or ""))
        name15, hint15 = _explain_returncode(-15, "umap")
        check("SIGTERM is named", name15 in ("SIGTERM", "signal 15"), str(name15))
        check("SIGTERM hint mentions the time limit", "time limit" in (hint15 or ""))
        check("ordinary failures get no signal", _explain_returncode(1, "x") == (None, None))
        check("success gets no signal", _explain_returncode(0, "x") == (None, None))
        check("unknown signal still names itself",
              _explain_returncode(-77, "x")[0] == "signal 77",
              str(_explain_returncode(-77, "x")))

        print("\n=== sample-size reconciliation ===")
        n, note = resolve_sample_sizes(exp, will_build=True)
        check("max() wins when umap needs more (6000 > 5000)", n == 6000, str(n))
        check("resolution is explained", "6000" in note and note != "", note)
        exp.umap["sample_size"] = 100
        n2, _ = resolve_sample_sizes(exp, will_build=True)
        check("build size kept when already larger", n2 == 5000, str(n2))
        exp.umap["source"] = "live"
        exp.umap["sample_size"] = 99999
        n3, note3 = resolve_sample_sizes(exp, will_build=True)
        check("live source leaves build size alone", n3 == 5000 and note3 == "", str(n3))
        exp.umap["source"] = "eval"
        exp.umap["sample_size"] = 99999
        n4, note4 = resolve_sample_sizes(exp, will_build=False)
        check("warns when build will not run", n4 == 5000 and "limited" in note4, note4)

        print("\n=== rebuild modes ===")
        cfg2 = load_analysis_config(cfg_path)
        e2 = cfg2.experiments[0]
        e2.eval_adata.write_text("stub", encoding="utf-8")   # pretend it exists
        steps_auto, skipped_auto = plan_experiment(e2, ROOT)
        check("rebuild auto skips build when eval.h5ad exists",
              "build" not in [s.name for s in steps_auto])
        check("skip reason mentions auto", "auto" in skipped_auto.get("build", ""))
        check("downstream steps still planned",
              "metrics" in [s.name for s in steps_auto])
        e2.build["rebuild"] = "always"
        check("rebuild always builds anyway",
              "build" in [s.name for s in plan_experiment(e2, ROOT)[0]])
        e2.eval_adata.unlink()
        e2.build["rebuild"] = "never"
        steps_never, skipped_never = plan_experiment(e2, ROOT)
        check("rebuild never + missing eval.h5ad skips eval-dependent steps",
              [s.name for s in steps_never] == ["benchmark"],
              str([s.name for s in steps_never]))
        check("each skip states the reason",
              all(skipped_never.get(k) for k in ("build", "metrics", "scib",
                                                 "diagnose", "umap")))
        e2.umap["source"] = "live"
        steps_live, _ = plan_experiment(e2, ROOT)
        check("umap with source: live survives a missing eval.h5ad",
              "umap" in [s.name for s in steps_live],
              str([s.name for s in steps_live]))

        print("\n=== per-experiment overrides ===")
        p3 = tmp / "cfg3.yaml"
        p3.write_text(
            f"""
vars:
  work: {work}
env: {{container_image: /i.sif, conda_env: e}}
defaults:
  build: {{modality_sample_size: 1234}}
experiments:
  - name: a
    ablation_dir: ${{work}}/abl_a
    adata_dir: ${{work}}/data_a
  - name: b
    ablation_dir: ${{work}}/abl_a
    steps: [diagnose, benchmark]
    build: {{rebuild: never}}
    metrics: {{group_column: cancer_type}}
""",
            encoding="utf-8",
        )
        cfg3 = load_analysis_config(p3)
        a, b = cfg3.experiments
        check("defaults override reaches experiments",
              a.build["modality_sample_size"] == 1234)
        check("per-experiment steps override", b.steps == ["diagnose", "benchmark"])
        check("per-experiment nested override", b.metrics["group_column"] == "cancer_type")
        check("sibling keeps the default",
              a.metrics["group_column"] == "tissue_general")
        check("adata_dir optional when neither build nor live umap run",
              b.adata_dir is None)

        print("\n=== validation ===")
        for body, why in [
            ("experiments: []", "empty experiments"),
            (f"experiments:\n  - name: a\n    ablation_dir: {work}\n"
             "    steps: [nope]", "unknown step"),
            (f"experiments:\n  - name: a\n    ablation_dir: {work}\n"
             "    build: {rebuild: sometimes}", "bad rebuild"),
            (f"experiments:\n  - name: a\n    ablation_dir: {work}\n"
             "    umap: {source: magic}", "bad umap source"),
            (f"experiments:\n  - name: a\n    ablation_dir: {work}\n"
             f"  - name: a\n    ablation_dir: {work}", "duplicate names"),
            (f"experiments:\n  - name: a\n    ablation_dir: {work}\n"
             "    steps: [build]", "build without adata_dir"),
            (f"slurm: {{mode: whenever}}\nexperiments:\n  - name: a\n"
             f"    ablation_dir: {work}\n    steps: [benchmark]", "bad slurm mode"),
        ]:
            p = tmp / "bad.yaml"
            p.write_text(body, encoding="utf-8")
            try:
                load_analysis_config(p)
                check(f"rejects {why}", False)
            except (KeyError, ValueError, TypeError):
                check(f"rejects {why}", True)

        print("\n=== dry run end to end ===")
        from evaluate.run_analysis import execute_experiment, submit
        cfg4 = load_analysis_config(cfg_path)
        rec = execute_experiment(cfg4.experiments[0], cfg4, None,
                                 dry_run=True, local=False)
        check("dry run plans every default step", len(rec["steps"]) == len(DEFAULT_STEPS),
              str(len(rec["steps"])))
        check("dry run runs nothing",
              all(s["status"] == "dry-run" for s in rec["steps"]))
        check("dry run writes no result file",
              not (cfg4.experiments[0].analysis_dir / "analysis_result.json").exists())
        check("every command is recorded", all("command" in s for s in rec["steps"]))
        check("record is JSON-serialisable", bool(json.dumps(rec)))

        # A dry run must not create anything: it is routinely used to inspect a
        # config whose paths belong to another machine, and mkdir(parents=True) on
        # those would build the whole tree locally.
        before = {p for p in cfg4.experiments[0].ablation_dir.rglob("*")}
        submit(cfg4, cfg4.experiments, None, "per-experiment", dry_run=True)
        submit(cfg4, cfg4.experiments, None, "single-job", dry_run=True)
        after = {p for p in cfg4.experiments[0].ablation_dir.rglob("*")}
        check("dry-run submit writes no payloads", before == after,
              str(sorted(p.name for p in after - before)))
        check("dry-run submit writes no manifest",
              not (cfg_path.parent / f"{cfg_path.stem}_manifest.json").exists())

        print("\n=== execution, failure handling and --payload ===")
        cfg5 = load_analysis_config(p3)
        exp_b = [e for e in cfg5.experiments if e.name == "b"][0]
        # Steps [diagnose, benchmark] on a stub eval.h5ad: the real CLIs will exit
        # non-zero, which is exactly the failure path being checked.
        exp_b.eval_adata.write_text("stub", encoding="utf-8")
        cfg5.env.python = sys.executable
        rec5 = execute_experiment(exp_b, cfg5, None, dry_run=False, local=True)
        check("failing steps are recorded, not raised", rec5["status"] == "failed")
        check("all steps attempted despite an early failure",
              len(rec5["steps"]) == 2, str(len(rec5["steps"])))
        check("failures carry a returncode",
              all("returncode" in s for s in rec5["steps"] if s["status"] == "failed"))
        check("result JSON written on a real run",
              (exp_b.analysis_dir / "analysis_result.json").exists())

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("ALL ANALYSIS-PLAN CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
