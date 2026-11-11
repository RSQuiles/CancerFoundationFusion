"""Self-check for evaluate/run_evaluation.py and evaluate/_preflight.py.

Offline and torch-free: fake step scripts (using ``parse_with_config`` like the real
ones) in a temporary package, run with this interpreter.

    python -m evaluate.check.check_run_evaluation
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from evaluate import run_evaluation as R  # noqa: E402

FAILURES: list[str] = []

FAKE_STEP = '''
import argparse, json, sys
from pathlib import Path
from evaluate._cli import parse_with_config


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None)
    p.add_argument("--out", required=False)
    p.add_argument("--model", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--fail", action="store_true")
    p.add_argument("--labels", nargs="*", default=None)
    return p


def main(argv=None):
    args = parse_with_config(build_parser(), "fake", argv)
    print("step running", args.model)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(vars(args), default=str) + "\\n")
    if args.fail:
        sys.exit(3)


if __name__ == "__main__":
    main()
'''

FAKE_CLI = '''
import argparse, json


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--items", nargs="+")
    p.add_argument("--flag", action="store_true")
    p.add_argument("--off", action="store_true")
    p.add_argument("--labels", nargs="*")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(vars(a)) + "\\n")
'''


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def _write_fakes(tmp: Path) -> None:
    pkg = tmp / "fakesteps"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "step.py").write_text(FAKE_STEP)
    (pkg / "clistep.py").write_text(FAKE_CLI)


def _config(tmp: Path, out: Path, **tasks) -> Path:
    cfg = {
        "project_root": str(tmp),
        "shared": {"root": str(tmp / "root"), "out": str(out)},
        "default_environment": "py",
        "environments": {"py": {
            "python": [sys.executable, "-u"],
            "env": {"PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), str(tmp)])},
        }},
        "tasks": tasks,
    }
    path = tmp / f"cfg_{len(list(tmp.glob('cfg_*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _main(*argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = R.main([str(a) for a in argv])
    return code, buf.getvalue()


def _records(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


def check_merge_and_expand() -> None:
    print("\n-- merge, foreach, selection --")
    cfg = {
        "shared": {"root": "/r", "models": ["top"]},
        "tasks": {
            "t1": {"shared": {"models": ["task"], "n": 1}, "steps": {
                "a": {"module": "m.a", "pass_as": "config", "args": {"n": 2},
                      "foreach": {"model": ["x", "y"], "n": [5]}},
                "b": {"module": "m.b", "drop": ["models"], "enabled": False},
            }},
            "t2": {"enabled": False, "steps": {"umap": {}}},
        },
    }
    envs = R.load_environments({})
    runs = R.build_runs(cfg, envs)
    a = [r for r in runs if r.step == "a"]
    check("foreach gives one run per combination", [r.args["model"] for r in a] == ["x", "y"])
    check("precedence: foreach > args > task shared > shared",
          a[0].args["n"] == 5 and a[0].args["models"] == ["task"] and a[0].args["root"] == "/r")
    b = next(r for r in runs if r.step == "b")
    check("drop removes a shared key", "models" not in b.args and b.args["n"] == 1)
    check("module steps default to cli", b.pass_as == "cli")
    umap = next(r for r in runs if r.task == "t2")
    check("alias resolves to its module", umap.module == "evaluate.representation.plot_umap")

    sel = R.select_runs(list(runs))
    check("disabled task/step left out by default", [r.step for r in sel] == ["a", "a"])
    sel = R.select_runs(list(runs), only=["t2", "t1.b"])
    check("--only overrides enabled: false", [r.name for r in sel] == ["t1.b", "t2.umap"])
    sel = R.select_runs(list(runs), only=["t1"], skip=["t1.a"])
    check("--skip removes a step", [r.name for r in sel] == ["t1.b"])
    sel = R.select_runs(list(runs), only=["t1", "t2"], start="t1.b")
    check("--from starts at the first match", [r.name for r in sel] == ["t1.b", "t2.umap"]
          and [r.index for r in sel] == [1, 2])

    def refused(c) -> bool:
        try:
            R.build_runs(c, envs)
        except R.ConfigError:
            return True
        return False

    check("unknown step key refused", refused({"tasks": {"t": {"steps": {"umap": {"argz": {}}}}}}))
    check("unknown alias refused", refused({"tasks": {"t": {"steps": {"nope": {}}}}}))
    check("unknown environment refused",
          refused({"tasks": {"t": {"steps": {"umap": {"environment": "nope"}}}}}))
    check("section-name key refused", refused({"tasks": {"t": {"steps": {"umap": {"args": {"pca": 1}}}}}}))
    try:
        R.select_runs(list(runs), only=["zzz"])
        check("unknown selector refused", False)
    except R.ConfigError:
        check("unknown selector refused", True)

    check("cli translation",
          R.to_cli_args({"stratify_by": ["a", "b"], "dry_run": True, "off": False, "x": None,
                         "labels": {"k": "V"}, "n": 3})
          == ["--stratify-by", "a", "b", "--dry-run", "--labels", "k=V", "--n", "3"])
    env = R.Environment("e", setup="conda activate x")
    cmd = env.argv("m", ["--config", "a b.yaml"])
    check("setup env runs through bash with quoting",
          cmd[:2] == ["bash", "-c"] and cmd[2].endswith("exec python -u -m m --config 'a b.yaml'"))


def check_execution(tmp: Path) -> None:
    print("\n-- execution --")
    _write_fakes(tmp)
    out = tmp / "out.jsonl"
    cfg = _config(tmp, out,
                  t1={"shared": {"labels": {"a": "A"}}, "steps": {
                      "s": {"module": "fakesteps.step", "pass_as": "config",
                            "foreach": {"model": ["m1", "m2"]}, "args": {"n": 7}},
                      "c": {"module": "fakesteps.clistep", "drop": ["root", "labels"],
                            "args": {"items": ["x", "y"], "flag": True, "off": False}},
                  }},
                  t2={"steps": {"s": {"module": "fakesteps.step", "pass_as": "config",
                                      "args": {"model": "late"}}}})
    run_dir = tmp / "run_ok"
    code, text = _main("--config", cfg, "--run-dir", run_dir)
    recs = _records(out)
    check("all steps succeed", code == 0, text[-2000:])
    check("steps run in order", [r.get("model") for r in recs] == ["m1", "m2", None, "late"], str(recs))
    check("config values reach the script", recs and recs[0]["n"] == 7 and recs[0]["labels"] == {"a": "A"}
          and recs[0]["root"] == str(tmp / "root"))
    check("cli step gets flags", len(recs) > 2 and recs[2]["items"] == ["x", "y"] and recs[2]["flag"]
          and not recs[2]["off"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    check("summary.json records every run", [s["status"] for s in summary["steps"]] == ["ok"] * 4)
    logs = sorted((run_dir / "logs").glob("*.log"))
    check("one log per run, holding the output",
          len(logs) == 4 and "step running m1" in logs[0].read_text(encoding="utf-8"))
    check("banners flag each step", "==== [1/4] t1.s (model=m1)" in text and "OK in" in text)

    # failure stops later steps unless continue_on_error
    out.unlink()
    cfg = _config(tmp, out, t={"steps": {
        "bad": {"module": "fakesteps.step", "pass_as": "config", "args": {"fail": True}},
        "after": {"module": "fakesteps.step", "pass_as": "config", "args": {"model": "after"}}}})
    code, text = _main("--config", cfg, "--run-dir", tmp / "run_fail")
    summary = json.loads((tmp / "run_fail" / "summary.json").read_text(encoding="utf-8"))
    check("failure stops the run", code == 1 and [s["status"] for s in summary["steps"]] == ["failed", "not_run"]
          and summary["steps"][0]["returncode"] == 3)
    out.unlink()
    code, text = _main("--config", cfg, "--run-dir", tmp / "run_cont", "--continue-on-error")
    check("--continue-on-error runs the rest", code == 1 and [r.get("model") for r in _records(out)] == [None, "after"])

    # preflight catches an unknown key before anything runs
    out.unlink()
    cfg = _config(tmp, out, t={"steps": {
        "good": {"module": "fakesteps.step", "pass_as": "config"},
        "typo": {"module": "fakesteps.step", "pass_as": "config", "args": {"modle": "x"}}}})
    code, text = _main("--config", cfg, "--run-dir", tmp / "run_pre")
    check("preflight refuses an unknown key, nothing runs", code == 1 and not out.exists()
          and "modle" in text, text[-1500:])
    cfg = _config(tmp, out, t={"steps": {
        "badtype": {"module": "fakesteps.step", "pass_as": "config", "args": {"n": "many"}}}})
    code, _ = _main("--config", cfg, "--preflight-only", "--run-dir", tmp / "run_pre2")
    check("preflight refuses a wrong type", code == 1 and not out.exists())

    # dry run writes step configs only
    cfg = _config(tmp, out, t={"steps": {"s": {"module": "fakesteps.step", "pass_as": "config"}}})
    code, text = _main("--config", cfg, "--dry-run", "--run-dir", tmp / "run_dry")
    step_files = list((tmp / "run_dry" / "steps").glob("*.yaml"))
    check("dry run runs nothing and writes the step configs", code == 0 and not out.exists()
          and len(step_files) == 1 and "fakesteps.step" in text)


def check_example() -> None:
    print("\n-- example config --")
    path = PROJECT_ROOT / "evaluate" / "configs" / "example_evaluation.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    envs = R.load_environments(cfg["environments"])
    runs = R.build_runs(cfg, envs)
    check("example config builds", len(runs) > 0)
    check("example covers the four tasks",
          {"representation", "alignment", "reconstruction", "geneicl"} <= {r.task for r in runs})


def main() -> int:
    check_merge_and_expand()
    check_example()
    with tempfile.TemporaryDirectory() as tmp:
        check_execution(Path(tmp))
    print(f"\n{'All checks passed.' if not FAILURES else f'{len(FAILURES)} check(s) FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
