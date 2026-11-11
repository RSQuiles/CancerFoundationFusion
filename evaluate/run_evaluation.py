"""Run every evaluation step of every task from ONE config, sequentially.

    python -m evaluate.run_evaluation --config evaluate/configs/example_evaluation.yaml
    python -m evaluate.run_evaluation --config <cfg> --only representation reconstruction.plot
    python -m evaluate.run_evaluation --config <cfg> --dry-run          # write step configs, print commands
    sbatch evaluate/run_evaluation.sh                                     # the cluster entry point

The coordinator knows no script option. Each step's options are passed through as they
are: they are merged (top ``shared`` < task ``shared`` < step ``args`` < ``foreach``
values), written to a flat YAML and handed to the script as ``--config <file>``, which
``evaluate._cli.parse_with_config`` reads and validates (unknown keys are errors). A new
or renamed script option is therefore fixed in the config alone, and a new task, step or
environment is a new config block.

Config::

    project_root: /path/to/CancerFoundationFusion   # cwd of every step (default: this repo)
    shared: {root: /path/to/experiments}           # passed to every step
    default_environment: container
    environments:
      container: {command: [singularity, run, --nv, ..., image.sif, python, -u]}
      bulkFM:    {setup: "source ~/.bashrc; conda activate bulkFM || exit 1"}
      host:      {}                                # `python` on PATH
    tasks:                                         # run in this order
      <task>:
        enabled: true
        shared: {...}                              # every step of this task
        steps:                                     # run in this order
          <step>:
            script: <alias>                        # default: the step key (see SCRIPTS)
            module: <python module>                # instead of an alias, for any script
            environment: <name>
            args: {...}                            # the script's own options
            foreach: {model: [a, b]}               # one run per combination
            drop: [models]                         # shared keys this step must not get
            argv: [--flag]                         # raw extra arguments
            pass_as: config | cli                  # cli: options become flags
            enabled: true
            continue_on_error: false

An environment is ``command`` (argv that starts python; ``-m <module> ...`` is appended)
or ``setup`` (shell lines run by ``bash`` before ``exec <python> -m <module> ...``,
``python`` naming the interpreter, default ``python -u``), plus optional ``env`` (extra
environment variables). ``{}`` runs ``python -u`` from PATH.

Every run gets a directory, ``<root>/results/_eval_runs/<config stem>_<timestamp>/``
(``--run-dir`` overrides), holding the config, each step's YAML under ``steps/``, its
output under ``logs/`` and ``summary.json``. Before anything runs, every step's arguments
are parsed by the script's own parser in its own environment (``evaluate._preflight``;
``--no-preflight`` skips it), so a typo fails in seconds instead of after the GPU steps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Script aliases a step may use (its key, or `script:`). Any other script: `module:`.
SCRIPTS: dict[str, str] = {
    # representation
    "umap": "evaluate.representation.plot_umap",
    "plot_umap": "evaluate.representation.plot_umap",
    "pca": "evaluate.representation.plot_pca",
    "plot_pca": "evaluate.representation.plot_pca",
    "alignment": "evaluate.representation.plot_alignment",
    "plot_alignment": "evaluate.representation.plot_alignment",
    # alignment (scIB)
    "scib": "evaluate.alignment.scib_benchmark",
    "scib_benchmark": "evaluate.alignment.scib_benchmark",
    # reconstruction
    "mask_plan": "evaluate.reconstruction.mask_plan",
    "reconstruction": "evaluate.reconstruction.run_reconstruction",
    "run_reconstruction": "evaluate.reconstruction.run_reconstruction",
    "compare": "evaluate.reconstruction.compare",
    "reconstruction_compare": "evaluate.reconstruction.compare",
    "reconstruction_plot": "evaluate.reconstruction.plot_reconstruction",
    "plot_reconstruction": "evaluate.reconstruction.plot_reconstruction",
    # geneicl
    "geneicl_embed": "evaluate.geneicl.embed_suite",
    "embed_suite": "evaluate.geneicl.embed_suite",
    "geneicl_probe": "evaluate.geneicl.run_probe",
    "run_probe": "evaluate.geneicl.run_probe",
    "geneicl_compare": "evaluate.geneicl.compare",
    "geneicl_plot": "evaluate.geneicl.plot_benchmark",
    "plot_benchmark": "evaluate.geneicl.plot_benchmark",
    # embeddings (prerequisite of representation and alignment)
    "embed_splits": "embed.embed_splits",
}

TOP_KEYS = {"project_root", "shared", "environments", "default_environment", "tasks"}
ENV_KEYS = {"command", "setup", "python", "env"}
TASK_KEYS = {"enabled", "shared", "steps"}
STEP_KEYS = {"script", "module", "environment", "args", "foreach", "drop", "argv", "pass_as",
             "enabled", "continue_on_error"}
PREFLIGHT_MODULE = "evaluate._preflight"
RULE = "=" * 100


class ConfigError(ValueError):
    pass


@dataclass
class Environment:
    name: str
    command: list[str] | None = None
    setup: str | None = None
    python: list[str] = field(default_factory=lambda: ["python", "-u"])
    env: dict[str, str] = field(default_factory=dict)

    def argv(self, module: str, rest: list[str]) -> list[str]:
        """The full command running ``python -m module *rest`` in this environment."""
        if self.command is not None:
            return [*self.command, "-m", module, *rest]
        launch = [*self.python, "-m", module, *rest]
        if self.setup is None:
            return launch
        return ["bash", "-c", f"{self.setup.rstrip()}\nexec {shlex.join(launch)}"]


@dataclass
class Run:
    """One execution of one step (one ``foreach`` combination)."""

    index: int
    task: str
    step: str
    label: str              # "model=a" for a foreach combination, else ""
    module: str
    environment: str
    args: dict[str, Any]
    argv: list[str]
    pass_as: str
    continue_on_error: bool
    enabled: bool = True     # False: left out unless --only names it
    config_path: Path | None = None
    log_path: Path | None = None
    status: str = "pending"  # pending | ok | failed | not_run | interrupted
    returncode: int | None = None
    seconds: float | None = None
    command: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.task}.{self.step}"

    @property
    def title(self) -> str:
        return f"{self.name} ({self.label})" if self.label else self.name

    @property
    def file_stem(self) -> str:
        stem = f"{self.index:02d}_{self.name}"
        if self.label:
            stem += "." + "".join(c if c.isalnum() or c in "-_=." else "_" for c in self.label.replace(" ", "."))
        return stem


# ---------------------------------------------------------------------------
# config -> runs
# ---------------------------------------------------------------------------

def _mapping(value, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _unknown(keys, allowed: set[str], where: str) -> None:
    extra = sorted(set(keys) - allowed)
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def _norm(key: str) -> str:
    return str(key).replace("-", "_")


def load_environments(raw: dict) -> dict[str, Environment]:
    envs: dict[str, Environment] = {}
    for name, spec in _mapping(raw, "environments").items():
        spec = _mapping(spec, f"environments.{name}")
        _unknown(spec, ENV_KEYS, f"environments.{name}")
        if "command" in spec and ("setup" in spec or "python" in spec):
            raise ConfigError(f"environments.{name}: give 'command', or 'setup' (+ 'python'), not both")
        command = spec.get("command")
        if command is not None:
            if isinstance(command, str):
                command = shlex.split(command)
            command = [str(c) for c in command]
        python = spec.get("python", ["python", "-u"])
        python = shlex.split(python) if isinstance(python, str) else [str(p) for p in python]
        envs[name] = Environment(
            name=name, command=command, setup=spec.get("setup"), python=python,
            env={str(k): str(v) for k, v in _mapping(spec.get("env"), f"environments.{name}.env").items()},
        )
    envs.setdefault("host", Environment(name="host"))
    return envs


def _foreach_combinations(foreach, where: str) -> list[dict[str, Any]]:
    foreach = _mapping(foreach, f"{where}.foreach")
    if not foreach:
        return [{}]
    keys = list(foreach)
    values = [v if isinstance(v, list) else [v] for v in foreach.values()]
    if any(not v for v in values):
        raise ConfigError(f"{where}.foreach: every key needs at least one value")
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _label(combo: dict[str, Any]) -> str:
    def show(v):
        return ",".join(map(str, v)) if isinstance(v, list) else str(v)
    return " ".join(f"{k}={show(v)}" for k, v in combo.items())


def build_runs(cfg: dict, envs: dict[str, Environment]) -> list[Run]:
    """Every step run of the config, in order, including disabled ones (``enabled`` False
    marks them; selection happens in :func:`select_runs`)."""
    from evaluate._cli import SECTIONS

    top_shared = _mapping(cfg.get("shared"), "shared")
    default_env = cfg.get("default_environment", "host")
    if default_env not in envs:
        raise ConfigError(f"default_environment {default_env!r} is not defined; known: {sorted(envs)}")

    runs: list[Run] = []
    tasks = _mapping(cfg.get("tasks"), "tasks")
    if not tasks:
        raise ConfigError("the config defines no tasks")
    for task, tspec in tasks.items():
        tspec = _mapping(tspec, f"tasks.{task}")
        _unknown(tspec, TASK_KEYS, f"tasks.{task}")
        task_on = tspec.get("enabled", True) is not False
        task_shared = _mapping(tspec.get("shared"), f"tasks.{task}.shared")
        steps = _mapping(tspec.get("steps"), f"tasks.{task}.steps")
        if not steps:
            raise ConfigError(f"tasks.{task} has no steps")
        for step, sspec in steps.items():
            where = f"tasks.{task}.steps.{step}"
            sspec = _mapping(sspec, where)
            _unknown(sspec, STEP_KEYS, where)
            if "module" in sspec and "script" in sspec:
                raise ConfigError(f"{where}: give 'script' or 'module', not both")
            if "module" in sspec:
                module = str(sspec["module"])
                default_pass = "cli"
            else:
                alias = str(sspec.get("script", step))
                if alias not in SCRIPTS:
                    raise ConfigError(f"{where}: unknown script {alias!r} (set 'script' to one of "
                                      f"{sorted(SCRIPTS)}, or 'module' to any python module)")
                module = SCRIPTS[alias]
                default_pass = "config"
            pass_as = sspec.get("pass_as", default_pass)
            if pass_as not in ("config", "cli"):
                raise ConfigError(f"{where}.pass_as must be 'config' or 'cli'")
            env = sspec.get("environment", default_env)
            if env not in envs:
                raise ConfigError(f"{where}: environment {env!r} is not defined; known: {sorted(envs)}")
            drop = {_norm(k) for k in (sspec.get("drop") or [])}
            extra = sspec.get("argv") or []
            if isinstance(extra, str):
                extra = shlex.split(extra)
            step_on = task_on and sspec.get("enabled", True) is not False

            for combo in _foreach_combinations(sspec.get("foreach"), where):
                merged: dict[str, Any] = {}
                for layer in (top_shared, task_shared, _mapping(sspec.get("args"), f"{where}.args"), combo):
                    for k, v in layer.items():
                        merged.pop(next((m for m in merged if _norm(m) == _norm(k)), None), None)
                        merged[k] = v
                merged = {k: v for k, v in merged.items() if _norm(k) not in drop}
                clash = sorted(k for k in merged if _norm(k) in SECTIONS)
                if clash and pass_as == "config":
                    raise ConfigError(f"{where}: key(s) {clash} are section names of evaluate/_cli.py "
                                      "and would be ignored by parse_with_config")
                run = Run(index=0, task=str(task), step=str(step), label=_label(combo), module=module,
                          environment=env, args=merged, argv=[str(a) for a in extra], pass_as=pass_as,
                          continue_on_error=bool(sspec.get("continue_on_error", False)), enabled=step_on)
                runs.append(run)
    return runs


def _matches(run: Run, selector: str) -> bool:
    return selector in (run.task, run.name)


def select_runs(runs: list[Run], only=None, skip=None, start=None) -> list[Run]:
    """``only``: tasks or ``task.step`` (overrides ``enabled: false``); ``skip``: the same,
    removed; ``start``: drop everything before the first match. Indices are renumbered."""
    for sel in [*(only or []), *(skip or []), *([start] if start else [])]:
        if not any(_matches(r, sel) for r in runs):
            known = sorted({r.task for r in runs} | {r.name for r in runs})
            raise ConfigError(f"selector {sel!r} matches no task or step; known: {known}")
    if only:
        chosen = [r for r in runs if any(_matches(r, s) for s in only)]
    else:
        chosen = [r for r in runs if r.enabled]
    chosen = [r for r in chosen if not any(_matches(r, s) for s in (skip or []))]
    if start:
        first = next((i for i, r in enumerate(chosen) if _matches(r, start)), None)
        if first is None:
            raise ConfigError(f"--from {start!r} is not among the selected steps")
        chosen = chosen[first:]
    for i, run in enumerate(chosen, 1):
        run.index = i
    return chosen


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------

def to_cli_args(values: dict[str, Any]) -> list[str]:
    """``{"stratify_by": ["a", "b"], "dry_run": True, "labels": {"x": "X"}}`` ->
    ``["--stratify-by", "a", "b", "--dry-run", "--labels", "x=X"]``. ``False`` and
    ``None`` are left out."""
    out: list[str] = []
    for key, value in values.items():
        flag = "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, dict):
            out += [flag, *(f"{k}={v}" for k, v in value.items())]
        elif isinstance(value, list):
            out += [flag, *map(str, value)]
        else:
            out += [flag, str(value)]
    return out


def script_args(run: Run) -> list[str]:
    if run.pass_as == "config":
        return ["--config", str(run.config_path), *run.argv]
    return [*to_cli_args(run.args), *run.argv]


def write_step_config(run: Run, steps_dir: Path) -> None:
    import yaml

    run.config_path = steps_dir / f"{run.file_stem}.yaml"
    header = (f"# {run.title}: {run.module}, environment {run.environment}\n"
              f"# written by evaluate.run_evaluation; passed as "
              f"{'--config' if run.pass_as == 'config' else 'CLI flags'}\n")
    run.config_path.write_text(header + yaml.safe_dump(run.args, sort_keys=False, default_flow_style=None),
                               encoding="utf-8")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _child_env(env: Environment) -> dict[str, str]:
    out = dict(os.environ)
    out.update(env.env)
    out["PYTHONUNBUFFERED"] = "1"
    return out


def execute(command: list[str], env: Environment, cwd: Path, log_path: Path | None,
            echo: bool = True) -> tuple[int, str]:
    """Run ``command``, streaming its output to stdout (``echo``) and ``log_path``.
    Returns the exit code and the last lines of output."""
    tail: list[str] = []
    log = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=_child_env(env), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
    except FileNotFoundError as exc:
        msg = f"cannot start {command[0]!r}: {exc}"
        if log:
            log.write(msg + "\n")
            log.close()
        return 127, msg
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            if log:
                log.write(line)
                log.flush()
            tail.append(line.rstrip("\n"))
            del tail[:-20]
        return proc.wait(), "\n".join(tail)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        if log:
            log.close()


def print_plan(runs: list[Run]) -> None:
    print(RULE)
    print(f"Evaluation plan: {len(runs)} step run(s)")
    print(RULE)
    width = max([len(r.title) for r in runs] + [4])
    for r in runs:
        print(f"  [{r.index:>2}/{len(runs)}] {r.title:<{width}}  env={r.environment:<12} {r.module}")
    print()


def preflight(runs: list[Run], envs: dict[str, Environment], cwd: Path) -> bool:
    print(RULE)
    print("Preflight: parsing every step's arguments with its own script, in its own environment")
    print(RULE)
    ok = True
    for r in runs:
        rest = [r.module, str(r.config_path), "--pass-as", r.pass_as, "--", *r.argv]
        code, tail = execute(envs[r.environment].argv(PREFLIGHT_MODULE, rest), envs[r.environment], cwd,
                             None, echo=False)
        mark = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"  [{r.index:>2}/{len(runs)}] {r.title}: {mark}")
        if code != 0:
            ok = False
            print("      " + "\n      ".join(tail.splitlines()[-8:]))
    print()
    return ok


def run_all(runs: list[Run], envs: dict[str, Environment], cwd: Path, logs_dir: Path,
            continue_on_error: bool) -> None:
    total = len(runs)
    stop = False
    for r in runs:
        if stop:
            r.status = "not_run"
            continue
        env = envs[r.environment]
        r.command = env.argv(r.module, script_args(r))
        r.log_path = logs_dir / f"{r.file_stem}.log"
        print(RULE)
        print(f"==== [{r.index}/{total}] {r.title} | env: {r.environment}")
        print(f"     module : {r.module}")
        print(f"     config : {r.config_path}")
        print(f"     log    : {r.log_path}")
        print(f"     command: {shlex.join(r.command)}")
        print(f"     started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(RULE, flush=True)
        t0 = time.monotonic()
        try:
            code, _ = execute(r.command, env, cwd, r.log_path)
        except KeyboardInterrupt:
            r.status, r.seconds = "interrupted", time.monotonic() - t0
            print(f"\n[INTERRUPTED] {r.title} after {_duration(r.seconds)}")
            for later in runs[r.index:]:
                later.status = "not_run"
            raise
        r.returncode, r.seconds = code, time.monotonic() - t0
        if code == 0:
            r.status = "ok"
            print(f"---- [{r.index}/{total}] {r.title}: OK in {_duration(r.seconds)}\n", flush=True)
        else:
            r.status = "failed"
            print(f"---- [{r.index}/{total}] {r.title}: FAILED (exit {code}) after "
                  f"{_duration(r.seconds)} -> {r.log_path}\n", flush=True)
            if not (continue_on_error or r.continue_on_error):
                stop = True
                print("Stopping: later steps are not run (--continue-on-error to go on).\n")


def print_summary(runs: list[Run]) -> None:
    print(RULE)
    print("Evaluation summary")
    print(RULE)
    width = max([len(r.title) for r in runs] + [4])
    for r in runs:
        took = _duration(r.seconds) if r.seconds is not None else "-"
        status = r.status.upper() + (f" (exit {r.returncode})" if r.status == "failed" else "")
        print(f"  [{r.index:>2}] {r.title:<{width}}  {status:<16} {took:>10}  {r.log_path or ''}")
    counts = {s: sum(r.status == s for r in runs) for s in ("ok", "failed", "not_run", "interrupted")}
    print("  " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))


def write_summary(path: Path, config: Path, runs: list[Run], started: datetime, dry_run: bool) -> None:
    summary = {
        "config": str(config),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "steps": [
            {"index": r.index, "task": r.task, "step": r.step, "label": r.label, "module": r.module,
             "environment": r.environment, "pass_as": r.pass_as, "status": r.status,
             "returncode": r.returncode, "seconds": r.seconds, "config": str(r.config_path),
             "log": str(r.log_path) if r.log_path else None, "command": r.command}
            for r in runs
        ],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the evaluation steps of one config, sequentially.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Global evaluation YAML")
    p.add_argument("--only", nargs="+", metavar="TASK[.STEP]",
                   help="Run only these tasks/steps (also ones with enabled: false)")
    p.add_argument("--skip", nargs="+", metavar="TASK[.STEP]", help="Leave these tasks/steps out")
    p.add_argument("--from", dest="start", metavar="TASK[.STEP]",
                   help="Start at this task/step, dropping the selected ones before it")
    p.add_argument("--list", action="store_true", help="Print the plan and exit (writes nothing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the step configs and print the commands; run nothing")
    p.add_argument("--no-preflight", action="store_true", help="Do not check the arguments first")
    p.add_argument("--preflight-only", action="store_true",
                   help="Check every step's arguments in its environment, then stop")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Run the later steps after a failure")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Where configs, logs and summary.json go "
                        "(default: <root>/results/_eval_runs/<config stem>_<timestamp>; a temp dir with --dry-run)")
    return p


def main(argv: list[str] | None = None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    try:
        _unknown(_mapping(cfg, str(args.config)), TOP_KEYS, str(args.config))
        envs = load_environments(cfg.get("environments"))
        runs = select_runs(build_runs(cfg, envs), args.only, args.skip, args.start)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if not runs:
        print("Nothing to run: no step is selected (all disabled?).")
        return 0
    cwd = Path(cfg.get("project_root") or PROJECT_ROOT)

    print_plan(runs)
    if args.list:
        return 0

    started = datetime.now()
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.dry_run:
        run_dir = Path(tempfile.mkdtemp(prefix=f"eval_{args.config.stem}_"))
    else:
        from experiments.layout import ExperimentLayout

        root = _mapping(cfg.get("shared"), "shared").get("root")
        run_dir = ExperimentLayout.from_root(root).eval_runs_dir / f"{args.config.stem}_{started:%Y%m%d-%H%M%S}"
    steps_dir, logs_dir = run_dir / "steps", run_dir / "logs"
    steps_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    for r in runs:
        write_step_config(r, steps_dir)
    print(f"Run directory: {run_dir}\n")

    if args.dry_run:
        for r in runs:
            r.command = envs[r.environment].argv(r.module, script_args(r))
            r.status = "not_run"
            print(f"[{r.index}/{len(runs)}] {r.title} | env: {r.environment}\n    {shlex.join(r.command)}")
        write_summary(run_dir / "summary.json", args.config, runs, started, dry_run=True)
        return 0

    if args.preflight_only or not args.no_preflight:
        passed = preflight(runs, envs, cwd)
        if args.preflight_only:
            print("Preflight passed." if passed else "Preflight failed.")
            return 0 if passed else 1
    else:
        passed = True
    if not passed:
        print("Preflight failed: nothing was run. Fix the config (or --no-preflight).", file=sys.stderr)
        for r in runs:
            r.status = "not_run"
        write_summary(run_dir / "summary.json", args.config, runs, started, dry_run=False)
        return 1

    code = 0
    try:
        run_all(runs, envs, cwd, logs_dir, args.continue_on_error)
    except KeyboardInterrupt:
        code = 130
    print_summary(runs)
    write_summary(run_dir / "summary.json", args.config, runs, started, dry_run=False)
    print(f"Run directory: {run_dir}")
    if code:
        return code
    return 1 if any(r.status != "ok" for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
