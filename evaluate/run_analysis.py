"""Run the post-training analyses over several ablation experiments from one config.

Replaces hand-editing one shell script per analysis per experiment
(``plot/plot_umap.sh``, ``check/build_and_check.sh``, ``check/check_models.sh``,
``plot/plot_benchmark.sh``). Steps are the existing CLIs — nothing is reimplemented
here; this module only builds argv, picks the right environment, and submits jobs.

The steps do not all run in the same environment, which is the main thing this
orchestrator exists to get right:

    build        check/build_eval_adata.py        container   GPU
    metrics      check/unified_metrics.py         container   GPU
    scib         check/unified_metrics.py --scib  conda     (container lacks scib)
    diagnose     check/diagnose_scib.py           plain
    umap         plot/umaps.py                    container   GPU
    paired_umap  plot/plot_paired_umap.py         container   GPU   (opt-in)
    benchmark    plot/plot_ablation_benchmark.py  plain

``paired_umap`` is opt-in: it needs a separate paired h5ad (cell-line matched SC and
bulk) rather than eval.h5ad or adata_dir, so it is not in the default step list.

Usage
-----
    # inspect the exact commands without running anything
    python evaluate/run_analysis.py --config evaluate/example_analysis_config.yaml --dry-run

    # run here and now, everything in this environment
    python evaluate/run_analysis.py --config <cfg> --local

    # submit to SLURM (mode comes from the config; override with --mode)
    python evaluate/run_analysis.py --config <cfg>

    # subset
    python evaluate/run_analysis.py --config <cfg> --only monitor_align --step umap benchmark

``--payload`` is the in-job entry point: each submitted job re-invokes this module
with a JSON payload, and the job then enters the container / conda env per step.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_analysis")

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from evaluate.analysis_config import (  # noqa: E402
    STEP_ORDER,
    AnalysisConfig,
    EnvConfig,
    ExperimentConfig,
    load_analysis_config,
)

# Environment each step needs. "container" = bionemo image, "conda" = the env with
# scib_metrics, "plain" = anything with scanpy/pandas.
STEP_ENV: dict[str, str] = {
    "build": "container",
    "metrics": "container",
    "scib": "conda",
    "diagnose": "plain",
    "umap": "container",
    "paired_umap": "container",
    "benchmark": "plain",
}

# Steps that read eval.h5ad and therefore depend on `build` having produced it.
# `umap` is conditional: only when umap.source == "eval". `paired_umap` never reads
# eval.h5ad — it builds its own SC/pseudobulk/bulk AnnData from its paired h5ad.
STEP_NEEDS_EVAL: dict[str, bool] = {
    "build": False,
    "metrics": True,
    "scib": True,
    "diagnose": True,
    "umap": False,   # decided per experiment
    "paired_umap": False,
    "benchmark": False,
}

SCRIPTS: dict[str, Path] = {
    "build": Path("evaluate/check/build_eval_adata.py"),
    "metrics": Path("evaluate/check/unified_metrics.py"),
    "scib": Path("evaluate/check/unified_metrics.py"),
    "diagnose": Path("evaluate/check/diagnose_scib.py"),
    "umap": Path("evaluate/plot/umaps.py"),
    "paired_umap": Path("evaluate/plot/plot_paired_umap.py"),
    "benchmark": Path("evaluate/plot/plot_ablation_benchmark.py"),
}


@dataclass
class Step:
    """One resolved command: what to run, where, and why it might be skipped."""

    name: str
    argv: list[str]          # python-level argv (script + flags), no wrapper
    env_kind: str            # container | conda | plain
    cwd: Path
    needs_eval: bool
    note: str = ""           # human-readable resolution note, printed on dry-run


# ---------------------------------------------------------------------------
# Sample-size reconciliation
# ---------------------------------------------------------------------------

def resolve_sample_sizes(exp: ExperimentConfig, will_build: bool) -> tuple[int, str]:
    """Return (build_sample_size, note).

    The two knobs are NOT the same unit:
      * build --sample-size = max cells per *filename prefix*
      * umap  --sample-size = total cells drawn from the raw h5ads, and it is only
        honoured on the live path; with --eval-adata, umaps.py plots whatever rows
        eval.h5ad holds and ignores the flag.

    So when an experiment both builds and plots from eval.h5ad, eval.h5ad has to be
    built large enough for the plot: take the max, once, and reuse it for both.
    """
    build_n = int(exp.build["sample_size"])
    umap_n = int(exp.umap["sample_size"])

    if exp.umap["source"] != "eval" or "umap" not in exp.steps:
        return build_n, ""
    if not will_build:
        if umap_n > build_n:
            return build_n, (
                f"umap wants {umap_n} cells but build is not running; the plot is "
                f"limited to whatever eval.h5ad already contains"
            )
        return build_n, ""
    if umap_n > build_n:
        return umap_n, (
            f"build sample_size raised {build_n} -> {umap_n} so the single eval.h5ad "
            f"also serves the UMAP (umap.sample_size won)"
        )
    return build_n, (
        f"build sample_size {build_n} already >= umap.sample_size {umap_n}; "
        f"eval.h5ad serves both"
    )


# ---------------------------------------------------------------------------
# argv builders — one per step, all pointing at existing CLIs
# ---------------------------------------------------------------------------

def _flag(argv: list[str], name: str, value: Any) -> None:
    """Append ``--name value`` unless value is None."""
    if value is not None:
        argv += [name, str(value)]


def _build_argv(exp: ExperimentConfig, root: Path, build_n: int) -> list[str]:
    b = exp.build
    argv = [str(root / SCRIPTS["build"]),
            "--adata-dir", str(exp.adata_dir),
            "--out", str(exp.eval_adata),
            "--ablation-dir", str(exp.ablation_dir),
            "--sample-size", str(build_n)]
    _flag(argv, "--batch-size", b["batch_size"])
    _flag(argv, "--seed", exp.seed)
    _flag(argv, "--group-column", exp.metrics["group_column"])
    _flag(argv, "--n-pca-components", b["n_pca_components"])
    _flag(argv, "--n-hvg-baseline", b["n_hvg_baseline"])
    _flag(argv, "--panel-strategy", b["panel_strategy"])
    if b["precomputed_pb"]:
        argv.append("--precomputed-pb")
    if b["per_modality_panel"]:
        argv.append("--per-modality-panel")
    if not exp.normalized:
        argv.append("--not-normalized")
    return argv


def _metrics_argv(exp: ExperimentConfig, root: Path, scib: bool) -> list[str]:
    m = exp.metrics
    argv = [str(root / SCRIPTS["metrics"]),
            "--eval-adata", str(exp.eval_adata),
            "--ablation-dir", str(exp.ablation_dir)]
    _flag(argv, "--batch-size", m["batch_size"])
    _flag(argv, "--mask-ratio", m["mask_ratio"])
    _flag(argv, "--n-sc-per-pb", m["n_sc_per_pb"])
    _flag(argv, "--n-synth-pb", m["n_synth_pb"])
    _flag(argv, "--group-column", m["group_column"])
    _flag(argv, "--seed", exp.seed)
    if not exp.normalized:
        argv.append("--not-normalized")
    if scib:
        # --skip-existing is panel-aware: a cache whose panel_hash differs from this
        # eval.h5ad is recomputed, so this cannot serve stale numbers.
        argv += ["--scib", "--skip-existing"]
        _flag(argv, "--scib-n-max", exp.scib["n_max"])
    elif m["skip_existing"]:
        argv.append("--skip-existing")
    return argv


def _diagnose_argv(exp: ExperimentConfig, root: Path) -> list[str]:
    d = exp.diagnose
    out_csv = d.get("out_csv") or exp.ablation_dir / "_scib_metrics" / "diagnosis.csv"
    argv = [str(root / SCRIPTS["diagnose"]),
            "--eval-adata", str(exp.eval_adata)]
    _flag(argv, "--group-column", exp.metrics["group_column"])
    _flag(argv, "--n-max", d["n_max"])
    _flag(argv, "--seed", exp.seed)
    _flag(argv, "--out-csv", out_csv)
    return argv


def _umap_argv(exp: ExperimentConfig, root: Path) -> list[str]:
    u = exp.umap
    argv = [str(root / SCRIPTS["umap"]), "--ablation-dir", str(exp.ablation_dir)]
    if u["source"] == "eval":
        # No --sample-size: umaps.py ignores it when reading eval.h5ad, and passing
        # it would suggest a subsampling that does not happen.
        argv += ["--eval-adata", str(exp.eval_adata)]
    else:
        argv += ["--adata-dir", str(exp.adata_dir),
                 "--sample-size", str(u["sample_size"])]
        _flag(argv, "--adata-prefix", u.get("adata_prefix"))
    if u["color"]:
        argv += ["--color", *[str(c) for c in u["color"]]]
    _flag(argv, "--neighbors", u["neighbors"])
    _flag(argv, "--min-dist", u["min_dist"])
    _flag(argv, "--seed", exp.seed)
    _flag(argv, "--embed-batch-size", u["embed_batch_size"])
    _flag(argv, "--n-sc-per-pb", exp.metrics["n_sc_per_pb"])
    _flag(argv, "--pb-group-column", exp.metrics["group_column"])
    if u["skip_unknown"]:
        argv.append("--skip-unknown")
    if u["plot_pb_only"]:
        argv.append("--plot-pb-only")
    if u["no_modality_split"]:
        argv.append("--no-modality-split")
    return argv


def _paired_umap_argv(exp: ExperimentConfig, root: Path) -> list[str]:
    """plot_paired_umap.py in --ablation-dir mode.

    Only the flags ``run_ablation()`` actually receives are passed. --out-prefix,
    --domain-col, --sc-label, --bulk-label, --is-log1p and --max-sc-per-line exist on
    the CLI but are dropped by main() in this mode, and `normalized` is hardcoded True
    there, so forwarding them would imply an effect they do not have.
    """
    p = exp.paired_umap
    out_dir = p.get("out_dir") or exp.ablation_dir / "paired_umaps"
    argv = [str(root / SCRIPTS["paired_umap"]),
            "--ablation-dir", str(exp.ablation_dir),
            "--input-h5ad", str(p["input_h5ad"]),
            "--out-dir", str(out_dir)]
    _flag(argv, "--cell-line-col", p["cell_line_col"])
    _flag(argv, "--n-cell-lines", p["n_cell_lines"])
    _flag(argv, "--batch-size", p["batch_size"])
    _flag(argv, "--neighbors", p["neighbors"])
    _flag(argv, "--min-dist", p["min_dist"])
    _flag(argv, "--seed", exp.seed)
    _flag(argv, "--dpi", p["dpi"])
    _flag(argv, "--device", p.get("device"))
    if p["no_sc"]:
        argv.append("--no-sc")
    return argv


def _benchmark_argv(exp: ExperimentConfig, root: Path) -> list[str]:
    b = exp.benchmark
    out = b.get("output") or exp.ablation_dir / "benchmark.png"
    argv = [str(root / SCRIPTS["benchmark"]),
            "--ablation-dir", str(exp.ablation_dir),
            "--output", str(out)]
    if b.get("no_show", True):
        argv.append("--no-show")
    if b.get("figsize"):
        argv += ["--figsize", *[str(x) for x in b["figsize"]]]
    primary = b.get("primary") or {}
    if primary:
        argv += ["--primary", *[f"{k}={v}" for k, v in primary.items()]]
    return argv


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_experiment(
    exp: ExperimentConfig, root: Path, only_steps: list[str] | None = None
) -> tuple[list[Step], dict[str, str]]:
    """Resolve one experiment into an ordered list of Steps.

    Returns (steps, skipped) where ``skipped`` maps step name -> reason, so a step
    dropped by ``rebuild: never`` or a missing eval.h5ad is reported rather than
    silently absent.
    """
    steps = [s for s in exp.steps if only_steps is None or s in only_steps]
    skipped: dict[str, str] = {}

    # ── does `build` run? ────────────────────────────────────────────────────
    eval_exists = exp.eval_adata.exists()
    will_build = False
    if "build" in steps:
        mode = exp.build["rebuild"]
        if mode == "always":
            will_build = True
        elif mode == "never":
            skipped["build"] = "build.rebuild = never"
        elif eval_exists:
            skipped["build"] = (
                f"build.rebuild = auto and {exp.eval_adata.name} already exists"
            )
        else:
            will_build = True

    build_n, size_note = resolve_sample_sizes(exp, will_build)

    umap_needs_eval = exp.umap["source"] == "eval"
    eval_available = will_build or eval_exists

    out: list[Step] = []
    for name in STEP_ORDER:
        if name not in steps:
            continue
        if name == "build" and not will_build:
            continue

        needs_eval = (
            umap_needs_eval if name == "umap" else STEP_NEEDS_EVAL[name]
        )
        if needs_eval and not eval_available:
            skipped[name] = (
                f"needs {exp.eval_adata} which does not exist and is not being built "
                f"(build.rebuild = {exp.build['rebuild']})"
            )
            continue

        if name == "build":
            argv, note = _build_argv(exp, root, build_n), size_note
        elif name == "metrics":
            argv, note = _metrics_argv(exp, root, scib=False), ""
        elif name == "scib":
            argv, note = _metrics_argv(exp, root, scib=True), ""
        elif name == "diagnose":
            argv, note = _diagnose_argv(exp, root), ""
        elif name == "umap":
            argv = _umap_argv(exp, root)
            note = (
                "reads eval.h5ad, so the figure and the metric tables come from "
                "identical vectors"
                if umap_needs_eval else
                "live embedding: the figure may describe different vectors than the "
                "metric tables"
            )
        elif name == "paired_umap":
            argv = _paired_umap_argv(exp, root)
            note = f"cell-line paired plot from {exp.paired_umap['input_h5ad']}"
        else:
            argv, note = _benchmark_argv(exp, root), ""

        out.append(
            Step(
                name=name,
                argv=argv,
                env_kind=STEP_ENV[name],
                cwd=(root / SCRIPTS[name]).parent,
                needs_eval=needs_eval,
                note=note,
            )
        )

    return out, skipped


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------

def wrap_command(step: Step, env: EnvConfig, local: bool) -> list[str]:
    """Turn a python-level argv into the command actually executed."""
    py = [env.python, "-u", *step.argv]

    if local or step.env_kind == "plain":
        return py

    if step.env_kind == "container":
        if not env.container_image:
            raise ValueError(
                f"step '{step.name}' needs the container but env.container_image is "
                "unset. Set it, or run with --local."
            )
        cmd = ["singularity", "run", "--pwd", str(step.cwd)]
        for b in env.binds:
            cmd += ["--bind", f"{b}:{b}"]
        if env.nv:
            cmd.append("--nv")
        cmd.append(env.container_image)
        return cmd + py

    if step.env_kind == "conda":
        if not env.conda_env:
            raise ValueError(
                f"step '{step.name}' needs the scib conda env but env.conda_env is "
                "unset. Set it, or run with --local."
            )
        # -l so ~/.bashrc is sourced and `conda activate` is available, matching what
        # the shell scripts this replaces did by hand.
        inner = f"conda activate {shlex.quote(env.conda_env)} && {shlex.join(py)}"
        return ["bash", "-lc", inner]

    raise ValueError(f"Unknown env_kind {step.env_kind!r}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_experiment(
    exp: ExperimentConfig,
    cfg: AnalysisConfig,
    only_steps: list[str] | None,
    dry_run: bool,
    local: bool,
) -> dict[str, Any]:
    """Run one experiment's steps in order. Never raises on step failure."""
    steps, skipped = plan_experiment(exp, cfg.repo_root, only_steps)

    log.info("=" * 78)
    log.info("[%s] %s", exp.name, exp.ablation_dir)
    log.info("  steps: %s", " -> ".join(s.name for s in steps) or "<none>")
    for name, why in skipped.items():
        log.info("  skipping %-10s %s", name, why)

    results: list[dict[str, Any]] = []
    failed: set[str] = set()

    for step in steps:
        # A step that reads eval.h5ad is pointless once build has failed.
        if step.needs_eval and "build" in failed:
            log.warning("  [%s] skipped: 'build' failed earlier", step.name)
            results.append({"step": step.name, "status": "skipped",
                            "reason": "build failed"})
            continue

        cmd = wrap_command(step, cfg.env, local)
        # ASCII only: these lines land on consoles that are not always UTF-8.
        log.info("  [%s] (%s)%s", step.name,
                 "local" if local else step.env_kind,
                 f"  -- {step.note}" if step.note else "")
        log.info("    %s", shlex.join(cmd))

        if dry_run:
            results.append({"step": step.name, "status": "dry-run",
                            "command": shlex.join(cmd)})
            continue

        # cwd is the script's own directory inside the repo; if it is missing that is
        # a real error, not something to paper over by creating it.
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(step.cwd))
        elapsed = round(time.time() - started, 1)

        if proc.returncode == 0:
            log.info("    [%s] ok (%.1fs)", step.name, elapsed)
            results.append({"step": step.name, "status": "ok",
                            "seconds": elapsed, "command": shlex.join(cmd)})
        else:
            log.error("    [%s] FAILED rc=%d (%.1fs)", step.name, proc.returncode, elapsed)
            failed.add(step.name)
            results.append({"step": step.name, "status": "failed",
                            "returncode": proc.returncode, "seconds": elapsed,
                            "command": shlex.join(cmd)})

    record = {
        "experiment": exp.name,
        "ablation_dir": str(exp.ablation_dir),
        "planned": [s.name for s in steps],
        "skipped": skipped,
        "steps": results,
        "status": "failed" if failed else "completed",
    }

    if not dry_run:
        exp.analysis_dir.mkdir(parents=True, exist_ok=True)
        out = exp.analysis_dir / "analysis_result.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.info("  result -> %s", out)

    return record


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------

def _write_payload(
    path: Path,
    cfg: AnalysisConfig,
    exp_names: list[str],
    only_steps: list[str] | None,
    dry_run: bool,
) -> Path:
    """Payload is JSON so the job needs no PyYAML to re-read it.

    A dry run writes nothing and creates no directories — otherwise inspecting a
    config would litter the filesystem (and, with paths for a machine you are not on,
    create the whole tree locally).
    """
    payload = {
        "config_path": str(cfg.config_path),
        "experiments": exp_names,
        "only_steps": only_steps,
    }
    if dry_run:
        log.info("  payload (not written): %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sbatch(
    cfg: AnalysisConfig, payload: Path, job_name: str, log_path: Path, dry_run: bool
) -> str | None:
    """Submit one job that re-invokes this module with the payload."""
    inner = shlex.join([cfg.env.python, "-u",
                        str(cfg.repo_root / "evaluate" / "run_analysis.py"),
                        "--payload", str(payload)])
    cmd = ["sbatch", "--parsable", *cfg.slurm.sbatch_args,
           "--job-name", job_name[:128],
           "--output", str(log_path),
           "--wrap", inner]

    log.info("  sbatch: %s", shlex.join(cmd))
    if dry_run:
        return None

    env = os.environ.copy()
    env.update(cfg.slurm.environment)
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    except FileNotFoundError:
        raise RuntimeError(
            "sbatch not found. Use --local to run here instead."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"sbatch failed for {job_name}: {exc.stderr.strip() or '<empty>'}"
        ) from exc
    return done.stdout.strip().split(";", maxsplit=1)[0].strip()


def submit(
    cfg: AnalysisConfig,
    experiments: list[ExperimentConfig],
    only_steps: list[str] | None,
    mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []

    if mode == "single-job":
        # One job for everything; payload carries every experiment.
        anchor = experiments[0].analysis_dir
        payload = _write_payload(
            anchor / "payload_all.json", cfg, [e.name for e in experiments],
            only_steps, dry_run,
        )
        job_name = f"{cfg.slurm.job_name_prefix}_all"
        job_id = _sbatch(cfg, payload, job_name,
                         anchor / "slurm-analysis-%j.out", dry_run)
        jobs.append({"job_name": job_name, "job_id": job_id,
                     "experiments": [e.name for e in experiments],
                     "payload": str(payload)})
    else:
        for exp in experiments:
            payload = _write_payload(
                exp.analysis_dir / "payload.json", cfg, [exp.name], only_steps, dry_run
            )
            job_name = f"{cfg.slurm.job_name_prefix}_{exp.name}"
            job_id = _sbatch(cfg, payload, job_name,
                             exp.analysis_dir / "slurm-%j.out", dry_run)
            jobs.append({"job_name": job_name, "job_id": job_id,
                         "experiments": [exp.name], "payload": str(payload)})

    manifest = {"config": str(cfg.config_path), "mode": mode,
                "only_steps": only_steps, "jobs": jobs}
    if not dry_run and cfg.config_path is not None:
        path = cfg.config_path.parent / f"{cfg.config_path.stem}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("Manifest -> %s", path)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Analysis config (.yaml/.yml/.json).")
    p.add_argument("--payload", type=Path, default=None,
                   help="Internal: JSON payload written by the submitter; runs the "
                        "listed experiments in this process.")
    p.add_argument("--only", nargs="*", default=None, metavar="NAME",
                   help="Restrict to these experiment names.")
    p.add_argument("--step", nargs="*", default=None, metavar="STEP",
                   choices=list(STEP_ORDER),
                   help=f"Restrict to these steps (of {list(STEP_ORDER)}).")
    p.add_argument("--mode", choices=["per-experiment", "single-job", "local"],
                   default=None, help="Override slurm.mode; 'local' implies --local.")
    p.add_argument("--local", action="store_true",
                   help="Run now in this environment, no SLURM and no container.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every resolved command and submit nothing.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    only_steps = list(args.step) if args.step else None
    only_names = list(args.only) if args.only else None

    if args.payload is not None:
        # In-job path: read the payload and execute locally=False so each step still
        # enters its own container / conda env.
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        cfg = load_analysis_config(payload["config_path"])
        only_steps = payload.get("only_steps") or only_steps
        wanted = set(payload.get("experiments") or [])
        experiments = [e for e in cfg.experiments if e.name in wanted]
        if not experiments:
            log.error("Payload lists %s but the config has none of them.", wanted)
            return 1
        records = [
            execute_experiment(e, cfg, only_steps, dry_run=False, local=False)
            for e in experiments
        ]
        return 0 if all(r["status"] == "completed" for r in records) else 1

    if args.config is None:
        log.error("--config is required (or --payload for the in-job path).")
        return 1

    cfg = load_analysis_config(args.config)
    experiments = cfg.experiments
    if only_names:
        unknown = sorted(set(only_names) - {e.name for e in experiments})
        if unknown:
            log.error("Unknown experiment name(s): %s. Available: %s",
                      unknown, [e.name for e in experiments])
            return 1
        experiments = [e for e in experiments if e.name in set(only_names)]

    mode = args.mode or ("local" if args.local else
                         (cfg.slurm.mode if cfg.slurm.enabled else "local"))
    local = args.local or mode == "local"

    log.info("Config: %s", cfg.config_path)
    log.info("Experiments: %s", [e.name for e in experiments])
    log.info("Mode: %s%s", mode, "  (dry run)" if args.dry_run else "")

    if local:
        records = [
            execute_experiment(e, cfg, only_steps, args.dry_run, local=True)
            for e in experiments
        ]
        ok = all(r["status"] == "completed" for r in records)
        log.info("=" * 78)
        for r in records:
            log.info("%-24s %s", r["experiment"], r["status"])
        return 0 if ok or args.dry_run else 1

    # SLURM. Show the resolved plan first so a dry run is informative.
    if args.dry_run:
        for e in experiments:
            execute_experiment(e, cfg, only_steps, dry_run=True, local=False)
    submit(cfg, experiments, only_steps, mode, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
