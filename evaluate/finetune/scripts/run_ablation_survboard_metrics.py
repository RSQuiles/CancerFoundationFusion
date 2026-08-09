"""
Compute SurvBoard metrics for every model of every ablation directory in a config.

``evaluate/finetune/tasks/evaluate_survboard_metrics.py`` evaluates ONE ablation
directory, taken from the ``ablation_dir`` key of a survival task config. This
script drives it over a list of ablation directories instead, so a whole sweep can
be evaluated with one command:

    python evaluate/finetune/scripts/run_ablation_survboard_metrics.py \\
        --config evaluate/finetune/scripts/ablation_survboard_metrics.yaml

Nothing is re-trained. The survival functions must already exist — they are written
by the ``survival`` downstream task (``run_ablation_downstream.py``) to

    {survboard_results_dir}/{COHORT}/{CANCER}/{model_name}/split_{fold}.csv

and this script only reads them and writes, per model,

    {ablation_dir}/{model_name}/metrics/results_survival.json          (aggregate)
    {ablation_dir}/{model_name}/metrics/results_survival_detailed.csv  (per fold)

The aggregate keeps any ``c_index`` the task itself wrote, so the JSON ends up with
all four metrics that ``evaluate/plot/plot_ablation_benchmark.py`` plots.

Requires the SurvBoard environment (pycox, sksurv, survival_evaluation) — the same
one ``scripts/survboard_metrics.sh`` activates. ``--dry-run`` and ``--list`` do not
import any of it, so the plan can be checked anywhere.

Model-name collisions
---------------------
Survival CSVs are keyed by MODEL NAME ONLY, not by ablation directory
(``survboard_task.py:985``). Two ablation directories that both contain, say,
``unified/`` and share one ``survboard_results_dir`` therefore write to the same
CSVs, and whichever ran last is what both get scored on. A pre-flight check reports
every such collision; give the affected ablations distinct ``survboard_results_dir``
values and re-run the survival task if you hit one.

Config
------
See ``evaluate/finetune/scripts/ablation_survboard_metrics.yaml``. Briefly::

    vars:            {work: /cluster/work/boeva/rquiles}    # ${work} interpolation
    task_config:     path to a survival task YAML; its finetune.survival block
                     supplies the defaults (data/splits/results dirs, cohorts,
                     cancer types), so this cannot drift from what wrote the CSVs
    defaults:        overrides on top of task_config, applied to every ablation
    ablations:       [{name, ablation_dir, <per-ablation overrides>}, ...]

Recognised keys, in ``defaults`` or per ablation: ``survboard_data_dir``,
``splits_dir``, ``survboard_results_dir``, ``cohorts``, ``cancer_types``,
``ibs_grid_len``, ``models``, ``skip_existing``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.analysis_config import deep_merge, interpolate, read_config_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("survboard_sweep")

# Keys read from a job, with their defaults. Anything else in the config is ignored,
# which is what lets `task_config` point at a full survival task YAML.
_DEFAULTS: dict[str, Any] = {
    "survboard_data_dir":    None,
    "splits_dir":            None,
    "survboard_results_dir": None,
    "cohorts":               [],
    "cancer_types":          [],
    "ibs_grid_len":          100,
    "models":                None,   # None -> discover under ablation_dir
    "skip_existing":         False,
}

_REQUIRED = ("survboard_data_dir", "splits_dir", "survboard_results_dir")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AblationJob:
    """One ablation directory plus everything needed to score it."""

    name:          str
    ablation_dir:  Path
    data_dir:      Path
    splits_dir:    Path
    results_dir:   Path
    cohorts:       list[str]
    cancer_types:  list[str]
    ibs_grid_len:  int
    models:        list[str] | None
    skip_existing: bool


def _resolve_vars(raw: dict[str, Any]) -> dict[str, str]:
    """Expand ``vars:`` against itself, so later entries may use earlier ones."""
    variables: dict[str, str] = {}
    for key, value in (raw.get("vars") or {}).items():
        variables[str(key)] = str(interpolate(str(value), variables))
    return variables


def _dedup(values: list[Any], what: str, where: str) -> list[str]:
    """Drop repeats, preserving order, and say so.

    ``evaluate_all`` loops cohorts x cancer_types, so a repeated entry scores the
    same (cohort, cancer) block twice: duplicate rows in the detailed CSV and double
    weight in the aggregate mean. ``survival_pred_config.yaml`` ships one such
    repeat — BRCA, listed again for METABRIC — so this is not hypothetical.
    """
    seen: set[str] = set()
    out: list[str] = []
    repeats: list[str] = []
    for value in values:
        text = str(value)
        (out if text not in seen else repeats).append(text)
        seen.add(text)
    if repeats:
        log.warning(
            "[%s] dropped duplicate %s: %s — each (cohort, cancer) pair is scored "
            "once, so a repeat would double its weight in the mean.",
            where, what, ", ".join(sorted(set(repeats))),
        )
    return out


def _task_config_defaults(path: Path) -> dict[str, Any]:
    """Return the ``finetune.survival`` block of a survival task config."""
    raw = read_config_file(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a mapping at the top level.")
    block = (raw.get("finetune") or {}).get("survival")
    if not isinstance(block, dict):
        raise KeyError(f"{path} has no 'finetune.survival' block.")
    return {k: v for k, v in block.items() if k in _DEFAULTS}


def load_jobs(config_path: Path) -> list[AblationJob]:
    """Load, interpolate, merge and validate a sweep config."""
    raw = read_config_file(config_path)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} must contain a mapping at the top level.")

    variables = _resolve_vars(raw)
    raw = interpolate({k: v for k, v in raw.items() if k != "vars"}, variables)

    base = dict(_DEFAULTS)
    task_config = raw.get("task_config")
    if task_config:
        task_path = Path(str(task_config))
        if not task_path.is_absolute():
            task_path = (PROJECT_ROOT / task_path).resolve()
        base = deep_merge(base, _task_config_defaults(task_path))
        log.info("Defaults inherited from %s", task_path)

    defaults = deep_merge(base, raw.get("defaults") or {})

    items = raw.get("ablations") or raw.get("experiments")
    if not items:
        raise KeyError(f"{config_path} must define a non-empty 'ablations' list.")

    jobs: list[AblationJob] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Each ablation must be a mapping, got {type(item)}")
        if "ablation_dir" not in item:
            raise KeyError(
                f"Ablation {item.get('name', '<unnamed>')} needs 'ablation_dir'."
            )

        ablation_dir = Path(str(item["ablation_dir"]))
        name = str(item.get("name") or ablation_dir.name)
        if name in seen:
            raise ValueError(f"Duplicate ablation name: {name!r}")
        seen.add(name)

        merged = deep_merge(
            defaults,
            {k: v for k, v in item.items() if k not in ("name", "ablation_dir")},
        )
        missing = [k for k in _REQUIRED if not merged.get(k)]
        if missing:
            raise KeyError(f"Ablation {name!r} is missing required key(s): {missing}")
        if not merged["cancer_types"]:
            raise KeyError(f"Ablation {name!r} has an empty 'cancer_types' list.")
        if not merged["cohorts"]:
            raise KeyError(f"Ablation {name!r} has an empty 'cohorts' list.")

        models = merged["models"]
        jobs.append(AblationJob(
            name          = name,
            ablation_dir  = ablation_dir,
            data_dir      = Path(str(merged["survboard_data_dir"])),
            splits_dir    = Path(str(merged["splits_dir"])),
            results_dir   = Path(str(merged["survboard_results_dir"])),
            cohorts       = _dedup(merged["cohorts"], "cohort", name),
            cancer_types  = _dedup(merged["cancer_types"], "cancer type", name),
            ibs_grid_len  = int(merged["ibs_grid_len"]),
            models        = [str(m) for m in models] if models else None,
            skip_existing = bool(merged["skip_existing"]),
        ))

    return jobs


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_models(ablation_dir: Path) -> list[str]:
    """Names of sub-directories holding a checkpoint or a metrics/ directory.

    Mirrors ``evaluate_survboard_metrics._discover_model_dirs``. Kept local so that
    ``--dry-run`` and ``--list`` work without pycox/sksurv installed — importing that
    module pulls in the whole SurvBoard stack. The metrics/ clause is what keeps
    ``pca_baseline/`` (no checkpoint) in the sweep.
    """
    if not ablation_dir.is_dir():
        return []
    names: list[str] = []
    for d in sorted(ablation_dir.iterdir()):
        if not d.is_dir():
            continue
        has_ckpt = any(d.glob("*.ckpt")) or any((d / "checkpoints").glob("*.ckpt"))
        if has_ckpt or (d / "metrics").is_dir():
            names.append(d.name)
    return names


def _has_metrics(ablation_dir: Path, model: str) -> bool:
    """True if SurvBoard metrics (not just the task's c_index) are already stored."""
    path = ablation_dir / model / "metrics" / "results_survival.json"
    if not path.exists():
        return False
    try:
        with open(path) as fh:
            return "antolini_concordance" in json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False


def check_collisions(plan: list[tuple[AblationJob, list[str]]]) -> int:
    """Warn where one model name is shared by ablations writing to one results dir.

    Returns the number of colliding names. See the module docstring: the survival
    CSVs carry no ablation identity, so such models are all scored on whichever run
    wrote last.
    """
    owners: dict[tuple[str, str], list[str]] = {}
    for job, models in plan:
        for model in models:
            owners.setdefault((str(job.results_dir), model), []).append(job.name)

    collisions = {k: v for k, v in owners.items() if len(v) > 1}
    for (results_dir, model), names in sorted(collisions.items()):
        log.warning(
            "COLLISION: model '%s' appears in %s, all sharing survboard_results_dir "
            "%s — they read the SAME survival CSVs, so these numbers cannot "
            "distinguish the ablations.",
            model, ", ".join(sorted(names)), results_dir,
        )
    if collisions:
        log.warning(
            "%d colliding model name(s). Give the affected ablations distinct "
            "'survboard_results_dir' values and re-run the survival task.",
            len(collisions),
        )
    return len(collisions)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def _missing_paths(job: AblationJob) -> list[str]:
    return [
        f"{label}={path}"
        for label, path in (
            ("ablation_dir", job.ablation_dir),
            ("survboard_data_dir", job.data_dir),
            ("splits_dir", job.splits_dir),
            ("survboard_results_dir", job.results_dir),
        )
        if not path.exists()
    ]


def run_job(job: AblationJob, models: list[str]) -> dict[str, str]:
    """Evaluate every model of one ablation. Returns model -> status."""
    from evaluate.finetune.tasks.evaluate_survboard_metrics import evaluate_all

    statuses: dict[str, str] = {}
    for model in models:
        if job.skip_existing and _has_metrics(job.ablation_dir, model):
            log.info("  [%s/%s] metrics already present — skipping", job.name, model)
            statuses[model] = "skip"
            continue

        log.info("  [%s/%s] evaluating ...", job.name, model)
        try:
            evaluate_all(
                data_dir     = job.data_dir.resolve(),
                splits_dir   = job.splits_dir.resolve(),
                results_dir  = job.results_dir.resolve(),
                ablation_dir = job.ablation_dir.resolve(),
                cohorts      = job.cohorts,
                cancer_types = job.cancer_types,
                model_name   = model,
                ibs_grid_len = job.ibs_grid_len,
            )
            statuses[model] = "ok"
        except SystemExit:
            # evaluate_all exits when it found no survival CSVs at all. That is a
            # missing model, not a broken sweep, so keep going.
            log.warning(
                "  [%s/%s] no survival CSVs under %s — has the survival task run "
                "for this model?",
                job.name, model, job.results_dir / "<COHORT>/<CANCER>" / model,
            )
            statuses[model] = "no-data"
        except Exception:
            log.error("  [%s/%s] FAILED:\n%s", job.name, model, traceback.format_exc())
            statuses[model] = "fail"

    return statuses


def _print_summary(summary: dict[str, dict[str, str]]) -> None:
    print(f"\n{'=' * 72}\nSurvBoard metric sweep — summary\n{'=' * 72}")
    counts: dict[str, int] = {}
    for name, statuses in summary.items():
        print(f"\n{name}")
        if not statuses:
            print("  (no models)")
        for model, status in statuses.items():
            print(f"  {model:<40} {status}")
            counts[status] = counts.get(status, 0) + 1
    tally = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing to do"
    print(f"\n{'-' * 72}\n{tally}\n{'=' * 72}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute SurvBoard metrics for every model of every ablation "
            "directory listed in a config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, type=Path, help="Sweep config (YAML/JSON).")
    p.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Evaluate only these ablations (by 'name').",
    )
    p.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Override the model list for every ablation.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a model whose results_survival.json already holds SurvBoard metrics.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the ablations and the models found in each, then exit.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report the plan (including collisions) without computing anything.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    jobs = load_jobs(args.config.expanduser().resolve())

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {j.name for j in jobs}
        if unknown:
            log.error("--only names not in the config: %s", sorted(unknown))
            sys.exit(1)
        jobs = [j for j in jobs if j.name in wanted]

    # Resolve the model list per job before doing anything, so --dry-run and the
    # collision check see exactly what a real run would.
    plan: list[tuple[AblationJob, list[str]]] = []
    for job in jobs:
        if args.skip_existing:
            job = replace(job, skip_existing=True)
        models = args.models or job.models or discover_models(job.ablation_dir)
        plan.append((job, list(models)))

    if args.list or args.dry_run:
        for job, models in plan:
            missing = _missing_paths(job)
            print(f"\n{job.name}")
            print(f"  ablation_dir  : {job.ablation_dir}")
            print(f"  results_dir   : {job.results_dir}")
            print(f"  cohorts       : {', '.join(job.cohorts)}")
            print(f"  cancer_types  : {len(job.cancer_types)} "
                  f"({', '.join(job.cancer_types[:6])}"
                  f"{', ...' if len(job.cancer_types) > 6 else ''})")
            print(f"  models        : {', '.join(models) if models else '(none found)'}")
            if missing:
                print(f"  MISSING PATHS : {'; '.join(missing)}")
        print()
        check_collisions(plan)
        sys.exit(0)

    n_collisions = check_collisions(plan)
    if n_collisions:
        log.warning("Continuing anyway — the numbers above will be ambiguous.")

    summary: dict[str, dict[str, str]] = {}
    for job, models in plan:
        log.info("=" * 72)
        log.info("[%s] %s", job.name, job.ablation_dir)

        missing = _missing_paths(job)
        if missing:
            log.error("[%s] skipped, path(s) do not exist: %s", job.name, "; ".join(missing))
            summary[job.name] = {"(all)": "missing-paths"}
            continue
        if not models:
            log.warning("[%s] no model directories found in %s", job.name, job.ablation_dir)
            summary[job.name] = {}
            continue

        log.info("[%s] %d model(s): %s", job.name, len(models), ", ".join(models))
        summary[job.name] = run_job(job, models)

    _print_summary(summary)

    failed = sum(
        1 for statuses in summary.values() for s in statuses.values()
        if s in ("fail", "missing-paths")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
