"""
Run downstream evaluation tasks for every model in an ablation directory.

Given an ablation directory structured as:

    {ablation_dir}/
        {model_name}/          ← contains a checkpoint (*.ckpt)
        {model_name}/
        ...

this script iterates over all model sub-directories, locates the best
checkpoint in each one, and runs the requested downstream tasks using
the configs in evaluate/finetune/configs/ (or a custom --config-dir).

Results are written to:
    {ablation_dir}/{model_name}/metrics/results_{task}.json

After all runs, an optional benchmark plot can be generated via
evaluate/plot/ablation_benchmark.py.

Usage
-----
    # Run two tasks for all models, then plot:
    python evaluate/finetune/run_ablation_downstream.py \\
        --ablation-dir path/to/ablation \\
        --tasks canc_type_class deconv \\
        --plot

    # Skip models that already have results:
    python evaluate/finetune/run_ablation_downstream.py \\
        --ablation-dir path/to/ablation \\
        --tasks survival \\
        --skip-existing

    # Use custom config directory:
    python evaluate/finetune/run_ablation_downstream.py \\
        --ablation-dir path/to/ablation \\
        --tasks canc_type_class \\
        --config-dir my_configs/
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.finetune.run_downstream_task import main as run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"

# Maps task name → config filename inside the config directory.
TASK_CONFIG_FILES: dict[str, str] = {
    "canc_type_class": "cancer_annot_config.yaml",
    "deconv":          "deconv_config.yaml",
    "survival":        "survival_pred_config.yaml",
    "proteome_pred":   "proteome_pred_config.yaml",
    "drug_sensitivity": "drug_sensitivity_config.yaml",
}

_EPOCH_RE = re.compile(r"epoch[=_]?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _epoch_num(p: Path) -> int:
    """Extract epoch number from checkpoint filename, or -1 if not found."""
    m = _EPOCH_RE.search(p.stem)
    return int(m.group(1)) if m else -1


def _find_best_ckpt(model_dir: Path) -> Path | None:
    """
    Find the best (highest-epoch) checkpoint in a model directory.

    Searches ``{model_dir}/*.ckpt`` and ``{model_dir}/checkpoints/*.ckpt``.
    Returns None if no checkpoint is found.
    """
    candidates: list[Path] = []
    candidates.extend(model_dir.glob("*.ckpt"))
    candidates.extend((model_dir / "checkpoints").glob("*.ckpt"))

    if not candidates:
        return None

    return max(candidates, key=lambda p: (_epoch_num(p), p.stat().st_mtime))


def _results_exist(model_dir: Path, task: str) -> bool:
    """Return True if results for *task* already exist under *model_dir*."""
    return (model_dir / "metrics" / f"results_{task}.json").exists()


def _discover_model_dirs(ablation_dir: Path) -> list[Path]:
    """Return sorted list of sub-directories that contain at least one checkpoint."""
    dirs = []
    for d in sorted(ablation_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpt = _find_best_ckpt(d)
        if ckpt is not None:
            dirs.append(d)
    return dirs


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def run_ablation(
    ablation_dir: Path,
    tasks: list[str],
    config_dir: Path,
    skip_existing: bool,
    plot: bool,
    plot_show: bool,
    plot_output: Path | None,
) -> None:
    model_dirs = _discover_model_dirs(ablation_dir)

    if not model_dirs:
        log.error("No model directories with checkpoints found in %s", ablation_dir)
        sys.exit(1)

    log.info("Found %d model(s): %s", len(model_dirs), [d.name for d in model_dirs])

    # Validate that all requested tasks have a config file.
    missing_configs: list[str] = []
    for task in tasks:
        cfg_file = config_dir / TASK_CONFIG_FILES.get(task, f"{task}_config.yaml")
        if not cfg_file.exists():
            missing_configs.append(f"{task} (expected {cfg_file})")
    if missing_configs:
        log.error("Config files not found for tasks: %s", missing_configs)
        sys.exit(1)

    results_summary: dict[str, dict[str, str]] = {}  # model → task → "ok" | "skip" | "fail"

    for model_dir in model_dirs:
        model_name = model_dir.name
        results_summary[model_name] = {}

        ckpt = _find_best_ckpt(model_dir)
        if ckpt is None:
            log.warning("No checkpoint found in %s — skipping", model_dir)
            continue

        log.info("Model '%s' — checkpoint: %s", model_name, ckpt.name)

        for task in tasks:
            output_dir = model_dir / "metrics"

            if skip_existing and _results_exist(model_dir, task):
                log.info("  [%s] results already exist — skipping", task)
                results_summary[model_name][task] = "skip"
                continue

            cfg_file = config_dir / TASK_CONFIG_FILES.get(task, f"{task}_config.yaml")

            log.info("  [%s] running with config %s ...", task, cfg_file.name)
            try:
                run_task(
                    config_path=str(cfg_file),
                    checkpoint_path=ckpt,
                    task_name=task,
                    output_dir=output_dir,
                )
                results_summary[model_name][task] = "ok"
                log.info("  [%s] done.", task)
            except Exception:
                log.error("  [%s] FAILED:\n%s", task, traceback.format_exc())
                results_summary[model_name][task] = "fail"

    # Print summary table.
    log.info("\n%s\nSummary\n%s", "=" * 60, "=" * 60)
    header = f"{'Model':<30} " + "  ".join(f"{t:<20}" for t in tasks)
    log.info(header)
    for model_name, task_statuses in results_summary.items():
        row = f"{model_name:<30} " + "  ".join(
            f"{task_statuses.get(t, 'n/a'):<20}" for t in tasks
        )
        log.info(row)
    log.info("=" * 60)

    if plot:
        _run_benchmark_plot(ablation_dir, plot_show, plot_output)


def _run_benchmark_plot(
    ablation_dir: Path,
    show: bool,
    output: Path | None,
) -> None:
    try:
        from evaluate.plot.ablation_benchmark import collect_metrics, plot_benchmark
    except ImportError as e:
        log.error("Could not import ablation_benchmark: %s", e)
        return

    log.info("Generating benchmark plot ...")
    results = collect_metrics(ablation_dir)
    if not results:
        log.warning("No metric JSON files found — skipping benchmark plot.")
        return

    out = output or (ablation_dir / "benchmark.png")
    plot_benchmark(
        results=results,
        primary_overrides={},
        output=out,
        show=show,
        figsize=None,
    )
    log.info("Benchmark plot saved to %s", out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run downstream evaluation tasks for all models in an ablation directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ablation-dir", "-d",
        required=True,
        type=Path,
        help="Path to the ablation directory containing model sub-directories.",
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        required=True,
        metavar="TASK",
        help=(
            "Downstream tasks to run. "
            f"Known tasks: {', '.join(sorted(TASK_CONFIG_FILES))}."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing task YAML configs. "
            f"Defaults to {_DEFAULT_CONFIG_DIR}."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a (model, task) pair if results_{task}.json already exists "
            "in {model_dir}/metrics/."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a benchmark comparison plot after all tasks complete.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window (only save to file).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save path for the benchmark plot. Defaults to {ablation_dir}/benchmark.png.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all known task names and their config files, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        print("Known tasks and their default config files:")
        for task, cfg_file in sorted(TASK_CONFIG_FILES.items()):
            print(f"  {task:<25} {cfg_file}")
        sys.exit(0)

    ablation_dir = args.ablation_dir.expanduser().resolve()
    if not ablation_dir.is_dir():
        log.error("--ablation-dir does not exist: %s", ablation_dir)
        sys.exit(1)

    config_dir = (args.config_dir or _DEFAULT_CONFIG_DIR).expanduser().resolve()
    if not config_dir.is_dir():
        log.error("--config-dir does not exist: %s", config_dir)
        sys.exit(1)

    run_ablation(
        ablation_dir=ablation_dir,
        tasks=args.tasks,
        config_dir=config_dir,
        skip_existing=args.skip_existing,
        plot=args.plot,
        plot_show=not args.no_show,
        plot_output=args.plot_output,
    )


if __name__ == "__main__":
    main()
