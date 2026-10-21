"""
Unified runner for downstream fine-tuning tasks using the task registry.

This script loads a task from the registry and runs the training loop.
Usage:
    python run_downstream_task.py --config <config_path> --task <task_name>
"""

import argparse
import logging
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf, DictConfig
import hydra

# Import task implementations to register them
from evaluate.finetune.tasks import (
    CancTypeClassTask,
    DeconvTask,
    SurvBoardTask,
    DrugSensitivityV2Task,
)
from evaluate.finetune.downstream_task import TaskRegistry
from evaluate.finetune.base_downstream_runner import BaseDownstreamRunner
from evaluate.finetune.normalization import (
    CANONICAL_KEY,
    SOURCE_KEY,
    drain_provenance,
    resolve_policy,
)
from evaluate.finetune.tasks.drug_sensitivity_v2 import (
    PrecomputedEmbedder,
    aggregate_drug_sensitivity_results,
    make_drug_endpoint_configs,
    precompute_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def load_runner_config(config_path: str | Path, checkpoint_path: str | Path | None) -> DictConfig:
    """
    Load YAML config into OmegaConf DictConfig.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file.
    checkpoint_path : str or Path or None
        Path to checkpoint file.

    Returns
    -------
    DictConfig
        Loaded configuration.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Loaded config is not a DictConfig: {type(cfg)}")

    if "finetune" not in cfg or cfg.finetune is None:
        raise ValueError("Config must contain 'finetune' section")
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        # Infer task from config keys and modify model path
        available_in_config = [
            key for key in cfg.finetune.keys()
            if cfg.finetune[key] is not None
        ]

        if len(available_in_config) < 1:
            raise ValueError(f"Faulty config, must specify downstream task")
        
        task = available_in_config[0]
        cfg.finetune[task]["pretrained_model_path"] = str(checkpoint_path)

    return cfg


def _normalization_record(policy) -> dict:
    """Provenance block written into results_<task>.json.

    ``normalize``/``source`` are what was asked for and where it came from;
    ``events`` is what each apply() actually did, including any safeguard skip.
    Without this you cannot tell later which preprocessing produced a given
    benchmark bar.
    """
    return {
        "normalize": policy.normalize,
        "source": policy.source,
        "summary": policy.describe(),
        "events": drain_provenance(),
    }


def main(
    config_path: str,
    checkpoint_path: str | Path | None = None,
    task_name: str | None = None,
    output_dir: str | Path | None = None,
    embedder=None,
    normalize: bool | None = None,
    ablation_dir: str | Path | None = None,
) -> dict:
    """
    Main entry point for running a downstream task.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    checkpoint_path : str or Path or None
        Path to checkpoint file.  Ignored when *embedder* is provided.
    task_name : str, optional
        Task name to run. If None, will infer from config or list available tasks.
    embedder : optional
        Pre-built embedder object (e.g. PCAEmbedder).  When supplied,
        checkpoint loading is skipped entirely.
    normalize : bool or None
        Global override for input normalization, in "do it?" space: True means the
        data must be CP10K+log1p normalized, False means nothing is applied
        anywhere.  None leaves the decision to the task config's ``normalize:``
        key, which itself defaults to False.  See
        ``evaluate/finetune/normalization.py``.
    ablation_dir : str or Path or None
        The ablation directory this model belongs to, folded into the task config.
        The survival task names its output directory ``{ablation}_{model}`` and the
        PCA baseline has no checkpoint to derive that from, so a sweep must pass it.
        See ``evaluate/finetune/survival_layout.py``.

    Returns
    -------
    dict
        Final evaluation metrics from the task.
    """
    # Start from a clean provenance collector: run_ablation_downstream calls this
    # once per (model, task) in one process, and a task that raised part-way would
    # otherwise leak its events into the next task's results JSON.
    drain_provenance()

    # Load config
    cfg = load_runner_config(config_path, checkpoint_path)
    log.info(f"Loaded config from {config_path}")

    # Determine task
    if task_name is None:
        # Try to infer from config keys
        available_in_config = [
            key for key in cfg.finetune.keys()
            if cfg.finetune[key] is not None
        ]

        if len(available_in_config) == 0:
            log.error("No tasks configured in finetune section")
            log.info(f"Available tasks: {', '.join(TaskRegistry.list_tasks())}")
            raise ValueError("Please specify --task or configure a task in 'finetune' section")

        if len(available_in_config) > 1 and not task_name:
            log.error(
                f"Multiple tasks found in config: {available_in_config}. "
                "Please specify which one to run with --task"
            )
            raise ValueError(f"Ambiguous config: specify --task from {available_in_config}")

        task_name = available_in_config[0]

    log.info(f"Running task: {task_name}")

    # Fold a CLI/orchestrator override into the config the tasks actually read, so
    # every task resolves the same policy from one place. SOURCE_KEY keeps the
    # reported provenance honest ("cli" rather than "config").
    if normalize is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.finetune[task_name][CANONICAL_KEY] = bool(normalize)
        cfg.finetune[task_name][SOURCE_KEY] = "cli"

    # Same idea for the ablation directory: the caller knows it, the config's own key
    # is hand-written and goes stale. Survival names its CSV directory from this.
    if ablation_dir is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.finetune[task_name]["ablation_dir"] = str(ablation_dir)
    policy = resolve_policy(cfg.finetune[task_name], None)
    log.info("Normalization policy for '%s': %s", task_name, policy.describe())

    # Get task from registry
    try:
        task = TaskRegistry.get_task(task_name)
    except KeyError as e:
        log.error(str(e))
        log.info(f"Available tasks: {', '.join(TaskRegistry.list_tasks())}")
        raise

    if task_name == "drug_sensitivity_v2":
        save_dir = Path(output_dir) if output_dir is not None else Path(cfg.finetune[task_name].pretrained_model_path).parent / "metrics"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load the model once and embed all cell lines once; every job then
        # performs a fast index lookup instead of re-running the transformer.
        if embedder is None:
            sys.path.insert(0, "../")
            from cancerfoundation.model.model import CancerFoundation
            checkpoint_path = str(cfg.finetune[task_name].pretrained_model_path)
            real_embedder = CancerFoundation.load_for_inference(checkpoint_path)
        else:
            real_embedder = embedder
        log.info("Pre-computing cell-line embeddings for drug sensitivity (runs once)...")
        emb_df = precompute_embeddings(real_embedder, cfg.finetune[task_name])
        log.info("Pre-computation done: %d cell lines x %d dims", *emb_df.shape)
        precomputed = PrecomputedEmbedder(emb_df)

        results = []
        for job in make_drug_endpoint_configs(cfg.finetune[task_name]):
            job_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            job_cfg.finetune[task_name].drug = job["drug"]
            job_cfg.finetune[task_name].endpoint = job["endpoint"]
            runner = BaseDownstreamRunner(job_cfg, TaskRegistry.get_task(task_name), embedder=precomputed)
            result = runner.run()
            results.append(result)
            if runner.is_master:
                with open(save_dir / f"results_{task_name}__{job['drug']}__{job['endpoint']}.json", "w") as f:
                    json.dump(result, f, indent=2)
        results = {
            "jobs": results,
            "aggregate": aggregate_drug_sensitivity_results(results),
            "normalization": _normalization_record(policy),
        }
        if runner.is_master:
            with open(save_dir / f"results_{task_name}.json", "w") as f:
                json.dump(results, f, indent=2)
        return results

    # Create and run runner
    runner = BaseDownstreamRunner(cfg, task, embedder=embedder)
    results = runner.run()

    if isinstance(results, dict):
        results["normalization"] = _normalization_record(policy)

    if runner.is_master:
        log.info("=" * 60)
        log.info(f"Task '{task_name}' completed successfully")
        log.info(f"Final metrics: {results}")
        log.info("=" * 60)

        if output_dir is not None:
            save_dir = Path(output_dir)
        else:
            save_dir = Path(cfg.finetune[task_name].pretrained_model_path).parent / "metrics"
        save_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / f"results_{task_name}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {results_path}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run downstream fine-tuning tasks on frozen pretrained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run cancer type classification
  python run_downstream_task.py --config cancer_anot_config.yaml --task canc_type_class

  # Run deconvolution (auto-detect from config)
  python run_downstream_task.py --config deconv_config.yaml

  # List available tasks
  python run_downstream_task.py --list-tasks
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file for the task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Name of the task to run (e.g., 'canc_type_class', 'deconv'). "
        "If not specified, will infer from config.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available registered tasks and exit.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force input normalization on (--normalize) or off (--no-normalize) for "
            "this task, overriding the config's 'normalize:' key. --normalize applies "
            "CP10K+log1p, skipping with a warning if the matrix does not look like "
            "raw counts. Omitted: the config decides, defaulting to off."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_tasks:
        print("Available downstream tasks:")
        for task_name in TaskRegistry.list_tasks():
            task = TaskRegistry.get_task(task_name)
            print(f"  - {task_name} (config key: {task.config_key})")
        sys.exit(0)

    if not args.config:
        print("ERROR: --config is required unless using --list-tasks")
        print("Use --help for more information.")
        sys.exit(1)

    try:
        results = main(args.config, task_name=args.task, normalize=args.normalize)
        sys.exit(0)
    except Exception as e:
        log.exception(f"Task failed with error: {e}")
        sys.exit(1)
