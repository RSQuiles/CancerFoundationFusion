"""
Unified runner for downstream fine-tuning tasks using the task registry.

This script loads a task from the registry and runs the training loop.
Usage:
    python run_downstream_task.py --config <config_path> --task <task_name>
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import hydra

# Import task implementations to register them
from evaluate.finetune.downstream_tasks_impl import CancTypeClassTask, DeconvTask
from evaluate.finetune.downstream_task import TaskRegistry
from base_downstream_runner import BaseDownstreamRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def load_runner_config(config_path: str | Path) -> DictConfig:
    """
    Load YAML config into OmegaConf DictConfig.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file.

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

    return cfg


def main(config_path: str, task_name: str | None = None) -> dict:
    """
    Main entry point for running a downstream task.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    task_name : str, optional
        Task name to run. If None, will infer from config or list available tasks.

    Returns
    -------
    dict
        Final evaluation metrics from the task.
    """
    # Load config
    cfg = load_runner_config(config_path)
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

    # Get task from registry
    try:
        task = TaskRegistry.get_task(task_name)
    except KeyError as e:
        log.error(str(e))
        log.info(f"Available tasks: {', '.join(TaskRegistry.list_tasks())}")
        raise

    # Create and run runner
    runner = BaseDownstreamRunner(cfg, task)
    results = runner.run()

    if runner.is_master:
        log.info("=" * 60)
        log.info(f"Task '{task_name}' completed successfully")
        log.info(f"Final metrics: {results}")
        log.info("=" * 60)

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
        results = main(args.config, args.task)
        sys.exit(0)
    except Exception as e:
        log.exception(f"Task failed with error: {e}")
        sys.exit(1)
