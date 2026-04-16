import argparse
import json
import os
import os
import sys
sys.path.insert(0, "./")
from pathlib import Path
from pytorch_lightning.callbacks import TQDMProgressBar
from enum import Enum

from utils_config import build_parser, _load_json_config, _flatten_sectioned_config, _parser_dest_names, _filter_known_config_keys, expand_env_vars, pretty_print_args

def get_args():
    parser = build_parser()

    # First pass: only read --config
    initial_args, remaining_argv = parser.parse_known_args()

    if initial_args.config is not None:
        nested_config = _load_json_config(initial_args.config)
        flat_config = _flatten_sectioned_config(nested_config)
        flat_config = _filter_known_config_keys(parser, flat_config)
        parser.set_defaults(**flat_config)

    # Second pass: full parse, with CLI overriding config defaults
    args = parser.parse_args()
    args = expand_env_vars(args)

    pretty_print_args(args)
    if args.save_dir is not None:
        save_resolved_config(args, Path(args.save_dir) / "config.resolved.json")

    return args

def save_resolved_config(args, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for key, value in vars(args).items():
        if isinstance(value, Enum):
            serializable[key] = value.value
        elif isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
