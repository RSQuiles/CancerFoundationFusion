import argparse
import csv
import json
import os
import os
import sys
sys.path.insert(0, "./")
from pathlib import Path
from pytorch_lightning.callbacks import TQDMProgressBar, Callback
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


class LossCSVLogger(Callback):
    """Append every logged train/ and val/ metric to a long-format CSV in save_dir.

    Columns: step, epoch, split, metric, value. Local, offline-plottable complement to W&B;
    active regardless of whether a W&B logger is configured. Rank-zero only.

    Train rows are written every ``interval`` optimizer steps (default: every step);
    validation rows are written once per validation run.
    """

    COLUMNS = ["step", "epoch", "split", "metric", "value"]

    def __init__(self, save_dir, interval=1, filename="metrics.csv"):
        super().__init__()
        self.path = os.path.join(save_dir, filename)
        self.interval = max(1, int(interval))
        self._need_header = not os.path.exists(self.path)

    def _append(self, trainer, metrics, split, prefix):
        if not trainer.is_global_zero:
            return
        step, epoch = trainer.global_step, trainer.current_epoch
        rows = []
        for k, v in metrics.items():
            if not k.startswith(prefix):
                continue
            try:
                val = float(v.item() if hasattr(v, "item") else v)
            except (ValueError, TypeError):
                continue
            rows.append([step, epoch, split, k[len(prefix):], val])
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            if self._need_header:
                w.writerow(self.COLUMNS)
                self._need_header = False
            w.writerows(rows)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.interval != 0:
            return
        self._append(trainer, trainer.logged_metrics, "train", "train/")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._append(trainer, trainer.callback_metrics, "val", "val/")


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
