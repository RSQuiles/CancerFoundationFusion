"""Self-check for ``experiments/layout.py`` and the launcher's path resolution.

Runs anywhere: no cluster, GPU or torch. The layout is the one thing every writer
and reader must agree on, so it is pinned here.

    python -m experiments.check.check_layout
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from experiments.config import load_launch_config
from experiments.launch import _resolve_model_runs, launch_models
from experiments.layout import (
    DEFAULT_ROOT,
    ROOT_ENV_VAR,
    ExperimentLayout,
    canonical_modality,
    resolve_root,
    validate_model_name,
)
from experiments.runtime import resolve_model_config

ROOT = Path("/exp")
L = ExperimentLayout(ROOT)


def check_paths() -> None:
    print("\n-- paths --")
    check("raw h5ad named after its modality dir",
          L.raw_h5ad("sc") == ROOT / "data/single_cell/single_cell.h5ad")
    check("modality aliases agree",
          L.raw_h5ad("single_cell") == L.raw_h5ad("sc") and L.raw_h5ad("pb") == L.raw_h5ad("pseudobulk"))
    check("split files", L.split_index_file("bulk", "s0", "test") == ROOT / "data/bulk/splits/s0/test.txt")
    check("pipeline_ready per split", L.pipeline_ready("sc", "s0") == ROOT / "data/single_cell/pipeline_ready/s0")
    check("mix store", L.mix_dir("m") == ROOT / "data/pipeline_ready/m")
    check("checkpoint dir", L.checkpoint_dir("m1") == ROOT / "checkpoints/m1")
    check("embedding file", L.embedding_file("m1", "single_cell", "test") == ROOT / "embeddings/m1/sc_test.h5ad")
    check("alignment dir", L.alignment_dir("m1") == ROOT / "results/representation/_alignment/m1")
    check("task dir", L.task_dir("canc", "m1") == ROOT / "results/tasks/canc/m1")
    check("unknown modality rejected", raises(lambda: canonical_modality("rna")))
    check("unknown embedding part rejected", raises(lambda: L.embedding_file("m1", "sc", "val")))


def check_names() -> None:
    print("\n-- model names --")
    check("plain name accepted", validate_model_name("unified_dat.v2-a") == "unified_dat.v2-a")
    for bad in ("", "a/b", "../x", "_launches", "with space", "-lead"):
        check(f"rejects {bad!r}", raises(lambda b=bad: validate_model_name(b)))


def check_root_resolution() -> None:
    print("\n-- root resolution --")
    saved = os.environ.pop(ROOT_ENV_VAR, None)
    try:
        check("default root", resolve_root() == DEFAULT_ROOT)
        os.environ[ROOT_ENV_VAR] = "/env/root"
        check("env var beats default", resolve_root() == Path("/env/root"))
        check("explicit beats env var", resolve_root("/cli") == Path("/cli"))
    finally:
        os.environ.pop(ROOT_ENV_VAR, None)
        if saved is not None:
            os.environ[ROOT_ENV_VAR] = saved


def check_model_config_resolution() -> None:
    print("\n-- model config resolution --")
    cfg, ref = resolve_model_config(L, "m1", {"data": {"train_path": {"mix": "abc"}}})
    check("save_dir derived", cfg["trainer"]["save_dir"] == str(ROOT / "checkpoints/m1"))
    check("wandb_name defaults to the name", cfg["trainer"]["wandb_name"] == "m1")
    check("mix reference resolved", cfg["data"]["train_path"] == str(ROOT / "data/pipeline_ready/abc"))
    check("reference recorded", ref == {"mix": "abc"})
    cfg, ref = resolve_model_config(L, "m1", {"data": {"train_path": {"pipeline_ready": "bulk:s0"}}})
    check("pipeline_ready reference resolved",
          cfg["data"]["train_path"] == str(ROOT / "data/bulk/pipeline_ready/s0"))
    cfg, ref = resolve_model_config(L, "m1", {"data": {"train_path": "/lit"}})
    check("literal path kept", cfg["data"]["train_path"] == "/lit" and ref is None)
    check("hand-written save_dir refused",
          raises(lambda: resolve_model_config(L, "m1", {"trainer": {"save_dir": "/x"}})))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def check_launcher_dry_run() -> None:
    print("\n-- launcher dry run --")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = {"train_entrypoint": "pretrain:main", "base_config": {"data": {"train_path": "/d"}}}
        cfg_path = _write(tmp / "cfg.json", {**base, "launch_name": "t",
                                             "models": [{"name": "a", "overrides": {}}, {"name": "b", "overrides": {}}]})
        cfg = load_launch_config(cfg_path, root_override=tmp / "root")
        layout = cfg.layout
        summary = launch_models(cfg, cli_dry_run=True)
        check("one checkpoint dir per model",
              all(layout.checkpoint_dir(m).is_dir() for m in ("a", "b")))
        check("run card written", json.loads(layout.run_card("a").read_text())["name"] == "a")
        check("placeholder checkpoint resolvable", layout.latest_checkpoint("a") is not None)
        check("launch summary lists the models",
              summary["models"] == ["a", "b"] and len(list(layout.launches_dir.glob("t_*.json"))) == 1)
        check("dry-run leftovers do not block a second launch",
              not raises(lambda: launch_models(load_launch_config(cfg_path, root_override=tmp / "root"),
                                               cli_dry_run=True)))

        # A real (non-dry-run) card makes the directory occupied.
        card = json.loads(layout.run_card("a").read_text())
        card["dry_run"] = False
        layout.run_card("a").write_text(json.dumps(card))
        check("existing model refused without resume",
              raises(lambda: launch_models(load_launch_config(cfg_path, root_override=tmp / "root"),
                                           only=["a"], cli_dry_run=True), FileExistsError))

        dup = _write(tmp / "dup.json", {**base, "models": [{"name": "a"}, {"name": "a"}]})
        check("duplicate names refused", raises(lambda: load_launch_config(dup)))
        legacy = _write(tmp / "legacy.json", {**base, "output_dir": "ablation_runs/x", "ablations": []})
        check("legacy output_dir refused", raises(lambda: load_launch_config(legacy)))
        with_save = _write(tmp / "save.json", {**base, "models": [
            {"name": "c", "overrides": {"trainer": {"save_dir": "/old/ablation_x/c"}}}]})
        check("legacy save_dir refused",
              raises(lambda: launch_models(load_launch_config(with_save, root_override=tmp / "root"),
                                           cli_dry_run=True)))
        check("--only with an unknown name refused",
              raises(lambda: launch_models(load_launch_config(cfg_path, root_override=tmp / "r2"),
                                           only=["zzz"], cli_dry_run=True)))

    example = PROJECT_ROOT / "experiments/configs/example_unifiedfm.json"
    check("example config loads", not raises(lambda: load_launch_config(example, root_override="/exp")))
    legacy_dir = PROJECT_ROOT / "experiments/configs/legacy"
    check("every legacy config fails fast",
          all(raises(lambda p=p: _resolve_model_runs(load_launch_config(p, root_override="/never")))
              for p in sorted(legacy_dir.glob("*.json"))))


def main() -> None:
    check_paths()
    check_names()
    check_root_resolution()
    check_model_config_resolution()
    check_launcher_dry_run()
    finish()


if __name__ == "__main__":
    main()
