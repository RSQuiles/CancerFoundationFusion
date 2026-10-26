"""Self-check for ``experiments/rename_model.py`` on a throwaway experiments root.

Builds every directory a model can own plus every reference file, renames the
model, and checks that the old name is gone from directory names and from the
JSON fields the layout says may hold it. Also checks the refusals and that an
injected failure rolls everything back.

    python -m experiments.check.check_rename
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from experiments.layout import ExperimentLayout
from experiments.rename_model import RenameError, apply_plan, build_plan

OLD, NEW, OTHER = "unified_v1", "unified_paired_dat_s0", "bulk_only"


def _no_jobs(_model: str) -> list[str]:
    return []


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_root(root: Path) -> tuple[ExperimentLayout, Path]:
    layout = ExperimentLayout(root)
    ck = layout.checkpoint_dir(OLD)
    save_dir = str(ck)
    _write(layout.run_card(OLD), {
        "name": OLD, "previous_names": [],
        "model_config": {"model_name": OLD, "trainer": {"save_dir": save_dir, "wandb_name": OLD}},
    })
    _write(ck / "model_config.json", {"model_name": OLD, "trainer": {"save_dir": save_dir, "wandb_name": OLD}})
    _write(ck / "job_payload.json", {"model_name": OLD, "model_config_path": f"{save_dir}/model_config.json"})
    _write(ck / "config.resolved.json", {"save_dir": save_dir, "wandb_name": OLD, "train_path": "/data/x"})
    _write(ck / "run_result.json", {"model_name": OLD, "checkpoint_path": f"{save_dir}/epoch_00.ckpt"})
    (ck / "epoch_00.ckpt").write_bytes(b"binary")
    _write(layout.embed_card(OLD), {"model": OLD, "checkpoint": f"{save_dir}/epoch_00.ckpt"})
    (layout.embedding_dir(OLD) / "sc_test.h5ad").write_bytes(b"h5")
    _write(layout.representation_dir(OLD) / "umap.json", {"model": OLD})
    _write(layout.alignment_dir(OLD) / "meta.json", {"model": OLD})
    _write(layout.task_dir("canc_type", OLD) / "results.json", {"model": OLD, "accuracy": 0.5})
    _write(layout.task_dir("canc_type", OTHER) / "results.json", {"model": OTHER})
    _write(layout.launches_dir / "launch_1.json", {
        "models": [OLD, OTHER],
        "runs": [{"model_name": OLD, "checkpoint_dir": save_dir}, {"model_name": OTHER}],
    })
    _write(layout.assessment_dir("scib") / "manifest.json", {"models": {OLD: "row1", OTHER: "row2"}})
    cfg = root.parent / "launch_cfg.json"
    _write(cfg, {"models": [{"name": OLD, "overrides": {"trainer": {"wandb_name": OLD}}}, {"name": OTHER}]})
    return layout, cfg


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _json_mentions_old(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("previous_names", None)
    return OLD in json.dumps(data)


def check_rename() -> None:
    print("\n-- rename --")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        layout, cfg = _build_root(root)
        plan = build_plan(layout, OLD, NEW, [cfg], job_lookup=_no_jobs)
        check("plans every owned directory", len(plan.moves) == 5)
        before = _snapshot(root)
        apply_plan(layout, plan)

        check("no directory still named OLD",
              not any(p.name == OLD for p in root.rglob("*")))
        check("every owned dir now under NEW",
              all(d.exists() for d in layout.model_owned_dirs(NEW)[:4]) and layout.task_dir("canc_type", NEW).exists())
        check("other model untouched", layout.task_dir("canc_type", OTHER).exists())
        leftovers = [p for p in root.rglob("*.json") if _json_mentions_old(p)]
        check("no JSON mentions OLD outside the history", not leftovers)
        card = json.loads(layout.run_card(NEW).read_text())
        check("run card renamed", card["name"] == NEW)
        check("history recorded", [h["name"] for h in card["previous_names"]] == [OLD])
        check("save_dir rewritten",
              json.loads((layout.checkpoint_dir(NEW) / "config.resolved.json").read_text())["save_dir"]
              == str(layout.checkpoint_dir(NEW)))
        check("non-path field left alone",
              json.loads((layout.checkpoint_dir(NEW) / "config.resolved.json").read_text())["train_path"] == "/data/x")
        check("binary checkpoint carried over", (layout.checkpoint_dir(NEW) / "epoch_00.ckpt").read_bytes() == b"binary")
        manifest = json.loads((layout.assessment_dir("scib") / "manifest.json").read_text())
        check("manifest dict key renamed", set(manifest["models"]) == {NEW, OTHER})
        launch = json.loads((layout.launches_dir / "launch_1.json").read_text())
        check("launch summary list renamed", launch["models"] == [NEW, OTHER])
        raw_cfg = json.loads(cfg.read_text())
        check("launch config entry renamed",
              raw_cfg["models"][0]["name"] == NEW
              and raw_cfg["models"][0]["overrides"]["trainer"]["wandb_name"] == NEW
              and raw_cfg["models"][1]["name"] == OTHER)
        log = layout.rename_log.read_text().strip().splitlines()
        check("rename logged", len(log) == 1 and json.loads(log[0])["new"] == NEW)
        check("tree changed", _snapshot(root) != before)

        # Renaming back restores the directory names.
        apply_plan(layout, build_plan(layout, NEW, OLD, job_lookup=_no_jobs))
        check("rename back works", layout.checkpoint_dir(OLD).is_dir() and not layout.checkpoint_dir(NEW).exists())


def check_refusals() -> None:
    print("\n-- refusals --")
    with tempfile.TemporaryDirectory() as tmp:
        layout, _ = _build_root(Path(tmp) / "root")
        layout.checkpoint_dir(OTHER).mkdir(parents=True)
        check("NEW already exists", raises(lambda: build_plan(layout, OLD, OTHER, job_lookup=_no_jobs), RenameError))
        check("OLD owns nothing", raises(lambda: build_plan(layout, "ghost", NEW, job_lookup=_no_jobs), RenameError))
        check("invalid NEW", raises(lambda: build_plan(layout, OLD, "bad/name", job_lookup=_no_jobs), ValueError))
        check("same name", raises(lambda: build_plan(layout, OLD, OLD, job_lookup=_no_jobs), RenameError))
        check("live job", raises(
            lambda: build_plan(layout, OLD, NEW, job_lookup=lambda m: [f"ufm_{m}"]), RenameError))
        plan = build_plan(layout, OLD, NEW, job_lookup=lambda m: None)
        check("missing squeue only warns", any("squeue" in w for w in plan.warnings))


def check_rollback() -> None:
    print("\n-- rollback --")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        layout, cfg = _build_root(root)
        before = _snapshot(root)
        plan = build_plan(layout, OLD, NEW, [cfg], job_lookup=_no_jobs)
        check("injected failure propagates",
              raises(lambda: apply_plan(layout, plan, _fail_after_moves=True), RuntimeError))
        check("tree restored byte for byte", _snapshot(root) == before)
        check("no rename logged", not layout.rename_log.exists())


def main() -> None:
    check_rename()
    check_refusals()
    check_rollback()
    finish()


if __name__ == "__main__":
    main()
