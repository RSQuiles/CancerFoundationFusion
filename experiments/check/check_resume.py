"""Self-check for ``utils/resume.py`` and the launcher's resume handling.

Runs anywhere: no cluster, GPU or torch (the checkpoint dicts are built by hand).
The ``EpochRandomSampler`` part runs only where torch imports, and is skipped
otherwise.

    python -m experiments.check.check_resume
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from experiments.layout import ExperimentLayout
from experiments.runtime import describe_resume, resolve_model_config
from utils.resume import (
    DRY_RUN_CHECKPOINT,
    append_segment,
    archive_resolved_config,
    check_resume_epochs,
    compare_hparams,
    find_last_checkpoint,
    find_wandb_run_id,
    load_wandb_run,
    read_segments,
    resolve_checkpoint,
    save_wandb_run,
    segment_record,
    summarize_checkpoint,
    trim_metrics_csv,
)


def _ckpt(epoch: int, step: int, last_batch: bool | None, **extra) -> dict:
    out = {"epoch": epoch, "global_step": step, **extra}
    if last_batch is not None:
        out["loops"] = {"fit_loop": {"epoch_loop.batch_progress": {"is_last_batch": last_batch}}}
    return out


def _touch(path: Path, age: float) -> Path:
    path.write_text("x", encoding="utf-8")
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def check_checkpoint_lookup(tmp: Path) -> None:
    print("\n-- checkpoint lookup --")
    d = tmp / "ckpts"
    d.mkdir()
    check("empty dir -> None", find_last_checkpoint(d) is None)
    check("missing dir -> None", find_last_checkpoint(tmp / "nope") is None)
    _touch(d / "epoch_03.ckpt", 30)
    newest_real = _touch(d / "step_000500_epoch_04.ckpt", 20)
    _touch(d / DRY_RUN_CHECKPOINT, 0)
    check("newest real checkpoint wins over a newer dry_run.ckpt", find_last_checkpoint(d) == newest_real)
    check("'last' resolves the same way", resolve_checkpoint("last", d) == newest_real)
    check("no spec -> None", resolve_checkpoint(None, d) is None)
    check("explicit dry_run.ckpt refused", raises(lambda: resolve_checkpoint(str(d / DRY_RUN_CHECKPOINT), d), ValueError))
    check("missing explicit path refused", raises(lambda: resolve_checkpoint(str(d / "gone.ckpt"), d), FileNotFoundError))

    layout = ExperimentLayout(tmp / "root")
    model_dir = layout.checkpoint_dir("m")
    model_dir.mkdir(parents=True)
    _touch(model_dir / "epoch_00.ckpt", 10)
    _touch(model_dir / DRY_RUN_CHECKPOINT, 0)
    check("layout.latest_checkpoint skips the placeholder too", layout.latest_checkpoint("m").name == "epoch_00.ckpt")


def check_summary() -> None:
    print("\n-- checkpoint summary --")
    end = summarize_checkpoint(_ckpt(4, 5000, True), Path("epoch_04.ckpt"), "continue")
    check("end-of-epoch checkpoint: epoch 4 means 5 completed", end.epochs_completed == 5)
    check("end-of-epoch checkpoint is not mid-epoch", end.mid_epoch is False)
    check("continue: segment starts at the restored step", end.segment_start == 5000)

    mid = summarize_checkpoint(_ckpt(2, 2300, False), Path("step.ckpt"), "continue")
    check("mid-epoch checkpoint: epoch 2 means 2 completed", mid.epochs_completed == 2 and mid.mid_epoch is True)

    exact_old = summarize_checkpoint(_ckpt(2, 2300, False), Path("step.ckpt"), "exact")
    check("exact on a pre-segment checkpoint: schedule started at 0", exact_old.segment_start == 0)
    exact_new = summarize_checkpoint(
        _ckpt(7, 9000, False, lr_segment={"start_step": 5000, "total_steps": 95000}), Path("s.ckpt"), "exact"
    )
    check("exact keeps the checkpoint's own segment start", exact_new.segment_start == 5000)
    check("unknown loop state -> mid_epoch None", summarize_checkpoint(_ckpt(1, 10, None), Path("a"), "exact").mid_epoch is None)
    check("bad mode refused", raises(lambda: summarize_checkpoint(_ckpt(0, 0, True), Path("a"), "resume"), ValueError))

    check("epochs == completed refused (nothing to train)", raises(lambda: check_resume_epochs(end, 5), ValueError))
    check("epochs > completed accepted", not raises(lambda: check_resume_epochs(end, 100)))


def check_hparams() -> None:
    print("\n-- hparam comparison --")
    hp = {"nlayers": 8, "embsize": 256, "lr": 1e-4, "vocab": {"a": 0, "b": 1}, "data_path": "/d/a"}
    info = summarize_checkpoint(
        _ckpt(4, 100, True, hyper_parameters=hp, datamodule_hyper_parameters={"batch_size": 64}),
        Path("epoch_04.ckpt"), "continue",
    )
    check("identical config -> no changes", compare_hparams(info, dict(hp), {"batch_size": 64}) == [])
    check("architecture mismatch refused", raises(lambda: compare_hparams(info, {**hp, "nlayers": 12}), ValueError))
    check("vocab size mismatch refused", raises(lambda: compare_hparams(info, {**hp, "vocab": {"a": 0}}), ValueError))
    changes = compare_hparams(info, {**hp, "lr": 5e-5, "data_path": "/d/b"}, {"batch_size": 32})
    check("lr, data_path and batch_size changes reported", len(changes) == 3)
    check("keys absent from an old checkpoint are not compared",
          compare_hparams(info, {**hp, "do_mvc": True}) == [])


def check_side_files(tmp: Path) -> None:
    print("\n-- side files --")
    d = tmp / "side"
    d.mkdir()
    check("no wandb_run.json -> None", load_wandb_run(d) is None)
    check("no W&B record at all -> no id", find_wandb_run_id(d) is None)
    (d / "wandb" / "run-20261001_101010-old42").mkdir(parents=True)
    check("pre-existing run dir is the fallback id", find_wandb_run_id(d) == "old42")
    save_wandb_run(d, "abc123", "ent", "proj", "https://w/abc123")
    first = load_wandb_run(d)
    save_wandb_run(d, "abc123", "ent", "proj", "https://w/abc123")
    check("wandb id round-trips", first["id"] == "abc123" and load_wandb_run(d)["id"] == "abc123")
    check("wandb_run.json wins over the run dir", find_wandb_run_id(d) == "abc123")
    check("re-saving the same run keeps its created time", load_wandb_run(d)["created"] == first["created"])

    info = summarize_checkpoint(_ckpt(4, 5000, True), d / "epoch_04.ckpt", "continue")
    s0 = append_segment(d, segment_record(None, epochs=5, lr=1e-4, warmup=1000, seed=0))
    s1 = append_segment(d, segment_record(info, epochs=100, lr=1e-4, warmup=1500, seed=5000))
    segments = read_segments(d)
    check("segments are numbered in order", [s["index"] for s in segments] == [0, 1] and s1["index"] == 1)
    check("first segment is 'fresh' from step 0", s0["mode"] == "fresh" and s0["start_step"] == 0)
    check("resumed segment records mode, start and target",
          segments[1]["mode"] == "continue" and segments[1]["start_step"] == 5000
          and segments[1]["start_epoch"] == 5 and segments[1]["epochs"] == 100)
    check("segments name the checkpoint file, never a path", segments[1]["from_checkpoint"] == "epoch_04.ckpt")

    metrics = d / "metrics.csv"
    with metrics.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "split", "metric", "value"])
        for step in (100, 200, 300, 400):
            w.writerow([step, 0, "train", "loss", 1.0])
    check("nothing after the checkpoint -> file untouched", trim_metrics_csv(metrics, 400) == 0)
    moved = trim_metrics_csv(metrics, 250)
    with metrics.open(newline="") as f:
        kept = [int(r["step"]) for r in csv.DictReader(f)]
    discarded = list(d.glob("metrics.discarded_*.csv"))
    check("rows after the checkpoint step are moved", moved == 2 and kept == [100, 200])
    check("moved rows are kept in one discarded file", len(discarded) == 1)
    check("missing metrics.csv is fine", trim_metrics_csv(d / "none.csv", 0) == 0)

    resolved = d / "config.resolved.json"
    check("no previous config -> nothing archived", archive_resolved_config(resolved) is None)
    resolved.write_text("{}", encoding="utf-8")
    first_archive = archive_resolved_config(resolved)
    resolved.write_text("{}", encoding="utf-8")
    second_archive = archive_resolved_config(resolved)
    check("previous configs archived as seg0, seg1",
          first_archive.name == "config.resolved.seg0.json" and second_archive.name == "config.resolved.seg1.json")


def check_launcher(tmp: Path) -> None:
    print("\n-- launcher --")
    layout = ExperimentLayout(tmp / "launch_root")
    cfg = {"trainer": {"epochs": 100, "resume_from_checkpoint": "last", "resume_mode": "rewind"}}
    check("unknown resume_mode refused", raises(lambda: resolve_model_config(layout, "m", cfg), ValueError))
    cfg["trainer"]["resume_mode"] = "continue"
    resolved, _ = resolve_model_config(layout, "m", cfg)
    check("no checkpoint yet -> 'starts from scratch'", "from scratch" in describe_resume(layout, "m", resolved))
    model_dir = layout.checkpoint_dir("m")
    model_dir.mkdir(parents=True)
    _touch(model_dir / "epoch_04.ckpt", 5)
    (model_dir / "wandb_run.json").write_text(json.dumps({"id": "run42"}), encoding="utf-8")
    line = describe_resume(layout, "m", resolved)
    check("dry-run line names the checkpoint, target and W&B run",
          "epoch_04.ckpt" in line and "epoch 100" in line and "run42" in line)
    check("no resume key -> no line", describe_resume(layout, "m", {"trainer": {}}) is None)


def check_sampler() -> None:
    print("\n-- EpochRandomSampler --")
    try:
        from cancerfoundation.data.data_module import EpochRandomSampler
    except Exception as exc:  # noqa: BLE001 - torch may not load here
        print(f"  SKIP  torch/cancerfoundation not importable ({type(exc).__name__})")
        return
    data = list(range(50))

    def order(seed: int, epoch: int) -> list[int]:
        sampler = EpochRandomSampler(data, seed=seed)
        sampler.set_epoch(epoch)
        return list(sampler)

    check("a permutation of the data", sorted(order(0, 0)) == data)
    check("same (seed, epoch) -> same order, whatever ran before", order(0, 5) == order(0, 5))
    check("epochs differ", order(0, 0) != order(0, 1))
    check("seeds differ", order(0, 3) != order(1, 3))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        check_checkpoint_lookup(tmp)
        check_summary()
        check_hparams()
        check_side_files(tmp)
        check_launcher(tmp)
    check_sampler()
    finish()


if __name__ == "__main__":
    main()
