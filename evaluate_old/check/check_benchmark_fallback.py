"""Self-check for the ``metrics_old`` fallback in ``plot_ablation_benchmark``.

Anything ``metrics/`` does not provide — a whole task file, or single metrics inside
one — is filled in from a sibling ``metrics_old/``. That is convenient while only
part of a sweep has been recomputed and dangerous if it happens quietly, so the
rules are pinned here: metrics/ always wins, borrowing is per key, and every
substitution is reported.

Needs matplotlib and numpy (imported by the module under test); no cluster, GPU,
checkpoints or metric JSONs of your own.

    python evaluate/check/check_benchmark_fallback.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.plot import plot_ablation_benchmark as pab  # noqa: E402

FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label
          + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILED.append(label)


def write(directory: Path, task: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"results_{task}.json").write_text(json.dumps(payload))


def collect(model_dir: Path, **kwargs) -> tuple[dict, str]:
    """collect_model_metrics with a clean event log and captured stderr."""
    pab._FALLBACK_EVENTS.clear()
    buf = io.StringIO()
    with redirect_stderr(buf):
        result = pab.collect_model_metrics(model_dir, **kwargs)
    return result, buf.getvalue()


def check_per_key_merge(root: Path) -> None:
    print("\n-- per-metric fallback --")
    model = root / "unified"
    # The real case: survival re-run has written c_index, but the SurvBoard metric
    # pass has not been repeated yet, so the other three live in metrics_old.
    write(model / "metrics", "survival", {"c_index": 0.70})
    write(model / "metrics_old", "survival", {
        "c_index": 0.61, "antolini_concordance": 0.62, "ibs": 0.18,
        "d_calibration": 4.2,
    })

    data, err = collect(model)
    surv = data["survival"]
    check("metrics/ wins on a shared key", surv["c_index"] == 0.70, str(surv["c_index"]))
    check("missing keys borrowed", surv["antolini_concordance"] == 0.62
          and surv["ibs"] == 0.18 and surv["d_calibration"] == 4.2)
    check("borrowing is reported", "[metrics_old]" in err and "survival" in err, err)
    check("report names the borrowed keys", "antolini_concordance" in err, err)
    check("event recorded", len(pab.fallback_events()) == 1)
    _, _, keys = pab.fallback_events()[0]
    check("event lists exactly the borrowed keys",
          set(keys) == {"antolini_concordance", "ibs", "d_calibration"}, str(keys))


def check_whole_task(root: Path) -> None:
    print("\n-- whole-task fallback --")
    model = root / "dat"
    write(model / "metrics", "survival", {"c_index": 0.7})
    write(model / "metrics_old", "deconv", {"mean_pearson_r_present": 0.55})

    data, err = collect(model)
    check("task absent from metrics/ is taken wholesale",
          data.get("deconv", {}).get("mean_pearson_r_present") == 0.55)
    check("task present in metrics/ is untouched", data["survival"] == {"c_index": 0.7})
    check("whole-task substitution reported", "taken from metrics_old/" in err, err)
    check("recorded with keys=None", pab.fallback_events()[0][2] is None)


def check_no_fallback_cases(root: Path) -> None:
    print("\n-- when nothing should happen --")
    model = root / "mmd"
    write(model / "metrics", "survival", {"c_index": 0.7, "ibs": 0.2})
    data, err = collect(model)
    check("no metrics_old/ -> unchanged", data == {"survival": {"c_index": 0.7, "ibs": 0.2}})
    check("and silent", err == "", err)

    write(model / "metrics_old", "survival", {"c_index": 0.1, "ibs": 0.9})
    data, err = collect(model)
    check("nothing missing -> nothing borrowed",
          data == {"survival": {"c_index": 0.7, "ibs": 0.2}}, str(data))
    check("and still silent", err == "" and pab.fallback_events() == [], err)

    data, err = collect(model, fallback_dirname=None)
    check("fallback_dirname=None disables it",
          data == {"survival": {"c_index": 0.7, "ibs": 0.2}} and err == "")


def check_only_old(root: Path) -> None:
    print("\n-- a model with only metrics_old/ --")
    model = root / "legacy_only"
    write(model / "metrics_old", "survival", {"c_index": 0.42})

    check("is_model_dir (metrics/ only) says no", not pab.is_model_dir(model))
    check("is_benchmark_model_dir says yes", pab.is_benchmark_model_dir(model))

    data, _ = collect(model)
    check("its results are read", data["survival"]["c_index"] == 0.42)

    # And it survives the directory walk, which is what puts it in the figure.
    pab._FALLBACK_EVENTS.clear()
    with redirect_stderr(io.StringIO()):
        results = pab.collect_metrics(root)
    check("collect_metrics includes it", "legacy_only" in results)
    check("collect_metrics still includes normal models", "unified" in results)


def check_empty_dirs(root: Path) -> None:
    print("\n-- degenerate directories --")
    empty = root / "empty"
    (empty / "metrics").mkdir(parents=True)
    (empty / "metrics_old").mkdir(parents=True)
    check("no results_*.json anywhere -> not a model dir",
          not pab.is_benchmark_model_dir(empty))

    unreadable = root / "broken"
    (unreadable / "metrics").mkdir(parents=True)
    (unreadable / "metrics" / "results_survival.json").write_text("{not json")
    write(unreadable / "metrics_old", "survival", {"c_index": 0.33})
    data, err = collect(unreadable)
    check("a corrupt file in metrics/ falls through to metrics_old",
          data.get("survival", {}).get("c_index") == 0.33, str(data))
    check("and the read failure is reported", "Could not read" in err, err)


def check_config_path(root: Path) -> None:
    """The config-driven selection must discover metrics_old-only runs too."""
    print("\n-- config-driven selection --")
    cfg = root / "cmp.yaml"
    cfg.write_text(
        "title: t\n"
        "groups:\n"
        f"  - name: G\n    dir: {root.as_posix()}\n    all_models: true\n",
        encoding="utf-8",
    )
    config = pab.load_config(cfg)
    picked = {name for _, members in config.groups for name, _ in members}
    check("all_models finds the metrics_old-only run", "legacy_only" in picked,
          str(sorted(picked)))
    check("and the empty directory is excluded", "empty" not in picked,
          str(sorted(picked)))

    pab._FALLBACK_EVENTS.clear()
    with redirect_stderr(io.StringIO()):
        results, groups = pab.collect_from_config(config)
    check("collect_from_config keeps it", "legacy_only" in results, str(sorted(results)))
    check("and merges the borrowed metrics",
          results["unified"]["survival"]["antolini_concordance"] == 0.62)


def check_plots_end_to_end(root: Path) -> None:
    """The merged values must actually reach a rendered figure."""
    print("\n-- end to end --")
    pab._FALLBACK_EVENTS.clear()
    with redirect_stderr(io.StringIO()):
        results = pab.collect_metrics(root)

    out = root / "bench.png"
    with redirect_stderr(io.StringIO()):
        written = pab.plot_benchmark(
            results=results,
            primary_overrides={},
            output=out,
            show=False,
            figsize=(10, 6),
            metric_subsets={"survival": ["c_index", "antolini_concordance"]},
        )
    check("figure written", written == [out] and out.exists())
    check("borrowed metric is plottable",
          "antolini_concordance" in results["unified"]["survival"])


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        check_per_key_merge(root)
        check_whole_task(root)
        check_no_fallback_cases(root)
        check_only_old(root)
        check_empty_dirs(root)
        check_config_path(root)
        check_plots_end_to_end(root)

    print()
    if FAILED:
        print(f"FAILURES ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
