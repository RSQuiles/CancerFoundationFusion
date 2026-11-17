"""Self-check for ``experiments/migrate_results_layout.py`` on a throwaway root.

    python -m experiments.check.check_migrate_results
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish  # noqa: E402
from experiments.layout import ExperimentLayout  # noqa: E402
from experiments.migrate_results_layout import main as migrate  # noqa: E402


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")


def _old_tree(root: Path) -> None:
    res = root / "results"
    old_run = res / "tasks" / "reconstruction" / "m1" / "sc_s0_test_r25"
    _write(old_run / "metrics.json", {"model": "m1", "head": "gen"})
    _write(res / "tasks" / "reconstruction" / "m1" / "sc_s0_test_r25_mvc" / "metrics.json", {"model": "m1", "head": "mvc"})
    _write(res / "tasks" / "reconstruction" / "_plots" / "sc_s0_test_r25" / "cmp" / "manifest.json",
           {"models": ["m1"], "plan": "sc:s0:test:r25"})
    _write(res / "tasks" / "geneicl_benchmark" / "m1" / "run.json",
           {"out_dir": str(res / "tasks" / "geneicl_benchmark" / "m1")})
    # comparison tables in assessment/, named <task>_<plan>[_<name>]
    _write(res / "assessment" / "reconstruction_sc_s0_test_r25_my_cmp" / "manifest.json",
           {"models": ["m1"], "plan": "sc:s0:test:r25"})
    _write(res / "assessment" / "reconstruction_sc_s0_test_r25" / "manifest.json",
           {"models": ["m1"], "plan": "sc:s0:test:r25"})
    _write(res / "assessment" / "geneicl_benchmark_main" / "manifest.json", {"models": ["m1"], "suite": "benchmark"})
    _write(res / "assessment" / "geneicl_non_benchmark" / "manifest.json", {"models": ["m1"], "suite": "non_benchmark"})
    _write(res / "assessment" / "scib_tissue" / "manifest.json", {"models": ["m1"]})
    _write(res / "assessment" / "store_check_sc_s0" / "report.json", {"store": "x"})
    _write(res / "assessment" / "mystery" / "notes.txt", "keep me")
    _write(res / "assessment" / "reconstruction_broken" / "table.csv", "no manifest")


def main() -> None:
    print("\n-- migrate results layout --")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _old_tree(root)
        layout = ExperimentLayout(root)
        res = layout.results_dir
        before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        check("dry run moves nothing", migrate(["--root", tmp]) == 0
              and sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == before)

        check("apply succeeds", migrate(["--root", tmp, "--apply"]) == 0)
        check("task runs moved up, both heads", (layout.task_dir("reconstruction", "m1") / "sc_s0_test_r25" / "metrics.json").exists()
              and (layout.task_dir("reconstruction", "m1") / "sc_s0_test_r25_mvc" / "metrics.json").exists())
        check("figures moved up", (layout.task_plot_dir("reconstruction", "sc_s0_test_r25", "cmp") / "manifest.json").exists())
        check("named reconstruction comparison -> _compare/<plan>/<name>",
              (layout.task_compare_dir("reconstruction", "sc_s0_test_r25", "my_cmp") / "manifest.json").exists())
        check("unnamed reconstruction comparison -> _compare/<plan>/default",
              (layout.task_compare_dir("reconstruction", "sc_s0_test_r25", "default") / "manifest.json").exists())
        check("GeneICL comparisons by their manifest suite",
              (layout.task_compare_dir("geneicl_benchmark", "main") / "manifest.json").exists()
              and (layout.task_compare_dir("geneicl_non_benchmark", "default") / "manifest.json").exists())
        check("scIB analysis -> scib_benchmark/<name>", (layout.scib_dir("tissue") / "manifest.json").exists())
        check("store check -> _checks/store_<mod>_<split>", (layout.check_dir("store_sc_s0") / "report.json").exists())
        card = json.loads((layout.task_dir("geneicl_benchmark", "m1") / "run.json").read_text())
        check("paths inside JSON rewritten", card["out_dir"] == str(layout.task_dir("geneicl_benchmark", "m1")))
        check("tasks/ removed (empty)", not (res / "tasks").exists())
        check("unknown and manifest-less entries left; assessment/ kept",
              (res / "assessment" / "mystery" / "notes.txt").exists()
              and (res / "assessment" / "reconstruction_broken" / "table.csv").exists())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _old_tree(root)
        layout = ExperimentLayout(root)
        # A new run already wrote the same scIB analysis: nothing may move.
        _write(layout.scib_dir("tissue") / "manifest.json", {"models": ["m2"]})
        _write(layout.task_dir("reconstruction", "m2") / "p" / "metrics.json", {"model": "m2"})
        snapshot = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        check("an existing destination stops the run before any move",
              migrate(["--root", tmp, "--apply"]) == 1
              and sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == snapshot)
        (layout.scib_dir("tissue") / "manifest.json").unlink()
        layout.scib_dir("tissue").rmdir()
        check("an existing task directory is merged into", migrate(["--root", tmp, "--apply"]) == 0
              and (layout.task_dir("reconstruction", "m2") / "p" / "metrics.json").exists()
              and (layout.task_dir("reconstruction", "m1") / "sc_s0_test_r25" / "metrics.json").exists())
    finish()


if __name__ == "__main__":
    main()
