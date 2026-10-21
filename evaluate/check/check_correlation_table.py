"""Self-check for ``correlate_metrics_downstream.py``.

    python evaluate/check/check_correlation_table.py

Builds a synthetic ablation tree whose correlations are known by construction, then
asserts the script recovers them — no cluster, GPU, checkpoints or real metrics.

The statistics are pinned against cases with an analytic answer (a perfect monotone
relationship, a reversed one, a constant column) rather than against whatever the
implementation happened to produce, so a sign flip or an off-by-one in the FDR
adjustment cannot pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from evaluate.check.correlate_metrics_downstream import (  # noqa: E402
    CorrConfig,
    benjamini_hochberg,
    collect,
    correlate,
    load_config,
    sort_rows,
    spearman,
)

FAILURES: list[str] = []


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Spearman of perfectly monotone data comes back as 0.9999999999999999,
    not 1.0 — the rho is a Pearson correlation computed on ranks. Comparing it
    exactly would make these checks pass or fail on the sample size."""
    return abs(float(a) - float(b)) < tol


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f'  [{detail}]' if detail else ''}")
    if not ok:
        FAILURES.append(label)


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

def make_run(
    group_dir: Path,
    name: str,
    accuracy: float | None,
    internal: dict,
    with_results: bool = True,
) -> Path:
    """One model dir with unified_metrics.json and (optionally) the task results."""
    d = group_dir / name / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "unified_metrics.json").write_text(
        json.dumps({**internal, "panel_hash": "p0", "metrics_version": 2})
    )
    if with_results and accuracy is not None:
        (d / "results_canc_type_class.json").write_text(
            json.dumps({"accuracy": accuracy, "f1_weighted": accuracy * 0.95})
        )
    return group_dir / name


def build_tree(root: Path, n: int = 8) -> Path:
    """One group of *n* runs with metrics of known relationship to accuracy."""
    group = root / "ablation_demo"
    for i in range(n):
        acc = 0.30 + 0.05 * i                      # strictly increasing
        make_run(group, f"m{i}", acc, {
            "perfect_up": float(i),                # rho = +1
            "perfect_down": float(-i),             # rho = -1
            "constant": 1.0,                       # undefined
            "noisy": float((i * 7) % n),           # something in between
            "recon_pearson_r": 0.5 + 0.01 * i,
            "geometry_pr_pooled": 40.0 - i,        # rho = -1
        })
    return group


def write_config(path: Path, group_dir: Path, out: Path, **extra) -> Path:
    lines = [f"output: {out.as_posix()}"]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    lines += [
        "groups:",
        "  - name: demo",
        f"    dir: {group_dir.as_posix()}",
        "    all_models: true",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #

def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    group = build_tree(tmp)

    # ── the statistics, standalone ──────────────────────────────────────────
    print("\nstatistics")
    x = np.arange(10.0)
    check("monotone increasing gives rho = +1", close(spearman(x, x * 3 + 1)[0], 1.0))
    check("monotone decreasing gives rho = -1", close(spearman(x, -x)[0], -1.0))
    check("rank-based, not value-based",
          close(spearman(x, np.exp(x))[0], 1.0), "exp() is monotone")
    check("constant column is undefined, not zero",
          np.isnan(spearman(x, np.ones(10))[0]))
    check("fewer than 3 points is undefined",
          np.isnan(spearman(np.arange(2.0), np.arange(2.0))[0]))

    q = benjamini_hochberg([0.001, 0.01, 0.5, float("nan")])
    check("BH preserves NaN in place", np.isnan(q[3]))
    check("BH is monotone and >= p",
          q[0] <= q[1] <= q[2] and all(
              qi >= pi for qi, pi in zip(q[:3], [0.001, 0.01, 0.5])),
          f"{[round(v, 4) for v in q[:3]]}")
    check("BH of a single p is that p", benjamini_hochberg([0.02])[0] == 0.02)
    check("largest p is unchanged by BH", abs(q[2] - 0.5) < 1e-12)

    # ── collection ──────────────────────────────────────────────────────────
    print("\ncollection")
    cfg_path = write_config(tmp / "cfg.yaml", group, tmp / "out.png")
    config = load_config(cfg_path)
    internal, target, groups = collect(config)
    check("every run collected", len(internal) == 8, f"{len(internal)}")
    check("target read from results_canc_type_class.json",
          abs(target["m0"] - 0.30) < 1e-9 and abs(target["m7"] - 0.65) < 1e-9)
    check("primary metric resolves to accuracy",
          config.resolved_target_metric == "accuracy")

    # A run with internal metrics but no task results must be dropped, not counted
    # with a missing y — that would silently change which runs a row describes.
    make_run(group, "no_results", None, {"perfect_up": 99.0}, with_results=False)
    internal2, _, _ = collect(load_config(cfg_path))
    check("run without task results is dropped", "no_results" not in internal2,
          f"{len(internal2)} runs")

    # ── correlation ─────────────────────────────────────────────────────────
    print("\ncorrelation")
    keys = ["perfect_up", "perfect_down", "constant", "noisy",
            "recon_pearson_r", "geometry_pr_pooled"]
    rows = {r.key: r for r in correlate(internal, target, groups, keys, config)}
    check("perfectly aligned metric gives rho = +1", close(rows["perfect_up"].rho, 1.0))
    check("perfectly reversed metric gives rho = -1", close(rows["perfect_down"].rho, -1.0))
    check("constant metric is unusable, not 0",
          not rows["constant"].usable and "constant" in rows["constant"].note)
    check("n is the paired count", rows["perfect_up"].n == 8, f"{rows['perfect_up'].n}")
    check("a decreasing metric correlates negatively",
          close(rows["geometry_pr_pooled"].rho, -1.0))
    check("significant q for a perfect correlation at n=8",
          rows["perfect_up"].q < 0.05, f"q={rows['perfect_up'].q:.4g}")
    check("catalogued metric picks up its label and direction",
          rows["recon_pearson_r"].label == "Pearson R"
          and rows["recon_pearson_r"].direction == "up")
    check("uncatalogued metric falls back to its key",
          rows["noisy"].label == "noisy")

    # min_runs must suppress scoring, not silently produce a 3-point rho.
    strict = CorrConfig(groups=config.groups, min_runs=20)
    strict_rows = {r.key: r for r in
                   correlate(internal, target, groups, ["perfect_up"], strict)}
    check("min_runs leaves an under-powered row unscored",
          not strict_rows["perfect_up"].usable
          and "run(s)" in strict_rows["perfect_up"].note)

    # ── sorting ─────────────────────────────────────────────────────────────
    print("\nsorting")
    ordered = sort_rows(list(rows.values()), "abs_rho")
    check("strongest |rho| first", close(abs(ordered[0].rho), 1.0))
    check("unusable rows sink to the bottom", not ordered[-1].usable)
    signed = sort_rows(list(rows.values()), "rho")
    check("sort by signed rho puts +1 first and -1 last among usable",
          close(signed[0].rho, 1.0)
          and close([r for r in signed if r.usable][-1].rho, -1.0))
    check("sort 'config' preserves the given order",
          [r.key for r in sort_rows(list(rows.values()), "config")] == list(rows))

    # ── within_group scope ──────────────────────────────────────────────────
    print("\nwithin_group scope")
    two = tmp / "two_groups"
    ga, gb = two / "abl_a", two / "abl_b"
    # Same within-group relationship, opposite between-group offset: pooled rho is
    # dragged toward 0 while within_group recovers the true +1. This is the
    # confounding case the scope option exists for.
    for i in range(6):
        make_run(ga, f"a{i}", 0.30 + 0.02 * i, {"m": float(i)})
        make_run(gb, f"b{i}", 0.70 + 0.02 * i, {"m": float(i) - 40.0})
    cfg2 = two / "cfg.yaml"
    cfg2.write_text(
        f"output: {(two / 'o.png').as_posix()}\n"
        "min_runs: 5\ngroups:\n"
        f"  - name: A\n    dir: {ga.as_posix()}\n    all_models: true\n"
        f"  - name: B\n    dir: {gb.as_posix()}\n    all_models: true\n"
    )
    c2 = load_config(cfg2)
    i2, t2, g2 = collect(c2)
    pooled = correlate(i2, t2, g2, ["m"], c2)[0]
    c2.scope = "within_group"
    grouped = correlate(i2, t2, g2, ["m"], c2)[0]
    check("within_group recovers the true relationship", close(grouped.rho, 1.0),
          f"pooled={pooled.rho:.3f} within={grouped.rho:.3f}")
    check("pooled is confounded by the between-group offset", pooled.rho < grouped.rho,
          f"pooled={pooled.rho:.3f}")
    check("group count reported", grouped.n_groups == 2)
    check("n stays the run count", grouped.n == 12, f"{grouped.n}")

    # ── CLI ─────────────────────────────────────────────────────────────────
    print("\ncli")
    script = PROJECT_ROOT / "evaluate" / "check" / "correlate_metrics_downstream.py"

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path),
                        "--list"], capture_output=True, text=True)
    check("--list exits 0", r.returncode == 0, r.stderr[-300:])
    check("--list shows the target values", "accuracy = 0.3000" in r.stdout)
    check("--list shows per-metric n", "n=8" in r.stdout)

    out = tmp / "fig.png"
    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path),
                        "--output", str(out), "--no-show"],
                       capture_output=True, text=True)
    check("figure written", r.returncode == 0 and out.is_file(),
          r.stderr[-500:] if r.returncode else f"{out.stat().st_size // 1024} KB")
    csv_path = out.with_suffix(".csv")
    check("csv written beside the figure", csv_path.is_file())
    if csv_path.is_file():
        text = csv_path.read_text()
        import csv as _csv
        got = {r["metric"]: r for r in _csv.DictReader(text.splitlines())}
        check("csv holds the raw rho",
              close(float(got["perfect_up"]["spearman_rho"]), 1.0)
              and close(float(got["perfect_down"]["spearman_rho"]), -1.0))
        check("csv leaves an unusable row blank",
              got["constant"]["spearman_rho"] == "")
        check("csv records the target", "canc_type_class:accuracy" in text)

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg2),
                        "--scope", "within_group", "--output", str(two / "g.png"),
                        "--no-show"], capture_output=True, text=True)
    check("--scope within_group runs", r.returncode == 0, r.stderr[-300:])

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg_path),
                        "--task", "nonexistent_task", "--no-show"],
                       capture_output=True, text=True)
    check("a task with no results exits 1 with a clear message",
          r.returncode == 1 and "results_nonexistent_task.json" in r.stderr)

    r = subprocess.run([sys.executable, str(script), "--config",
                        str(tmp / "nope.yaml"), "--no-show"],
                       capture_output=True, text=True)
    check("missing config exits 1", r.returncode == 1)

    # ── shipped example config ──────────────────────────────────────────────
    print("\nshipped example config")
    example = PROJECT_ROOT / "evaluate" / "check" / "example_correlation_config.yaml"
    check("example config exists", example.is_file())
    if example.is_file():
        import re
        raw = example.read_text()
        check("no unexpanded ${vars} outside the vars block",
              not re.search(r"\$\{(?!work|save)\w+\}", raw))
        try:
            import yaml
            parsed = yaml.safe_load(raw)
            check("example config parses", isinstance(parsed, dict))
            check("target task is the cancer-type one",
                  (parsed.get("target") or {}).get("task") == "canc_type_class")
            check("every listed metric is spelled like a real key",
                  all(isinstance(m, str) and m for m in parsed.get("metrics", [])))
            check("scope/sort are valid",
                  parsed.get("scope") in ("pooled", "within_group")
                  and parsed.get("sort") in ("abs_rho", "rho", "config"))
        except ImportError:
            print("  SKIP  PyYAML absent - example config not parsed")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("All correlation-table checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
