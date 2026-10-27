"""Self-checks for the unified-metrics comparison table.

Run directly; needs only matplotlib, numpy, pandas and PyYAML (no cluster, no GPU, no
container, no checkpoints):

    python evaluate/check/check_unified_table.py

Exits non-zero on failure. Everything is exercised against a temporary directory tree
of synthetic model dirs, so it never touches a real ablation directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from evaluate.plot.plot_unified_metrics_table import (  # noqa: E402
    DOWN,
    NONE,
    UP,
    MetricSpec,
    _read_scib_table,
    _SCIB_CACHE,
    collect_from_config,
    collect_scib_metrics,
    collect_unified_metrics,
    dense_ranks,
    fmt_value,
    is_unified_model_dir,
    load_config,
    plot_bars,
    plot_table,
    resolve_metrics,
    score_column,
    warn_panel_mismatch,
)

FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{('  - ' + detail) if detail else ''}")
        FAILED.append(label)


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

def make_model(ablation: Path, name: str, **overrides) -> Path:
    """One synthetic model dir with a metrics/unified_metrics.json."""
    metrics = {
        "recon_pearson_r": 0.60,
        "recon_mae_bins": 4.2,
        "recon_mae_per_bin": {"5": 3.1, "10": 4.4},     # nested - must be dropped
        "paired_cosine_sim_mean": 0.80,
        "paired_rank_mean": 3.0,
        "paired_l2_mean": 1.10,
        "paired_rank_l2_mean": 3.5,
        "paired_random_baseline_cosine": 0.10,
        "paired_n_pairs": 120,
        "agg_paired_cosine_pb_to_mean_sc": 0.90,
        "agg_synth_cosine_pb_to_mean_sc": 0.85,
        "agg_paired_l2_pb_to_mean_sc": 1.4,
        "agg_synth_l2_pb_to_mean_sc": 1.6,
        "contrastive_cross_cosine_mean": 0.55,
        "contrastive_within_bulk_cosine": 0.71,
        "contrastive_within_pb_cosine": 0.69,
        "contrastive_cross_l2_mean": 2.0,
        "contrastive_within_bulk_l2": 1.9,
        "contrastive_within_pb_l2": 1.8,
        "contrastive_mmd": 0.05,
        "contrastive_wasserstein": 0.31,
        "contrastive_bulk_source": "bulk",              # str - not plottable
        "skipped_families": {"recon": "no ckpt"},       # nested - must be dropped
        "panel_hash": "abc123def456",
        "panel_strategy": "consensus",
    }
    metrics.update(overrides)

    model_dir = ablation / name
    (model_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with (model_dir / "metrics" / "unified_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    return model_dir


def make_scib(ablation: Path, tag: str, model_names: list[str]) -> Path:
    """A scIB table in the real shape: X_cf_ prefixes plus a Metric Type row."""
    scib_dir = ablation / "_scib_metrics"
    scib_dir.mkdir(parents=True, exist_ok=True)
    csv = scib_dir / f"scib_{tag}.csv"

    header = ",Isolated labels,Silhouette label,cLISI,BRAS,iLISI,Batch correction,Total"
    lines = [header]
    for i, name in enumerate(model_names):
        v = 0.30 + 0.10 * i
        lines.append(
            f"X_cf_{name},{0.5 + 0.01 * i},{0.6},{0.9},{0.4},{v},{0.45},{0.55}"
        )
    lines.append("X_pca,0.40,0.50,0.85,0.20,0.15,0.25,0.35")
    # scib_metrics appends this descriptive row; it is not a model.
    lines.append("Metric Type,Bio conservation,Bio conservation,Bio conservation,"
                 "Batch correction,Batch correction,Aggregate score,Aggregate score")
    csv.write_text("\n".join(lines) + "\n")
    return csv


def write_config(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_scoring() -> None:
    print("\n[scoring]")

    # Direction flipping: for a ↓ metric the smallest value must score best.
    v = np.array([1.0, 2.0, 3.0])
    s_down = score_column(v, DOWN, "minmax")
    check("lower-is-better metric: lowest value scores 1.0",
          s_down[0] == 1.0 and s_down[-1] == 0.0, str(s_down))
    s_up = score_column(v, UP, "minmax")
    check("higher-is-better metric: highest value scores 1.0",
          s_up[-1] == 1.0 and s_up[0] == 0.0, str(s_up))

    # Degenerate columns must not invent an ordering.
    check("constant column scores 0.5 throughout",
          np.allclose(score_column(np.array([2.0, 2.0, 2.0]), UP, "minmax"), 0.5))
    check("single run scores 0.5",
          np.allclose(score_column(np.array([0.7]), UP, "minmax"), 0.5))
    check("all-NaN column stays NaN",
          np.all(np.isnan(score_column(np.array([np.nan, np.nan]), UP, "minmax"))))

    # NaN must not drag the finite values around.
    s = score_column(np.array([1.0, np.nan, 3.0]), UP, "minmax")
    check("NaN cell stays NaN, others still span 0..1",
          np.isnan(s[1]) and s[0] == 0.0 and s[2] == 1.0, str(s))

    # No-direction metrics get no score at all (drawn grey).
    check("direction=none produces no score",
          np.all(np.isnan(score_column(v, NONE, "minmax"))))

    # zscore and rank stay in range and keep the ordering.
    for mode in ("zscore", "rank"):
        s = score_column(np.array([1.0, 5.0, 9.0]), UP, mode)
        check(f"{mode}: in [0,1] and monotone",
              bool(np.all((s >= 0) & (s <= 1)) and s[0] < s[1] < s[2]), str(s))

    # Ranks: 1 = best, ties share the average.
    r = dense_ranks(np.array([0.9, 0.5, 0.7]), UP)
    check("ranks: 1 = best for higher-is-better", list(r) == [1.0, 3.0, 2.0], str(r))
    r = dense_ranks(np.array([0.9, 0.5, 0.7]), DOWN)
    check("ranks: 1 = best for lower-is-better", list(r) == [3.0, 1.0, 2.0], str(r))
    r = dense_ranks(np.array([1.0, 1.0, 2.0]), UP)
    check("ranks: ties get the average", list(r) == [2.5, 2.5, 1.0], str(r))
    r = dense_ranks(np.array([1.0, np.nan, 2.0]), UP)
    check("ranks: NaN stays NaN", r[0] == 2.0 and np.isnan(r[1]) and r[2] == 1.0, str(r))


def check_formatting() -> None:
    print("\n[formatting]")
    cases = {
        0.612345: "0.612",
        12.3456: "12.35",
        123.456: "123.5",
        0.0: "0",
        123456.0: "1.2e+05",
        float("nan"): "n/a",
        None: "n/a",
    }
    for value, expected in cases.items():
        got = fmt_value(value)
        check(f"fmt_value({value!r}) == {expected!r}", got == expected, got)


def check_collection(tmp: Path) -> tuple[dict, list]:
    print("\n[collection]")
    abl_a = tmp / "ablation_a"
    abl_b = tmp / "ablation_b"

    make_model(abl_a, "unified")
    make_model(abl_a, "cdd", recon_pearson_r=0.66, paired_rank_mean=2.0,
               contrastive_wasserstein=0.22)
    make_model(abl_a, "pca_baseline", recon_pearson_r=0.30)
    make_model(abl_b, "mix", recon_pearson_r=0.55, panel_hash="zzz999")
    make_model(abl_b, "mix2", recon_pearson_r=0.58)
    (abl_b / "not_a_model").mkdir(parents=True, exist_ok=True)

    make_scib(abl_a, "bulk_vs_pb", ["unified", "cdd", "pca_baseline"])

    check("is_unified_model_dir accepts a model dir",
          is_unified_model_dir(abl_a / "unified"))
    check("is_unified_model_dir rejects a dir without metrics",
          not is_unified_model_dir(abl_b / "not_a_model"))

    data = collect_unified_metrics(abl_a / "unified")
    check("nested recon_mae_per_bin is dropped", "recon_mae_per_bin" not in data)
    check("nested skipped_families is dropped", "skipped_families" not in data)
    check("scalar metrics survive", data.get("recon_pearson_r") == 0.60)
    check("provenance survives", data.get("panel_hash") == "abc123def456")

    _SCIB_CACHE.clear()
    table = _read_scib_table(abl_a, "bulk_vs_pb")
    check("scIB: X_cf_ prefix stripped", "unified" in table, str(list(table)))
    check("scIB: Metric Type row dropped", "Metric Type" not in table, str(list(table)))
    check("scIB: X_pca row kept as-is", "X_pca" in table, str(list(table)))

    scib = collect_scib_metrics(abl_a / "cdd", ["bulk_vs_pb"], ["iLISI", "BRAS"])
    check("scIB: joined on the model DIR name, not a display name",
          scib.get("scib:bulk_vs_pb:iLISI") == 0.40, str(scib))
    check("scIB: requested metrics only",
          set(scib) == {"scib:bulk_vs_pb:iLISI", "scib:bulk_vs_pb:BRAS"}, str(scib))

    cfg_path = write_config(tmp / "cfg.yaml", f"""
title: "Fixture"
style: heatmap
scib:
  tags: [bulk_vs_pb]
  metrics: [iLISI]
groups:
  - name: "A"
    dir: {abl_a.as_posix()}
    all_models: true
    exclude: [pca_baseline]
  - name: "B"
    dir: {abl_b.as_posix()}
    experiments:
      - {{name: "Mix", model: mix}}
""")
    config = load_config(cfg_path)
    results, groups = collect_from_config(config)

    check("all_models found both models", len(groups[0][1]) == 2, str(groups))
    check("exclude honoured", "pca_baseline" not in results, str(list(results)))
    check("'not_a_model' skipped by all_models",
          "not_a_model" not in results, str(list(results)))
    check("display name honoured", "Mix" in results, str(list(results)))
    check("groups preserved", [g for g, _ in groups] == ["A", "B"], str(groups))
    check("scIB column attached to group A runs",
          "scib:bulk_vs_pb:iLISI" in results["unified"], str(list(results["unified"])))
    check("no scIB table in ablation_b -> no scIB column there",
          "scib:bulk_vs_pb:iLISI" not in results["Mix"])

    specs = resolve_metrics(results, config)
    check("default metric set resolved", len(specs) > 10, str(len(specs)))
    check("scIB column is last", specs[-1].key == "scib:bulk_vs_pb:iLISI",
          specs[-1].key)
    check("contrastive_mmd is off by default",
          all(s.key != "contrastive_mmd" for s in specs))
    check("string metrics never become columns",
          all(s.key != "contrastive_bulk_source" for s in specs))
    check("count metrics not in the default set",
          all(s.key != "paired_n_pairs" for s in specs))

    footnote = warn_panel_mismatch(results)
    check("mismatched panel_hash is flagged", footnote is not None, str(footnote))
    check("matching panel_hash is not flagged",
          warn_panel_mismatch({k: v for k, v in results.items() if k != "Mix"}) is None)

    # Explicit metric list: honoured verbatim, in order, unknown keys dropped.
    cfg2 = write_config(tmp / "cfg2.yaml", f"""
metrics: [contrastive_wasserstein, recon_pearson_r, no_such_metric]
groups:
  - name: "A"
    dir: {abl_a.as_posix()}
    experiments:
      - {{name: "U", model: unified}}
""")
    c2 = load_config(cfg2)
    r2, _ = collect_from_config(c2)
    s2 = resolve_metrics(r2, c2)
    check("explicit metrics keep their order",
          [s.key for s in s2] == ["contrastive_wasserstein", "recon_pearson_r"],
          str([s.key for s in s2]))

    return results, groups


def check_config_errors(tmp: Path) -> None:
    print("\n[config errors]")
    abl = tmp / "ablation_a"

    def run_expecting_exit(body: str) -> str:
        path = write_config(tmp / "bad.yaml", body)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "evaluate" / "plot" /
                                 "plot_unified_metrics_table.py"),
             "--config", str(path), "--list"],
            capture_output=True, text=True,
        )
        return f"rc={proc.returncode} {proc.stdout} {proc.stderr}"

    out = run_expecting_exit(f"""
groups:
  - name: "A"
    dir: {abl.as_posix()}
    experiments:
      - {{name: "Same", model: unified}}
      - {{name: "Same", model: cdd}}
""")
    check("duplicate display names rejected",
          "rc=1" in out and "duplicate experiment name" in out, out[:300])

    out = run_expecting_exit(f"""
style: pie_chart
groups:
  - {{name: "A", dir: {abl.as_posix()}, experiments: [{{model: unified}}]}}
""")
    check("unknown style rejected", "rc=1" in out and "'style'" in out, out[:300])

    out = run_expecting_exit(f"""
scib: {{tags: [not_a_tag]}}
groups:
  - {{name: "A", dir: {abl.as_posix()}, experiments: [{{model: unified}}]}}
""")
    check("unknown scib tag rejected",
          "rc=1" in out and "unknown scib tag" in out, out[:300])

    out = run_expecting_exit(f"""
experiments: [{{model: unified, dir: {abl.as_posix()}}}]
groups:
  - {{name: "A", dir: {abl.as_posix()}, experiments: [{{model: cdd}}]}}
""")
    check("'groups' + 'experiments' together rejected",
          "rc=1" in out and "both 'groups' and 'experiments'" in out, out[:300])


def check_rendering(tmp: Path, results: dict, groups: list) -> None:
    print("\n[rendering]")
    cfg_path = tmp / "cfg.yaml"
    config = load_config(cfg_path)
    specs = resolve_metrics(results, config)
    footnote = warn_panel_mismatch(results)

    for style in ("heatmap", "rank_table", "bars"):
        for transpose in (False, True):
            if style == "bars" and transpose:
                continue
            config.style = style
            config.transpose = transpose
            out = tmp / f"fig_{style}{'_T' if transpose else ''}.png"
            plot = plot_bars if style == "bars" else plot_table
            try:
                plot(results, specs, groups, config, out,
                     show=False, figsize=None, footnote=footnote)
                ok = out.is_file() and out.stat().st_size > 5_000
                detail = f"{out.stat().st_size} bytes" if out.is_file() else "missing"
            except Exception as exc:  # noqa: BLE001 — the check is "does it render"
                ok, detail = False, repr(exc)
            label = f"{style}{' (transposed)' if transpose else ''} renders"
            check(label, ok, detail)

    # A single run is the degenerate case min-max cannot rank — it must still draw.
    one = {k: v for k, v in list(results.items())[:1]}
    one_groups = [(None, list(one))]
    config.style, config.transpose = "heatmap", False
    out = tmp / "fig_single.png"
    try:
        plot_table(one, specs, one_groups, config, out,
                   show=False, figsize=None, footnote=None)
        ok = out.is_file() and out.stat().st_size > 5_000
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"       {exc!r}")
    check("single-run table renders", ok)

    # A metric no run has must not crash the grid.
    config.style = "heatmap"
    out = tmp / "fig_missing.png"
    ghost = specs + [MetricSpec("never_computed", "Ghost", UP, "Other")]
    try:
        plot_table(results, ghost, groups, config, out,
                   show=False, figsize=None, footnote=None)
        ok = out.is_file()
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"       {exc!r}")
    check("all-missing column renders as n/a", ok)


def check_csv_output(tmp: Path, results: dict, groups: list) -> None:
    """The figure's numbers are written beside it as {stem}.csv."""
    import csv as csv_module

    from evaluate.plot.plot_unified_metrics_table import (
        csv_output_path,
        write_table_csv,
    )

    print("\n[csv]")
    config = load_config(tmp / "cfg.yaml")
    specs = resolve_metrics(results, config)

    out = tmp / "csv_test.png"
    check("path is the figure's, with a .csv suffix",
          csv_output_path(out) == tmp / "csv_test.csv")

    path = write_table_csv(out, results, specs, groups)
    check("csv written", path.is_file())

    rows = list(csv_module.DictReader(path.open(encoding="utf-8")))
    run_order = [name for _, names in groups for name in names]
    check("one row per run, in plot order",
          [r["run"] for r in rows] == run_order, str([r["run"] for r in rows]))
    check("group column filled",
          all(r["group"] for r in rows if any(g for g, _ in groups)),
          str(rows[0]))
    check("one column per plotted metric, keyed by metric key",
          set(rows[0]) == {"group", "run"} | {s.key for s in specs},
          str(sorted(set(rows[0]) - {"group", "run"})))

    # Values must be the raw numbers, not the colour scores.
    first = rows[0]
    for spec in specs:
        raw = results[run_order[0]].get(spec.key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            check(f"value round-trips ({spec.key})",
                  abs(float(first[spec.key]) - float(raw)) < 1e-12,
                  f"{first[spec.key]} vs {raw}")
            break

    # A metric no run has becomes an empty cell rather than a crash.
    ghost = specs + [MetricSpec("never_computed", "Ghost", UP, "Other")]
    rows = list(csv_module.DictReader(
        write_table_csv(tmp / "csv_ghost.png", results, ghost, groups)
        .open(encoding="utf-8")))
    check("a metric no run has is an empty cell",
          all(r["never_computed"] == "" for r in rows))


def check_cli(tmp: Path) -> None:
    print("\n[cli]")
    script = ROOT / "evaluate" / "plot" / "plot_unified_metrics_table.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(tmp / "cfg.yaml"), "--list"],
        capture_output=True, text=True,
    )
    check("--list exits 0", proc.returncode == 0, proc.stderr[-400:])
    check("--list groups by family", "Reconstruction" in proc.stdout,
          proc.stdout[:300])
    check("--list marks the selection", "*" in proc.stdout)

    out = tmp / "cli.png"
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(tmp / "cfg.yaml"),
         "--style", "rank_table", "--output", str(out), "--no-show"],
        capture_output=True, text=True,
    )
    check("--style/--output/--no-show writes a figure",
          proc.returncode == 0 and out.is_file(), proc.stderr[-400:])

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(tmp / "nope.yaml"), "--no-show"],
        capture_output=True, text=True,
    )
    check("missing config exits 1", proc.returncode == 1, proc.stderr[-200:])


def check_vars(tmp: Path) -> None:
    """``vars:`` interpolation, shared with run_analysis.py's config."""
    print("\n[vars interpolation]")
    abl = tmp / "ablation_a"

    cfg = write_config(tmp / "vars.yaml", f"""
vars:
  root: {abl.parent.as_posix()}
  abl: ${{root}}/ablation_a
title: "Runs under ${{root}}"
groups:
  - name: "A"
    dir: ${{abl}}
    experiments:
      - {{name: "U", model: unified}}
""")
    config = load_config(cfg)
    name, model_dir = config.groups[0][1][0]
    check("${var} expands in a group dir",
          model_dir == (abl / "unified").resolve(), str(model_dir))
    check("nested ${var} inside another var expands",
          "${" not in str(model_dir), str(model_dir))
    check("${var} expands in scalars too",
          "${" not in config.title, config.title)

    bad = write_config(tmp / "badvars.yaml", """
groups:
  - name: "A"
    dir: ${no_such_var}/x
    experiments: [{model: unified}]
""")
    try:
        load_config(bad)
        ok, detail = False, "no error raised"
    except SystemExit as exc:
        ok, detail = "no_such_var" in str(exc), str(exc)
    check("undefined ${var} is a clear error", ok, detail)


def check_example_config() -> None:
    """The shipped example config must stay loadable as the code evolves.

    It names every model explicitly (no ``all_models``), so parsing it touches no
    files and works on a machine that has never seen the cluster.
    """
    print("\n[shipped example config]")
    path = ROOT / "evaluate" / "plot" / "example_unified_metrics_config.yaml"
    check("example config exists", path.is_file(), str(path))
    if not path.is_file():
        return

    try:
        config = load_config(path)
    except SystemExit as exc:
        check("example config parses", False, str(exc))
        return
    check("example config parses", True)

    runs = [(n, d) for _, members in config.groups for n, d in members]
    names = [n for n, _ in runs]
    check("names are unique", len(names) == len(set(names)),
          str([n for n in names if names.count(n) > 1]))
    check("every group has runs", all(members for _, members in config.groups))
    check("all groups are named", all(g for g, _ in config.groups))
    check("no unexpanded ${vars} remain",
          not any("${" in str(d) for _, d in runs)
          and "${" not in str(config.output or ""),
          str(config.output))

    ablations = {d.parent.name for _, d in runs}
    check("covers the 9 real ablation dirs", len(ablations) == 9,
          f"{len(ablations)}: {sorted(ablations)}")
    check("every dir is an ablation_* directory",
          all(a.startswith("ablation_") for a in ablations), str(sorted(ablations)))
    check("selects every trained run", len(runs) == 39, str(len(runs)))
    check("style/normalize are valid",
          config.style in ("heatmap", "rank_table", "bars")
          and config.normalize in ("minmax", "zscore", "rank"))


def check_benchmark_untouched() -> None:
    """The step-1 refactor must not change plot_ablation_benchmark's public surface."""
    print("\n[regression: plot_ablation_benchmark]")
    try:
        from evaluate.plot import plot_ablation_benchmark as pab

        names = ["collect_metrics", "collect_model_metrics", "plot_benchmark",
                 "resolve_task", "load_config", "is_model_dir",
                 "METRIC_LABELS", "SKIP_METRICS", "TASK_PRIMARY_METRIC",
                 "TASK_LABELS", "LOWER_IS_BETTER"]
        missing = [n for n in names if not hasattr(pab, n)]
        check("public names still importable", not missing, str(missing))
        check("is_model_dir still defaults to results_*.json",
              pab.is_model_dir.__defaults__ is not None
              or "results_*.json" in str(pab.is_model_dir.__doc__))
    except Exception as exc:  # noqa: BLE001
        check("plot_ablation_benchmark imports", False, repr(exc))

    try:
        import evaluate.plot.replot_ablation_benchmark  # noqa: F401

        check("replot_ablation_benchmark imports", True)
    except Exception as exc:  # noqa: BLE001
        check("replot_ablation_benchmark imports", False, repr(exc))


def check_font_sizes(tmp: Path) -> None:
    """The font_scale / font_sizes mechanism, and the layout it has to keep valid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import evaluate.plot.plot_unified_metrics_table as table_mod
    from evaluate.plot.plot_unified_metrics_table import (
        FONT_ROLES,
        FontSizes,
        METRIC_BY_KEY,
        TableConfig,
        resolve_font_sizes,
    )

    print("\n[font sizes]")

    defaults = FontSizes()
    check("defaults are the sizes the figure has always used",
          (defaults.row_label, defaults.col_label, defaults.value,
           defaults.suptitle) == (9, 8, 7.5, 13.0))
    check("font_scale multiplies every role",
          all(getattr(resolve_font_sizes(2.0), r) == getattr(defaults, r) * 2
              for r in FONT_ROLES))
    pinned = resolve_font_sizes(2.0, {"value": 6})
    check("a font_sizes entry is absolute, not scaled",
          pinned.value == 6.0 and pinned.row_label == defaults.row_label * 2)

    for bad_scale, bad_over, why in (
        (0.0, None, "zero scale"),
        (-1.0, None, "negative scale"),
        (1.0, {"value": 0}, "zero size"),
        (1.0, {"bar_name": 9}, "a role from the benchmark, not this figure"),
    ):
        try:
            resolve_font_sizes(bad_scale, bad_over)
            check(f"rejects {why}", False)
        except (ValueError, KeyError):
            check(f"rejects {why}", True)

    # ── layout: the colour bar must clear the family brackets ────────────────
    # These hang below the axes, so the pad needed is largest on a SHORT table.
    # A 3-row figure overlapped at the default font size before the pad was made
    # size-aware, which is exactly what raising font_scale would have made worse.
    keys = ["recon_pearson_r", "recon_mae_bins", "contrastive_cross_cosine_mean",
            "contrastive_energy_distance", "geometry_pr_pooled"]
    specs = [METRIC_BY_KEY[k] for k in keys]
    captured: dict = {}
    original_finish = table_mod._finish

    def _capture(fig, output, show, dpi, tight=True):
        if tight:
            fig.tight_layout(rect=[0, 0.01, 1, 0.98])
        captured["fig"] = fig

    table_mod._finish = _capture
    try:
        worst = None
        for n_rows in (2, 3, 8, 20):
            for scale in (1.0, 1.5, 2.2):
                names = [f"run{i}" for i in range(n_rows)]
                data = {n: {k: 0.5 + 0.01 * i for k in keys}
                        for i, n in enumerate(names)}
                cfg = TableConfig(groups=[], font_scale=scale)
                table_mod.plot_table(
                    data, specs, [("demo", names)], cfg, None,
                    show=False, figsize=None,
                    fonts=resolve_font_sizes(scale),
                )
                fig = captured["fig"]
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                main_ax, cbar_ax = fig.axes[0], fig.axes[1]
                family_bottom = min(
                    t.get_window_extent(renderer).y0
                    for t in main_ax.texts if t.get_style() == "italic"
                )
                gap = family_bottom - cbar_ax.get_window_extent().y1
                if worst is None or gap < worst[0]:
                    worst = (gap, n_rows, scale)
                plt.close(fig)
        check("colour bar never overlaps the family brackets",
              worst is not None and worst[0] > 0,
              f"tightest gap {worst[0]:.1f}px at {worst[1]} rows, scale {worst[2]}")
    finally:
        table_mod._finish = original_finish


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_scoring()
        check_formatting()
        results, groups = check_collection(tmp)
        check_config_errors(tmp)
        check_vars(tmp)
        check_rendering(tmp, results, groups)
        check_csv_output(tmp, results, groups)
        check_cli(tmp)
        check_font_sizes(tmp)
        check_example_config()
        check_benchmark_untouched()

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for label in FAILED:
            print(f"  - {label}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
