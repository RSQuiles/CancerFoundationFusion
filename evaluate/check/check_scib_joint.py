"""Self-check for ``evaluate/plot/plot_scib_joint.py``.

    python evaluate/check/check_scib_joint.py

Builds synthetic ``_scib_metrics/scib_<tag>.csv`` files and checks that the merged
table is assembled correctly — no cluster, GPU or real benchmark.

``scib_metrics`` is not importable outside the conda env that has it, so the render
path is exercised against a **stub** ``Benchmarker`` whose ``plot_results_table``
does what the real one does with the object: call ``self.get_results(...)``, read
``self._embedding_obsm_keys``, drop the ``Metric Type`` row, sort by ``Total`` and
group each column by its metric type. That pins the contract this script relies on —
it does not prove the real renderer draws, which needs the conda env.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from evaluate.plot.plot_scib_joint import (  # noqa: E402
    METRIC_TYPE_ROW,
    assemble,
    has_scib_row,
    load_config,
    read_scib_csv,
    scib_csv_path,
)

FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {label}{('  - ' + detail) if not cond and detail else ''}")
    if not cond:
        FAILED.append(label)


# --------------------------------------------------------------------------- #
# Fixture — the real CSV shape: X_cf_ rows, an X_pca row, a Metric Type row
# --------------------------------------------------------------------------- #

COLUMNS = ["Isolated labels", "Silhouette label", "cLISI", "BRAS", "iLISI",
           "Batch correction", "Bio conservation", "Total"]
TYPES = ["Bio conservation", "Bio conservation", "Bio conservation",
         "Batch correction", "Batch correction",
         "Aggregate score", "Aggregate score", "Aggregate score"]


def write_scib_csv(ablation: Path, tag: str, models: dict[str, float],
                   columns: list[str] | None = None) -> Path:
    cols = columns or COLUMNS
    types = [TYPES[COLUMNS.index(c)] for c in cols]
    out = ablation / "_scib_metrics"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["," + ",".join(cols)]
    for name, base in models.items():
        idx = name if name == "X_pca" else f"X_cf_{name}"
        lines.append(idx + "," + ",".join(f"{base + 0.01 * i:.4f}"
                                          for i in range(len(cols))))
    lines.append("Metric Type," + ",".join(types))
    path = out / f"scib_{tag}.csv"
    path.write_text("\n".join(lines) + "\n")
    # The model dirs themselves must exist for the config to resolve them.
    for name in models:
        if name != "X_pca":
            (ablation / name).mkdir(parents=True, exist_ok=True)
    return path


def write_config(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# The stub renderer
# --------------------------------------------------------------------------- #

def install_stub_scib(record: dict) -> None:
    """Install a scib_metrics whose plot_results_table mimics the real one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class Benchmarker:
        def __init__(self, *a, **k):
            raise AssertionError(
                "plot_scib_joint must NOT call Benchmarker.__init__ — it would want "
                "an AnnData with embeddings and would re-run the benchmark."
            )

        def plot_results_table(self, min_max_scale=True, show=True, save_dir=None):
            # Exactly the state the real implementation touches.
            num_embeds = len(self._embedding_obsm_keys)
            df = self.get_results(min_max_scale=min_max_scale)
            plot_df = df.drop(METRIC_TYPE_ROW, axis=0)
            plot_df = plot_df.sort_values(by="Total", ascending=False).astype(float)
            groups = [df.loc[METRIC_TYPE_ROW, c] for c in df.columns]

            record["min_max_scale"] = min_max_scale
            record["num_embeds"] = num_embeds
            record["rows"] = list(plot_df.index)
            record["columns"] = list(df.columns)
            record["groups"] = groups
            record["values"] = plot_df.copy()

            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25,
                                            3 + 0.3 * num_embeds))
            ax.axis("off")
            table = types.SimpleNamespace(ax=ax)
            return table

    module = types.ModuleType("scib_metrics")
    benchmark = types.ModuleType("scib_metrics.benchmark")
    benchmark.Benchmarker = Benchmarker
    module.benchmark = benchmark
    sys.modules["scib_metrics"] = module
    sys.modules["scib_metrics.benchmark"] = benchmark


# --------------------------------------------------------------------------- #

def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    save = tmp / "save_CFF"
    a = save / "ablation_a"
    b = save / "ablation_b"
    write_scib_csv(a, "bulk_vs_pb",
                   {"unified": 0.50, "dat": 0.40, "X_pca": 0.20})
    write_scib_csv(b, "bulk_vs_pb", {"unified": 0.60, "mmd": 0.30})
    write_scib_csv(a, "paired_bulk_vs_paired_pb", {"unified": 0.55, "dat": 0.45})

    print("\n[reading]")
    values, metric_type = read_scib_csv(scib_csv_path(a, "bulk_vs_pb"))
    check("X_cf_ prefix stripped from the row index",
          list(values.index) == ["unified", "dat", "X_pca"], str(list(values.index)))
    check("Metric Type row is separated out, not left as data",
          METRIC_TYPE_ROW not in values.index)
    check("metric types are read per column",
          metric_type["iLISI"] == "Batch correction"
          and metric_type["Total"] == "Aggregate score")
    check("values are numeric", values.dtypes.apply(
        lambda d: d.kind == "f").all())
    check("has_scib_row finds a listed model", has_scib_row(a / "unified", "bulk_vs_pb"))
    check("has_scib_row rejects an unlisted one",
          not has_scib_row(a / "ghost", "bulk_vs_pb"))

    bad = tmp / "bad" / "_scib_metrics"
    bad.mkdir(parents=True)
    (bad / "scib_bulk_vs_pb.csv").write_text(",iLISI\nX_cf_x,0.5\n")
    try:
        read_scib_csv(bad / "scib_bulk_vs_pb.csv")
        check("a CSV without a Metric Type row is rejected", False)
    except ValueError as exc:
        check("a CSV without a Metric Type row is rejected",
              METRIC_TYPE_ROW in str(exc))

    print("\n[assembly]")
    cfg = write_config(tmp / "cfg.yaml", f"""
output: {(tmp / 'out.svg').as_posix()}
tag: bulk_vs_pb
groups:
  - name: A
    dir: {a.as_posix()}
    experiments:
      - {{name: "Unified A", model: unified}}
      - {{name: "DAT A", model: dat}}
  - name: B
    dir: {b.as_posix()}
    experiments:
      - {{name: "Unified B", model: unified}}
  - name: Ref
    experiments:
      - {{name: "PCA", dir: "{a.as_posix()}", model: X_pca}}
""")
    config = load_config(cfg)
    table, names = assemble(config)
    check("one row per selected run, in config order",
          names == ["Unified A", "DAT A", "Unified B", "PCA"], str(names))
    check("custom display names replace the model dir names",
          list(table.index) == names + [METRIC_TYPE_ROW], str(list(table.index)))
    check("the Metric Type row is the last row, as get_results returns it",
          table.index[-1] == METRIC_TYPE_ROW)
    check("rows keep their own file's numbers",
          abs(float(table.loc["Unified A", "Isolated labels"]) - 0.50) < 1e-9
          and abs(float(table.loc["Unified B", "Isolated labels"]) - 0.60) < 1e-9)
    check("the same model name in two dirs stays two distinct rows",
          float(table.loc["Unified A", "iLISI"])
          != float(table.loc["Unified B", "iLISI"]))
    check("X_pca is selectable like any other row",
          abs(float(table.loc["PCA", "Isolated labels"]) - 0.20) < 1e-9)
    check("column order follows the source file",
          list(table.columns) == COLUMNS, str(list(table.columns)))

    # A file with fewer columns must not silently mix metrics between rows.
    c = save / "ablation_c"
    write_scib_csv(c, "bulk_vs_pb", {"unified": 0.7},
                   columns=[col for col in COLUMNS if col != "BRAS"])
    cfg2 = write_config(tmp / "cfg2.yaml", f"""
output: {(tmp / 'out2.svg').as_posix()}
groups:
  - name: A
    dir: {a.as_posix()}
    experiments: [{{name: "Unified A", model: unified}}]
  - name: C
    dir: {c.as_posix()}
    experiments: [{{name: "Unified C", model: unified}}]
""")
    table2, _ = assemble(load_config(cfg2))
    check("a column missing from one run is dropped from all",
          "BRAS" not in table2.columns, str(list(table2.columns)))

    cfg3 = write_config(tmp / "cfg3.yaml", f"""
output: {(tmp / 'out3.svg').as_posix()}
metrics: [iLISI, Total]
groups:
  - name: A
    dir: {a.as_posix()}
    experiments: [{{name: "Unified A", model: unified}}]
""")
    table3, _ = assemble(load_config(cfg3))
    check("a metrics: list restricts the columns",
          list(table3.columns) == ["iLISI", "Total"], str(list(table3.columns)))

    print("\n[tag]")
    cfg4 = write_config(tmp / "cfg4.yaml", f"""
output: {(tmp / 'out4.svg').as_posix()}
tag: paired_bulk_vs_paired_pb
groups:
  - name: A
    dir: {a.as_posix()}
    experiments: [{{name: "Unified A", model: unified}}]
""")
    table4, _ = assemble(load_config(cfg4))
    check("the tag selects a different source file",
          abs(float(table4.loc["Unified A", "Isolated labels"]) - 0.55) < 1e-9)

    print("\n[min-max scaling]")
    cfg5 = write_config(tmp / "cfg5.yaml", f"""
output: {(tmp / 'out5.svg').as_posix()}
min_max_scale: true
groups:
  - name: A
    dir: {a.as_posix()}
    experiments:
      - {{name: "Unified A", model: unified}}
      - {{name: "DAT A", model: dat}}
      - {{name: "PCA", model: X_pca}}
""")
    table5, _ = assemble(load_config(cfg5))
    metrics_only = [c for c in table5.columns
                    if table5.loc[METRIC_TYPE_ROW, c] != "Aggregate score"]
    block = table5.loc[["Unified A", "DAT A", "PCA"], metrics_only].astype(float)
    check("scaled metrics span exactly [0, 1]",
          abs(block.min().min()) < 1e-9 and abs(block.max().max() - 1.0) < 1e-9)
    check("aggregates are recomputed from the scaled metrics, not copied",
          abs(float(table5.loc["Unified A", "Batch correction"])
              - float(block.loc["Unified A", ["BRAS", "iLISI"]].mean())) < 1e-9)

    print("\n[rendering contract]")
    record: dict = {}
    install_stub_scib(record)
    from evaluate.plot.plot_scib_joint import render   # after the stub is installed

    out = tmp / "fig.svg"
    render(table, names, out, dpi=150, figsize=None, min_max_scale=False)
    check("the figure is written", out.is_file())
    check("scIB's renderer is asked for the values as saved",
          record.get("min_max_scale") is False)
    check("it receives the Metric Type row it needs to group columns",
          record.get("groups") and record["groups"][-1] == "Aggregate score")
    check("num_embeds is the number of runs, sizing the figure",
          record.get("num_embeds") == len(names), str(record.get("num_embeds")))
    check("scIB sorts the rows by Total, overriding the config order",
          record.get("rows") == ["Unified B", "Unified A", "DAT A", "PCA"],
          str(record.get("rows")))
    check("every selected run reaches the renderer",
          set(record.get("rows", [])) == set(names))

    print("\n[cli]")
    script = PROJECT_ROOT / "evaluate" / "plot" / "plot_scib_joint.py"

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg),
                        "--list"], capture_output=True, text=True)
    check("--list exits 0", r.returncode == 0, r.stderr[-400:])
    check("--list names the source files and their rows",
          "scib_bulk_vs_pb.csv" in r.stdout and "unified" in r.stdout)

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg),
                        "--output", str(tmp / "dry.svg"), "--dry-run"],
                       capture_output=True, text=True)
    check("--dry-run exits 0 without scib_metrics", r.returncode == 0,
          r.stderr[-400:])
    check("--dry-run writes the merged CSV", (tmp / "dry.csv").is_file())
    if (tmp / "dry.csv").is_file():
        back = pd.read_csv(tmp / "dry.csv", index_col=0)
        check("the merged CSV round-trips to the same shape",
              list(back.index) == names + [METRIC_TYPE_ROW])
    check("--dry-run stops before importing scib_metrics",
          "stopping before scib_metrics" in r.stdout)
    check("cross-directory pooling is warned about",
          "several benchmarks" in r.stderr or "different eval.h5ad" in r.stderr,
          r.stderr[-300:])

    r = subprocess.run([sys.executable, str(script), "--config", str(cfg),
                        "--tag", "nonsense"], capture_output=True, text=True)
    check("an unknown --tag is rejected by argparse", r.returncode != 0)

    ghost = write_config(tmp / "ghost.yaml", f"""
output: {(tmp / 'ghost.svg').as_posix()}
groups:
  - name: G
    dir: {(save / 'ablation_missing').as_posix()}
    experiments: [{{name: "Nope", model: unified}}]
""")
    r = subprocess.run([sys.executable, str(script), "--config", str(ghost),
                        "--dry-run"], capture_output=True, text=True)
    check("a directory with no scIB table exits 1 with the fix",
          r.returncode == 1 and "scib" in r.stderr.lower(), r.stderr[-200:])

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for label in FAILED:
            print(f"  - {label}")
        return 1
    print("All scIB joint-table checks passed.")
    print("NOTE: the real scib_metrics renderer was stubbed. Run the script once in "
          "the bulkFM env to confirm it draws.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
