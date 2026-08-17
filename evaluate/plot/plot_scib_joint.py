"""
One scIB benchmark table over runs from several ablation directories.

``unified_metrics.py --scib`` runs the benchmark once per ablation directory and
writes the result to

    {ablation_dir}/_scib_metrics/scib_{tag}.csv

which is exactly what ``Benchmarker.get_results(min_max_scale=False,
clean_names=True)`` returned. This script picks named rows out of several such
files, concatenates them into one table, and hands it to **scib_metrics' own
renderer** — ``Benchmarker.plot_results_table`` — so the figure is a real scIB
benchmark table (PRGn circles per metric, bars for the aggregate scores, columns
bracketed by metric type), not a lookalike drawn here.

Nothing is recomputed: the numbers are the ones already on disk.

How the renderer is reached
---------------------------
``plot_results_table`` is a method that calls ``self.get_results(...)`` and reads
``self._embedding_obsm_keys`` for the figure height, and nothing else about the
benchmark. :class:`_JointTable` subclasses ``Benchmarker`` and supplies exactly those
two things from the merged CSV, so no AnnData, embedding or neighbour graph is
needed and no benchmark is re-run. If a future scib_metrics needs more state than
that, the failure is caught and reported with the attribute it asked for.

Usage
-----
    # assemble and check the merged table — no scib_metrics needed
    python evaluate/plot/plot_scib_joint.py --config cfg.yaml --dry-run

    # what rows do the source CSVs actually contain?
    python evaluate/plot/plot_scib_joint.py --config cfg.yaml --list

    # the figure (needs the conda env that has scib_metrics + plottable)
    python evaluate/plot/plot_scib_joint.py --config cfg.yaml

Read this before comparing rows
-------------------------------
Every source CSV was produced from its own ``eval.h5ad`` — its own cells, its own
gene panel, its own subsample at ``--scib-n-max``. Putting the rows in one table
does not make them one benchmark: it is a side-by-side of several, and a difference
between two rows from different directories confounds the model with the dataset.
The script warns per source file so the composition is on the record.

Two consequences of using scIB's renderer rather than our own:

* **Row order is scIB's, not the config's.** ``plot_results_table`` sorts by the
  ``Total`` column, descending. The config order only controls which rows are read.
* **Values are plotted as saved** (``min_max_scale=False``), because that is how
  ``run_scib_benchmark`` computed them. scIB's min-max option rescales *inside*
  ``get_results`` before the aggregates are formed, so honouring it here would mean
  recomputing ``Batch correction`` / ``Bio conservation`` / ``Total`` by this
  script's own formula — inventing numbers the label would attribute to scIB. Use
  ``--min-max-scale`` only if you accept that; it is off by default.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from evaluate.plot.experiment_selection import (  # noqa: E402
    load_raw_config,
    parse_figsize,
    parse_groups,
)

# The row scib_metrics appends to say which family each column belongs to. Its name
# and its three values are part of the CSV contract, not our choice.
METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE = "Aggregate score"

# The four benchmarks run_scib_benchmark writes; the default is the bulk-vs-pseudobulk
# one, which is the only benchmark every dataset produces.
SCIB_TAGS: tuple[str, ...] = (
    "bulk_vs_pb",
    "paired_vs_nonpaired",
    "paired_bulk_vs_paired_pb",
    "nonpaired_bulk_vs_synth_pb",
)
DEFAULT_TAG = "bulk_vs_pb"

# Rows in the CSV are keyed by obsm key; the model directory name is what a config
# names, so the prefix is stripped on read and re-added nowhere.
OBSM_PREFIX = "X_cf_"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def scib_csv_path(ablation_dir: Path, tag: str) -> Path:
    return ablation_dir / "_scib_metrics" / f"scib_{tag}.csv"


def read_scib_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(values, metric_type)`` for one ``scib_{tag}.csv``.

    ``values`` is indexed by model directory name (the ``X_cf_`` prefix removed,
    ``X_pca`` left alone) with one column per metric; ``metric_type`` is the trailing
    row saying which family each column belongs to.
    """
    df = pd.read_csv(path, index_col=0)
    if METRIC_TYPE_ROW not in df.index:
        raise ValueError(
            f"{path} has no '{METRIC_TYPE_ROW}' row — it does not look like a table "
            "written by Benchmarker.get_results(). scIB's renderer needs that row to "
            "group the columns."
        )

    metric_type = df.loc[METRIC_TYPE_ROW].astype(str)
    values = df.drop(index=METRIC_TYPE_ROW)
    values = values.apply(pd.to_numeric, errors="coerce")
    values.index = [
        name[len(OBSM_PREFIX):] if str(name).startswith(OBSM_PREFIX) else str(name)
        for name in values.index
    ]
    return values, metric_type


def has_scib_row(model_dir: Path, tag: str) -> bool:
    """True if this model has a row in its ablation's scIB table (for all_models)."""
    path = scib_csv_path(model_dir.parent, tag)
    if not path.is_file():
        return False
    try:
        values, _ = read_scib_csv(path)
    except Exception:
        return False
    return model_dir.name in values.index


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class JointConfig:
    groups: list[tuple[str | None, list[tuple[str, Path]]]]
    tag: str = DEFAULT_TAG
    output: Path | None = None
    figsize: tuple[float, float] | None = None
    dpi: int = 300
    min_max_scale: bool = False
    metrics: list[str] = field(default_factory=list)   # [] -> every shared column


def load_config(path: Path) -> JointConfig:
    """
    Parse a YAML (or JSON) config.

    The ``groups:`` / ``experiments:`` half is the grammar shared with every other
    comparison config — see
    :func:`evaluate.plot.experiment_selection.parse_groups`. Each entry names a model
    directory; the scIB table is read from that directory's *parent*::

        vars:
          save: /cluster/work/boeva/rquiles/outputs/save_CFF

        tag:    bulk_vs_pb          # which of the four benchmarks to read
        output: figures/scib.svg    # optional; else <config>.svg
        dpi:    300
        min_max_scale: false        # see the module docstring before turning on

        metrics:                    # optional; omitted -> every shared column
          - iLISI
          - cLISI

        groups:
          - name: "Paired"          # groups are flattened: scIB's table is a flat
            dir: ${save}/ablation_paired_counts   # ranking sorted by Total
            experiments:
              - {name: "Unified (counts)", model: unified}
              - {name: "DAT (counts)",     model: dat}
          - name: "Baseline"
            experiments:
              - {name: "PCA", dir: "${save}/ablation_paired_counts", model: X_pca}
    """
    raw = load_raw_config(path)

    tag = str(raw.get("tag") or DEFAULT_TAG)
    if tag not in SCIB_TAGS:
        sys.exit(
            f"ERROR: unknown tag '{tag}' in {path}. Known tags: {', '.join(SCIB_TAGS)}."
        )

    groups = parse_groups(raw, path, lambda d: has_scib_row(d, tag))

    metrics = raw.get("metrics") or []
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]

    return JointConfig(
        groups=groups,
        tag=tag,
        output=Path(raw["output"]).expanduser() if raw.get("output") else None,
        figsize=parse_figsize(raw, path),
        dpi=int(raw.get("dpi") or 300),
        min_max_scale=bool(raw.get("min_max_scale", False)),
        metrics=[str(m) for m in metrics],
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble(config: JointConfig) -> tuple[pd.DataFrame, list[str]]:
    """Merge the requested rows into one ``get_results()``-shaped frame.

    Returns ``(table, display_names)`` where *table* carries the metric rows followed
    by the ``Metric Type`` row, exactly as scib_metrics produced it — that shape is
    what its renderer consumes.
    """
    rows: dict[str, pd.Series] = {}
    order: list[str] = []
    metric_types: dict[Path, pd.Series] = {}
    sources: dict[Path, list[str]] = {}
    cache: dict[Path, tuple[pd.DataFrame, pd.Series]] = {}

    for _, members in config.groups:
        for display_name, model_dir in members:
            ablation_dir = model_dir.parent
            csv_path = scib_csv_path(ablation_dir, config.tag)
            if csv_path not in cache:
                if not csv_path.is_file():
                    print(
                        f"[skip] {display_name}: no {csv_path}. Run the 'scib' step "
                        f"for {ablation_dir.name} first.",
                        file=sys.stderr,
                    )
                    continue
                try:
                    cache[csv_path] = read_scib_csv(csv_path)
                except Exception as exc:
                    print(f"[skip] {display_name}: {exc}", file=sys.stderr)
                    continue

            values, metric_type = cache[csv_path]
            if model_dir.name not in values.index:
                print(
                    f"[skip] {display_name}: '{model_dir.name}' is not a row in "
                    f"{csv_path.name}. Rows: {', '.join(map(str, values.index))}",
                    file=sys.stderr,
                )
                continue

            rows[display_name] = values.loc[model_dir.name]
            order.append(display_name)
            metric_types[csv_path] = metric_type
            sources.setdefault(csv_path, []).append(display_name)

    if not rows:
        sys.exit(
            "ERROR: no requested model has a row in any scIB table. Run the 'scib' "
            "step (evaluate/run_analysis.py --step scib) for these ablation "
            "directories first."
        )

    # Columns must agree, or the table would be plotting different metrics per row.
    common = set.intersection(*(set(s.index) for s in rows.values()))
    dropped = set.union(*(set(s.index) for s in rows.values())) - common
    if dropped:
        print(
            f"[warning] {len(dropped)} column(s) are not present in every selected "
            f"run and are dropped: {', '.join(sorted(map(str, dropped)))}",
            file=sys.stderr,
        )

    if config.metrics:
        missing = [m for m in config.metrics if m not in common]
        for name in missing:
            print(f"[warning] metric '{name}' is not in every run — ignoring.",
                  file=sys.stderr)
        chosen = [m for m in config.metrics if m in common]
        if not chosen:
            sys.exit("ERROR: none of the requested metrics is available.")
        # An aggregate column whose components were dropped would be misleading, but
        # the aggregates come from the source file and describe that file's own run,
        # so they are kept and simply reported below.
    else:
        chosen = None

    reference = next(iter(metric_types.values()))
    for path, mt in metric_types.items():
        shared = [c for c in common if c in mt.index and c in reference.index]
        disagree = [c for c in shared if mt[c] != reference[c]]
        if disagree:
            print(
                f"[warning] {path} disagrees on the metric type of "
                f"{', '.join(disagree)} — using the first file's.",
                file=sys.stderr,
            )

    # Preserve the source file's column order rather than a set's arbitrary one.
    columns = [c for c in reference.index if c in common]
    if chosen is not None:
        columns = [c for c in columns if c in chosen]

    table = pd.DataFrame(
        [rows[name].reindex(columns) for name in order],
        index=order,
        columns=columns,
    )
    metric_type_row = reference.reindex(columns)

    if config.min_max_scale:
        table = _min_max_scale(table, metric_type_row)

    # The renderer needs the Metric Type row back on the bottom, as get_results
    # returns it. Values stay object-typed there, which is what scib does too.
    out = pd.concat([table, metric_type_row.to_frame(METRIC_TYPE_ROW).T])

    print(f"\n{len(order)} run(s) from {len(sources)} scIB table(s):")
    for path, names in sources.items():
        print(f"  {path}")
        print(f"    {', '.join(names)}")
    if len(sources) > 1:
        print(
            "[warning] these rows come from different eval.h5ad files — different "
            "cells, and possibly different gene panels. The table shows several "
            "benchmarks side by side, not one benchmark.",
            file=sys.stderr,
        )

    return out, order


def _min_max_scale(table: pd.DataFrame, metric_type: pd.Series) -> pd.DataFrame:
    """Column-wise min-max over the pooled rows, aggregates recomputed after.

    Off by default and documented as this script's arithmetic, not scIB's: scIB
    scales inside ``get_results`` before forming the aggregates, so there is no way
    to honour the flag from a saved table without recomputing them here.
    """
    scaled = table.copy().astype(float)
    metric_cols = [c for c in table.columns if metric_type.get(c) != AGGREGATE_SCORE]
    for col in metric_cols:
        lo, hi = scaled[col].min(), scaled[col].max()
        scaled[col] = 0.5 if hi == lo else (scaled[col] - lo) / (hi - lo)

    by_family: dict[str, list[str]] = {}
    for col in metric_cols:
        by_family.setdefault(str(metric_type.get(col)), []).append(col)

    for agg in (c for c in table.columns if metric_type.get(c) == AGGREGATE_SCORE):
        if agg in by_family:                      # "Batch correction", "Bio conservation"
            scaled[agg] = scaled[by_family[agg]].mean(axis=1)
        elif agg == "Total":
            # scIB's published weighting.
            bio = scaled[by_family.get("Bio conservation", [])]
            batch = scaled[by_family.get("Batch correction", [])]
            if not bio.empty and not batch.empty:
                scaled[agg] = 0.6 * bio.mean(axis=1) + 0.4 * batch.mean(axis=1)
    return scaled


# --------------------------------------------------------------------------- #
# Rendering — scib_metrics' own table
# --------------------------------------------------------------------------- #

def render(
    table: pd.DataFrame,
    display_names: list[str],
    output: Path,
    dpi: int,
    figsize: tuple[float, float] | None = None,
    min_max_scale: bool = False,
):
    """Draw *table* with ``Benchmarker.plot_results_table`` and save it.

    The scaling has already been applied (or not) during assembly, so the renderer is
    always asked for ``min_max_scale=False`` — anything else would make it rescale a
    table it did not compute.
    """
    try:
        from scib_metrics.benchmark import Benchmarker
    except ImportError:
        sys.exit(
            "ERROR: scib_metrics is not importable here, and this script plots with "
            "ITS renderer rather than a local reimplementation.\n"
            "       Run in the conda env that has it (the same one "
            "'run_analysis.py --step scib' uses):\n"
            "           conda activate bulkFM\n"
            "       Use --dry-run to assemble and inspect the merged table without "
            "scib_metrics."
        )

    class _JointTable(Benchmarker):
        """A Benchmarker that already has its results.

        Deliberately does NOT call ``Benchmarker.__init__``: that wants an AnnData
        with embeddings, and there is nothing to compute — the numbers are read from
        disk. ``plot_results_table`` needs only the results frame and the number of
        embeddings, both supplied here.
        """

        def __init__(self, results: pd.DataFrame, keys: list[str]):
            self._results_table = results
            self._embedding_obsm_keys = list(keys)
            self._benchmarked = True

        def get_results(self, min_max_scale: bool = True, clean_names: bool = True):
            # The frame is already a get_results() output, read back from disk.
            return self._results_table

    bench = _JointTable(table, display_names)

    try:
        tab = bench.plot_results_table(min_max_scale=False, show=False)
    except AttributeError as exc:
        sys.exit(
            f"ERROR: this scib_metrics version's plot_results_table needs benchmark "
            f"state this script does not reconstruct ({exc}).\n"
            "       The merged table itself is fine — re-run with --dry-run to write "
            "it as CSV, and plot it with Benchmarker directly."
        )
    except Exception as exc:
        sys.exit(f"ERROR: scib_metrics failed to draw the table: {exc!r}")

    try:
        fig = tab.ax.get_figure()
        facecolor = tab.ax.get_facecolor()
    except AttributeError:
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        facecolor = fig.get_facecolor()

    if figsize is not None:
        fig.set_size_inches(*figsize)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=facecolor, dpi=dpi, bbox_inches="tight")
    print(f"Saved to {output}")
    return tab


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def print_inventory(config: JointConfig) -> None:
    """Every row and column the selected ablation directories offer."""
    seen: set[Path] = set()
    for _, members in config.groups:
        for display_name, model_dir in members:
            path = scib_csv_path(model_dir.parent, config.tag)
            if path in seen:
                continue
            seen.add(path)
            print(f"\n{path}")
            if not path.is_file():
                print("  (missing — run the 'scib' step for this ablation)")
                continue
            try:
                values, metric_type = read_scib_csv(path)
            except Exception as exc:
                print(f"  (unreadable: {exc})")
                continue
            print(f"  rows:    {', '.join(map(str, values.index))}")
            print("  columns:")
            for col in values.columns:
                print(f"    {str(col):<24} {metric_type.get(col, '?')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One scIB benchmark table over runs from several ablation directories, "
            "drawn by scib_metrics' own Benchmarker.plot_results_table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="YAML config selecting the runs. See load_config().")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Figure path (.svg, .png, .pdf). Defaults to the "
                             "config's 'output:', else the config path with .svg.")
    parser.add_argument("--tag", choices=SCIB_TAGS, default=None,
                        help="Override the config's 'tag:'.")
    parser.add_argument("--min-max-scale", action="store_true",
                        help="Rescale each metric across the selected runs and "
                             "recompute the aggregates. Read the module docstring: "
                             "that arithmetic is this script's, not scIB's.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"),
                        default=None, help="Override the figure size, in inches.")
    parser.add_argument("--dpi", type=int, default=None, help="Override the dpi.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assemble the merged table, write it as CSV beside the "
                             "output, and stop. Does not need scib_metrics.")
    parser.add_argument("--list", action="store_true",
                        help="List the rows and columns of every source table, "
                             "then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path: Path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"ERROR: --config does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    if args.tag:
        config.tag = args.tag
    if args.min_max_scale:
        config.min_max_scale = True
    if args.figsize:
        config.figsize = (args.figsize[0], args.figsize[1])
    if args.dpi:
        config.dpi = args.dpi

    if args.list:
        print_inventory(config)
        return

    print(f"Loading scIB tables from {config_path} (tag '{config.tag}') ...")
    table, display_names = assemble(config)

    output: Path = (
        args.output or config.output or config_path.with_suffix(".svg")
    ).expanduser()

    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path)
    print(f"Merged table -> {csv_path}")

    if args.dry_run:
        print("\n" + table.to_string())
        print("\n--dry-run: stopping before scib_metrics is imported.")
        return

    render(table, display_names, output, config.dpi, config.figsize,
           config.min_max_scale)


if __name__ == "__main__":
    main()
