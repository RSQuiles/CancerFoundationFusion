"""Self-check of the ``recompute`` switch helpers (pandas only, no scib_metrics).

    python -m evaluate.check.check_recompute

- ``_recompute.skip_existing``: present -> skip, missing -> compute, recompute -> compute;
- ``scib_plot.merge_results``: stored rows + newly computed rows give the same raw and
  min-max tables as one joint computation, and ``check_aggregates`` catches aggregates that
  do not follow from the metrics.
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILURES.append(name)


def check_skip() -> None:
    print("\n-- skip_existing --")
    from evaluate._recompute import skip_existing

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "a.txt"
        f.write_text("x", encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            results = (skip_existing("x", [f], False), skip_existing("x", [f, Path(tmp) / "b"], False),
                       skip_existing("x", [f], True))
        check("present skipped, missing computed, recompute computes", results == (True, False, False))
        check("flags printed", "[skip]" in buf.getvalue() and "[compute]" in buf.getvalue())


def check_scib_merge() -> None:
    print("\n-- scIB merge (stored + new rows) --")
    from evaluate.integration.scib_plot import (METRIC_TYPE, check_aggregates, merge_results,
                                                results_table)

    rng = np.random.default_rng(0)
    cols = ["Isolated labels", "KMeans NMI", "Silhouette label", "iLISI", "BRAS", "PCR comparison"]
    types = pd.Series(["Bio conservation"] * 3 + ["Batch correction"] * 3, index=cols)
    values = pd.DataFrame(rng.random((6, 6)), index=[f"m{i}" for i in range(6)], columns=cols)
    joint_raw, joint_mm = results_table(values, types, False), results_table(values, types, True)

    v = values
    total = 0.4 * v[cols[3:]].mean(axis=1) + 0.6 * v[cols[:3]].mean(axis=1)
    check("Total = 0.4 batch + 0.6 bio",
          np.allclose(joint_raw.drop(index=METRIC_TYPE)["Total"].astype(float), total))
    mm = (v - v.min()) / (v.max() - v.min())
    check("min-max scaling per metric before the aggregates",
          np.allclose(joint_mm.drop(index=METRIC_TYPE)["Total"].astype(float),
                      0.4 * mm[cols[3:]].mean(axis=1) + 0.6 * mm[cols[:3]].mean(axis=1)))

    buf = io.StringIO()
    results_table(values.iloc[:4], types, False).to_csv(buf)   # round trip through a CSV
    buf.seek(0)
    stored = pd.read_csv(buf, index_col=0)
    check_aggregates(stored)
    new = results_table(values.iloc[4:], types, False)
    raw, scaled = merge_results(stored, new, list(values.index))
    check("merged raw table equals the joint one",
          np.allclose(raw.drop(index=METRIC_TYPE).astype(float).values,
                      joint_raw.drop(index=METRIC_TYPE).astype(float).values)
          and list(raw.columns) == list(joint_raw.columns)
          and list(raw.loc[METRIC_TYPE]) == list(joint_raw.loc[METRIC_TYPE]))
    check("merged min-max table equals the joint one",
          np.allclose(scaled.drop(index=METRIC_TYPE).astype(float).values,
                      joint_mm.drop(index=METRIC_TYPE).astype(float).values))
    raw_only, _ = merge_results(stored, None, ["m3", "m0"])
    check("only stored rows: subset and order follow the requested rows",
          list(raw_only.index[:-1]) == ["m3", "m0"])
    bio = results_table(values[cols[:3]], types[cols[:3]], False)
    check("bio only: no Total", "Total" not in bio.columns)

    bad = stored.copy()
    bad.loc["m0", "Total"] = 5
    try:
        check_aggregates(bad)
        check("inconsistent aggregates refused", False)
    except ValueError:
        check("inconsistent aggregates refused", True)
    other = results_table(values[cols[:5]], types[cols[:5]], False)
    try:
        merge_results(stored, other, list(values.index))
        check("other metrics refused", False)
    except ValueError:
        check("other metrics refused", True)


def main() -> int:
    check_skip()
    check_scib_merge()
    print(f"\n{'All checks passed.' if not FAILURES else f'{len(FAILURES)} check(s) FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
