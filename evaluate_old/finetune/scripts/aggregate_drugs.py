"""
Aggregate drug sensitivity results from a directory of JSON files.

Reads all files matching results_drug_sensitivity*.json in the given directory,
aggregates them using aggregate_drug_sensitivity_results(), and writes the
output to results_drug_sensitivity_v2.json in the same directory.

Usage
-----
    python aggregate_drug_sensitivity.py --dir /path/to/metrics/dir
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate_drug_sensitivity_results(
    metrics: Iterable[dict[str, float | str]],
) -> dict[str, float]:
    """Aggregate per-drug/per-endpoint metrics separately."""
    df = pd.DataFrame(metrics)
    out: dict[str, float] = {}

    for endpoint, endpoint_df in df.groupby("endpoint"):
        numeric = endpoint_df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            if not col.startswith("n_"):
                out[f"{endpoint}_mean_{col}"] = float(numeric[col].mean())

    # Cross-endpoint summary aliases for plot_ablation_benchmark.py compatibility
    rho_vals = [v for k, v in out.items() if k.endswith("_mean_pearson_rho")]
    if rho_vals:
        out["mean_pearson_r"] = float(np.mean(rho_vals))
    auroc_vals = [v for k, v in out.items() if k.endswith("_mean_auroc")]
    if auroc_vals:
        out["mean_auroc"] = float(np.mean(auroc_vals))

    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sanitize_for_json(obj):
    """Replace NaN/Inf with None for valid JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_json_tolerant(path: Path) -> dict:
    """Load JSON file, tolerating NaN values written by Python."""
    with open(path) as f:
        content = f.read()
    # Replace bare NaN with null for valid JSON parsing
    content = content.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(content)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate drug sensitivity JSON results in a directory."
    )
    parser.add_argument(
        "--dir", "-d",
        required=True,
        type=Path,
        help="Directory containing results_drug_sensitivity*.json files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <dir>/results_drug_sensitivity_v2.json.",
    )
    args = parser.parse_args()

    metrics_dir = args.dir.expanduser().resolve()
    if not metrics_dir.is_dir():
        raise ValueError(f"Directory not found: {metrics_dir}")

    output_path = args.output or (metrics_dir / "results_drug_sensitivity_v2.json")

    # Find all matching JSON files
    json_files = sorted(metrics_dir.glob("results_drug_sensitivity*.json"))
    if not json_files:
        raise ValueError(f"No results_drug_sensitivity*.json files found in {metrics_dir}")

    print(f"Found {len(json_files)} file(s):")
    for jf in json_files:
        print(f"  {jf.name}")

    # Load and collect all per-drug metrics
    all_metrics: list[dict] = []
    for jf in json_files:
        data = load_json_tolerant(jf)
        all_metrics.append(data)

    if not all_metrics:
        raise ValueError("No per-drug metrics found in the loaded files.")

    print(f"\nLoaded {len(all_metrics)} per-drug metric entries.")

    # Aggregate
    aggregated = aggregate_drug_sensitivity_results(all_metrics)
    # aggregated = sanitize_for_json(aggregated)

    # Write output
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated metrics saved to: {output_path}")
    print("\n--- Aggregated metrics ---")
    for k, v in aggregated.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()