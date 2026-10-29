"""Compare several models' reconstruction of one mask plan on the entries they all scored.

    python -m evaluate.reconstruction.compare --models unified_v1 bulkformer --plan bulk:leiden_s0:test:r25

Reads ``results/tasks/reconstruction/<model>/<plan>/entries.parquet`` per model, keeps
the masked (sample, gene) entries every model scored, and recomputes the space-free
metrics there (per-sample Pearson / Spearman, zero AUROC, global Spearman), which are
comparable across target spaces. Native-space metrics are listed only for models that
share a space. Writes ``results/assessment/reconstruction_<plan>[_<name>]/``:
``table.csv``, ``per_sample_spearman.png`` and ``manifest.json`` listing the models.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_with_config  # noqa: E402
from evaluate.reconstruction import metrics as M  # noqa: E402
from evaluate.reconstruction.mask_plan import load_plan, resolve_plan_path  # noqa: E402
from evaluate.reconstruction.run_reconstruction import RESULTS_TASK, plan_slug  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models' reconstruction of one mask plan.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", default=None, help="Output names of run_reconstruction")
    parser.add_argument("--plan", default=None, help="MODALITY:SPLIT:PART:NAME or a .npz path")
    parser.add_argument("--name", default=None, help="Suffix of the assessment directory")
    parser.add_argument("--baselines", action="store_true",
                        help="Also list each model's two baselines (computed in that model's space)")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    from experiments.layout import ExperimentLayout

    args = parse_with_config(build_parser(), "compare", argv)
    if not (args.models and args.plan):
        sys.exit("Error: give --models and --plan")
    layout = ExperimentLayout.from_root(args.root)
    plan = load_plan(resolve_plan_path(args.plan, args.root))
    slug = plan_slug(plan)

    frames, spaces = {}, {}
    for model in args.models:
        run_dir = layout.task_dir(RESULTS_TASK, model) / slug
        path = run_dir / "entries.parquet"
        if not path.exists():
            sys.exit(f"Error: {path} not found; run run_reconstruction for '{model}' on this plan first")
        frames[model] = pd.read_parquet(path)
        spaces[model] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["space"]

    keys = None
    for frame in frames.values():
        k = set(zip(frame["sample"], frame["gene"]))
        keys = k if keys is None else keys & k
    if not keys:
        sys.exit("Error: the models share no scored masked entry")
    print(f"{len(keys)} masked entries scored by all {len(frames)} models "
          f"(of {', '.join(f'{m}: {len(f)}' for m, f in frames.items())})")

    rows, per_sample = [], []
    shared_space = len(set(spaces.values())) == 1
    for model, frame in frames.items():
        sub = frame[[key in keys for key in zip(frame["sample"], frame["gene"])]]
        predictors = M.PREDICTORS if args.baselines else ("pred",)
        for col in predictors:
            sub_c = sub[sub[col].notna()]
            row = {"model": model if col == "pred" else f"{model}:{col}", "space": spaces[model],
                   **M.space_free(sub_c, col)}
            if shared_space:
                row.update(M.native(sub_c, spaces[model], col))
            rows.append(row)
            ps = M.per_sample(sub_c, col)
            ps["model"] = row["model"]
            per_sample.append(ps)
    table = pd.DataFrame(rows)

    out_dir = layout.assessment_dir(f"reconstruction_{slug}" + (f"_{args.name}" if args.name else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table.csv", index=False)
    _plot(pd.concat(per_sample), out_dir / "per_sample_spearman.png", plan.plan_id, args.dpi)
    (out_dir / "manifest.json").write_text(json.dumps({
        "analysis": "reconstruction", "plan": plan.plan_id, "models": list(frames),
        "spaces": spaces, "n_shared_entries": len(keys),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print(table[["model", "space", "n_entries", "spearman_mean", "spearman_nonzero_mean", "auroc_zero"]])
    print(f"wrote {out_dir}")
    return table


def _plot(per_sample: pd.DataFrame, path: Path, title: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(per_sample["model"]))
    data = [per_sample.loc[per_sample["model"] == m, "spearman"].dropna().to_numpy() for m in labels]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 4))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
    ax.set_ylabel("Per-sample Spearman (masked genes)")
    ax.set_title(f"Reconstruction: {title}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
