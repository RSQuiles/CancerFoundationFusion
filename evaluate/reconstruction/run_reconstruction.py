"""Score one model's reconstruction of a mask plan's masked entries.

    # CancerFoundation model from the layout (newest checkpoint in checkpoints/<model>/)
    python -m evaluate.reconstruction.run_reconstruction --model unified_v1 --plan bulk:leiden_s0:test:r25
    # any registered reconstructor (see reconstructors.py), named in the outputs by --name
    python -m evaluate.reconstruction.run_reconstruction --reconstructor bulkformer --name bulkformer \\
        --plan bulk:leiden_s0:test:r25 --option repo=/path/to/bulkformer-latest --option gene_column=gene_id

``--plan`` is ``MODALITY:SPLIT:PART:NAME`` (``data/<mod>/splits/<split>/masks/``) or a
path. The rows are read from the h5ad the plan was built on (``--input`` overrides).
Writes ``results/tasks/reconstruction/<name>/<plan>/``:

- ``metrics.json``: model and baselines (native-space and space-free metrics), by group,
  with the plan, the target space and how many masked entries were scored or dropped;
- ``entries.parquet``: one row per scored masked entry (target, prediction, baselines),
  so any further metric needs no rerun;
- ``per_sample.csv`` and ``mae_by_target.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_mapping, parse_with_config  # noqa: E402
from evaluate.reconstruction import metrics as M  # noqa: E402
from evaluate.reconstruction.mask_plan import load_plan, read_rows, resolve_plan_path  # noqa: E402
from evaluate.reconstruction.reconstructors import build_reconstructor  # noqa: E402

RESULTS_TASK = "reconstruction"


def plan_slug(plan) -> str:
    return plan.plan_id.replace(":", "_").replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a model's reconstruction of a mask plan.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--plan", required=False, default=None, help="MODALITY:SPLIT:PART:NAME or a .npz path")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--model", default=None, help="CancerFoundation model name (checkpoints/<model>/)")
    backend.add_argument("--reconstructor", default=None, help="Registered reconstructor (e.g. bulkformer)")
    parser.add_argument("--name", default=None, help="Output name with --reconstructor (default: its name)")
    parser.add_argument("--ckpt", default=None, help="Checkpoint in checkpoints/<model>/ or a path (default: newest)")
    parser.add_argument("--option", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Constructor arguments for the reconstructor (values parsed as YAML)")
    parser.add_argument("--head", default="gen", choices=["gen", "mvc"], help="CancerFoundation decoder head")
    parser.add_argument("--context", default="full", choices=["full", "plan"],
                        help="bulkformer: all other genes visible (full) or only the plan's context genes")
    parser.add_argument("--condition", nargs="*", default=None, metavar="COL=VALUE",
                        help="CancerFoundation: force a condition value for every sample")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="CancerFoundation: training store mapping.json (default: <data_path>/mapping.json)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--group-by", nargs="*", default=["tissue_general"],
                        help="obs columns to break metrics down by (modality always)")
    parser.add_argument("--input", type=Path, default=None, help="h5ad holding the plan's rows (default: the plan's source)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results/tasks/reconstruction/<name>/<plan>/)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def resolve_source(plan, override: Path | None) -> Path:
    source = override or Path(plan.meta["source"]["path"])
    if not source.exists():
        sys.exit(f"Error: {source} not found; pass --input")
    expected = plan.meta.get("source", {}).get("bytes")
    if override is None and expected is not None and source.stat().st_size != expected:
        print(f"  [warn] {source.name} changed size since the plan was built; its rows may differ")
    return source


def make_reconstructor(args, layout):
    from embed.embed_any import parse_options

    options = parse_options(args.option)
    if args.model:
        from experiments.layout import validate_model_name

        name = validate_model_name(args.model)
        ckpt = Path(args.ckpt) if args.ckpt else layout.latest_checkpoint(name)
        if ckpt is not None and not ckpt.exists() and args.ckpt:
            ckpt = layout.checkpoint_dir(name) / args.ckpt
        if ckpt is None or not ckpt.exists():
            sys.exit(f"Error: no checkpoint for '{name}' ({layout.checkpoint_dir(name)})")
        options.update(checkpoint=str(ckpt), head=args.head, batch_size=args.batch_size, cpu=args.cpu,
                       seed=args.seed, mapping=str(args.mapping) if args.mapping else None,
                       condition=parse_mapping(args.condition, "--condition"))
        return name, build_reconstructor("cancerfoundation", options)
    if not args.reconstructor:
        sys.exit("Error: give --model or --reconstructor")
    if args.reconstructor == "bulkformer":
        options.setdefault("context", args.context)
    name = validate_name(args.name or args.reconstructor)
    return name, build_reconstructor(args.reconstructor, options)


def validate_name(name: str) -> str:
    from experiments.layout import validate_model_name

    return validate_model_name(name)


def _clean(value):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(args, adata, plan, name: str, reconstructor, out_dir: Path) -> dict:
    result = reconstructor.predict(adata, plan)
    group_cols = list(dict.fromkeys(["modality", *(args.group_by or [])]))
    frame = M.entries_frame(result, plan, adata.obs, group_cols)
    space = reconstructor.space
    summary = M.evaluate(frame, space)
    by_group = {}
    for col in group_cols:
        if col in frame.columns:
            by_group[col] = {str(value): {**M.native(g, space), **M.space_free(g)}
                             for value, g in frame.groupby(col) if len(g)}
    n_masked = int(plan.masked_ptr[-1])
    report = {
        "model": name, "plan": plan.plan_id, "space": space,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_settings": {k: plan.meta.get(k) for k in ("seq_len", "mask_ratio", "zero_fraction", "seed", "n_samples")},
        "n_masked": n_masked, "n_scored": int(len(frame)), "n_dropped": int(result.get("n_dropped", 0)),
        "provenance": reconstructor.provenance(),
        "metrics": summary, "by_group": by_group,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    frame.to_parquet(out_dir / "entries.parquet", index=False)
    per_sample = M.per_sample(frame)
    obs_cols = [c for c in group_cols if c in adata.obs.columns]
    if obs_cols:
        per_sample = per_sample.merge(adata.obs[obs_cols].astype(str), left_on="sample", right_index=True, how="left")
    per_sample.to_csv(out_dir / "per_sample.csv", index=False)
    M.mae_by_target(frame, space).to_csv(out_dir / "mae_by_target.csv", index=False)

    print(f"\n{name} on {plan.plan_id} ({space}): {len(frame)} of {n_masked} masked entries scored")
    keys = ["mae", "acc_within1", "spearman_mean", "spearman_nonzero_mean", "auroc_zero"]
    print(f"  {'predictor':12s}" + "".join(f"{k:>22s}" for k in keys))
    for predictor, values in summary.items():
        print(f"  {predictor:12s}" + "".join(
            f"{values.get(k, float('nan')):>22.4f}" if values.get(k) is not None else f"{'-':>22s}" for k in keys))
    print(f"  wrote {out_dir}")
    return report


def main(argv: list[str] | None = None) -> dict:
    from experiments.layout import ExperimentLayout

    args = parse_with_config(build_parser(), "reconstruction", argv)
    if not args.plan:
        sys.exit("Error: give --plan")
    layout = ExperimentLayout.from_root(args.root)
    plan = load_plan(resolve_plan_path(args.plan, args.root))
    name, reconstructor = make_reconstructor(args, layout)
    source = resolve_source(plan, args.input)
    print(f"Plan {plan.plan_id}: {plan.n_samples} samples, {len(plan.genes)} genes; rows from {source.name}")
    adata = read_rows(source, plan.obs_names)
    out_dir = args.out_dir or layout.task_dir(RESULTS_TASK, name) / plan_slug(plan)
    return run(args, adata, plan, name, reconstructor, out_dir)


if __name__ == "__main__":
    main()
