"""Compare several models' reconstruction of one mask plan on the entries they all scored.

    python -m evaluate.reconstruction.compare --models unified_v1 bulkformer --plan bulk:leiden_s0:test:r25

Reads ``results/reconstruction/<model>/<plan>/entries.parquet`` per model, keeps
the masked (sample, gene) entries every model scored, and recomputes the space-free
metrics there (per-sample Pearson / Spearman, zero AUROC, global Spearman), which are
comparable across target spaces; for bins runs with tie ranges the correlations use the
tie midpoint as the target (``metrics.correlation_target``, column ``tie_midpoint_target``).
Native-space metrics are listed only for models that share a space. Writes ``results/reconstruction/_compare/<plan>/<name>/`` (``--name``, default ``default``):
``table.csv``, ``per_sample_spearman.png`` and ``manifest.json`` listing the models.

**Rows and heads.** A ``--models`` entry is a run: ``model`` (the gen head, or any
non-CancerFoundation reconstructor) or ``model@mvc`` (the run with ``--head mvc``, read
from ``<model>/<plan>_mvc/``), so one model's two decoders can be compared side by side.
The manifest lists plain model names under ``"models"`` (what ``rename_model`` rewrites)
and the row labels under ``"rows"``.

**Context gain.** Each model row also has ``spearman_gain``, ``spearman_nonzero_gain``
and ``auroc_zero_gain``: the row's metric minus that of its own gene-mean baseline on the
same entries (``metrics.context_gain``). Near 0 means the model ranks a sample's masked
genes no better than each gene's usual level does.
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
from evaluate.reconstruction.run_reconstruction import (  # noqa: E402
    HEADS,
    RESULTS_TASK,
    card_head,
    parse_row,
    plan_slug,
    row_label,
    run_dir,
    run_subdir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models' reconstruction of one mask plan.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Runs to compare: run_reconstruction output names, model@mvc for a model's MVC head")
    parser.add_argument("--plan", default=None, help="MODALITY:SPLIT:PART:NAME or a .npz path")
    parser.add_argument("--name", default=None,
                        help="Name of this comparison: results/reconstruction/_compare/<plan>/<name>/ "
                             "(default: default)")
    parser.add_argument("--baselines", action="store_true",
                        help="Also list each model's two baselines (computed in that model's space)")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def available_models(layout, slug: str) -> list[str]:
    """Runs with ``entries.parquet`` for the plan ``slug``: ``model`` and ``model@mvc`` rows, sorted."""
    root = layout.task_root(RESULTS_TASK)
    if not root.is_dir():
        return []
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        for head in HEADS:
            run = p / run_subdir(slug, head)
            if not (run / "entries.parquet").exists():
                continue
            # <slug>_mvc could also be another plan's slug: trust the head the run recorded.
            card = run / "metrics.json"
            if card.exists() and card_head(json.loads(card.read_text(encoding="utf-8"))) != head:
                continue
            rows.append(row_label(p.name, head))
    return rows


def base_models(rows) -> list[str]:
    """The plain model names of ``rows`` (``model@mvc`` -> ``model``), in order, once each."""
    return list(dict.fromkeys(parse_row(r)[0] for r in rows))


def load_runs(layout, models: list[str], slug: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """``(entries per row, metrics.json per row)`` for ``model`` / ``model@head`` rows.

    Exits when a run is missing or was scored with another head than its row names.
    """
    frames, cards = {}, {}
    for row in models:
        try:
            directory = run_dir(layout, row, slug)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        path = directory / "entries.parquet"
        if not path.exists():
            _, head = parse_row(row)
            sys.exit(f"Error: {path} not found; run run_reconstruction for '{parse_row(row)[0]}'"
                     f"{' --head ' + head if head != HEADS[0] else ''} on this plan first")
        card = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if card_head(card) != parse_row(row)[1]:
            sys.exit(f"Error: {directory} was scored with head '{card_head(card)}', not '{parse_row(row)[1]}'")
        frames[row] = pd.read_parquet(path)
        cards[row] = card
    return frames, cards


def shared_entries(frames: dict[str, pd.DataFrame]) -> tuple[set, dict[str, pd.DataFrame]]:
    """The (sample, gene) entries every model scored, and each frame restricted to them."""
    keys = None
    for frame in frames.values():
        k = set(zip(frame["sample"], frame["gene"]))
        keys = k if keys is None else keys & k
    keys = keys or set()
    subs = {m: f[pd.MultiIndex.from_frame(f[["sample", "gene"]]).isin(list(keys))].reset_index(drop=True)
            for m, f in frames.items()}
    return keys, subs


def main(argv: list[str] | None = None) -> pd.DataFrame:
    from experiments.layout import ExperimentLayout

    args = parse_with_config(build_parser(), "compare", argv)
    if not (args.models and args.plan):
        sys.exit("Error: give --models and --plan")
    layout = ExperimentLayout.from_root(args.root)
    plan = load_plan(resolve_plan_path(args.plan, args.root))
    slug = plan_slug(plan)

    frames, cards = load_runs(layout, args.models, slug)
    spaces = {m: card["space"] for m, card in cards.items()}
    keys, subs = shared_entries(frames)
    if not keys:
        sys.exit("Error: the models share no scored masked entry")
    print(f"{len(keys)} masked entries scored by all {len(frames)} models "
          f"(of {', '.join(f'{m}: {len(f)}' for m, f in frames.items())})")

    rows, per_sample = [], []
    # Native metrics only when every model's targets are on one scale: the same space,
    # n_bins and binning method (metrics.space_group).
    groups = {m: M.space_group(card) for m, card in cards.items()}
    shared_space = len(set(groups.values())) == 1
    if not shared_space and len(set(spaces.values())) == 1:
        print(f"  [note] native metrics left out: the models' targets are binned differently {groups}")
    for model, sub in subs.items():
        predictors = M.PREDICTORS if args.baselines else ("pred",)
        for col in predictors:
            sub_c = sub[sub[col].notna()]
            row = {"model": model if col == "pred" else f"{model}:{col}", "space": spaces[model],
                   **M.space_free(sub_c, col)}
            if col == "pred":
                row.update(M.context_gain(sub))
            if shared_space:
                row.update(M.native(sub_c, spaces[model], col))
            rows.append(row)
            ps = M.per_sample(sub_c, col)
            ps["model"] = row["model"]
            per_sample.append(ps)
    table = pd.DataFrame(rows)

    from experiments.layout import DEFAULT_COMPARISON

    try:
        out_dir = layout.task_compare_dir(RESULTS_TASK, slug, args.name or DEFAULT_COMPARISON)
    except ValueError as exc:
        sys.exit(f"Error: --name: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table.csv", index=False)
    _plot(pd.concat(per_sample), out_dir / "per_sample_spearman.png", plan.plan_id, args.dpi)
    (out_dir / "manifest.json").write_text(json.dumps({
        "analysis": "reconstruction", "plan": plan.plan_id, "models": base_models(frames),
        "rows": list(frames), "gain_reference": M.GAIN_REFERENCE,
        "spaces": spaces, "space_groups": groups, "n_shared_entries": len(keys),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 160, "display.max_columns", 12):
        shown = ["model", "space", "n_entries", "spearman_mean", "spearman_gain", "spearman_nonzero_mean",
                 "auroc_zero", "auroc_zero_gain"]
        print(table[[c for c in shown if c in table.columns]])
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
