"""Transfer SIS: does multimodal training / alignment make the bulk embedding more useful?

    python -m evaluate.geneicl.transfer --name leiden_s1_alignment \\
        --baselines alignment=align_ctrl --models align_mmd align_cdd ... [--null-control align_ctrl]

The Synergistic Information Score of Richter et al. (Nat. Mach. Intell. 2026, "Multimodal
cell foundation models require synergistic integration beyond alignment") is the relative
linear-probe gain of a fused representation over the best unimodal one. The GeneICL tasks
see bulk only, so nothing is fused at test time; this script compares **models** instead:

    tSIS_t(m | B) = (P_t(m) - max_{b in B} P_t(b)) / max_{b in B} P_t(b)

per task ``t``, with ``P`` the GeneICL probe score (linear: PCA-95 % + elastic net, the same
5 CV folds for every model) and ``B`` a baseline set naming the question (``align_ctrl``:
does the alignment help; ``bulk_baseline``: does multimodal pretraining help; ``pca``: does
the foundation model help). Only the ``probe_*.csv`` files on disk are read.

Per model, baseline set and metric (default ``f1_macro`` and ``r2_pooled_oof``, the paper's
F1 macro and R2):

- **suite tSIS** = ``(mean_t P_m - mean_t B_t) / mean_t B_t`` over the tasks whose baseline
  score reaches ``--min-baseline`` (a ratio of means: small denominators of single tasks
  do not dominate); ``median_tsis`` per task, ``win_rate``, ``mean_delta`` (all tasks);
- **95 % CI** from a cluster bootstrap over **datasets** (tasks of one dataset share
  samples), model and baseline resampled together; a sign-flip permutation **p** on the
  per-dataset mean deltas, Benjamini-Hochberg **q** across the models of a set and metric;
- **noise floor**: the same tSIS for a pair that should not differ. By default a
  seed replicate of ``--null-control`` (``<control>_s1``) if its probe results exist, else
  ``<control>@<--null-cv-tag>`` (the control re-probed with another CV seed through
  ``run_probe --run-tag``; probe noise only, a lower bound), else none. ``--null-pairs
  A=B`` sets it explicitly. A model ``exceeds_null`` when ``|suite tSIS|`` is larger than
  the null pair's;
- **verdict**: ``gain`` (CI above 0 and exceeds the null), ``loss`` (CI below 0 and
  exceeds the null), else ``no gain``. When the floor is not a seed pair, the verdict
  says the training-seed variance is not measured.

Writes ``results/geneicl_<suite>/_compare/transfer/<name>/``: ``per_task.csv``,
``summary.csv``, ``null.csv``, ``transfer_sis.{png,pdf}`` (forest plot),
``transfer_sis_table.{png,pdf}`` and ``manifest.json`` (``"models"``: every model read).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_mapping, parse_with_config  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402
from evaluate.geneicl.compare import METRICS, available_models, load_runs, per_task_table  # noqa: E402
from evaluate.geneicl.run_probe import OUT_STEM, TAG_SEP, probe_dir, split_row, task_name  # noqa: E402

DEFAULT_METRICS = ["f1_macro", "r2_pooled_oof"]
METRIC_TYPE = {m: tt for tt, ms in METRICS.items() for m in ms} | {"r2_pooled_oof": "regression"}
# A relative gain over a near-zero baseline is meaningless: such tasks get no tSIS
# (they still enter mean_delta). Metrics not listed have no minimum.
DEFAULT_MIN_BASELINE = {"f1_macro": 0.2, "r2_pooled_oof": 0.05, "r2": 0.05}
NULL_SEED_SUFFIX = "_s1"
NOTE_SEED = "seed noise floor"
NOTE_PROBE = "probe noise only (CV seed); training-seed variance not measured"
NOTE_NONE = "no noise floor; training-seed variance not measured"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transfer SIS on a GeneICL suite: relative probe gain of "
                                            "models over a baseline set, with a cluster-bootstrap CI.")
    p.add_argument("--root", type=Path, default=None, help="Experiments root")
    p.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
    p.add_argument("--name", default=None,
                   help="Analysis name: results/geneicl_<suite>/_compare/transfer/<name>/ (default: default)")
    p.add_argument("--baselines", nargs="+", default=None, metavar="SET=B1,B2",
                   help="Baseline sets (a mapping in YAML), e.g. alignment=align_ctrl multimodal=bulk_baseline; "
                        "per task the best baseline of the set is the reference")
    p.add_argument("--models", nargs="*", default=None, metavar="M | SET=M1,M2",
                   help="Models compared: one list for every set, or SET=M1,M2 per set (a mapping in YAML). "
                        "Default: every model with probe results, minus the set's baselines")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help=f"Probe metrics (default: {' '.join(DEFAULT_METRICS)}; any of "
                        f"{sorted(METRIC_TYPE)})")
    p.add_argument("--min-baseline", nargs="*", default=None, metavar="METRIC=V",
                   help=f"Minimum baseline score for a task's tSIS (default: {DEFAULT_MIN_BASELINE})")
    p.add_argument("--null-control", default=None,
                   help="Control whose replicate gives the noise floor (default: the single baseline of "
                        "the first set)")
    p.add_argument("--null-cv-tag", default="cv43",
                   help="run_probe --run-tag of the control re-probed with another CV seed, the fallback "
                        "noise floor when no <control>_s1 exists")
    p.add_argument("--null-pairs", nargs="*", default=None, metavar="A=B",
                   help="Explicit noise-floor pair(s) instead of the automatic one (e.g. align_ctrl_s1=align_ctrl)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-labels", nargs="*", default=None, metavar="MODEL=LABEL")
    p.add_argument("--highlight-top", type=int, default=2)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--dpi", type=int, default=200)
    from evaluate._recompute import add_recompute_arg

    add_recompute_arg(p, comparison=True)
    return p


# ---------------------------------------------------------------------------- arguments

def parse_sets(baselines, models, available: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """``{set: (models, baselines)}``."""
    base = {k: [x for x in v.split(",") if x] for k, v in parse_mapping(baselines, "--baselines").items()}
    if not base:
        raise ValueError("give --baselines SET=B1,B2")
    if models and all(isinstance(m, str) and "=" not in m for m in models) and not isinstance(models, dict):
        per_set = {k: list(models) for k in base}
    else:
        per_set = {k: [x for x in v.split(",") if x] for k, v in parse_mapping(models, "--models").items()}
        unknown = sorted(set(per_set) - set(base))
        if unknown:
            raise ValueError(f"--models names sets without baselines: {unknown}")
    out = {}
    for k, bs in base.items():
        ms = per_set.get(k) or [m for m in available if TAG_SEP not in m]
        out[k] = ([m for m in ms if m not in bs], bs)
    return out


def min_baselines(value) -> dict[str, float]:
    out = dict(DEFAULT_MIN_BASELINE)
    out.update({k: float(v) for k, v in parse_mapping(value, "--min-baseline").items()})
    return out


def has_probe(layout, suite: str, row: str) -> bool:
    prefix = probe_dir(layout, suite, *split_row(row)) / OUT_STEM
    return any((prefix.parent / f"{prefix.name}_{tt}.csv").exists() for tt in METRICS)


def resolve_null(layout, suite: str, args, sets) -> tuple[list[tuple[str, str]], str]:
    """The noise-floor pairs ``[(replicate, original)]`` and their kind (seed / probe / none)."""
    if args.null_pairs:
        pairs = [tuple(x.strip() for x in spec.split("=", 1)) for spec in args.null_pairs]
        bad = [p for p in pairs if len(p) != 2]
        if bad:
            raise ValueError(f"--null-pairs must look like A=B, got {bad}")
        kind = "probe" if all(TAG_SEP in a for a, _ in pairs) else "seed"
        return pairs, kind
    control = args.null_control
    if control is None:
        first = next(iter(sets.values()))[1]
        control = first[0] if len(first) == 1 else None
    if control is None:
        return [], "none"
    for candidate, kind in ((control + NULL_SEED_SUFFIX, "seed"), (f"{control}{TAG_SEP}{args.null_cv_tag}", "probe")):
        if has_probe(layout, suite, candidate):
            return [(candidate, control)], kind
        print(f"  [null] {candidate}: no probe results")
    return [], "none"


# ---------------------------------------------------------------------------- statistics

def bh_q(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def score_pair(pm: pd.Series, bt: pd.Series, dataset: pd.Series, min_base: float,
               n_boot: int, n_perm: int, rng: np.random.Generator) -> dict:
    """Suite statistics of one model's per-task scores ``pm`` against baseline scores ``bt``."""
    ok = pm.notna() & bt.notna()
    pm, bt, ds = pm[ok], bt[ok], dataset[ok]
    delta = pm - bt
    elig = bt >= min_base
    tsis = (delta / bt).where(elig)
    res = {"n_tasks": int(len(pm)), "n_low_baseline": int((~elig).sum()),
           "mean_delta": float(delta.mean()) if len(delta) else np.nan,
           "win_rate": float((delta > 0).mean()) if len(delta) else np.nan,
           "median_tsis": float(tsis.median()) if tsis.notna().any() else np.nan}
    if not elig.any():
        return res | {"suite_tsis": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    frame = pd.DataFrame({"ds": ds[elig], "p": pm[elig], "b": bt[elig]})
    per_ds = frame.groupby("ds")[["p", "b"]].sum()
    sp, sb = per_ds["p"].to_numpy(), per_ds["b"].to_numpy()
    res["suite_tsis"] = float((sp.sum() - sb.sum()) / sb.sum())
    # cluster bootstrap over datasets: multinomial weights per dataset
    w = rng.multinomial(len(sp), np.full(len(sp), 1.0 / len(sp)), size=n_boot)
    boot = (w @ sp - w @ sb) / (w @ sb)
    res["ci_lo"], res["ci_hi"] = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    # sign-flip permutation on the per-dataset mean deltas (all tasks, absolute scale)
    d = (pd.DataFrame({"ds": ds, "d": delta}).groupby("ds")["d"].mean()).to_numpy()
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    res["p"] = float(((np.abs((flips * d).mean(axis=1)) >= obs - 1e-15).sum() + 1) / (n_perm + 1))
    return res


def verdict(row: dict, null_abs: float | None) -> str:
    lo, hi, s = row.get("ci_lo"), row.get("ci_hi"), row.get("suite_tsis")
    if not np.isfinite(s if s is not None else np.nan):
        return "n/a"
    beyond = null_abs is None or not np.isfinite(null_abs) or abs(s) > null_abs
    if lo > 0 and beyond:
        return "gain"
    if hi < 0 and beyond:
        return "loss"
    return "no gain"


def metric_frame(table: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.Series]:
    """``(task x row scores, task -> dataset)`` of one metric."""
    sub = table[table["task_type"] == METRIC_TYPE[metric]]
    if metric not in sub.columns or sub.empty:
        return pd.DataFrame(), pd.Series(dtype=object)
    scores = sub.pivot_table(index="task_id", columns="row", values=metric, aggfunc="first")
    dataset = sub.drop_duplicates("task_id").set_index("task_id")["dataset"].reindex(scores.index)
    return scores, dataset


def compute(table: pd.DataFrame, sets: dict, metrics: list[str], min_base: dict[str, float],
            null_pairs: list[tuple[str, str]], n_boot: int, n_perm: int, seed: int):
    """``(per_task, summary, null)`` frames."""
    rng = np.random.default_rng(seed)
    per_task, rows, nulls = [], [], []
    for metric in metrics:
        scores, dataset = metric_frame(table, metric)
        if scores.empty:
            print(f"  [warn] no {metric} results; skipped")
            continue
        mb = float(min_base.get(metric, -np.inf))
        null_abs = None
        for a, b in null_pairs:
            if a in scores and b in scores:
                st = score_pair(scores[a], scores[b], dataset, mb, n_boot, n_perm, rng)
                nulls.append({"metric": metric, "replicate": a, "original": b, **st})
                null_abs = max(null_abs or 0.0, abs(st["suite_tsis"])) if np.isfinite(st["suite_tsis"]) else null_abs
        for name, (models, baselines) in sets.items():
            present = [b for b in baselines if b in scores]
            if not present:
                print(f"  [warn] {name}/{metric}: no baseline results ({baselines}); skipped")
                continue
            bt = scores[present].max(axis=1)
            used = scores[present].idxmax(axis=1)
            block = []
            for m in models:
                if m not in scores:
                    continue
                st = score_pair(scores[m], bt, dataset, mb, n_boot, n_perm, rng)
                block.append({"set": name, "baselines": ",".join(present), "metric": metric,
                              "task_type": METRIC_TYPE[metric], "model": m, **st})
                d = scores[m] - bt
                per_task.append(pd.DataFrame({
                    "set": name, "model": m, "metric": metric, "task_type": METRIC_TYPE[metric],
                    "task_id": scores.index, "dataset": dataset.to_numpy(), "score": scores[m].to_numpy(),
                    "baseline_score": bt.to_numpy(), "baseline_used": used.to_numpy(), "delta": d.to_numpy(),
                    "tsis": (d / bt).where(bt >= mb).to_numpy()}))
            q = bh_q(np.array([r["p"] for r in block]))
            for r, qq in zip(block, q):
                r["q"] = float(qq)
                r["null_abs_tsis"] = null_abs
                r["exceeds_null"] = (None if null_abs is None
                                     else bool(np.isfinite(r["suite_tsis"]) and abs(r["suite_tsis"]) > null_abs))
                r["verdict"] = verdict(r, null_abs)
            rows += block
    per_task = pd.concat(per_task, ignore_index=True) if per_task else pd.DataFrame()
    return per_task, pd.DataFrame(rows), pd.DataFrame(nulls)


# ---------------------------------------------------------------------------- figures

def forest(summary: pd.DataFrame, labels: dict[str, str], note: str, title: str):
    from evaluate._plot_style import HAIRLINE, INK, INK_2, MUTED, SEQ_RAMP, SERIES, pyplot

    plt = pyplot()
    sets = list(dict.fromkeys(summary["set"]))
    metrics = list(dict.fromkeys(summary["metric"]))
    n_models = summary.groupby(["set", "metric"]).size().max()
    fig, axes = plt.subplots(len(sets), len(metrics), squeeze=False,
                             figsize=(4.2 * len(metrics) + 1.8, 0.32 * n_models * len(sets) + 1.4 * len(sets)))
    colors = {"gain": SERIES[2], "loss": SERIES[7], "no gain": INK_2, "n/a": MUTED}
    for i, s in enumerate(sets):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            sub = summary[(summary["set"] == s) & (summary["metric"] == m)].reset_index(drop=True)
            if sub.empty:
                ax.axis("off")
                continue
            y = np.arange(len(sub))[::-1]
            null_abs = sub["null_abs_tsis"].iloc[0]
            if null_abs is not None and np.isfinite(null_abs):
                ax.axvspan(-null_abs, null_abs, color=SEQ_RAMP[0], zorder=0, label="noise floor")
            ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
            for yy, (_, r) in zip(y, sub.iterrows()):
                c = colors.get(r["verdict"], MUTED)
                ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, linewidth=1.6, zorder=2)
                ax.plot(r["suite_tsis"], yy, "o", color=c, markersize=5, zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels([labels.get(mm, mm) for mm in sub["model"]], fontsize=8)
            ax.grid(axis="x", color=HAIRLINE, linewidth=0.6)
            ax.set_title(f"{s} (vs {sub['baselines'].iloc[0]}): {m}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("transfer SIS (relative gain)", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, 0.005, "Point: suite tSIS; bar: 95% cluster-bootstrap CI over datasets; green gain, red loss "
             f"(CI excludes 0 and beyond the noise floor). Shaded: noise floor ({note}).",
             fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def table_figure(summary: pd.DataFrame, labels: dict[str, str], note: str, title: str, highlight_top: int):
    from evaluate._plot_style import Col, draw_metric_table, highlight_note

    wide = summary.pivot_table(index="model", columns=["set", "metric"], values="suite_tsis", aggfunc="first")
    order = list(dict.fromkeys(summary["model"]))
    wide = wide.reindex(order)
    frame = pd.DataFrame({f"{s}:{m}": wide[(s, m)].to_numpy() for s, m in wide.columns})
    cols = [Col(f"{s}:{m}", m, s, "max", "{:+.3f}") for s, m in wide.columns]
    foot = f"Suite tSIS = relative probe gain over the set's best baseline. Noise floor: {note}. " + \
        highlight_note(highlight_top)
    return draw_metric_table(frame, [labels.get(m, m) for m in order], cols, title, foot,
                             highlight_top=highlight_top)


# ---------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> Path:
    from evaluate._recompute import note_comparison
    from experiments.layout import DEFAULT_COMPARISON, ExperimentLayout

    args = parse_with_config(build_parser(), "geneicl_transfer", argv)
    note_comparison(args)
    layout = ExperimentLayout.from_root(args.root)
    labels = parse_mapping(args.row_labels, "--row-labels")
    unknown = sorted(set(args.metrics) - set(METRIC_TYPE))
    if unknown:
        sys.exit(f"Error: unknown --metrics {unknown}; choose from {sorted(METRIC_TYPE)}")
    try:
        sets = parse_sets(args.baselines, args.models, available_models(layout, args.suite))
        min_base = min_baselines(args.min_baseline)
        null_pairs, null_kind = resolve_null(layout, args.suite, args, sets)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    note = {"seed": NOTE_SEED, "probe": NOTE_PROBE, "none": NOTE_NONE}[null_kind]
    print(f"Noise floor: {null_pairs or 'none'} ({note})")

    rows = list(dict.fromkeys([*(m for ms, bs in sets.values() for m in [*ms, *bs]),
                               *(x for pair in null_pairs for x in pair)]))
    missing = [r for r in rows if not has_probe(layout, args.suite, r)]
    for r in missing:
        print(f"  [warn] {r}: no probe results; left out")
    rows = [r for r in rows if r not in missing]
    runs, _ = load_runs(layout, args.suite, rows)
    table, _ = per_task_table(runs)
    if table.empty:
        sys.exit("Error: no task scored by every model")
    per_task, summary, nulls = compute(table, sets, args.metrics, min_base, null_pairs,
                                       args.n_boot, args.n_perm, args.seed)
    if summary.empty:
        sys.exit("Error: nothing to compare")
    summary["noise_floor"] = note

    out_dir = layout.task_compare_dir(task_name(args.suite), "transfer", args.name or DEFAULT_COMPARISON)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_task.to_csv(out_dir / "per_task.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    nulls.to_csv(out_dir / "null.csv", index=False)
    show = ["set", "metric", "model", "suite_tsis", "ci_lo", "ci_hi", "win_rate", "q", "verdict"]
    print(summary[show].round(3).to_string(index=False))

    from evaluate._plot_style import save

    title = f"Transfer SIS on GeneICL {args.suite}" + (f": {args.name}" if args.name else "")
    figures = save(forest(summary, labels, note, title), out_dir / "transfer_sis", args.formats, args.dpi)
    figures += save(table_figure(summary, labels, note, title, args.highlight_top),
                    out_dir / "transfer_sis_table", args.formats, args.dpi)
    models = list(dict.fromkeys(split_row(r)[0] for r in rows))
    (out_dir / "manifest.json").write_text(json.dumps({
        "analysis": "geneicl_transfer", "suite": args.suite, "models": models, "row_labels": labels,
        "sets": {k: {"models": ms, "baselines": bs} for k, (ms, bs) in sets.items()},
        "metrics": args.metrics, "min_baseline": min_base, "n_boot": args.n_boot, "n_perm": args.n_perm,
        "seed": args.seed, "null_pairs": [list(p) for p in null_pairs], "noise_floor": null_kind,
        "figures": figures, "highlight_top": args.highlight_top,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
