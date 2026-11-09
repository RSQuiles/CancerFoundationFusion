"""Does the reconstruction evaluation see what training saw? Three read-only parts.

1. **Same input, two forward paths.** One collator batch of the store's validation rows
   goes through the training forward (``generative_forward`` + the training criterion)
   and, unchanged, through ``CancerFoundationReconstructor._forward``. Predictions must
   agree.
2. **Same rows, store bins vs evaluation bins.** For those rows, every token's training
   bin (from the store) must lie in the bin range the evaluation computes from the
   evaluation h5ad (``var_names`` through the vocabulary, binned over all vocabulary genes,
   ``bin_tie_range`` for ties). Catches gene-id and value mismatches end to end.
3. **Metric scale.** Validation rows and test rows, each through an in-memory mask plan in
   the training layout (the collator's mask ratio) and the r25 layout (0.25), scored with
   ``metrics.py`` for ``pred``, ``gene_mean`` (transductive), ``sample_mean`` and
   ``gene_mean_train`` (each gene's mean bin over training rows). Plus the training-path
   MSE on the same validation rows, comparable with W&B ``val/loss_expr``.

    python -m evaluate.reconstruction.check_train_eval --model bulk_only --modality bulk --split leiden_s0

Runs in the container (torch, bionemo SCDL; GPU if available). Writes
``results/tasks/reconstruction/<model>/_checks/train_eval_<split>/report.json`` and exits 1
when part 1 or part 2 fails. Only generative (``both``) models without conditions are
supported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRED_TOL = 1e-3
REPORT_KEYS = ("mse", "mae", "mae_nonzero_target", "spearman_mean", "auroc_zero")


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--model", required=True, help="CancerFoundation model name (checkpoints/<model>/)")
    p.add_argument("--ckpt", default=None, help="Checkpoint path or file name (default: newest)")
    p.add_argument("--modality", default="bulk")
    p.add_argument("--split", required=True, help="Split whose test.txt gives the test rows")
    p.add_argument("--store", type=Path, default=None, help="Training store (default: the checkpoint's data_path)")
    p.add_argument("--plan", default=None,
                   help="Mask plan whose source the evaluation reads (default: <mod>:<split>:test:r25 if present)")
    p.add_argument("--eval-source", type=Path, default=None,
                   help="h5ad the evaluation reads (default: the plan's source, else layout.raw_h5ad)")
    p.add_argument("--train-source", type=Path, default=None,
                   help="h5ad the store was built from, for gene_mean_train (default: split.json source)")
    p.add_argument("--n-exact", type=int, default=64, help="Validation rows in parts 1 and 2")
    p.add_argument("--n-eval", type=int, default=500, help="Rows per set in part 3")
    p.add_argument("--n-gene-ref", type=int, default=2000, help="Training rows behind gene_mean_train")
    p.add_argument("--mask-ratio", type=float, default=None,
                   help="Training-layout mask ratio (default: the checkpoint's collator value)")
    p.add_argument("--eval-mask-ratio", type=float, default=0.25, help="Second plan shape (r25)")
    p.add_argument("--seq-len", type=int, default=None, help="Default: the model's max_seq_len")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def resolve_paths(args, layout, model):
    from evaluate.reconstruction.mask_plan import load_plan, resolve_plan_path

    store = Path(args.store or model.data_path)
    plan_path = resolve_plan_path(args.plan or f"{args.modality}:{args.split}:test:r25", args.root)
    eval_source = args.eval_source
    if eval_source is None and plan_path.exists():
        eval_source = Path(load_plan(plan_path).meta["source"]["path"])
    eval_source = Path(eval_source or layout.raw_h5ad(args.modality))
    train_source = args.train_source
    card = layout.split_card(args.modality, args.split)
    if train_source is None and card.exists():
        train_source = Path(json.loads(card.read_text(encoding="utf-8"))["source"]["path"])
    train_source = Path(train_source or eval_source)
    test_file = layout.split_index_file(args.modality, args.split, "test")
    if not test_file.exists():
        sys.exit(f"Error: {test_file} not found")
    test_names = test_file.read_text(encoding="utf-8").split()
    return store, plan_path, eval_source, train_source, test_names


def make_collator(model, mask_ratio: float):
    from cancerfoundation.data.data_collator import AnnDataCollator

    binned = model.input_style == "binned"
    return AnnDataCollator(
        do_padding=model.max_seq_len is not None,
        pad_token_id=model.pad_token_id,
        pad_value=model.pad_value,
        do_mlm=True,
        do_binning=binned,
        normalise_bins=model.normalise_bins,
        mask_ratio=mask_ratio,
        mask_value=model.mask_value,
        max_length=model.max_seq_len,
        sampling=model.TRUNC_BY_SAMPLE,
        data_style="both",
        n_bins=model.n_bins if binned else None,
        conditions=None,
        zero_percentages=model.zero_percentages,
        condition_token=False,
    )


def val_indices(n: int, seed: int = 42) -> np.ndarray:
    """The store rows ``BulkSCDataModule.setup`` held out for validation."""
    import torch
    from torch.utils.data import random_split

    _, val = random_split(range(n), [0.95, 0.05], generator=torch.Generator().manual_seed(seed))
    return np.asarray(val.indices, dtype=np.int64)


def training_forward(model, batch, device, torch):
    """``(gen_preds, gen_target, valid)`` exactly as ``TransformerModule.forward`` computes them."""
    pg, pv = batch["pcpt_gene"].to(device), batch["pcpt_expr"].to(device)
    pkpm, gg = batch["pcpt_key_padding_mask"].to(device), batch["gen_gene"].to(device)
    gkpm = batch["gen_key_padding_mask"].to(device)
    with torch.no_grad():
        out = model.model.generative_forward(pg, pv, pkpm, gg, gkpm, torch.cat([pkpm, gkpm], 1),
                                             batch["attn_mask"].to(device), conditions=None)
    return out["gen_preds"].float(), batch["gen_expr_target"].to(device).float(), ~gkpm


def batch_to_reconstructor_inputs(batch):
    """Per-sample context ids/values and masked ids, as ``predict`` would build them."""
    ctx_ids, ctx_vals, msk_ids = [], [], []
    for j in range(batch["pcpt_gene"].shape[0]):
        keep = ~batch["pcpt_key_padding_mask"][j, 1:]
        ctx_ids.append(batch["pcpt_gene"][j, 1:][keep].numpy())
        ctx_vals.append(batch["pcpt_expr"][j, 1:][keep].numpy())
        msk_ids.append(batch["gen_gene"][j][~batch["gen_key_padding_mask"][j]].numpy())
    return ctx_ids, ctx_vals, msk_ids


# --------------------------------------------------------------------------- #
# parts
# --------------------------------------------------------------------------- #


def part1(model, recon, batch, device, torch) -> dict:
    preds, target, valid = training_forward(model, batch, device, torch)
    train_loss = float(model.model.criterion(preds, target, valid))
    ctx_ids, ctx_vals, msk_ids = batch_to_reconstructor_inputs(batch)
    raw = recon._forward(ctx_ids, ctx_vals, msk_ids, {}, device, torch)
    diffs, sq_train, sq_recon, n = [], 0.0, 0.0, 0
    for j in range(preds.shape[0]):
        a = preds[j][valid[j]].cpu().numpy()
        b = np.asarray(raw[j][: len(a)], dtype=np.float32)
        t = target[j][valid[j]].cpu().numpy()
        diffs.append(float(np.abs(a - b).max()) if a.size else 0.0)
        if a.ndim == 1:  # mse: one value per gene
            sq_train += float(((a - t) ** 2).sum())
            sq_recon += float(((b - t) ** 2).sum())
            n += len(t)
    res = {"rows": int(preds.shape[0]), "max_abs_pred_diff": max(diffs),
           "training_criterion": train_loss,
           "masked_mse_training_path": sq_train / n if n else None,
           "masked_mse_reconstructor_path": sq_recon / n if n else None}
    res["status"] = "PASS" if res["max_abs_pred_diff"] < PRED_TOL else "FAIL"
    return res


def part2(model, batch, eval_adata) -> dict:
    """Training bins of every batch token vs the evaluation's tie range for that gene."""
    import scipy.sparse as sp

    from evaluate.reconstruction.reconstructors import bin_tie_range

    vocab = model.vocab
    inv = {v: k for k, v in vocab.items()}
    var_names = np.asarray(eval_adata.var_names)
    vocab_cols = np.flatnonzero(np.array([g in vocab for g in var_names]))
    col_ids = np.array([vocab[g] for g in var_names[vocab_cols]])
    id2pos: dict[int, int] = {}
    for pos, tok in enumerate(col_ids):
        id2pos.setdefault(int(tok), pos)
    dup_tokens = int(len(col_ids) - len(id2pos))
    scale = model.n_bins if model.normalise_bins else 1

    inside = outside = missing = 0
    examples = []
    for j in range(batch["pcpt_gene"].shape[0]):
        row = eval_adata.X[j]
        row = np.asarray(row.toarray() if sp.issparse(row) else row, dtype=np.float32).ravel()[vocab_cols]
        lo, hi = bin_tie_range(row, model.n_bins)
        pk = ~batch["pcpt_key_padding_mask"][j, 1:]
        gk = ~batch["gen_key_padding_mask"][j]
        ids = np.concatenate([batch["pcpt_gene"][j, 1:][pk].numpy(), batch["gen_gene"][j][gk].numpy()])
        bins = np.concatenate([batch["pcpt_expr"][j, 1:][pk].numpy(), batch["gen_expr_target"][j][gk].numpy()])
        bins = np.rint(bins * scale)
        for tok, b in zip(ids, bins):
            pos = id2pos.get(int(tok))
            if pos is None:
                missing += 1
                continue
            if lo[pos] <= b <= hi[pos]:
                inside += 1
            else:
                outside += 1
                if len(examples) < 8:
                    examples.append(f"{eval_adata.obs_names[j]} {inv.get(int(tok))}: training bin {int(b)}, "
                                    f"evaluation range [{lo[pos]}, {hi[pos]}], evaluation value {row[pos]:.4g}")
    total = inside + outside
    res = {"tokens": total + missing, "inside_eval_range": inside, "outside_eval_range": outside,
           "share_outside": outside / total if total else None, "tokens_without_eval_column": missing,
           "eval_columns_sharing_a_token": dup_tokens, "examples": examples}
    res["status"] = "PASS" if outside == 0 and missing == 0 else "FAIL"
    return res


def train_gene_means(vocab, adata, n_bins: int) -> pd.Series:
    """Each gene's mean bin over ``adata``'s rows, binned as the evaluation bins them."""
    import scipy.sparse as sp

    from cancerfoundation.data.preprocess import binning

    var_names = np.asarray(adata.var_names)
    cols = np.flatnonzero(np.array([g in vocab for g in var_names]))
    total = np.zeros(len(cols))
    for i in range(adata.n_obs):
        row = adata.X[i]
        row = np.asarray(row.toarray() if sp.issparse(row) else row, dtype=np.float32).ravel()[cols]
        total += np.asarray(binning(row, n_bins), dtype=float)
    return pd.Series(total / max(adata.n_obs, 1), index=var_names[cols])


def score_set(recon, adata, genes, seq_len, ratio, seed, gm_train: pd.Series) -> dict:
    from evaluate.reconstruction import metrics as M
    from evaluate.reconstruction.mask_plan import build_plan

    plan = build_plan(adata, genes, seq_len, ratio, seed)
    result = recon.predict(adata, plan)
    frame = M.entries_frame(result, plan, adata.obs)
    frame["gene_mean_train"] = gm_train.reindex(frame["gene"]).to_numpy()
    out = {}
    for col in ("pred", "gene_mean", "sample_mean", "gene_mean_train"):
        sub = frame[np.isfinite(frame[col].to_numpy(dtype=float))]
        if not len(sub):
            continue
        vals = {**M.native(sub, "bins", col), **M.space_free(sub, col)}
        out[col] = {k: vals.get(k) for k in REPORT_KEYS} | {"n_entries": int(len(sub))}
    return out


def training_path_mse(model, ds, rows, collator, batch_size, device, torch) -> float:
    sq, n = 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch = collator([ds[int(i)] for i in rows[start:start + batch_size]])
        preds, target, valid = training_forward(model, batch, device, torch)
        if preds.dim() != 2:
            return float("nan")  # non-mse losses: compare via the reconstructor metrics instead
        sq += float((((preds - target) ** 2) * valid).sum())
        n += int(valid.sum())
    return sq / n if n else float("nan")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import torch

    from cancerfoundation.data.dataset import SingleCellDataset
    from embed.embed import load_model
    from evaluate.reconstruction.mask_plan import read_rows
    from evaluate.reconstruction.reconstructors import CancerFoundationReconstructor
    from experiments.layout import ExperimentLayout, canonical_modality

    args = build_parser().parse_args(argv)
    args.modality = canonical_modality(args.modality)
    layout = ExperimentLayout.from_root(args.root)
    ckpt = Path(args.ckpt) if args.ckpt else layout.latest_checkpoint(args.model)
    if ckpt is not None and not ckpt.exists() and args.ckpt:
        ckpt = layout.checkpoint_dir(args.model) / args.ckpt
    if ckpt is None or not ckpt.exists():
        sys.exit(f"Error: no checkpoint for {args.model}")

    model = load_model(ckpt, args.cpu)
    if not model.model.use_generative_training:
        sys.exit("Error: only generative (training_tasks=both) models are supported")
    if model.model.conditions:
        sys.exit("Error: models with conditions are not supported by this check yet")
    device = next(model.parameters()).device
    recon = CancerFoundationReconstructor(checkpoint=str(ckpt), batch_size=args.batch_size,
                                          cpu=args.cpu, seed=args.seed)
    recon.model = model  # reuse the loaded model instead of loading it twice
    recon.loss = getattr(model.loss_type, "value", model.loss_type)
    recon.generative, recon.conditions = True, []

    store, plan_path, eval_source, train_source, test_names = resolve_paths(args, layout, model)
    mask_ratio = args.mask_ratio or float(model.hparams.get("mask_ratio", 0.4))
    seq_len = args.seq_len or int(model.max_seq_len)
    print(f"Checkpoint:   {ckpt}\nStore:        {store}\nEval source:  {eval_source}\n"
          f"Train source: {train_source}\nPlan:         {plan_path}\n"
          f"Collator: mask_ratio {mask_ratio}, max_seq_len {model.max_seq_len}, n_bins {model.n_bins}, "
          f"trunc_by_sample {model.TRUNC_BY_SAMPLE}, loss {recon.loss}")

    ds = SingleCellDataset(store, pad_value=model.pad_value)
    vocab_same = ds.vocab == model.vocab
    print(f"Store vocab == checkpoint vocab: {vocab_same} ({len(ds.vocab)} vs {len(model.vocab)} tokens)")
    report = {"model": args.model, "checkpoint": str(ckpt), "store": str(store), "eval_source": str(eval_source),
              "train_source": str(train_source), "plan": str(plan_path), "mask_ratio": mask_ratio,
              "seq_len": seq_len, "store_vocab_equals_checkpoint_vocab": vocab_same}

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    val_idx = val_indices(len(ds))
    exact_idx = np.sort(rng.choice(val_idx, size=min(args.n_exact, len(val_idx)), replace=False))
    collator = make_collator(model, mask_ratio)
    batch = collator([ds[int(i)] for i in exact_idx])

    print("\n== Part 1: training forward vs reconstructor forward (same batch) ==")
    report["part1"] = p1 = part1(model, recon, batch, device, torch)
    print(json.dumps(p1, indent=2, default=_jsonable))

    print("\n== Part 2: store bins vs evaluation bins (same rows) ==")
    exact_names = list(ds.obs.index.astype(str)[exact_idx])
    try:
        eval_rows = read_rows(eval_source, exact_names)
        p2 = part2(model, batch, eval_rows)
    except KeyError as exc:
        p2 = {"status": "FAIL", "error": f"validation rows missing from the evaluation source: {exc}"}
    report["part2"] = p2
    print(json.dumps(p2, indent=2, default=_jsonable))

    print("\n== Part 3: metric scale ==")
    gene_ref_pool = np.setdiff1d(np.arange(len(ds)), val_idx)
    ref_names = list(ds.obs.index.astype(str)[np.sort(rng.choice(gene_ref_pool, size=min(args.n_gene_ref, len(gene_ref_pool)), replace=False))])
    gm_train = train_gene_means(model.vocab, read_rows(train_source, ref_names), model.n_bins)
    eval_val_idx = np.sort(rng.choice(val_idx, size=min(args.n_eval, len(val_idx)), replace=False))
    sets = {"val": list(ds.obs.index.astype(str)[eval_val_idx]),
            "test": [test_names[i] for i in np.sort(rng.choice(len(test_names), size=min(args.n_eval, len(test_names)), replace=False))]}
    torch.manual_seed(args.seed)
    report["part3"] = p3 = {"training_path_mse_val": training_path_mse(
        model, ds, eval_val_idx, make_collator(model, mask_ratio), args.batch_size, device, torch)}
    print(f"  training-path masked MSE on {len(eval_val_idx)} val rows: {p3['training_path_mse_val']:.2f}")
    ratios = list(dict.fromkeys([mask_ratio, args.eval_mask_ratio]))
    for set_name, names in sets.items():
        try:
            adata = read_rows(eval_source, names)
        except KeyError as exc:
            print(f"  [warn] {set_name} rows missing from the evaluation source ({exc}); reading the training source")
            adata = read_rows(train_source, names)
        genes = list(adata.var_names)
        for ratio in ratios:
            key = f"{set_name}_mask{ratio:g}"
            p3[key] = score_set(recon, adata, genes, seq_len, ratio, args.seed, gm_train)
    print_table(p3, ratios)
    report["verdict"] = verdict(p1, p2, p3, mask_ratio, args.eval_mask_ratio)
    print("\nVerdict:\n  " + "\n  ".join(report["verdict"]))

    out_dir = args.out_dir or layout.task_dir("reconstruction", args.model) / "_checks" / f"train_eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if p1["status"] == "FAIL" or p2["status"] == "FAIL" else 0


def print_table(p3: dict, ratios: list[float]) -> None:
    print(f"\n  {'rows / plan':18s}{'predictor':18s}" + "".join(f"{k:>20s}" for k in REPORT_KEYS))
    for key, block in p3.items():
        if not isinstance(block, dict):
            continue
        for predictor, vals in block.items():
            cells = "".join(f"{vals[k]:>20.4f}" if vals.get(k) is not None else f"{'-':>20s}" for k in REPORT_KEYS)
            print(f"  {key:18s}{predictor:18s}{cells}")


def _mse(p3, key, predictor="pred"):
    return (p3.get(key) or {}).get(predictor, {}).get("mse")


def verdict(p1, p2, p3, train_ratio, eval_ratio) -> list[str]:
    """Heuristic reading of the three parts (thresholds are rules of thumb, not tests)."""
    lines = []
    if p1["status"] == "FAIL":
        lines.append("Part 1 FAIL: the reconstructor forward differs from the training forward on identical input.")
    if p2["status"] == "FAIL":
        lines.append("Part 2 FAIL: training bins and evaluation bins disagree for the same rows: gene ids or "
                     "values differ between the store and the evaluation h5ad (see check_store_alignment).")
    if lines:
        return lines
    tr = p3.get("training_path_mse_val")
    v_train, v_eval = _mse(p3, f"val_mask{train_ratio:g}"), _mse(p3, f"val_mask{eval_ratio:g}")
    t_train = _mse(p3, f"test_mask{train_ratio:g}")
    gm_val = _mse(p3, f"val_mask{train_ratio:g}", "gene_mean_train")
    gm_test = _mse(p3, f"test_mask{train_ratio:g}", "gene_mean_train")
    lines.append("Parts 1-2 PASS: same input gives the same prediction, and the evaluation bins the training values.")
    if tr and v_train and v_train > 1.3 * tr:
        lines.append(f"Val rows: evaluation path MSE {v_train:.1f} vs training path {tr:.1f}: a remaining "
                     "difference in how the evaluation builds sequences.")
    if v_train and v_eval and v_eval > 1.3 * v_train:
        lines.append(f"Val rows: MSE {v_eval:.1f} at mask {eval_ratio:g} vs {v_train:.1f} at {train_ratio:g}: "
                     "sensitive to the evaluation layout.")
    if v_train and t_train and t_train > 1.3 * v_train:
        shift = gm_val and gm_test and gm_test > 1.3 * gm_val
        lines.append(f"Test rows: MSE {t_train:.1f} vs {v_train:.1f} on val rows"
                     + (f"; gene_mean_train also degrades ({gm_val:.1f} -> {gm_test:.1f}): the test rows differ "
                        "from the training rows." if shift else
                        "; gene_mean_train does not degrade as much: the model generalizes worse than a "
                        "per-gene prior to these rows."))
    if len(lines) == 1:
        lines.append("No large gap between training path, val rows and test rows at these thresholds.")
    return lines


if __name__ == "__main__":
    sys.exit(main())
