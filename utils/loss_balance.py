"""Calibrate the alignment-loss weights against reconstruction (``--calibrate-loss-steps``).

The alignment losses differ widely in scale (MMD ~0.01-0.1, contrastive up to ~3, the
paired loss acts on one batch in ``paired_every_n``), so equal weights do not mean equal
pressure. This probe measures, on the warm-started model before any training step, how
strongly each term pulls on the shared backbone compared with reconstruction, and
suggests fixed weights that give every alignment term the same share of it.

For each training batch and each loss term ``k`` present in it:

    ratio_k = || d L_k / d theta_backbone ||  /  || d L_recon / d theta_backbone ||

with ``L_k`` the raw (unweighted) term and ``L_recon`` the reconstruction loss as the model
weights it (``w_rec * (expr + w_mvc * mvc + gen)``). The suggested weight is

    w_k = target_share / median_k(ratio_k)

so that, at the parent, ``w_k * ||g_k|| = target_share * ||g_recon||`` on the batches where
``k`` acts. A term that acts on few batches (paired) gets that share on its own batches.

``suggest_weights`` and ``format_report`` are numpy-only (offline self-check:
``python scripts/check_loss_balance.py``); ``probe`` needs torch and runs inside
``pretrain.main``.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

RECON_TERMS = ("expr", "mvc", "gen")
# term -> (launch-config key of its weight, CancerFoundation/module attribute of that weight)
ALIGN_TERMS = {
    "mmd": ("loss_weight_mmd", "weight_mmd"),
    "contrastive": ("loss_weight_contrastive", "weight_contrastive"),
    "cdd": ("loss_weight_cdd", "weight_cdd"),
    "paired": ("loss_weight_paired", "weight_paired"),
    "agg": ("loss_weight_agg", "weight_agg"),
}
NOTES = {
    "mmd": "marginal bulk vs pseudobulk (multi-RBF MMD on CLS)",
    "contrastive": "tissue InfoNCE bulk <-> pseudobulk (anchors need the tissue in both domains)",
    "cdd": "per-tissue MMD minus between-tissue MMD (tissues in both domains)",
    "paired": "MSE pb<->bulk + cells mean<->bulk/pb, paired batches only",
    "agg": "MSE pseudobulk vs mean of its sampled cells (mean over pseudobulks)",
}
MIN_BATCHES = 10        # fewer batches carrying a term: the suggestion is flagged unreliable
SPREAD_WARN = 1.0       # IQR / median of the ratio above this: flagged as variable


def _round_sig(x: float, digits: int = 2) -> float:
    if not math.isfinite(x) or x == 0:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + digits - 1)


def _stats(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q25": float(q25), "q75": float(q75)}


def suggest_weights(records: list[dict], weights: dict[str, float], target_share: float,
                    n_batches: int | None = None) -> dict[str, Any]:
    """Per-term statistics and suggested weights from the probe's per-batch records.

    ``records``: one dict per batch, ``{"paired": bool, "recon_grad": float,
    "terms": {term: {"loss": float, "grad": float}}}``. ``weights``: the current weight of
    each alignment term (launch-config values). Returns ``{"terms": {...}, ...}``; a term
    with no batch keeps its weight and says so.
    """
    n_batches = len(records) if n_batches is None else n_batches
    out: dict[str, Any] = {"target_share": target_share, "n_batches": n_batches,
                           "n_paired_batches": int(sum(bool(r.get("paired")) for r in records)),
                           "recon_grad": _stats([r["recon_grad"] for r in records]),
                           "terms": {}}
    for term in list(RECON_TERMS) + list(ALIGN_TERMS):
        rows = [(r["terms"][term], r["recon_grad"]) for r in records if term in r.get("terms", {})]
        losses = [t["loss"] for t, _ in rows]
        grads = [t["grad"] for t, _ in rows]
        ratios = [t["grad"] / g for t, g in rows if g > 0]
        entry: dict[str, Any] = {"batches": len(rows), "loss": _stats(losses), "grad": _stats(grads),
                                 "ratio": _stats(ratios)}
        if term in ALIGN_TERMS:
            key, _ = ALIGN_TERMS[term]
            current = float(weights.get(term, float("nan")))
            ratio = entry["ratio"]["median"]
            entry["config_key"] = key
            entry["current_weight"] = current
            entry["share_at_current"] = current * ratio if math.isfinite(ratio) else float("nan")
            flags = []
            if not rows:
                flags.append("never computed: weight kept")
                entry["suggested_weight"] = current
            elif not (ratio > 0):
                flags.append("zero gradient on the backbone: weight kept")
                entry["suggested_weight"] = current
            else:
                entry["suggested_weight"] = _round_sig(target_share / ratio)
                if len(rows) < MIN_BATCHES:
                    flags.append(f"only {len(rows)} batches (< {MIN_BATCHES}): unreliable")
                spread = (entry["ratio"]["q75"] - entry["ratio"]["q25"]) / ratio
                if spread > SPREAD_WARN:
                    flags.append(f"variable across batches (IQR/median {spread:.1f})")
            entry["flags"] = flags
        out["terms"][term] = entry
    return out


def format_report(result: dict[str, Any]) -> str:
    """The decision table printed at the end of the probe."""
    R = result["target_share"]
    lines = [
        "",
        "=" * 100,
        f"[calibrate] loss-weight suggestion: {result['n_batches']} batches "
        f"({result['n_paired_batches']} paired), target share R = {R}",
        "  ratio = ||grad of the raw term on the backbone|| / ||grad of the weighted reconstruction||",
        "  suggested weight = R / median(ratio): at the parent each term then pulls with R x reconstruction",
        f"  reconstruction gradient norm: median {result['recon_grad']['median']:.4g} "
        f"[IQR {result['recon_grad']['q25']:.4g} - {result['recon_grad']['q75']:.4g}]",
        "-" * 100,
        f"  {'term':12s}{'batches':>8s}{'loss (median)':>15s}{'ratio median [IQR]':>28s}"
        f"{'weight now':>12s}{'share now':>11s}{'suggested':>11s}",
    ]
    for term, e in result["terms"].items():
        r = e["ratio"]
        ratio_txt = (f"{r['median']:.3g} [{r['q25']:.3g}-{r['q75']:.3g}]" if e["batches"] else "-")
        loss_txt = f"{e['loss']['median']:.4g}" if e["batches"] else "-"
        if term in ALIGN_TERMS:
            now = f"{e['current_weight']:.3g}"
            share = f"{e['share_at_current']:.3g}" if math.isfinite(e["share_at_current"]) else "-"
            sug = f"{e['suggested_weight']:.3g}"
        else:
            now, share, sug = "(recon)", "", ""
        lines.append(f"  {term:12s}{e['batches']:>8d}{loss_txt:>15s}{ratio_txt:>28s}{now:>12s}{share:>11s}{sug:>11s}")
    lines.append("-" * 100)
    for term, e in result["terms"].items():
        if term not in ALIGN_TERMS:
            continue
        if e["batches"]:
            why = (f"ratio {e['ratio']['median']:.3g} at weight 1 -> {R} / {e['ratio']['median']:.3g} "
                   f"= {e['suggested_weight']:.3g}")
        else:
            why = "not computed in any batch"
        flags = f"  [!] {'; '.join(e['flags'])}" if e["flags"] else ""
        lines.append(f"  {term:12s}{NOTES[term]}\n  {'':12s}{why}{flags}")
    lines.append("=" * 100)
    return "\n".join(lines)


def config_snippet(result: dict[str, Any]) -> str:
    """The ``features`` weights to paste into the launch config."""
    weights = {e["config_key"]: e["suggested_weight"] for t, e in result["terms"].items() if t in ALIGN_TERMS}
    return json.dumps(weights, indent=2)


# --------------------------------------------------------------------------- #
# The probe (torch)
# --------------------------------------------------------------------------- #


def _to_device(obj, device):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _grad_norm(term, params) -> float:
    import torch

    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    total = sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None)
    return math.sqrt(total)


def probe(model, datamodule, args) -> Path:
    """Measure every loss term's backbone gradient on ``args.calibrate_loss_steps`` batches.

    ``model`` is the warm-started CancerFoundation, ``datamodule`` set up for fit. Writes
    ``<save_dir>/loss_balance.json`` and prints the decision table plus a config snippet.
    Nothing is trained: no optimizer step is taken.
    """
    import torch

    from utils.warm_start import classify, normalize_key

    save_dir = Path(args.save_dir)
    out_path = save_dir / "loss_balance.json"  # pretrain.main returns early when it exists

    steps, R = int(args.calibrate_loss_steps), float(args.calibrate_target_share)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus != 0 else "cpu")
    core = getattr(model.model, "_orig_mod", model.model)  # no torch.compile for the probe
    core.to(device).train()

    # What the LightningModule does at fit time: excluded class codes, full CDD weight
    # (the probe measures the term itself, so the warmup schedule does not apply).
    mapping = getattr(datamodule.dataset, "mapping", {}) or {}
    class_map = mapping.get(core.class_column, {})
    core.class_exclude_codes = tuple(int(c) for n, c in class_map.items() if n in (args.class_exclude_labels or []))
    core._cdd_w = core.weight_cdd

    named = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
    backbone = [p for n, p in named if classify(normalize_key(f"model.{n}") or "") == "backbone"]
    if not backbone:
        raise RuntimeError("[calibrate] no backbone parameter found (utils.warm_start.classify)")
    weights = {t: float(getattr(core, attr)) for t, (_, attr) in ALIGN_TERMS.items()}
    active = {"mmd": core.mmd, "contrastive": core.contrastive, "cdd": core.cdd,
              "paired": core.paired_alignment, "agg": core.aggregation}
    w_rec, w_mvc = float(core.weight_reconstruction), float(core.weight_mvc)

    print("\n" + "=" * 100)
    print(f"[calibrate] loss-weight probe: {steps} training batches, no optimizer step, device {device}")
    print(f"[calibrate] backbone: {len(backbone)} tensors, {sum(p.numel() for p in backbone):,} parameters "
          f"(utils.warm_start group 'backbone': gene/value encoders + transformer)")
    print(f"[calibrate] reconstruction reference: {w_rec} x (expr + {w_mvc} x mvc + gen); "
          f"use_cell_embedding on (as after step 1000)")
    print(f"[calibrate] alignment terms on: {[t for t, on in active.items() if on] or 'none'}; "
          f"off (not measured): {[t for t, on in active.items() if not on] or 'none'}")
    print(f"[calibrate] current weights: {weights}")
    print(f"[calibrate] excluded class codes {core.class_exclude_codes} "
          f"({args.class_exclude_labels}); target share R = {R}")
    off = [t for t, on in active.items() if not on]
    if off:
        print(f"[calibrate] [WARNING] {off} are off in this config, so they get no suggestion; "
              f"turn every loss on in the calibration model")
    # The datamodule reads trainer.world_size / global_rank to build its sampler; Lightning
    # attaches a trainer only during fit, so stand in for one process (one GPU).
    if getattr(datamodule, "trainer", None) is None:
        import types

        datamodule.trainer = types.SimpleNamespace(world_size=1, global_rank=0)
        print("[calibrate] no Lightning trainer: dataloader built for 1 process (world_size 1)")
    print("=" * 100, flush=True)

    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if str(args.precision).startswith("bf16") and device.type == "cuda" else None)
    records: list[dict] = []
    started = time.perf_counter()
    loader = iter(datamodule.train_dataloader())
    for step in range(steps):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(datamodule.train_dataloader())
            batch = next(loader)
        batch = _to_device(batch, device)
        paired = batch.get("is_paired_batch", False)
        paired = bool(paired.item()) if isinstance(paired, torch.Tensor) else bool(paired)

        core.loss_terms = {}
        if autocast is not None:
            with autocast:
                core(batch, use_cell_embedding=True, apply_dat=False)
        else:
            core(batch, use_cell_embedding=True, apply_dat=False)
        terms, core.loss_terms = core.loss_terms, None

        recon = w_rec * (terms.get("expr", 0.0) + w_mvc * terms.get("mvc", 0.0) + terms.get("gen", 0.0))
        rec: dict[str, Any] = {"paired": paired, "recon_grad": _grad_norm(recon, backbone), "terms": {}}
        for name, value in terms.items():
            if not (isinstance(value, torch.Tensor) and value.requires_grad):
                continue  # a skipped term returns a constant zero
            if name in ALIGN_TERMS and float(value.detach()) == 0.0:
                continue  # no anchor / class in this batch: the term did not act
            rec["terms"][name] = {"loss": float(value.detach().float()), "grad": _grad_norm(value, backbone)}
        records.append(rec)
        del terms, recon
        model.zero_grad(set_to_none=True)

        if (step + 1) % 25 == 0 or step == 0 or step + 1 == steps:
            seen = {t: sum(t in r["terms"] for r in records) for t in ALIGN_TERMS if active[t]}
            ratios = {t: np.median([r["terms"][t]["grad"] / r["recon_grad"] for r in records
                                    if t in r["terms"] and r["recon_grad"] > 0])
                      for t, n in seen.items() if n}
            print(f"[calibrate] batch {step + 1}/{steps} ({time.perf_counter() - started:.0f} s): "
                  f"batches per term {seen}; running median ratio "
                  + ", ".join(f"{t}={v:.3g}" for t, v in ratios.items()), flush=True)

    result = suggest_weights(records, weights, R)
    print(format_report(result))
    snippet = config_snippet(result)
    print("[calibrate] paste into base_config.features of the launch config:\n" + snippet)
    print("[calibrate] the share is measured at the parent, where the alignment losses are largest: "
          "as they shrink in training, so does their pull, so R is a starting share.", flush=True)

    result.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": steps, "weights_at_probe": weights, "reconstruction": {"weight": w_rec, "weight_mvc": w_mvc},
        "snippet": json.loads(snippet), "init_from": getattr(args, "init_from", None),
        "paired_every_n": getattr(args, "paired_every_n", None),
    })
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[calibrate] wrote {out_path}")
    return out_path
