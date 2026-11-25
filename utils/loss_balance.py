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
    "dat": ("loss_weight_dat", "weight_dat"),
    # The paired form the probed model does NOT train on (mse <-> infonce), measured on the
    # same paired batches, so one probe calibrates both. Its weight is a per-run override of
    # loss_weight_paired for runs that switch paired_loss; never part of the shared snippet.
    "paired_alt": ("loss_weight_paired", "weight_paired"),
    # The same for the aggregation form (agg_loss mse <-> infonce), on the same batches.
    "agg_alt": ("loss_weight_agg", "weight_agg"),
}
# alt term -> (its main term, the config key that switches the form)
ALT_TERMS = {"paired_alt": ("paired", "paired_loss"), "agg_alt": ("agg", "agg_loss")}
NOTES = {
    "mmd": "marginal bulk vs pseudobulk (multi-RBF MMD on CLS)",
    "contrastive": "tissue InfoNCE bulk <-> pseudobulk (anchors need the tissue in both domains)",
    "cdd": "per-tissue MMD minus between-tissue MMD (tissues in both domains)",
    "paired": "paired batches only: symmetric InfoNCE over the batch's pairs (paired_loss infonce) "
              "or MSE pb<->bulk + cells mean<->bulk/pb (mse)",
    "agg": "pseudobulk vs mean of its sampled cells: symmetric InfoNCE over the batch's pseudobulks "
           "(agg_loss infonce) or MSE (mse)",
    "dat": "adversarial bulk vs pseudobulk discriminator on CLS, through gradient reversal; measured "
           "against a discriminator first trained on the frozen parent (the strongest adversary at the start)",
    "paired_alt": "the other paired form (not trained here): loss_weight_paired for runs that switch paired_loss",
    "agg_alt": "the other aggregation form (not trained here): loss_weight_agg for runs that switch agg_loss",
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
        forms = result.get("paired_forms") or {}
        aforms = result.get("agg_forms") or {}
        form = {"paired": ("paired_loss", forms.get("trained")), "paired_alt": ("paired_loss", forms.get("alt")),
                "agg": ("agg_loss", aforms.get("trained")), "agg_alt": ("agg_loss", aforms.get("alt"))}.get(term)
        note = NOTES[term] + (f" [{form[0]} {form[1]}]" if form and form[1] else "")
        lines.append(f"  {term:12s}{note}\n  {'':12s}{why}{flags}")
    lines.append("=" * 100)
    return "\n".join(lines)


def config_snippet(result: dict[str, Any]) -> str:
    """The ``features`` weights to paste into the launch config."""
    weights = {e["config_key"]: e["suggested_weight"] for t, e in result["terms"].items()
               if t in ALIGN_TERMS and t not in ALT_TERMS}
    return json.dumps(weights, indent=2)


def forms_line(result: dict[str, Any], alt_term: str) -> str | None:
    """Both forms' weights of a loss with two forms (``paired_alt`` / ``agg_alt``): the
    shared one and the per-run override for runs that switch to the other form."""
    main_term, switch = ALT_TERMS[alt_term]
    forms = result.get(f"{main_term}_forms")
    alt = result["terms"].get(alt_term)
    if not forms or not alt or not alt["batches"]:
        return None
    key = ALIGN_TERMS[main_term][0]
    main = result["terms"][main_term]["suggested_weight"]
    return (f"[calibrate] {main_term} forms: {switch} {forms['trained']} -> {key} {main:.3g} "
            f"(shared snippet); runs with {switch} {forms['alt']} -> override {key} "
            f"{alt['suggested_weight']:.3g}")


def paired_forms_line(result: dict[str, Any]) -> str | None:
    """Both paired forms' weights: the shared one and the per-run override for the other."""
    return forms_line(result, "paired_alt")


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


def _batch_rows(batch) -> int:
    """Rows of a collated batch: the first dimension of its gene-id tensor."""
    import torch

    for key in ("gene", "genes", "pcpt_gene", "input_gene_ids"):
        value = batch.get(key) if isinstance(batch, dict) else None
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    mod = batch.get("conditions", {}).get("modality") if isinstance(batch, dict) else None
    return int(mod.shape[0]) if isinstance(mod, torch.Tensor) else -1


def _grad_norm(term, params) -> float:
    import torch

    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    total = sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None)
    return math.sqrt(total)


def _train_discriminators(core, next_batch, n_steps: int, autocast) -> None:
    """Train the DAT discriminators alone on the frozen model for ``n_steps`` batches.

    A warm-started model gets fresh discriminators (the parent had none, or ``heads`` is not
    in ``init_from``), and the gradient a random discriminator sends back through the
    reversal layer says nothing about DAT's pull. Training it first on the parent's
    embeddings gives the adversary DAT faces at the start of training. Only the
    discriminator parameters move (AdamW, lr 1e-3); every other parameter is frozen for
    this phase and restored afterwards, and the probe takes no other step.
    """
    import torch

    discs = getattr(core, "grad_reverse_discriminators", None)
    if discs is None or n_steps <= 0:
        return
    disc_params = list(discs.parameters())
    disc_ids = {id(p) for p in disc_params}
    frozen = [p for p in core.parameters() if p.requires_grad and id(p) not in disc_ids]
    for p in frozen:
        p.requires_grad_(False)
    opt = torch.optim.AdamW(disc_params, lr=1e-3)
    conf_key = None
    first, last = [], []
    print(f"[calibrate] DAT: training the discriminators alone for {n_steps} batches on the frozen "
          f"model (backbone unchanged), so DAT is measured against a trained adversary", flush=True)
    try:
        for step in range(n_steps):
            batch = next_batch()
            core.loss_terms = {}
            if autocast is not None:
                with autocast:
                    out = core(batch, use_cell_embedding=True, apply_dat=True)
            else:
                out = core(batch, use_cell_embedding=True, apply_dat=True)
            terms, core.loss_terms = core.loss_terms, None
            term = terms.get("dat")
            if term is None or not term.requires_grad:
                continue
            opt.zero_grad(set_to_none=True)
            term.float().backward()
            opt.step()
            conf_key = conf_key or next((k for k in out if k.endswith("_confidence")), None)
            if conf_key:
                (first if step < 10 else last).append(float(out[conf_key]))
                last[:] = last[-10:]
            if (step + 1) % 25 == 0 or step + 1 == n_steps:
                print(f"[calibrate] DAT discriminator {step + 1}/{n_steps}: loss {float(term.detach()):.4f}"
                      + (f", {conf_key} {float(out[conf_key]):.3f}" if conf_key else ""), flush=True)
    finally:
        for p in frozen:
            p.requires_grad_(True)
        core.zero_grad(set_to_none=True)
    if first and last:
        print(f"[calibrate] DAT: {conf_key} {sum(first) / len(first):.3f} (first 10) -> "
              f"{sum(last) / len(last):.3f} (last 10); 0.5 = cannot tell the domains apart", flush=True)


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
              "paired": core.paired_alignment, "agg": core.aggregation,
              "paired_alt": core.paired_alignment, "agg_alt": core.aggregation,
              "dat": bool(getattr(core, "do_dat", False))}
    paired_forms = {"trained": core.paired_loss,
                    "alt": "mse" if core.paired_loss == "infonce" else "infonce"}
    agg_form = getattr(core, "agg_loss", "mse")
    agg_forms = {"trained": agg_form, "alt": "mse" if agg_form == "infonce" else "infonce"}
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
    print(f"[calibrate] memory settings: attn_impl={getattr(args, 'attn_impl', None)}, "
          f"grad_checkpoint={getattr(args, 'grad_checkpoint', None)}, batch_size={args.batch_size} "
          f"(bulk_ratio {getattr(args, 'bulk_ratio', None)}, pb_ratio {getattr(args, 'pb_ratio', None)}), "
          f"n_sc_per_pseudobulk={getattr(args, 'n_sc_per_pseudobulk', None)}")
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
    reported = {False: False, True: False}  # first normal / first paired batch: memory line
    started = time.perf_counter()
    loader = iter(datamodule.train_dataloader())

    def next_batch():
        nonlocal loader
        try:
            b = next(loader)
        except StopIteration:
            loader = iter(datamodule.train_dataloader())
            b = next(loader)
        return _to_device(b, device)

    if active["dat"]:
        _train_discriminators(core, next_batch, int(getattr(args, "calibrate_dat_steps", 100)), autocast)

    for step in range(steps):
        batch = next_batch()
        paired = batch.get("is_paired_batch", False)
        paired = bool(paired.item()) if isinstance(paired, torch.Tensor) else bool(paired)

        n_rows = _batch_rows(batch)
        if device.type == "cuda" and not reported[paired]:
            torch.cuda.reset_peak_memory_stats(device)
        try:
            core.loss_terms = {}
            if autocast is not None:
                with autocast:
                    core(batch, use_cell_embedding=True, apply_dat=active["dat"])
            else:
                core(batch, use_cell_embedding=True, apply_dat=active["dat"])
            terms, core.loss_terms = core.loss_terms, None

            recon = w_rec * (terms.get("expr", 0.0) + w_mvc * terms.get("mvc", 0.0) + terms.get("gen", 0.0))
            rec: dict[str, Any] = {"paired": paired, "rows": n_rows, "recon_grad": _grad_norm(recon, backbone),
                                   "terms": {}}
            for name, value in terms.items():
                if not (isinstance(value, torch.Tensor) and value.requires_grad):
                    continue  # a skipped term returns a constant zero
                if name in ALIGN_TERMS and float(value.detach()) == 0.0:
                    continue  # no anchor / class in this batch: the term did not act
                rec["terms"][name] = {"loss": float(value.detach().float()), "grad": _grad_norm(value, backbone)}
        except torch.OutOfMemoryError as exc:
            raise torch.OutOfMemoryError(
                f"[calibrate] out of GPU memory on a {'paired' if paired else 'normal'} batch of {n_rows} rows "
                f"(batch_size {args.batch_size}, n_sc_per_pseudobulk {getattr(args, 'n_sc_per_pseudobulk', None)}, "
                f"attn_impl {getattr(args, 'attn_impl', None)}, grad_checkpoint "
                f"{getattr(args, 'grad_checkpoint', None)}). Lower n_sc_per_pseudobulk, then batch_size "
                f"(keep the ratios); the training runs use the same batches. {exc}") from exc
        records.append(rec)
        del terms, recon
        model.zero_grad(set_to_none=True)

        if device.type == "cuda" and not reported[paired]:
            reported[paired] = True
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            total = torch.cuda.get_device_properties(device).total_memory / 2**30
            print(f"[calibrate] first {'paired' if paired else 'normal'} batch: {n_rows} rows; peak GPU memory "
                  f"{peak:.1f} of {total:.1f} GiB (forward + every gradient of the probe; training needs "
                  f"one backward, so a little less)", flush=True)

        if (step + 1) % 25 == 0 or step == 0 or step + 1 == steps:
            seen = {t: sum(t in r["terms"] for r in records) for t in ALIGN_TERMS if active[t]}
            ratios = {t: np.median([r["terms"][t]["grad"] / r["recon_grad"] for r in records
                                    if t in r["terms"] and r["recon_grad"] > 0])
                      for t, n in seen.items() if n}
            print(f"[calibrate] batch {step + 1}/{steps} ({time.perf_counter() - started:.0f} s): "
                  f"batches per term {seen}; running median ratio "
                  + ", ".join(f"{t}={v:.3g}" for t, v in ratios.items()), flush=True)

    result = suggest_weights(records, weights, R)
    result["paired_forms"] = paired_forms
    result["agg_forms"] = agg_forms
    print(format_report(result))
    snippet = config_snippet(result)
    print("[calibrate] paste into base_config.features of the launch config:\n" + snippet)
    for alt_term in ALT_TERMS:
        line = forms_line(result, alt_term)
        if line:
            print(line)
    print("[calibrate] the share is measured at the parent, where the alignment losses are largest: "
          "as they shrink in training, so does their pull, so R is a starting share.", flush=True)

    result.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": steps, "weights_at_probe": weights, "reconstruction": {"weight": w_rec, "weight_mvc": w_mvc},
        "snippet": json.loads(snippet), "paired_alt_weight": result["terms"]["paired_alt"]["suggested_weight"],
        "agg_alt_weight": result["terms"]["agg_alt"]["suggested_weight"],
        "init_from": getattr(args, "init_from", None),
        "paired_every_n": getattr(args, "paired_every_n", None),
    })
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[calibrate] wrote {out_path}")
    return out_path
