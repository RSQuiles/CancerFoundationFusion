"""Verify the --grad-checkpoint and --attn-impl flags, and measure the memory ceiling.

Run inside the training container (needs torch + a GPU for the memory sweep):

    python scripts/check_attention_memory.py                # equivalence checks only
    python scripts/check_attention_memory.py --sweep        # + batch-size memory sweep

Checks
------
1. ``--grad-checkpoint`` is numerically identical to the baseline (forward and
   gradients), because ``use_reentrant=False`` preserves RNG state across the
   recompute.
2. ``--attn-impl flex`` reproduces the ``mha`` attention topology on the same
   weights, so checkpoints are interchangeable between the two.
3. Peak allocated memory as a function of batch size, for each combination.

Note on check 2: run it with ``dropout=0``. ``nn.MultiheadAttention`` applies
dropout to the attention weights, which FlexAttention has no equivalent for, so
the two paths only agree in eval mode or at zero attention dropout.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancerfoundation.model.layers import CFGenerator, CFLayer  # noqa: E402

D_MODEL = 128
NHEAD = 8
D_HID = 256
NLAYERS = 6


def build(grad_checkpoint: bool, attn_impl: str, dropout: float, device, seed: int = 0):
    torch.manual_seed(seed)
    layer = CFLayer(
        D_MODEL, NHEAD, D_HID, dropout, batch_first=True, norm_scheme="post"
    )
    return CFGenerator(
        encoder_layer=layer,
        num_layers=NLAYERS,
        grad_checkpoint=grad_checkpoint,
        attn_impl=attn_impl,
    ).to(device)


def make_batch(batch, pcpt_len, gen_len, device, pad: bool = True):
    torch.manual_seed(1234)
    pcpt = torch.randn(batch, pcpt_len, D_MODEL, device=device, requires_grad=True)
    gen = torch.randn(batch, gen_len, D_MODEL, device=device, requires_grad=True)
    kpm = torch.zeros(batch, pcpt_len + gen_len, dtype=torch.bool, device=device)
    if pad:
        # Pad the tail of a few rows, as the collator does for short cells.
        kpm[0, -5:] = True
        kpm[min(1, batch - 1), -12:] = True
    return pcpt, gen, kpm


def run_once(model, pcpt, gen, kpm, seed: int = 7):
    torch.manual_seed(seed)
    out_p, out_g = model(pcpt, gen, src_key_padding_mask=kpm)
    loss = out_p.square().mean() + out_g.square().mean()
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return out_p.detach(), out_g.detach(), grads.detach()


def check_grad_checkpoint(device, dropout: float):
    print(f"\n[1] grad-checkpoint equivalence (dropout={dropout})")
    pcpt, gen, kpm = make_batch(4, 64, 32, device)

    base = build(False, "mha", dropout, device)
    ckpt = build(True, "mha", dropout, device)
    ckpt.load_state_dict(base.state_dict())
    base.train()
    ckpt.train()

    op_a, og_a, g_a = run_once(base, pcpt, gen, kpm)
    pcpt.grad = gen.grad = None
    op_b, og_b, g_b = run_once(ckpt, pcpt, gen, kpm)

    for name, a, b in [("pcpt out", op_a, op_b), ("gen out", og_a, og_b), ("grads", g_a, g_b)]:
        delta = (a - b).abs().max().item()
        print(f"    max|delta| {name:9s} = {delta:.3e}  {'OK' if delta < 1e-5 else 'MISMATCH'}")


def check_flex(device):
    print("\n[2] mha vs flex on identical weights (dropout=0)")
    pcpt, gen, kpm = make_batch(4, 64, 32, device)

    mha = build(False, "mha", 0.0, device).eval()
    flex = build(False, "flex", 0.0, device).eval()
    flex.load_state_dict(mha.state_dict())

    with torch.no_grad():
        p_a, g_a = mha(pcpt, gen, src_key_padding_mask=kpm)
        p_b, g_b = flex(pcpt, gen, src_key_padding_mask=kpm)

    # Compare only rows that are not key-padded: for a padded *query* the two
    # paths legitimately differ (flex keeps the diagonal so the softmax has at
    # least one key and cannot produce NaN), and those rows are dropped by
    # positions_to_match in TransformerModule.forward.
    valid = ~kpm
    vp, vg = valid[:, : pcpt.shape[1]], valid[:, pcpt.shape[1] :]
    dp = (p_a[vp] - p_b[vp]).abs().max().item()
    dg = (g_a[vg] - g_b[vg]).abs().max().item()
    print(f"    max|delta| pcpt out = {dp:.3e}  {'OK' if dp < 1e-3 else 'MISMATCH'}")
    print(f"    max|delta| gen out  = {dg:.3e}  {'OK' if dg < 1e-3 else 'MISMATCH'}")
    print("    (tolerance is loose: different reduction order, not different math)")


def sweep(device, pcpt_len: int, gen_len: int):
    print(f"\n[3] peak memory sweep (L={pcpt_len + gen_len}, {NLAYERS} layers, {NHEAD} heads)")
    print(f"    {'rows':>6} {'baseline':>12} {'+ckpt':>12} {'+flex':>12} {'+both':>12}")
    for rows in (32, 64, 128, 172, 256, 354):
        cells = []
        for gc, impl in ((False, "mha"), (True, "mha"), (False, "flex"), (True, "flex")):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = build(gc, impl, 0.0, device).train()
                pcpt, gen, kpm = make_batch(rows, pcpt_len, gen_len, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_p, out_g = model(pcpt, gen, src_key_padding_mask=kpm)
                    (out_p.float().square().mean() + out_g.float().square().mean()).backward()
                cells.append(f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
            except torch.cuda.OutOfMemoryError:
                cells.append("OOM")
            finally:
                del model
                torch.cuda.empty_cache()
        print(f"    {rows:>6} " + " ".join(f"{c:>12}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also run the memory sweep (needs a GPU)")
    ap.add_argument("--pcpt-len", type=int, default=450)
    ap.add_argument("--gen-len", type=int, default=150)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} on {device}")

    check_grad_checkpoint(device, args.dropout)
    if device.type == "cuda":
        check_flex(device)
    else:
        print("\n[2] skipped: FlexAttention requires CUDA")

    if args.sweep:
        if device.type != "cuda":
            print("\n[3] skipped: --sweep requires CUDA")
        else:
            sweep(device, args.pcpt_len, args.gen_len)


if __name__ == "__main__":
    main()
