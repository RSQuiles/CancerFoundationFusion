"""Self-check of the DAT ramp (``--dat-ramp-steps``) and the flex recompile guard.

    python scripts/check_dat_ramp.py

Needs torch (the container). Checks that a ramped discriminator returns the same logits,
sends ``-dat_scale * rev_factor`` times the plain gradient back to the encoder, leaves
the discriminator's own gradient unchanged, and that the ramp factor runs 0 -> 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAILED: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILED.append(name)


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch not available; skipped")
        return
    from types import SimpleNamespace

    from cancerfoundation.model import layers  # noqa: F401  (applies the recompile guard)
    from cancerfoundation.model.model import CancerFoundation
    from cancerfoundation.model.module import AdversarialDiscriminator

    import torch._dynamo.config as dcfg

    limit = getattr(dcfg, "recompile_limit", getattr(dcfg, "cache_size_limit", None))
    check("dynamo recompile limit raised to >= 256", limit is None or limit >= 256)

    torch.manual_seed(0)
    disc = AdversarialDiscriminator(16, n_cls=2, scale=0.5, no_invert_dat=False)
    x = torch.randn(8, 16)

    def grads(ramped: bool, factor: float):
        disc.ramped = ramped
        disc.rev_factor.fill_(factor)
        disc.zero_grad()
        xi = x.clone().requires_grad_(True)
        out = disc(xi)
        out.square().sum().backward()
        return out.detach(), xi.grad.clone(), disc.out_layer.weight.grad.clone()

    out0, gx0, gw0 = grads(False, 1.0)
    for f in (1.0, 0.3, 0.0):
        out, gx, gw = grads(True, f)
        check(f"factor {f}: same logits (exactly)", torch.equal(out, out0))
        check(f"factor {f}: encoder gradient = factor x plain", torch.allclose(gx, f * gx0, atol=1e-6))
        check(f"factor {f}: discriminator gradient unchanged", torch.allclose(gw, gw0))
    check("rev_factor is not saved in the state_dict", "rev_factor" not in disc.state_dict())

    fake = SimpleNamespace(dat_ramp_steps=1000, dat_start_step=0)
    ramp = [CancerFoundation.dat_ramp_factor(fake, s) for s in (0, 499, 999, 5000)]
    check("ramp 0 -> 1 over dat_ramp_steps", ramp[0] == 0.001 and ramp[1] == 0.5 and ramp[2] == ramp[3] == 1.0)
    fake.dat_ramp_steps = 0
    check("no ramp: factor 1", CancerFoundation.dat_ramp_factor(fake, 0) == 1.0)

    print("\nAll checks passed." if not FAILED else f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
