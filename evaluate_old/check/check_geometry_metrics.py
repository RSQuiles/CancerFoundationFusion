"""Self-check for the scale-free contrastive metrics and the geometry family.

    python evaluate/check/check_geometry_metrics.py

Runs offline on synthetic clouds whose geometry is known in closed form — no
eval.h5ad, checkpoint, GPU or cluster. ``anndata`` and ``torch`` are stubbed because
``unified_metrics`` imports them at module level while none of the functions checked
here touch either.

Why this exists: these metrics are easy to get backwards. A ratio of participation
ratios looks like it should rise with a batch effect and in fact falls, because the
between-modality offset contributes one dominant eigenvalue and so *concentrates* the
spectrum. Every assertion below pins a value that theory predicts independently, so a
sign error cannot pass.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.no_grad = lambda: (lambda f: f)
    _torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None
    )
    sys.modules["torch"] = _torch

if "anndata" not in sys.modules:
    _ad = types.ModuleType("anndata")
    _ad.AnnData = object
    sys.modules["anndata"] = _ad

import numpy as np  # noqa: E402

from evaluate.check.unified_metrics import (  # noqa: E402
    _modality_variance_fraction,
    _participation_ratio,
    _pr_isotropic_ref,
    compute_contrastive_metrics,
    compute_geometry_metrics,
)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f'  [{detail}]' if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    rng = np.random.default_rng(0)
    d = 64

    # ── participation ratio ────────────────────────────────────────────────
    print("\nparticipation ratio")
    # k equal eigenvalues and the rest zero must give exactly k.
    for k in (1, 3, 17):
        basis = np.linalg.qr(rng.standard_normal((d, k)))[0]
        pr = _participation_ratio(rng.standard_normal((4000, k)) @ basis.T)
        check(f"rank-{k} cloud scores ~{k}", abs(pr - k) < 0.15 * k, f"PR={pr:.3f}")

    X = rng.standard_normal((2000, d))
    check("scale-invariant",
          abs(_participation_ratio(X) - _participation_ratio(1000 * X)) < 1e-6)

    # An uncentred metric would collapse to ~1 here; PR must not notice the offset.
    off = X + 500.0
    check("offset from the origin does not look like collapse",
          abs(_participation_ratio(X) - _participation_ratio(off)) < 1e-3,
          f"PR={_participation_ratio(off):.2f}")

    # The Marchenko-Pastur reference must track isotropic data even at gamma > 1.
    for n, dd in ((5000, 64), (500, 512), (2000, 512)):
        pr = _participation_ratio(rng.standard_normal((n, dd)))
        ref = _pr_isotropic_ref(n, dd)
        check(f"isotropic n={n} d={dd} matches reference",
              abs(pr / ref - 1.0) < 0.02, f"PR={pr:.1f} ref={ref:.1f}")

    check("degenerate inputs give NaN, not a fake number",
          np.isnan(_participation_ratio(np.zeros((5, 4))))
          and np.isnan(_participation_ratio(np.ones((1, 4))))
          and np.isnan(_pr_isotropic_ref(1, 10)))

    # ── energy distance ────────────────────────────────────────────────────
    print("\nnormalised energy distance")
    A, B = rng.standard_normal((900, d)), rng.standard_normal((900, d))
    e_same = compute_contrastive_metrics(A, B, seed=0)["contrastive_energy_distance"]
    check("~0 for two samples of one distribution", abs(e_same) < 0.01, f"e={e_same:.5f}")

    # Closed form for equal-spread clouds whose centroids sit r within-cloud
    # distances apart: e = 1 - 1/sqrt(1 + r^2).
    for r in (0.5, 1.0, 2.0):
        mu = np.zeros(d)
        mu[0] = r * np.sqrt(2 * d)
        e = compute_contrastive_metrics(
            rng.standard_normal((1500, d)), rng.standard_normal((1500, d)) + mu, n_max=1500, seed=0
        )["contrastive_energy_distance"]
        want = 1 - 1 / np.sqrt(1 + r * r)
        check(f"r={r} matches closed form", abs(e - want) < 0.02, f"e={e:.4f} want={want:.4f}")

    m = compute_contrastive_metrics(
        rng.standard_normal((300, d)) * 3.0, rng.standard_normal((400, d)) + 1.2, seed=0
    )
    identity = 1 - (m["contrastive_within_bulk_over_cross_l2"]
                    + m["contrastive_within_pb_over_cross_l2"]) / 2
    check("e == 1 - (ratio_bulk + ratio_pb)/2",
          abs(m["contrastive_energy_distance"] - identity) < 1e-12)

    # The whole point: the scale-free numbers must ignore a global rescaling that
    # moves cross_l2 by three orders of magnitude.
    A = rng.standard_normal((900, d)) * 1e-3
    B = rng.standard_normal((900, d)) * 1e-3 + np.eye(d)[0] * 8e-3
    small = compute_contrastive_metrics(A, B, seed=0)
    big = compute_contrastive_metrics(A * 1000, B * 1000, seed=0)
    check("energy distance survives a x1000 rescale",
          abs(small["contrastive_energy_distance"] - big["contrastive_energy_distance"]) < 1e-9)
    check("raw cross_l2 does not (which is why the ratios exist)",
          big["contrastive_cross_l2_mean"] > 100 * small["contrastive_cross_l2_mean"])
    check("a tiny cross_l2 still reports separation",
          small["contrastive_cross_l2_mean"] < 0.05
          and small["contrastive_energy_distance"] > 0.1,
          f"cross_l2={small['contrastive_cross_l2_mean']:.5f} "
          f"e={small['contrastive_energy_distance']:.4f}")

    # A clump inside a diffuse cloud pushes one ratio above 1. That is meaningful,
    # not a bug, so it must not be clipped away.
    nested = compute_contrastive_metrics(
        rng.standard_normal((900, d)), rng.standard_normal((900, d)) * 0.02, seed=0
    )
    check("nested clouds give ratio > 1 on the wider modality",
          nested["contrastive_within_bulk_over_cross_l2"] > 1.2
          and nested["contrastive_within_pb_over_cross_l2"] < 0.1,
          f"rb={nested['contrastive_within_bulk_over_cross_l2']:.3f} "
          f"rp={nested['contrastive_within_pb_over_cross_l2']:.3f}")

    coincident = np.ones((50, d))
    check("coincident clouds give NaN, never 'perfectly aligned' 0",
          np.isnan(compute_contrastive_metrics(
              coincident, coincident.copy(), seed=0)["contrastive_energy_distance"]))

    # ── modality variance share ────────────────────────────────────────────
    print("\nmodality variance share")
    A, B = rng.standard_normal((900, d)), rng.standard_normal((900, d))
    check("~0 when the modalities coincide",
          _modality_variance_fraction(A, B) < 0.05,
          f"{_modality_variance_fraction(A, B):.4f}")
    far = _modality_variance_fraction(A, B + np.eye(d)[0] * 40)
    check("~1 when the offset dominates", far > 0.8, f"{far:.4f}")
    check("monotone in the offset", all(
        _modality_variance_fraction(A, B + np.eye(d)[0] * lo)
        < _modality_variance_fraction(A, B + np.eye(d)[0] * hi)
        for lo, hi in ((0.5, 2.0), (2.0, 8.0), (8.0, 30.0))))
    check("identical centroids give exactly 0",
          _modality_variance_fraction(A, A.copy()) == 0.0)

    # ── the 2x2 the two families exist to separate ─────────────────────────
    print("\ncollapse vs mixing (the case the contrastive family alone cannot call)")
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    A = np.outer(rng.standard_normal(900), u)
    B = np.outer(rng.standard_normal(900), u) + 0.01 * u
    g = compute_geometry_metrics({"bulk": A, "pb": B}, seed=0)
    e = compute_contrastive_metrics(A, B, seed=0)["contrastive_energy_distance"]
    check("collapsed embedding: energy distance says 'mixed'", abs(e) < 0.02, f"e={e:.4f}")
    check("collapsed embedding: PR exposes it", g["geometry_pr_pooled"] < 1.05,
          f"PR={g['geometry_pr_pooled']:.3f}")
    check("collapsed embedding: PR fraction is near zero",
          g["geometry_pr_frac_pooled"] < 0.05, f"frac={g['geometry_pr_frac_pooled']:.4f}")

    healthy = compute_geometry_metrics(
        {"bulk": rng.standard_normal((900, d)), "pb": rng.standard_normal((900, d))}, seed=0
    )
    check("healthy embedding: PR fraction near 1",
          healthy["geometry_pr_frac_pooled"] > 0.9,
          f"frac={healthy['geometry_pr_frac_pooled']:.4f}")

    # ── shape of the emitted dict ──────────────────────────────────────────
    print("\nemitted keys")
    check("single-modality data still gets geometry",
          "geometry_pr_sc" in compute_geometry_metrics({"sc": rng.standard_normal((300, d))}))
    check("modality share needs two modalities",
          "geometry_modality_var_frac"
          not in compute_geometry_metrics({"sc": rng.standard_normal((300, d))}))
    check("row cap is applied and recorded",
          compute_geometry_metrics(
              {"sc": rng.standard_normal((900, d))}, max_rows=100
          )["geometry_pr_n_sc"] == 100)

    import json
    payload = {**compute_contrastive_metrics(A, B, seed=0),
               **compute_geometry_metrics({"bulk": A, "pb": B}, seed=0)}
    json.dumps(payload)   # raises if a numpy scalar slipped through
    check("everything is JSON-serialisable",
          not [k for k, v in payload.items() if isinstance(v, np.generic)])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("All geometry / energy metric checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
