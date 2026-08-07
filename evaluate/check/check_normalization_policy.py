"""Exercise the downstream normalization policy and its orchestrator wiring.

Run directly:  python evaluate/check/check_normalization_policy.py

Stubs torch so ``looks_like_counts`` is reachable, then asserts the three things the
``--normalize`` switch promises:

  * --no-normalize  -> nothing is applied, anywhere, and every embedder call is a
    pass-through (``normalized=True, log1p_only=False``);
  * --normalize     -> CP10K+log1p applied exactly once to counts-looking input;
  * --normalize on already-log1p input -> the safeguard reports and skips, so the
    matrix is never log1p'd twice.

Plus the resolution order (CLI > config > off), the obsolete-key report, and that
``run_analysis.py`` turns ``downstream.normalize`` into the right CLI flag.

Needs only numpy + PyYAML. No cluster, GPU, checkpoint or training stack.
"""

import logging
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- stub torch so cancerfoundation.data.preprocess imports ----------------
# looks_like_counts only uses torch for an isinstance() check; the real package
# __init__ would drag in the whole model stack, so load the module by path.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

for pkg in ("cancerfoundation", "cancerfoundation.data"):
    if pkg not in sys.modules:
        sys.modules[pkg] = types.ModuleType(pkg)

if "cancerfoundation.data.preprocess" not in sys.modules:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cancerfoundation.data.preprocess",
        ROOT / "cancerfoundation" / "data" / "preprocess.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cancerfoundation.data.preprocess"] = module
    spec.loader.exec_module(module)

from evaluate.finetune.normalization import (  # noqa: E402
    CANONICAL_KEY,
    SOURCE_KEY,
    NormalizationPolicy,
    drain_provenance,
    resolve_policy,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}{('  -- ' + detail) if detail else ''}")


class FakeAnnData:
    """Minimal stand-in: apply() only touches .X and .copy()."""

    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.float32)

    def copy(self):
        return FakeAnnData(self.X.copy())


class Cfg:
    """Stand-in for an OmegaConf task config; getattr is all resolve_policy uses."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


# Counts: integral and max > 20, so looks_like_counts says True.
COUNTS = np.array([[100.0, 300.0, 0.0], [200.0, 400.0, 400.0]], dtype=np.float32)
# The same matrix after CP10K + log1p — what a pre-normalized input looks like.
_ROWSUM = COUNTS.sum(axis=1, keepdims=True)
LOG1P = np.log1p(COUNTS / _ROWSUM * 1e4).astype(np.float32)


def expected_cp10k_log1p(X: np.ndarray) -> np.ndarray:
    rows = X.sum(axis=1, keepdims=True)
    return np.log1p(X / np.where(rows == 0, 1.0, rows) * 1e4).astype(np.float32)


# ---------------------------------------------------------------------------

def check_disabled() -> None:
    print("\n=== --no-normalize: nothing applied, anywhere ===")
    drain_provenance()
    policy = NormalizationPolicy(normalize=False, source="cli")

    out, prov = policy.apply(FakeAnnData(COUNTS), task="t")
    check("counts pass through untouched", np.array_equal(out.X, COUNTS))
    check("provenance says not applied", prov["applied"] is False)

    out, _ = policy.apply(FakeAnnData(LOG1P), task="t")
    check("log1p passes through untouched", np.allclose(out.X, LOG1P))

    check(
        "embed kwargs are a pass-through for both embedders",
        policy.embed_kwargs() == {"normalized": True, "log1p_only": False},
        str(policy.embed_kwargs()),
    )


def check_enabled_on_counts() -> None:
    print("\n=== --normalize on counts: CP10K+log1p, exactly once ===")
    drain_provenance()
    policy = NormalizationPolicy(normalize=True, source="cli")

    source = FakeAnnData(COUNTS)
    out, prov = policy.apply(source, task="t")

    check("applied", prov["applied"] is True)
    check("matches CP10K+log1p", np.allclose(out.X, expected_cp10k_log1p(COUNTS), atol=1e-5))
    check("max drops out of counts range", out.X.max() < 12.0, f"max={out.X.max():.3f}")
    check("input was not mutated in place", np.array_equal(source.X, COUNTS))
    check("zeros stay zero", out.X[0, 2] == 0.0)

    # Applying twice must not compound: the second pass hits the safeguard.
    twice, prov2 = policy.apply(out, task="t")
    check("second application is refused", prov2["applied"] is False)
    check("matrix unchanged by the second pass", np.allclose(twice.X, out.X))

    check(
        "kwargs are a pass-through, since apply() already normalized",
        policy.embed_kwargs() == {"normalized": True, "log1p_only": False},
    )


def check_safeguard() -> None:
    print("\n=== --normalize on already-log1p input: safeguard ===")
    drain_provenance()
    policy = NormalizationPolicy(normalize=True, source="cli")

    out, prov = policy.apply(FakeAnnData(LOG1P), task="t")
    check("skipped", prov["applied"] is False)
    check("reason names the safeguard", "safeguard" in prov["reason"], prov["reason"])
    check("looks_like_counts recorded False", prov["looks_like_counts"] is False)
    check("matrix untouched", np.allclose(out.X, LOG1P))


def check_resolution_order() -> None:
    print("\n=== resolution: CLI > config > off ===")
    check(
        "no config, no CLI -> off",
        resolve_policy(None, None) == NormalizationPolicy(False, "default"),
    )
    check(
        "empty config -> off",
        resolve_policy(Cfg(), None) == NormalizationPolicy(False, "default"),
    )
    check(
        "config true is honoured",
        resolve_policy(Cfg(**{CANONICAL_KEY: True}), None)
        == NormalizationPolicy(True, "config"),
    )
    check(
        "CLI False overrides config True",
        resolve_policy(Cfg(**{CANONICAL_KEY: True}), False)
        == NormalizationPolicy(False, "cli"),
    )
    check(
        "CLI True overrides config False",
        resolve_policy(Cfg(**{CANONICAL_KEY: False}), True)
        == NormalizationPolicy(True, "cli"),
    )
    check(
        "folded-in CLI override still reports source=cli",
        resolve_policy(Cfg(**{CANONICAL_KEY: True, SOURCE_KEY: "cli"}), None).source
        == "cli",
    )


def check_obsolete_keys_reported() -> None:
    print("\n=== obsolete keys are reported, not silently honoured ===")
    log = logging.getLogger("evaluate.finetune.normalization")

    for key, value in (("normalized", False), ("normalize_input", True)):
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())  # noqa: B023
        log.addHandler(handler)
        try:
            policy = resolve_policy(Cfg(**{key: value}), None)
        finally:
            log.removeHandler(handler)

        check(f"'{key}' warns", any(key in m and "obsolete" in m for m in records))
        check(
            f"'{key}' does not change the outcome",
            policy == NormalizationPolicy(False, "default"),
            str(policy),
        )


def check_orchestrator_flag() -> None:
    """downstream.normalize -> the right run_ablation_downstream.py flag."""
    print("\n=== run_analysis.py: downstream.normalize -> CLI flag ===")
    import json
    import tempfile

    from evaluate.analysis_config import DEFAULT_STEPS, STEP_ORDER, load_analysis_config
    from evaluate.run_analysis import STEP_ENV, plan_experiment

    check("'downstream' is a known step", "downstream" in STEP_ORDER)
    check("'downstream' is opt-in", "downstream" not in DEFAULT_STEPS)
    check("'downstream' runs in the container", STEP_ENV["downstream"] == "container")
    check(
        "'downstream' is ordered before 'benchmark'",
        STEP_ORDER.index("downstream") < STEP_ORDER.index("benchmark"),
    )

    cases = {True: "--normalize", False: "--no-normalize", None: None}
    for value, expected in cases.items():
        cfg_dict = {
            "defaults": {"steps": ["downstream"]},
            "experiments": [{
                "name": "e",
                "ablation_dir": "/work/abl",
                "downstream": {"tasks": ["canc_type_class"], "normalize": value},
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg_dict), encoding="utf-8")
            cfg = load_analysis_config(path)
            steps, _ = plan_experiment(cfg.experiments[0], cfg.repo_root)

        argv = next(s.argv for s in steps if s.name == "downstream")
        if expected is None:
            check(
                "normalize: null passes neither flag",
                "--normalize" not in argv and "--no-normalize" not in argv,
            )
        else:
            check(f"normalize: {value} passes {expected}", expected in argv, str(argv))
        check(
            f"normalize: {value} still passes --tasks",
            "--tasks" in argv and "canc_type_class" in argv,
        )

    # An enabled step with no tasks must fail loudly rather than run empty.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        path.write_text(json.dumps({
            "defaults": {"steps": ["downstream"]},
            "experiments": [{"name": "e", "ablation_dir": "/work/abl"}],
        }), encoding="utf-8")
        try:
            load_analysis_config(path)
            raised = False
        except KeyError as exc:
            raised = "downstream.tasks" in str(exc)
    check("empty downstream.tasks is rejected", raised)


def check_task_configs_migrated() -> None:
    """Every shipped task config carries the canonical key and no retired one."""
    print("\n=== shipped task configs use the canonical key ===")
    import yaml

    config_dir = ROOT / "evaluate" / "finetune" / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        section = raw.get("finetune") or {}
        task_key = next(iter(section), None)
        task_cfg = section.get(task_key) or {}

        check(f"{path.name} defines '{CANONICAL_KEY}'", CANONICAL_KEY in task_cfg)
        check(
            f"{path.name} drops the retired keys",
            "normalized" not in task_cfg and "normalize_input" not in task_cfg,
            str([k for k in ("normalized", "normalize_input") if k in task_cfg]),
        )


def main() -> int:
    # WARNING-only: the safeguard check asserts on a warning being emitted, and the
    # INFO lines would drown the check output.
    logging.basicConfig(level=logging.WARNING, format="        %(levelname)s %(message)s")

    check_disabled()
    check_enabled_on_counts()
    check_safeguard()
    check_resolution_order()
    check_obsolete_keys_reported()
    check_orchestrator_flag()
    check_task_configs_migrated()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("ALL NORMALIZATION-POLICY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
