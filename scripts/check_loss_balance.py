"""Self-check of the loss-weight calibration arithmetic and of launch-config ``base_from``.

Offline, numpy only (no torch): ``utils.loss_balance.suggest_weights`` on synthetic probe
records, and ``experiments.config.load_launch_config`` with a config inheriting its base.

    python scripts/check_loss_balance.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises  # noqa: E402
from utils.loss_balance import config_snippet, format_report, suggest_weights  # noqa: E402


def _records(n: int = 100, paired_every: int = 10, seed: int = 0) -> list[dict]:
    """Batches where each term's gradient is a fixed multiple of reconstruction's (+ noise)."""
    rng = np.random.default_rng(seed)
    true_ratio = {"mmd": 0.02, "contrastive": 1.5, "cdd": 0.1, "agg": 0.6, "paired": 3.0}
    out = []
    for i in range(n):
        g = float(rng.uniform(1, 2))
        paired = i % paired_every == 0
        terms = {"expr": {"loss": 1.0, "grad": g}, "mvc": {"loss": 1.2, "grad": 0.5 * g}}
        names = ["paired"] if paired else ["mmd", "contrastive", "cdd", "agg"]
        for t in names:
            terms[t] = {"loss": 0.5, "grad": g * true_ratio[t] * float(rng.uniform(0.95, 1.05))}
        out.append({"paired": paired, "recon_grad": g, "terms": terms})
    return out, true_ratio


def check_suggest() -> None:
    print("-- suggest_weights --")
    records, true_ratio = _records()
    weights = {"mmd": 1.0, "contrastive": 1.0, "cdd": 0.3, "paired": 1.0, "agg": 1.0}
    res = suggest_weights(records, weights, target_share=0.3)
    t = res["terms"]
    check("each term reaches the target share at its suggested weight",
          all(abs(t[k]["suggested_weight"] * t[k]["ratio"]["median"] - 0.3) < 0.01 for k in true_ratio))
    check("weight = R / ratio (mmd small ratio -> large weight)",
          t["mmd"]["suggested_weight"] > 10 and t["contrastive"]["suggested_weight"] < 1)
    check("paired counted on its own batches only", t["paired"]["batches"] == 10
          and res["n_paired_batches"] == 10 and t["mmd"]["batches"] == 90)
    check("share at the current weight reported", abs(t["cdd"]["share_at_current"] - 0.3 * 0.1) < 0.005)
    check("reconstruction terms carry no weight", "suggested_weight" not in t["expr"])
    check("paired with 10 batches is not flagged as too few", not any("only" in f for f in t["paired"]["flags"]))

    few = suggest_weights(records[:30], weights, 0.3)["terms"]["paired"]
    check("a term in < 10 batches is flagged unreliable", any("unreliable" in f for f in few["flags"]))

    no_agg = [{**r, "terms": {k: v for k, v in r["terms"].items() if k != "agg"}} for r in records]
    a = suggest_weights(no_agg, weights, 0.3)["terms"]["agg"]
    check("a term never computed keeps its weight and says so",
          a["suggested_weight"] == 1.0 and a["batches"] == 0 and "never computed" in a["flags"][0])

    noisy, _ = _records(seed=1)
    rng = np.random.default_rng(2)
    for r in noisy:
        if "mmd" in r["terms"]:
            r["terms"]["mmd"]["grad"] *= float(rng.lognormal(0, 1.5))
    m = suggest_weights(noisy, weights, 0.3)["terms"]["mmd"]
    check("a ratio varying across batches is flagged", any("variable" in f for f in m["flags"]))

    from utils.loss_balance import paired_forms_line

    alt_records = [{**r, "terms": {**r["terms"], **({"paired_alt": {"loss": 0.1, "grad": r["recon_grad"] * 0.03}}
                                                    if r["paired"] else {})}} for r in records]
    alt = suggest_weights(alt_records, {**weights, "paired_alt": 1.0}, 0.3)
    alt["paired_forms"] = {"trained": "infonce", "alt": "mse"}
    check("other paired form calibrated on the same batches, kept out of the shared snippet",
          abs(alt["terms"]["paired_alt"]["suggested_weight"] - 10) < 0.6
          and "paired_alt" not in config_snippet(alt) and json.loads(config_snippet(alt))["loss_weight_paired"]
          == alt["terms"]["paired"]["suggested_weight"])
    line = paired_forms_line(alt) or ""
    check("paired forms line names both forms and weights", "infonce" in line and "mse" in line and "override" in line)

    from utils.loss_balance import forms_line

    agg_records = [{**r, "terms": {**r["terms"], **({"agg_alt": {"loss": 3.0, "grad": r["recon_grad"] * 0.06}}
                                                    if "agg" in r["terms"] else {})}} for r in records]
    aa = suggest_weights(agg_records, {**weights, "agg_alt": 1.0}, 0.3)
    aa["agg_forms"] = {"trained": "mse", "alt": "infonce"}
    aline = forms_line(aa, "agg_alt") or ""
    check("other aggregation form calibrated on the same batches, kept out of the shared snippet",
          abs(aa["terms"]["agg_alt"]["suggested_weight"] - 5) < 0.3 and aa["terms"]["agg_alt"]["batches"] == 90
          and "agg_alt" not in config_snippet(aa)
          and "agg_loss infonce -> override loss_weight_agg" in aline)

    report = format_report(res)
    check("report lists every term and the rule", all(k in report for k in true_ratio) and "R / median" in report)
    snippet = json.loads(config_snippet(res))
    check("snippet holds the five launch-config keys",
          set(snippet) == {"loss_weight_mmd", "loss_weight_contrastive", "loss_weight_cdd",
                           "loss_weight_paired", "loss_weight_agg"})
    print(report)


def check_base_from() -> None:
    print("\n-- launch config base_from --")
    from experiments.config import load_launch_config

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        full = {"train_entrypoint": "pretrain:main", "experiments_root": str(tmp / "root"),
                "base_config": {"model": {"lr": 1e-4}, "features": {"loss_weight_mmd": 7.0}},
                "slurm": {"enabled": True, "job_name_prefix": "ufm"},
                "models": [{"name": "a", "overrides": {}}, {"name": "b", "overrides": {}}]}
        (tmp / "full.json").write_text(json.dumps(full), encoding="utf-8")
        (tmp / "test.json").write_text(json.dumps(
            {"base_from": "full.json", "models": [{"name": "c", "overrides": {"model": {"lr": 2e-4}}}]}),
            encoding="utf-8")
        cfg = load_launch_config(tmp / "test.json")
        check("base config, slurm and entrypoint inherited",
              cfg.base_config["features"]["loss_weight_mmd"] == 7.0 and cfg.slurm.enabled
              and cfg.slurm.job_name_prefix == "ufm" and cfg.train_entrypoint == "pretrain:main")
        check("models are not inherited", [m.name for m in cfg.models] == ["c"])
        (tmp / "loop.json").write_text(json.dumps({"base_from": "loop.json", "models": []}), encoding="utf-8")
        check("a base_from cycle is refused", raises(lambda: load_launch_config(tmp / "loop.json"), ValueError))


if __name__ == "__main__":
    check_suggest()
    check_base_from()
    finish()
