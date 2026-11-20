"""Self-check for ``utils/sampling.py`` on synthetic obs (numpy + pandas only).

    python -m experiments.check.check_sampling
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from utils.sampling import (
    allocate,
    filter_mask,
    filter_tag,
    parse_filter_specs,
    parse_fractions,
    parse_group_specs,
    select_linked_rows,
    split_groups,
    split_summary,
    stratified_group_split,
    stratified_subsample,
    stratum_labels,
    stratum_quotas,
)


def _obs(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tissue = rng.choice(["Lung", "breast", "Colon", "skin"], size=n, p=[0.5, 0.3, 0.15, 0.05])
    assay = rng.choice(["10x", "smart-seq"], size=n)
    donor = rng.integers(0, 150, size=n)
    # Pairs live inside a donor (a cell line / sample), like real paired data.
    paired = np.where(rng.random(n) < 0.2, donor + 1, 0)
    obs = pd.DataFrame(
        {"tissue": tissue, "assay": assay, "donor": donor.astype(str), "paired": paired},
        index=[f"c{i}" for i in range(n)],
    )
    obs.loc[obs.index[:5], "tissue"] = np.nan
    return obs


def check_filters() -> None:
    print("\n-- filters --")
    obs = _obs()
    specs = parse_filter_specs(["tissue=lung,BREAST", "assay=10x"])
    check("specs parsed", specs == {"tissue": ["lung", "BREAST"], "assay": ["10x"]})
    mask = filter_mask(obs, specs)
    expected = obs["tissue"].astype(str).str.lower().isin({"lung", "breast"}) & (obs["assay"] == "10x")
    check("case-insensitive AND filter", bool((mask == expected.to_numpy()).all()))
    check("hyphen and underscore fold together",
          filter_mask(obs, {"assay": ["smart_seq"]}).sum() == (obs["assay"] == "smart-seq").sum())
    check("no filter keeps everything", filter_mask(obs, None).all())
    check("empty selection raises", raises(lambda: filter_mask(obs, {"tissue": ["brain"]})))
    check("missing column raises", raises(lambda: filter_mask(obs, {"nope": ["x"]})))
    check("malformed spec raises", raises(lambda: parse_filter_specs(["tissue"])))
    check("filter tag", filter_tag(specs) == "lung-breast-10x" and filter_tag(None) == "")


def check_allocation() -> None:
    print("\n-- allocation --")
    sizes = pd.Series({"a": 900, "b": 90, "c": 9, "d": 1})
    alloc = allocate(sizes, 100, min_per_stratum=1)
    check("sums to n", int(alloc.sum()) == 100)
    check("rare strata kept", bool((alloc >= 1).all()))
    check("never above size", bool((alloc <= sizes).all()))
    check("roughly proportional", abs(int(alloc["a"]) - 90) <= 2)
    check("n above total returns sizes", allocate(sizes, 5000).equals(sizes))


def check_subsample() -> None:
    print("\n-- stratified subsample --")
    obs = _obs()
    picked = stratified_subsample(obs, 600, ["tissue"], seed=1)
    check("size", len(picked) == 600)
    check("unique and in original order",
          picked.is_unique and list(picked) == [i for i in obs.index if i in set(picked)])
    full = stratum_labels(obs, ["tissue"]).value_counts(normalize=True)
    sub = stratum_labels(obs.loc[picked], ["tissue"]).value_counts(normalize=True)
    check("proportions preserved (+-2%)", bool((full - sub.reindex(full.index).fillna(0)).abs().max() < 0.02))
    check("missing labels become their own stratum", "unknown" in sub.index)
    check("deterministic under a seed",
          list(picked) == list(stratified_subsample(obs, 600, ["tissue"], seed=1)))
    check("different seeds differ", list(picked) != list(stratified_subsample(obs, 600, ["tissue"], seed=2)))
    check("n=None keeps all", len(stratified_subsample(obs, None, ["tissue"])) == len(obs))
    check("unstratified works", len(stratified_subsample(obs, 50, None)) == 50)


def check_split() -> None:
    print("\n-- group-aware split --")
    obs = _obs()
    groups_spec = parse_group_specs(["donor", "paired:0"])
    check("group specs parsed", set(groups_spec) == {"donor", "paired"} and "0" in groups_spec["paired"]
          and "0" not in groups_spec["donor"])
    assign = stratified_group_split(obs, ["tissue"], 0.2, seed=0, group_cols=groups_spec)
    check("every row assigned", set(assign.unique()) == {"train", "test"} and len(assign) == len(obs))
    groups = split_groups(obs, groups_spec)
    sides = pd.DataFrame({"g": groups, "s": assign}).groupby("g")["s"].nunique()
    check("no group straddles the split", bool((sides == 1).all()))
    paired = obs["paired"] > 0
    pair_sides = pd.DataFrame({"p": obs.loc[paired, "paired"], "s": assign[paired]}).groupby("p")["s"].nunique()
    check("no pair straddles the split", bool((pair_sides == 1).all()))
    frac = (assign == "test").mean()
    check(f"overall test fraction near 0.2 (got {frac:.3f})", abs(frac - 0.2) < 0.05)
    table = split_summary(assign, stratum_labels(obs, ["tissue"]), ["train", "test"])
    big = table[table[["train", "test"]].sum(axis=1) >= 50]  # tiny strata ride inside other strata's groups
    check("every sizeable stratum on both sides", bool((big[["train", "test"]] > 0).all().all()))
    check("deterministic",
          assign.equals(stratified_group_split(obs, ["tissue"], 0.2, seed=0, group_cols=groups_spec)))
    rowwise = stratified_group_split(obs, ["tissue", "assay"], 0.3, seed=0)
    check("row-level split without groups", abs((rowwise == "test").mean() - 0.3) < 0.02)

    three = stratified_group_split(obs, ["tissue"], parse_fractions(["train=0.7", "val=0.1", "test=0.2"]),
                                   seed=0, group_cols=groups_spec)
    realized = three.value_counts(normalize=True)
    check(f"three parts near 0.7/0.1/0.2 (got {realized.round(3).to_dict()})",
          set(realized.index) == {"train", "val", "test"}
          and all(abs(realized[k] - v) < 0.05 for k, v in {"train": 0.7, "val": 0.1, "test": 0.2}.items()))
    three_table = split_summary(three, stratum_labels(obs, ["tissue"]), ["train", "val", "test"])
    big3 = three_table[three_table[["train", "val", "test"]].sum(axis=1) >= 50]
    check("every sizeable stratum in every part", bool((big3[["train", "val", "test"]] > 0).all().all()))
    check("fractions must sum to 1", raises(lambda: parse_fractions(["train=0.7", "test=0.2"])))

    bridge = pd.DataFrame({"donor": ["a", "a", "b", "b", "c", "c"], "paired": [1, 0, 1, 0, 0, 0]},
                          index=[f"r{i}" for i in range(6)])
    g = split_groups(bridge, parse_group_specs(["donor", "paired:0"]))
    check("a pair spanning two donors merges them", g["r0"] == g["r1"] == g["r2"] == g["r3"] != g["r4"])
    g0 = split_groups(bridge, ["donor", "paired"])
    check("without ':0', paired=0 is itself a group (links everything)", g0.nunique() == 1)
    tiny = obs.iloc[:3].assign(tissue=["x", "y", "z"])
    tiny_assign = stratified_group_split(tiny, ["tissue"], 0.34, seed=0)
    check("single-group strata pooled, not all forced to train", (tiny_assign == "test").sum() >= 1)
    check("bad test_size raises", raises(lambda: stratified_group_split(obs, None, 1.5)))


def chk(name: str, ok: bool, detail: str = "") -> None:
    """``check`` that prints ``detail`` when it fails."""
    if not ok and detail:
        print(f"        {detail}")
    check(name, ok)


def check_modes() -> None:
    print("\n-- allocation modes, unknown cap --")
    sizes = pd.Series({"blood": 5000, "brain": 3000, "lung": 400, "liver": 100, "eye": 10})
    bal = allocate(sizes, 1000, mode="balanced")
    chk("balanced sums to n", int(bal.sum()) == 1000, str(bal.to_dict()))
    chk("balanced: small strata take all, large share the rest equally",
          bal["eye"] == 10 and bal["liver"] == 100
          and bal[["blood", "brain", "lung"]].max() - bal[["blood", "brain", "lung"]].min() <= 1,
          str(bal.to_dict()))
    sq = allocate(sizes, 1000, mode="sqrt")
    chk("sqrt sums to n and sits between proportional and balanced",
          int(sq.sum()) == 1000 and allocate(sizes, 1000)["blood"] > sq["blood"] > bal["blood"], str(sq.to_dict()))
    chk("unknown mode raises", raises(lambda: allocate(sizes, 10, mode="nope")))

    labels = pd.Series(["unknown"] * 7000 + ["lung"] * 2000 + ["liver"] * 1000)
    q = stratum_quotas(labels, 1000, mode="proportional", unknown_frac=0.05)
    chk("unknown capped at unknown_frac * n, the rest goes to tissues",
          q["unknown"] == 50 and int(q.sum()) == 1000, str(q.to_dict()))
    q0 = stratum_quotas(labels, 1000, mode="balanced", unknown_frac=0.0)
    chk("unknown_frac 0 leaves unknown out", q0["unknown"] == 0 and int(q0.sum()) == 1000, str(q0.to_dict()))
    joint = pd.Series(["lung | unknown"] * 500 + ["lung | 10x"] * 500)
    chk("a joint stratum with an unknown part counts as unknown",
          stratum_quotas(joint, 100, unknown_frac=0.1)["lung | unknown"] == 10)

    obs = _obs()
    a = stratified_subsample(obs, 600, ["tissue"], seed=3)
    b = stratified_subsample(obs, 600, ["tissue"], seed=3, mode="proportional", unknown_frac=None)
    chk("explicit defaults give the same draw", list(a) == list(b))
    kept = obs.index[:40]
    c = stratified_subsample(obs, 600, ["tissue"], seed=3, mode="balanced", keep=kept)
    chk("keep rows are always in the sample, total stays n", set(kept) <= set(c) and len(c) == 600, str(len(c)))


def check_links() -> None:
    print("\n-- linked rows (pairs) --")
    rng = np.random.default_rng(0)
    tissues = ["blood"] * 6 + ["lung"] * 3 + ["eye"]

    def pool(n, per_pair, n_pairs, prefix):
        t = rng.choice(tissues, n)
        paired = np.zeros(n, dtype=int)
        for k in range(1, n_pairs + 1):
            paired[(k - 1) * per_pair:k * per_pair] = k
            t[(k - 1) * per_pair:k * per_pair] = "lung" if k % 2 else "blood"
        return pd.DataFrame({"tissue": t, "paired": paired}, index=[f"{prefix}{i}" for i in range(n)])

    sc, bulk, pb = pool(20000, 40, 60, "c"), pool(6000, 1, 60, "b"), pool(300, 1, 60, "p")
    pools = {"sc": sc, "bulk": bulk, "pseudobulk": pb}
    budgets = {"sc": 2000, "bulk": 1000, "pseudobulk": None}
    kept, info = select_linked_rows(pools, budgets, ["paired:0"], ["tissue"], mode="balanced",
                                    linked_frac=0.25, per_key=10, seed=0)
    keys = {m: set(pools[m].loc[kept[m], "paired"]) for m in pools}
    chk("the same pairs in every modality", keys["sc"] == keys["bulk"] == keys["pseudobulk"],
          str({m: len(k) for m, k in keys.items()}))
    chk("linked rows within linked_frac * budget",
          len(kept["sc"]) <= 500 and len(kept["bulk"]) <= 250, str(info["rows"]))
    chk("per_key caps the cells of a pair",
          pools["sc"].loc[kept["sc"]].groupby("paired").size().max() <= 10)
    q = stratum_quotas(stratum_labels(sc, ["tissue"]), 2000, mode="balanced")
    per_t = sc.loc[kept["sc"], "tissue"].value_counts()
    chk("linked rows within each stratum's quota", bool((per_t <= q.reindex(per_t.index)).all()),
          f"{per_t.to_dict()} vs {q.to_dict()}")
    tight, tinfo = select_linked_rows(pools, {"sc": 2000, "bulk": 20, "pseudobulk": None}, ["paired:0"],
                                      ["tissue"], linked_frac=0.25, per_key=10, seed=0)
    tkeys = {m: set(pools[m].loc[tight[m], "paired"]) for m in pools}
    chk("a pair that does not fit one modality is dropped from all",
          len(tkeys["bulk"]) <= 5 and tkeys["sc"] == tkeys["bulk"] == tkeys["pseudobulk"],
          str(tinfo))
    again, _ = select_linked_rows({"bulk": bulk}, {"bulk": 1000}, ["paired:0"], ["tissue"], mode="balanced",
                                  linked_frac=0.25, per_key=10, seed=0)
    chk("key order does not depend on the other pools (subset call agrees)",
          set(again["bulk"]) >= set(kept["bulk"]))
    sub = stratified_subsample(bulk, 1000, ["tissue"], mode="balanced", keep=kept["bulk"], seed=1)
    chk("subsample keeps the linked rows and has n rows",
          set(kept["bulk"]) <= set(sub) and len(sub) == 1000)
    linked_sc = sc.index[sc["paired"] > 0]
    sub_sc = stratified_subsample(sc, 2000, ["tissue"], mode="balanced", keep=kept["sc"],
                                  exclude=linked_sc, seed=1)
    in_pairs = sc.loc[sub_sc].query("paired > 0")
    chk("exclude: linked rows enter only through keep (per_key holds, no extra pairs)",
        set(in_pairs.index) == set(kept["sc"]) and in_pairs.groupby("paired").size().max() <= 10
        and len(sub_sc) == 2000)


def main() -> None:
    check_filters()
    check_allocation()
    check_subsample()
    check_modes()
    check_links()
    check_split()
    finish()


if __name__ == "__main__":
    main()
