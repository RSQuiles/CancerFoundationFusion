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
                                    linked_frac=0.25, per_key=10, seed=0, linked_quota="stratum")
    keys = {m: set(pools[m].loc[kept[m], "paired"]) for m in pools}
    chk("the same pairs in every modality", keys["sc"] == keys["bulk"] == keys["pseudobulk"],
          str({m: len(k) for m, k in keys.items()}))
    chk("linked rows within linked_frac * budget",
          len(kept["sc"]) <= 500 and len(kept["bulk"]) <= 250, str(info["rows"]))
    chk("per_key caps the cells of a pair",
          pools["sc"].loc[kept["sc"]].groupby("paired").size().max() <= 10)
    q = stratum_quotas(stratum_labels(sc, ["tissue"]), 2000, mode="balanced")
    per_t = sc.loc[kept["sc"], "tissue"].value_counts()
    chk("linked_quota=stratum: linked rows within each stratum's quota", bool((per_t <= q.reindex(per_t.index)).all()),
          f"{per_t.to_dict()} vs {q.to_dict()}")
    tight, tinfo = select_linked_rows(pools, {"sc": 2000, "bulk": 20, "pseudobulk": None}, ["paired:0"],
                                      ["tissue"], linked_frac=0.25, per_key=10, seed=0, linked_quota="stratum")
    tkeys = {m: set(pools[m].loc[tight[m], "paired"]) for m in pools}
    chk("a pair that does not fit one modality is dropped from all",
          len(tkeys["bulk"]) <= 5 and tkeys["sc"] == tkeys["bulk"] == tkeys["pseudobulk"],
          str(tinfo))
    again, _ = select_linked_rows({"bulk": bulk}, {"bulk": 1000}, ["paired:0"], ["tissue"], mode="balanced",
                                  linked_frac=0.25, per_key=10, seed=0, linked_quota="stratum")
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


def check_per_link() -> None:
    """Per-link budgets: pairs and pseudobulks in one subsample, each with its own share."""
    print("\n-- per-link budgets (pairs + pseudobulks) --")
    from utils.sampling import parse_per_link

    rng = np.random.default_rng(2)
    n_pairs, per_pair, n_pb, per_pb = 30, 60, 400, 30
    n_free = 30000
    paired = np.r_[np.repeat(np.arange(1, n_pairs + 1), per_pair), np.zeros(n_pb * per_pb + n_free, dtype=int)]
    pbid = np.r_[np.full(n_pairs * per_pair, "0"), np.repeat([f"pb{i}" for i in range(n_pb)], per_pb),
                 np.full(n_free, "0")]
    sc = pd.DataFrame({"tissue": rng.choice(["lung", "blood", "eye"], len(paired)), "paired": paired,
                       "pseudobulk_id": pbid}, index=[f"c{i}" for i in range(len(paired))])
    pb = pd.DataFrame({"tissue": "lung", "paired": 0, "pseudobulk_id": [f"pb{i}" for i in range(n_pb)]},
                      index=[f"p{i}" for i in range(n_pb)])
    pools, budgets = {"sc": sc, "pseudobulk": pb}, {"sc": 10000, "pseudobulk": None}
    specs = ["paired:0", "pseudobulk_id:0,-1"]
    kept, info = select_linked_rows(pools, budgets, specs, ["tissue"], seed=0,
                                    linked_frac={"paired": 0.2, "pseudobulk_id": 0.3},
                                    per_key={"paired": 50, "pseudobulk_id": 15})
    rows = sc.loc[kept["sc"]]
    pair_rows, pb_rows = rows[rows["paired"] > 0], rows[rows["paired"] == 0]
    chk("each link column has its own budget (pairs <= 2000, pseudobulk cells <= 3000)",
        len(pair_rows) <= 2000 and len(pb_rows) <= 3000, str(info["per_link"]))
    chk("pairs come first and are all kept (their budget is their own)",
        info["per_link"]["paired"]["keys_accepted"] == n_pairs and len(pair_rows) == n_pairs * 50)
    chk("per-link caps: 50 cells per pair, 15 per pseudobulk",
        pair_rows.groupby("paired").size().max() == 50 and set(pb_rows.groupby("pseudobulk_id").size()) == {15})
    chk("pseudobulks fill their budget with whole keys (200 x 15)",
        info["per_link"]["pseudobulk_id"]["keys_accepted"] == 200 and len(pb_rows) == 3000)
    chk("kept pseudobulk rows match the kept cells' keys",
        set(pb.loc[kept["pseudobulk"], "pseudobulk_id"]) == set(pb_rows["pseudobulk_id"]))
    scalar_a, _ = select_linked_rows(pools, budgets, specs, ["tissue"], seed=0, linked_frac=0.25, per_key=10)
    scalar_b, _ = select_linked_rows(pools, budgets, specs, ["tissue"], seed=0, linked_frac=0.25,
                                     per_key={"paired": 10, "pseudobulk_id": 10})
    chk("a per_key mapping with equal values matches the scalar", list(scalar_a["sc"]) == list(scalar_b["sc"]))
    chk("an unknown column in a mapping is refused",
        raises(lambda: select_linked_rows(pools, budgets, specs, linked_frac={"donor": 0.1})))
    chk("parse_per_link: scalar, mapping and COL=VALUE list",
        parse_per_link(0.3) == 0.3 and parse_per_link(["0.3"]) == 0.3 and parse_per_link(None) is None
        and parse_per_link(["paired=0.06", "pseudobulk_id=0.3"]) == {"paired": 0.06, "pseudobulk_id": 0.3}
        and parse_per_link({"paired": "50"}, int) == {"paired": 50})
    chk("a malformed entry is refused", raises(lambda: parse_per_link(["paired=0.1", "oops"])))


def check_caps() -> None:
    print("\n-- label caps, pairs that form their own stratum --")
    from utils.sampling import label_caps, parse_label_caps

    sizes = pd.Series({"paired": 6000, "cell line | 10x": 3000, "lung": 500, "liver": 500})
    caps = label_caps(sizes, 1000, {"paired": 0.05, "Cell Line": 0.1})
    chk("a capped label gets at most frac * n (joint stratum and case-insensitive match)",
        caps == {"paired": 50, "cell line | 10x": 100}, str(caps))
    q = allocate(sizes, 1000, mode="proportional", caps=caps)
    chk("capped share goes to the other strata", q["paired"] == 50 and q["cell line | 10x"] == 100
        and int(q.sum()) == 1000 and q["lung"] == q["liver"] == 425, str(q.to_dict()))
    chk("caps parse from a mapping or KEY=VALUE list",
        parse_label_caps({"paired": 0.05}) == parse_label_caps(["paired=0.05"]) == {"paired": 0.05})
    chk("a cap outside [0, 1] is refused", raises(lambda: parse_label_caps(["x=2"])))
    labels = pd.Series(["unknown"] * 700 + ["lung"] * 300)
    chk("the unknown cap is unchanged by label caps",
        stratum_quotas(labels, 100, unknown_frac=0.05).to_dict()
        == stratum_quotas(labels, 100, unknown_frac=0.05, caps={}).to_dict())

    # single cell: every pair's cells form the 'paired' stratum (as in SC_pretraining)
    rng = np.random.default_rng(1)
    n_pairs, per = 40, 30
    t = np.array(["paired"] * (n_pairs * per) + list(rng.choice([f"t{i}" for i in range(30)], 20000)))
    paired = np.r_[np.repeat(np.arange(1, n_pairs + 1), per), np.zeros(20000, dtype=int)]
    sc = pd.DataFrame({"tissue": t, "paired": paired}, index=[f"c{i}" for i in range(len(t))])
    bulk = pd.DataFrame({"tissue": ["unknown"] * n_pairs + ["lung"] * 3000,
                         "paired": np.r_[np.arange(1, n_pairs + 1), np.zeros(3000, dtype=int)]},
                        index=[f"b{i}" for i in range(n_pairs + 3000)])
    pools, budgets = {"sc": sc, "bulk": bulk}, {"sc": 2000, "bulk": 1000}
    capd = {"paired": 0.05}
    strict, sinfo = select_linked_rows(pools, budgets, ["paired:0"], ["tissue"], mode="balanced", caps=capd,
                                       unknown_frac=0.05, linked_frac=0.25, per_key=10, seed=0,
                                       linked_quota="stratum")
    loose, linfo = select_linked_rows(pools, budgets, ["paired:0"], ["tissue"], mode="balanced", caps=capd,
                                      unknown_frac=0.05, linked_frac=0.25, per_key=10, seed=0)
    chk("stratum rule: the capped 'paired' stratum admits only a few pairs",
        sinfo["keys_accepted"] <= 10, str(sinfo))
    chk("budget rule (default): every pair kept, bounded by linked_frac",
        linfo["keys_accepted"] == n_pairs and linfo["rows"]["sc"] == n_pairs * 10, str(linfo))
    sub_sc = stratified_subsample(sc, 2000, ["tissue"], mode="balanced", caps=capd, keep=loose["sc"],
                                  exclude=sc.index[sc["paired"] > 0], seed=0)
    comp = sc.loc[sub_sc, "tissue"].value_counts()
    chk("pairs above the stratum's cap: total still n, 'paired' = the pair cells only",
        len(sub_sc) == 2000 and comp["paired"] == n_pairs * 10, str(comp.head().to_dict()))
    others = comp.drop("paired")
    chk("the other strata stay balanced", others.max() - others.min() <= 1, str(others.describe().to_dict()))
    chk("unknown linked_quota refused",
        raises(lambda: select_linked_rows(pools, budgets, ["paired:0"], linked_quota="nope")))


def main() -> None:
    check_filters()
    check_allocation()
    check_subsample()
    check_modes()
    check_links()
    check_per_link()
    check_caps()
    check_split()
    finish()


if __name__ == "__main__":
    main()
