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
    split_groups,
    split_summary,
    stratified_group_split,
    stratified_subsample,
    stratum_labels,
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


def main() -> None:
    check_filters()
    check_allocation()
    check_subsample()
    check_split()
    finish()


if __name__ == "__main__":
    main()
