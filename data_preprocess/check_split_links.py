"""Check that linked rows of several modalities sit in the same part of a split.

    python data_preprocess/check_split_links.py --split leiden_s1 \\
        --modalities single_cell bulk pseudobulk --links pseudobulk_id:0,-1 paired:0 [--max-leak 0]

Reads, per modality, ``data/<modality>/splits/<split>/`` (part lists + ``split.json``)
and the link columns of the source h5ad recorded in that ``split.json`` (obs only, via
``h5ad_io.SourceReader``; no torch). For every link column (``COL[:NOLINK,...]``, with
empty/NaN/0/-1 always meaning "no link", see ``utils.sampling.link_keys``) it reports:

- link keys per part and modality (how many pairs / pseudobulks each part holds);
- keys whose rows sit in more than one part (a train/test leak), with row counts and
  examples;
- keys present in only one modality (e.g. a pair with a pseudobulk but no bulk), which
  the matching losses cannot use.

Exit code 1 when the number of leaking keys of any link exceeds ``--max-leak``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from h5ad_io import SourceReader  # noqa: E402
from split_train_test import SPLIT_CARD, parse_link_specs, read_parts  # noqa: E402
from utils.sampling import link_keys  # noqa: E402


def load_modality(split_dir: Path, columns: list[str], source: Path | None = None) -> pd.DataFrame:
    """``part`` plus the requested link columns for every row of one modality's split."""
    card = json.loads((split_dir / SPLIT_CARD).read_text(encoding="utf-8"))
    path = source or Path(card["source"]["path"])
    with SourceReader(path) as reader:
        obs = reader.obs
        frame = pd.DataFrame(index=obs.index.astype(str))
        for col in columns:
            if col in obs.columns:
                frame[col] = obs[col].to_numpy()
    part = pd.Series(pd.NA, index=frame.index, dtype=object)
    for name, rows in read_parts(split_dir).items():
        part[part.index.isin(rows)] = name
    frame["part"] = part
    return frame[frame["part"].notna()]


def check_link(frames: dict[str, pd.DataFrame], col: str, extra: list[str], max_examples: int) -> int:
    """Print the report for one link column; returns the number of leaking keys."""
    rows = []
    for modality, frame in frames.items():
        if col not in frame.columns:
            print(f"  {modality}: no '{col}' column")
            continue
        keys = link_keys(frame[col], extra)
        linked = keys.notna()
        rows.append(pd.DataFrame({"key": keys[linked].to_numpy(), "part": frame["part"][linked].to_numpy(),
                                  "modality": modality}))
    if not rows:
        print("  no modality has this column")
        return 0
    links = pd.concat(rows, ignore_index=True)
    print(f"  {len(links)} linked rows, {links['key'].nunique()} keys")

    per_part = links.drop_duplicates(["key", "part", "modality"]).pivot_table(
        index="part", columns="modality", values="key", aggfunc="count", fill_value=0)
    print("  keys per part and modality:")
    print("    " + per_part.to_string().replace("\n", "\n    "))

    parts_per_key = links.groupby("key")["part"].nunique()
    leaking = parts_per_key[parts_per_key > 1].index
    if len(leaking):
        leak_rows = links[links["key"].isin(leaking)]
        print(f"  [LEAK] {len(leaking)} keys span several parts ({len(leak_rows)} rows), e.g.:")
        for key in list(leaking)[:max_examples]:
            sides = leak_rows[leak_rows["key"] == key].groupby(["modality", "part"]).size()
            print(f"    {key}: " + ", ".join(f"{m}/{p}={n}" for (m, p), n in sides.items()))
    else:
        print("  [OK] every key sits in one part")

    modalities_per_key = links.groupby("key")["modality"].agg(lambda s: tuple(sorted(set(s))))
    counts = modalities_per_key.value_counts()
    print("  keys by the modalities that carry them:")
    for combo, n in counts.items():
        print(f"    {' + '.join(combo)}: {n}")
    return int(len(leaking))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that linked rows share a split part.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--split", required=True, help="Split name under data/<modality>/splits/")
    parser.add_argument("--modalities", nargs="+", default=["single_cell", "bulk", "pseudobulk"])
    parser.add_argument("--links", nargs="+", default=["pseudobulk_id:0,-1", "paired:0"],
                        metavar="COL[:NOLINK,...]", help="Link columns and extra 'no link' values")
    parser.add_argument("--max-leak", type=int, default=0, help="Leaking keys tolerated per link")
    parser.add_argument("--examples", type=int, default=5, help="Leaking keys printed per link")
    args = parser.parse_args(argv)

    from experiments.layout import ExperimentLayout

    layout = ExperimentLayout.from_root(args.root)
    specs = parse_link_specs(args.links)
    frames = {}
    for modality in args.modalities:
        split_dir = layout.split_dir(modality, args.split)
        frames[modality] = load_modality(split_dir, list(specs))
        print(f"{modality}: {len(frames[modality])} rows in {split_dir} "
              f"({frames[modality]['part'].value_counts().to_dict()})")

    worst = 0
    for col, extra in specs.items():
        print(f"\n== link '{col}'")
        worst = max(worst, check_link(frames, col, extra, args.examples))

    if worst > args.max_leak:
        print(f"\nFAILED: up to {worst} leaking keys per link (tolerated: {args.max_leak})")
        return 1
    print("\nOK: linked rows share their part" + (f" (<= {args.max_leak} leaks tolerated)" if args.max_leak else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
