"""Self-check for split_train_test.py, build_mix.py and bulk_sc_data_preprocessing.py.

Runs without torch or bionemo: the SCDL collection and ``DatasetDir`` are replaced by
small fakes that record what they are given, so the check covers everything this
code decides (rows, genes, gene ids, obs encoding, order, untouched sources,
transient-file cleanup) but not SCDL's own conversion. Needs anndata and scanpy.

    cd data_preprocess && python check_preprocessing.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for p in (str(HERE), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from experiments.check._harness import check, finish, raises  # noqa: E402

# --------------------------------------------------------------------------- #
# Fakes for the torch-dependent imports
# --------------------------------------------------------------------------- #


class FakeDatasetDir:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def mkdir(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = property(lambda self: self.data_dir / "vocab.json")
    obs_path = property(lambda self: self.data_dir / "obs.parquet")
    mapping_path = property(lambda self: self.data_dir / "mapping.json")
    memmap_path = property(lambda self: self.data_dir / "mem.map")


class FakeCollection:
    """Records each transient file at load time; ``flatten`` writes the concatenation."""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.fname_to_mmap = {}

    def load_h5ad(self, h5ad_path, use_X_not_raw=False):
        import anndata as ad

        assert use_X_not_raw, "X must be used, not raw.X"
        a = ad.read_h5ad(h5ad_path)
        self.fname_to_mmap[self.data_path / Path(h5ad_path).stem] = a

    def flatten(self, out, destroy_on_copy=False):
        out = Path(out)
        out.mkdir(parents=True)
        blocks = list(self.fname_to_mmap.values())
        np.save(out / "ids.npy", np.concatenate([np.repeat(b.var["_cf_gene_id"].to_numpy()[None], b.n_obs, 0).ravel()
                                                 for b in blocks]))
        rows = [(name, b.var["_cf_gene_id"].tolist(), b.X.toarray().tolist(), list(b.obs_names))
                for name, b in zip(self.fname_to_mmap, blocks)]
        (out / "rows.json").write_text(json.dumps([[str(n), ids, X, names] for n, ids, X, names in rows]))


def install_fakes() -> None:
    for name in ("cancerfoundation", "cancerfoundation.data"):
        sys.modules.setdefault(name, types.ModuleType(name))
    dataset = types.ModuleType("cancerfoundation.data.dataset")
    dataset.DatasetDir = FakeDatasetDir
    sys.modules["cancerfoundation.data.dataset"] = dataset
    for name in ("bionemo", "bionemo.scdl", "bionemo.scdl.io"):
        sys.modules.setdefault(name, types.ModuleType(name))
    coll = types.ModuleType("bionemo.scdl.io.single_cell_collection")
    coll.SingleCellCollection = FakeCollection
    sys.modules["bionemo.scdl.io.single_cell_collection"] = coll


install_fakes()
import bulk_sc_data_preprocessing as pre  # noqa: E402
import build_mix  # noqa: E402
import split_train_test as stt  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic experiments root
# --------------------------------------------------------------------------- #

RNG = np.random.default_rng(0)


def _write(path: Path, n: int, genes: list[str], obs: dict, dense: bool = False) -> None:
    import anndata as ad

    X = RNG.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(X=X if dense else sp.csr_matrix(X), obs=pd.DataFrame(obs, index=[f"{path.stem}_{i}" for i in range(n)]),
                   var=pd.DataFrame(index=genes))
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)


def _sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def build_root(root: Path) -> dict[str, Path]:
    g_all = [f"G{i}" for i in range(40)]
    n = 600
    donor = np.arange(n) // 20
    paths = {
        "sc": root / "data/single_cell/single_cell.h5ad",
        "bulk": root / "data/bulk/bulk.h5ad",
        "pb": root / "data/pseudobulk/pseudobulk.h5ad",
    }
    _write(paths["sc"], n, g_all, {"tissue_general": RNG.choice(["lung", "colon"], n), "donor_id": donor.astype(str),
                                   "paired": np.where(donor < 10, donor + 1, 0), "pseudobulk_id": (np.arange(n) // 10).astype(str)})
    _write(paths["bulk"], 60, g_all[5:], {"tissue_general": RNG.choice(["lung", "colon"], 60), "modality": "bulk",
                                          "paired": np.where(np.arange(60) < 10, np.arange(60) + 1, 0), "assay": "rna"},
           dense=True)
    _write(paths["pb"], 60, g_all[:35], {"tissue_general": RNG.choice(["lung", "colon"], 60),
                                         "pseudobulk_id": np.arange(60).astype(str)})
    return paths


def run_split(argv: list[str]) -> None:
    sys.argv = ["split_train_test.py", *argv]
    stt.main(stt.get_args())


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_split(root: Path, paths: dict[str, Path]) -> None:
    print("\n-- split_train_test.py --")
    before = {k: _sha(p) for k, p in paths.items()}
    R = ["--root", str(root)]
    run_split([*R, "--modality", "sc", "--split-name", "s0", "--stratify-by", "tissue_general",
               "--group-by", "donor_id", "paired:0", "--fractions", "train=0.7", "val=0.1", "test=0.2"])
    split = root / "data/single_cell/splits/s0"
    parts = stt.read_parts(split)
    check("three named parts written", set(parts) == {"train", "val", "test"})
    names = [n for v in parts.values() for n in v]
    check("every row in exactly one part", len(names) == len(set(names)) == 600)
    import anndata as ad

    obs = ad.read_h5ad(paths["sc"], backed="r").obs
    part_of = {n: p for p, v in parts.items() for n in v}
    donors = obs.assign(part=[part_of[n] for n in obs.index]).groupby("donor_id")["part"].nunique()
    check("no donor straddles parts", bool((donors == 1).all()))

    run_split([*R, "--modality", "pb", "--split-name", "s0", "--fractions", "train=0.7", "val=0.1", "test=0.2",
               "--inherit-from", "sc:s0", "--link-col", "pseudobulk_id"])
    pb_parts = stt.read_parts(root / "data/pseudobulk/splits/s0")
    sc_pb_part = obs.assign(part=[part_of[n] for n in obs.index]).groupby("pseudobulk_id")["part"].first()
    pb_obs = ad.read_h5ad(paths["pb"], backed="r").obs
    pb_part_of = {n: p for p, v in pb_parts.items() for n in v}
    linked = pb_obs["pseudobulk_id"].isin(sc_pb_part.index)
    check("pseudobulk inherits its cells' part",
          all(pb_part_of[n] == sc_pb_part[k] for n, k in pb_obs.loc[linked, "pseudobulk_id"].items()))

    run_split([*R, "--modality", "bulk", "--split-name", "s0", "--inherit-from", "sc:s0", "--link-col", "paired",
               "--fractions", "train=0.7", "val=0.1", "test=0.2", "--group-by", "paired:0"])
    run_split(["--input", str(paths["sc"]), "--out-dir", str(root / "explicit"), "--cluster", "leiden",
               "--cluster-max-rows", "150", "--n-hvg", "20", "--n-pcs", "5", "--filter", "tissue_general=lung"])
    card = json.loads((root / "explicit/split.json").read_text())
    check("explicit paths + filter + leiden", card["rows_selected"] == int((obs.tissue_general == "lung").sum())
          and card["cluster"]["assigned_by_centroid"] > 0)
    check("explicit split has no layout fields", "split_name" not in card)
    check("parent with unknown parts refused", raises(lambda: run_split(
        [*R, "--modality", "bulk", "--split-name", "bad", "--inherit-from", "sc:s0", "--link-col", "paired"]),
        SystemExit))
    check("existing split refused without --overwrite", raises(lambda: run_split(
        [*R, "--modality", "sc", "--split-name", "s0"]), FileExistsError))
    check("sources untouched by splitting", before == {k: _sha(p) for k, p in paths.items()})


def check_store(root: Path, paths: dict[str, Path]) -> None:
    print("\n-- build store from a split --")
    before = _sha(paths["sc"])
    R = ["--root", str(root)]
    run_split([*R, "--modality", "sc", "--split-name", "s1", "--group-by", "donor_id", "--build-store",
               "--obs-columns", "tissue_general", "paired", "assay", "--pb-id-column", "pseudobulk_id"])
    store = root / "data/single_cell/pipeline_ready/s1"
    train = stt.read_parts(root / "data/single_cell/splits/s1")["train"]
    obs = pd.read_parquet(store / "obs.parquet")
    mapping = json.loads((store / "mapping.json").read_text())
    vocab = json.loads((store / "vocab.json").read_text())
    rows = json.loads((store / "mem.map/rows.json").read_text())
    check("store holds exactly the train rows, in order", rows[0][3] == train and len(obs) == len(train))
    check("modality from the layout label", list(mapping["modality"]) == ["sc"])
    check("missing column filled with '0'", list(mapping["assay"]) == ["0"])
    check("pseudobulk ids prefixed", all(k.startswith("pb:") for k in mapping["pseudobulk_id"]))
    check("vocab is <cls>, <pad>, then sorted genes", list(vocab)[:3] == ["<cls>", "<pad>", "G0"])
    import anndata as ad

    src = ad.read_h5ad(paths["sc"])
    expected = src[train].X.toarray()
    check("values stored as in the source (no normalization)", np.allclose(np.array(rows[0][2]), expected))
    check("gene ids are vocab ids", rows[0][1] == [vocab[g] for g in src.var_names])
    check("transient files removed", not (store / pre.TRANSIENT_DIR).exists())
    check("source untouched by the store build", _sha(paths["sc"]) == before)


def check_mix(root: Path, paths: dict[str, Path]) -> None:
    print("\n-- build_mix.py --")
    before = {k: _sha(p) for k, p in paths.items()}
    sys.argv = ["build_mix.py", "--root", str(root), "--name", "m", "--source", "sc:s0", "bulk:s0", "pb:s0:val",
                "--obs-columns", "tissue_general", "paired"]
    build_mix.main(build_mix.get_args())
    store = root / "data/pipeline_ready/m"
    obs = pd.read_parquet(store / "obs.parquet")
    mapping = json.loads((store / "mapping.json").read_text())
    vocab = json.loads((store / "vocab.json").read_text())
    rows = json.loads((store / "mem.map/rows.json").read_text())
    n_expected = (len(stt.read_parts(root / "data/single_cell/splits/s0")["train"])
                  + len(stt.read_parts(root / "data/bulk/splits/s0")["train"])
                  + len(stt.read_parts(root / "data/pseudobulk/splits/s0")["val"]))
    check("rows of every source part", len(obs) == n_expected == sum(len(r[3]) for r in rows))
    check("intersection vocab, sorted", list(vocab)[2:] == sorted(f"G{i}" for i in range(5, 35)))
    inv = {v: k for k, v in mapping["modality"].items()}
    codes = obs["modality"].map(inv)
    check("modality per source: layout label or the file's own column",
          set(codes) == {"sc", "bulk", "pseudobulk"} and codes.iloc[-1] == "pseudobulk")
    check("blocks in source order", [Path(r[0]).name.split("_", 1)[1] for r in rows] == ["single_cell", "bulk", "pseudobulk"])
    check("dense source read too", len(rows[1][3]) > 0)
    check("no staged copies left", not (store / pre.TRANSIENT_DIR).exists() and not list(store.rglob("*.h5ad")))
    check("sources untouched by the mix", before == {k: _sha(p) for k, p in paths.items()})


def check_cli(root: Path, paths: dict[str, Path]) -> None:
    print("\n-- bulk_sc_data_preprocessing.py CLI --")
    rows_file = root / "rows.txt"
    rows_file.write_text("\n".join(f"bulk_{i}" for i in range(10)))
    sys.argv = ["bulk_sc_data_preprocessing.py", "--h5ad-path", str(paths["bulk"]), "--data-path", str(root / "cli"),
                "--obs-columns", "tissue_general", "--rows", str(rows_file), "--vocab-mode", "union"]
    pre.main(pre.get_args())
    obs = pd.read_parquet(root / "cli/obs.parquet")
    card = json.loads((root / "cli/store.json").read_text())
    check("--rows selects rows", len(obs) == 10 and card["n_rows"] == 10)
    check("file's own modality kept", list(json.loads((root / "cli/mapping.json").read_text())["modality"]) == ["bulk"])
    check("non-empty output refused without --overwrite", raises(lambda: pre.main(pre.get_args()), FileExistsError))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = build_root(root)
        check_split(root, paths)
        check_store(root, paths)
        check_mix(root, paths)
        check_cli(root, paths)
    finish()


if __name__ == "__main__":
    main()
