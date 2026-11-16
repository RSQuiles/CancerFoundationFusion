"""Self-check for embed.py / embed_splits.py / embed_any.py (no torch, checkpoint or cluster).

A fake model with ``CancerFoundation.embed``'s signature stands in for the real one,
so the chunking and the one-panel-for-everything rule are tested without torch.
Needs anndata, scanpy and scikit-learn.

    python -m embed.check_embed
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed.embed import EMBED_CARD, embed_adata, panel_hash, panel_record, write_embeddings  # noqa: E402
from embed.embed_any import EMBEDDERS, Embedder, build_embedder, parse_options, register, run_embedder  # noqa: E402
from experiments.check._harness import check, finish, raises  # noqa: E402
from experiments.layout import ExperimentLayout  # noqa: E402


class FakeModel:
    """Mimics ``CancerFoundation.embed``: fits a panel unless given one, returns (df, genes)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def embed(self, adata, gene_subset=None, modality_col=None, modality="sc", **kwargs):
        genes = list(gene_subset) if gene_subset is not None else list(adata.var_names[: 10 + adata.n_obs % 7])
        self.calls.append({"n": adata.n_obs, "given_panel": gene_subset is not None,
                           "modality_col": modality_col, "genes": genes, "kwargs": kwargs})
        X = adata[:, genes].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        df = pd.DataFrame(X[:, :4].astype(np.float32), index=adata.obs_names)
        return df, genes


def _adata(n: int, mods=("sc",), seed: int = 0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"modality": rng.choice(list(mods), size=n),
                        "tissue_general": rng.choice(["lung", "colon"], size=n)},
                       index=[f"c{i}" for i in range(n)])
    X = rng.poisson(3.0, size=(n, 50)).astype(np.float32)
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(50)]))


def _cancerfoundation_importable() -> bool:
    """The PCA embedder normalizes through cancerfoundation, whose package init loads torch."""
    try:
        import cancerfoundation.data.preprocess  # noqa: F401
    except Exception:  # noqa: BLE001 - torch may not load here
        return False
    return True


def check_machinery() -> None:
    print("\n-- embed.py machinery --")
    model = FakeModel()
    adata = _adata(30, mods=("sc", "bulk"))
    emb, genes = embed_adata(model, adata, modality_col="modality")
    check("small input embedded in one call", len(model.calls) == 1 and emb.shape == (30, 4))
    check("multi-modality call asks for a shared panel", model.calls[0]["modality_col"] == "modality")

    model = FakeModel()
    big = _adata(250, mods=("sc", "bulk"))
    emb, genes = embed_adata(model, big, modality_col="modality", chunk_rows=100, panel_sample=40)
    check("panel fitted once, then 3 chunks", [c["given_panel"] for c in model.calls] == [False, True, True, True])
    check("every chunk uses the fitted panel", all(c["genes"] == genes for c in model.calls[1:]))
    check("all rows embedded in order", emb.shape == (250, 4)
          and np.allclose(emb, big[:, genes].X[:, :4]))
    check("chunking refuses per-modality panels",
          raises(lambda: embed_adata(FakeModel(), big, modality_col="modality", chunk_rows=100, shared_panel=False)))

    with tempfile.TemporaryDirectory() as tmp:
        write_embeddings(Path(tmp), "a", adata.obs, emb[:30], genes, provenance={"method": "x"})
        write_embeddings(Path(tmp), "b", adata.obs, emb[:30], None)
        card = json.loads((Path(tmp) / EMBED_CARD).read_text())
        check("one card, one entry per file", set(card["files"]) == {"a.h5ad", "b.h5ad"})
        check("panel saved once by hash", (Path(tmp) / f"panel_{card['files']['a.h5ad']['panel_hash']}.txt").exists())
        check("no-panel embedder recorded as such", card["files"]["b.h5ad"]["panel_hash"] is None)
        import anndata as ad

        out = ad.read_h5ad(Path(tmp) / "a.h5ad")
        check("obs + obsm only", out.n_vars == 0 and out.obsm["X_emb"].shape == (30, 4))

        skipped = emb[:30].copy()
        skipped[[2, 5]] = np.nan
        write_embeddings(Path(tmp), "c", adata.obs, skipped, genes)
        entry = json.loads((Path(tmp) / EMBED_CARD).read_text())["files"]["c.h5ad"]
        written = ad.read_h5ad(Path(tmp) / "c.h5ad")
        check("all-NaN rows (skipped by the embedder) left out of the file",
              written.n_obs == 28 and not {adata.obs_names[2], adata.obs_names[5]} & set(written.obs_names))
        check("dropped rows recorded in embed.json",
              entry["n_dropped"] == 2 and entry["dropped_rows"] == [adata.obs_names[2], adata.obs_names[5]])

    from embed.embed_any import _count_like_rows

    rows = np.array([[1, 2, 0], [0, 0, 0], [1, -1, 3], [1.5, 2, 1], [np.nan, 1, 1], [0, 0, 7]])
    check("count-like rows: integral, non-negative, finite, positive total",
          list(_count_like_rows(rows)) == [True, False, False, False, False, True])


def _preprocess_module():
    """``cancerfoundation/data/preprocess.py`` by path: numpy only, no package init (torch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_check_preprocess", PROJECT_ROOT / "cancerfoundation" / "data" / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_input_modes() -> None:
    print("\n-- input modes (train_matched) --")
    pp = _preprocess_module()
    rng = np.random.default_rng(3)
    n_genes, budget, cap = 200, 40, 16
    ranked = rng.permutation(n_genes)

    sc_row = np.zeros(n_genes, dtype=np.float32)
    sc_row[rng.choice(n_genes, 15, replace=False)] = rng.integers(1, 4, 15)
    cols, bins, n_nz, n_z = pp.train_matched_input(sc_row, ranked, 51, budget, cap, method="quantile", seed=7)
    order = {g: k for k, g in enumerate(ranked)}
    check("sc-like row: every expressed gene plus zero_cap zeros", n_nz == 15 and n_z == cap and len(cols) == 31)
    check("non-zeros first, each block in rank order",
          (sc_row[cols[:n_nz]] > 0).all() and (sc_row[cols[n_nz:]] == 0).all()
          and all(order[a] < order[b] for blk in (cols[:n_nz], cols[n_nz:]) for a, b in zip(blk, blk[1:])))
    top_zeros = [g for g in ranked if sc_row[g] == 0][:cap]
    check("the zeros are the top-ranked zero genes", list(cols[n_nz:]) == top_zeros)
    np.random.seed(7)
    full = pp.binning(sc_row, 51, method="quantile")
    check("bins come from binning the whole row", np.array_equal(bins, full[cols]))
    np.random.seed(123)
    again = pp.train_matched_input(sc_row, ranked, 51, budget, cap, method="quantile", seed=7)
    check("seeded per cell: same input whatever ran before", np.array_equal(again[1], bins))

    bulk_row = rng.integers(1, 500, n_genes).astype(np.float32)
    bulk_row[ranked[[3, 50, 120]]] = 0
    cols, bins, n_nz, n_z = pp.train_matched_input(bulk_row, ranked, 51, budget, cap, method="midrank")
    check("bulk-like row: its few zeros, the rest expressed, full budget",
          n_z == 3 and n_nz == budget - 3 and len(cols) == budget)
    check("midrank bins from the whole row",
          np.array_equal(bins, pp.binning(bulk_row, 51, method="midrank")[cols]))
    _, _, n_nz, n_z = pp.train_matched_input(sc_row, ranked, 51, budget, 0)
    check("zero_cap 0 (nonzero mode): no zeros", n_z == 0 and n_nz == 15)
    sub = ranked[:50]
    cols, *_ = pp.train_matched_input(bulk_row, sub, 51, budget, cap)
    check("a cut ranking (panel_size) is never left", set(cols) <= set(sub))
    check("cell seed is stable", pp.cell_seed("c1") == pp.cell_seed("c1") != pp.cell_seed("c2"))

    print("\n-- panel records --")
    genes = ["G3", "G1", "G2"]
    check("fixed panel: hash of the sorted genes, as before input modes",
          panel_record(genes)[0] == panel_hash(genes) and panel_record(genes, "legacy_panel")[0] == panel_hash(genes))
    h1, text, n = panel_record(genes, "train_matched", 0.4)
    check("ranking file keeps the order and the zero fraction",
          text.splitlines() == ["# input_mode=train_matched", "# zero_fraction=0.4", "G3", "G1", "G2"] and n == 3)
    check("ranking hash depends on order and zero fraction",
          h1 != panel_record(["G1", "G2", "G3"], "train_matched", 0.4)[0]
          and h1 != panel_record(genes, "train_matched", 0.2)[0])
    _, text, n = panel_record({"sc": ["G1", "G2"], "bulk": ["G2", "G3"]}, "train_matched", 0.4)
    check("per-modality rankings: one section each", "## bulk" in text and "## sc" in text and n == 3)

    model = FakeModel()
    embed_adata(model, _adata(30))
    kw = model.calls[0]["kwargs"]
    check("default input mode is train_matched with zero fraction 0.4",
          kw.get("input_mode") == "train_matched" and kw.get("zero_fraction") == 0.4 and "panel_size" in kw)
    model = FakeModel()
    embed_adata(model, _adata(30), input_mode="legacy_panel")
    kw = model.calls[0]["kwargs"]
    check("legacy_panel passes no ranking options", kw.get("input_mode") == "legacy_panel" and "zero_fraction" not in kw)
    model = FakeModel()
    big = _adata(250, mods=("sc", "bulk"))
    _, genes = embed_adata(model, big, modality_col="modality", chunk_rows=100, panel_sample=40)
    check("chunks reuse the ranking in its order", all(c["genes"] == genes for c in model.calls[1:]))
    args = _splits_args()
    check("embed_splits defaults: train_matched, 0.4, no panel cut",
          (args.input_mode, args.zero_fraction, args.panel_size) == ("train_matched", 0.4, None))
    check("embed_splits takes --input-mode legacy_panel",
          _splits_args("--input-mode", "legacy_panel").input_mode == "legacy_panel")
    from embed.embed import input_mode_provenance

    check("provenance: nonzero records zero fraction 0",
          input_mode_provenance(_splits_args("--input-mode", "nonzero"))["zero_fraction"] == 0.0)
    check("provenance: legacy_panel records no ranking options",
          input_mode_provenance(_splits_args("--input-mode", "legacy_panel")) ==
          {"input_mode": "legacy_panel", "zero_fraction": None, "panel_size": None})


def check_any() -> None:
    print("\n-- embed_any.py --")
    check("pca and bulkformer registered", {"pca", "bulkformer"} <= set(EMBEDDERS))
    check("options parsed as YAML scalars",
          parse_options(["n_pcs=5", "normalized=true", "checkpoint=/a/b"]) == {"n_pcs": 5, "normalized": True,
                                                                             "checkpoint": "/a/b"})
    train, test = _adata(200, seed=1), _adata(60, seed=2)
    check("pca normalizes with cpm_log1p by default", build_embedder("pca").normalize == "cpm_log1p")
    check("pca normalize=none means values as stored",
          build_embedder("pca", parse_options(["normalize=none"])).normalize is None)
    check("pca refuses an unknown normalize", raises(lambda: build_embedder("pca", {"normalize": "tpm"}), ValueError))
    check("pca refuses the retired normalized/log1p_only options",
          raises(lambda: build_embedder("pca", {"normalized": True}), ValueError))
    if not _cancerfoundation_importable():
        print("  SKIP  pca fit/embed: cancerfoundation (torch) not importable here")
    else:
        pca = build_embedder("pca", {"n_pcs": 5, "n_hvg": 20})
        emb = run_embedder(pca, test, fit_adata=train)
        check("pca fits on one set, embeds another", emb.shape == (60, 5) and len(pca.genes) == 20)
        raw = build_embedder("pca", {"n_pcs": 5, "n_hvg": 20, "normalize": None})
        emb_raw = run_embedder(raw, test, fit_adata=train)
        check("pca panel does not depend on normalize", raw.genes == pca.genes)
        check("normalize changes the embedding", not np.allclose(emb, emb_raw))
    check("bulkformer refuses a missing checkout",
          raises(lambda: build_embedder("bulkformer", {"repo": "/nonexistent"}).embed(test), FileNotFoundError))
    check("bulkformer refuses an unknown input scale",
          raises(lambda: build_embedder("bulkformer", {"repo": "/x", "input": "fpkm"}), ValueError))
    check("unknown embedder refused", raises(lambda: build_embedder("nope"), KeyError))

    @register("_bad_shape")
    class Bad(Embedder):
        def embed(self, adata):
            return np.zeros((3, 2))

    check("wrong output shape refused", raises(lambda: run_embedder(Bad(), test), ValueError))
    EMBEDDERS.pop("_bad_shape")


def _write_split_root(tmp: str) -> ExperimentLayout:
    """Raw sc and bulk h5ads with a split s0 (30 test rows, 90 train rows each)."""
    layout = ExperimentLayout(Path(tmp))
    for i, mod in enumerate(("sc", "bulk")):
        a = _adata(120, seed=10 + i)
        a.obs_names = [f"{mod}_{j}" for j in range(a.n_obs)]
        a.obs = a.obs.drop(columns=["modality"])
        raw = layout.raw_h5ad(mod)
        raw.parent.mkdir(parents=True)
        a.write_h5ad(raw)
        split = layout.split_dir(mod, "s0")
        split.mkdir(parents=True)
        names = list(a.obs_names)
        (split / "test.txt").write_text("\n".join(names[:30]))
        (split / "train.txt").write_text("\n".join(names[30:]))
    return layout


def _splits_args(*argv: str):
    from embed import embed_splits

    old = sys.argv
    sys.argv = ["embed_splits.py", "--model", "x", "--split", "s0", *argv]
    try:
        return embed_splits.parse_args()
    finally:
        sys.argv = old


def check_selection() -> None:
    """Which rows embed_splits and embed.py read; needs no model, so it always runs."""
    print("\n-- row selection --")
    from embed import embed_splits
    from embed.embed import read_input

    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_split_root(tmp)
        args = _splits_args()
        check("default parts: test only", args.parts == ["test"])
        pieces = embed_splits.collect_rows(layout, args, "bulk")
        check("test only -> one piece with every test row",
              len(pieces) == 1 and pieces[0].n_obs == 30 and set(pieces[0].obs["_embed_part"]) == {"test"})
        pieces = embed_splits.collect_rows(layout, _splits_args("--test-subset", "10"), "bulk")
        all_test = set(_adata(120, seed=11).obs["tissue_general"].iloc[:30])
        check("--test-subset: stratified subsample of the test rows",
              pieces[0].n_obs == 10 and set(pieces[0].obs["tissue_general"]) == all_test
              and set(pieces[0].obs_names) <= {f"bulk_{j}" for j in range(30)})
        check("--max-test still works as an alias", _splits_args("--max-test", "7").test_subset == 7)
        pieces = embed_splits.collect_rows(layout, _splits_args("--parts", "train_subset", "--train-subset", "0"), "sc")
        check("train only, all rows with --train-subset 0",
              len(pieces) == 1 and pieces[0].n_obs == 90 and set(pieces[0].obs["_embed_part"]) == {"train_subset"})
        pieces = embed_splits.collect_rows(
            layout, _splits_args("--parts", "test", "train_subset", "--train-subset", "20", "--test-subset", "5"), "sc")
        check("both parts on request", sorted(p.n_obs for p in pieces) == [5, 20])
        check("--train-stratify-by still accepted", _splits_args("--train-stratify-by", "a").stratify_by == ["a"])

        raw = layout.raw_h5ad("sc")
        check("embed.py reads every row by default", read_input(raw).n_obs == 120)
        sub = read_input(raw, 12, ["tissue_general", "not_a_column"], seed=1)
        check("embed.py --subsample: stratified, only those rows read",
              sub.n_obs == 12 and sub.obs["tissue_general"].nunique() == 2)
        check("embed.py --subsample is deterministic under a seed",
              list(read_input(raw, 12, ["tissue_general"], seed=1).obs_names) == list(sub.obs_names))


def check_splits() -> None:
    print("\n-- embed_splits.py --")
    if not _cancerfoundation_importable():
        print("  SKIP  embed_splits --embedder pca: cancerfoundation (torch) not importable here")
        return
    import anndata as ad

    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_split_root(tmp)
        cmd = [sys.executable, str(PROJECT_ROOT / "embed/embed_splits.py"), "--root", tmp, "--embedder", "pca",
               "--split", "s0", "--modalities", "sc", "bulk", "--parts", "test", "train_subset",
               "--train-subset", "50",
               "--option", "n_pcs=4", "--option", "n_hvg=20"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode:
            print(out.stdout[-2000:], out.stderr[-2000:])
        check("embed_splits --embedder pca runs", out.returncode == 0)
        files = sorted(p.name for p in layout.embedding_dir("pca").glob("*.h5ad"))
        check("four files under embeddings/pca/",
              files == ["bulk_test.h5ad", "bulk_train_subset.h5ad", "sc_test.h5ad", "sc_train_subset.h5ad"])
        card = json.loads(layout.embed_card("pca").read_text())
        check("one panel for every file", len({f["panel_hash"] for f in card["files"].values()}) == 1)
        check("fitted on the train subset only", all(f["fitted_on"] == "train_subset" for f in card["files"].values()))
        test_rows = ad.read_h5ad(layout.embedding_file("pca", "sc", "test"))
        check("test file holds exactly the test rows", set(test_rows.obs_names) == {f"sc_{j}" for j in range(30)})
        check("train subset size", ad.read_h5ad(layout.embedding_file("pca", "bulk", "train_subset")).n_obs == 50)
        check("card names the model", card["model"] == "pca")


def main() -> None:
    check_machinery()
    check_input_modes()
    check_any()
    check_selection()
    check_splits()
    finish()


if __name__ == "__main__":
    main()
