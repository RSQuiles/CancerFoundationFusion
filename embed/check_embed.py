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

from embed.embed import EMBED_CARD, embed_adata, write_embeddings  # noqa: E402
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
                           "modality_col": modality_col, "genes": genes})
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


def check_any() -> None:
    print("\n-- embed_any.py --")
    check("pca and bulkformer registered", {"pca", "bulkformer"} <= set(EMBEDDERS))
    check("options parsed as YAML scalars",
          parse_options(["n_pcs=5", "normalized=true", "checkpoint=/a/b"]) == {"n_pcs": 5, "normalized": True,
                                                                             "checkpoint": "/a/b"})
    train, test = _adata(200, seed=1), _adata(60, seed=2)
    pca = build_embedder("pca", {"n_pcs": 5, "n_hvg": 20})
    emb = run_embedder(pca, test, fit_adata=train)
    check("pca fits on one set, embeds another", emb.shape == (60, 5) and len(pca.genes) == 20)
    check("bulkformer stub says it is not implemented",
          raises(lambda: run_embedder(build_embedder("bulkformer"), test), NotImplementedError))
    check("unknown embedder refused", raises(lambda: build_embedder("nope"), KeyError))

    @register("_bad_shape")
    class Bad(Embedder):
        def embed(self, adata):
            return np.zeros((3, 2))

    check("wrong output shape refused", raises(lambda: run_embedder(Bad(), test), ValueError))
    EMBEDDERS.pop("_bad_shape")


def check_splits() -> None:
    print("\n-- embed_splits.py --")
    import anndata as ad

    with tempfile.TemporaryDirectory() as tmp:
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
        cmd = [sys.executable, str(PROJECT_ROOT / "embed/embed_splits.py"), "--root", tmp, "--embedder", "pca",
               "--split", "s0", "--modalities", "sc", "bulk", "--train-subset", "50",
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
    check_any()
    check_splits()
    finish()


if __name__ == "__main__":
    main()
