"""Self-check for the representation scripts on synthetic embeddings.

Writes a throwaway experiments root with ``embeddings/<model>/{sc,bulk,pseudobulk}_test.h5ad``
and ``embed.json`` (what ``embed/embed_splits.py`` produces), then runs ``plot_umap`` and
``plot_alignment`` end to end. Needs anndata, scanpy and matplotlib; no torch,
checkpoint or cluster.

    python -m evaluate.check.check_representation
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from experiments.layout import ExperimentLayout
from evaluate.representation import plot_alignment, plot_umap
from evaluate._cli import format_title, parse_mapping, parse_quotas

MODEL = "toy_model"
SIZES = {"sc": 800, "pseudobulk": 60, "bulk": 40}


def _write_root(root: Path, mixed_panels: bool = False) -> ExperimentLayout:
    import anndata as ad

    layout = ExperimentLayout(root)
    rng = np.random.default_rng(0)
    tissues = np.array(["Lung", "breast", "colon", "unknown"])
    files = {}
    for i, (mod, n) in enumerate(SIZES.items()):
        tissue = rng.choice(tissues, size=n, p=[0.4, 0.3, 0.2, 0.1])
        emb = rng.normal(size=(n, 16)).astype(np.float32) + (tissue == "Lung")[:, None] * 3
        paired = np.where(np.arange(n) < 20, np.arange(n) % 10 + 1, 0)
        obs = pd.DataFrame({"tissue_general": tissue, "paired": paired, "modality": mod},
                           index=[f"{mod}_{j}" for j in range(n)])
        path = layout.embedding_file(MODEL, mod, "test")
        path.parent.mkdir(parents=True, exist_ok=True)
        ad.AnnData(obs=obs, obsm={"X_emb": emb}).write_h5ad(path)
        files[path.name] = {"panel_hash": f"p{i}" if mixed_panels else "same", "n_rows": n}
    layout.embed_card(MODEL).write_text(json.dumps({"model": MODEL, "files": files}), encoding="utf-8")
    return layout


def check_umap() -> None:
    print("\n-- plot_umap --")
    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_root(Path(tmp))
        plot_umap.main(["--root", tmp, "--models", MODEL, "--modality", "single_cell",
                        "--filter", "tissue_general=lung,breast", "--subsample", "200",
                        "--color", "tissue_general", "paired", "--skip-unknown"])
        out = layout.representation_dir(MODEL)
        png = out / "umap_sc_test_lung-breast.png"
        check("figure written under results/representation/<model>/", png.exists())
        coords = pd.read_parquet(out / "umap_sc_test_lung-breast.parquet")
        check("subsample size respected", len(coords) == 200)
        check("filter applied before sampling",
              set(coords["tissue_general"].str.lower()) <= {"lung", "breast"})
        check("coordinates saved", {"umap_1", "umap_2"} <= set(coords.columns))

        ext = Path(tmp) / "external.h5ad"
        import anndata as ad

        src = ad.read_h5ad(layout.embedding_file(MODEL, "bulk", "test"))
        src.obsm["X_pca"] = src.obsm.pop("X_emb")
        src.write_h5ad(ext)
        plot_umap.main(["--input", str(ext), "--obsm-key", "X_pca", "--subsample", "0"])
        check("--input mode writes next to the file", (Path(tmp) / "umap_external.png").exists())
        check("missing embeddings reported",
              raises(lambda: plot_umap.main(["--root", tmp, "--models", "nope"]), FileNotFoundError))

        cfg = Path(tmp) / "cfg.yaml"
        cfg.write_text(f"root: {tmp}\nmodels: [{MODEL}]\numap:\n  modality: bulk\n  subsample: 0\n"
                       "alignment:\n  per_modality: [sc=100]\n", encoding="utf-8")
        plot_umap.main(["--config", str(cfg)])
        check("config section applied", (out / "umap_bulk_test.png").exists())
        bad = Path(tmp) / "bad.yaml"
        bad.write_text("modalty: sc\n", encoding="utf-8")
        check("unknown config key refused", raises(lambda: plot_umap.main(["--config", str(bad)]), SystemExit))


def check_alignment() -> None:
    print("\n-- plot_alignment --")
    check("quotas parsed", parse_quotas(["sc=1_000", "pb=all"], ["sc", "pseudobulk", "bulk"])
          == {"sc": 1000, "pseudobulk": None, "bulk": None})
    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_root(Path(tmp))
        plot_alignment.main(["--root", tmp, "--models", MODEL, "--per-modality", "sc=300",
                             "--connect-pairs", "--skip-unknown"])
        out = layout.alignment_dir(MODEL)
        coords = pd.read_parquet(out / "alignment_test.parquet")
        check("figure written under _alignment/<model>/", (out / "alignment_test.png").exists())
        counts = coords["modality"].value_counts().to_dict()
        check("per-modality quotas", counts == {"sc": 300, "pseudobulk": 60, "bulk": 40})
        check("unpaired rows labelled", "unpaired" in set(coords["paired"]))

        plot_alignment.main(["--root", tmp, "--models", MODEL, "--paired-only"])
        paired = pd.read_parquet(out / "alignment_test_paired.parquet")
        check("--paired-only keeps only paired rows", "unpaired" not in set(paired["paired"]))

        segs = plot_alignment.pair_segments(
            np.zeros((4, 2)),
            pd.DataFrame({"modality": ["bulk", "pseudobulk", "sc", "bulk"]}),
            pd.Series(["1", "1", "1", np.nan]),
        )
        check("one segment per bulk point and partner modality", len(segs) == 2)

    with tempfile.TemporaryDirectory() as tmp:
        _write_root(Path(tmp), mixed_panels=True)
        check("mixed gene panels refused",
              raises(lambda: plot_alignment.main(["--root", tmp, "--models", MODEL]), SystemExit))
        plot_alignment.main(["--root", tmp, "--models", MODEL, "--allow-mixed-panels"])
        check("--allow-mixed-panels overrides",
              ExperimentLayout(Path(tmp)).alignment_dir(MODEL).joinpath("alignment_test.png").exists())


def check_titles_labels() -> None:
    print("\n-- titles and labels --")
    import matplotlib

    matplotlib.use("Agg")
    from evaluate.representation.render import draw_umap_panels

    check("title template filled", format_title("{model} ({part})", "dflt", model="m", part="test") == "m (test)")
    check("no template -> default title", format_title(None, "dflt", model="m") == "dflt")
    check("plain string kept", format_title("My plot", "dflt", model="m") == "My plot")
    check("unknown title field refused", raises(lambda: format_title("{nope}", "d", model="m"), ValueError))
    check("labels from a YAML dict", parse_mapping({"tissue_general": "Tissue"}) == {"tissue_general": "Tissue"})
    check("labels from the CLI", parse_mapping(["tissue_general=Tissue", "a=b=c"]) == {"tissue_general": "Tissue", "a": "b=c"})
    check("malformed label refused", raises(lambda: parse_mapping(["tissue_general"]), ValueError))

    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"modality": rng.choice(["sc", "bulk"], 50), "tissue_general": rng.choice(["lung", "colon"], 50)})
    fig = draw_umap_panels(rng.normal(size=(50, 2)), obs, ["modality", "tissue_general"], modality_col="modality",
                           title="T", labels={"tissue_general": "Tissue", "modality": "Data type"})
    titles = [ax.get_title() for ax in fig.axes]
    from matplotlib.legend import Legend

    legends = [c.get_title().get_text() for ax in fig.axes for c in ax.get_children() if isinstance(c, Legend)]
    check("panel titles use the labels", titles == ["Data type", "Tissue"])
    check("legend titles use the labels", "Tissue" in legends and "Data type" in legends)
    check("suptitle drawn", fig._suptitle is not None and fig._suptitle.get_text() == "T")

    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_root(Path(tmp))
        cfg = Path(tmp) / "cfg.yaml"
        import yaml

        cfg.write_text(yaml.safe_dump({
            "root": tmp, "models": [MODEL], "labels": {"tissue_general": "Tissue"},
            "umap": {"modality": "bulk", "subsample": 0, "title": "{model} / {modality_name}"},
            "alignment": {"per_modality": ["sc=100"], "title": "{model}: {counts}"},
        }), encoding="utf-8")
        plot_umap.main(["--config", str(cfg)])
        plot_alignment.main(["--config", str(cfg)])
        coords = pd.read_parquet(layout.representation_dir(MODEL) / "umap_bulk_test.parquet")
        check("parquet keeps the raw column names", "tissue_general" in coords.columns)
        check("labels + title from config accepted by both scripts",
              (layout.alignment_dir(MODEL) / "alignment_test.png").exists())


def main() -> None:
    check_umap()
    check_alignment()
    check_titles_labels()
    finish()


if __name__ == "__main__":
    main()
