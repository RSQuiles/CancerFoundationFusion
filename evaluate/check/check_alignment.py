"""Self-check for ``evaluate/alignment/scib_benchmark.py`` on synthetic embeddings.

Covers loading several models on their shared rows, the filter and the label-stratified
per-modality quotas, the label x batch pre-flight report and the dry-run outputs. The
Benchmarker itself runs only where ``scib_metrics`` imports (the bulkFM env); otherwise
it is skipped.

    python -m evaluate.check.check_alignment
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

from experiments.check._harness import check, finish, raises  # noqa: E402
from experiments.layout import ExperimentLayout  # noqa: E402
from evaluate.alignment import scib_benchmark  # noqa: E402

SIZES = {"bulk": 80, "pseudobulk": 120}


def _write_root(root: Path) -> ExperimentLayout:
    import anndata as ad

    layout = ExperimentLayout(root)
    rng = np.random.default_rng(0)
    for model, drop in (("m1", 0), ("m2", 5), ("pca", 0)):
        out = layout.embedding_dir(model)
        out.mkdir(parents=True)
        files = {}
        for mod, n in SIZES.items():
            # Tissue "skin" exists only in bulk, so it is not usable by per-label metrics.
            tissues = np.where(np.arange(n) % 4 == 0, "skin" if mod == "bulk" else "lung",
                               np.where(np.arange(n) % 2 == 0, "lung", "colon"))
            obs = pd.DataFrame({"modality": mod, "tissue_general": tissues}, index=[f"{mod}{i}" for i in range(n)])
            obs = obs.iloc[drop:]  # m2 did not embed the first rows of each modality
            emb = rng.normal(size=(len(obs), 8)).astype(np.float32)
            name = f"{mod}_test.h5ad"
            ad.AnnData(obs=obs, obsm={"X_emb": emb}).write_h5ad(out / name)
            files[name] = {"panel_hash": f"hash_{model}"}
        (out / "embed.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return layout


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        layout = _write_root(Path(tmp))
        print("\n-- loading and selection --")
        obs, embs, hashes = scib_benchmark.load_models(layout, ["m1", "m2"], ["bulk", "pseudobulk"], "test")
        check("rows restricted to those every model embedded", len(obs) == 200 - 10)
        check("one aligned embedding per model", all(e.shape == (190, 8) for e in embs.values()))
        check("row names carry the modality", obs.index[0].startswith("bulk:"))
        check("panel hashes collected", hashes["m1"]["bulk_test.h5ad"] == "hash_m1")

        args = scib_benchmark.build_parser().parse_args(
            ["--models", "m1", "--name", "x", "--per-modality", "bulk=40", "pseudobulk=all"])
        mask = scib_benchmark.select_rows(obs, args, ["bulk", "pseudobulk"])
        picked = obs[mask]
        check("per-modality quota applied", (picked["modality"] == "bulk").sum() == 40
              and (picked["modality"] == "pseudobulk").sum() == 115)
        check("quota stratified by label (all bulk tissues kept)",
              set(picked.loc[picked["modality"] == "bulk", "tissue_general"]) == {"skin", "lung", "colon"})

        _, summary = scib_benchmark.label_batch_report(picked, "modality", "tissue_general", 10)
        check("pre-flight: skin is bulk-only, so 2 of 3 labels usable",
              summary["n_labels"] == 3 and summary["n_labels_usable"] == 2)

        print("\n-- dry run --")
        out = scib_benchmark.main(["--root", tmp, "--name", "t", "--models", "m1", "m2", "pca",
                                   "--pre-integrated", "pca", "--dry-run"])
        check("written under results/assessment/scib_<name>/", out == layout.assessment_dir("scib_t"))
        manifest = json.loads((out / "manifest.json").read_text())
        check("manifest lists every model; the reference is not a row",
              manifest["models"] == ["m1", "m2", "pca"] and manifest["rows"] == ["m1", "m2"])
        check("label x batch table saved", (out / "label_batch.csv").exists())
        meta = json.loads((out / "meta.json").read_text())
        check("meta records keys and counts", meta["batch_key"] == "modality" and meta["n_rows"] == 190)
        check("a missing label column is refused",
              raises(lambda: scib_benchmark.main(["--root", tmp, "--name", "t2", "--models", "m1",
                                                  "--label-key", "nope", "--dry-run"]), SystemExit))
        check("a single modality is refused",
              raises(lambda: scib_benchmark.main(["--root", tmp, "--name", "t3", "--models", "m1",
                                                  "--modalities", "bulk", "--dry-run"]), SystemExit))

        print("\n-- Benchmarker --")
        try:
            import scib_metrics  # noqa: F401
        except ImportError:
            print("  SKIP  scib_metrics not importable here (it lives in the bulkFM env)")
        else:
            out = scib_benchmark.main(["--root", tmp, "--name", "full", "--models", "m1", "m2",
                                       "--pre-integrated", "pca"])
            results = pd.read_csv(out / "scib.csv", index_col=0)
            check("one row per model plus the Metric Type row", {"m1", "m2", "Metric Type"} <= set(results.index))
    finish()


if __name__ == "__main__":
    main()
