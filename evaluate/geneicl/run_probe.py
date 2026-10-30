"""Score a model's GeneICL-suite embeddings with GeneICL's own frozen-embedding probe.

    python -m evaluate.geneicl.run_probe --model unified_v1 [--suite benchmark] \\
        --geneicl-python /path/to/geneicl-benchmark/bin/python [--resume] [--datasets GSE107422 ...]

Reads ``embeddings/<model>/geneicl_<suite>/`` (``embed_suite.py``) and runs GeneICL's
``eval/benchmark/benchmark.py`` on it through ``_probe_shim.py``, with the python of the
GeneICL eval environment (``--geneicl-python``, else ``$GENEICL_PYTHON``, else this
python). Every scored task gets 5-fold CV (patient-grouped when the task has a group,
``seed`` 42), PCA keeping 95% of the variance and an untuned elastic net (regression) or
elastic-net logistic regression (classification), fitted per fold: the protocol of the
FM rows of the GeneICL paper. Writes ``results/tasks/geneicl_<suite>/<model>/``:

- ``probe_{classification,regression}.csv``: per-fold metrics (benchmark.py columns);
- ``probe_regression.oof.csv`` and ``probe_classification.clfoof.npz``: per-sample
  predictions, so pooled metrics need no rerun;
- ``run.json``: probe key, GeneICL commit, embedding card summary and the command.

The ``model`` column of the CSVs holds the probe key (``cancerfoundation``, or
``cf_<embedder>``), never the model name, so renaming a model only moves directories.
``--datasets`` scores only the tasks of those datasets, through a filtered copy of the
suite index under the task directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_with_config  # noqa: E402
from evaluate.geneicl import suite as S  # noqa: E402

SHIM = Path(__file__).resolve().parent / "_probe_shim.py"
PYTHON_ENV_VAR = "GENEICL_PYTHON"
OUT_STEM = "probe"
SUBSET_DIR = "_suite_subset"
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./")


def task_name(suite: str) -> str:
    """The results/tasks/ directory of a suite."""
    return f"geneicl_{suite}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score GeneICL-suite embeddings with GeneICL's probe.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--geneicl-repo", type=Path, default=None,
                        help=f"GeneICL checkout (default: ${S.GENEICL_ENV_VAR}, else ../GeneICL)")
    parser.add_argument("--geneicl-python", default=None,
                        help=f"python of the GeneICL eval env (default: ${PYTHON_ENV_VAR}, else this python)")
    parser.add_argument("--suite", default="benchmark", choices=list(S.SUITES))
    parser.add_argument("--model", default=None, help="Model whose embeddings/<model>/geneicl_<suite>/ is scored")
    parser.add_argument("--emb-dir", type=Path, default=None,
                        help="Embedding directory (default: embeddings/<model>/geneicl_<suite>/)")
    parser.add_argument("--probe-key", default=None,
                        help="model= key given to benchmark.py (default: cancerfoundation, or cf_<embedder>)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Score only the tasks of these datasets")
    parser.add_argument("--seed", type=int, default=42, help="CV seed (GeneICL's reported runs use 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Keep folds already on disk (benchmark.py overwrite=false)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and stop")
    return parser


def hydra_value(value) -> str:
    """A Hydra override value: bare when safe, else single-quoted."""
    text = value.as_posix() if isinstance(value, Path) else str(value)
    if text and set(text) <= _SAFE:
        return text
    if "'" in text:
        raise ValueError(f"Cannot pass {text!r} as a Hydra override value")
    return f"'{text}'"


def default_probe_key(card: dict) -> str:
    method = card.get("method") or "cancerfoundation"
    return "cancerfoundation" if method == "cancerfoundation" else f"cf_{method}"


def subset_suite(suite_path: Path, datasets: list[str], dest: Path) -> Path:
    """A copy of the suite's index and scored task files restricted to ``datasets``.

    In embedding mode benchmark.py reads only ``index.json`` and ``tasks/``, never
    ``datasets/``, so nothing else is needed.
    """
    index = S.load_index(suite_path)
    keep = set(datasets)
    index["tasks"] = [t for t in S.scored_tasks(index) if t["dataset"] in keep]
    index["datasets"] = {k: v for k, v in index.get("datasets", {}).items() if k in keep}
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "tasks").mkdir(parents=True)
    for t in index["tasks"]:
        shutil.copy2(suite_path / t["file"], dest / t["file"])
    (dest / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return dest


def build_command(python: str, repo: Path, key: str, emb_dir: Path, data_dir: Path,
                  out_dir: Path, seed: int, resume: bool) -> list[str]:
    overrides = {
        "emb_dir": emb_dir,
        "data_dir": data_dir,
        "out": out_dir / OUT_STEM,
        "overwrite": "false" if resume else "true",
        "seed": seed,
        "hydra.run.dir": out_dir / ".hydra",
        "hydra.output_subdir": "null",
    }
    return [python, str(SHIM), str(repo), key, *(f"{k}={hydra_value(v)}" for k, v in overrides.items())]


def main(argv: list[str] | None = None) -> dict:
    from experiments.layout import ExperimentLayout, validate_model_name

    args = parse_with_config(build_parser(), "geneicl_probe", argv)
    if not args.model:
        sys.exit("Error: give --model")
    name = validate_model_name(args.model)
    layout = ExperimentLayout.from_root(args.root)
    repo = S.resolve_geneicl(args.geneicl_repo)
    suite_path = S.suite_dir(repo, args.suite)
    emb_dir = args.emb_dir or layout.geneicl_embedding_dir(name, args.suite)
    if not emb_dir.is_dir():
        sys.exit(f"Error: {emb_dir} not found; run evaluate.geneicl.embed_suite for '{name}' first")
    card_path = emb_dir / "embed.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}

    datasets = S.list_datasets(suite_path, args.datasets)
    missing = [d for d in datasets if not (emb_dir / f"{d}.npz").exists()]
    if missing:
        sys.exit(f"Error: {len(missing)} datasets have no embeddings in {emb_dir}: {missing}. "
                 "Embed them, or restrict the run with --datasets")
    out_dir = layout.task_dir(task_name(args.suite), name)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = subset_suite(suite_path, datasets, out_dir / SUBSET_DIR) if args.datasets else suite_path

    key = args.probe_key or default_probe_key(card)
    python = args.geneicl_python or os.environ.get(PYTHON_ENV_VAR) or sys.executable
    cmd = build_command(python, repo, key, emb_dir, data_dir, out_dir, args.seed, args.resume)
    print("Command:\n  " + " ".join(cmd))
    if args.dry_run:
        return {"command": cmd}

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    subprocess.run(cmd, check=True)
    run = {
        "model": name, "suite": args.suite, "probe_key": key, "seed": args.seed, "resume": args.resume,
        "datasets": args.datasets or None, "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "geneicl_repo": str(repo), "geneicl_commit": S.geneicl_commit(repo), "python": python,
        "emb_dir": str(emb_dir),
        "embedding": {k: card.get(k) for k in ("method", "checkpoint", "options", "gene_ids", "normalize",
                                               "modality", "geneicl_commit")},
        "panel_hashes": {f: e.get("panel_hash") for f, e in card.get("files", {}).items()},
        "command": cmd,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")
    return run


if __name__ == "__main__":
    main()
