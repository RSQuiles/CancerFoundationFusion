"""Reading a GeneICL checkout: where it is, its task suites, datasets and genes.

Torch-free and anndata-free, so the probe driver and the self-check import it anywhere.

A suite is ``<geneicl>/eval/data/<suite>/`` (``benchmark``: 52 tasks over 32 datasets;
``non_benchmark``: GeneICL's training-time monitor) holding

    index.json               tasks (task_id, dataset, task_type, file), dataset row counts
    reference_genes.txt      the gene symbols, i.e. the column order of every dataset
    datasets/<dataset>.npz   X (log1p-CPM, samples x genes) + obs_names
    tasks/<task_id>.npz      sample_ids, y [, group, event]

The GeneICL checkout is ``--geneicl-repo``, else ``$GENEICL_REPO``, else ``../GeneICL``
next to this repository.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENEICL_ENV_VAR = "GENEICL_REPO"
DEFAULT_GENEICL = PROJECT_ROOT.parent / "GeneICL"
SUITES = ("benchmark", "non_benchmark")
BENCHMARK_SCRIPT = Path("eval") / "benchmark" / "benchmark.py"
SYMBOL_TO_ENSEMBL = PROJECT_ROOT / "gene_mappings" / "symbol_to_ensembl.json"
GENE_IDS = ("auto", "symbol", "ensembl")

_RESTORE_HINT = ("restore the checkout first: git -C {repo} checkout -- . "
                 "(after git restore --staged . if files show as deleted), then git -C {repo} lfs pull")


def resolve_geneicl(path: str | Path | None = None) -> Path:
    """The GeneICL checkout: ``path``, else ``$GENEICL_REPO``, else ``../GeneICL``."""
    if path is not None:
        repo = Path(os.path.expandvars(str(path))).expanduser()
    elif os.environ.get(GENEICL_ENV_VAR):
        repo = Path(os.environ[GENEICL_ENV_VAR]).expanduser()
    else:
        repo = DEFAULT_GENEICL
    if not repo.is_dir():
        raise FileNotFoundError(f"No GeneICL checkout at {repo}; pass --geneicl-repo or set ${GENEICL_ENV_VAR}")
    if not (repo / BENCHMARK_SCRIPT).exists():
        raise FileNotFoundError(f"{repo / BENCHMARK_SCRIPT} not found; " + _RESTORE_HINT.format(repo=repo))
    return repo


def suite_dir(repo: Path, suite: str) -> Path:
    """``<repo>/eval/data/<suite>/``, checked for its index."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {SUITES}")
    path = Path(repo) / "eval" / "data" / suite
    if not (path / "index.json").exists():
        raise FileNotFoundError(f"{path / 'index.json'} not found; " + _RESTORE_HINT.format(repo=repo))
    return path


def geneicl_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def load_index(suite_path: Path) -> dict:
    return json.loads((Path(suite_path) / "index.json").read_text(encoding="utf-8"))


def scored_tasks(index: dict) -> list[dict]:
    """The tasks GeneICL's benchmark.py scores (survival tasks are quickstart-only)."""
    return [t for t in index["tasks"] if t["task_type"] in ("classification", "regression")]


def list_datasets(suite_path: Path, datasets: list[str] | None = None) -> list[str]:
    """The datasets of the scored tasks, sorted, or the requested ones (checked)."""
    known = sorted({t["dataset"] for t in scored_tasks(load_index(suite_path))})
    if not datasets:
        return known
    unknown = sorted(set(datasets) - set(known))
    if unknown:
        raise ValueError(f"Not datasets of this suite: {unknown}. Known: {known}")
    return [d for d in known if d in set(datasets)]


def reference_genes(suite_path: Path) -> list[str]:
    path = Path(suite_path) / "reference_genes.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(suite_path: Path, dataset: str) -> tuple[np.ndarray, list[str]]:
    """``(X float32 log1p-CPM, obs_names)`` of one dataset."""
    path = Path(suite_path) / "datasets" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (git lfs pull in the GeneICL checkout?)")
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        obs = [str(o) for o in z["obs_names"].tolist()]
    if X.shape[0] != len(obs):
        raise ValueError(f"{path}: {X.shape[0]} rows but {len(obs)} obs_names")
    return X, obs


def map_genes(
    symbols: list[str],
    gene_ids: str = "auto",
    vocab=None,
    mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str], str]:
    """Columns to keep and their names in the identifier space the model reads.

    ``symbol`` keeps the symbols; ``ensembl`` maps them through ``mapping``
    (``gene_mappings/symbol_to_ensembl.json`` by default), dropping unmapped symbols
    and every symbol after the first that maps to the same Ensembl ID. ``auto``
    picks whichever space overlaps ``vocab`` more. Returns ``(column indices, names,
    space)``.
    """
    if gene_ids not in GENE_IDS:
        raise ValueError(f"gene_ids must be one of {GENE_IDS}, got {gene_ids!r}")
    keep_sym = np.arange(len(symbols))

    def to_ensembl():
        table = mapping if mapping is not None else json.loads(SYMBOL_TO_ENSEMBL.read_text(encoding="utf-8"))
        cols, names, seen = [], [], set()
        for i, s in enumerate(symbols):
            ens = table.get(s)
            if ens and ens not in seen:
                seen.add(ens)
                cols.append(i)
                names.append(ens)
        return np.asarray(cols, dtype=np.int64), names

    if gene_ids == "symbol":
        return keep_sym, list(symbols), "symbol"
    if gene_ids == "ensembl":
        cols, names = to_ensembl()
        return cols, names, "ensembl"
    if vocab is None:
        raise ValueError("gene_ids=auto needs the model vocabulary")
    vocab_keys = set(vocab)
    cols, names = to_ensembl()
    n_sym = len(vocab_keys.intersection(symbols))
    n_ens = len(vocab_keys.intersection(names))
    if n_ens > n_sym:
        return cols, names, "ensembl"
    return keep_sym, list(symbols), "symbol"
