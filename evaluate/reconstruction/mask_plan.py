"""Build a model-independent mask plan for a reconstruction benchmark.

A plan fixes, for each evaluated sample, which genes are **masked** (to be
reconstructed) and which form the **context** a sequence model such as
CancerFoundation sees. Every model is then scored on exactly the same masked
(sample, gene) entries, whatever its input format.

Per sample it mirrors CancerFoundation's "both" training mode: ``seq_len - 1`` genes
(CLS takes the first position) are drawn from the gene universe, uniformly or with a
share of zeros as ``zero_percentages`` does, and ``int((seq_len - 1) * mask_ratio)``
of them are masked; the rest are the context. Zeros can be masked too, as in training.

    # a split part under the experiments layout -> data/<mod>/splits/<split>/masks/<part>_<name>.npz
    python -m evaluate.reconstruction.mask_plan --modality bulk --split leiden_s0 --part test \\
        --name r25 --mask-ratio 0.25 --n-samples 2000 --stratify-by tissue_general
    # any h5ad
    python -m evaluate.reconstruction.mask_plan --input data.h5ad --out plan.npz --name r25

``--gene-universe genes.txt`` restricts the genes that can be drawn (e.g. those every
compared model knows), so no model has to drop masked entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate._cli import parse_with_config  # noqa: E402

PLAN_FORMAT = 1


@dataclass
class MaskPlan:
    """A loaded plan. Gene indices point into ``genes``; row ``i`` is ``obs_names[i]``."""

    obs_names: np.ndarray
    genes: np.ndarray
    masked_ptr: np.ndarray
    masked_idx: np.ndarray
    context_ptr: np.ndarray
    context_idx: np.ndarray
    meta: dict
    path: Path | None = None

    @property
    def n_samples(self) -> int:
        return len(self.obs_names)

    def masked(self, i: int) -> np.ndarray:
        return self.masked_idx[self.masked_ptr[i]:self.masked_ptr[i + 1]]

    def context(self, i: int) -> np.ndarray:
        return self.context_idx[self.context_ptr[i]:self.context_ptr[i + 1]]

    @property
    def plan_id(self) -> str:
        return str(self.meta.get("plan_id") or (self.path.stem if self.path else "plan"))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs_names=self.obs_names.astype(str), genes=self.genes.astype(str),
            masked_ptr=self.masked_ptr, masked_idx=self.masked_idx,
            context_ptr=self.context_ptr, context_idx=self.context_idx,
            meta=np.array(json.dumps(self.meta)),
        )
        return path


def load_plan(path: str | Path) -> MaskPlan:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != PLAN_FORMAT:
            raise ValueError(f"{path}: unsupported mask plan format {meta.get('format')!r}")
        return MaskPlan(
            obs_names=data["obs_names"], genes=data["genes"],
            masked_ptr=data["masked_ptr"], masked_idx=data["masked_idx"],
            context_ptr=data["context_ptr"], context_idx=data["context_idx"],
            meta=meta, path=path,
        )


def resolve_plan_path(spec: str, root: Path | None = None) -> Path:
    """``MODALITY:SPLIT:PART:NAME`` under the layout, or a path to a ``.npz``."""
    if spec.endswith(".npz") or Path(spec).exists():
        return Path(spec)
    fields = spec.split(":")
    if len(fields) != 4:
        raise ValueError(f"--plan {spec!r} must be MODALITY:SPLIT:PART:NAME or a .npz path")
    from experiments.layout import ExperimentLayout, canonical_modality

    mod, split, part, name = fields
    return ExperimentLayout.from_root(root).mask_plan(canonical_modality(mod), split, part, name)


def sample_plan_row(values: np.ndarray, seq_len: int, mask_ratio: float, rng: np.random.Generator,
                    zero_fraction: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(masked, context)`` positions into ``values`` (one sample over the gene universe)."""
    n = len(values)
    length = min(seq_len - 1, n)
    if zero_fraction is None:
        chosen = rng.choice(n, size=length, replace=False)
    else:
        zeros, nonzeros = np.flatnonzero(values == 0), np.flatnonzero(values != 0)
        n_zero = min(int(round(zero_fraction * length)), len(zeros))
        n_nonzero = min(length - n_zero, len(nonzeros))
        n_zero = min(length - n_nonzero, len(zeros))  # top up with zeros if nonzeros ran short
        chosen = np.concatenate([rng.choice(zeros, n_zero, replace=False),
                                 rng.choice(nonzeros, n_nonzero, replace=False)])
        chosen = rng.permutation(chosen)
    n_masked = int(length * mask_ratio)
    if n_masked < 1:
        raise ValueError(f"mask_ratio={mask_ratio} masks no gene of a {length}-gene sequence")
    return np.sort(chosen[:n_masked]), np.sort(chosen[n_masked:])


def build_plan(adata, genes: list[str], seq_len: int, mask_ratio: float, seed: int = 0,
               zero_fraction: float | None = None, meta: dict | None = None) -> MaskPlan:
    """Plan for every row of ``adata`` over ``genes`` (a subset of ``adata.var_names``)."""
    import scipy.sparse as sp

    if not 0 < mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    if seq_len < 3:
        raise ValueError(f"seq_len must be at least 3, got {seq_len}")
    positions = adata.var_names.get_indexer(genes)
    if (positions < 0).any():
        raise ValueError("Every universe gene must be an input gene")
    rng = np.random.default_rng(seed)
    masked_ptr, context_ptr = [0], [0]
    masked_idx, context_idx = [], []
    for i in range(adata.n_obs):
        row = adata.X[i]
        row = np.asarray(row.toarray() if sp.issparse(row) else row).ravel()[positions]
        masked, context = sample_plan_row(row, seq_len, mask_ratio, rng, zero_fraction)
        masked_idx.append(masked)
        context_idx.append(context)
        masked_ptr.append(masked_ptr[-1] + len(masked))
        context_ptr.append(context_ptr[-1] + len(context))
    return MaskPlan(
        obs_names=adata.obs_names.to_numpy().astype(str), genes=np.asarray(genes, dtype=str),
        masked_ptr=np.asarray(masked_ptr, dtype=np.int64),
        masked_idx=np.concatenate(masked_idx).astype(np.int32),
        context_ptr=np.asarray(context_ptr, dtype=np.int64),
        context_idx=np.concatenate(context_idx).astype(np.int32),
        meta={"format": PLAN_FORMAT, "seq_len": seq_len, "mask_ratio": mask_ratio, "seed": seed,
              "zero_fraction": zero_fraction, "n_samples": int(adata.n_obs), "n_genes": len(genes),
              **(meta or {})},
    )


def read_rows(path: Path, names) -> object:
    """The rows ``names`` of an h5ad, in that order, in memory (all genes)."""
    import anndata as ad

    backed = ad.read_h5ad(path, backed="r")
    try:
        index = backed.obs_names.get_indexer(list(names))
        if (index < 0).any():
            missing = [n for n, i in zip(names, index) if i < 0][:5]
            raise KeyError(f"{path.name} lacks {int((index < 0).sum())} requested rows, e.g. {missing}")
        order = np.argsort(index)
        sub = backed[np.sort(index)].to_memory()
    finally:
        backed.file.close()
    return sub[np.argsort(order)].copy()


def _file_digest(path: Path) -> dict:
    stat = path.stat()
    head = hashlib.sha1(path.open("rb").read(1 << 20)).hexdigest()[:16]
    return {"path": str(path.resolve()), "bytes": stat.st_size, "head_sha1": head}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a model-independent reconstruction mask plan.")
    parser.add_argument("--root", type=Path, default=None, help="Experiments root")
    parser.add_argument("--modality", default=None, help="sc | bulk | pseudobulk (layout mode)")
    parser.add_argument("--split", default=None, help="Split name (layout mode)")
    parser.add_argument("--part", default="test", help="Split part whose rows are used (layout mode)")
    parser.add_argument("--input", type=Path, default=None, help="Any h5ad instead of a layout split part")
    parser.add_argument("--out", type=Path, default=None, help="Plan path (required with --input)")
    parser.add_argument("--name", required=False, default=None, help="Plan name, e.g. r25")
    parser.add_argument("--mask-ratio", type=float, default=0.25,
                        help="Share of each sample's sequence that is masked (training: 0.25/0.5/0.75)")
    parser.add_argument("--seq-len", type=int, default=1200, help="Sequence length incl. CLS (model max_seq_len)")
    parser.add_argument("--zero-fraction", type=float, default=None,
                        help="Share of zeros in each drawn sequence (default: uniform draw)")
    parser.add_argument("--gene-universe", type=Path, default=None,
                        help="Text file of gene names that may be drawn (default: all input genes)")
    parser.add_argument("--n-samples", type=int, default=2000, help="Samples to evaluate (0 or less = all)")
    parser.add_argument("--stratify-by", nargs="*", default=["tissue_general"],
                        help="obs columns the sample subset is stratified on (missing ones skipped)")
    parser.add_argument("--filter", nargs="*", default=[], metavar="COL=V1,V2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Path:
    import anndata as ad

    from utils.sampling import filter_mask, parse_filter_specs, stratified_subsample

    args = parse_with_config(build_parser(), "mask_plan", argv)
    if not args.name:
        sys.exit("Error: give --name")
    if args.input is not None:
        if args.out is None:
            sys.exit("Error: --input needs --out")
        source, out, rows, plan_id = args.input, args.out, None, args.name
    else:
        if not (args.modality and args.split):
            sys.exit("Error: give --modality and --split (or --input and --out)")
        from experiments.layout import ExperimentLayout, canonical_modality

        layout = ExperimentLayout.from_root(args.root)
        label = canonical_modality(args.modality)
        source = layout.raw_h5ad(label)
        part_file = layout.split_dir(label, args.split) / f"{args.part}.txt"
        if not part_file.exists():
            sys.exit(f"Error: {part_file} not found")
        rows = part_file.read_text(encoding="utf-8").split()
        out = args.out or layout.mask_plan(label, args.split, args.part, args.name)
        plan_id = f"{label}:{args.split}:{args.part}:{args.name}"
    if out.exists() and not args.overwrite:
        sys.exit(f"Error: {out} exists; pass --overwrite to replace it")

    backed = ad.read_h5ad(source, backed="r")
    try:
        obs = backed.obs.copy()
        var_names = backed.var_names.copy()
    finally:
        backed.file.close()
    if rows is not None:
        obs = obs.loc[obs.index.intersection(rows)]
    filters = parse_filter_specs(args.filter)
    if filters:
        obs = obs[filter_mask(obs, filters, label=source.name)]
    strata = [c for c in args.stratify_by if c in obs.columns]
    n = args.n_samples if args.n_samples and args.n_samples > 0 else None
    names = list(stratified_subsample(obs, n, strata, seed=args.seed))
    print(f"Plan '{plan_id}': {len(names)} samples from {source.name}"
          + (f" (stratified on {strata})" if strata and n else ""))

    genes = list(var_names)
    if args.gene_universe is not None:
        wanted = set(args.gene_universe.read_text(encoding="utf-8").split())
        genes = [g for g in var_names if g in wanted]
        print(f"  gene universe: {len(genes)} of {len(var_names)} input genes ({len(wanted)} listed)")
        if not genes:
            sys.exit("Error: no input gene is in --gene-universe")

    adata = read_rows(source, names)
    plan = build_plan(adata, genes, args.seq_len, args.mask_ratio, args.seed, args.zero_fraction, meta={
        "plan_id": plan_id, "source": _file_digest(source), "part": args.part, "filters": filters,
        "stratify_by": strata, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    per_sample = np.diff(plan.masked_ptr)
    print(f"  {int(per_sample.mean())} masked + {int(np.diff(plan.context_ptr).mean())} context genes per sample")
    print(f"  wrote {plan.save(out)}")
    return out


if __name__ == "__main__":
    main()
