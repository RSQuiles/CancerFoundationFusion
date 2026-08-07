"""Config schema for the post-training analysis orchestrator (``run_analysis.py``).

One config describes N experiment directories and which analysis steps to run over
each. Per-experiment blocks are deep-merged on top of ``defaults``, so the common
case is a short ``experiments:`` list and nothing else.

See ``evaluate/example_analysis_config.yaml`` for a documented example.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Steps in dependency order. ``run_analysis`` executes whatever subset an experiment
# asks for, but always in this order.
STEP_ORDER: tuple[str, ...] = (
    "build",
    "metrics",
    "scib",
    "diagnose",
    "umap",
    "paired_umap",
    "downstream",
    "benchmark",
)

# Steps enabled unless an experiment says otherwise.
#   ``paired_umap`` is opt-in: it needs a separate paired h5ad (cell-line matched SC
#     + bulk) that most datasets do not have, and it errors out rather than no-oping.
#   ``downstream`` is opt-in: it fine-tunes and evaluates every model on every
#     requested task, which is hours of GPU time and needs task datasets (TCGA,
#     CPTAC, PRISM, ...) that an arbitrary ablation may not have.
_OPT_IN_STEPS: frozenset[str] = frozenset({"paired_umap", "downstream"})
DEFAULT_STEPS: tuple[str, ...] = tuple(s for s in STEP_ORDER if s not in _OPT_IN_STEPS)

_DEFAULT_DEFAULTS: dict[str, Any] = {
    "steps": list(DEFAULT_STEPS),
    "seed": 0,
    # False -> the underlying CLIs get --not-normalized. The repo's on-disk data is
    # raw counts and binning is scale-invariant, so True (embed as-is, matching
    # training) is the right default; see CancerFoundation.embed.
    "normalized": True,
    "build": {
        "rebuild": "auto",          # auto | always | never
        # Max cells PER MODALITY (per filename prefix), NOT a total: with sc + bulk +
        # paired_sc + paired_pb + paired_bulk present, 75_000 here means ~300k rows in
        # eval.h5ad. Named for the unit because conflating it with umap.sample_size
        # (a total) is an easy and expensive mistake.
        "modality_sample_size": 5000,
        "precomputed_pb": True,
        "panel_strategy": "consensus",
        "per_modality_panel": False,
        "n_pca_components": 50,
        "n_hvg_baseline": 2000,
        "batch_size": 64,
    },
    "metrics": {
        "batch_size": 64,
        "group_column": "tissue_general",
        "skip_existing": True,
        "mask_ratio": 0.15,
        "n_sc_per_pb": 10,
        "n_synth_pb": 200,
    },
    "scib": {"n_max": 500},
    "diagnose": {"n_max": 500, "out_csv": None},
    "umap": {
        "source": "eval",           # eval | live
        "sample_size": 6000,        # TOTAL cells; only used when source == "live"
        "color": ["tissue_general"],
        "skip_unknown": True,
        "neighbors": 15,
        "min_dist": 0.5,
        "plot_pb_only": False,
        "no_modality_split": False,
        "adata_prefix": None,
        "embed_batch_size": 64,
    },
    # plot_paired_umap.py --ablation-dir. Only the keys below reach run_ablation():
    # --out-prefix / --domain-col / --sc-label / --bulk-label / --is-log1p /
    # --max-sc-per-line exist on the CLI but are ignored in ablation-dir mode, so they
    # are deliberately absent here rather than offered and silently dropped.
    "paired_umap": {
        "input_h5ad": None,     # required when this step runs; no sensible default
        "out_dir": None,        # None -> {ablation_dir}/paired_umaps
        "cell_line_col": "cell_line",
        "n_cell_lines": 50,
        "batch_size": 64,
        "neighbors": 15,
        "min_dist": 0.5,
        "dpi": 200,
        "no_sc": False,
        "device": None,
    },
    # run_ablation_downstream.py: fine-tune + evaluate every model on every task,
    # producing the {model}/metrics/results_{task}.json that 'benchmark' then plots.
    # Opt-in; 'tasks' is required once the step is enabled.
    "downstream": {
        "tasks": [],            # e.g. [canc_type_class, deconv, survival]
        "config_dir": None,     # None -> evaluate/finetune/configs
        "models": None,         # None -> every model dir with a checkpoint
        "skip_existing": True,
        "pca_baseline": False,
        "pca_n_components": 128,
        # NOTE the polarity, and note that this is NOT the experiment-level
        # 'normalized' key above. This one is in "do it?" space:
        #   true  -> CP10K+log1p is applied to every task's input (skipped, with a
        #            warning, where the matrix does not look like raw counts)
        #   false -> nothing is applied at any point, for any task
        #   null  -> each task config's own 'normalize:' key decides
        # 'normalized' (above) means the opposite thing: "the data is ALREADY
        # normalized, so tell build/metrics to skip normalising it".
        "normalize": False,
    },
    "benchmark": {"output": None, "no_show": True, "figsize": None, "primary": {}},
}

_VALID_REBUILD = ("auto", "always", "never")
_VALID_UMAP_SOURCE = ("eval", "live")
_VALID_SLURM_MODE = ("per-experiment", "single-job")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Where each kind of step runs.

    The steps do not all live in the same environment: ``build`` / ``metrics`` /
    ``umap`` need the bionemo container (``umaps.py`` imports ``cancerfoundation`` at
    module level), while ``scib`` needs the conda env that has ``scib_metrics``, which
    the container lacks. ``diagnose`` and ``benchmark`` run anywhere.
    """

    container_image: str | None = None
    binds: list[str] = field(default_factory=lambda: ["/cluster"])
    conda_env: str | None = None
    python: str = "python"
    nv: bool = True


@dataclass
class SlurmConfig:
    enabled: bool = False
    mode: str = "per-experiment"
    sbatch_args: list[str] = field(default_factory=list)
    job_name_prefix: str = "analysis"
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """One ablation directory plus the resolved options for each step."""

    name: str
    ablation_dir: Path
    adata_dir: Path | None
    steps: list[str]
    seed: int
    normalized: bool
    build: dict[str, Any]
    metrics: dict[str, Any]
    scib: dict[str, Any]
    diagnose: dict[str, Any]
    umap: dict[str, Any]
    paired_umap: dict[str, Any]
    downstream: dict[str, Any]
    benchmark: dict[str, Any]

    @property
    def eval_adata(self) -> Path:
        return self.ablation_dir / "eval.h5ad"

    @property
    def analysis_dir(self) -> Path:
        """Where payloads, logs and the result JSON go. Never holds real outputs."""
        return self.ablation_dir / "analysis"


@dataclass
class AnalysisConfig:
    experiments: list[ExperimentConfig]
    env: EnvConfig
    slurm: SlurmConfig
    repo_root: Path
    config_path: Path | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_config_file(path: Path) -> dict[str, Any]:
    """Read a .json / .yaml / .yml config into a dict.

    Mirrors ``ablate/config.py:_read_config_file`` so both config families behave the
    same way, including the "YAML asked for but PyYAML missing" message.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Use JSON or install pyyaml."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format '{suffix}'. Use .json, .yaml, or .yml.")


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate(obj: Any, variables: dict[str, str]) -> Any:
    """Recursively expand ``${name}`` using ``variables``, then ``os.environ``.

    Note ``${work}`` in ``ablate/configs/*.json`` is expanded by nothing in the repo
    and survives as a literal path component; this loader actually resolves it.
    """
    if isinstance(obj, str):
        def _sub(m: re.Match) -> str:
            key = m.group(1)
            if key in variables:
                return str(variables[key])
            if key in os.environ:
                return os.environ[key]
            raise KeyError(
                f"Config references ${{{key}}} but it is defined neither in 'vars:' "
                f"nor in the environment."
            )
        # Loop so a var may itself contain ${...}; bounded to avoid cycles.
        out = obj
        for _ in range(10):
            new = _VAR_RE.sub(_sub, out)
            if new == out:
                return new
            out = new
        raise ValueError(f"Variable interpolation did not converge for: {obj!r}")
    if isinstance(obj, dict):
        return {k: interpolate(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate(v, variables) for v in obj]
    return obj


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins. Lists replace, they do not append."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _resolve_vars(raw: dict[str, Any]) -> dict[str, str]:
    """Expand the ``vars:`` block against itself so later entries may use earlier ones."""
    variables: dict[str, str] = {}
    for key, value in (raw.get("vars") or {}).items():
        variables[str(key)] = str(interpolate(str(value), variables))
    return variables


def load_analysis_config(config_path: Path | str) -> AnalysisConfig:
    """Load, interpolate, merge and validate an analysis config."""
    config_path = Path(config_path).expanduser().resolve()
    raw = read_config_file(config_path)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} must contain a mapping at the top level.")

    variables = _resolve_vars(raw)
    raw = interpolate({k: v for k, v in raw.items() if k != "vars"}, variables)

    repo_root = Path(__file__).resolve().parents[1]

    env_raw = raw.get("env") or {}
    env = EnvConfig(
        container_image=env_raw.get("container_image"),
        binds=[str(b) for b in (env_raw.get("binds") or ["/cluster"])],
        conda_env=env_raw.get("conda_env"),
        python=str(env_raw.get("python", "python")),
        nv=bool(env_raw.get("nv", True)),
    )

    slurm_raw = raw.get("slurm") or {}
    slurm = SlurmConfig(
        enabled=bool(slurm_raw.get("enabled", False)),
        mode=str(slurm_raw.get("mode", "per-experiment")),
        sbatch_args=[str(a) for a in (slurm_raw.get("sbatch_args") or [])],
        job_name_prefix=str(slurm_raw.get("job_name_prefix", "analysis")),
        environment={str(k): str(v) for k, v in (slurm_raw.get("environment") or {}).items()},
    )
    if slurm.mode not in _VALID_SLURM_MODE:
        raise ValueError(
            f"slurm.mode must be one of {_VALID_SLURM_MODE}, got {slurm.mode!r}"
        )

    defaults = deep_merge(_DEFAULT_DEFAULTS, raw.get("defaults") or {})

    experiments_raw = raw.get("experiments")
    if not experiments_raw:
        raise KeyError("Config must define a non-empty 'experiments' list.")

    experiments: list[ExperimentConfig] = []
    seen: set[str] = set()
    for item in experiments_raw:
        if not isinstance(item, dict):
            raise TypeError(f"Each experiment must be a mapping, got {type(item)}")
        if "ablation_dir" not in item:
            raise KeyError(f"Experiment {item.get('name', '<unnamed>')} needs 'ablation_dir'.")

        ablation_dir = Path(str(item["ablation_dir"]))
        name = str(item.get("name") or ablation_dir.name)
        if name in seen:
            raise ValueError(f"Duplicate experiment name {name!r}.")
        seen.add(name)

        merged = deep_merge(defaults, {k: v for k, v in item.items()
                                      if k not in ("name", "ablation_dir", "adata_dir")})
        steps = [str(s) for s in merged.get("steps", [])]
        unknown = [s for s in steps if s not in STEP_ORDER]
        if unknown:
            raise ValueError(
                f"[{name}] unknown step(s) {unknown}; valid steps: {list(STEP_ORDER)}"
            )
        # Always execute in dependency order regardless of how they were listed.
        steps = [s for s in STEP_ORDER if s in steps]

        if "sample_size" in (item.get("build") or {}) or "sample_size" in (
            (raw.get("defaults") or {}).get("build") or {}
        ):
            raise KeyError(
                f"[{name}] 'build.sample_size' was renamed to "
                "'build.modality_sample_size': it is a per-modality cap, not a total "
                "(sc + bulk + paired_* each get that many cells, so the eval.h5ad row "
                "count is roughly that times the number of modalities). Rename the "
                "key; the meaning is unchanged."
            )

        if merged["build"]["rebuild"] not in _VALID_REBUILD:
            raise ValueError(
                f"[{name}] build.rebuild must be one of {_VALID_REBUILD}, "
                f"got {merged['build']['rebuild']!r}"
            )
        if merged["umap"]["source"] not in _VALID_UMAP_SOURCE:
            raise ValueError(
                f"[{name}] umap.source must be one of {_VALID_UMAP_SOURCE}, "
                f"got {merged['umap']['source']!r}"
            )

        adata_dir = Path(str(item["adata_dir"])) if item.get("adata_dir") else None
        needs_adata = "build" in steps or (
            "umap" in steps and merged["umap"]["source"] == "live"
        )
        if needs_adata and adata_dir is None:
            raise KeyError(
                f"[{name}] needs 'adata_dir': it runs 'build' and/or 'umap' with "
                "source: live, both of which read the raw h5ads."
            )

        if "downstream" in steps and not merged["downstream"].get("tasks"):
            raise KeyError(
                f"[{name}] runs the 'downstream' step but 'downstream.tasks' is empty. "
                "List the downstream tasks to evaluate, e.g. "
                "downstream: {tasks: [canc_type_class, deconv]}. Without them "
                "run_ablation_downstream.py has nothing to do."
            )

        if "paired_umap" in steps and not merged["paired_umap"].get("input_h5ad"):
            raise KeyError(
                f"[{name}] runs the 'paired_umap' step but 'paired_umap.input_h5ad' is "
                "unset. That step needs a single paired h5ad holding cell-line matched "
                "SC and bulk observations (e.g. .../processed/paired_samples.h5ad); it "
                "does not read adata_dir or eval.h5ad."
            )

        experiments.append(
            ExperimentConfig(
                name=name,
                ablation_dir=ablation_dir,
                adata_dir=adata_dir,
                steps=steps,
                seed=int(merged["seed"]),
                normalized=bool(merged["normalized"]),
                build=merged["build"],
                metrics=merged["metrics"],
                scib=merged["scib"],
                diagnose=merged["diagnose"],
                umap=merged["umap"],
                paired_umap=merged["paired_umap"],
                downstream=merged["downstream"],
                benchmark=merged["benchmark"],
            )
        )

    return AnalysisConfig(
        experiments=experiments,
        env=env,
        slurm=slurm,
        repo_root=repo_root,
        config_path=config_path,
    )
