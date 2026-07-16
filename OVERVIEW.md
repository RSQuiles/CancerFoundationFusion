# CancerFoundationFusion — Alignment Overview

## Codebase structure

This is a **PyTorch Lightning** codebase whose core deliverable is the in-house
foundation model itself (**CancerFoundation "Fusion"**), plus a Hydra-based
downstream-evaluation framework layered on top. Training is `argparse`/JSON-config
driven; downstream eval is Hydra-driven.

Main parts:

- `pretrain.py` — central training entry point (argparse; `utils_config.py` holds ~80 hyperparameters).
- `cancerfoundation/model/model.py` — `CancerFoundation` LightningModule (training loop, losses, and the `embed()` API used downstream).
- `cancerfoundation/model/module.py` / `layers.py` — core scGPT-style transformer backbone + custom attention/generator variants.
- `cancerfoundation/data/` — memory-mapped h5ad dataset, masking/binning collator, balanced sampler, **paired bulk/SC** data + collator.
- `evaluate/finetune/` — downstream task runners (`run_downstream_task.py`, `base_downstream_runner.py`) + plugin task registry under `tasks/`.
- `ablate/` — config-driven ablation runner (JSON base config + per-ablation overrides, local or SLURM).
- `submits_biomed/`, `submits_cscs/` — SLURM scripts (LeoMed Singularity; CSCS Alps Enroot/Pyxis).
- `data_preprocess/` — h5ad → mem-map conversion, bulk & paired bulk/SC preprocessing, ESM/RNABert gene embeddings.
- Analysis: `embed.ipynb` (cell embeddings), `evaluate/plot/` (umaps/utils).

## Main in-house model

scGPT-style transformer adapted for **joint single-cell + bulk + pseudobulk**
RNA-seq (the "Fusion"/Unified FM is the novel contribution).

Core setup (current default, from `debug.sh`):

- Gene vocabulary: **28,725 genes** (`cancerfoundation/assets/vocab.json`).
- Max sequence length: **1,200 tokens** (incl. CLS); truncation by sampling.
- Transformer: **6 layers, 8 heads, embedding dim 128, FF dim 256**; configurable pre/post-LayerNorm, LayerNorm/RMSNorm, relu/gelu/swiglu.
- Expression discretized into **51 bins**; input styles `binned`/`log1p`/`normed_raw`, value-encoding `mine`/`theirs`.
- Objectives: masked binned-expression reconstruction (`pcpt`) + generation (`gen`) + CLS/MVC. Losses: MSE / ordinal-CE / CORN / ZINB; optional explicit zero-probability modeling.
- During pretraining, genes are **sampled from the full vocabulary** (not fixed HVGs), with controllable zero-expression sampling (`--zero-percentages`); no HVG filtering.

**Unified/Fusion-specific machinery** (`--unified`): mixed batches of
SC/bulk/pseudobulk (`--bulk-ratio`, `--pb-ratio`), on-the-fly pseudobulk
aggregation, and alignment losses — **contrastive** (pseudobulk↔bulk),
**aggregation-consistency** (pseudobulk ≈ mean of constituent cells),
**paired-alignment** (1-to-1 PB↔bulk), and **DAT** (gradient-reversal
domain-adversarial training over modality/technology). Per-loss weights are
exposed. Optional ESM/RNABert gene embeddings, denoising, and metadata
conditioning.

## Downstream tasks

Plugin tasks (registered via `TaskRegistry`, under `evaluate/finetune/tasks/`):

- **Cancer type classification** — TCGA bulk/SC (`canc_type_class`).
- **Deconvolution** — pseudobulk → cell-type proportions (`deconv`).
- **Drug sensitivity** — expression + drug features (`drug_sensitivity_v2`).
- **Proteome prediction** — expression → protein abundance (`proteome_pred`).
- **Survival** — SurvBoard-style Cox/survival-function evaluation (`survival`, with a separate `evaluate_survboard_metrics.py` for pycox/sksurv C-index).

## Finetuning modes

Two modes (a single `finetune` boolean), **no adapters**:

- **Frozen** — embedder frozen, train only the head (embeddings precomputed via `embed_for_finetune`).
- **Full fine-tuning** — backbone trainable, raw expression flows through for end-to-end gradients.

Heads are MLP prediction heads (`components.EmbeddingPredHead`) on CLS / pooled
gene embeddings; balanced loss for imbalanced classification. DDP + checkpointing
handled in `BaseDownstreamRunner`.

## Baselines

- **PCA + downstream head** — `PCAEmbedder` (StandardScaler → PCA, refit per fold) as a drop-in replacement for the transformer embedder, used for both our model and as a raw-expression baseline.

## Gene selection / preprocessing

**Modality-aware HVG selection** (`model._select_genes`, deterministic, on log1p data):

- **single-cell** → scanpy `seurat` dispersion HVG;
- **bulk / pseudobulk** → **log1p + MAD** (robust to composition-driven variance);

modality is inferred per-row (`_classify_modality`) so mixed AnnData is embedded
group-by-group with its own gene set. Pretraining uses no HVG filter (random gene
sampling from the full vocab).

## Outputs and analysis

Downstream runs write per-task metrics/config/predictions (with per-fold results);
ablations fan out one run per override. Embeddings via `embed.ipynb`; UMAP/plots
via `evaluate/plot/`.

## Main caveats to communicate

- The **novel contribution is on the pretraining side** (the Fusion/Unified multi-modality alignment), so alignment with the collaborator's framework is cleanest at the **downstream-eval and embedding-comparison** layer.
- Vocab (~28.7k) and default architecture (6L/8H/128d) **differ from the 13,004-gene / 256-dim setup** — a shared-gene intersection and dim-normalization are needed before head-to-head embedding comparison.
- Finetuning is **frozen-head vs full-FT only** (no adapter mode) — a gap relative to a three-way `head_only`/`adapters`/`full_ft` comparison.
- HVG is **modality-aware (seurat for SC, MAD for bulk)** rather than a single MAD scheme; worth reconciling so both codebases select genes identically on bulk.
- Survival is SurvBoard-style with an external-env metrics step (pycox/sksurv), so it's not directly the BulkRNABert Cox benchmark.
