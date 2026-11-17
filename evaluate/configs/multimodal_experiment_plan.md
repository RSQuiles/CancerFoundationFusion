# Multimodal experiment program: does sc (+pseudobulk) help bulk downstream performance?

## Context
Goal of the thesis: decide whether training on single-cell + bulk (+ pseudobulk) improves
downstream bulk performance (GeneICL), and if it does not, explain why from the latent
space. Existing runs (leiden_s0, 8 layers / 256-d / 1200 genes, mse, both, MVC):
- `bulk_pretrain` (B): bulk only, from scratch (quantile binning, the default).
- `sc_quantile`, `sc_midrank`, `sc_mr_{ordinal,corn,mvc5,ordinal_mvc5}` (S*): sc only, from scratch.
- `bulk_then_sc`, `bulk_then_sc75_bulk25`: warm-started from B (epoch 20), sc-only and
  sc+bulk (`--unified`, bulk 0.25, pb 0).

Decisions from the user: SB and SBP from scratch; alignment variants warm-start from SBP.
Precomputed pseudobulks with pair ids exist for leiden_s0.

## 1. Objectives implemented and what each tests
All of them act on the CLS embedding. Modality is a label-only condition
(`where_condition: none`), so the encoder never sees which modality a row is.

| Objective | Flag | Signal | Granularity | What it tests |
|---|---|---|---|---|
| Reconstruction (expr + MVC + gen) | always | masked bins | per sample | Is the data useful by itself (shared weights, no explicit alignment)? |
| MMD | `--mmd` | multi-RBF MMD, bulk vs pb | marginal, no tissue | Is closing the global domain gap enough? Can blur tissue structure if tissue mixes differ |
| CDD | `--cdd [--cdd-class-aware]` | intra minus inter tissue MMD (pb vs bulk); MMD warmup, then ramp | class-conditional (given tissue labels) | Does tissue-matched alignment transfer sc semantics to bulk? |
| Contrastive | `--contrastive-training` | InfoNCE (every bulk is a positive for every pb; sc are negatives) + VICReg | marginal, no tissue | Same question as MMD, plus pushing sc away; watch for loss of tissue structure |
| DAT | `--do-dat --dat-columns modality` | gradient-reversal modality discriminator | marginal, adversarial | Domain invariance by adversary |
| Aggregation consistency | `--agg-consistency` | pb embedding ≈ mean of its cells' embeddings | sc ↔ pb (structural) | Does the bridge sc→pb make embeddings compositional, which is what lets sc information reach bulk through pb? |
| Paired alignment | `--paired-sampling` + paired loss | MSE bulk ↔ its own pb (↔ mean of its cells) | sample-level | The strongest bridge: the same biological sample in two modalities |

Logic of the chain: sc → (aggregation) → pb → (alignment) → bulk. A benefit for bulk
needs both links. Every run can log the losses it does not train with
`--monitor-losses mmd cdd contrastive aggregation paired`, so the control runs still
record the domain gap.

## 2. Experiment groups (about 14 new runs, most cut by screening)
The key control is **budget**: B sees only bulk per step; SB/SBP see bulk in 25% of each
batch. GeneICL on B's intermediate checkpoints gives a bulk-exposure learning curve with
no new run. Every multimodal model is compared at matched bulk exposure as well as at its
end.

**Recipe decision first.** B uses quantile binning and S* uses midrank. SB/SBP must use
B's recipe, or B is rerun with the new one (+1 run). Recommended: pick the S* winner's
loss and binning after the sc_pretrain evaluation, and rerun B with it if it differs
(otherwise the comparison confounds data and recipe).

**G1, modality composition (from scratch, same steps as each other)**: 2 runs.
- `sb`: sc + bulk, `--unified`, bulk 0.25, pb 0.
- `sbp`: sc + bulk + pb, bulk 0.25, pb 0.25 (equal so paired sampling stays possible later),
  **precomputed pseudobulks** (`precomputed_pb: true`, `pb_label: pseudobulk`,
  `pb_id_column: pseudobulk_id`, `pb_group_column: tissue_general`), no alignment.
  Monitors every alignment loss. Notes: the pb pool is finite (one row per real
  pseudobulk, unlike on-the-fly augmentation), so check its size against the steps
  (repetition / overfitting); aggregation consistency then uses only pseudobulks whose
  cells are in the split and the `mean` reduction.
- Both read one new mix `sc_bulk_pb_leiden_s0` (sc + bulk + pb train parts, obs
  `tissue_general assay paired pseudobulk_id`), so SB and SBP differ only in `pb_ratio`.
- Answers: does sc help bulk (`sb` vs B, vs S), and does pb as an extra input modality help
  by itself (`sbp` vs `sb`)?

**G2, alignment screen (warm start from `sbp`, N extra epochs each)**: 5 runs.
- `sbp_cont`: no new loss. Control for the extra steps.
- `sbp_mmd`, `sbp_cdd` (class-aware, with the MMD warmup built in), `sbp_contrastive`,
  `sbp_dat` (modality, `dat_start_step: 0`).
- One factor at a time against `sbp_cont`; no factorial.

**G3, aggregation (warm start from `sbp`)**: 2 runs.
- `sbp_agg` (agg alone) and `sbp_<best>_agg` (best G2 alignment + agg): a 2×2 with
  `sbp_cont` and `sbp_<best>` gives the main effects and the interaction.

**G4, paired (warm start from `sbp`)**: 1 run.
- `sbp_<best combo>_paired`: `paired_sampling`, `paired_every_n` 10, paired loss on.
- Precomputed pb rows feed both the standard and the paired batches.
- Check first that the paired test bulk samples are not in train (the split used
  `--group-by paired:0`).

**G4b, joint confirmation (from scratch, same budget as `sb`/`sbp`)**: 2 runs.
- The winning combination trained jointly from step 0, using the losses' own warmups
  (the CDD MMD→CDD ramp, `dat_start_step`) instead of a separate phase; optionally the
  runner-up.
- Phase screens are cheap and paired (same start, so differences belong to the loss), but
  they can understate a loss that shapes the geometry early. The headline comparison
  against B / `sb` / `sbp` at a matched budget comes from these runs.

**G5, noise floor**: 2 runs.
- Second seed of `sbp_cont` and of the final winner. Without it a GeneICL delta of
  ~0.01 AUROC cannot be read.

Running order: G1 → evaluate → G2 → pick best → G3 → G4 → G5. A "no effect" in G1 still
justifies G2: alignment is the mechanism that could make sc information usable for bulk.

## 3. Analysis
**Downstream (primary).**
- GeneICL with `embed_suite` / `run_probe` / `compare` / `plot_benchmark`.
- References: B (and its checkpoint curve), `pca`, `bulkformer`.
- Reported as per-task deltas against B and against `sbp_cont`, with a Wilcoxon
  signed-rank test across the 52 tasks and the fraction of tasks improved. This is new:
  a small `evaluate/geneicl/paired_stats.py` reading compare's per-task table.

**Fidelity.**
- Reconstruction on `bulk:leiden_s0:test:r25` and `sc:leiden_s0:test:r25`, plus a pb plan.
- Detects negative transfer to bulk and checks that alignment does not break
  reconstruction.

**Latent space (the "why").**
- Existing tools:
  - per-modality UMAP/PCA (`own_panel`);
  - joint `plot_alignment` (sc/pb/bulk, one panel, `--connect-pairs` for the paired test samples);
  - `scib_benchmark` with batch = modality (bulk vs pb) and label = tissue: iLISI/BRAS
    for mixing, silhouette/NMI/ARI/isolated labels for tissue structure.
- New, `evaluate/representation/latent_diagnostics.py`, one script over
  `embeddings/<model>/`:
  1. **Cross-modal transfer:** a tissue logistic probe trained on pb, tested on bulk (and
     the reverse). It measures whether the alignment is semantically correct, not just
     overlapping.
  2. **Residual domain gap:** a linear bulk-vs-pb probe, with accuracy and AUROC.
  3. **Per-tissue bulk–pb centroid distance** relative to the within-tissue spread.
  4. **Paired-sample distance:** each bulk's distance to its own pb, as a rank among
     all pbs.
  5. **Effective rank / participation ratio** of the bulk embeddings, to catch collapse
     from MMD, contrastive or DAT.
  6. **Linear CKA** of each model's GeneICL bulk embeddings with B's: how much the bulk
     geometry moved.
- Across runs, correlate these with the GeneICL deltas: about 12 points is weak, so it is
  descriptive only.

**Hypotheses for a null result, each mapped to its diagnostic:**
- (a) **Budget dilution:** the learning curve of B at matched exposure.
- (b) **sc carries no bulk-relevant signal:** S* GeneICL vs `pca`; CKA.
- (c) **The modalities stay separate:** domain probe and iLISI in G1.
- (d) **Alignment overlaps the clouds without matching tissues:** the domain probe falls
  but cross-modal transfer does not rise (expected for MMD and contrastive, less so for
  CDD and paired).
- (e) **Alignment erases tissue structure:** silhouette and isolated labels drop,
  effective rank drops.
- (f) **pb ≠ real bulk** (composition, protocol): paired distance stays large even with
  the paired loss.

## 4. Deliverables once approved
1. `data_preprocess/build_mix.py --name sc_bulk_pb_leiden_s0 --source single_cell:leiden_s0 bulk:leiden_s0 pseudobulk:leiden_s0 --obs-columns tissue_general assay paired pseudobulk_id --pb-id-column pseudobulk_id` (if the pseudobulk split is not built yet: `split_train_test.py --modality pseudobulk --split-name leiden_s0 --inherit-from ...`).
2. `experiments/configs/multimodal_baselines.json` (`sb`, `sbp`, optional `bulk_<recipe>` rerun), sharing the architecture and step budget of B.
3. `experiments/configs/pb_alignment.json` (G2–G5, `init_from: {"model": "sbp", "checkpoint": "last"}`, same extra epochs; `bulk_ratio = pb_ratio`; `conditions: [tissue_general]` as label for CDD).
4. `evaluate/configs/multimodal_evaluation.yaml`: embed (joint + own_panel, sc/bulk/pb test), UMAP/alignment, scIB, reconstruction on three plans, GeneICL for all + B checkpoints, compare/plots.
5. New scripts: `evaluate/representation/latent_diagnostics.py`, `evaluate/geneicl/paired_stats.py`, each with an offline check and a coordinator step.

## Verification
- `python -m experiments.launch --config <cfg> --dry-run` for both launch configs (resolved store, warm-start line, sbatch line).
- One short smoke run of `sbp` and `sbp_<best>_paired` (`epoch_size` small) in the container: batch composition log (n_bulk/n_pb/n_sc), all monitored losses present in `metrics.csv`, and that `pb_ratio 0` on the 3-modality mix never draws pb rows.
- `python -m evaluate.run_evaluation --config evaluate/configs/multimodal_evaluation.yaml --preflight-only`.
- Offline checks of the new analysis scripts on synthetic embeddings with a known answer (e.g., two shifted Gaussians: domain probe ~1, transfer ~ chance).
