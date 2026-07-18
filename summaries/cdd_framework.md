# Contrastive Domain Discrepancy (CDD) Framework

A class-conditional alignment between **real bulk RNA-seq** and **pseudobulk built from
single cells**, adapted from the Contrastive Adaptation Network (CAN, Kang et al., CVPR
2019) to the CancerFoundation pretraining pipeline. It is fully optional and composes
with the existing unified-mode losses (marginal MMD, DAT, VICReg+InfoNCE contrastive,
paired/aggregation alignment).

- **Source domain** = pseudobulk (`modality == 2`)
- **Target domain** = real bulk (`modality == 0`)
- **Class** = tissue (`tissue_general`)

---

## 1. Goal and the core difficulty

CDD pulls **same-tissue** source/target embeddings together and pushes **different-tissue**
apart. Unlike marginal MMD (which matches the two clouds as wholes, ignoring tissue), CDD
preserves tissue structure across the two data sources.

The difficulty is **labels**:

- **Pseudobulk tissue** comes straight from CellxGene `tissue_general` (inherited from the
  constituent single cells). Reliable.
- **Bulk tissue** is normalized from free-text metadata. A large fraction is the literal
  `"unknown"`, and several bulk tissues (e.g. `esophagus`, `thyroid`, `lymph node`) have
  **no single-cell counterpart** at all.

So a big part of the framework exists to *earn* usable tissue labels for the bulk before it
can participate in the class-conditional loss.

---

## 2. The training timeline

Everything hinges on one boundary, `--cdd-cluster-warmup-steps` (default 2000), followed by
a ramp of `--cdd-ramp-steps` (default 1000).

| Phase | Steps (defaults) | MMD weight | CDD weight | What happens |
|-------|------------------|-----------|-----------|--------------|
| **Warmup** | `0 … 2000` | `--loss-weight-mmd` (full) | `0` | MMD pulls the whole bulk cloud toward the whole pseudobulk cloud, ignoring tissue. |
| **First clustering** | at `2000` | — | — | Bulk tissue labels are inferred for the first time (§4). |
| **Ramp / handover** | `2000 … 3000` | `1.0 → --cdd-mmd-final-weight` (0) | `0 → --loss-weight-cdd` (0.3) | The two losses swap roles smoothly. |
| **Steady state** | `> 3000` | `--cdd-mmd-final-weight` (0) | `--loss-weight-cdd` (0.3) | CDD does the alignment; clustering refreshes once per epoch. |

**Why warmup exists:** the clustering that labels the bulk relies on the bulk and pseudobulk
clouds *overlapping*. At initialization they are separated by modality, so clustering would
be meaningless. Warmup uses marginal MMD to close that gap first. If `--mmd` was not passed,
it is **auto-enabled** for the warmup window (and, since the final weight defaults to 0, it
turns itself back off after the ramp).

**Why hand over rather than stack both:** marginal MMD forces
`P(f(bulk)) ≈ P(f(pseudobulk))` while ignoring class. Because the bulk tissue distribution
differs from the single-cell tissue distribution, perfect marginal alignment *distorts*
class structure to make the mismatched proportions match — the exact failure CAN was written
against. MMD is ideal for the bootstrap and counterproductive once CDD is doing conditional
alignment, so it fades out as CDD fades in.

> **Note:** the schedule applies to MMD whenever `--cdd` is on, *including* an explicitly
> passed `--mmd`. So `--mmd --cdd` decays MMD to zero after the ramp unless you set
> `--cdd-mmd-final-weight`.

---

## 3. Sampling — making batches non-trivial

CDD needs the **same tissues present in both the bulk and pseudobulk halves** of a batch, or
there is nothing to compare. The standard sampler draws bulk and pseudobulk tissues
independently, so matches are rare and the loss degenerates.

**Class-aware sampling** (`--cdd-class-aware`) fixes this per batch:

1. Pick `K` tissues shared by both domains.
2. Build the pseudobulk half tissue-pure from those `K`.
3. **Split the bulk half** via `--cdd-bulk-class-frac` (default 0.6):
   - the matched fraction (0.6) is drawn from the same `K` tissues → CDD has paired classes;
   - the free fraction (0.4) is drawn uniformly from **all** bulk.

**Why the free fraction is essential.** If every bulk slot were reserved for the shared
tissues (the original behaviour, equivalent to `frac = 1.0`), then `"unknown"` and bulk-only
rows would appear in **no batch, ever** — no gradient from any loss, and never embedded for
the clustering meant to recover them. That was a real chicken-and-egg bug:
`--cdd-class-aware --cdd-infer-labels` was effectively a no-op. A regression test confirms
that with `frac = 1.0`, 7 of 15 synthetic bulk rows are starved — exactly the unknown +
bulk-only rows — while `frac = 0.6` reaches all 15.

Paired batches (when `--paired-sampling` is on) **skip CDD** entirely and use the paired
alignment loss instead; CDD runs only on the non-paired batches.

---

## 4. The clustering — inferring bulk tissue labels

Enabled by `--cdd-infer-labels`. Runs once per epoch (or every `--cdd-cluster-interval`
steps), and produces a `pseudo_label` per bulk row: the real label where known, an inferred
label where recovered, `-1` where still ambiguous.

Everything the clustering compares is produced inside **one `no_grad`/`eval` block**, so all
embeddings come from a single model state.

1. **Re-encode all bulk.** Every bulk row is embedded fresh under current weights.
   - This is the key correctness fix over a streamed memory bank: previously bank rows were
     written at different steps by different weights and then compared as if commensurable.
   - It also fills the bank for **every** bulk row regardless of what the sampler drew —
     which is what dissolves the chicken-and-egg from §3.

2. **Re-encode source class means.** For each tissue, draw `--cdd-cluster-source-pb`
   (default 8) pseudobulks, aggregate + embed them, and average → the source mean for that
   tissue. Recomputed each event (not an EMA — see §6), so it shares the bulk's model state.

3. **Seed centroids** (see §5 for the source-seeding detail):
   - *Preferred:* a tissue's centroid = mean of its **known-labeled bulk** (reliable, same
     domain as the rows being clustered).
   - *Fallback:* a tissue known only to the single-cell data has no bulk anchors, so its
     centroid comes from the **source mean, translated into the bulk frame** (§5).

4. **Assign the unlabeled bulk** by spherical k-means (cosine, a few Lloyd iterations) to the
   nearest centroid. Two purity filters:
   - too far from every centroid (`1 − cos > --cdd-cluster-ambiguity`, default 0.05) → stays
     `-1`, counted as `cdd_orphans`;
   - a class winning fewer than `--cdd-cluster-min-size` (default 3) rows → dropped.
   Known labels stay fixed unless `--cdd-relabel-known`.

5. **Write `pseudo_label`**, then re-key the sampler's tissue pools on it so newly-recovered
   rows become drawable in the matched half (not just the free half).

**Label-space widening.** The CDD class space is the set of tissues the **source** can
supply (`sc_tissues − excludes`) — a tissue with no pseudobulk can never form a valid pair. A
bulk label is trusted only if it names a class in that space; everything else (literal
`"unknown"` **and** bulk-only tissues) is treated as unlabeled and handed to the clustering,
instead of carrying a "known" label that matches no class and silently never trains.

---

## 5. Source-seeded centroids: "translating into the bulk frame"

This is the mechanism that lets **single-cell-only tissues** be reached. On by default;
disable with `--cdd-no-source-fallback`.

### The problem it solves

For a tissue with no bulk examples, the only possible centroid comes from the pseudobulk
side. But that centroid is then used to label **bulk** rows by cosine-nearest-centroid, and
bulk and pseudobulk occupy different regions of embedding space — the model encodes modality
as a strong, easily-learned signal. A raw pseudobulk centroid sits in the *pseudobulk*
neighborhood, so every bulk row is closer to any bulk-anchored centroid than to it. The
source centroid wins nothing, or the true rows get grabbed by the wrong bulk centroid.

*Measured in a unit test:* a raw source centroid scored **0.248** against its own true rows,
while a **wrong-tissue** bulk centroid scored **0.596** — wrong tissue wins, rows mislabeled.

### The intuition

Read each embedding as two added parts:

```
embedding  ≈   (which modality)     +   (which tissue)
                bulk vs pseudobulk        tissue identity
```

The **modality** part is a large shift shared by all rows of a domain. The **tissue** part is
the finer structure we care about. A source centroid carries both; when compared to bulk, the
modality shift dominates and drowns out the tissue signal.

Fix: **strip the modality shift, keep the tissue signal, re-plant the centroid in the bulk
neighborhood** where the rows being labeled actually live.

### How it works

Estimate the modality shift as the difference between the two clouds' centers of mass:

- `src_global` = mean of all source (pseudobulk) class means → where pseudobulk lives
- `bulk_global` = mean of all bulk embeddings → where bulk lives

Then for a single-cell-only tissue `c`:

```
C_src[c] = normalize( src_mean[c]  −  src_global  +  bulk_global )
                       └ tissue+modality  └ remove pseudobulk  └ add bulk
                                            center             center
```

`src_mean[c] − src_global` cancels the modality component, leaving the tissue's position
*relative to its own cloud's center*; adding `bulk_global` re-plants that same relative
position around the bulk cloud's center. The tissue offset survives; the modality offset is
removed.

*Same test, after correction:* the score moves **0.248 → 0.914**, clearly beating the
wrong-tissue bulk centroid at 0.596.

### Assumptions

1. **The modality difference is roughly a single shared translation** — one shift vector,
   about the same for every tissue. First-order; if the domains differ by rotation/scaling or
   tissue-dependent offsets, the correction is only partial.
2. **Tissue structure is preserved across domains** — pseudobulk-lung sits relative to
   pseudobulk-average like bulk-lung sits relative to bulk-average. This is the premise of
   doing any cross-domain alignment.
3. **Warmup has already shrunk the gap.** The correction only needs to be approximate,
   because MMD spent warmup pulling the clouds together; the residual shift is small, so a
   crude first-order fix is enough to flip the argmax. The correction handles the leftover,
   not the full raw gap.

### Why it is self-limiting (the safety)

The corrected centroid only has to **win the assignment argmax once**. As soon as the tissue
collects bulk rows, the next k-means iteration recomputes its centroid from *those actual
bulk rows*, so it stops being a translated-source estimate and becomes a genuine bulk
centroid, snapping into the bulk cloud. The translation is a bootstrap for the first few
rows, nothing more.

If the assumptions do not hold well enough (clouds still too far apart), the rows simply fail
the ambiguity filter and stay `-1` — they show up as `cdd_orphans`. It degrades to "couldn't
label these," never to confidently-wrong labels. **`cdd_orphans` is the number to watch:**
high orphans alongside empty source-seeded classes means the translation was insufficient,
i.e. warmup did not close the gap — not that anything was mislabeled.

---

## 6. Why source means are recomputed, not EMA'd

Pseudobulks are re-aggregated from randomly drawn cells every step. An EMA of pseudobulk
embeddings would therefore smear over **both** a changing pseudobulk composition **and** a
changing model state. Worse, the §5 offset correction would then subtract a **stale**
`src_global` while adding a **fresh** `bulk_global`, leaking model drift straight into the
centroid position — reintroducing exactly the staleness §4 removes on the bulk side.

Re-drawing and re-encoding the pseudobulks at each clustering event makes `src_mean[c]` a
clean estimate of `E[emb(PB_c)]` under current weights, and guarantees `src_mean`,
`src_global`, `bulk_global`, and `bank_emb` all share one model state. It also removes the
momentum hyperparameter entirely.

*(Implementation detail: the synthesized pseudobulk rows must be tagged `modality == 2`
explicitly; `_fill_missing_conditions` copies conditions from the constituent single cells,
which would otherwise mark them `modality == 1`.)*

---

## 7. The CDD loss itself

On each non-paired batch, past warmup (`_cdd_w > 0`):

1. Take CLS embeddings; split source (`modality == 2`) / target (`modality == 0`).
2. Label each row by tissue (bulk rows use `pseudo_label` when inference is on).
3. For each tissue `c`, a multi-bandwidth-RBF-kernel MMD gives:
   - `e1(c)` — source spread within tissue `c`
   - `e2(c)` — target spread within tissue `c`
   - `e3(c, c')` — source-`c` vs target-`c'` closeness
4. `D(c, c') = e1(c) + e2(c') − 2·e3(c, c')`
5. The loss:

```
CDD = mean_c D(c, c)   −   mean_{c ≠ c'} D(c, c')
      └ intra: pull same tissue together   └ inter: push different tissues apart
```

Only tissues present in **both** domains with ≥ `--cdd-min-class-count` samples, not in the
exclude set, and with a non-negative label are used — so `-1` rows drop out automatically.
Added as `_cdd_w · CDD`, logged as `train/loss_cdd`. Bandwidth is the detached median
heuristic over the pooled sample; kernel means are biased (diagonal included), matching the
existing `_mmd_rbf`.

---

## 8. Distributed (DDP) behaviour

Because each rank re-encodes the **full** bulk set (and the same seeded pseudobulks, drawn
with a step-derived seed shared across ranks) inside the refresh pass, every rank clusters on
identical data and produces identical `pseudo_label`s with **no cross-rank communication**.
The earlier `all_gather` bank-merge is therefore unnecessary once the refresh is
unconditional.

---

## 9. One-paragraph summary

MMD pulls the bulk and pseudobulk clouds together during warmup → clustering then labels the
messy bulk against **known-bulk** centroids (and, for single-cell-only tissues,
**offset-corrected source** centroids) → class-aware sampling puts matched tissues in each
batch while a free bulk fraction keeps every row training → CDD pulls same-tissue together and
different-tissue apart as MMD fades out. Watch `cdd_orphans`: source-seeded classes that stay
empty with high orphans mean the clouds have not met yet.

---

## 10. Flags reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--cdd` | off | Enable the CDD loss. Requires `--unified`. |
| `--loss-weight-cdd` | 0.3 | CDD weight (β), post-ramp. |
| `--cdd-class-column` | `tissue_general` | Condition column used as the class. Must be in `--conditions`. |
| `--cdd-min-class-count` | 2 | Min samples per class per domain to contribute. |
| `--cdd-exclude-labels` | `["unknown"]` | Label names excluded from the class space. |
| `--cdd-class-aware` | off | Same tissues in both halves of each batch (§3). |
| `--cdd-bulk-class-frac` | 0.6 | Fraction of bulk slots reserved for the matched tissues; rest drawn freely. |
| `--cdd-infer-labels` | off | Cluster to infer bulk tissue labels (§4). Requires `--cdd`. |
| `--cdd-cluster-warmup-steps` | 2000 | Steps before the first clustering / CDD onset. |
| `--cdd-cluster-interval` | 0 | Re-cluster every N steps; 0 = once per epoch. |
| `--cdd-cluster-iters` | 10 | Max Lloyd iterations per event. |
| `--cdd-cluster-ambiguity` | 0.05 | `D0`: cosine-distance above which a row stays `-1`. |
| `--cdd-cluster-min-size` | 3 | `N0`: min members per class/anchor. |
| `--cdd-relabel-known` | off | Also re-infer known-labeled bulk (fully unsupervised CAN). |
| `--cdd-no-source-fallback` | off | Disable source-seeded centroids (§5); restricts classes to tissues the bulk already knows. |
| `--cdd-cluster-source-pb` | 8 | Pseudobulks encoded per class to estimate the source mean. |
| `--cdd-ramp-steps` | 1000 | Steps over which CDD ramps in and MMD decays out. |
| `--cdd-mmd-final-weight` | 0.0 | MMD weight after the ramp. Applies whenever `--cdd` is on, including an explicit `--mmd`. |

### Example

```bash
python pretrain.py --unified --cdd --cdd-class-aware --cdd-infer-labels \
    --loss-weight-cdd 0.3 --cdd-bulk-class-frac 0.6 \
    --cdd-cluster-warmup-steps 2000 --cdd-ramp-steps 1000 \
    --pb-group-column tissue_general --conditions tissue_general assay --verbose
```

With `--verbose`, startup reports the bulk split (how many rows are explicitly unknown vs
unmatched tissue), and each clustering event logs `cdd_classes_used`, `cdd_source_seeded`,
`cdd_unknown_assigned`, `cdd_orphans`, and `cdd_bank_filled`.

---

## 11. Code map

| Concern | Location |
|---------|----------|
| CDD loss, valid-class filter, per-tissue kernels | `cancerfoundation/model/module.py` → `_cdd_loss` |
| Target-label lookup (raw or pseudo-labels) | `module.py` → `_cdd_target_labels` |
| Bank + source buffers, spherical k-means, source-seeding | `module.py` → `init_target_bank`, `set_source_means`, `recluster` |
| Single-batch CLS encode for refresh passes | `module.py` → `encode_cls` |
| Bulk re-encode + source-mean recompute + cluster trigger | `cancerfoundation/model/model.py` → `_refresh_bulk_bank`, `_refresh_source_means`, `_run_target_clustering` |
| MMD→CDD handover schedule | `model.py` → `_update_cdd_schedule` (in `on_train_batch_start`) |
| Label-space widening + bank init | `model.py` → `setup` |
| Class-aware batch, bulk-slot split, `refresh_cdd_labels` | `cancerfoundation/data/bulk_sc_data.py` → `sample_class_aware_batch`, `refresh_cdd_labels` |
| Per-row refresh collator | `cancerfoundation/data/data_module.py` → `make_cdd_refresh_collator` |
| Flag definitions | `utils_config.py` |
| Flag wiring, guards, MMD auto-enable | `pretrain.py` |
