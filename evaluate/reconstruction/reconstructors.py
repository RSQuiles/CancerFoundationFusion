"""Models that reconstruct masked expression values, behind one interface.

A ``Reconstructor`` scores the masked entries of a mask plan
(``evaluate/reconstruction/mask_plan.py``) in its own target space. The contract every
implementation follows, so models stay comparable:

1. Score exactly the plan's masked (sample, gene) entries; an entry the model cannot
   see (gene outside its vocabulary) is dropped and counted, never guessed.
2. Build targets from the FULL sample, in the model's own space, and say which space
   (``space``): ``"bins"`` for CancerFoundation, ``"ln_tpm1"`` for BulkFormer.
3. Never let a masked value reach the model input.

``predict(adata, plan)`` returns ``targets``, an ``(n_samples, n_plan_genes)`` matrix in
the model's space (NaN where the model has no value: used for the baselines), and
``row``/``gene``/``pred`` arrays for the masked entries it scored. Add a model by
subclassing ``Reconstructor`` and decorating it with ``@register("<name>")``;
``--option key=value`` pairs become constructor arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RECONSTRUCTORS: dict[str, type] = {}

# Model-facing modality codes (cancerfoundation/data/bulk_sc_collator.py); fixed, not
# the stored mapping.json codes.
MODALITY_CODES = {"bulk": 0, "sc": 1, "pseudobulk": 2}


def register(name: str):
    def wrap(cls):
        if name in RECONSTRUCTORS:
            raise ValueError(f"Reconstructor {name!r} registered twice")
        cls.name = name
        RECONSTRUCTORS[name] = cls
        return cls
    return wrap


def build_reconstructor(name: str, options: dict[str, Any] | None = None) -> "Reconstructor":
    if name not in RECONSTRUCTORS:
        raise KeyError(f"Unknown reconstructor {name!r}; registered: {sorted(RECONSTRUCTORS)}")
    return RECONSTRUCTORS[name](**(options or {}))


class Reconstructor:
    name: str = "base"
    space: str = "unknown"

    def __init__(self, **options: Any) -> None:
        self.options = options

    def predict(self, adata, plan) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {"method": self.name, "space": self.space,
                "options": {k: str(v) for k, v in self.options.items()}}


# --------------------------------------------------------------------------- #
# Decoding CancerFoundation outputs into bins (numpy, so it is testable without torch)
# --------------------------------------------------------------------------- #


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_bins(raw: np.ndarray, loss: str, n_bins: int, normalise_bins: bool = False) -> np.ndarray:
    """Expected bin per position from the decoder output (see cancerfoundation/loss.py).

    - ``mse``: one value per position (scaled back by ``n_bins`` when bins were normalised);
    - ``ordinal_cross_entropy``: ``n_bins`` cumulative logits, where class 0 is always
      true, so the expected bin is ``sum(sigmoid(logits[..., 1:]))``;
    - ``corn``: ``n_bins - 1`` conditional logits, expected bin ``sum(cumprod(sigmoid))``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if loss == "mse":
        # One value per position, possibly with a trailing singleton axis.
        pred = raw[..., 0] if raw.ndim >= 2 and raw.shape[-1] == 1 else raw
        return pred * n_bins if normalise_bins else pred
    if loss == "ordinal_cross_entropy":
        return _sigmoid(raw[..., 1:]).sum(-1)
    if loss == "corn":
        return np.cumprod(_sigmoid(raw), axis=-1).sum(-1)
    raise ValueError(f"Cannot decode loss {loss!r} into bins (zinb predicts counts, not bins)")


def _str_value(x) -> str:
    return getattr(x, "value", x) if not isinstance(x, str) else x


# --------------------------------------------------------------------------- #
# CancerFoundation
# --------------------------------------------------------------------------- #


@register("cancerfoundation")
class CancerFoundationReconstructor(Reconstructor):
    """A CancerFoundation checkpoint; targets and predictions are bins.

    Each sample is binned per row over all of its vocabulary genes, as training bins
    the whole stored row before sampling a sequence (ties broken at random, seeded).
    Generative models (``training_tasks`` gen/both) get the plan's context genes as
    the perception stream and its masked genes as gen tokens, exactly the training
    layout; gen tokens attend only to the context and themselves, so all masked genes
    are predicted in one pass without seeing each other. Perception-only models get
    one sequence with the masked values replaced by ``mask_value``.

    Options: ``checkpoint`` (required), ``head`` (``gen``: the expression decoder;
    ``mvc``: the MVC decoder), ``batch_size`` (16), ``cpu``, ``seed`` (0), ``mapping``
    (the training store's mapping.json; default ``<model data_path>/mapping.json``),
    ``condition`` (``{column: value}`` forced for every sample), ``modality_col``
    (obs column with sc/bulk/pseudobulk, default ``modality``).
    """

    space = "bins"

    def __init__(self, checkpoint: str | None = None, head: str = "gen", batch_size: int = 16,
                 cpu: bool = False, seed: int = 0, mapping: str | None = None,
                 condition: dict | None = None, modality_col: str = "modality", **options: Any) -> None:
        if checkpoint is None:
            raise ValueError("cancerfoundation needs checkpoint=<.ckpt>")
        if head not in ("gen", "mvc"):
            raise ValueError(f"head must be gen or mvc, got {head!r}")
        super().__init__(checkpoint=checkpoint, head=head, batch_size=batch_size, cpu=cpu, seed=seed,
                         mapping=mapping, condition=condition, modality_col=modality_col, **options)
        self.checkpoint, self.head, self.batch_size = Path(checkpoint), head, int(batch_size)
        self.cpu, self.seed, self.mapping_path = bool(cpu), int(seed), mapping
        self.condition = dict(condition or {})
        self.modality_col = modality_col
        self.model = None

    # -- setup -----------------------------------------------------------------

    def _load(self) -> None:
        if self.model is not None:
            return
        from embed.embed import load_model

        self.model = load_model(self.checkpoint, self.cpu)
        self.loss = _str_value(self.model.loss_type)
        if self.loss == "zinb":
            raise ValueError("A zinb model predicts counts, not bins; bin reconstruction does not apply")
        if self.head == "mvc" and not getattr(self.model.model, "do_mvc", False):
            raise ValueError("head=mvc but this model has no MVC decoder")
        self.generative = bool(self.model.model.use_generative_training)
        self.conditions = list((self.model.model.conditions or {}).keys())
        print(f"  [cf] loss {self.loss}, {'generative' if self.generative else 'perception-only'}, "
              f"n_bins {self.model.n_bins}, conditions {self.conditions or 'none'}")

    def _condition_codes(self, obs) -> dict[str, np.ndarray]:
        """Model-facing condition codes per sample."""
        codes: dict[str, np.ndarray] = {}
        mapping = None
        for cond in self.conditions:
            if cond == "modality":
                if "modality" in self.condition:
                    values = np.full(len(obs), str(self.condition["modality"]))
                elif self.modality_col in obs.columns:
                    values = obs[self.modality_col].astype(str).to_numpy()
                else:
                    raise KeyError(f"The model is conditioned on modality but obs has no "
                                   f"'{self.modality_col}' column; pass condition={{modality: bulk}}")
                from experiments.layout import canonical_modality

                codes[cond] = np.array([MODALITY_CODES[canonical_modality(v)] for v in values])
                continue
            if mapping is None:
                import json

                path = Path(self.mapping_path or Path(str(self.model.data_path)) / "mapping.json")
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found: the training store's mapping.json is needed "
                                            f"to encode condition '{cond}'; pass mapping=<path>")
                mapping = json.loads(path.read_text(encoding="utf-8"))
            table = {str(k): int(v) for k, v in mapping.get(cond, {}).items()}
            if cond in self.condition:
                values = np.full(len(obs), str(self.condition[cond]))
            elif cond in obs.columns:
                values = obs[cond].astype(str).to_numpy()
            else:
                raise KeyError(f"The model is conditioned on '{cond}' but obs lacks it; pass condition={{{cond}: ...}}")
            unknown = sorted(set(values) - set(table))
            if unknown:
                raise ValueError(f"Condition '{cond}' values {unknown[:5]} are not in the training mapping "
                                 f"(known: {sorted(table)[:10]}...)")
            codes[cond] = np.array([table[v] for v in values])
        return codes

    # -- prediction --------------------------------------------------------------

    def predict(self, adata, plan) -> dict[str, np.ndarray]:
        import scipy.sparse as sp
        import torch

        from cancerfoundation.data.preprocess import binning

        self._load()
        model = self.model
        vocab = model.vocab
        var_pos = adata.var_names.get_indexer(plan.genes)
        if (var_pos < 0).any():
            raise ValueError("The plan's genes must all be in the input h5ad")
        in_vocab = np.array([g in vocab for g in plan.genes])
        gene_ids = np.array([vocab[g] if ok else -1 for g, ok in zip(plan.genes, in_vocab)])
        # Binning runs over every vocabulary gene of the sample, as in training.
        vocab_cols = np.flatnonzero(np.array([g in vocab for g in adata.var_names]))
        plan_in_row = np.searchsorted(vocab_cols, var_pos)  # position of each plan gene in the binned row
        print(f"  [cf] {int(in_vocab.sum())} of {len(plan.genes)} plan genes in the vocabulary; "
              f"binning over {len(vocab_cols)} vocabulary genes per sample")

        codes = self._condition_codes(adata.obs)
        device = next(model.parameters()).device
        np.random.seed(self.seed)  # binning breaks ties with np.random
        n, n_genes = plan.n_samples, len(plan.genes)
        targets = np.full((n, n_genes), np.nan, dtype=np.float32)
        rows, genes, preds = [], [], []
        norm = model.normalise_bins

        for start in range(0, n, self.batch_size):
            batch = range(start, min(start + self.batch_size, n))
            ctx_ids, ctx_vals, msk_ids, msk_genes = [], [], [], []
            for i in batch:
                row = adata.X[i]
                row = np.asarray(row.toarray() if sp.issparse(row) else row, dtype=np.float32).ravel()
                binned = binning(row[vocab_cols], model.n_bins).astype(np.float32)
                t = np.full(n_genes, np.nan, dtype=np.float32)
                t[in_vocab] = binned[plan_in_row[in_vocab]]
                targets[i] = t
                ctx = plan.context(i)
                ctx = ctx[in_vocab[ctx]]
                msk = plan.masked(i)
                msk = msk[in_vocab[msk]]
                values = t[ctx] / model.n_bins if norm else t[ctx]
                ctx_ids.append(gene_ids[ctx])
                ctx_vals.append(values)
                msk_ids.append(gene_ids[msk])
                msk_genes.append(msk)

            raw = self._forward(ctx_ids, ctx_vals, msk_ids, {k: v[list(batch)] for k, v in codes.items()},
                                device, torch)
            for j, i in enumerate(batch):
                k = len(msk_ids[j])
                rows.append(np.full(k, i))
                genes.append(msk_genes[j])
                preds.append(decode_bins(raw[j][:k], self.loss, model.n_bins, norm))
            if (start // self.batch_size) % 20 == 0:
                print(f"  [cf] {min(start + self.batch_size, n)}/{n} samples")

        return {"targets": targets, "row": np.concatenate(rows), "gene": np.concatenate(genes),
                "pred": np.concatenate(preds).astype(np.float32),
                "n_dropped": int(sum(len(plan.masked(i)) for i in range(n)) - sum(len(g) for g in genes))}

    def _forward(self, ctx_ids, ctx_vals, msk_ids, codes, device, torch) -> list[np.ndarray]:
        """Raw decoder outputs at the masked positions, one array per sample."""
        model = self.model
        pad_id, pad_val, cls_id = model.pad_token_id, model.pad_value, model.cls_token_id
        b = len(ctx_ids)
        conditions = ({k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in codes.items()}
                      if self.conditions else None)

        if self.generative:
            lp = 1 + max(len(c) for c in ctx_ids)
            lg = max(1, max(len(m) for m in msk_ids))
            pg = torch.full((b, lp), pad_id, dtype=torch.long)
            pv = torch.full((b, lp), float(pad_val))
            gg = torch.full((b, lg), pad_id, dtype=torch.long)
            for j in range(b):
                pg[j, 0] = cls_id
                pg[j, 1:1 + len(ctx_ids[j])] = torch.as_tensor(ctx_ids[j])
                pv[j, 1:1 + len(ctx_vals[j])] = torch.as_tensor(ctx_vals[j])
                gg[j, :len(msk_ids[j])] = torch.as_tensor(msk_ids[j])
            pkpm, gkpm = pg.eq(pad_id), gg.eq(pad_id)
            attn = torch.zeros((lp + lg, lp + lg), dtype=torch.bool)
            attn[lp:, lp:] = True
            attn[lp:, lp:].fill_diagonal_(False)
            with torch.no_grad():
                out = model.model.generative_forward(
                    pg.to(device), pv.to(device), pkpm.to(device), gg.to(device), gkpm.to(device),
                    torch.cat([pkpm, gkpm], 1).to(device), attn.to(device), conditions=conditions)
            raw = out["gen_preds"] if self.head == "gen" else out["mvc_output"][:, -lg:]
            raw = raw.float().cpu().numpy()
            return [raw[j] for j in range(b)]

        # Perception-only: [CLS, context..., masked...] with masked values hidden.
        lengths = [1 + len(c) + len(m) for c, m in zip(ctx_ids, msk_ids)]
        length = max(lengths)
        src = torch.full((b, length), pad_id, dtype=torch.long)
        vals = torch.full((b, length), float(pad_val))
        for j in range(b):
            nc, nm = len(ctx_ids[j]), len(msk_ids[j])
            src[j, 0] = cls_id
            src[j, 1:1 + nc] = torch.as_tensor(ctx_ids[j])
            src[j, 1 + nc:1 + nc + nm] = torch.as_tensor(msk_ids[j])
            vals[j, 1:1 + nc] = torch.as_tensor(ctx_vals[j])
            vals[j, 1 + nc:1 + nc + nm] = float(model.mask_value)
        kpm = src.eq(pad_id)
        with torch.no_grad():
            out = model.model.perceptual_forward(src.to(device), vals.to(device), kpm.to(device),
                                                 conditions=conditions)
        raw = (out["mlm_output"] if self.head == "gen" else out["mvc_output"]).float().cpu().numpy()
        return [raw[j, 1 + len(ctx_ids[j]):1 + len(ctx_ids[j]) + len(msk_ids[j])] for j in range(b)]

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        if self.model is not None:
            out.update(loss=self.loss, generative=self.generative, n_bins=int(self.model.n_bins),
                       conditions=self.conditions)
        return out


# --------------------------------------------------------------------------- #
# BulkFormer
# --------------------------------------------------------------------------- #


@register("bulkformer")
class BulkFormerReconstructor(Reconstructor):
    """bulkformer-latest in its documented imputation mode; targets are ln(TPM+1).

    Loading, gene matching and the conversion to ln(TPM+1) are those of
    ``embed/embed_any.BulkFormerEmbedder`` (same options: ``repo``, ``model_dir``,
    ``input``, ``gene_column``, ``match_on``, ``batch_size``, ``chunk_rows``, ``device``,
    ``amp_dtype``). Targets are computed from the full sample, so the TPM denominator
    spans every vocabulary-matched gene. The plan's masked genes are then set to -10
    and the expression head's predictions read back for them.

    ``context``: ``full`` (default, BulkFormer's native setting: every other available
    gene is visible) or ``plan`` (only the plan's context genes are visible, the rest
    -10, so the model sees the same context as CancerFoundation).
    """

    space = "ln_tpm1"

    def __init__(self, context: str = "full", **options: Any) -> None:
        if context not in ("full", "plan"):
            raise ValueError(f"context must be full or plan, got {context!r}")
        from embed.embed_any import BulkFormerEmbedder

        self._bf = BulkFormerEmbedder(**options)
        super().__init__(context=context, **self._bf.options)
        self.context = context

    def predict(self, adata, plan) -> dict[str, np.ndarray]:
        import pandas as pd
        import scipy.sparse as sp

        bf = self._bf
        bf._ensure_loaded()
        api, vocab = bf._api, bf._vocab
        positions, ensg = bf._match_genes(adata)
        vocab_genes = vocab["canonical_ensg"].astype(str).tolist()
        vocab_index = {g: k for k, g in enumerate(vocab_genes)}
        pos_to_vocab = {int(p): vocab_index[e] for p, e in zip(positions, ensg)}
        var_pos = adata.var_names.get_indexer(plan.genes)
        if (var_pos < 0).any():
            raise ValueError("The plan's genes must all be in the input h5ad")
        plan_vocab = np.array([pos_to_vocab.get(int(p), -1) for p in var_pos])
        print(f"  [bulkformer] {int((plan_vocab >= 0).sum())} of {len(plan.genes)} plan genes in the "
              f"vocabulary; context: {self.context}")
        input_type = bf._resolve_input(adata.X[:512][:, positions])
        mask_value = float(bf._model.config.mask_value)

        n, n_genes = plan.n_samples, len(plan.genes)
        targets = np.full((n, n_genes), np.nan, dtype=np.float32)
        rows, genes, preds = [], [], []
        for start in range(0, n, bf.chunk_rows):
            stop = min(start + bf.chunk_rows, n)
            X = adata.X[start:stop][:, positions]
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            frame = pd.DataFrame(X.astype(np.float32), index=adata.obs_names[start:stop], columns=ensg)
            aligned, _, _ = api.main_gene_selection(X_df=bf._to_log_tpm(frame, input_type), gene_list=vocab_genes)
            truth = aligned.to_numpy(dtype=np.float32)
            inputs = truth.copy()
            for j, i in enumerate(range(start, stop)):
                ok = plan_vocab >= 0
                t = np.full(n_genes, np.nan, dtype=np.float32)
                t[ok] = truth[j, plan_vocab[ok]]
                t[t == mask_value] = np.nan  # a gene the source lacks has no target
                targets[i] = t
                msk = plan.masked(i)
                msk = msk[(plan_vocab[msk] >= 0) & ~np.isnan(t[msk])]
                if self.context == "plan":
                    keep = plan_vocab[plan.context(i)]
                    visible = inputs[j, keep[keep >= 0]].copy()
                    inputs[j, :] = mask_value
                    inputs[j, keep[keep >= 0]] = visible
                inputs[j, plan_vocab[msk]] = mask_value
                rows.append(np.full(len(msk), i))
                genes.append(msk)
            out = api.extract_feature(model=bf._model, expr_array=inputs, output_feature_type="expr_level",
                                      batch_size=bf.batch_size, amp_dtype=bf.amp_dtype)
            for j, i in enumerate(range(start, stop)):
                preds.append(out[j, plan_vocab[genes[i]]])
            print(f"  [bulkformer] {stop}/{n} samples")
        n_masked = sum(len(plan.masked(i)) for i in range(n))
        return {"targets": targets, "row": np.concatenate(rows), "gene": np.concatenate(genes),
                "pred": np.concatenate(preds).astype(np.float32),
                "n_dropped": int(n_masked - sum(len(g) for g in genes))}

    def provenance(self) -> dict[str, Any]:
        out = super().provenance()
        out["bulkformer"] = self._bf.provenance()
        return out
