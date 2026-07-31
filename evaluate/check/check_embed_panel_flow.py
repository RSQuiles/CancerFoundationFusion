"""Exercise CancerFoundation.embed()'s gene-selection control flow.

Run directly:  python evaluate/check/check_embed_panel_flow.py

Stubs pytorch_lightning/transformers/safetensors/bionemo/tokenizers and replaces the
forward pass with a recorder, so the real branch logic in ``embed()`` runs and we can
assert which genes each cell was actually embedded through. Needs only
numpy/pandas/anndata/scanpy/torch.

Asserts that ``shared_panel=True`` yields ONE dense pass over all cells through one
gene set; that ``shared_panel=False`` reproduces the old per-modality behaviour (and
reports how far the two panels diverge); and that single-modality callers — the
majority of them — are unaffected either way.
"""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- stub the heavy deps that are not installed locally -------------------
pl = types.ModuleType("pytorch_lightning")
class _LM:
    def __init__(self, *a, **k): pass
    def save_hyperparameters(self, *a, **k): pass
    def eval(self): return self
    def to(self, *a, **k): return self
pl.LightningModule = _LM
pl.utilities = types.ModuleType("pytorch_lightning.utilities")
pl.utilities.types = types.ModuleType("pytorch_lightning.utilities.types")
pl.utilities.types.OptimizerLRSchedulerConfig = dict
sys.modules["pytorch_lightning"] = pl
sys.modules["pytorch_lightning.utilities"] = pl.utilities
sys.modules["pytorch_lightning.utilities.types"] = pl.utilities.types

tf = types.ModuleType("transformers")
tf.get_scheduler = lambda *a, **k: None
sys.modules["transformers"] = tf

st = types.ModuleType("safetensors")
st.safe_open = lambda *a, **k: None
sys.modules["safetensors"] = st

bn = types.ModuleType("bionemo")
for name in ("bionemo.scdl", "bionemo.scdl.io",
             "bionemo.scdl.io.single_cell_memmap_dataset"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["bionemo"] = bn
sys.modules["bionemo.scdl.io.single_cell_memmap_dataset"].SingleCellMemMapDataset = object

tk = types.ModuleType("tokenizers")
for attr in ("Tokenizer", "models", "pre_tokenizers", "trainers"):
    setattr(tk, attr, object)
sys.modules["tokenizers"] = tk


import anndata as ad
import pandas as pd
import torch

from cancerfoundation.model.model import CancerFoundation

N_GENES, N_TOP = 400, 100
GENES = [f"g{i}" for i in range(N_GENES)]
rng = np.random.default_rng(0)


class FakeModel(CancerFoundation):
    """Only what embed() touches: vocab, n_top_genes, and a recording forward pass."""

    def __init__(self):
        # bypass the real __init__ (needs the full transformer stack)
        self.vocab = {g: i + 2 for i, g in enumerate(GENES)}
        self.n_top_genes = N_TOP
        self.input_style = "continuous"
        self.n_bins = 51
        self.embsize = 8
        self.calls = []  # (n_obs, tuple_of_genes)

        inner = types.SimpleNamespace()
        inner.eval = lambda: None
        inner.parameters = lambda: iter([torch.zeros(1)])
        self.model = inner

    # embed() calls these
    def _run_dense_embed(self, data, batch_size, device):
        self.calls.append((data.n_obs, tuple(data.var_names)))
        return torch.zeros((data.n_obs, self.embsize))


def make_adata():
    A = np.abs(rng.normal(2, 0.05, (120, N_GENES)))
    A[:, 0:50] += np.abs(rng.normal(0, 3, (120, 50)))
    A[:, 100:150] += np.abs(rng.normal(0, 3, (120, 50)))
    B = np.abs(rng.normal(2, 0.05, (120, N_GENES)))
    B[:, 50:100] += np.abs(rng.normal(0, 3, (120, 50)))
    B[:, 100:150] += np.abs(rng.normal(0, 3, (120, 50)))
    return ad.AnnData(
        X=np.vstack([A, B]).astype(np.float32),
        obs=pd.DataFrame(
            {"_eval_modality": ["bulk"] * 120 + ["pseudobulk"] * 120},
            index=[f"c{i}" for i in range(240)],
        ),
        var=pd.DataFrame(index=GENES),
    )


# ---- 1. shared panel (default): ONE dense embed, all cells, same genes ----
print("=== shared_panel=True (default) ===")
m = FakeModel()
df, gene_set = m.embed(make_adata(), normalized=True, modality_col="_eval_modality")
assert len(m.calls) == 1, f"expected 1 dense embed, got {len(m.calls)}"
assert m.calls[0][0] == 240, f"expected all 240 cells in one pass, got {m.calls[0][0]}"
assert len(m.calls[0][1]) == N_TOP, f"expected {N_TOP} genes, got {len(m.calls[0][1])}"
assert isinstance(gene_set, list), f"gene_set_used must be flat, got {type(gene_set)}"
assert len(gene_set) == N_TOP
assert df.shape == (240, 8)
print(f"  1 dense embed, 240 cells, {len(gene_set)} shared genes, flat gene_set: OK")

# ---- 2. shared_panel=False: per-modality panels, embeds diverge -----------
print("\n=== shared_panel=False (legacy per-modality) ===")
m2 = FakeModel()
df2, gs2 = m2.embed(make_adata(), normalized=True, modality_col="_eval_modality",
                    shared_panel=False)
assert len(m2.calls) == 2, f"expected 2 dense embeds, got {len(m2.calls)}"
assert isinstance(gs2, dict), f"expected per-modality dict, got {type(gs2)}"
g_bulk, g_pb = set(gs2["bulk"]), set(gs2["pseudobulk"])
jac = len(g_bulk & g_pb) / len(g_bulk | g_pb)
print(f"  2 dense embeds; bulk/pb panel Jaccard overlap = {jac:.2f}")
assert jac < 1.0, "per-modality panels should differ (that was the bug)"
print("  per-modality dict + divergent panels: OK (this is what shared_panel fixes)")

# ---- 3. single modality: shared panel is a no-op --------------------------
print("\n=== single modality group ===")
a1 = make_adata()
a1.obs["_eval_modality"] = "bulk"
m3 = FakeModel()
df3, gs3 = m3.embed(a1, normalized=True, modality_col="_eval_modality")
m3b = FakeModel()
df3b, gs3b = m3b.embed(a1, normalized=True, modality_col="_eval_modality",
                       shared_panel=False)
assert isinstance(gs3, dict) and isinstance(gs3b, dict), (gs3, gs3b)
assert gs3 == gs3b, "single-group data must be unaffected by shared_panel"
print("  identical with and without shared_panel: OK")

# ---- 4. no modality_col at all (the ~10 single-modality callers) ----------
print("\n=== no modality_col ===")
a2 = make_adata()
m4 = FakeModel()
df4, gs4 = m4.embed(a2, normalized=True, modality="bulk")
m4b = FakeModel()
df4b, gs4b = m4b.embed(a2, normalized=True, modality="bulk", shared_panel=False)
assert gs4 == gs4b, "callers without modality_col must be unaffected"
assert isinstance(gs4, list) and len(gs4) == N_TOP
print(f"  identical with and without shared_panel, {len(gs4)} genes: OK")

# ---- 5. explicit gene_subset still wins ----------------------------------
print("\n=== explicit gene_subset overrides the shared panel ===")
want = GENES[:37]
m5 = FakeModel()
df5, gs5 = m5.embed(make_adata(), normalized=True, modality_col="_eval_modality",
                    gene_subset=want)
assert list(m5.calls[0][1]) == want, m5.calls[0][1]
assert gs5 == want
print("  gene_subset respected: OK")

# ---- 6. panel_strategy is honoured ---------------------------------------
print("\n=== panel_strategy=pooled ===")
m6 = FakeModel()
_, gs6 = m6.embed(make_adata(), normalized=True, modality_col="_eval_modality",
                  panel_strategy="pooled")
assert isinstance(gs6, list) and len(gs6) == N_TOP
assert set(gs6) != set(gene_set), "pooled should differ from consensus here"
print("  pooled differs from consensus: OK")

print("\nALL EMBED-FLOW CHECKS PASSED")
