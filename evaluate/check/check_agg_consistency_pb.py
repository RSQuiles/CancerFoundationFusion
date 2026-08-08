"""Self-checks for aggregation consistency over precomputed pseudobulks.

Run directly; no cluster, no GPU, no checkpoints, no memory-mapped dataset:

    python evaluate/check/check_agg_consistency_pb.py

Covers the pieces that have to agree for the loss to mean anything:

  1. BulkSCDataset._build_pb_id_index  — grouping SC rows by pseudobulk id, and the
     sentinel handling that keeps rows without a pseudobulk out of the groups.
  2. BulkSCSampler._build_agg_pb_pools / _draw_agg_sc — restricting the PB pool to
     pseudobulks whose cells are present, and drawing those cells.
  3. BulkSCCollator.__call__ — slicing the [sc, pb, agg_sc, bulk] block layout and
     emitting is_sc_for_pb / sample_pseudobulk_index for the loss.
  4. The aggregation block itself, driven by those tensors.

Needs torch (the data modules import it at module scope), so run it in the bionemo
container — everything else it touches is fabricated in memory. Exits non-zero on
failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    print(
        "SKIPPED: this check needs torch, which cancerfoundation.data imports at "
        "module scope. Nothing was verified — run it inside the bionemo container:\n"
        "  singularity run --bind /cluster <sif> python "
        "evaluate/check/check_agg_consistency_pb.py",
        file=sys.stderr,
    )
    raise SystemExit(0)


def _load_data_modules():
    """Import cancerfoundation.data.* without executing the package __init__.

    ``cancerfoundation/__init__.py`` pulls in the model, and through it scanpy and
    lightning. None of that is involved in batch assembly, so the two data modules are
    bound under stub parent packages instead — that keeps this check runnable in any
    environment with torch, not only in the full training container.
    """
    import types

    for name, rel in (
        ("cancerfoundation", "cancerfoundation"),
        ("cancerfoundation.data", "cancerfoundation/data"),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(ROOT / rel)]
            sys.modules[name] = pkg

    import importlib

    collator = importlib.import_module("cancerfoundation.data.bulk_sc_collator")
    data = importlib.import_module("cancerfoundation.data.bulk_sc_data")
    return collator, data


_collator_mod, _data_mod = _load_data_modules()
BULK_MODALITY = _collator_mod.BULK_MODALITY
PB_MODALITY = _collator_mod.PB_MODALITY
SC_MODALITY = _collator_mod.SC_MODALITY
BulkSCCollator = _collator_mod.BulkSCCollator
BulkSCDataset = _data_mod.BulkSCDataset
BulkSCSampler = _data_mod.BulkSCSampler

FAILED: list[str] = []

PAD_ID = 0
CLS_ID = 1
MAX_LEN = 8


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{('  - ' + detail) if detail else ''}")
        FAILED.append(label)


# --------------------------------------------------------------------------- #
# 1. Dataset index
# --------------------------------------------------------------------------- #

def make_dataset(pb_codes: dict[str, int], sc_ids, pb_ids, verbose=False):
    """A BulkSCDataset with only the fields _build_pb_id_index touches.

    Built with object.__new__ so the real method is exercised without a memmap: the
    on-disk store needs bionemo and a multi-GB fixture, and none of it is involved in
    the indexing logic under test.
    """
    ds = object.__new__(BulkSCDataset)
    ds.verbose = verbose
    ds.pb_id_column = "pseudobulk_id"
    ds.sc_pb_to_indices = {}
    ds.pb_id_fill_code = None
    ds.mapping = {"pseudobulk_id": pb_codes}

    n_sc, n_pb = len(sc_ids), len(pb_ids)
    ds.sc_indices = np.arange(n_sc, dtype=np.int64)
    ds.pb_indices = np.arange(n_sc, n_sc + n_pb, dtype=np.int64)
    ds._obs_arrays = {
        "pseudobulk_id": np.array(list(sc_ids) + list(pb_ids), dtype=np.int64)
    }
    ds._build_pb_id_index()
    return ds


def check_dataset_index() -> None:
    print("\n[dataset: pseudobulk_id index]")

    # Codes as preprocessing would emit them: "0" is the reindex fill for rows with no
    # pseudobulk, the rest are prefixed string ids.
    codes = {"0": 0, "pb:10": 1, "pb:11": 2, "pb:12": 3}
    #        sc rows: two cells of pb:10, three of pb:11, one with no pseudobulk
    sc_ids = [1, 1, 2, 2, 2, 0]
    #        pb rows: pb:10, pb:11, pb:12 (pb:12 has no cells)
    pb_ids = [1, 2, 3]
    ds = make_dataset(codes, sc_ids, pb_ids)

    check("fill category resolved", ds.pb_id_fill_code == 0, str(ds.pb_id_fill_code))
    check("groups keyed by code", set(ds.sc_pb_to_indices) == {1, 2},
          str(sorted(ds.sc_pb_to_indices)))
    check("cells grouped correctly",
          list(ds.sc_pb_to_indices[1]) == [0, 1]
          and list(ds.sc_pb_to_indices[2]) == [2, 3, 4],
          str({k: list(v) for k, v in ds.sc_pb_to_indices.items()}))
    check("row with no pseudobulk excluded",
          all(5 not in list(v) for v in ds.sc_pb_to_indices.values()))
    check("pseudobulk with no cells absent from the index",
          3 not in ds.sc_pb_to_indices)

    # No fill category at all (every row has a pseudobulk) must still work.
    ds2 = make_dataset({"pb:1": 0, "pb:2": 1}, [0, 0, 1], [0, 1])
    check("no fill category -> nothing excluded",
          set(ds2.sc_pb_to_indices) == {0, 1}, str(sorted(ds2.sc_pb_to_indices)))

    # The trap the "pb:" prefix exists to prevent: bare integer ids make the fill
    # value 0 indistinguishable from a real id 0.
    ds3 = make_dataset({"0": 0, "1": 1, "2": 2}, [0, 1, 2], [0, 1, 2])
    check("integer ids: id 0 dropped rather than merged with the fill",
          0 not in ds3.sc_pb_to_indices, str(sorted(ds3.sc_pb_to_indices)))


# --------------------------------------------------------------------------- #
# 2. Sampler
# --------------------------------------------------------------------------- #

def make_sampler(
    n_sc_per_pb=3, n_pb=2, n_sc=4, n_bulk=2, group_pools=False, subset=False
):
    """A BulkSCSampler with only the fields the agg pools/draw touch.

    ``subset=True`` mimics the random_split path, where the sampler's indices are
    Subset-local and ``subset_base_indices`` maps them back to obs rows. Note that
    ``subset_indices`` is deliberately never set: it only exists on the non-Subset
    construction path, so touching it here would hide a real AttributeError.
    """
    sp = object.__new__(BulkSCSampler)
    sp.verbose = False
    sp.n_sc_per_pb = n_sc_per_pb
    sp.n_pb, sp.n_sc, sp.n_bulk = n_pb, n_sc, n_bulk
    sp.rng = np.random.default_rng(0)

    # 12 SC rows (0..11), 3 PB rows (12,13,14). pb id 3 has no cells.
    sp.sc_indices = np.arange(12, dtype=np.int64)
    sp.pb_indices = np.array([12, 13, 14], dtype=np.int64)
    sp.sc_pb_to_indices = {
        1: np.arange(0, 5, dtype=np.int64),
        2: np.arange(5, 12, dtype=np.int64),
    }
    sp.pb_group_to_indices = (
        {"lung": np.array([12, 14]), "liver": np.array([13])} if group_pools else None
    )
    sp.pb_group_to_indices_agg = sp.pb_group_to_indices
    sp.pb_indices_with_sc = sp.pb_indices
    sp.precomputed_agg = True

    pb_ids = np.array([0] * 12 + [1, 2, 3], dtype=np.int64)
    if subset:
        # A non-identity local -> base map: base row b holds local row 14-b, so an
        # implementation that skipped the translation would read the wrong ids.
        sp.subset_base_indices = np.arange(14, -1, -1, dtype=np.int64)
        pb_ids = pb_ids[::-1].copy()
    else:
        sp.subset_base_indices = np.arange(15, dtype=np.int64)

    base = object.__new__(BulkSCDataset)
    base.pb_id_column = "pseudobulk_id"
    base._obs_arrays = {"pseudobulk_id": pb_ids}
    sp.base_dataset = base
    sp._build_agg_pb_pools()
    return sp


def check_sampler() -> None:
    print("\n[sampler: pools and constituent draws]")

    sp = make_sampler()
    check("PB without cells excluded from the pool",
          list(sp.pb_indices_with_sc) == [12, 13], str(list(sp.pb_indices_with_sc)))
    check("row -> pseudobulk code lookup built",
          sp.pb_row_to_pb_id == {12: 1, 13: 2, 14: 3}, str(sp.pb_row_to_pb_id))

    drawn = sp._draw_agg_sc([12, 13])
    check("draws n_sc_per_pb cells per pseudobulk",
          len(drawn) == 2 * sp.n_sc_per_pb, str(len(drawn)))
    check("first block comes from pseudobulk 1 only",
          all(0 <= i < 5 for i in drawn[:3]), str(drawn[:3]))
    check("second block comes from pseudobulk 2 only",
          all(5 <= i < 12 for i in drawn[3:]), str(drawn[3:]))

    # Order must follow the input, not the pool order — the collator slices by position.
    drawn = sp._draw_agg_sc([13, 12])
    check("block order follows the PB order given",
          all(5 <= i < 12 for i in drawn[:3]) and all(0 <= i < 5 for i in drawn[3:]),
          str(drawn))

    # A pseudobulk with fewer cells than n_sc_per_pb must still fill its block.
    sp2 = make_sampler(n_sc_per_pb=9)
    drawn = sp2._draw_agg_sc([12])
    check("small pseudobulk sampled with replacement",
          len(drawn) == 9 and all(0 <= i < 5 for i in drawn), str(drawn))

    # The random_split path: indices are Subset-local and must be translated through
    # subset_base_indices before touching _obs_arrays. This is the construction path
    # training actually uses, and the one where a missing translation shows up.
    sub = make_sampler(subset=True)
    check("Subset path: builds without touching non-existent attributes", True)
    check("Subset path: ids read through subset_base_indices",
          sub.pb_row_to_pb_id == {12: 1, 13: 2, 14: 3}, str(sub.pb_row_to_pb_id))
    check("Subset path: unusable PB still excluded",
          list(sub.pb_indices_with_sc) == [12, 13], str(list(sub.pb_indices_with_sc)))
    drawn = sub._draw_agg_sc([12, 13])
    check("Subset path: draws from the right pools",
          all(0 <= i < 5 for i in drawn[:3]) and all(5 <= i < 12 for i in drawn[3:]),
          str(drawn))

    # Group pools are restricted too, so a group-aware draw cannot pick an unusable PB.
    sp3 = make_sampler(group_pools=True)
    check("group pools restricted to usable pseudobulks",
          list(sp3.pb_group_to_indices_agg.get("lung", [])) == [12]
          and list(sp3.pb_group_to_indices_agg.get("liver", [])) == [13],
          str({k: list(v) for k, v in sp3.pb_group_to_indices_agg.items()}))

    # No id column at all -> a message naming the fix, not an AttributeError.
    sp4 = object.__new__(BulkSCSampler)
    sp4.base_dataset = object.__new__(BulkSCDataset)
    sp4.base_dataset.pb_id_column = None
    try:
        sp4._build_agg_pb_pools()
        ok, detail = False, "no error raised"
    except ValueError as exc:
        ok, detail = "pseudobulk_id" in str(exc), str(exc)[:120]
    check("missing id column gives an actionable error", ok, detail)

    # Every pseudobulk unusable -> refuse rather than train on nothing.
    sp5 = object.__new__(BulkSCSampler)
    sp5.verbose = False
    sp5.subset_base_indices = np.arange(13, dtype=np.int64)
    sp5.pb_indices = np.array([12], dtype=np.int64)
    sp5.sc_pb_to_indices = {}
    sp5.pb_group_to_indices = None
    base = object.__new__(BulkSCDataset)
    base.pb_id_column = "pseudobulk_id"
    base._obs_arrays = {"pseudobulk_id": np.array([0] * 12 + [7], dtype=np.int64)}
    sp5.base_dataset = base
    try:
        sp5._build_agg_pb_pools()
        ok, detail = False, "no error raised"
    except ValueError as exc:
        ok, detail = "not one of" in str(exc), str(exc)[:120]
    check("no usable pseudobulk gives an actionable error", ok, detail)


# --------------------------------------------------------------------------- #
# 3. Collator
# --------------------------------------------------------------------------- #

def make_collator(n_sc_per_pb=3, batch_size=8, agg=True, precomputed=True):
    return BulkSCCollator(
        normalise_bins=False,
        condition_token=False,
        do_padding=True,
        pad_token_id=PAD_ID,
        max_length=MAX_LEN,
        do_mlm=True,
        mask_ratio=0.15,
        keep_first_n_tokens=1,
        data_style="pcpt",
        conditions=["modality"],
        batch_size=batch_size,
        bulk_ratio=0.25,
        pb_ratio=0.25,
        n_sc_per_pseudobulk=n_sc_per_pb,
        agg_consistency=agg,
        precomputed_pb=precomputed,
        pb_id_column="pseudobulk_id",
        sampling=True,
    )


def make_sample(pb_id=0, n_genes=5):
    genes = np.arange(2, 2 + n_genes, dtype=np.int64)
    return {
        "genes": torch.from_numpy(np.insert(genes, 0, CLS_ID)),
        "expressions": torch.tensor(
            np.insert(np.random.rand(n_genes).astype(np.float32), 0, 0.0)
        ),
        "modality": 0,
        "pseudobulk_id": pb_id,
        "_row_index": 0,
    }


def build_examples(col, pb_ids):
    """[sc, pb, agg_sc, bulk] in the order the sampler emits."""
    ex = [make_sample() for _ in range(col.n_sc)]
    ex += [make_sample(pb_id=p) for p in pb_ids]
    for p in pb_ids:
        ex += [make_sample(pb_id=p) for _ in range(col.n_sc_per_pseudobulk)]
    ex += [make_sample() for _ in range(col.n_bulk)]
    return ex


def check_collator() -> None:
    print("\n[collator: block layout and emitted tensors]")

    col = make_collator()
    expected = col.n_bulk + col.n_sc + col.n_pb * (1 + col.n_sc_per_pseudobulk)
    check("raw_batch_size accounts for the constituent cells",
          col.raw_batch_size == expected, f"{col.raw_batch_size} != {expected}")

    pb_ids = [11, 22]
    examples = build_examples(col, pb_ids)
    check("fixture length matches raw_batch_size",
          len(examples) == col.raw_batch_size,
          f"{len(examples)} != {col.raw_batch_size}")

    out = col(examples)
    modality = out["conditions"]["modality"]
    is_sc_for_pb = out["is_sc_for_pb"]
    sample_pb_idx = out["sample_pseudobulk_index"]

    n_agg = col.n_pb * col.n_sc_per_pseudobulk
    check("constituent cells tagged is_sc_for_pb",
          int(is_sc_for_pb.sum()) == n_agg, str(int(is_sc_for_pb.sum())))
    check("unified batch holds every row",
          len(modality) == col.n_bulk + col.n_sc + col.n_pb + n_agg, str(len(modality)))
    check("pseudobulk rows kept their modality",
          int((modality == PB_MODALITY).sum()) == col.n_pb,
          str(int((modality == PB_MODALITY).sum())))
    check("constituent cells enter as SC rows",
          all(modality[i] == SC_MODALITY for i in (is_sc_for_pb == 1).nonzero()[:, 0]))
    check("bulk rows unaffected",
          int((modality == BULK_MODALITY).sum()) == col.n_bulk)
    check("precomputed flag set on the batch",
          bool(out["is_precomputed_pb_batch"]))

    # The mapping the loss consumes: each constituent cell must name its pseudobulk,
    # in contiguous per-pseudobulk blocks.
    agg_positions = (is_sc_for_pb == 1).nonzero()[:, 0].tolist()
    assigned = [int(sample_pb_idx[i]) for i in agg_positions]
    check("constituent cells map onto 0..n_pb-1 in blocks",
          assigned == [0] * col.n_sc_per_pseudobulk + [1] * col.n_sc_per_pseudobulk,
          str(assigned))
    check("pseudobulk_sizes reports the real constituent count",
          out["pseudobulk_sizes"].tolist() == [col.n_sc_per_pseudobulk] * col.n_pb,
          str(out["pseudobulk_sizes"].tolist()))
    check("not flagged as a paired batch", not bool(out["is_paired_batch"]))

    # The invariant guard: a reordered agg block must be rejected, not silently used.
    bad = build_examples(col, pb_ids)
    start = col.n_sc + col.n_pb
    bad[start], bad[start + col.n_sc_per_pseudobulk] = (
        bad[start + col.n_sc_per_pseudobulk], bad[start],
    )
    try:
        col(bad)
        ok, detail = False, "no error raised"
    except ValueError as exc:
        ok, detail = "mismatch" in str(exc), str(exc)[:120]
    check("mismatched constituent cell is rejected", ok, detail)

    # precomputed without agg: unchanged short batch, no is_sc_for_pb rows.
    col2 = make_collator(agg=False)
    check("without agg the batch stays short",
          col2.raw_batch_size == col2.n_bulk + col2.n_sc + col2.n_pb,
          str(col2.raw_batch_size))
    ex2 = (
        [make_sample() for _ in range(col2.n_sc)]
        + [make_sample(pb_id=p) for p in pb_ids]
        + [make_sample() for _ in range(col2.n_bulk)]
    )
    out2 = col2(ex2)
    check("without agg no constituent rows are emitted",
          int(out2["is_sc_for_pb"].sum()) == 0)

    # Regression: the on-the-fly path must keep its original layout.
    col3 = make_collator(agg=True, precomputed=False)
    check("on-the-fly raw_batch_size unchanged",
          col3.raw_batch_size
          == col3.n_bulk + col3.n_sc + col3.n_pb * col3.n_sc_per_pseudobulk,
          str(col3.raw_batch_size))
    ex3 = (
        [make_sample() for _ in range(col3.n_sc)]
        + [make_sample() for _ in range(col3.n_pb * col3.n_sc_per_pseudobulk)]
        + [make_sample() for _ in range(col3.n_bulk)]
    )
    out3 = col3(ex3)
    check("on-the-fly still emits constituent rows",
          int(out3["is_sc_for_pb"].sum()) == col3.n_pb * col3.n_sc_per_pseudobulk,
          str(int(out3["is_sc_for_pb"].sum())))
    check("on-the-fly not flagged as precomputed",
          not bool(out3["is_precomputed_pb_batch"]))


# --------------------------------------------------------------------------- #
# 4. The loss block
# --------------------------------------------------------------------------- #

def agg_loss_from_batch(out, embeddings, agg_fn="mean"):
    """Reimplementation of module.py's aggregation block, on the collator's tensors.

    Kept in the check rather than imported because TransformerModule.forward cannot run
    without a full model; the point is to prove the *tensors* drive it correctly.
    """
    import torch.nn.functional as F

    sc_assignment: dict = {}
    for idx in range(len(embeddings)):
        if out["is_sc_for_pb"][idx] == 1:
            sc_assignment.setdefault(
                int(out["sample_pseudobulk_index"][idx]), []
            ).append(idx)

    pb_global_pos = (out["conditions"]["modality"] == PB_MODALITY).nonzero(
        as_tuple=True
    )[0]
    total = torch.tensor(0.0)
    for pb_local_idx, sc_indices in sc_assignment.items():
        pb_e = embeddings[pb_global_pos[pb_local_idx]]
        sc_e = embeddings[sc_indices]
        agg = sc_e.mean(dim=0) if agg_fn == "mean" else sc_e.sum(dim=0)
        total = total + F.mse_loss(pb_e, agg)
    return total, sc_assignment


def check_loss() -> None:
    print("\n[loss: the tensors drive the aggregation]")

    col = make_collator()
    out = col(build_examples(col, [11, 22]))
    n_rows = len(out["conditions"]["modality"])
    d_model = 4

    torch.manual_seed(0)
    emb = torch.randn(n_rows, d_model)
    loss, assignment = agg_loss_from_batch(out, emb)
    check("loss recovers one group per pseudobulk",
          set(assignment) == {0, 1}, str(sorted(assignment)))
    check("each group holds n_sc_per_pb cells",
          all(len(v) == col.n_sc_per_pseudobulk for v in assignment.values()),
          str({k: len(v) for k, v in assignment.items()}))
    check("loss is positive on random embeddings", float(loss) > 0, str(float(loss)))

    # Make each pseudobulk equal the mean of its own cells -> the loss must vanish.
    pb_pos = (out["conditions"]["modality"] == PB_MODALITY).nonzero(as_tuple=True)[0]
    for pb_local, sc_idx in assignment.items():
        emb[pb_pos[pb_local]] = emb[sc_idx].mean(dim=0)
    loss_zero, _ = agg_loss_from_batch(out, emb)
    check("loss is zero when the pseudobulk equals its cell mean",
          float(loss_zero) < 1e-6, str(float(loss_zero)))

    # Sum would not vanish here — the check that the reduction actually matters.
    loss_sum, _ = agg_loss_from_batch(out, emb, agg_fn="sum")
    check("sum reduction does NOT vanish (why mean is forced)",
          float(loss_sum) > 1e-6, str(float(loss_sum)))


def main() -> None:
    check_dataset_index()
    check_sampler()
    check_collator()
    check_loss()

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for label in FAILED:
            print(f"  - {label}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
