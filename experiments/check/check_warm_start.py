"""Self-check for ``utils/warm_start.py`` and the launcher's ``trainer.init_from``.

The planner (which weight goes where), the parent lookup, the launcher and the rename
of a parent run anywhere: no cluster, GPU or torch. The part that copies tensors runs
only where torch imports, on a small stand-in module with CancerFoundation's key layout
(the real model is covered by one run in the container).

    python -m experiments.check.check_warm_start
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.check._harness import check, finish, raises
from experiments.layout import ExperimentLayout
from experiments.rename_model import apply_plan, build_plan
from experiments.runtime import describe_warm_start, resolve_model_config, write_run_card
from utils.resume import segment_record
from utils.warm_start import (
    WarmStartError,
    WarmStartOptions,
    check_vocab,
    classify,
    decoder_condition_layout,
    find_parent_mapping,
    normalize_key,
    plan_warm_start,
)

D = 16
BACKBONE = {
    "gene_encoder.embedding.weight": (100, D),
    "gene_encoder.enc_norm.weight": (D,),
    "value_encoder.linear1.weight": (D, 1),
    "transformer_encoder.layers.0.linear1.weight": (32, D),
    "generative_flag": (D,),
}


def _shapes(tech_rows=3, decoder_in=D, mvc_in=None, heads=None, extra=None, conditions=("technology", "modality")):
    shapes = dict(BACKBONE)
    for col in conditions:
        rows = 3 if col == "modality" else tech_rows
        shapes[f"condition_encoders.{col}.embedding.weight"] = (rows, D)
        shapes[f"condition_encoders.{col}.enc_norm.weight"] = (D,)
    shapes["decoder.fc.0.weight"] = (D, decoder_in)
    shapes["mvc_decoder.gene2query.weight"] = (D, mvc_in or D * (len(conditions) + 1))
    shapes.update(heads or {})
    shapes.update(extra or {})
    return shapes


PARENT_MAP = {"technology": {"10x": 0, "smartseq": 1, "visium": 2}, "modality": {"bulk": 0, "sc": 1, "pb": 2}}
# Same column in another store: one category missing, one new, codes shifted.
CHILD_MAP = {"technology": {"10x": 0, "dropseq": 1, "smartseq": 2, "visium": 3}, "modality": {"bulk": 0, "sc": 1, "pb": 2}}


def check_keys() -> None:
    print("\n-- keys and groups --")
    check("model. prefix stripped", normalize_key("model.gene_encoder.embedding.weight") == "gene_encoder.embedding.weight")
    check("compile prefix stripped", normalize_key("model._orig_mod.decoder.fc.0.weight") == "decoder.fc.0.weight")
    check("keys outside the module -> None", normalize_key("criterion.weight") is None)
    check("groups", [classify(k) for k in (
        "gene_encoder.x", "encoder.embedding.weight", "transformer_encoder.l", "generative_flag",
        "flag_encoder.weight", "decoder.fc", "mvc_decoder.w", "condition_encoders.a.embedding.weight",
        "grad_reverse_discriminators.modality.out_layer.weight", "known_label", "generative_flag_x",
    )] == ["backbone"] * 5 + ["decoders"] * 2 + ["conditions", "heads", "other", "other"])
    check("layout begin: only MVC takes conditions",
          decoder_condition_layout("begin", ["t", "m"]) == {"decoder": (), "mvc_decoder": ("t", "m")})
    check("layout end: both", decoder_condition_layout("end", ["m"]) == {"decoder": ("m",), "mvc_decoder": ("m",)})
    check("layout none / no conditions: neither",
          decoder_condition_layout("none", ["m"]) == decoder_condition_layout("end", []) == {"decoder": (), "mvc_decoder": ()})
    check("unknown group refused", raises(lambda: WarmStartOptions(include=("backbone", "dat")), WarmStartError))
    check("exclude wins over include",
          WarmStartOptions(include=("backbone", "heads"), exclude=("heads",)).groups == ("backbone",))


def check_planner() -> None:
    print("\n-- planner --")
    opts = WarmStartOptions()
    heads = {"grad_reverse_discriminators.modality.out_layer.weight": (2, D)}
    plan = plan_warm_start(
        _shapes(), _shapes(tech_rows=4, heads=heads), options=opts,
        parent_mapping=PARENT_MAP, child_mapping=CHILD_MAP, parent_where="begin", child_where="begin",
    )
    copied = {a.key for a in plan.actions if a.rows is None}
    check("whole backbone copied", set(BACKBONE) <= copied)
    check("decoders copied (same layout)", {"decoder.fc.0.weight", "mvc_decoder.gene2query.weight"} <= copied)
    check("modality encoder copied as is", "condition_encoders.modality.embedding.weight" in copied)
    rows = {a.key: a.rows for a in plan.actions if a.rows is not None}
    check("technology remapped by name",
          rows.get("condition_encoders.technology.embedding.weight") == [(0, 0), (2, 1), (3, 2)])
    info = plan.remapped.get("condition_encoders.technology.embedding.weight", {})
    check("new category reported", info.get("new_categories") == ["dropseq"] and info.get("rows") == 4)
    check("technology norm copied", "condition_encoders.technology.enc_norm.weight" in copied)
    check("new DAT head left as built (heads are off by default)", "not transferred" in plan.reinitialized.get(
        "grad_reverse_discriminators.modality.out_layer.weight", ""))
    plan = plan_warm_start(
        _shapes(), _shapes(tech_rows=4, heads=heads),
        options=WarmStartOptions(include=("backbone", "decoders", "conditions", "heads")),
        parent_mapping=PARENT_MAP, child_mapping=CHILD_MAP, parent_where="begin", child_where="begin",
    )
    check("new DAT head with heads on: not in the parent", "not in the parent" in plan.reinitialized.get(
        "grad_reverse_discriminators.modality.out_layer.weight", ""))

    # Another condition set: the MVC decoder's input concatenates different conditions.
    plan = plan_warm_start(
        _shapes(), _shapes(conditions=("modality",)), options=opts,
        parent_mapping=PARENT_MAP, child_mapping=CHILD_MAP, parent_where="begin", child_where="begin",
    )
    check("MVC decoder re-initialized when its conditions change",
          "concatenates" in plan.reinitialized.get("mvc_decoder.gene2query.weight", ""))
    check("expression decoder still copied (begin)",
          any(a.key == "decoder.fc.0.weight" for a in plan.actions))
    check("dropped condition reported as skipped",
          "condition_encoders.technology.embedding.weight" in plan.skipped)

    # where_condition="end" with the same count but other columns: same shapes, other meaning.
    plan = plan_warm_start(
        _shapes(decoder_in=3 * D, conditions=("technology", "modality")),
        _shapes(decoder_in=3 * D, conditions=("modality", "technology")), options=opts,
        parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP, parent_where="end", child_where="end",
    )
    check("end: decoder re-initialized when the condition order differs",
          "decoder.fc.0.weight" in plan.reinitialized and "mvc_decoder.gene2query.weight" in plan.reinitialized)
    plan = plan_warm_start(
        _shapes(decoder_in=3 * D), _shapes(decoder_in=3 * D), options=opts,
        parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP, parent_where=None, child_where="end",
    )
    check("no parent where_condition: shape-only match, noted",
          any(a.key == "decoder.fc.0.weight" for a in plan.actions) and plan.notes)

    # Backbone mismatches.
    bigger = _shapes()
    bigger["transformer_encoder.layers.0.linear1.weight"] = (64, D)
    check("backbone shape mismatch refused",
          raises(lambda: plan_warm_start(_shapes(), bigger, options=opts, parent_where="begin", child_where="begin"),
                 WarmStartError))
    loose = plan_warm_start(_shapes(), bigger, options=WarmStartOptions(strict_backbone=False),
                            parent_where="begin", child_where="begin")
    check("loose backbone: mismatch re-initialized",
          "transformer_encoder.layers.0.linear1.weight" in loose.reinitialized)
    orig = {("encoder." + k[len("gene_encoder."):] if k.startswith("gene_encoder.") else k): v
            for k, v in _shapes().items()}
    check("gen_method orig vs theirs (encoder.* vs gene_encoder.*) refused",
          raises(lambda: plan_warm_start(orig, _shapes(), options=opts), WarmStartError))

    # Category names.
    plan = plan_warm_start(_shapes(), _shapes(tech_rows=4), options=opts, parent_mapping=None,
                           child_mapping=CHILD_MAP, parent_where="begin", child_where="begin")
    check("no parent mapping: technology re-initialized, never copied by code",
          "unknown" in plan.reinitialized.get("condition_encoders.technology.embedding.weight", "")
          and "condition_encoders.technology.enc_norm.weight" in plan.reinitialized)
    check("no parent mapping: modality still copied",
          any(a.key == "condition_encoders.modality.embedding.weight" for a in plan.actions))
    wrong = {"technology": {"10x": 0, "smartseq": 1}}
    plan = plan_warm_start(_shapes(), _shapes(tech_rows=4), options=opts, parent_mapping=wrong,
                           child_mapping=CHILD_MAP, parent_where="begin", child_where="begin")
    check("mapping with the wrong cardinality rejected",
          "not the mapping" in plan.reinitialized.get("condition_encoders.technology.embedding.weight", ""))

    # Heads.
    old_head = {"grad_reverse_discriminators.modality.out_layer.weight": (3, D),
                "grad_reverse_discriminators.technology.out_layer.weight": (3, D)}
    new_head = {"grad_reverse_discriminators.modality.out_layer.weight": (2, D),
                "grad_reverse_discriminators.technology.out_layer.weight": (3, D)}
    with_heads = WarmStartOptions(include=("backbone", "decoders", "conditions", "heads"))
    plan = plan_warm_start(_shapes(heads=old_head), _shapes(heads=new_head), options=with_heads,
                           parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP,
                           parent_where="begin", child_where="begin")
    check("old 3-class modality head re-initialized",
          "3-class" in plan.reinitialized.get("grad_reverse_discriminators.modality.out_layer.weight", ""))
    check("head with identical class codes copied",
          any(a.key == "grad_reverse_discriminators.technology.out_layer.weight" for a in plan.actions))
    plan = plan_warm_start(_shapes(heads=old_head), _shapes(heads=new_head), options=with_heads,
                           parent_mapping=PARENT_MAP,
                           child_mapping={**PARENT_MAP, "technology": {"10x": 0, "visium": 1, "smartseq": 2}},
                           parent_where="begin", child_where="begin")
    check("head with other class codes re-initialized",
          "class codes" in plan.reinitialized.get("grad_reverse_discriminators.technology.out_layer.weight", ""))
    plan = plan_warm_start(_shapes(heads=old_head), _shapes(heads=new_head), options=opts,
                           parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP,
                           parent_where="begin", child_where="begin")
    check("heads excluded by default",
          all(classify(a.key) != "heads" for a in plan.actions)
          and "not transferred" in plan.reinitialized.get("grad_reverse_discriminators.technology.out_layer.weight", ""))

    # Group switches and other state.
    plan = plan_warm_start(_shapes(), _shapes(), options=WarmStartOptions(exclude=("decoders",)),
                           parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP,
                           parent_where="begin", child_where="begin")
    check("excluded group left as built", "not transferred" in plan.reinitialized.get("decoder.fc.0.weight", ""))
    plan = plan_warm_start(_shapes(extra={"known_label": (50,)}), _shapes(extra={"known_label": (70,)}),
                           options=opts, parent_mapping=PARENT_MAP, child_mapping=PARENT_MAP,
                           parent_where="begin", child_where="begin")
    check("other state never transferred, whatever its shape",
          all(a.key != "known_label" for a in plan.actions) and "known_label" in plan.reinitialized)
    summary = plan.summary()
    check("summary counts per group", summary.get("backbone", {}).get("copied") == len(BACKBONE))


def check_lookup(tmp: Path) -> None:
    print("\n-- parent mapping and vocabulary --")
    store = tmp / "store"
    store.mkdir()
    (store / "mapping.json").write_text(json.dumps(PARENT_MAP), encoding="utf-8")
    moved = tmp / "moved_store"
    moved.mkdir()
    (moved / "mapping.json").write_text(json.dumps(CHILD_MAP), encoding="utf-8")
    ckpt_dir = tmp / "parent"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "epoch_00.ckpt"

    found, source = find_parent_mapping({"condition_mapping": CHILD_MAP, "hyper_parameters": {"data_path": str(store)}},
                                        ckpt_path)
    check("checkpoint's own mapping first", found == CHILD_MAP and source == "checkpoint")
    found, source = find_parent_mapping({"hyper_parameters": {"data_path": str(store)}}, ckpt_path, str(moved))
    check("explicit mapping (a store dir) wins", found == CHILD_MAP and source.endswith("mapping.json"))
    found, _ = find_parent_mapping({"hyper_parameters": {"data_path": str(store)}}, ckpt_path)
    check("existing checkpoint: data_path's mapping.json", found == PARENT_MAP)
    (ckpt_dir / "config.resolved.json").write_text(json.dumps({"train_path": str(moved)}), encoding="utf-8")
    found, _ = find_parent_mapping({"hyper_parameters": {"data_path": str(tmp / "gone")}}, ckpt_path)
    check("data_path gone: config.resolved.json train_path", found == CHILD_MAP)
    (ckpt_dir / "config.resolved.json").unlink()
    found, source = find_parent_mapping({"hyper_parameters": {}}, ckpt_path)
    check("nothing found -> None", found is None and source == "none")
    check("unreadable explicit mapping refused",
          raises(lambda: find_parent_mapping({}, ckpt_path, str(tmp / "nope.json")), FileNotFoundError))

    vocab = {"<pad>": 0, "<cls>": 1, "A": 2}
    check("same vocabulary passes", check_vocab(vocab, dict(vocab)) is None)
    check("missing parent vocabulary is a note", check_vocab(None, vocab) is not None)
    check("different vocabulary refused", raises(lambda: check_vocab(vocab, {**vocab, "A": 3}), WarmStartError))


def check_launcher(tmp: Path) -> None:
    print("\n-- launcher, run card, rename --")
    layout = ExperimentLayout(tmp / "root")
    parent_dir = layout.checkpoint_dir("parent")
    parent_dir.mkdir(parents=True)
    (parent_dir / "dry_run.ckpt").write_text("x", encoding="utf-8")
    cfg = {"trainer": {"epochs": 5, "init_from": {"model": "parent"}}}
    check("parent with only a dry-run placeholder refused",
          raises(lambda: resolve_model_config(layout, "child", cfg), FileNotFoundError))
    (parent_dir / "epoch_04.ckpt").write_text("x", encoding="utf-8")
    cfg = {"trainer": {"epochs": 5, "init_from": {"model": "parent", "include": ["backbone"]}}}
    resolved, _ = resolve_model_config(layout, "child", cfg)
    trainer = resolved["trainer"]
    check("init_from pinned to the parent's newest checkpoint",
          trainer["init_from"] == str(parent_dir / "epoch_04.ckpt"))
    check("reference kept by name", resolved["init_from_ref"] == {"model": "parent", "checkpoint": "epoch_04.ckpt"})
    check("dict options become flat trainer keys", trainer["init_include"] == ["backbone"])
    line = describe_warm_start(layout, "child", resolved)
    check("dry-run line names parent, file and groups",
          "parent/epoch_04.ckpt" in line and "backbone" in line and "decoders" not in line)
    check("named file resolved",
          resolve_model_config(layout, "child", {"trainer": {"init_from": {"model": "parent", "checkpoint": "epoch_04.ckpt"}}})[0]
          ["trainer"]["init_from"].endswith("epoch_04.ckpt"))
    check("plain path accepted",
          resolve_model_config(layout, "child", {"trainer": {"init_from": str(parent_dir)}})[0]
          ["init_from_ref"]["checkpoint"] == "epoch_04.ckpt")
    check("self-reference refused",
          raises(lambda: resolve_model_config(layout, "parent", {"trainer": {"init_from": {"model": "parent"}}}), ValueError))
    check("own directory as path refused",
          raises(lambda: resolve_model_config(layout, "parent", {"trainer": {"init_from": str(parent_dir)}}), ValueError))
    check("unknown init_from key refused",
          raises(lambda: resolve_model_config(layout, "child", {"trainer": {"init_from": {"model": "parent", "lr": 1}}}), ValueError))
    check("conflicting option refused",
          raises(lambda: resolve_model_config(layout, "child", {"trainer": {
              "init_include": ["heads"], "init_from": {"model": "parent", "include": ["backbone"]}}}), ValueError))
    check("missing parent refused",
          raises(lambda: resolve_model_config(layout, "child", {"trainer": {"init_from": {"model": "ghost"}}}), FileNotFoundError))

    # Resuming the warm-started child: its own checkpoint wins.
    child_dir = layout.checkpoint_dir("child")
    child_dir.mkdir(parents=True)
    write_run_card(layout, "child", resolved, "launch", None, None)
    card = json.loads(layout.run_card("child").read_text(encoding="utf-8"))
    check("run card records the parent",
          card["init_from"] == {"model": "parent", "checkpoint": "epoch_04.ckpt", "path": str(parent_dir / "epoch_04.ckpt")})
    (child_dir / "epoch_01.ckpt").write_text("x", encoding="utf-8")
    resumed = json.loads(json.dumps(resolved))
    resumed["trainer"]["resume_from_checkpoint"] = "last"
    check("resume of own checkpoint: warm start ignored",
          "ignored" in describe_warm_start(layout, "child", resumed))
    check("no init_from -> no line", describe_warm_start(layout, "child", {"trainer": {}}) is None)

    seg = segment_record(None, epochs=5, lr=1e-4, warmup=0.1, seed=0,
                         init_from={"model": "parent", "file": "epoch_04.ckpt", "global_step": 10})
    check("segment mode warm_start", seg["mode"] == "warm_start" and seg["init_from"]["model"] == "parent")
    check("plain fresh segment unchanged",
          segment_record(None, epochs=5, lr=1e-4, warmup=0.1, seed=0)["mode"] == "fresh")

    # Renaming the parent rewrites the child's reference (name and path).
    (child_dir / "warm_start.json").write_text(json.dumps({"from": {
        "model": "parent", "checkpoint": str(parent_dir / "epoch_04.ckpt")}}), encoding="utf-8")
    write_run_card(layout, "parent", {"trainer": {}}, "launch", None, None)
    plan = build_plan(layout, "parent", "base", job_lookup=lambda _: [])
    check("no file listed twice", len(plan.json_files) == len(set(plan.json_files)))
    apply_plan(layout, plan)
    card = json.loads(layout.run_card("child").read_text(encoding="utf-8"))
    new_dir = layout.checkpoint_dir("base")
    check("child's run card follows the rename",
          card["init_from"]["model"] == "base"
          and Path(card["init_from"]["path"]) == new_dir / "epoch_04.ckpt")
    record = json.loads((child_dir / "warm_start.json").read_text(encoding="utf-8"))
    check("child's warm_start.json follows the rename",
          record["from"]["model"] == "base" and Path(record["from"]["checkpoint"]) == new_dir / "epoch_04.ckpt")
    parent_card = json.loads(layout.run_card("base").read_text(encoding="utf-8"))
    check("parent's history appended once", len(parent_card["previous_names"]) == 1)


def check_torch(tmp: Path) -> None:
    print("\n-- copying tensors (torch) --")
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  torch not importable ({type(exc).__name__})")
        return
    from utils.warm_start import warm_start_model

    class Cond(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.embedding = nn.Embedding(n, D)
            self.enc_norm = nn.LayerNorm(D)

    class Core(nn.Module):
        def __init__(self, tech_rows, dat_rows):
            super().__init__()
            self.gene_encoder = nn.Embedding(10, D)
            self.transformer_encoder = nn.Linear(D, D)
            self.generative_flag = nn.Parameter(torch.randn(D))
            self.condition_encoders = nn.ModuleDict({"technology": Cond(tech_rows), "modality": Cond(3)})
            self.decoder = nn.Linear(D, 1)
            self.mvc_decoder = nn.Linear(3 * D, 1)
            if dat_rows:
                self.grad_reverse_discriminators = nn.ModuleDict({"modality": nn.Linear(D, dat_rows)})

    class Wrapper(nn.Module):
        def __init__(self, tech_rows, dat_rows=0):
            super().__init__()
            self.model = Core(tech_rows, dat_rows)
            self.vocab = {"<pad>": 0, "<cls>": 1}
            self.where_condition = "begin"

    parent = Wrapper(3, dat_rows=3)  # a current checkpoint: 3-class DAT head, compiled keys
    store = tmp / "parent_store"
    store.mkdir()
    (store / "mapping.json").write_text(json.dumps(PARENT_MAP), encoding="utf-8")
    state = {k.replace("model.", "model._orig_mod.", 1): v for k, v in parent.state_dict().items()}
    ckpt = tmp / "parent.ckpt"
    torch.save({"state_dict": state, "epoch": 3, "global_step": 40,
                "hyper_parameters": {"vocab": parent.vocab, "where_condition": "begin", "data_path": str(store)}}, ckpt)

    child = Wrapper(4, dat_rows=2)
    before_new = child.model.condition_encoders["technology"].embedding.weight[1].clone()
    before_head = child.model.grad_reverse_discriminators["modality"].weight.clone()
    record = warm_start_model(child, ckpt, CHILD_MAP, WarmStartOptions(include=("backbone", "decoders", "conditions", "heads")))
    p, c = parent.model, child.model
    check("backbone equal", torch.equal(p.gene_encoder.weight, c.gene_encoder.weight)
          and torch.equal(p.transformer_encoder.weight, c.transformer_encoder.weight)
          and torch.equal(p.generative_flag, c.generative_flag))
    pt, ct = p.condition_encoders["technology"].embedding.weight, c.condition_encoders["technology"].embedding.weight
    check("rows remapped by name", torch.equal(ct[0], pt[0]) and torch.equal(ct[2], pt[1]) and torch.equal(ct[3], pt[2]))
    check("new category keeps its initialization", torch.equal(ct[1], before_new))
    check("modality encoder copied",
          torch.equal(p.condition_encoders["modality"].embedding.weight, c.condition_encoders["modality"].embedding.weight))
    check("old 3-class head not loaded into the 2-class one",
          torch.equal(c.grad_reverse_discriminators["modality"].weight, before_head))
    check("record names the source", record["from"]["global_step"] == 40
          and record["mapping_source"].endswith("mapping.json"))
    check("parameters still trainable leaves", c.gene_encoder.weight.requires_grad)

    bad = Wrapper(3)
    bad.vocab = {"<pad>": 0, "<cls>": 2}
    check("vocabulary mismatch refused before any copy",
          raises(lambda: warm_start_model(bad, ckpt, PARENT_MAP), WarmStartError))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_keys()
        check_planner()
        check_lookup(tmp)
        check_launcher(tmp)
        check_torch(tmp)
    finish()


if __name__ == "__main__":
    main()
