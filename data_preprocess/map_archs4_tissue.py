"""ARCHS4 samples -> CellxGene ``tissue_general`` categories, and that map applied to an h5ad.

Two subcommands (the approach of ``notebooks/map_archs4_tissue.ipynb``, made a script):

``build-map``: ``metadata.csv`` -> one row per ARCHS4 sample, saved with a run card.
  1. The ``characteristics`` field is split into key: value pairs. Each sample gets
     a tissue value (first of ``TISSUE_KEYS``), else its ``source_name``, and a
     ``has_cell_line`` flag (any of ``CELL_LINE_KEYS``).
  2. Each distinct string is matched to one UBERON term by ``Matcher.match`` (manual
     table, tumor names, exclusions, special rules, UBERON labels and synonyms, organ
     words; see its docstring). The hand-curated tables live in
     ``archs4_tissue_rules.py``.
  3. Each sample gets a category, decided per sample in this order:
     - ``cell line``: a cell-line key, or a string that names a cell line;
     - ``xenograft`` and ``in vitro``: from the string;
     - ``tissue``: a UBERON term with a CellxGene ``tissue_general`` ancestor
       (``cellxgene_ontology_guide``);
     - ``unmatched``: everything else (control RNA, a string without a tissue,
       ambiguous, no match, or a term with no tissue_general). The reason is kept
       in ``match``.

     ``tissue_general`` is the CellxGene label for tissues, the category name for
     the other three, and empty when unmatched.

``apply``: write the map into one obs column (``tissue_general``) of an h5ad.
  - Rows are matched on the obs index (or ``--id-column``).
  - Unmatched and unknown ids keep their current value.
  - The file is edited with h5py: only ``obs/<column>`` is rewritten, so X, var,
    uns and the other obs columns stay byte for byte as they were. That is checked
    after writing.

``curate``: make ``tissue_general`` hold curated labels only (for the class-aware losses
  and the evaluation), keeping the old labels.
  - bulk: the old column moves to ``tissue_noncurated``. The new ``tissue_general`` is
    the map's CellxGene label (``tissue``) or category name (``cell line``,
    ``in vitro``, ``xenograft``), and ``unknown`` for everything else: ``unmatched``
    ARCHS4 rows and non-ARCHS4 rows, whose labels came from keyword matching plus a
    centroid fill (``add_fields.py``). ``--trust-heuristic`` keeps a non-ARCHS4 label,
    renamed to its CellxGene name (``tissue_names.CELLXGENE_NAME``), when keywords
    alone reproduce it.
  - single_cell / pseudobulk: their labels are CellxGene already; nothing is relabelled.
  - Every modality gets ``tissue_label_source`` (cellxgene, archs4, archs4_culture,
    keyword, heuristic, none), so evaluation can filter on it in every modality.
  - ``--in-place`` edits obs only, without copying the file, and undoes every step on
    failure.

    python map_archs4_tissue.py build-map --metadata .../archs4/metadata.csv \\
        --uberon-obo .../uberon-basic.obo [--out-dir archs4_tissue_map]
    python map_archs4_tissue.py apply --h5ad .../BULK_pretraining.h5ad --overwrite [--dry-run]
    python map_archs4_tissue.py apply --h5ad in.h5ad --out relabelled.h5ad
    python map_archs4_tissue.py curate --h5ad .../BULK_pretraining.h5ad --modality bulk \\
        --vocab-h5ad .../pseudobulk.h5ad (--in-place | --out FILE | --dry-run)
    python map_archs4_tissue.py curate --h5ad .../pseudobulk.h5ad --modality pseudobulk --in-place

``build-map`` needs obonet, networkx and cellxgene_ontology_guide; ``apply`` and
``curate`` need h5py, pandas and anndata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import archs4_tissue_rules as R

HERE = Path(__file__).resolve().parent
DEFAULT_MAP_DIR = HERE / "archs4_tissue_map"
MAP_FILE = "archs4_tissue_map.csv.gz"
CARD_FILE = "archs4_tissue_map.json"

CELL_LINE, XENOGRAFT, IN_VITRO = "cell line", "xenograft", "in vitro"
TISSUE, UNMATCHED = "tissue", "unmatched"
CATEGORIES = [TISSUE, CELL_LINE, IN_VITRO, XENOGRAFT, UNMATCHED]
UNKNOWN = "unknown"          # value of a new column on rows the map leaves alone

SYN = re.compile(r'^"(.*)"\s+(EXACT|RELATED|NARROW|BROAD)\b')   # obonet keeps synonyms as raw OBO strings


def pct(n: int, total: int) -> str:
    return f"{n:>9,} ({n / total:6.1%})" if total else f"{n:>9,}"


def banner(title: str) -> None:
    print(f"\n==== {title} " + "=" * max(0, 72 - len(title)))


# ======================================================================================
# 1) metadata -> per-sample strings
# ======================================================================================
def norm(s) -> str:
    """Lowercase, ``_`` -> space, single spaces; NA-like strings -> ""."""
    s = re.sub(r"\s+", " ", re.sub(r"_+", " ", str(s).lower())).strip(" ,;")
    return "" if s in R.NA else s


def parse_characteristics(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (sample_id, key) with its value, from the ``characteristics`` field."""
    rows = []
    for sid, txt in meta["characteristics"].dropna().items():
        for part in R.SPLIT.split(txt):
            if ":" in part:
                k, v = part.split(":", 1)
                k, v = norm(k), norm(v)
                if k and v:
                    rows.append((sid, k, v))
    return pd.DataFrame(rows, columns=["sample_id", "key", "value"]).drop_duplicates(["sample_id", "key"])


def pick(kv: pd.DataFrame, keys: list[str]) -> tuple[pd.Series, pd.Series]:
    """Value and key of the first key in ``keys`` each sample has."""
    rank = {k: i for i, k in enumerate(keys)}
    sub = kv[kv["key"].isin(rank)].assign(r=lambda d: d["key"].map(rank)).sort_values(["sample_id", "r"])
    sub = sub.drop_duplicates("sample_id").set_index("sample_id")
    return sub["value"], sub["key"]


def build_samples(meta: pd.DataFrame) -> pd.DataFrame:
    kv = parse_characteristics(meta)
    print(f"{len(kv):,} key: value pairs")
    key_counts = kv.groupby("key")["sample_id"].nunique().sort_values(ascending=False)
    listed = set(R.TISSUE_KEYS + R.CELL_TYPE_KEYS + R.CELL_LINE_KEYS + R.FALLBACK_TISSUE_KEYS) | R.SKIP_KEYS
    suggest = key_counts[[k for k in key_counts.index if R.SUGGEST.search(k) and k not in listed]]
    if len(suggest):
        print("keys not in the rule lists that look relevant (consider adding to archs4_tissue_rules.py):")
        print(suggest.head(15).to_string())

    tissue_v, tissue_k = pick(kv, R.TISSUE_KEYS)
    celltype_v, _ = pick(kv, R.CELL_TYPE_KEYS)
    cellline_v, _ = pick(kv, R.CELL_LINE_KEYS)
    fallback_v, _ = pick(kv, R.FALLBACK_TISSUE_KEYS)

    samples = pd.DataFrame(index=meta.index)
    samples.index.name = "sample_id"
    samples["series_id"] = meta["series_id"]
    samples["source_name"] = meta["source_name"].map(norm)
    samples["tissue_key"] = tissue_k
    samples["tissue_value"] = tissue_v
    samples["cell_type_value"] = celltype_v
    samples["cell_line_value"] = cellline_v
    samples["tissue_fallback"] = fallback_v
    samples = samples.replace("", np.nan)
    samples["has_cell_line"] = samples["cell_line_value"].notna()
    samples["field"] = np.where(samples["tissue_value"].notna(), "tissue", "source_name")
    samples["string"] = samples["tissue_value"].fillna(samples["source_name"])

    n = len(samples)
    print("\nfield presence:")
    for c in ["source_name", "tissue_value", "cell_type_value", "cell_line_value", "tissue_fallback"]:
        print(f"  {c:<16} {pct(int(samples[c].notna().sum()), n)}")
    print(f"  no tissue value and no source name: {pct(int(samples['string'].isna().sum()), n)}")
    return samples


# ======================================================================================
# 2) UBERON
# ======================================================================================
def load_uberon(obo: Path):
    """``terms`` (id -> name) and ``lookup`` (label / exact / related -> {ids}), lowercase."""
    import obonet

    g = obonet.read_obo(str(obo))                  # obsolete terms are skipped
    terms = {n: d["name"] for n, d in g.nodes(data=True) if n.startswith("UBERON:") and d.get("name")}
    lookup = {"label": defaultdict(set), "exact": defaultdict(set), "related": defaultdict(set)}
    for n, d in g.nodes(data=True):
        if n not in terms:
            continue
        lookup["label"][d["name"].lower()].add(n)
        for s in d.get("synonym", []):
            m = SYN.match(s)
            if m and m.group(2) in ("EXACT", "RELATED"):
                lookup[m.group(2).lower()][m.group(1).lower()].add(n)
    version = g.graph.get("data-version") or g.graph.get("date") or "?"
    print(f"{len(terms):,} UBERON terms ({obo.name}, version {version})")
    return terms, lookup, str(version)


class Matcher:
    """The notebook's string -> UBERON matcher. The first step that fires decides:

    1. clean the text (MOJIBAKE, CHAR_FIXES, TYPOS) and build variants (prefixes and
       suffixes stripped, adjective -> noun, plural, cell suffixes, + tissue / organ, no
       parentheses, no tumor / "adjacent normal" words);
    2. MANUAL on any variant; 3. TUMOR_ORGAN on any variant;
    4. exclusions, no id: not a tissue, control rna, xenograft, cell line name, in vitro;
    5. special rules: hematopoietic, nasal epithelium, nasopharynx, blood (only when no
       other site is named);
    6. tumor abbreviations as single words; 7. MANUAL / TUMOR_ORGAN on the text inside
       parentheses;
    8. UBERON labels and EXACT synonyms on every variant, then RELATED synonyms
       (several terms -> "ambiguous"); 9. organ words (several organs -> ambiguous);
    10. "not a tissue" for generic leftovers and sample codes, else "no match".
    """

    def __init__(self, terms: dict[str, str], lookup: dict) -> None:
        self.terms, self.lookup = terms, lookup
        uid = self.uid
        self.manual = {k: uid(v) for k, v in R.MANUAL.items()}
        self.tumor_organ = {k: uid(v) for k, v in R.TUMOR_ORGAN.items()}
        self.organ_words = {k: uid(v) for k, v in R.ORGAN_WORDS.items()}
        self.blood, self.hemato = uid(R.BLOOD), uid(R.HEMATO)
        self.nasal_epi, self.nasopharynx = uid("nasal cavity epithelium"), uid("nasopharynx")

    def uid(self, label: str) -> str:
        """The one UBERON id of a label or EXACT synonym; stops on a missing or ambiguous one."""
        ids = self.lookup["label"].get(label) or self.lookup["exact"].get(label)
        if not ids or len(ids) != 1:
            raise SystemExit(f"Error: archs4_tissue_rules.py names {label!r}, which is "
                             f"{'ambiguous: ' + str(sorted(ids)) if ids else 'not a UBERON label or EXACT synonym'}")
        return next(iter(ids))

    # ---- text --------------------------------------------------------------------------
    @staticmethod
    def fix_text(s: str) -> str:
        for bad, good in R.MOJIBAKE.items():
            s = s.replace(bad, good)
        for bad, good in R.CHAR_FIXES.items():
            s = s.replace(bad, good)
        s = R.MANGLED_NBSP.sub("", s)
        return " ".join(R.TYPOS.get(w, w) for w in s.split())

    @staticmethod
    def _variants(s: str) -> list[str]:
        out, seen = [], set()

        def add(x):
            x = x.strip(" ,;-")
            if x and x not in seen:
                seen.add(x)
                out.append(x)

        add(s)
        x = s
        for _ in range(3):
            x = R.SUFFIX.sub("", R.PREFIX.sub("", x))
            add(x)
        for v in list(out):                                   # adjective -> noun ("rectal biopsy" -> "rectum")
            w = v.split(" ", 1)
            if w[0] in R.ADJ:
                add(R.SUFFIX.sub("", " ".join([R.ADJ[w[0]]] + w[1:])))
        for v in list(out):
            if v.endswith("s") and not v.endswith("ss"):
                add(v[:-1])
        for v in list(out):                                   # weaker variants last
            add(R.CELLS.sub("", v))
            add(v + " tissue")
            add(v + " organ")
        return out

    def variants(self, s: str) -> list[str]:
        """Variants of s, then of s without (parentheses) / "week N", then without tumor words."""
        out = self._variants(s)
        no_paren = re.sub(r"\s*\(.*?\)", "", s).strip()
        no_paren = re.sub(r"\s+(of\s+)?(gestational\s+)?weeks?\s+\d+", "", no_paren).strip()
        no_tumor = re.sub(r"\s+", " ", R.NORMAL_WORDS.sub("", R.TUMOR_WORDS.sub("", no_paren))).strip()
        for extra in (no_paren, no_tumor):
            if extra and extra != s:
                out += [v for v in self._variants(extra) if v not in out]
        return out

    def paren_variants(self, s: str) -> list[str]:
        """Variants of the text inside parentheses (4+ characters): the weakest evidence."""
        out = []
        for inner in re.findall(r"\(([^()]{4,})\)", s):
            out += [v for v in self._variants(inner.strip()) if v not in out]
        return out

    # ---- matching ----------------------------------------------------------------------
    def _hit(self, tid: str, how: str):
        return tid, self.terms[tid], how

    def match(self, s: str):
        """``(uberon_id, uberon_label, how)``; id and label are None when nothing matched."""
        s = self.fix_text(s)
        vs = self.variants(s)
        ps = [v for v in self.paren_variants(s) if v not in vs]
        tumor = "+tumor" if R.TUMOR_WORDS.search(s) else ""
        for v in vs:
            if v in self.manual:
                return self._hit(self.manual[v], "manual")
        for v in vs:
            if v in self.tumor_organ:
                return self._hit(self.tumor_organ[v], "tumor name+tumor")
        if s in R.NOT_A_TISSUE:
            return None, None, "not a tissue"
        if R.CONTROL_RNA.search(s):
            return None, None, "control rna"
        if R.XENOGRAFT.search(s):
            return None, None, "xenograft"
        if R.CELL_LINE_NAMES.search(s):
            return None, None, "cell line name"
        if R.IN_VITRO.search(s):
            return None, None, "in vitro"
        if R.HEMATO_RE.search(s):
            return self._hit(self.hemato, "hematopoietic rule")
        if R.NASAL_EPI.search(s):
            return self._hit(self.nasal_epi, "nasal rule")
        if R.NASOPHARYNX.search(s):
            return self._hit(self.nasopharynx, "nasal rule")
        if (R.BLOOD_WORD.search(s) or R.BLOOD_CELL_WORD.search(s)) and not R.NON_BLOOD_SITE.search(s):
            return self._hit(self.blood, "blood rule")
        for w in re.findall(r"[a-z0-9]+", s):                 # tumor abbreviations inside longer strings
            if w in self.tumor_organ:
                return self._hit(self.tumor_organ[w], "tumor name+tumor")
        for v in ps:                                          # explicit decisions on the text in parentheses
            if v in self.manual:
                return self._hit(self.manual[v], "manual")
            if v in self.tumor_organ:
                return self._hit(self.tumor_organ[v], "tumor name+tumor")
        for kinds in (("label", "exact"), ("related",)):    # strong evidence on all variants first
            for v in (vs + ps if kinds[0] == "label" else vs):
                if v in R.NOT_A_TISSUE:                       # "normal tissue" -> "tissue" says nothing
                    continue
                for kind in kinds:
                    if kind == "related" and v != s and v.endswith((" tissue", " organ")):
                        continue                              # "frontal" + " organ" is a synonym of the pineal body
                    ids = self.lookup[kind].get(v)
                    if ids:
                        if len(ids) > 1:
                            return None, None, f"ambiguous {kind}"
                        return self._hit(next(iter(ids)), kind + ("" if v == s else "+variant") + tumor)
        organs = {self.organ_words[m]
                  for m in R.ORGAN_RE.findall(" ".join(vs[:1] + [R.ADJ.get(w, w) for w in s.split()]))}
        if len(organs) == 1:
            return self._hit(organs.pop(), "organ word" + tumor)
        if len(organs) > 1:
            return None, None, "ambiguous organ words"
        if R.SAMPLE_CODE.search(s) or any(v in R.NOT_A_TISSUE for v in vs):
            return None, None, "not a tissue"
        return None, None, "no match"


# ======================================================================================
# 3) UBERON -> CellxGene tissue_general
# ======================================================================================
def tissue_general_of(ids) -> tuple[dict[str, tuple[str, str] | None], str]:
    """UBERON id -> (tissue_general id, label) or None; and the package version."""
    from importlib.metadata import version

    from cellxgene_ontology_guide.curated_ontology_term_lists import get_curated_ontology_term_list
    from cellxgene_ontology_guide.entities import CuratedOntologyTermList
    from cellxgene_ontology_guide.ontology_parser import OntologyParser

    parser = OntologyParser()
    general = get_curated_ontology_term_list(CuratedOntologyTermList.TISSUE_GENERAL)
    out = {}
    for tid in ids:
        try:
            hits = parser.get_high_level_terms(tid, general)
        except (KeyError, ValueError, IndexError):   # id unknown to CellxGene's UBERON version
            hits = []
        out[tid] = (hits[0], parser.get_term_label(hits[0])) if hits else None
    try:
        pkg = version("cellxgene-ontology-guide")
    except Exception:  # noqa: BLE001
        pkg = "?"
    return out, pkg


# ======================================================================================
# build-map
# ======================================================================================
def build_map(args) -> pd.DataFrame:
    started = datetime.now().isoformat(timespec="seconds")
    banner("1) metadata -> per-sample strings")
    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False).set_index("sample_id")
    meta.index = meta.index.astype(str)
    if not meta.index.is_unique:
        sys.exit("Error: duplicated sample_id in metadata.csv")
    print(f"{len(meta):,} samples in {args.metadata}")
    samples = build_samples(meta)
    n = len(samples)

    banner("2) strings -> UBERON")
    terms, lookup, obo_version = load_uberon(args.uberon_obo)
    matcher = Matcher(terms, lookup)
    has = samples.dropna(subset=["string"])
    queue = (has.groupby(["field", "string"])
             .agg(n_samples=("series_id", "size"), n_series=("series_id", "nunique"),
                  frac_cell_line=("has_cell_line", "mean"))
             .sort_values("n_samples", ascending=False).reset_index())
    print(f"{len(queue):,} distinct (field, string) pairs")
    queue[["uberon_id", "uberon_label", "match"]] = pd.DataFrame(
        queue["string"].map(matcher.match).tolist(), index=queue.index)

    # the notebook's per-string categories, for comparison with its numbers
    nb_cat = np.select(
        [(queue["frac_cell_line"] >= 0.5) | (queue["match"] == "cell line name"), queue["match"] == "in vitro",
         queue["match"] == "xenograft", queue["match"] == "control rna", queue["match"] == "not a tissue",
         queue["uberon_id"].notna()],
        ["cell_line", "in_vitro", "xenograft", "control_rna", "needs_context", "tissue"], default="unmatched")
    print("\nper-string categories as in the notebook (share of all samples):")
    print((pd.Series(queue["n_samples"].to_numpy(), index=nb_cat).groupby(level=0).sum()
           .sort_values(ascending=False) / n).map("{:.1%}".format).to_string())

    banner("3) UBERON -> CellxGene tissue_general")
    tg, tg_version = tissue_general_of(queue["uberon_id"].dropna().unique())
    no_tg = sorted(t for t, v in tg.items() if v is None)
    print(f"{len(tg):,} UBERON ids -> {len({v for v in tg.values() if v})} tissue_general categories; "
          f"{len(no_tg)} ids without one")

    # ---- per sample ----
    key = pd.MultiIndex.from_arrays([samples["field"], samples["string"]])
    lab = queue.set_index(["field", "string"])
    m = pd.DataFrame({c: lab[c].reindex(key).to_numpy() for c in ["uberon_id", "uberon_label", "match"]},
                     index=samples.index)
    m.loc[samples["string"].isna(), "match"] = "no string"
    tg_id = m["uberon_id"].map(lambda t: tg[t][0] if isinstance(t, str) and tg.get(t) else None)
    tg_label = m["uberon_id"].map(lambda t: tg[t][1] if isinstance(t, str) and tg.get(t) else None)

    cell_line = samples["has_cell_line"].to_numpy() | (m["match"] == "cell line name").to_numpy()
    category = np.select(
        [cell_line, m["match"] == "xenograft", m["match"] == "in vitro", tg_id.notna()],
        [CELL_LINE, XENOGRAFT, IN_VITRO, TISSUE], default=UNMATCHED)
    out = pd.DataFrame({
        "series_id": samples["series_id"],
        "category": category,
        "tissue_general": np.where(category == TISSUE, tg_label.astype(object),
                                   np.where(category == UNMATCHED, None, category)),
        "tissue_general_ontology_term_id": np.where(category == TISSUE, tg_id.astype(object), None),
        "uberon_id": m["uberon_id"],
        "uberon_label": m["uberon_label"],
        "match": m["match"],
        "tumor": m["match"].fillna("").str.endswith("+tumor"),
        "has_cell_line": samples["has_cell_line"],
        "field": samples["field"].where(samples["string"].notna()),
        "string": samples["string"],
    }, index=samples.index)
    out.loc[(out["uberon_id"].notna()) & (category == UNMATCHED), "match"] += " (no tissue_general)"

    stats = map_stats(out, queue, terms, no_tg, args.n_review)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.out_dir / MAP_FILE
    out.reset_index().to_csv(map_path, index=False, compression="gzip")
    rules = HERE / "archs4_tissue_rules.py"
    card = {
        "created": started,
        "metadata": {"path": str(args.metadata), "bytes": args.metadata.stat().st_size, "n_samples": n},
        "uberon_obo": {"path": str(args.uberon_obo), "version": obo_version},
        "cellxgene_ontology_guide": tg_version,
        "rules": {"path": str(rules), "sha256": hashlib.sha256(rules.read_bytes()).hexdigest()},
        "cell_line_rule": "per sample: a cell-line key, or a string naming a cell line",
        "map": str(map_path),
        "stats": stats,
    }
    (args.out_dir / CARD_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"\nmap written to {map_path} ({len(out):,} samples), run card {args.out_dir / CARD_FILE}")
    return out


def map_stats(out: pd.DataFrame, queue: pd.DataFrame, terms: dict, no_tg: list[str], n_review: int) -> dict:
    n = len(out)
    banner("statistics")
    cats = out["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    print("samples per category:")
    for c, k in cats.items():
        print(f"  {c:<10} {pct(int(k), n)}")

    tissue = out[out["category"] == TISSUE]
    print("\nmatch type among tissue samples:")
    print(tissue["match"].value_counts().map(lambda k: pct(int(k), len(tissue))).to_string())
    print(f"  tumor context: {pct(int(tissue['tumor'].sum()), len(tissue))}")

    labels = out["tissue_general"].value_counts()
    print(f"\n{labels.size} tissue_general values; top 40:")
    print(labels.head(40).map(lambda k: pct(int(k), n)).to_string())

    if no_tg:
        print(f"\nUBERON ids without a tissue_general ({len(no_tg)}; their samples are unmatched): "
              + ", ".join(f"{t} ({terms.get(t, '?')})" for t in no_tg[:15]) + (" ..." if len(no_tg) > 15 else ""))

    unmatched = out[out["category"] == UNMATCHED]
    print("\nunmatched samples by reason:")
    reason = pd.Series(np.where(unmatched["match"].str.endswith("(no tissue_general)"), "no tissue_general",
                                unmatched["match"].str.replace(r"\+.*", "", regex=True)))
    print(reason.value_counts().map(lambda k: pct(int(k), n)).to_string())

    string_is_tissue = (queue.set_index(["field", "string"])["uberon_id"].notna()
                        .reindex(pd.MultiIndex.from_arrays([out["field"], out["string"]]))
                        .eq(True).to_numpy())
    moved = out["has_cell_line"].to_numpy(dtype=bool) & string_is_tissue
    print(f"\nper-sample cell-line rule: {pct(int(moved.sum()), n)} samples with a cell-line key whose string "
          "maps to a tissue are labelled 'cell line'")

    top = queue.head(n_review)
    print(f"\ntop {len(top):,} strings: {top['uberon_id'].notna().mean():.1%} auto-mapped, covering "
          f"{top['n_samples'].sum() / n:.1%} of samples")
    review = top[top["match"].isin(["no match", "ambiguous label", "ambiguous exact", "ambiguous related",
                                    "ambiguous organ words"]) & (top["frac_cell_line"] < 0.5)]
    if len(review):
        print("most frequent strings without a decision (add them to archs4_tissue_rules.py):")
        print(review[["field", "string", "n_samples", "n_series", "match"]].head(25).to_string(index=False))

    return {
        "n_samples": n,
        "per_category": {c: int(k) for c, k in cats.items()},
        "tissue_general": {str(k): int(v) for k, v in labels.items()},
        "match_among_tissue": {str(k): int(v) for k, v in tissue["match"].value_counts().items()},
        "uberon_without_tissue_general": no_tg,
        "cell_line_rule_moved": int(moved.sum()),
    }


# ======================================================================================
# apply
# ======================================================================================
def _io():
    try:
        from anndata.io import read_elem, write_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem, write_elem
    return read_elem, write_elem


def load_map(path: Path) -> pd.DataFrame:
    path = path / MAP_FILE if path.is_dir() else path
    if not path.exists():
        sys.exit(f"Error: {path} not found; run build-map first")
    m = pd.read_csv(path, dtype=str, usecols=["sample_id", "category", "tissue_general"],
                    keep_default_na=False, na_values=[""]).set_index("sample_id")
    if m.index.has_duplicates:
        sys.exit(f"Error: {path} has {int(m.index.duplicated().sum()):,} duplicated sample ids")
    print(f"map: {path} ({len(m):,} samples)")
    return m


def _tree(f) -> dict[str, tuple]:
    """Every group and dataset with its shape, dtype and attributes (attrs compared by value)."""
    import h5py

    out = {}

    def visit(name, obj):
        attrs = tuple(sorted((k, repr(np.asarray(v).tolist())) for k, v in obj.attrs.items()))
        if isinstance(obj, h5py.Dataset):
            out[name] = ("dataset", obj.shape, str(obj.dtype), attrs)
        else:
            out[name] = ("group", attrs)

    out["/"] = ("group", tuple(sorted((k, repr(np.asarray(v).tolist())) for k, v in f.attrs.items())))
    f.visititems(visit)
    return out


def _decode(values) -> pd.Series:
    s = pd.Series(values)
    return s.map(lambda x: x.decode() if isinstance(x, bytes) else x)


def apply_map(args) -> None:
    import h5py

    read_elem, write_elem = _io()
    if not args.dry_run and not args.overwrite and args.out is None:
        sys.exit("Error: give --out FILE, --overwrite or --dry-run")
    if args.out is not None and args.overwrite:
        sys.exit("Error: --out and --overwrite are exclusive")
    if args.out is not None and args.out.resolve() == args.h5ad.resolve():
        sys.exit("Error: --out is the input; use --overwrite")

    mapping = load_map(args.map)
    col = args.column
    with h5py.File(args.h5ad, "r") as f:
        obs = f["obs"]
        if obs.attrs.get("encoding-type") != "dataframe":
            sys.exit(f"Error: {args.h5ad.name}: obs is not an anndata >= 0.7 dataframe group")
        index_key = obs.attrs["_index"]
        index_key = index_key.decode() if isinstance(index_key, bytes) else index_key
        obs_names = _decode(read_elem(obs[index_key])).astype(str)
        if args.id_column:
            if args.id_column not in obs:
                sys.exit(f"Error: obs has no column {args.id_column!r}")
            ids = _decode(read_elem(obs[args.id_column])).astype(str)
        else:
            ids = obs_names
        existed = col in obs
        old = _decode(read_elem(obs[col])) if existed else pd.Series([np.nan] * len(ids))
        old_kind = obs[col].attrs.get("encoding-type") if existed else None
        if isinstance(old_kind, bytes):
            old_kind = old_kind.decode()
    if existed and old_kind not in ("categorical", "string-array"):
        sys.exit(f"Error: obs/{col} is stored as {old_kind!r}; only categorical or string columns are replaced")

    ids = ids.str.strip().to_numpy()
    n = len(ids)
    found = pd.Index(mapping.index).get_indexer(ids) >= 0
    banner(f"apply -> {args.h5ad.name} obs['{col}']")
    print(f"rows: {n:,}; ids from {'obs[' + repr(args.id_column) + ']' if args.id_column else 'the obs index'}")
    print(f"  found in the map: {pct(int(found.sum()), n)}")
    if found.mean() < args.min_overlap:
        sys.exit(f"Error: only {found.mean():.1%} of the ids are in the map (< --min-overlap {args.min_overlap:.0%}); "
                 "check the id format. Nothing written.")
    hit = mapping.reindex(ids)
    print("  map category of the found rows:")
    for c, k in hit["category"].value_counts().reindex(CATEGORIES, fill_value=0).items():
        print(f"    {c:<10} {pct(int(k), n)}")
    if (~found).any():
        print(f"  ids not in the map, e.g.: {list(ids[~found][:5])}")

    new_label = hit["tissue_general"].to_numpy(dtype=object)
    relabel = pd.notna(new_label)
    old_v = old.to_numpy(dtype=object)
    new = np.where(relabel, new_label, old_v)
    if not existed:
        new = np.where(pd.isna(new), UNKNOWN, new)
    new = pd.Series(new, dtype=object)
    changed = (new.astype(str) != pd.Series(old_v).astype(str)).to_numpy()
    print(f"  relabelled from the map: {pct(int(relabel.sum()), n)}; kept as they were: {pct(int((~relabel).sum()), n)}")
    print(f"  value changed: {pct(int(changed.sum()), n)}" + ("" if existed else f" (new column, '{UNKNOWN}' elsewhere)"))

    if changed.any():
        pairs = (pd.DataFrame({"before": pd.Series(old_v).fillna("<NA>").astype(str)[changed],
                               "after": new[changed].astype(str)})
                 .value_counts().head(25))
        print("\nmost frequent changes (before -> after):")
        for (b, a), k in pairs.items():
            print(f"  {b[:35]:>35} -> {a:<30} {k:>9,}")
    after = new.fillna("<NA>").value_counts()
    print(f"\n{col} after ({after.size} values); top 40:")
    print(after.head(40).map(lambda k: pct(int(k), n)).to_string())

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    target = args.h5ad.with_name(args.h5ad.name + ".relabel.tmp") if args.overwrite else args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"\ncopying {args.h5ad} -> {target} ...")
    shutil.copyfile(args.h5ad, target)
    try:
        with h5py.File(target, "r+") as f:
            obs = f["obs"]
            if existed:
                del obs[col]
            values = new.astype(str).to_numpy(dtype=object)
            if old_kind == "string-array":
                write_elem(obs, col, values)
            else:
                write_elem(obs, col, pd.Categorical(values))
            if not existed:
                order = [c.decode() if isinstance(c, bytes) else str(c) for c in obs.attrs.get("column-order", [])]
                obs.attrs["column-order"] = np.array(order + [col], dtype=object)
        verify(args.h5ad, target, col, existed)
        if args.overwrite:
            os.replace(target, args.h5ad)
            target = args.h5ad
    except BaseException:
        if args.overwrite and target.exists():
            target.unlink()
        raise
    print(f"written: {target} (only obs/{col} changed)")


def verify(src: Path, dst: Path, col: str, existed: bool) -> None:
    """Fail unless ``dst`` differs from ``src`` only in obs/<col> (and obs column-order for a new column)."""
    import h5py

    read_elem, _ = _io()
    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        ta, tb = _tree(a), _tree(b)
        own = lambda k: k == f"obs/{col}" or k.startswith(f"obs/{col}/")   # noqa: E731
        if not existed:   # obs gains the column in column-order; its other attributes must stay
            order = lambda f: [c.decode() if isinstance(c, bytes) else str(c)   # noqa: E731
                               for c in f["obs"].attrs.get("column-order", [])]
            if order(b) != order(a) + [col]:
                raise RuntimeError(f"obs column-order is {order(b)}, expected {order(a) + [col]}")
            strip = lambda t: (t[0], tuple(x for x in t[1] if x[0] != "column-order"))   # noqa: E731
            ta["obs"], tb["obs"] = strip(ta["obs"]), strip(tb["obs"])
        diff = sorted(k for k in set(ta) | set(tb) if not own(k) and ta.get(k) != tb.get(k))
        if diff:
            raise RuntimeError(f"unexpected changes besides obs/{col}: {diff[:10]}")
        index_key = a["obs"].attrs["_index"]
        n_old = len(read_elem(a["obs"][index_key.decode() if isinstance(index_key, bytes) else index_key]))
        n_new = len(read_elem(b["obs"][col]))
        if n_old != n_new:
            raise RuntimeError(f"obs/{col} has {n_new} rows, obs has {n_old}")
    print("verified: every other group, dataset and attribute is unchanged")


# ======================================================================================
# curate
# ======================================================================================
SRC_CELLXGENE, SRC_ARCHS4, SRC_CULTURE = "cellxgene", "archs4", "archs4_culture"
SRC_KEYWORD, SRC_HEURISTIC, SRC_NONE = "keyword", "heuristic", "none"
TMP = "__curate_tmp_"


def _str(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _column_order(obs) -> list[str]:
    return [_str(c) for c in obs.attrs.get("column-order", [])]


def _read_obs(path: Path, columns: list[str]) -> tuple[pd.Index, dict[str, pd.Series | None]]:
    """obs index and the requested columns (None when absent), via h5py."""
    import h5py

    read_elem, _ = _io()
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        if obs.attrs.get("encoding-type") != "dataframe":
            sys.exit(f"Error: {path.name}: obs is not an anndata >= 0.7 dataframe group")
        names = pd.Index(_decode(read_elem(obs[_str(obs.attrs["_index"])])).astype(str))
        cols = {c: (_decode(read_elem(obs[c])) if c in obs else None) for c in columns}
    return names, cols


def _h5ad_labels(paths: list[Path], column: str) -> set[str]:
    """Labels of ``column`` in the --vocab-h5ad files (CellxGene h5ads: pseudobulk, single cell)."""
    labels: set[str] = set()
    for path in paths:
        _, cols = _read_obs(path, [column])
        if cols[column] is None:
            sys.exit(f"Error: --vocab-h5ad {path} has no obs column {column!r}")
        found = set(cols[column].dropna().astype(str)) - {UNKNOWN}
        print(f"vocabulary: {len(found)} labels from {path.name}")
        labels |= found
    return labels


def curate_values(args, ids: np.ndarray, old: pd.Series, keyword_meta: pd.DataFrame | None,
                  mapping: pd.DataFrame | None, vocab: set[str]) -> tuple[pd.Series, pd.Series, dict]:
    """New curated labels and their source per row (see ``curate``)."""
    n = len(ids)
    old_s = old.astype(object).where(old.notna(), None)
    if args.modality != "bulk":
        label = old_s.fillna(UNKNOWN).astype(str)
        return label, pd.Series(np.full(n, SRC_CELLXGENE, dtype=object)), {}

    hit = mapping.reindex(ids)
    cat = hit["category"].to_numpy(dtype=object)
    tg = hit["tissue_general"].to_numpy(dtype=object)
    label = np.full(n, UNKNOWN, dtype=object)
    source = np.where(pd.Series(old_s).isin([None, UNKNOWN, ""]).to_numpy(), SRC_NONE, SRC_HEURISTIC).astype(object)
    is_tissue = (cat == TISSUE) & pd.notna(tg)
    is_culture = np.isin(cat, [CELL_LINE, IN_VITRO, XENOGRAFT])
    label[is_tissue], source[is_tissue] = tg[is_tissue], SRC_ARCHS4
    label[is_culture], source[is_culture] = cat[is_culture], SRC_CULTURE

    renamed: dict = {}
    if args.trust_heuristic:
        from tissue_names import CELLXGENE_NAME, keyword_tissue

        not_archs4 = pd.isna(cat)
        rows = np.flatnonzero(not_archs4 & (source == SRC_HEURISTIC))
        cache: dict[tuple, str] = {}
        kw = np.full(n, None, dtype=object)
        values = keyword_meta.astype(str).to_numpy()
        for i in rows:
            key = tuple(values[i])
            if key not in cache:
                cache[key] = keyword_tissue(key)
            kw[i] = cache[key]
        reproduced = np.array([kw[i] is not None and kw[i] == old_s.iloc[i] for i in range(n)])
        for i in np.flatnonzero(reproduced):
            target = CELLXGENE_NAME.get(old_s.iloc[i])
            renamed.setdefault((old_s.iloc[i], target, target in vocab), 0)
            renamed[(old_s.iloc[i], target, target in vocab)] += 1
            if target in vocab:
                label[i], source[i] = target, SRC_KEYWORD
    return pd.Series(label, dtype=object), pd.Series(source, dtype=object), renamed


def _shared_report(old: pd.Series, new: pd.Series, vocab_pb: set[str] | None) -> None:
    if not vocab_pb:
        return
    for name, s in (("before", old), ("after", new)):
        v = s.astype(str)
        known = ~v.isin([UNKNOWN, "nan", "None", ""])
        shared = v.isin(vocab_pb) & known
        print(f"  {name}: {v[known].nunique():>3} labels, {v[shared].nunique():>3} shared with --vocab-h5ad; "
              f"rows with a shared label {pct(int(shared.sum()), len(v))}")


def curate(args) -> None:
    """``tissue_general`` -> ``tissue_noncurated``; curated ``tissue_general`` + ``tissue_label_source``."""
    import h5py

    _, write_elem = _io()
    modes = sum([bool(args.in_place), args.out is not None, bool(args.dry_run)])
    if modes != 1:
        sys.exit("Error: give exactly one of --in-place, --out FILE, --dry-run")
    if args.out is not None and args.out.resolve() == args.h5ad.resolve():
        sys.exit("Error: --out is the input; use --in-place")
    if args.trust_heuristic and not args.keyword_columns:
        sys.exit("Error: --trust-heuristic needs --keyword-columns (the obs strings the heuristic labels came from)")
    bulk = args.modality == "bulk"
    col, nc_col, src_col = args.column, args.noncurated_column, args.source_column

    wanted = [col, nc_col, src_col] + ([args.id_column] if args.id_column else []) + list(args.keyword_columns or [])
    names, cols = _read_obs(args.h5ad, wanted)
    for c in (src_col,) + ((nc_col,) if bulk else ()):
        if cols[c] is not None:
            sys.exit(f"Error: obs/{c} already exists in {args.h5ad.name}: curate was run on this file already")
    if cols[col] is None:
        sys.exit(f"Error: obs has no column {col!r}")
    missing_kw = [c for c in args.keyword_columns or [] if cols[c] is None]
    if missing_kw:
        sys.exit(f"Error: --keyword-columns not in obs: {missing_kw}")
    old = cols[col]
    ids = (cols[args.id_column] if args.id_column else pd.Series(names)).astype(str).str.strip().to_numpy()
    n = len(ids)

    mapping = load_map(args.map) if bulk else None
    if bulk:
        found = pd.Index(mapping.index).get_indexer(ids) >= 0
        if found.mean() < args.min_overlap:
            sys.exit(f"Error: only {found.mean():.1%} of the ids are in the map (< --min-overlap "
                     f"{args.min_overlap:.0%}); check the id format. Nothing written.")
    vocab_pb = _h5ad_labels(args.vocab_h5ad, col) if (bulk and args.vocab_h5ad) else set()
    vocab = (set(mapping.loc[mapping["category"] == TISSUE, "tissue_general"].dropna()) | vocab_pb
             if bulk else set())

    noncurated = old
    if bulk and args.noncurated_from is not None:
        o_names, o_cols = _read_obs(args.noncurated_from, [col])
        if o_cols[col] is None:
            sys.exit(f"Error: --noncurated-from {args.noncurated_from} has no obs column {col!r}")
        orig = pd.Series(o_cols[col].to_numpy(), index=o_names)
        orig = orig[~orig.index.duplicated()]
        aligned = orig.reindex(names)
        n_from = int(aligned.notna().sum())
        noncurated = pd.Series(np.where(aligned.notna(), aligned.to_numpy(dtype=object), old.to_numpy(dtype=object)),
                               dtype=object)
        print(f"{nc_col}: {n_from:,} of {n:,} rows from {args.noncurated_from}, the rest from the current {col}")
    elif bulk:
        print(f"[note] {nc_col} = the current {col}. Rows an earlier 'apply --overwrite' relabelled hold that "
              f"label, not their original one (give --noncurated-from ORIGINAL.h5ad to restore it).")

    keyword_meta = pd.DataFrame({c: cols[c] for c in args.keyword_columns}) if args.trust_heuristic else None
    label, source, renamed = curate_values(args, ids, noncurated if bulk else old, keyword_meta, mapping, vocab)

    banner(f"curate {args.modality} -> {args.h5ad.name}")
    print(f"rows: {n:,}")
    print(f"{src_col}:")
    for s, k in source.value_counts().items():
        print(f"  {s:<16} {pct(int(k), n)}")
    if renamed:
        print("--trust-heuristic: heuristic label -> CellxGene name (rows reproduced by keywords alone)")
        for (a, b, ok), k in sorted(renamed.items(), key=lambda x: -x[1]):
            print(f"  {a:>20} -> {str(b):<22} {k:>9,}" + ("" if ok else "   [not in the vocabulary: unknown]"))
    if bulk:
        before = old.astype(object).where(old.notna(), "<NA>").astype(str)
        changed = (before != label.astype(str)).to_numpy()
        print(f"{col} changed: {pct(int(changed.sum()), n)}")
        if changed.any():
            pairs = pd.DataFrame({"before": before[changed], "after": label[changed].astype(str)}).value_counts().head(25)
            print("most frequent changes (before -> after):")
            for (b, a), k in pairs.items():
                print(f"  {b[:35]:>35} -> {a:<30} {k:>9,}")
        after = label.value_counts()
        print(f"\n{col} after ({after.size} values); top 40:")
        print(after.head(40).map(lambda k: pct(int(k), n)).to_string())
        if vocab_pb:
            print("labels shared with the --vocab-h5ad files (what the class-aware losses can pair):")
            _shared_report(old, label, vocab_pb)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    target = args.h5ad
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"\ncopying {args.h5ad} -> {args.out} ...")
        shutil.copyfile(args.h5ad, args.out)
        target = args.out
    with h5py.File(target, "r") as f:
        before_tree = _tree(f)
        order0 = _column_order(f["obs"])

    new_cols = [src_col] + ([nc_col] if bulk else [])
    order1 = order0 + [c for c in new_cols if c not in order0]
    done: list[tuple] = []   # undo log
    try:
        with h5py.File(target, "r+") as f:
            obs = f["obs"]
            write_elem(obs, TMP + "src", pd.Categorical(source.astype(str).to_numpy(dtype=object)))
            done.append(("del", TMP + "src"))
            if bulk:
                write_elem(obs, TMP + "new", pd.Categorical(label.astype(str).to_numpy(dtype=object)))
                done.append(("del", TMP + "new"))
                if args.noncurated_from is not None:
                    nc_values = noncurated.astype(object).where(noncurated.notna(), UNKNOWN).astype(str)
                    write_elem(obs, TMP + "nc", pd.Categorical(nc_values.to_numpy(dtype=object)))
                    done.append(("del", TMP + "nc"))
                    obs.move(col, TMP + "old")
                    done.append(("move", TMP + "old", col))
                    obs.move(TMP + "nc", nc_col)
                    done.append(("move", nc_col, TMP + "nc"))
                else:
                    obs.move(col, nc_col)
                    done.append(("move", nc_col, col))
                obs.move(TMP + "new", col)
                done.append(("move", col, TMP + "new"))
            obs.move(TMP + "src", src_col)
            done.append(("move", src_col, TMP + "src"))
            obs.attrs["column-order"] = np.array(order1, dtype=object)
            done.append(("order", order0))
        verify_curated(before_tree, target, order1, [col, nc_col, src_col, TMP + "old"])
        with h5py.File(target, "r+") as f:
            if TMP + "old" in f["obs"]:
                del f["obs"][TMP + "old"]   # last step: --noncurated-from holds the original labels
        done = []
    except BaseException:
        if done:
            print("[error] undoing the changes ...")
            with h5py.File(target, "r+") as f:
                obs = f["obs"]
                for op in reversed(done):
                    if op[0] == "del" and op[1] in obs:
                        del obs[op[1]]
                    elif op[0] == "move" and op[1] in obs:
                        obs.move(op[1], op[2])
                    elif op[0] == "order":
                        obs.attrs["column-order"] = np.array(op[1], dtype=object)
        if args.out is not None and target.exists():
            target.unlink()
        raise
    what = f"obs/{col} curated, old values in obs/{nc_col}, obs/{src_col} added" if bulk else f"obs/{src_col} added"
    print(f"written: {target} ({what})")


def verify_curated(before: dict, path: Path, order1: list[str], own_cols: list[str]) -> None:
    """Fail unless only the ``own_cols`` obs groups and obs column-order changed."""
    import h5py

    read_elem, _ = _io()
    with h5py.File(path, "r") as f:
        after = _tree(f)
        if _column_order(f["obs"]) != order1:
            raise RuntimeError(f"obs column-order is {_column_order(f['obs'])}, expected {order1}")
        n_obs = len(read_elem(f["obs"][_str(f["obs"].attrs["_index"])]))
        for c in own_cols:
            if c in f["obs"] and len(read_elem(f["obs"][c])) != n_obs:
                raise RuntimeError(f"obs/{c} does not have {n_obs} rows")
        if any(k.startswith(f"obs/{TMP}") and not k.startswith(f"obs/{TMP}old") for k in after):
            raise RuntimeError("temporary obs groups left behind")
    own = lambda k: any(k == f"obs/{c}" or k.startswith(f"obs/{c}/") for c in own_cols)   # noqa: E731
    strip = lambda t: (t[0], tuple(x for x in t[1] if x[0] != "column-order"))   # noqa: E731
    before, after = dict(before), dict(after)
    before["obs"], after["obs"] = strip(before["obs"]), strip(after["obs"])
    diff = sorted(k for k in set(before) | set(after) if not own(k) and before.get(k) != after.get(k))
    if diff:
        raise RuntimeError(f"unexpected changes besides {own_cols}: {diff[:10]}")
    print(f"verified: only obs/{{{', '.join(c for c in own_cols if not c.startswith(TMP))}}} "
          "and the obs column order changed")


# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-map", help="metadata.csv -> per-sample tissue_general map",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    b.add_argument("--metadata", type=Path, required=True, help="ARCHS4 metadata.csv (sample_id, characteristics, ...)")
    b.add_argument("--uberon-obo", type=Path, required=True, help="uberon-basic.obo")
    b.add_argument("--out-dir", type=Path, default=DEFAULT_MAP_DIR, help="Where the map and its run card go")
    b.add_argument("--n-review", type=int, default=2000, help="Top strings summarized for review")

    a = sub.add_parser("apply", help="write the map into obs[--column] of an h5ad",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("--h5ad", type=Path, required=True)
    a.add_argument("--map", type=Path, default=DEFAULT_MAP_DIR, help=f"Map directory or {MAP_FILE}")
    out = a.add_mutually_exclusive_group()
    out.add_argument("--out", type=Path, default=None, help="Write a relabelled copy here")
    out.add_argument("--overwrite", action="store_true", help="Relabel the input (via a temporary copy)")
    a.add_argument("--column", default="tissue_general", help="obs column to write")
    a.add_argument("--id-column", default=None, help="obs column with the ARCHS4 sample ids (default: the obs index)")
    a.add_argument("--min-overlap", type=float, default=0.05,
                   help="Refuse when fewer ids than this share are in the map (an id format mismatch)")
    a.add_argument("--dry-run", action="store_true", help="Print the statistics, write nothing")

    c = sub.add_parser(
        "curate", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="curated tissue_general (old one -> tissue_noncurated) + tissue_label_source")
    c.add_argument("--h5ad", type=Path, required=True)
    c.add_argument("--modality", required=True, choices=["bulk", "single_cell", "pseudobulk"],
                   help="bulk: curate from the map; single_cell/pseudobulk: CellxGene labels, only the source column is added")
    c.add_argument("--map", type=Path, default=DEFAULT_MAP_DIR, help=f"Map directory or {MAP_FILE} (bulk)")
    mode = c.add_mutually_exclusive_group()
    mode.add_argument("--in-place", action="store_true",
                      help="Edit the file itself (obs only, no copy; undone on failure)")
    mode.add_argument("--out", type=Path, default=None, help="Write a curated copy here")
    mode.add_argument("--dry-run", action="store_true", help="Print the statistics, write nothing")
    c.add_argument("--column", default="tissue_general", help="Label column to curate")
    c.add_argument("--noncurated-column", default="tissue_noncurated", help="Where the old labels go (bulk)")
    c.add_argument("--noncurated-from", type=Path, default=None,
                   help="Original h5ad whose --column holds the labels from before 'apply --overwrite' (bulk)")
    c.add_argument("--source-column", default="tissue_label_source",
                   help="New column: cellxgene, archs4, archs4_culture, keyword, heuristic or none")
    c.add_argument("--id-column", default=None, help="obs column with the ARCHS4 sample ids (default: the obs index)")
    c.add_argument("--min-overlap", type=float, default=0.05,
                   help="Refuse when fewer ids than this share are in the map (bulk)")
    c.add_argument("--vocab-h5ad", type=Path, nargs="*", default=[],
                   help="CellxGene h5ads (e.g. pseudobulk) whose labels extend the vocabulary and are "
                        "reported as shared classes (bulk)")
    c.add_argument("--trust-heuristic", action="store_true",
                   help="Keep a non-ARCHS4 row's old label (renamed to its CellxGene name) when keyword "
                        "matching over --keyword-columns reproduces it (bulk)")
    c.add_argument("--keyword-columns", nargs="*", default=[], help="obs string columns for --trust-heuristic")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-map":
        build_map(args)
    elif args.command == "curate":
        curate(args)
    else:
        apply_map(args)


if __name__ == "__main__":
    main()
