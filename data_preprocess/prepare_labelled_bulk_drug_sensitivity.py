from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("/cluster/work/boeva/bulkFM/data/raw/drug_sens_prediction")
DEFAULT_EXPRESSION_PATH = (DEFAULT_DATA_ROOT / "rnaseq_merged_rsem_expected_count_20250922.csv")
DEFAULT_LABELS_PATH = DEFAULT_DATA_ROOT / "bulk_download.csv"
DEFAULT_MODEL_LIST_PATH = DEFAULT_DATA_ROOT / "model_list_latest.csv"
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[1] / "vocab.json"
DEFAULT_OUTPUT_DIR = Path("/cluster/work/boeva/bulkFM/data/processed/drug_sens_prediction")

DEFAULT_DRUGS = (
    "Tanespimycin",
    "Idelalisib",
    "Regorafenib",
    "Lenvatinib",
    "Venetoclax",
    "Nutlin-3a",
    "Navitoclax",
    "azacitidine",
    "Vorinostat",
    "Decitabine",
    "Irinotecan",
    "Teniposide",
    "Topotecan",
    "Vincristine",
    "Paclitaxel",
    "Docetaxel",
    "Temsirolimus",
    "Rapamycin",
    "Nelarabine",
    "cytarabine hydrochloride",
    "Methotrexate",
    "Vismodegib",
    "Bexarotene",
    "Tretinoin",
    "Pevonedistat",
    "Bortezomib",
    "Tamoxifen",
    "Fulvestrant",
    "Bicalutamide",
    "Lenalidomide",
    "Olaparib",
    "Rucaparib",
    "Veliparib",
    "Carmustine",
    "Carboplatin",
    "Temozolomide",
    "bleomycin A2",
    "Doxorubicin",
    "Mitoxantrone",
    "omacetaxine mepesuccinate",
)
DEFAULT_CMAX = {
    "Tanespimycin": 8.07,
    "Idelalisib": 0.761,
    "Regorafenib": 4.48,
    "Lenvatinib": np.nan,
    "Venetoclax": 4.6,
    "Nutlin-3a": 3.07,
    "Navitoclax": 1.2,
    "azacitidine": 0.323,
    "Vorinostat": 5.78,
    "Decitabine": 23.14,
    "Irinotecan": 0.0149,
    "Teniposide": 0.0065,
    "Topotecan": 4.27,
    "Vincristine": 5.47,
    "Paclitaxel": 0.567,
    "Docetaxel": 0.016,
    "Temsirolimus": 16.81,
    "Rapamycin": 54.4,
    "Nelarabine": 1.3,
    "cytarabine hydrochloride": 33.9,
    "Methotrexate": 3.38,
    "Vismodegib": 1.15,
    "Bexarotene": 6.3,
    "Tretinoin": 0.312,
    "Pevonedistat": 0.107,
    "Bortezomib": 0.041,
    "Tamoxifen": 1.78,
    "Fulvestrant": 1.73,
    "Bicalutamide": 13.11,
    "Lenalidomide": 6.0,
    "Olaparib": 5.2,
    "Rucaparib": 19.38,
    "Veliparib": 134.7,
    "Carmustine": 37.59,
    "Carboplatin": 706.4,
    "Temozolomide": 6.73,
    "bleomycin A2": 0.715,
    "Doxorubicin": 0.046,
    "Mitoxantrone": np.nan,
    "omacetaxine mepesuccinate": np.nan,
}


def clean_step_1(text: Any) -> str:
    """Strict cleaning copied from PRISM match_bulk.ipynb."""
    if pd.isna(text):
        return ""
    return str(text).upper().replace(" ", "")


def clean_step_2(text: Any) -> str:
    """Aggressive cell-line-name cleaning copied from PRISM match_bulk.ipynb."""
    base = clean_step_1(text)
    base = re.split(r"[\[\(]", base)[0]
    return base.replace(";", "").replace("/", "").replace("-", "").replace(".", "").strip()


def response_from_ic50_slope(
    ic50_log2: pd.Series | np.ndarray,
    slope: pd.Series | np.ndarray,
    dose_linear: float,
) -> np.ndarray:
    """PRISM response at a linear dose from log2 IC50 and slope."""
    if pd.isna(dose_linear) or dose_linear <= 0:
        return np.full_like(ic50_log2, np.nan, dtype=np.float64)
    dose_log2 = float(np.log2(float(dose_linear)))
    ic50 = np.asarray(ic50_log2, dtype=np.float64)
    slope_arr = np.asarray(slope, dtype=np.float64)
    response = 1.0 / (1.0 + np.power(2.0, -(slope_arr * (dose_log2 - ic50))))
    return np.clip(response, 0.0, 1.0)


def viability_from_ic50_slope(
    ic50_log2: pd.Series | np.ndarray,
    slope: pd.Series | np.ndarray,
    dose_linear: float,
) -> np.ndarray:
    """PRISM viability at a linear dose. Higher response means lower viability."""
    return 1.0 - response_from_ic50_slope(ic50_log2, slope, dose_linear)


def load_vocab(vocab_path: Path) -> dict[str, int]:
    with vocab_path.open() as f:
        vocab = json.load(f)
    return {
        str(gene): int(idx)
        for gene, idx in vocab.items()
        if not str(gene).startswith("<")
    }


def prepare_expression(
    expression_path: Path,
    vocab_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Load raw Cell Model Passports counts and align gene columns to FM vocab."""
    vocab_genes = load_vocab(vocab_path)

    expression = pd.read_csv(
        expression_path,
        index_col=[0, 1, 2],
        header=[0, 1, 2],
        low_memory=False,
    ).T
    expression.index = expression.index.set_names(["model_id", "model_name", "data_source"])

    ensembl = (
        pd.Index(expression.columns.get_level_values("ensembl_gene_id").astype(str))
        .str.replace(r"\.\d+$", "", regex=True)
    )
    in_vocab = ensembl.isin(vocab_genes)
    expression = expression.loc[:, in_vocab].copy()
    ensembl = ensembl[in_vocab]

    gene_meta = pd.DataFrame(
        {
            "gene_symbol": expression.columns.get_level_values("gene_symbol").astype(str),
            "ensembl_gene_id": ensembl,
            "gene_id": expression.columns.get_level_values("gene_id").astype(str),
        }
    )

    numeric_expr = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_expr.columns = ensembl
    grouped_expr = numeric_expr.T.groupby(level=0, sort=False).sum().T

    gene_meta = (
        gene_meta.groupby("ensembl_gene_id", sort=False)
        .agg(
            {
                "gene_symbol": lambda values: "|".join(pd.unique(values.astype(str))),
                "gene_id": lambda values: "|".join(pd.unique(values.astype(str))),
            }
        )
        .reset_index()
    )

    ordered_genes = sorted(grouped_expr.columns, key=lambda gene: vocab_genes[gene])
    grouped_expr = grouped_expr.loc[:, ordered_genes]
    gene_meta = gene_meta.set_index("ensembl_gene_id").loc[ordered_genes].reset_index()

    expr_out = grouped_expr.copy()
    model_ids = expr_out.index.get_level_values("model_id").astype(str)
    cleaned_model_names = expr_out.index.get_level_values("model_name").map(clean_step_2)
    model_id_to_clean_name = dict(zip(model_ids, cleaned_model_names))
    expr_out.index = pd.Index(cleaned_model_names, name="model_name")

    duplicated_samples = expr_out.index[expr_out.index.duplicated()].unique().tolist()
    if duplicated_samples:
        expr_out = expr_out.loc[~expr_out.index.duplicated(keep="first")]

    report = {
        "n_expression_rows_raw": int(expression.shape[0]),
        "n_expression_rows_after_clean_dedup": int(expr_out.shape[0]),
        "n_gene_columns_raw": int(len(in_vocab)),
        "n_gene_columns_in_vocab_before_grouping": int(in_vocab.sum()),
        "n_gene_columns_after_duplicate_sum": int(grouped_expr.shape[1]),
        "n_duplicate_ensembl_columns_summed_extra": int(in_vocab.sum() - grouped_expr.shape[1]),
        "n_duplicate_sample_names_dropped": int(len(duplicated_samples)),
        "duplicate_sample_names_dropped_first_20": duplicated_samples[:20],
        "first_10_genes": ordered_genes[:10],
    }
    return expr_out, gene_meta, model_id_to_clean_name, report


def match_cell_lines(annotations_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Match annotations to label cellosaurus IDs using match_bulk.ipynb logic."""
    ref_ids = labels_df["cellosaurus_id"].dropna().unique()
    ref_map_1 = {clean_step_1(x): x for x in ref_ids}
    ref_map_2 = {clean_step_2(x): x for x in ref_ids}

    results = []
    for idx, row in annotations_df.iterrows():
        match_found = None
        match_source = None
        match_method = None
        candidates = []

        if "model_name" in row and pd.notna(row["model_name"]):
            candidates.append(row["model_name"])
        if "synonyms" in row and pd.notna(row["synonyms"]):
            candidates.extend(s.strip() for s in str(row["synonyms"]).split(";"))
        if "CCLE_ID" in row and pd.notna(row["CCLE_ID"]):
            candidates.append(re.split(r"_", str(row["CCLE_ID"]))[0].strip())
        candidates = list(dict.fromkeys(candidates))

        for cand in candidates:
            clean_cand = clean_step_1(cand)
            if clean_cand in ref_map_1:
                match_found = ref_map_1[clean_cand]
                match_source = cand
                match_method = "Step 1 (Strict)"
                break

        if not match_found:
            for cand in candidates:
                clean_cand = clean_step_2(cand)
                if clean_cand in ref_map_2:
                    match_found = ref_map_2[clean_cand]
                    match_source = cand
                    match_method = "Step 2 (Aggressive)"
                    break

        results.append(
            {
                "original_index": idx,
                "matched_id": match_found,
                "match_source_name": match_source,
                "match_method": match_method,
            }
        )

    results_df = pd.DataFrame(results)
    final_df = annotations_df.merge(results_df, left_index=True, right_on="original_index")
    return final_df.merge(labels_df, left_on="matched_id", right_on="cellosaurus_id", how="left")


def prepare_labels(
    labels_path: Path,
    model_list_path: Path,
    expression_index: pd.Index,
    model_id_to_clean_name: dict[str, str],
    drugs: list[str],
    cmax_by_drug: dict[str, float],
    max_fitted_mae: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(labels_path, index_col=0)
    annotations = pd.read_csv(model_list_path, index_col=0)

    labels = labels[labels["drug_name"].isin(drugs)].copy()
    labels = labels[pd.to_numeric(labels["fitted_mae"], errors="coerce") < float(max_fitted_mae)]
    labels["IC50"] = pd.to_numeric(labels["IC50"], errors="coerce")
    labels["slope"] = pd.to_numeric(labels["slope"], errors="coerce")
    labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=["IC50", "slope"])

    for drug in drugs:
        dose = cmax_by_drug[drug]
        drug_mask = labels["drug_name"] == drug
        labels.loc[drug_mask, "Cmax_viability"] = viability_from_ic50_slope(
            labels.loc[drug_mask, "IC50"],
            labels.loc[drug_mask, "slope"],
            dose,
        )

    matched_data = match_cell_lines(annotations, labels)
    matched_data = matched_data[matched_data["matched_id"].notna()].copy()
    matched_data = matched_data[matched_data["original_index"].isin(model_id_to_clean_name)]
    matched_data["model_name"] = matched_data["original_index"].map(model_id_to_clean_name)
    matched_data = matched_data[matched_data["model_name"].isin(expression_index)]

    cols = ["IC50", "slope", "Cmax_viability"]
    y_long = matched_data[["model_name", "drug_name"] + cols].drop_duplicates(
        subset=["model_name", "drug_name"] + cols
    )
    y_wide = y_long.pivot_table(
        index="model_name",
        columns="drug_name",
        values=cols,
        aggfunc="first",
    )
    y_wide = y_wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    y_wide.columns = [f"{drug}_{metric}" for drug, metric in y_wide.columns]
    y_wide.index.name = "model_name"

    report = {
        "selected_drugs": drugs,
        "cmax_by_drug": cmax_by_drug,
        "max_fitted_mae": float(max_fitted_mae),
        "n_label_rows_after_drug_filter": int(labels.shape[0]),
        "n_matched_label_rows": int(matched_data.shape[0]),
        "n_labelled_models": int(y_wide.shape[0]),
        "label_non_null_counts": {
            str(col): int(y_wide[col].notna().sum()) for col in y_wide.columns
        },
    }
    return y_wide, report


def write_expression_csv(
    expr: pd.DataFrame,
    gene_meta: pd.DataFrame,
    output_path: Path,
) -> None:
    metadata = pd.DataFrame(
        [
            gene_meta["gene_symbol"].to_numpy(),
            gene_meta["ensembl_gene_id"].to_numpy(),
            gene_meta["gene_id"].to_numpy(),
            [""] * gene_meta.shape[0],
        ],
        index=["gene_symbol", "ensembl_gene_id", "gene_id", "model_name"],
        columns=gene_meta["ensembl_gene_id"].to_numpy(),
    )
    out = pd.concat([metadata, expr], axis=0)
    out.index.name = "gene_symbol"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)


def parse_cmax(values: list[str], drugs: list[str]) -> dict[str, float]:
    cmax = dict(DEFAULT_CMAX)
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --cmax value '{item}'. Expected DRUG=DOSE.")
        drug, dose = item.split("=", 1)
        cmax[drug] = float(dose)
    missing = [drug for drug in drugs if drug not in cmax]
    if missing:
        raise ValueError(f"Missing Cmax values for selected drugs: {missing}")
    return {drug: float(cmax[drug]) for drug in drugs}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare labelled bulk expression and drug-response CSVs for CFF fine-tuning."
    )
    parser.add_argument("--expression-path", type=Path, default=DEFAULT_EXPRESSION_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--model-list-path", type=Path, default=DEFAULT_MODEL_LIST_PATH)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drugs", nargs="+", default=list(DEFAULT_DRUGS))
    parser.add_argument(
        "--cmax",
        nargs="*",
        default=[f"{drug}={dose}" for drug, dose in DEFAULT_CMAX.items()],
        help="Linear Cmax doses as DRUG=DOSE.",
    )
    parser.add_argument("--max-fitted-mae", type=float, default=0.3)
    parser.add_argument("--expression-output-name", default="gene_expression.csv")
    parser.add_argument("--labels-output-name", default="drug_response.csv")
    parser.add_argument("--report-output-name", default="labelled_bulk_report.json")
    args = parser.parse_args()

    drugs = [str(drug) for drug in args.drugs]
    cmax_by_drug = parse_cmax(args.cmax, drugs)

    expr, gene_meta, model_id_to_clean_name, expression_report = prepare_expression(
        args.expression_path,
        args.vocab_path,
    )
    y, label_report = prepare_labels(
        args.labels_path,
        args.model_list_path,
        expr.index,
        model_id_to_clean_name,
        drugs,
        cmax_by_drug,
        args.max_fitted_mae,
    )

    shared_models = expr.index.intersection(y.index)
    expr = expr.loc[shared_models]
    y = y.loc[shared_models]

    expression_output = args.output_dir / args.expression_output_name
    labels_output = args.output_dir / args.labels_output_name
    report_output = args.output_dir / args.report_output_name

    write_expression_csv(expr, gene_meta, expression_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    y.to_csv(labels_output)

    report = {
        "expression_path": str(args.expression_path),
        "labels_path": str(args.labels_path),
        "model_list_path": str(args.model_list_path),
        "vocab_path": str(args.vocab_path),
        "expression_output": str(expression_output),
        "labels_output": str(labels_output),
        "n_shared_labelled_models": int(len(shared_models)),
        "expression": expression_report,
        "labels": label_report,
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
