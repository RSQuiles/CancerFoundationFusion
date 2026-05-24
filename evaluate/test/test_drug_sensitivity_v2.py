from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from evaluate.finetune.tasks.drug_sensitivity_v2 import (
    DrugSensitivityV2Task,
    binarize_response,
    compute_prism_response,
    load_drug_response_labels,
    load_expression_csv,
)


def test_prism_response_formula_and_binarization() -> None:
    response = compute_prism_response(
        ic50_log2=np.array([0.0, 1.0, 2.0]),
        slope=np.array([1.0, 1.0, 1.0]),
        dose=2.0,
    )

    expected = 1.0 / (1.0 + 2.0 ** (-(np.array([1.0, 0.0, -1.0]))))
    np.testing.assert_allclose(response, expected)
    np.testing.assert_array_equal(binarize_response(response, 0.5), np.array([1.0, 1.0, 0.0]))


def test_expression_parser_transposed_header_shape(tmp_path) -> None:
    x_path = tmp_path / "x.csv"
    pd.DataFrame(
        {
            "row": ["gene_symbol", "ensembl_gene_id", "model_name", "CL_A", "CL_B"],
            "g1": ["TP53", "ENSG00000141510.17", "ignored", "1.0", "2.0"],
            "g2": ["KRAS", "ENSG00000133703.13", "ignored", "3.0", "4.0"],
        }
    ).to_csv(x_path, index=False)

    adata = load_expression_csv(x_path)

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["CL_A", "CL_B"]
    assert adata.obs["model_name"].tolist() == ["CL_A", "CL_B"]
    assert adata.var_names.tolist() == ["ENSG00000141510", "ENSG00000133703"]
    expected = np.log1p(
        np.array(
            [
                [1.0 / 4.0 * 1e4, 3.0 / 4.0 * 1e4],
                [2.0 / 6.0 * 1e4, 4.0 / 6.0 * 1e4],
            ],
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(adata.X, expected)


def test_drug_response_parser_ic50_and_slope(tmp_path) -> None:
    y_path = tmp_path / "y.csv"
    pd.DataFrame(
        {
            "model_name": ["CL_A", "CL_B", "CL_C"],
            "DrugX_IC50": [0.0, 2.0, np.nan],
            "DrugX_slope": [1.0, 1.0, 1.0],
        }
    ).to_csv(y_path, index=False)

    labels = load_drug_response_labels(y_path, "DrugX", dose=2.0, response_threshold=0.5)

    assert labels.index.tolist() == ["CL_A", "CL_B"]
    assert labels["label"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(labels.loc["CL_A", "response"], 2.0 / 3.0)


def test_load_data_with_synthetic_csvs_and_dose_mapping(tmp_path) -> None:
    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"

    pd.DataFrame(
        {
            "row": ["gene_symbol", "ensembl_gene_id", "CL_A", "CL_B", "CL_C", "CL_D"],
            "g1": ["TP53", "ENSG1", "1.0", "2.0", "3.0", "4.0"],
            "g2": ["KRAS", "ENSG2", "2.0", "3.0", "4.0", "5.0"],
        }
    ).to_csv(x_path, index=False)
    pd.DataFrame(
        {
            "model_name": ["CL_A", "CL_B", "CL_C", "CL_D"],
            "DrugX_IC50": [0.0, 2.0, 0.5, 3.0],
            "DrugX_slope": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(y_path, index=False)

    cfg = OmegaConf.create(
        {
            "pretrained_model_path": "unused.ckpt",
            "head_learning_rate": 0.001,
            "x_path": str(x_path),
            "y_path": str(y_path),
            "drug": "DrugX",
            "drug_doses": {"DrugX": 2.0},
            "test_size": 0.5,
            "train_test_split_version": 1,
        }
    )

    task = DrugSensitivityV2Task()
    output_dim, train_adata, test_adata, train_targets, test_targets = task.load_data(cfg, embedder=None)

    assert output_dim == 1
    assert train_adata.n_obs == 2
    assert test_adata.n_obs == 2
    assert train_targets.shape == (2,)
    assert test_targets.shape == (2,)
    assert set(np.concatenate([train_targets, test_targets]).tolist()) == {0.0, 1.0}
