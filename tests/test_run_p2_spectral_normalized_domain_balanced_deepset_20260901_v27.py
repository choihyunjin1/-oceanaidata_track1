from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_spectral_normalized_domain_balanced_deepset_20260901_v27 as M  # noqa: E402


def test_config_is_one_fixed_v13_operator_norm_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    spectral = training["spectral_normalization"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert spectral["target_operator_norm"] == 1.0
    assert spectral["n_power_iterations_per_training_forward"] == 1
    assert spectral["coefficient_sweep"] is False
    assert spectral["selective_layer_application"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_spectral_norm_mask_and_layer_permutation_contract() -> None:
    receipt = M._spectral_contract_receipt()
    assert receipt["spectral_normalized_linear_count"] == 5
    assert receipt["maximum_operator_norm_abs_error_from_one"] <= 1e-4
    assert receipt["masked_token_isolation_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_invariance_maximum_abs_error"] <= 1e-6
    assert receipt["bias_normalized"] is False


def test_prefix_cutoffs_and_prospective_gate_are_sealed() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        expected = pd.Timestamp(start) - pd.Timedelta(days=7)
        assert pd.Timestamp(preflight["prefix_cutoffs"][fold]) == expected
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
    assert preflight["gate_amendment_sha256"] == config["authorization_evidence"][
        "prospective_gate_amendment_sha256"
    ]
    assert preflight["data_rows_read"] == 0
    assert preflight["model_fits"] == 0
    assert preflight["official_rows_read"] == 0
    assert preflight["hidden_rows_read"] == 0


def test_zero_operation_preflight_is_byte_identical() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0


def test_prospective_fold_layer_gate_requires_coverage_and_tolerance() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    values = [-0.1] * 8 + [0.002]
    record = {"by_fold_layer": {}}
    for fold_index, fold in enumerate(("a", "b", "c")):
        record["by_fold_layer"][fold] = {
            str(layer): {"delta_rmse": values[fold_index * 3 + offset]}
            for offset, layer in enumerate((2, 3, 4))
        }
    assert M.prospective_fold_layer_gate(record, config)["pass"] is True
    record["by_fold_layer"]["c"]["4"]["delta_rmse"] = 0.004
    assert M.prospective_fold_layer_gate(record, config)["pass"] is False
    record["by_fold_layer"]["c"]["4"]["delta_rmse"] = 0.002
    record["by_fold_layer"]["c"]["3"]["delta_rmse"] = 0.001
    assert M.prospective_fold_layer_gate(record, config)["pass"] is False


def test_tiny_training_has_finite_spectral_receipts() -> None:
    rng = np.random.default_rng(27)
    rows = 24
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.ones(rows, dtype=np.float32)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 12
    prediction, receipt = M.train_predict_seed(
        tokens,
        mask,
        context,
        target,
        weights,
        tokens[:7],
        mask[:7],
        context[:7],
        config,
        27,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["spectral_normalized_linear_count"] == 5
    assert receipt["effective_spectral_norm_min"] >= 0.98
    assert receipt["effective_spectral_norm_max"] <= 1.02
    assert receipt["input_gradient_penalty"] == 0
    assert receipt["parameter_neighborhood_perturbation"] == 0
