"""Contract tests for sealed P2 v46 LayerNorm DeepSets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_hidden_activation_layer_normalized_deepset_20260901_v46 as runner  # noqa: E402


def test_negative_fingerprint_and_primary_source_are_sealed() -> None:
    config = runner.load_config()
    assert config["semantic_audit"]["exact_execution_hits_before_preregistration"] == 0
    assert config["primary_source"]["url"] == "https://arxiv.org/abs/1607.06450"
    assert config["primary_source"]["use"].endswith("no P2 performance transfer")


def test_exact_four_layernorm_placements_and_parameter_contract() -> None:
    receipt = runner._layernorm_contract_receipt()
    assert receipt["layernorm_names"] == [
        "element.1",
        "element.4",
        "head.1",
        "head.4",
    ]
    assert receipt["normalized_shapes"] == [[32], [32], [32], [32]]
    assert receipt["eps"] == [1e-5, 1e-5, 1e-5, 1e-5]
    assert all(receipt["elementwise_affine"])
    assert max(receipt["initial_weight_errors"]) == 0.0
    assert max(receipt["initial_bias_errors"]) == 0.0
    assert receipt["parameters"] == 5121
    assert receipt["parameter_tensors"] == 18
    assert receipt["buffers"] == 0


def test_linear_parameters_are_exact_v13_but_normalized_function_is_new() -> None:
    receipt = runner._layernorm_contract_receipt()
    assert receipt["linear_parameter_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["initial_function_maximum_abs_difference_vs_v13"] > 0.0
    assert receipt["gradients_finite"] is True


def test_layernorm_has_no_batch_statistics_or_stochastic_modules() -> None:
    receipt = runner._layernorm_contract_receipt()
    assert receipt["batch_composition_maximum_abs_error"] <= 1e-6
    assert receipt["repeat_maximum_abs_error"] == 0.0
    assert receipt["normalization_mean_maximum_abs_error"] <= 1e-5
    assert receipt["normalization_variance_maximum_abs_error"] <= 1e-3
    assert receipt["batchnorm_count"] == 0
    assert receipt["dropout_count"] == 0


def test_permutation_and_masked_future_isolation() -> None:
    receipt = runner._isolation_receipt()
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["masked_or_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["repeat_maximum_abs_error"] == 0.0


def test_two_target_free_preflights_are_byte_identical() -> None:
    left = runner.preflight()
    right = runner.preflight()
    assert left == right
    assert left["status"] == "ZERO_OPERATION_PREFLIGHT_READY"
    assert left["preflight_sha256"] == right["preflight_sha256"]
    assert all(
        left[name] == 0
        for name in (
            "data_rows_read",
            "model_fits",
            "artifacts_written",
            "official_rows_read",
            "hidden_rows_read",
            "submission_csv_created",
            "uploads",
        )
    )


def test_v13_science_and_v26a_gate_are_unchanged() -> None:
    config = runner.load_config()
    training = config["training"]
    gate = config["evaluation"]["safety_gate"]
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["objective"] == "exact_v13_weighted_SmoothL1_beta_1.0"
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["maximum_fit_count"] == 9
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
