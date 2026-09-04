"""Contract tests for sealed P2 v48 diagonal-bilinear profile pooling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_diagonal_bilinear_profile_pooling_deepset_20260901_v48 as runner  # noqa: E402


def test_negative_fingerprint_and_primary_source_are_sealed() -> None:
    config = runner.load_config()
    semantic = config["semantic_audit"]
    assert semantic["exact_execution_hits_before_preregistration"] == 0
    assert semantic["semantic_execution_hits_before_preregistration"] == 0
    assert config["primary_source"]["url"].startswith(
        "https://openaccess.thecvf.com/content_iccv_2015/"
    )
    assert config["primary_source"]["use"].endswith("no P2 performance transfer")


def test_diagonal_bilinear_descriptor_is_identity_initialized() -> None:
    receipt = runner._bilinear_contract_receipt()
    assert receipt["descriptor"] == "masked_mean(element_embedding_squared)"
    assert receipt["embedding_width"] == 32
    assert receipt["descriptor_width"] == 107
    assert receipt["new_head_columns"] == 32
    assert receipt["new_head_columns_initial_maximum_abs"] == 0.0
    assert receipt["parameters"] == 5889
    assert receipt["parameter_tensors"] == 10
    assert receipt["buffers"] == 0
    assert receipt["linear_count"] == 5


def test_initial_v13_function_and_second_moment_contract() -> None:
    receipt = runner._bilinear_contract_receipt()
    assert receipt["unchanged_parameter_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["initial_function_maximum_abs_error_vs_v13"] == 0.0
    assert receipt["descriptor_forward_maximum_abs_error"] == 0.0
    assert receipt["diagonal_second_moment_maximum_abs_error"] == 0.0
    assert receipt["all_missing_second_moment_maximum_abs"] == 0.0
    assert receipt["learned_bilinear_columns_change_function_maximum_abs"] > 0.0
    assert receipt["gradients_finite"] is True
    assert receipt["new_head_column_gradient_finite_nonzero"] is True


def test_bilinear_axis_has_no_normalization_dropout_attention_or_batch_leak() -> None:
    receipt = runner._bilinear_contract_receipt()
    assert receipt["normalization_count"] == 0
    assert receipt["dropout_count"] == 0
    assert receipt["attention_count"] == 0
    assert receipt["batch_composition_maximum_abs_error"] <= 1e-6


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
    assert left["historical_block_support"]["supported_folds"] == 3
    assert (
        left["historical_block_support"]["finite_pooled_profile_context_width"]
        == 107
    )
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
