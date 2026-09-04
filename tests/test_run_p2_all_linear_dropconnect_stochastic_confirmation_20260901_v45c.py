"""Contract tests for P2 v45c stochastic confirmation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_all_linear_dropconnect_stochastic_confirmation_20260901_v45c as runner  # noqa: E402


def test_config_is_exact_v45_contract_with_only_new_disjoint_seeds() -> None:
    config = runner.load_config()
    source = json.loads(
        (ROOT / config["authorization_evidence"]["v45_config"]).read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["seeds"] == [20260904, 20260905, 20260906]
    assert source["training"]["seeds"] == [20260901, 20260902, 20260903]
    assert not set(config["training"]["seeds"]) & set(source["training"]["seeds"])
    assert runner._without_seeds(config["training"]) == runner._without_seeds(
        source["training"]
    )
    assert config["evaluation"] == source["evaluation"]
    assert config["source_contract"] == source["source_contract"]


def test_two_target_free_preflights_are_byte_identical() -> None:
    left = runner.preflight()
    right = runner.preflight()
    assert left == right
    assert left["status"] == "ZERO_OPERATION_CONFIRMATION_PREFLIGHT_READY"
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


def test_dropconnect_architecture_and_inference_are_exactly_v45() -> None:
    receipt = runner.v45._dropconnect_contract_receipt()
    assert receipt["module_count"] == 5
    assert receipt["module_shapes"] == [
        [32, 8],
        [32, 32],
        [32, 75],
        [32, 32],
        [1, 32],
    ]
    assert receipt["parameters"] == 4865
    assert receipt["parameter_tensors"] == 10
    assert receipt["buffers"] == 0
    assert receipt["evaluation_initial_function_maximum_abs_error"] == 0.0
    assert receipt["evaluation_rng_unchanged"] is True
    assert receipt["dropout_module_count"] == 0


def test_permutation_and_masked_future_isolation() -> None:
    receipt = runner.v45._isolation_receipt()
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["masked_or_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["repeat_maximum_abs_error"] == 0.0


def test_confirmation_claim_does_not_call_exposed_blocks_fresh() -> None:
    config = runner.load_config()
    contract = config["confirmation_contract"]
    assert contract["same_exposed_folds_not_fresh_temporal_surface"] is True
    assert contract["candidate_selection_between_seed_trios"] is False
    assert contract["v45_original_commitment_remains_representative"] is True
    assert contract["automatic_retry_count"] == 0


def test_v26a_gate_and_access_contract_are_unchanged() -> None:
    config = runner.load_config()
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
    assert config["operation_limits"] == {
        "maximum_candidate_count": 1,
        "maximum_fit_count": 9,
        "automatic_retry_count": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def test_frozen_source_hashes_and_status() -> None:
    config = runner.load_config()
    evidence = config["authorization_evidence"]
    source = json.loads(
        (ROOT / evidence["v45_result"]).read_text(encoding="utf-8")
    )
    assert source["status"] == "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
    assert source["candidate"]["prospective_fold_layer_gate"]["pass"] is True
    assert source["candidate"]["prospective_fold_layer_gate"]["non_harm_cells"] == 9
    assert runner.v12.sha256_file(ROOT / evidence["v45_result"]) == evidence[
        "v45_result_sha256"
    ]
    assert runner.v12.sha256_file(ROOT / evidence["v45_prediction"]) == evidence[
        "v45_prediction_sha256"
    ]
