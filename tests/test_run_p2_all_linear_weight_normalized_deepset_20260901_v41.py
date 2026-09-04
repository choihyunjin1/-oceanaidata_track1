from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_all_linear_weight_normalized_deepset_20260901_v41 as runner  # noqa: E402


def test_config_is_exact_v13_plus_five_linear_weight_norms() -> None:
    config = runner.load_config()
    training = config["training"]
    norm = training["weight_normalization"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert norm["dimension"] == 0
    assert norm["linear_module_count"] == 5
    assert norm["learned_per_output_magnitude"]
    assert norm["unit_direction"]
    assert not norm["spectral_normalization"]
    assert not norm["power_iteration"]
    assert not norm["sweep"]


def test_weight_norm_initial_function_and_parametrization_contract() -> None:
    receipt = runner._weight_norm_contract_receipt()
    assert receipt["initial_function_preserved"]
    assert receipt["initial_function_maximum_abs_error"] <= 1e-6
    assert receipt["linear_names_exact"]
    assert receipt["linear_module_count"] == 5
    assert receipt["parametrization_class_names"] == ["_WeightNorm"] * 5
    assert receipt["maximum_initial_magnitude_error"] <= 1e-7
    assert receipt["maximum_unit_direction_norm_error"] <= 1e-6
    assert receipt["maximum_effective_magnitude_error"] <= 1e-6
    assert not receipt["bias_parametrized"]
    assert not receipt["buffer_names"]
    assert not receipt["spectral_or_power_iteration_state"]
    assert receipt["deterministic_inference"]


def test_weight_norm_model_is_masked_permutation_and_repeat_invariant() -> None:
    receipt = runner._isolation_receipt()
    assert max(receipt.values()) <= 1e-6


def test_semantic_audit_does_not_reopen_spectral_family() -> None:
    audit = runner.semantic_audit(runner.load_config())
    assert audit["repository_p2_exact_execution_hits"] == 0
    assert audit["v27_spectral_family_not_reopened"]
    assert audit["v27_code_import_count"] == 0
    assert audit["spectral_tolerance_changes"] == 0
    assert audit["v29_lookahead_distinguished"]
    assert audit["v34_gradient_centralization_distinguished"]
    assert audit["v35_radam_distinguished"]
    assert audit["v40_dropout_consistency_distinguished"]
    assert not audit["official_v23_feedback_used_for_selection"]


def test_two_preflights_are_byte_identical_and_zero_operation() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    for key in (
        "data_rows_read",
        "model_fits",
        "artifacts_written",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert first[key] == 0


def test_tiny_weight_norm_training_contract_is_finite() -> None:
    generator = np.random.default_rng(41)
    rows = 24
    config = runner.load_config()
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = rows
    output, receipt = runner.train_predict_seed(
        generator.normal(size=(rows, 5, 8)).astype(np.float32),
        np.ones((rows, 5), dtype=np.float32),
        generator.normal(size=(rows, 11)).astype(np.float32),
        generator.normal(size=rows).astype(np.float32),
        np.ones(rows, dtype=np.float32),
        generator.normal(size=(8, 5, 8)).astype(np.float32),
        np.ones((8, 5), dtype=np.float32),
        generator.normal(size=(8, 11)).astype(np.float32),
        config,
        41,
    )
    assert output.shape == (8,)
    assert np.isfinite(output).all()
    assert receipt["optimizer_steps"] == 1
    assert receipt["weight_norm_module_count"] == 5
    assert receipt["parameters"] == 4994
    assert receipt["parameter_tensors"] == 15
    assert receipt["spectral_buffer_count"] == 0
    assert receipt["batch_stat_buffer_count"] == 0
    assert not receipt["bias_parametrized"]
    assert len(receipt["initial_parametrization_state_sha256"]) == 64
    assert len(receipt["final_parametrization_state_sha256"]) == 64
    assert receipt["initial_parametrization_state_sha256"] != receipt[
        "final_parametrization_state_sha256"
    ]
