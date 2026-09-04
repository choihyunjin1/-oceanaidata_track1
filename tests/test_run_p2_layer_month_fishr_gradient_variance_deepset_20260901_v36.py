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

import run_p2_layer_month_fishr_gradient_variance_deepset_20260901_v36 as M  # noqa: E402


def test_config_is_exact_v13_with_one_fixed_fishr_change() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    fishr = training["fishr"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0001
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert fishr["coefficient"] == 1.0
    assert fishr["environment"] == "target_layer_x_KST_calendar_month"
    assert fishr["minimum_rows_per_environment_in_batch"] == 2
    assert fishr["minimum_supported_environments_in_batch"] == 2
    assert fishr["ema"] is False
    assert fishr["warmup"] is False
    assert fishr["sweep"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_environment_ids_are_exact_layer_x_kst_calendar_month() -> None:
    ids = M.environment_ids(
        np.array([2, 2, 3, 4]),
        pd.DatetimeIndex(
            [
                "2025-01-01T00:00:00+09:00",
                "2025-02-01T00:00:00+09:00",
                "2025-01-01T00:00:00+09:00",
                "2025-12-01T00:00:00+09:00",
            ]
        ),
    )
    assert ids.tolist() == [201, 202, 301, 412]


def test_forward_with_final_features_matches_original_model() -> None:
    receipt = M._forward_contract_receipt()
    assert receipt["prediction_maximum_abs_error"] <= 1e-7
    assert receipt["final_feature_shape"] == [5, 32]
    assert receipt["final_head_gradient_dimension"] == 33


def test_fishr_contract_is_permutation_invariant_and_differentiable() -> None:
    receipt = M._fishr_contract_receipt()
    assert receipt["permutation_invariant"] is True
    assert receipt["gradients_finite"] is True
    assert receipt["coefficient"] == 1.0
    assert receipt["supported_environment_count"] == 2
    assert receipt["skipped_environment_count"] == 0
    assert receipt["gradient_dimension"] == 3
    assert receipt["variance_correction"] == 0
    assert receipt["penalty_finite"] is True


def test_masked_future_permutation_and_prospective_gate_are_sealed() -> None:
    isolation = M._isolation_receipt()
    assert isolation["masked_token_maximum_abs_error"] <= 1e-6
    assert isolation["permutation_maximum_abs_error"] <= 1e-6
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003


def test_two_target_free_preflights_are_byte_identical_and_access_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["environment_contract"]["exact"] is True
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0
    assert first["semantic_audit"]["official_v23_feedback_used_for_selection"] is False
    for name in (
        "data_rows_read",
        "model_fits",
        "artifacts_written",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert first[name] == 0


def test_tiny_training_has_finite_fishr_receipt() -> None:
    rng = np.random.default_rng(36)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows).astype(np.float32)
    M._CURRENT_ENVIRONMENT_IDS = np.repeat([201, 202, 301], 10).astype(np.int64)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = rows
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
        36,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["optimizer_steps"] == 2
    assert receipt["fishr_penalty_steps"] == 2
    assert receipt["coefficient"] == 1.0
    assert receipt["supported_environment_count_min"] == 3
    assert receipt["final_head_gradient_dimension"] == 33
    assert receipt["variance_correction"] == 0
    assert receipt["ema_steps"] == 0
    assert receipt["warmup_steps"] == 0
    assert receipt["row_deletion"] == 0
    assert np.isfinite(
        [
            receipt["base_loss_first"],
            receipt["base_loss_last"],
            receipt["fishr_penalty_first"],
            receipt["fishr_penalty_last"],
        ]
    ).all()


def test_semantic_axis_is_distinct_and_official_feedback_selection_is_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    audit = M.semantic_audit(config)
    assert audit["classification"] == "NEW_P2_LAYER_MONTH_FINAL_HEAD_DIAGONAL_FISHR"
    assert audit["repository_p2_exact_execution_hits"] == 0
    for name in (
        "v18_group_dro_distinguished",
        "v19_vrex_distinguished",
        "v23_input_gradient_distinguished",
        "v28_pcgrad_distinguished",
        "v30_irm_distinguished",
        "v31_dann_distinguished",
        "v33_mldg_distinguished",
    ):
        assert audit[name] is True
    assert audit["official_v23_feedback_used_for_selection"] is False
