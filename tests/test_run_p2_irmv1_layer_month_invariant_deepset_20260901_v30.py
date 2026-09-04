from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_irmv1_layer_month_invariant_deepset_20260901_v30 as M  # noqa: E402


def test_config_is_one_fixed_v13_irmv1_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    irm = training["irmv1"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert training["optimizer"] == "exact_v13_AdamW"
    assert irm["environment_axis"] == "target_layer_x_calendar_month"
    assert irm["dummy_classifier_scale"] == 1.0
    assert irm["coefficient"] == 1.0
    assert irm["annealing"] is False
    assert irm["coefficient_sweep"] is False
    assert irm["environment_router"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_irmv1_penalty_formula_and_determinism() -> None:
    receipt = M._irmv1_contract_receipt()
    assert receipt["toy_environment_count"] == 2
    assert receipt["toy_expected_penalty"] == 2.0
    assert receipt["toy_observed_penalty"] == 2.0
    assert receipt["formula_exact"] is True
    assert receipt["byte_identical_replay"] is True
    assert receipt["annealing"] is False
    assert receipt["coefficient_sweep"] is False


def test_penalty_uses_separate_environment_optimality_gradients() -> None:
    prediction = torch.tensor([0.0, 2.0, 1.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0, 1.0])
    weights = torch.ones(3)
    grouped, count = M.irmv1_penalty(
        prediction, target, weights, torch.tensor([201, 202, 202])
    )
    pooled, pooled_count = M.irmv1_penalty(
        prediction, target, weights, torch.tensor([201, 201, 201])
    )
    assert count == 2
    assert pooled_count == 1
    assert torch.isfinite(grouped)
    assert torch.isfinite(pooled)
    assert not torch.isclose(grouped, pooled)


def test_layer_month_environment_capture_preserves_v13_weights() -> None:
    layers = np.resize(np.array([2, 3, 4], dtype=int), 30)
    local = pd.date_range("2024-05-01", periods=30, freq="12h", tz="Asia/Seoul")
    weights, receipt = M.domain_balanced_weights(layers, local)
    assert len(weights) == 30
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)
    assert np.isclose(weights.mean(), 1.0)
    assert M._ACTIVE_ENVIRONMENT_IDS is not None
    expected = layers * 100 + local.month.to_numpy()
    assert np.array_equal(M._ACTIVE_ENVIRONMENT_IDS, expected)
    assert receipt["irmv1_environment_count"] == len(np.unique(expected))


def test_masked_future_permutation_prefix_and_gate_contracts() -> None:
    receipt = M._isolation_receipt()
    assert receipt["masked_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        expected = pd.Timestamp(start) - pd.Timedelta(days=7)
        assert pd.Timestamp(preflight["prefix_cutoffs"][fold]) == expected
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
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


def test_tiny_training_has_finite_active_irmv1_receipts() -> None:
    rng = np.random.default_rng(30)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    layers = np.resize(np.array([2, 3, 4], dtype=int), rows)
    local = pd.date_range("2024-05-01", periods=rows, freq="12h", tz="Asia/Seoul")
    weights, _ = M.domain_balanced_weights(layers, local)
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
        30,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["irmv1_coefficient"] == 1.0
    assert receipt["dummy_classifier_scale"] == 1.0
    assert receipt["minimum_batch_environment_count"] >= 3
    assert receipt["irmv1_penalty_max"] > 0.0
    assert receipt["annealing"] == 0
    assert receipt["coefficient_sweep"] == 0
    assert receipt["environment_router"] == 0
    assert receipt["row_deletion"] == 0
