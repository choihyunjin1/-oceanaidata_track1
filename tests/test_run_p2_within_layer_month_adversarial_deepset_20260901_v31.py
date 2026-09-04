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

import run_p2_within_layer_month_adversarial_deepset_20260901_v31 as M  # noqa: E402


def test_config_is_fixed_v13_task_plus_one_domain_adversary() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    adversary = training["domain_adversary"]
    assert training["task_architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert training["optimizer"] == "exact_v13_AdamW"
    assert adversary["gradient_reversal_coefficient"] == 0.1
    assert adversary["domain_loss_multiplier"] == 1.0
    assert adversary["task_context_adversarialized"] is False
    assert adversary["schedule"] is False
    assert adversary["coefficient_sweep"] is False
    assert adversary["month_router"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_gradient_reversal_formula_is_exact() -> None:
    receipt = M._gradient_reversal_contract_receipt()
    assert receipt["coefficient"] == 0.1
    assert np.allclose(receipt["observed_gradient"], [-0.3, -0.4])
    assert receipt["formula_exact"] is True
    assert receipt["schedule"] is False
    assert receipt["coefficient_sweep"] is False


def test_task_initialization_and_permutation_match_exact_v13() -> None:
    receipt = M._model_contract_receipt()
    assert receipt["v13_task_state_byte_equal_at_initialization"] is True
    assert receipt["v13_task_output_maximum_abs_error"] == 0.0
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["task_context_adversarialized"] is False
    assert receipt["domain_head_count"] == 3
    assert receipt["domain_classes_per_head"] == 12


def test_layer_month_capture_and_conditional_heads() -> None:
    layers = np.resize(np.array([2, 3, 4], dtype=int), 30)
    local = pd.date_range("2024-05-01", periods=30, freq="12h", tz="Asia/Seoul")
    weights, receipt = M.domain_balanced_weights(layers, local)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.mean(), 1.0)
    assert M._ACTIVE_LAYER_IDS is not None
    assert M._ACTIVE_MONTH_IDS is not None
    assert np.array_equal(M._ACTIVE_LAYER_IDS, layers)
    assert np.array_equal(M._ACTIVE_MONTH_IDS, local.month.to_numpy())
    assert receipt["adversary_target_layers"] == [2, 3, 4]
    model = M.MonthAdversarialDeepSet()
    pooled = torch.randn(9, 64, requires_grad=True)
    layer_tensor = torch.tensor([2, 3, 4] * 3)
    month_tensor = torch.tensor([5, 5, 5, 6, 6, 6, 7, 7, 7])
    loss, accuracy, heads = M.domain_adversarial_loss(
        model,
        pooled,
        layer_tensor,
        month_tensor,
        torch.ones(9),
        0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert heads == 3
    assert pooled.grad is not None
    assert torch.isfinite(pooled.grad).all()


def test_masked_future_prefix_and_prospective_gate_are_sealed() -> None:
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


def test_two_zero_operation_preflights_are_byte_identical() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["semantic_audit"]["repository_p2_exact_execution_hits"] == 0
    assert first["semantic_audit"]["p1_code_or_output_reused"] is False


def test_tiny_training_has_finite_active_domain_receipts() -> None:
    rng = np.random.default_rng(31)
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
        31,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["gradient_reversal_coefficient"] == 0.1
    assert receipt["domain_loss_multiplier"] == 1.0
    assert receipt["minimum_domain_heads_per_batch"] == 3
    assert np.isfinite(receipt["domain_loss_first"])
    assert 0.0 <= receipt["domain_accuracy_last"] <= 1.0
    assert receipt["task_context_adversarialized"] == 0
    assert receipt["schedule"] == 0
    assert receipt["coefficient_sweep"] == 0
    assert receipt["month_router"] == 0
    assert receipt["row_deletion"] == 0
