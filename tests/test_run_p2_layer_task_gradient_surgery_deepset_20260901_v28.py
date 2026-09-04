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

import run_p2_layer_task_gradient_surgery_deepset_20260901_v28 as M  # noqa: E402


def test_config_is_one_fixed_v13_pcgrad_change_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    surgery = training["gradient_surgery"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert surgery["task_labels"] == [2, 3, 4]
    assert surgery["task_order"] == [2, 3, 4]
    assert surgery["negative_dot_only"] is True
    assert surgery["task_order_sweep"] is False
    assert surgery["task_reweight_sweep"] is False
    assert training["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_pcgrad_formula_order_no_conflict_noop_and_determinism() -> None:
    receipt = M._projection_contract_receipt()
    assert receipt["fixed_task_order"] == [2, 3, 4]
    assert receipt["opposite_gradient_projection_count"] == 2
    assert receipt["opposite_gradient_residual_l2_max"] <= 1e-12
    assert receipt["zero_conflict_exact_noop"] is True
    assert receipt["byte_identical_replay"] is True
    assert receipt["projection_reference"] == "unmodified_other_task_gradient"


def test_masked_future_and_layer_permutation_isolation() -> None:
    receipt = M._isolation_receipt()
    assert receipt["masked_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6


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


def test_flat_gradient_assignment_round_trip() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    parameters = [value for value in model.parameters() if value.requires_grad]
    expected = torch.arange(sum(value.numel() for value in parameters), dtype=torch.float32)
    M._assign_flat_gradient(parameters, expected)
    actual = torch.cat([value.grad.reshape(-1) for value in parameters])
    assert torch.equal(actual, expected)


def test_tiny_training_has_finite_pcgrad_receipts() -> None:
    rng = np.random.default_rng(28)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    layers = np.resize(np.array([2, 3, 4]), rows)
    local = pd.date_range("2024-05-01", periods=rows, freq="10min", tz="Asia/Seoul")
    weights, _receipt = M.domain_balanced_weights(layers, local)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 1
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
        28,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["task_labels"] == [2, 3, 4]
    assert receipt["missing_task_batches"] == 0
    assert receipt["directional_task_pairs"] == 6
    assert 0.0 <= receipt["negative_dot_projection_share"] <= 1.0
    assert receipt["row_deletion"] == 0
