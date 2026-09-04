from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_score_aligned_weighted_mse_deepset_20260901_v32 as M  # noqa: E402


def test_config_is_exact_v13_with_one_uncapped_mse_change() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    loss = training["loss_contract"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["objective"] == "weighted_MSE_uncapped_residual"
    assert training["optimizer"] == "exact_v13_AdamW"
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert loss["extra_coefficient"] is None
    assert loss["residual_clipping"] is False
    assert loss["winsorization"] is False
    assert loss["row_downweighting_beyond_v13_domain_weights"] is False
    assert loss["row_deletion"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_weighted_mse_formula_and_gradient_are_exact() -> None:
    receipt = M._weighted_mse_contract_receipt()
    assert receipt["toy_raw_squared_residuals"] == [1.0, 1.0]
    assert receipt["toy_weighted_mse"] == 1.0
    assert np.allclose(receipt["toy_observed_gradient"], [-0.5, 1.5])
    assert receipt["formula_exact"] is True
    assert receipt["gradient_exact"] is True
    assert receipt["extra_coefficient"] is None
    assert receipt["residual_clipping"] is False
    assert receipt["winsorization"] is False
    assert receipt["row_deletion"] is False


def test_large_residual_is_not_clipped_or_winsorized() -> None:
    prediction = torch.tensor([0.0, 10.0], requires_grad=True)
    target = torch.zeros(2)
    weights = torch.ones(2)
    loss, raw = M.weighted_mse(prediction, target, weights)
    loss.backward()
    assert raw.detach().tolist() == [0.0, 100.0]
    assert float(loss.detach()) == 50.0
    assert prediction.grad is not None
    assert prediction.grad.tolist() == [0.0, 10.0]


def test_masked_future_permutation_and_gate_are_sealed() -> None:
    isolation = M._isolation_receipt()
    assert isolation["masked_token_maximum_abs_error"] <= 1e-6
    assert isolation["permutation_maximum_abs_error"] <= 1e-6
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003


def test_zero_operation_preflight_has_prefix_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    preflight = M.preflight()
    for fold, start in config["training"]["fold_starts_kst"].items():
        expected = M.pd.Timestamp(start) - M.pd.Timedelta(days=7)
        assert M.pd.Timestamp(preflight["prefix_cutoffs"][fold]) == expected
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


def test_tiny_training_has_finite_pure_mse_receipt() -> None:
    rng = np.random.default_rng(32)
    rows = 30
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows).astype(np.float32)
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
        32,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["loss"] == "weighted_MSE_uncapped_residual"
    assert receipt["extra_coefficient"] is None
    assert receipt["residual_clipping"] == 0
    assert receipt["winsorization"] == 0
    assert receipt["extra_row_downweighting"] == 0
    assert receipt["row_deletion"] == 0
