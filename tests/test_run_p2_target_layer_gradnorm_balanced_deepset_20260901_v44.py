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

import run_p2_target_layer_gradnorm_balanced_deepset_20260901_v44 as M  # noqa: E402


def test_config_is_exact_v13_plus_fixed_gradnorm_and_access_zero() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    gradnorm = training["gradnorm"]
    assert config["experiment_id"] == M.EXPERIMENT_ID
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert gradnorm["tasks"] == [2, 3, 4]
    assert gradnorm["initial_task_weights"] == [1.0, 1.0, 1.0]
    assert gradnorm["asymmetry_alpha"] == 1.5
    assert gradnorm["shared_parameter"] == "head.0.weight"
    assert gradnorm["task_weight_learning_rate"] == 0.025
    assert gradnorm["task_weight_minimum"] == 0.001
    assert gradnorm["task_weight_sum"] == 3.0
    assert not gradnorm["sampler_change"]
    assert not gradnorm["projection"]
    assert not gradnorm["coordinate_mask"]
    assert not gradnorm["sweep"]
    assert config["operation_limits"]["official_rows_read"] == 0
    assert config["operation_limits"]["hidden_rows_read"] == 0
    assert config["operation_limits"]["submission_csv_created"] == 0
    assert config["operation_limits"]["uploads"] == 0


def test_gradnorm_equal_fixed_point_and_unequal_update_contract() -> None:
    receipt = M._gradnorm_contract_receipt()
    assert receipt["tasks"] == [2, 3, 4]
    assert receipt["shared_parameter"] == "head.0.weight"
    assert receipt["alpha"] == 1.5
    assert receipt["task_weight_learning_rate"] == 0.025
    assert receipt["equal_fixed_point_maximum_abs_error"] <= 1e-7
    assert receipt["unequal_update_maximum_abs"] > 0.0
    assert abs(receipt["updated_weight_sum"] - 3.0) <= 1e-6
    assert receipt["updated_weight_minimum"] >= 0.001 - 1e-8
    assert not receipt["gradnorm_autograd_populated_model_or_weight_grad"]
    assert receipt["model_loss_task_weight_gradient_absent"]
    assert receipt["missing_task_update_exact_noop"]
    assert receipt["finite"]


def test_projected_sgd_positive_sum_three_and_deterministic() -> None:
    left = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    gradient = torch.tensor([100.0, -1.0, 2.0])
    M.apply_projected_task_weight_sgd_(left, gradient, 0.025, 0.001, 3.0)
    M.apply_projected_task_weight_sgd_(right, gradient, 0.025, 0.001, 3.0)
    assert torch.equal(left, right)
    assert float(left.detach().min()) >= 0.001 - 1e-8
    assert abs(float(left.detach().sum()) - 3.0) <= 1e-6


def test_exact_v13_architecture_permutation_and_future_isolation() -> None:
    receipt = M._architecture_isolation_receipt()
    assert receipt["model_class"] == "VerticalDeepSet"
    assert receipt["parameters"] == 4865
    assert receipt["parameter_tensors"] == 10
    assert receipt["state_identical_initial_function_maximum_abs_error"] == 0.0
    assert receipt["masked_or_future_token_maximum_abs_error"] <= 1e-6
    assert receipt["permutation_maximum_abs_error"] <= 1e-6
    assert receipt["extra_inference_parameters"] == 0


def test_target_free_preflight_is_byte_identical_and_namespace_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0
    assert first["hidden_rows_read"] == 0
    assert first["submission_csv_created"] == 0
    assert first["uploads"] == 0


def test_tiny_gradnorm_training_updates_weights_without_inference_change() -> None:
    rng = np.random.default_rng(44)
    rows = 18
    tokens = rng.normal(size=(rows, 5, 8)).astype(np.float32)
    tokens[:, :, 4:] = 1.0
    mask = np.ones((rows, 5), dtype=np.float32)
    context = rng.normal(size=(rows, 11)).astype(np.float32)
    context[:, 1:4] = np.eye(3, dtype=np.float32)[np.arange(rows) % 3]
    target = rng.normal(size=rows).astype(np.float32)
    weights = np.linspace(0.5, 1.5, rows, dtype=np.float32)
    M._ACTIVE_LAYER_IDS = np.array([2, 3, 4] * 6, dtype=np.int64)
    config = copy.deepcopy(json.loads(M.CONFIG.read_text(encoding="utf-8")))
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 9
    prediction, receipt = M.train_predict_seed(
        tokens,
        mask,
        context,
        target,
        weights,
        tokens[:4],
        mask[:4],
        context[:4],
        config,
        20260901,
    )
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert receipt["optimizer_steps"] == 2
    assert receipt["task_weight_updates"] == 2
    assert receipt["missing_task_batches"] == 0
    assert receipt["parameters"] == 4865
    assert receipt["shared_parameter"] == "head.0.weight"
    assert receipt["shared_parameter_shape"] == [32, 75]
    assert abs(receipt["task_weight_sum"] - 3.0) <= 1e-6
    assert min(receipt["final_task_weights"]) >= 0.001 - 1e-8
    assert receipt["initial_task_weight_sha256"] != receipt["final_task_weight_sha256"]
    assert receipt["gradient_projection"] == 0
    assert receipt["coordinate_mask"] == 0
    assert receipt["extra_inference_parameters"] == 0
    assert receipt["loss_finite"]


def test_prospective_gate_is_unchanged() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003
    assert (
        config["authorization_evidence"]["prospective_gate_amendment_sha256"]
        == "c7fde8c5d9f535ab8080eb561bf082c55e5c7172117d00e12e1479f9b4417680"
    )
