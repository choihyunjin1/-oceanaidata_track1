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

import run_p2_gradient_centralized_domain_balanced_deepset_20260901_v34 as M  # noqa: E402


def test_config_is_exact_v13_with_one_gradient_centralization_change() -> None:
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    central = training["gradient_centralization"]
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["objective"] == "weighted_SmoothL1_beta_1.0_with_fixed_gradient_centralization"
    assert training["optimizer"] == "exact_v13_AdamW_after_gradient_centralization"
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0001
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["epochs"] == 60
    assert training["maximum_fit_count"] == 9
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert training["maximum_final_action_C"] == 0.5
    assert central["coefficient"] is None
    assert central["bias_or_1d_change"] is False
    assert central["second_loss"] is False
    assert central["task_split"] is False
    assert central["parameter_reparameterization"] is False
    assert central["slow_weights"] is False
    assert central["sweep"] is False
    for name in (
        "automatic_retry_count",
        "official_rows_read",
        "hidden_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert config["operation_limits"][name] == 0


def test_gradient_centralization_formula_is_exact_and_bias_is_untouched() -> None:
    receipt = M._gradient_centralization_contract_receipt()
    assert receipt["centralized_gradient"] == [[-1.0, 1.0], [-2.0, 2.0]]
    assert receipt["formula_exact"] is True
    assert receipt["row_means_zero"] is True
    assert receipt["bias_unchanged"] is True
    assert receipt["eligible_count"] == 1
    assert receipt["coefficient"] is None


def test_only_linear_multidimensional_weight_gradients_change() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1),
    )
    for parameter in model.parameters():
        parameter.grad = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter) + 1
    bias_before = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if name.endswith("bias")
    }
    receipt = M.centralize_linear_weight_gradients(model)
    assert receipt["eligible_names"] == ["0.weight", "2.weight"]
    assert receipt["eligible_count"] == 2
    assert receipt["maximum_abs_mean_after"] <= 1e-6
    for name, parameter in model.named_parameters():
        if name.endswith("weight"):
            assert torch.allclose(
                parameter.grad.mean(dim=tuple(range(1, parameter.grad.ndim))),
                torch.zeros(parameter.shape[0]),
                atol=1e-6,
                rtol=0.0,
            )
        else:
            assert torch.equal(parameter.grad, bias_before[name])


def test_masked_future_permutation_and_prospective_gate_are_sealed() -> None:
    isolation = M._isolation_receipt()
    assert isolation["masked_token_maximum_abs_error"] <= 1e-6
    assert isolation["permutation_maximum_abs_error"] <= 1e-6
    config = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    gate = config["evaluation"]["safety_gate"]
    assert gate["minimum_fold_layer_non_harm_cells"] == 8
    assert gate["total_fold_layer_cells"] == 9
    assert gate["maximum_any_fold_layer_delta_rmse_C"] == 0.003


def test_two_zero_operation_preflights_are_byte_identical_and_access_zero() -> None:
    first = M.preflight()
    second = M.preflight()
    assert M.v12.sha256_json(first) == M.v12.sha256_json(second)
    assert first["preflight_sha256"] == second["preflight_sha256"]
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


def test_tiny_training_has_finite_gradient_centralization_receipt() -> None:
    rng = np.random.default_rng(34)
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
        34,
    )
    assert prediction.shape == (7,)
    assert np.isfinite(prediction).all()
    assert receipt["loss_finite"] is True
    assert receipt["epochs"] == 2
    assert receipt["optimizer_steps"] == 2
    assert receipt["centralization_steps"] == 2
    assert receipt["eligible_count"] == 5
    assert receipt["maximum_abs_mean_after"] <= 1e-6
    assert receipt["coefficient"] is None
    assert receipt["second_loss"] == 0
    assert receipt["task_split"] == 0
    assert receipt["bias_or_1d_change"] == 0
    assert receipt["row_deletion"] == 0
