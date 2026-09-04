from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p2_fixed_student_t_robust_deepset_20260901_v21.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("p2_v21_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_one_fixed_nine_fit_contract() -> None:
    runner = load_runner()
    config = runner.load_config()
    training = config["training"]
    assert training["student_t_degrees_of_freedom"] == 4.0
    assert training["student_t_normalized_scale"] == 1.0
    assert training["weight_decay"] == 0.0001
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert not training["row_deletion"]
    assert not config["result_adaptive_tuning"]


def test_student_t_loss_matches_registered_closed_form() -> None:
    runner = load_runner()
    prediction = torch.tensor([-2.0, -0.5, 0.0, 1.5], dtype=torch.float64)
    target = torch.tensor([0.0, 0.5, 0.0, -0.5], dtype=torch.float64)
    residual = prediction - target
    expected = 2.5 * torch.log1p(residual.square() / 4.0)
    actual = runner.student_t_location_loss(prediction, target, 4.0, 1.0)
    assert torch.equal(actual, expected)


def test_student_t_extreme_residual_influence_redescends_without_deletion() -> None:
    runner = load_runner()
    residual = torch.tensor([1.0, 2.0, 10.0, 100.0], dtype=torch.float64)
    influence = runner.student_t_absolute_influence(residual, 4.0, 1.0)
    assert influence[0] == 1.0
    assert influence[-1] < influence[-2] < influence[0]
    receipt = runner.synthetic_student_t_receipt(runner.load_config())
    assert receipt["row_deletion"] == 0
    assert receipt["extreme_influence_below_unit_residual_influence"]


def test_model_parameterization_and_public_layer_permutation_match_v13() -> None:
    runner = load_runner()
    torch.manual_seed(29)
    model = runner.v12.VerticalDeepSet(8, 11, hidden=32).eval()
    assert sum(value.numel() for value in model.parameters()) == 4865
    tokens = torch.randn(6, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0]] * 6, dtype=torch.float32)
    context = torch.randn(6, 11)
    order = torch.tensor([4, 2, 0, 3, 1])
    original = model(tokens, mask, context)
    permuted = model(tokens[:, order], mask[:, order], context)
    assert torch.allclose(original, permuted, atol=1e-7, rtol=0.0)


def test_loss_and_weighted_reduction_are_row_permutation_invariant() -> None:
    runner = load_runner()
    torch.manual_seed(31)
    prediction = torch.randn(23)
    target = torch.randn(23)
    weight = torch.rand(23) + 0.1
    raw = runner.student_t_location_loss(prediction, target, 4.0, 1.0)
    original = (raw * weight).sum() / weight.sum()
    order = torch.randperm(23)
    permuted_raw = runner.student_t_location_loss(
        prediction[order], target[order], 4.0, 1.0
    )
    permuted = (permuted_raw * weight[order]).sum() / weight[order].sum()
    assert torch.allclose(original, permuted, atol=1e-7, rtol=0.0)


def test_preflight_is_byte_stable_zero_operation_and_new_axis() -> None:
    runner = load_runner()
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == first["data_rows_read"] == 0
    assert first["official_rows_read"] == first["hidden_rows_read"] == 0
    assert first["submission_csv_created"] == first["uploads"] == 0
    semantic = first["semantic_audit"]
    assert semantic["prior_p2_student_t_runners"] == []
    assert semantic["prior_p2_student_t_artifacts"] == []
    assert semantic["prior_p2_student_t_reports"] == []
    assert semantic["v10_zero_fit_public_difference_action_shrink_distinguished"]
