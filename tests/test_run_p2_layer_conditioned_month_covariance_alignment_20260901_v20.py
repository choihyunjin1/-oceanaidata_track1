from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_p2_layer_conditioned_month_covariance_alignment_20260901_v20.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("p2_v20_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_one_fixed_nine_fit_contract() -> None:
    runner = load_runner()
    config = runner.load_config()
    training = config["training"]
    assert training["coral_coefficient"] == 1.0
    assert training["weight_decay"] == 0.0001
    assert training["epochs"] == 60
    assert training["seeds"] == [20260901, 20260902, 20260903]
    assert training["maximum_fit_count"] == 9
    assert not training["row_deletion"]
    assert not config["result_adaptive_tuning"]


def test_latent_model_preserves_v13_parameterization_and_permutation() -> None:
    runner = load_runner()
    torch.manual_seed(23)
    model = runner.LatentVerticalDeepSet(8, 11, hidden=32).eval()
    baseline = runner.v12.VerticalDeepSet(8, 11, hidden=32).eval()
    baseline.load_state_dict(model.state_dict())
    tokens = torch.randn(7, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0]] * 7, dtype=torch.float32)
    context = torch.randn(7, 11)
    prediction, latent = model.latent_and_prediction(tokens, mask, context)
    assert torch.equal(prediction, baseline(tokens, mask, context))
    assert latent.shape == (7, 32)
    order = torch.tensor([4, 0, 3, 1, 2])
    permuted, _ = model.latent_and_prediction(tokens[:, order], mask[:, order], context)
    assert torch.allclose(prediction, permuted, atol=1e-7, rtol=0.0)


def test_covariance_uses_unbiased_centered_second_moment() -> None:
    runner = load_runner()
    values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 8.0]])
    expected = torch.cov(values.T, correction=1)
    assert torch.allclose(runner.covariance_matrix(values), expected)


def test_coral_pairs_months_only_within_each_target_layer() -> None:
    runner = load_runner()
    receipt = runner.synthetic_coral_receipt()
    assert receipt["within_layer_pair_count"] == 2
    assert receipt["cross_layer_pairs"] == 0
    assert receipt["minimum_rows_per_covariance"] == 2
    assert receipt["loss"] >= 0.0


def test_coral_is_row_permutation_invariant() -> None:
    runner = load_runner()
    torch.manual_seed(17)
    latent = torch.randn(24, 6)
    environment = torch.arange(6).repeat_interleave(4)
    groups = {"layer2": [0, 1], "layer3": [2, 3], "layer4": [4, 5]}
    original, pairs, minimum = runner.layer_conditioned_month_coral(
        latent, environment, groups
    )
    order = torch.randperm(len(latent))
    permuted, permuted_pairs, permuted_minimum = runner.layer_conditioned_month_coral(
        latent[order], environment[order], groups
    )
    assert torch.allclose(original, permuted, atol=1e-7, rtol=0.0)
    assert (pairs, minimum) == (permuted_pairs, permuted_minimum) == (3, 4)


def test_preflight_is_zero_operation_and_semantically_distinct() -> None:
    runner = load_runner()
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == first["data_rows_read"] == 0
    assert first["official_rows_read"] == first["hidden_rows_read"] == 0
    assert first["submission_csv_created"] == first["uploads"] == 0
    semantic = first["semantic_audit"]
    assert semantic["prior_p2_coral_runners"] == []
    assert semantic["prior_p2_coral_artifacts"] == []
    assert semantic["prior_p2_coral_reports"] == []
    assert semantic["alignment_is_within_target_layer_only"]
