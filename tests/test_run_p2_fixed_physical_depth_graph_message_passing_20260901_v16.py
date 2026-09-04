from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_fixed_physical_depth_graph_message_passing_20260901_v16.py"
SPEC = importlib.util.spec_from_file_location("p2_v16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    tokens = torch.randn(8, 5, 8, generator=generator)
    tokens[:, :, 3] = torch.tensor([-0.40, -0.20, 0.0, 0.20, 0.50])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 0, 1]], dtype=torch.float32)
    context = torch.randn(8, 11, generator=generator)
    return tokens, mask, context


def test_fixed_contract() -> None:
    config = MODULE.load_config()
    assert config["representation"]["message_blocks"] == 1
    assert config["representation"]["bandwidth_normalized_depth"] == 0.20
    assert config["representation"]["learned_adjacency"] is False
    assert config["training"]["maximum_fit_count"] == 9
    assert config["training"]["embargo_days"] == 7
    assert config["training"]["champion_preserving_weight"] == 0.8
    assert config["training"]["model_weight"] == 0.2


def test_adjacency_is_fixed_normalized_and_has_no_self_edges() -> None:
    tokens, mask, _ = inputs()
    adjacency = MODULE.fixed_depth_adjacency(tokens, mask, 0.20)
    valid_sums = adjacency.sum(dim=2)[mask.bool()]
    torch.testing.assert_close(valid_sums, torch.ones_like(valid_sums), atol=1e-6, rtol=0)
    assert torch.max(torch.abs(torch.diagonal(adjacency, dim1=1, dim2=2))) == 0
    assert adjacency[2, 1, 2] > adjacency[2, 1, 4]


def test_graph_encoder_equivariance_and_prediction_invariance() -> None:
    torch.manual_seed(9)
    model = MODULE.FixedDepthGraphEncoder(8, 11, hidden=32, blocks=1, bandwidth_normalized_depth=0.20).eval()
    tokens, mask, context = inputs()
    order = torch.tensor([4, 1, 3, 0, 2])
    with torch.inference_mode():
        encoded = model.encode(tokens, mask)
        encoded_permuted = model.encode(tokens[:, order], mask[:, order])
        prediction = model(tokens, mask, context)
        prediction_permuted = model(tokens[:, order], mask[:, order], context)
    torch.testing.assert_close(encoded_permuted, encoded[:, order], atol=1e-6, rtol=0)
    torch.testing.assert_close(prediction_permuted, prediction, atol=1e-6, rtol=0)


def test_future_rows_cannot_change_prior_predictions() -> None:
    torch.manual_seed(9)
    model = MODULE.FixedDepthGraphEncoder(8, 11, hidden=32, blocks=1, bandwidth_normalized_depth=0.20).eval()
    tokens, mask, context = inputs()
    changed_tokens = tokens.clone()
    changed_context = context.clone()
    changed_tokens[4:] += 10000
    changed_context[4:] -= 10000
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(changed_tokens, mask, changed_context)
    torch.testing.assert_close(right[:4], left[:4], atol=0, rtol=0)


def test_no_target_features_no_temporal_axis_and_half_degree_bound() -> None:
    config = MODULE.load_config()
    representation = config["representation"]
    assert representation["public_layers"] == [1, 5, 6, 7, 8]
    assert representation["target_layer_values_used_as_features"] is False
    assert representation["temporal_axis"] is False
    assert config["training"]["model_minus_champion_clip_C"] * config["training"]["model_weight"] == 0.5


def test_preflight_byte_identical_and_zero_operation() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["semantic_audit"]["spline_axis_closed_as_duplicate"]
    receipt = first["architecture_contract"]
    assert receipt["encoder_equivariance_maximum_abs_error"] <= 1e-6
    assert receipt["prediction_invariance_maximum_abs_error"] <= 1e-6
    assert receipt["future_batch_perturbation_maximum_abs_error_on_prior_rows"] == 0
