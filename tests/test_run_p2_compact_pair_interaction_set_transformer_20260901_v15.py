from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "run_p2_compact_pair_interaction_set_transformer_20260901_v15.py"
)
SPEC = importlib.util.spec_from_file_location("p2_v15", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    tokens = torch.randn(8, 5, 8, generator=generator)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    context = torch.randn(8, 11, generator=generator)
    return tokens, mask, context


def test_contract_is_one_candidate_nine_fits_and_fixed_architecture() -> None:
    config = MODULE.load_config()
    assert config["operation_limits"]["maximum_candidate_count"] == 1
    assert config["training"]["maximum_fit_count"] == 9
    assert config["representation"]["attention_blocks"] == 1
    assert config["representation"]["attention_heads"] == 2
    assert config["representation"]["positional_encoding"] is False
    assert config["representation"]["temporal_axis"] is False
    assert config["training"]["row_deletion"] is False


def test_encoder_is_public_layer_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = MODULE.CompactSetTransformer(8, 11, hidden=32, heads=2, blocks=1).eval()
    tokens, mask, _ = make_inputs()
    order = torch.tensor([4, 1, 3, 0, 2])
    with torch.inference_mode():
        left = model.encode(tokens, mask)
        right = model.encode(tokens[:, order], mask[:, order])
    torch.testing.assert_close(right, left[:, order], atol=1e-6, rtol=0)


def test_prediction_is_public_layer_permutation_invariant() -> None:
    torch.manual_seed(5)
    model = MODULE.CompactSetTransformer(8, 11, hidden=32, heads=2, blocks=1).eval()
    tokens, mask, context = make_inputs()
    order = torch.tensor([3, 0, 4, 2, 1])
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(tokens[:, order], mask[:, order], context)
    torch.testing.assert_close(right, left, atol=1e-6, rtol=0)


def test_future_rows_cannot_change_prior_row_predictions() -> None:
    """Batch rows are timestamps; the architecture never attends over that axis."""
    torch.manual_seed(5)
    model = MODULE.CompactSetTransformer(8, 11, hidden=32, heads=2, blocks=1).eval()
    tokens, mask, context = make_inputs()
    changed_tokens = tokens.clone()
    changed_context = context.clone()
    changed_tokens[4:] += 10_000.0
    changed_context[4:] -= 10_000.0
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(changed_tokens, mask, changed_context)
    torch.testing.assert_close(right[:4], left[:4], atol=0, rtol=0)


def test_current_target_values_are_forbidden_and_only_public_tokens_are_used() -> None:
    config = MODULE.load_config()
    representation = config["representation"]
    assert representation["public_layers"] == [1, 5, 6, 7, 8]
    assert representation["target_layer_values_used_as_features"] is False
    assert representation["base"] == "v12_public_layer_tokens"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "design.normalized_target[train_mask]" in source
    assert "tokens[positions[query_mask]]" in source
    assert "temporal_attention" not in source


def test_prefix_purge_blend_and_action_bound_are_fixed() -> None:
    config = MODULE.load_config()
    training = config["training"]
    assert training["embargo_days"] == 7
    assert training["weighting"].startswith("equal_total_mass_per_target_layer_x_calendar_month")
    assert training["champion_preserving_weight"] == 0.8
    assert training["model_weight"] == 0.2
    assert training["model_minus_champion_clip_C"] == 2.5
    assert 0.2 * 2.5 == 0.5


def test_preflight_is_byte_identical_and_zero_operation() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    contract = first["architecture_contract"]
    assert contract["attention_blocks"] == 1
    assert contract["attention_heads"] == 2
    assert contract["encoder_equivariance_maximum_abs_error"] <= 1e-6
    assert contract["prediction_invariance_maximum_abs_error"] <= 1e-6
    assert contract["future_batch_perturbation_maximum_abs_error_on_prior_rows"] == 0.0
