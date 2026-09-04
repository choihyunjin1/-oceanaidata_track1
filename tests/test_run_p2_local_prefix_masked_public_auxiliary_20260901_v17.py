from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_local_prefix_masked_public_auxiliary_20260901_v17.py"
SPEC = importlib.util.spec_from_file_location("p2_v17", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    tokens = torch.randn(8, 5, 8, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 0, 1]], dtype=torch.float32)
    context = torch.randn(8, 11, generator=generator)
    return tokens, mask, context


def test_fixed_contract_and_no_external_data() -> None:
    config = MODULE.load_config()
    assert config["auxiliary"]["weight"] == 0.25
    assert config["auxiliary"]["mask_schedule"] == "epoch_modulo_five_fixed_cycle"
    assert config["representation"]["external_data"] is False
    assert config["representation"]["temporal_axis"] is False
    assert config["training"]["maximum_fit_count"] == 9
    assert config["training"]["embargo_days"] == 7


def test_mask_public_index_masks_exactly_one_slot_without_deleting_rows() -> None:
    tokens, mask, _ = inputs()
    masked_tokens, masked_mask, eligible = MODULE.mask_public_index(tokens, mask, 2)
    assert torch.count_nonzero(masked_tokens[:, 2]) == 0
    assert torch.count_nonzero(masked_mask[:, 2]) == 0
    torch.testing.assert_close(masked_tokens[:, :2], tokens[:, :2])
    torch.testing.assert_close(masked_tokens[:, 3:], tokens[:, 3:])
    assert eligible.dtype == torch.bool
    assert len(eligible) == len(tokens)


def test_target_and_reconstruction_shapes_are_fixed() -> None:
    torch.manual_seed(17)
    model = MODULE.MaskedPublicAuxiliaryEncoder(8, 5, 11, hidden=64, latent=32).eval()
    tokens, mask, context = inputs()
    with torch.inference_mode():
        target = model(tokens, mask, context)
        reconstruction = model.reconstruct(tokens, mask, context)
    assert target.shape == (8,)
    assert reconstruction.shape == (8, 5)


def test_future_rows_cannot_change_prior_predictions() -> None:
    torch.manual_seed(17)
    model = MODULE.MaskedPublicAuxiliaryEncoder(8, 5, 11, hidden=64, latent=32).eval()
    tokens, mask, context = inputs()
    changed_tokens = tokens.clone()
    changed_context = context.clone()
    changed_tokens[4:] += 10000
    changed_context[4:] -= 10000
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(changed_tokens, mask, changed_context)
    torch.testing.assert_close(right[:4], left[:4], atol=0, rtol=0)


def test_target_features_forbidden_and_action_bound_fixed() -> None:
    config = MODULE.load_config()
    assert config["representation"]["target_layer_values_used_as_features"] is False
    assert config["representation"]["public_layers"] == [1, 5, 6, 7, 8]
    assert config["training"]["model_minus_champion_clip_C"] * config["training"]["model_weight"] == 0.5
    assert config["training"]["row_deletion"] is False


def test_preflight_byte_identical_zero_operation_and_no_overlap() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["external_rows_read"] == 0
    assert first["official_rows_read"] == 0
    audit = first["semantic_audit"]
    assert set(audit["external_depth_query_execution"].values()) == {False}
    assert audit["external_depth_query_artifact_exists"] is False
    assert audit["target_masked_tcn_or_csdi_overlap"] is False
