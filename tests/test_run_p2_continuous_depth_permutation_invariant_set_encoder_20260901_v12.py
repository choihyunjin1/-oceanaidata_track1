from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12.py"
SPEC = importlib.util.spec_from_file_location("p2_v12", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preregistered_contract_is_single_frozen_candidate() -> None:
    config = MODULE.load_config()
    assert config["operation_limits"]["maximum_candidate_count"] == 1
    assert config["training"]["champion_preserving_weight"] == 0.8
    assert config["training"]["model_weight"] == 0.2
    assert config["training"]["row_deletion"] is False
    assert config["training"]["outer_fold_tuning"] is False
    assert len(config["training"]["seeds"]) == 3


def test_zero_operation_preflight_is_stable() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["preflight_sha256"] == second["preflight_sha256"]


def test_set_encoder_is_permutation_invariant() -> None:
    torch.manual_seed(19)
    model = MODULE.VerticalDeepSet(8, 11, 32).eval()
    tokens = torch.randn(7, 5, 8)
    mask = (torch.rand(7, 5) > 0.25).float()
    context = torch.randn(7, 11)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    with torch.inference_mode():
        expected = model(tokens, mask, context)
        observed = model(tokens[:, permutation], mask[:, permutation], context)
    torch.testing.assert_close(expected, observed, atol=1e-6, rtol=0.0)


def test_masked_token_content_cannot_change_output() -> None:
    torch.manual_seed(23)
    model = MODULE.VerticalDeepSet(8, 11, 32).eval()
    tokens = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    mask[:, -1] = 0.0
    changed = tokens.clone()
    changed[:, -1] = 1e6
    context = torch.randn(3, 11)
    with torch.inference_mode():
        expected = model(tokens, mask, context)
        observed = model(changed, mask, context)
    torch.testing.assert_close(expected, observed, atol=1e-6, rtol=0.0)


def test_context_width_matches_preregistration() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert len(config["training"]["context_features"]) == 11
    assert len(config["training"]["token_features"]) == 8


def test_action_geometry_is_finite_and_bounded() -> None:
    truth = np.asarray([0.0, 1.0, 2.0])
    reference = np.asarray([0.5, 0.5, 2.5])
    candidate = np.asarray([0.4, 0.6, 2.4])
    receipt = MODULE.action_geometry(truth, reference, candidate)
    assert receipt["active_rows_gt_1e_12"] == 3
    assert 0.0 <= receipt["active_share_gt_1e_12"] <= 1.0
    assert receipt["abs_action_max_C"] <= 0.1 + 1e-12
