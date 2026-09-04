from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_regularized_layer_month_group_dro_20260901_v18.py"
SPEC = importlib.util.spec_from_file_location("p2_v18", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_group_dro_contract_has_no_sweep_or_router() -> None:
    config = MODULE.load_config()
    training = config["training"]
    assert training["group_definition"] == "target_layer_x_calendar_month"
    assert training["group_dro_eta"] == 0.1
    assert training["weight_decay"] == 0.001
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["maximum_fit_count"] == 9
    assert training["model_weight"] == 0.2
    assert training["row_deletion"] is False
    assert config["result_adaptive_tuning"] is False


def test_group_contract_equalizes_layer_month_mass_and_kst_days() -> None:
    layers = np.array([2, 2, 2, 3, 3, 3])
    times = pd.DatetimeIndex(
        [
            "2024-05-01T00:00:00+09:00",
            "2024-05-01T01:00:00+09:00",
            "2024-05-02T00:00:00+09:00",
            "2024-05-01T00:00:00+09:00",
            "2024-05-02T00:00:00+09:00",
            "2024-05-02T01:00:00+09:00",
        ]
    )
    groups, weights, labels, receipt = MODULE.build_group_contract(layers, times)
    assert labels == ["layer2:month05", "layer3:month05"]
    assert set(groups.tolist()) == {0, 1}
    np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-7)
    group_mass = [weights[groups == index].sum() for index in range(2)]
    np.testing.assert_allclose(group_mass, [3.0, 3.0], atol=1e-6)
    raw_mass = [item["raw_weight_sum"] for item in receipt["groups"].values()]
    np.testing.assert_allclose(raw_mass, [0.5, 0.5], atol=1e-12)


def test_exponentiated_update_upweights_the_worst_group() -> None:
    initial = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    losses = torch.tensor([0.1, 0.4, 0.2], dtype=torch.float64)
    updated = MODULE.exponentiated_group_update(initial, losses, 0.1)
    assert updated[1] > updated[2] > updated[0]
    torch.testing.assert_close(updated.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(
        MODULE.exponentiated_group_update(initial, torch.ones(3), 0.1),
        initial,
    )


def test_v13_encoder_remains_permutation_invariant() -> None:
    receipt = MODULE.v12.permutation_invariance_receipt()
    assert receipt["maximum_abs_error"] <= 1e-6
    config = MODULE.load_config()
    assert config["training"]["token_features"] == 8
    assert config["training"]["context_features"] == 11


def test_future_batch_rows_cannot_change_prior_predictions() -> None:
    torch.manual_seed(31)
    model = MODULE.v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(8, 5, 8)
    mask = torch.ones(8, 5)
    context = torch.randn(8, 11)
    changed_tokens = tokens.clone()
    changed_context = context.clone()
    changed_tokens[4:] += 10000
    changed_context[4:] -= 10000
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(changed_tokens, mask, changed_context)
    torch.testing.assert_close(right[:4], left[:4], atol=0, rtol=0)


def test_preflight_is_byte_identical_zero_operation_and_new_for_p2() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["maximum_fit_count"] == 9
    assert first["model_fits"] == 0
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0
    assert first["hidden_rows_read"] == 0
    audit = first["semantic_audit"]
    assert audit["classification"] == "NEW_P2_REGULARIZED_LAYER_MONTH_GROUP_DRO_OBJECTIVE"
    assert audit["prior_p2_group_dro_runners"] == []
    assert audit["prior_p2_group_dro_artifacts"] == []
    assert audit["prior_p2_group_dro_reports"] == []
    assert audit["v13_static_domain_balance_only"] is True
    assert audit["p1_group_dro_is_cross_problem_only"] is True
