from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_layer_month_risk_variance_rex_20260901_v19.py"
SPEC = importlib.util.spec_from_file_location("p2_v19", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_vrex_contract_has_no_sweep_or_router() -> None:
    config = MODULE.load_config()
    training = config["training"]
    assert training["risk_variance_coefficient"] == 10.0
    assert training["weight_decay"] == 0.001
    assert training["environment_definition"] == "target_layer_x_calendar_month"
    assert training["architecture"].startswith("v13_exact_DeepSets")
    assert training["maximum_fit_count"] == 9
    assert training["model_weight"] == 0.2
    assert training["row_deletion"] is False
    assert config["result_adaptive_tuning"] is False


def test_population_risk_variance_objective_is_exact() -> None:
    losses = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
    weights = torch.ones(3, dtype=torch.float64)
    groups = torch.arange(3)
    objective, risks, present = MODULE.risk_variance_objective(
        losses,
        weights,
        groups,
        3,
        10.0,
    )
    torch.testing.assert_close(risks, losses)
    torch.testing.assert_close(
        objective,
        losses.mean() + 10.0 * losses.var(unbiased=False),
    )
    assert present.tolist() == [True, True, True]


def test_equal_environment_risks_reduce_to_plain_mean() -> None:
    losses = torch.tensor([0.2, 0.2, 0.2, 0.2])
    weights = torch.ones(4)
    groups = torch.tensor([0, 0, 1, 1])
    objective, risks, _ = MODULE.risk_variance_objective(
        losses,
        weights,
        groups,
        2,
        10.0,
    )
    torch.testing.assert_close(risks, torch.tensor([0.2, 0.2]))
    torch.testing.assert_close(objective, torch.tensor(0.2))


def test_environment_contract_equalizes_layer_month_mass() -> None:
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
    groups, weights, labels, receipt = MODULE.v18.build_group_contract(layers, times)
    assert labels == ["layer2:month05", "layer3:month05"]
    np.testing.assert_allclose(
        [weights[groups == index].sum() for index in range(2)],
        [3.0, 3.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [item["raw_weight_sum"] for item in receipt["groups"].values()],
        [0.5, 0.5],
        atol=1e-12,
    )


def test_v13_encoder_is_permutation_invariant_and_row_local() -> None:
    receipt = MODULE.v12.permutation_invariance_receipt()
    assert receipt["maximum_abs_error"] <= 1e-6
    torch.manual_seed(53)
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
    assert first["fixed_risk_variance_coefficient"] == 10.0
    assert first["synthetic_population_variance_exact"] is True
    assert first["model_fits"] == 0
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0
    audit = first["semantic_audit"]
    assert audit["classification"] == "NEW_P2_LAYER_MONTH_RISK_VARIANCE_REX_OBJECTIVE"
    assert audit["prior_p2_vrex_runners"] == []
    assert audit["prior_p2_vrex_artifacts"] == []
    assert audit["prior_p2_vrex_reports"] == []
    assert audit["v18_minimax_adversary_not_risk_variance"] is True
