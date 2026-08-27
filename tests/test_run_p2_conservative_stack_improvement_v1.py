from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _runner():
    path = Path("scripts/run_p2_conservative_stack_improvement_v1.py").resolve()
    spec = importlib.util.spec_from_file_location("p2_stack_improvement_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(module):
    return json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _metric(rmse: float, folds: dict[str, float]):
    return {
        "fold_equal_official_layer_weighted_rmse_c": rmse,
        "by_fold": {
            name: {"official_layer_weighted_rmse_c": value} for name, value in folds.items()
        },
    }


def test_canonical_config_and_three_branch_budget() -> None:
    module = _runner()
    config = _config(module)
    assert (
        module._canonical_preflight(config, module.DEFAULT_CONFIG, module.DEFAULT_OUTPUT) == config
    )
    assert config["stack"]["branches"] == module.EXPECTED_BRANCHES


def test_direct_call_rejects_mutated_branch_before_any_fit() -> None:
    module = _runner()
    config = copy.deepcopy(_config(module))
    config["stack"]["branches"][1]["candidate_weight"] = 0.6
    with pytest.raises(ValueError, match="differs from the canonical"):
        module._dry_run(
            config,
            module.DEFAULT_CONFIG,
            Path("does-not-exist"),
            module.DEFAULT_OUTPUT,
        )


def test_stack_prediction_is_exact_convex_combination() -> None:
    module = _runner()
    baseline = np.array([1.0, 2.0, 3.0])
    candidate = np.array([3.0, 0.0, 7.0])
    actual = module.stack_prediction(baseline, candidate, 0.625)
    np.testing.assert_allclose(actual, np.array([2.25, 0.75, 5.5]), rtol=0, atol=0)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_stack_prediction_rejects_extrapolation(weight: float) -> None:
    module = _runner()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        module.stack_prediction(np.array([1.0]), np.array([2.0]), weight)


def test_guard_requires_two_fold_improvements_and_reduced_directionality_gap() -> None:
    module = _runner()
    previous = _metric(1.1, {"a": 0.7, "b": 1.3, "c": 1.2})
    outer = _metric(1.0, {"a": 0.75, "b": 1.1, "c": 1.1})
    inner = _metric(1.25, {"a": 1.2, "b": 1.2, "c": 1.3})
    report = module.branch_guard_report(
        branch_id="test",
        outer_report=outer,
        inner_report=inner,
        outer_baseline_rmse=1.2,
        inner_baseline_rmse=1.2,
        predecessor_outer=previous,
        predecessor_inner_rmse=1.4,
        guards={
            "minimum_outer_fold_improvements_vs_predecessor": 2,
            "maximum_outer_fold_regression_vs_predecessor_c": 0.15,
            "inner_outer_directionality_gap_strictly_below_predecessor_c": 0.4,
        },
    )
    assert report["eligible"] is True
    assert report["outer_fold_improvement_count"] == 2


def test_winner_selection_is_metric_then_branch_order() -> None:
    module = _runner()
    branches = [
        {"id": "a", "guard": {"eligible": True, "outer_primary_rmse_c": 1.0}},
        {"id": "b", "guard": {"eligible": True, "outer_primary_rmse_c": 0.9}},
        {"id": "c", "guard": {"eligible": False, "outer_primary_rmse_c": 0.8}},
    ]
    assert module.select_winner(branches)["id"] == "b"


def test_planned_paths_are_below_new_generation() -> None:
    module = _runner()
    paths = module._planned_paths(_config(module))
    for path in paths.values():
        path.relative_to(module.DEFAULT_OUTPUT)
    assert len(paths) == len(set(paths.values()))
