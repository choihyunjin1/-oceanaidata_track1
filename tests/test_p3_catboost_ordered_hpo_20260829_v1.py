from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.catboost_ordered_hpo import (
    CONTROL_ID,
    apply_frozen_kma_alpha,
    control_candidate,
    evaluate_confirmation_gate,
    evaluate_selection_gate,
    materialize_grid,
    metric_deltas,
    paired_case_bootstrap,
    rank_candidates,
    validate_schedule,
    validate_windows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.json"
GRID_PATH = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.grid.json"
RUNNER_PATH = ROOT / "scripts/run_p3_catboost_ordered_hpo_20260829_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_catboost_ordered_hpo_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_grid_is_exact_cartesian_48_plus_control_and_budget_174() -> None:
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    challengers = materialize_grid(grid)
    assert len(challengers) == 48
    assert len({json.dumps(row["parameters"], sort_keys=True) for row in challengers}) == 48
    assert control_candidate(grid)["candidate_id"] == CONTROL_ID
    config = _config()
    assert validate_schedule(config) == 174
    validate_windows(config)


def test_rank_tolerance_prefers_control_then_simpler_regularized_candidate() -> None:
    aggregate = pd.DataFrame(
        {
            "candidate_id": [CONTROL_ID, "challenger_01", "challenger_02"],
            "squared_error_sum": [100.0, 99.99, 98.0],
            "row_count": [100, 100, 100],
        }
    )
    parameters = {
        CONTROL_ID: {"depth": 6, "l2_leaf_reg": 8.0},
        "challenger_01": {"depth": 7, "l2_leaf_reg": 20.0},
        "challenger_02": {"depth": 5, "l2_leaf_reg": 8.0},
    }
    ranked = rank_candidates(aggregate, parameters, tie_tolerance_rmse_m=0.0005)
    # challenger_02 is more than 0.0005m better, so it must precede the conservative tie set.
    assert ranked.iloc[0]["candidate_id"] == "challenger_02"


def test_kma_alpha_is_exact_short_noop_and_fixed_long_blend() -> None:
    base = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    source = base + 1.0
    leads = np.asarray([3, 6, 9, 12, 18, 24])
    result = apply_frozen_kma_alpha(base, source, leads)
    assert np.array_equal(result[:4], base[:4])
    assert np.allclose(result[4:], np.asarray([5.4, 6.4]))


def _metric_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in range(9):
        for lead in [3, 6, 9, 12, 18, 24]:
            target = 2.0
            control = 2.1
            challenger = control if lead in [3, 6, 9, 12] else 2.0
            rows.append(
                {
                    "fold": f"fold_{case % 3}",
                    "anchor_id": case,
                    "station": f"station_{case % 3}",
                    "lead_h": lead,
                    "target_hs": target,
                    "control_prediction": control,
                    "challenger_prediction": challenger,
                }
            )
    return pd.DataFrame(rows)


def test_all_selection_and_confirmation_gates_are_conjunctive() -> None:
    frame = _metric_frame()
    metrics = metric_deltas(frame)
    selection_gate = {
        "pooled_delta_rmse_m_max": -0.002,
        "minimum_nonworse_folds": 2,
        "minimum_nonworse_stations": 2,
        "lead_18_delta_rmse_m_max": 0.0,
        "lead_24_delta_rmse_m_max": 0.0,
    }
    assert evaluate_selection_gate(metrics, selection_gate)["pass"] is True
    bootstrap = paired_case_bootstrap(frame, replicates=100, seed=7)
    confirmation_gate = {
        "pooled_delta_rmse_m_max": -0.002,
        "paired_case_bootstrap_ci90_upper_strictly_below_m": 0.0,
        "minimum_nonworse_folds": 2,
        "minimum_nonworse_stations": 2,
        "lead_18_delta_rmse_m_max": 0.0,
        "lead_24_delta_rmse_m_max": 0.0,
        "short_lead_pooled_delta_rmse_m_max": 0.0005,
        "worst_station_by_lead_delta_rmse_m_max": 0.005,
    }
    decision = evaluate_confirmation_gate(metrics, bootstrap, confirmation_gate)
    assert decision["pass"] is True
    assert all(decision["checks"].values())


def test_static_preflight_reads_metadata_but_fits_no_model() -> None:
    receipt_path = (
        ROOT
        / "artifacts/p3_catboost_ordered_hpo_20260829_v1/static_preflight.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "STATIC_PREFLIGHT_PASS_ROOT_AUTHORIZED"
    assert receipt["grid"]["challenger_count"] == 48
    assert receipt["grid"]["maximum_authorized_future_fit_count"] == 174
    assert receipt["grid"]["catboost_fit_count"] == 0
    assert receipt["schema"]["compact_feature_count_from_schema"] == 591
    assert receipt["schema"]["official_or_confirmation_rows_materialized"] == 0
    assert receipt["execution_boundary"]["official_rows_read"] == 0
    assert receipt["execution_boundary"]["attempt_lock_created"] is False


def test_static_grid_check_rejects_ordered_depthwise_before_fit() -> None:
    config = _config()
    with pytest.raises(
        RUNNER.HPOContractError,
        match=(
            "challenger_37 uses Ordered boosting with grow_policy=Depthwise; "
            "Ordered boosting requires SymmetricTree"
        ),
    ):
        RUNNER._grid_checks(config, {"grid": GRID_PATH})


def test_execute_rejects_wrong_token_before_attempt_lock_or_data_access(tmp_path: Path) -> None:
    lock = (
        ROOT
        / "artifacts/p3_catboost_ordered_hpo_20260829_v1/one_shot/attempt.lock"
    )
    before = lock.read_bytes() if lock.exists() else None
    with pytest.raises(RUNNER.HPOContractError, match="authorization token differs"):
        RUNNER.execute_hpo(CONFIG_PATH, tmp_path, "WRONG_TOKEN")
    after = lock.read_bytes() if lock.exists() else None
    assert after == before
