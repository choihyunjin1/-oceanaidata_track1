from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.era5_safe_advantage_router import (
    ROUTER_BASE_FEATURES,
    ROUTER_FEATURES,
    AdvantageRouterError,
    apply_bounded_router,
    build_inner_block_plan,
    calibrate_tau,
    exact_incumbent_fallback,
    fit_advantage_router,
    select_spaced_anchor_ids,
    validate_router_feature_names,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_p3_era5_incumbent_safe_advantage_router_20260828_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_p3_safe_advantage_router_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_router_feature_surface_is_exact_and_forbids_station_or_lead_one_hot() -> None:
    validate_router_feature_names(ROUTER_FEATURES)
    with pytest.raises(AdvantageRouterError):
        validate_router_feature_names((*ROUTER_FEATURES, "station"))
    with pytest.raises(AdvantageRouterError):
        validate_router_feature_names((*ROUTER_BASE_FEATURES, "lead_24"))


def test_inner_blocks_are_backward_only_and_station_spaced() -> None:
    times = pd.date_range("2024-01-01", "2024-06-30", freq="6h", tz="UTC")
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(len(times) * 2),
            "station": np.repeat(["G-ORS", "S-ORS"], len(times)),
            "anchor_time": np.tile(times, 2),
        }
    )
    selected = select_spaced_anchor_ids(anchors)
    selected_rows = anchors.set_index("anchor_id").loc[list(selected)].reset_index()
    for _station, group in selected_rows.groupby("station"):
        gaps = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        assert gaps.ge(pd.Timedelta(hours=78)).all()

    plan = build_inner_block_plan(
        anchors,
        (("outer", "2024-07-01", "2024-11-01"),),
    )["outer"]
    assert [block.name for block in plan] == ["I1", "I2", "I3", "I4"]
    assert all(block.end <= pd.Timestamp("2024-06-27T18:00:00Z") for block in plan)
    assert all((block.end - block.start) == pd.Timedelta(days=60) for block in plan)
    assert all(plan[index + 1].start - plan[index].end == pd.Timedelta(hours=78) for index in range(3))


def _synthetic_rows(cases: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(20260828)
    rows = cases * 6
    frame = pd.DataFrame(
        {
            "fold": np.repeat("past", rows),
            "anchor_id": np.repeat(np.arange(cases), 6),
            "station": np.repeat("G-ORS", rows),
            "lead_h": np.tile([3, 6, 9, 12, 18, 24], cases),
            "incumbent_prediction": rng.uniform(1.0, 3.0, rows),
        }
    )
    frame["transfer_prediction"] = frame["incumbent_prediction"] + rng.normal(0.0, 0.2, rows)
    frame["target_hs"] = frame["transfer_prediction"] + rng.normal(0.0, 0.05, rows)
    for name in ROUTER_BASE_FEATURES:
        frame[name] = rng.normal(size=rows)
    frame.loc[0, "wind_input_proxy_current"] = np.nan
    frame["lead_h_div_24"] = frame["lead_h"] / 24.0
    frame["transfer_minus_incumbent"] = (
        frame["transfer_prediction"] - frame["incumbent_prediction"]
    )
    frame["abs_transfer_minus_incumbent"] = frame["transfer_minus_incumbent"].abs()
    return frame


def test_fixed_ridge_tau_and_exact_noop_or_fixed_blend() -> None:
    fit = _synthetic_rows(24)
    calibration = _synthetic_rows(14)
    model = fit_advantage_router(fit)
    assert float(model.ridge.alpha) == 100.0
    tau = calibrate_tau(model, calibration)
    assert tau >= 0.0
    predicted = model.predict(calibration.loc[:, ROUTER_FEATURES])
    candidate, active = apply_bounded_router(calibration, predicted, tau)
    incumbent = calibration["incumbent_prediction"].to_numpy(float)
    transfer = calibration["transfer_prediction"].to_numpy(float)
    assert candidate[~active].tobytes() == incumbent[~active].tobytes()
    np.testing.assert_array_equal(candidate[active], incumbent[active] + 0.20 * (transfer[active] - incumbent[active]))

    fallback, inactive = exact_incumbent_fallback(calibration)
    assert fallback.tobytes() == incumbent.tobytes()
    assert not inactive.any()


def test_actual_check_only_preserves_frozen_support_topology_and_writes_nothing() -> None:
    module = _load_runner()
    _config, paths = module._load_contract(ROOT)
    output_before = paths.output.exists()
    lock_before = paths.attempt_lock.exists()
    result = module.check_only(ROOT)
    assert result["passed"] is True
    assert result["writes"] == 0
    assert result["outcome_values_read"] == 0
    assert result["incumbent_bytes_reproduced"] is True
    assert result["eligible_outer_folds"] == ["2025_h1"]
    assert result["preflight_fallback_outer_folds"] == [
        "2024_h2_storm",
        "winter_transition",
    ]
    assert paths.output.exists() is output_before
    assert paths.attempt_lock.exists() is lock_before
