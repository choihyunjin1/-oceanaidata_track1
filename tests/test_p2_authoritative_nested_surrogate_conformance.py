from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from p2_restore.authoritative_nested_surrogate_conformance import (
    COMPONENTS,
    TARGET_LAYERS,
    adapt_panel_for_full_prefix,
    adapt_panel_for_inner_fold,
    build_epoch_refit_receipt,
    build_prefix_plan,
    child_seed,
    fit_nnls_stack,
    joint_mask_target_observations,
    merge_component_oof,
)
from p2_restore.deep_data import P2Panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_conformance_20260825_v1.json"
)


def _metadata(periods: int = 120) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=periods, freq="12h", tz="Asia/Seoul")
    return pd.DataFrame(
        [("S", layer, time.isoformat()) for time in times for layer in range(1, 9)],
        columns=["station", "layer", "time"],
    )


def _plan():
    return build_prefix_plan(
        _metadata(),
        outer_fold="outer",
        validation_start_kst="2024-04-01T00:00:00+09:00",
        validation_stop_kst="2024-05-01T00:00:00+09:00",
        fraction=0.55,
    )


def _panel() -> P2Panel:
    times = pd.DatetimeIndex(pd.to_datetime(_metadata()["time"], utc=True).unique()).sort_values()
    rows = len(times)
    return P2Panel(
        times=times,
        inputs=np.zeros((rows, 2), dtype=np.float32),
        input_names=("public_a", "public_b"),
        baseline=np.ones((rows, 3), dtype=float),
        target=np.ones((rows, 3), dtype=float) * 2,
        target_mask=np.ones((rows, 3), dtype=bool),
        segment_ids=np.arange(rows, dtype=np.int32),
    )


def _component_oof() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    base = pd.DataFrame(
        [
            {
                "inner_fold": f"inner_{inner}",
                "station": "S",
                "layer": layer,
                "time": pd.Timestamp("2024-02-01", tz="UTC")
                + pd.Timedelta(days=inner),
                "truth": 10.0 + layer,
            }
            for inner in (1, 2, 3)
            for layer in TARGET_LAYERS
        ]
    )
    frames: dict[str, pd.DataFrame] = {}
    for number, component in enumerate(COMPONENTS):
        current = base.copy()
        current["prediction"] = current["truth"] + number * 0.01
        frames[component] = current
    return frames, base.loc[:, ["inner_fold", "station", "layer", "time"]]


def _runner() -> ModuleType:
    path = (
        PROJECT_ROOT
        / "scripts/validate_p2_authoritative_nested_surrogate_conformance_v1.py"
    )
    spec = importlib.util.spec_from_file_location("p2_auth_conformance_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_count_ties_and_seven_day_inner_embargo() -> None:
    plan = _plan()
    assert len(plan.prefix_times) == int(np.ceil(0.55 * len(plan.eligible_times)))
    assert plan.prefix_times[-1] == plan.cutoff_utc
    assert len(plan.inner_folds) == 3
    for inner in plan.inner_folds:
        assert inner.train_times[-1] < inner.validation_start_utc - pd.Timedelta(days=7)
        assert not inner.train_times.intersection(inner.validation_times).size


def test_joint_temp_psal_mask_preserves_public_layers() -> None:
    metadata = _metadata(8)
    values = metadata.assign(temp=np.arange(len(metadata), dtype=float), psal=30.0)
    allowed = pd.to_datetime(metadata["time"].unique()[:3], utc=True)
    masked, receipt = joint_mask_target_observations(values, allowed)
    times = pd.to_datetime(masked["time"], utc=True)
    outside_target = masked["layer"].isin(TARGET_LAYERS) & ~times.isin(allowed)
    assert masked.loc[outside_target, ["temp", "psal"]].isna().all().all()
    assert masked.loc[~masked["layer"].isin(TARGET_LAYERS), "temp"].equals(
        values.loc[~values["layer"].isin(TARGET_LAYERS), "temp"]
    )
    assert receipt["temp_rows_masked"] == receipt["psal_rows_masked"]
    assert receipt["public_rows_changed"] == 0


def test_deep_panel_adapters_remove_future_and_embargo_complement() -> None:
    plan = _plan()
    panel = _panel()
    adapted, receipt = adapt_panel_for_inner_fold(panel, plan.inner_folds[1])
    validation = (adapted.times >= plan.inner_folds[1].validation_start_utc) & (
        adapted.times < plan.inner_folds[1].validation_stop_utc
    )
    assert adapted.times[~validation].equals(plan.inner_folds[1].train_times)
    assert receipt["future_or_embargo_time_count_in_panel"] == 0
    full, full_receipt = adapt_panel_for_full_prefix(panel, plan)
    assert full.times.equals(plan.prefix_times)
    assert full_receipt["later_time_count_in_panel"] == 0


def test_child_seed_is_deterministic_and_phase_specific() -> None:
    first = child_seed(20260823, "router_400", "outer", 0.4, "inner_1")
    assert first == child_seed(20260823, "router_400", "outer", 0.4, "inner_1")
    assert first != child_seed(20260823, "router_400", "outer", 0.4, "full")
    assert first != child_seed(20260824, "router_400", "outer", 0.4, "inner_1")


def test_component_oof_requires_exact_order_and_registered_keys() -> None:
    frames, expected = _component_oof()
    merged, receipt = merge_component_oof(frames, expected_keys=expected)
    assert len(merged) == 9
    assert receipt["same_ordered_key_and_truth_across_components"] is True
    changed = {name: frame.copy() for name, frame in frames.items()}
    changed["lsti_style"] = changed["lsti_style"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered key/truth surface differs"):
        merge_component_oof(changed, expected_keys=expected)


def test_nnls_stack_is_nonnegative_sum_one_and_uniform_on_zero_solution() -> None:
    frame = pd.DataFrame(
        {
            "layer": np.repeat(TARGET_LAYERS, 4),
            "truth": np.ones(12),
            "a": np.zeros(12),
            "b": np.zeros(12),
        }
    )
    weights = fit_nnls_stack(frame, ("a", "b"))
    for layer in TARGET_LAYERS:
        assert np.allclose(weights[layer], [0.5, 0.5])
        assert np.isclose(weights[layer].sum(), 1.0)


def test_epoch_receipt_uses_earliest_tie_and_middle_of_three() -> None:
    histories = {
        component: {
            "inner_1": [{"epoch": 2, "rmse": 0.4}, {"epoch": 4, "rmse": 0.4}],
            "inner_2": [{"epoch": 8, "rmse": 0.3}],
            "inner_3": [{"epoch": 12, "rmse": 0.2}],
        }
        for component in COMPONENTS[1:]
    }
    receipt = build_epoch_refit_receipt(histories)
    assert receipt["depth_query_bitcn"]["best_epoch_by_inner"]["inner_1"] == 2
    assert receipt["depth_query_bitcn"]["full_prefix_epochs"] == 8
    assert receipt["depth_query_bitcn"]["frozen_epoch_reused"] is False


def test_runner_static_check_pins_parent_and_does_not_fit() -> None:
    result = _runner().run(CONFIG, execute=False)
    assert result["status"] == "PASS_STATIC_CHECK_ONLY"
    assert result["parent_contract_unchanged"] is True
    assert result["source_api_conformance"] == "PASS"
    assert result["new_model_fits"] == 0
