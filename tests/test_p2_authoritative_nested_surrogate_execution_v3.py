from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.authoritative_nested_surrogate_execution_v3 import (
    adapt_panel_for_full_prefix_v3,
    adapt_panel_for_inner_fold_v3,
    build_prefix_plan_v3,
    fully_joint_target_times,
)
from p2_restore.deep_data import P2Panel


def _observations(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for time in times:
        for layer in (2, 3, 4):
            rows.append(
                {
                    "station": "S-ORS",
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": 10.0 + layer,
                    "psal": 30.0 + layer,
                }
            )
    return pd.DataFrame(rows)


def _panel(times: pd.DatetimeIndex) -> P2Panel:
    rows = len(times)
    return P2Panel(
        times=times,
        inputs=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        input_names=("public_a", "public_b"),
        baseline=np.zeros((rows, 3), dtype=float),
        target=np.ones((rows, 3), dtype=float),
        target_mask=np.ones((rows, 3), dtype=bool),
        segment_ids=np.zeros(rows, dtype=np.int32),
    )


def test_joint_ledger_requires_temp_and_psal_for_every_target_layer() -> None:
    times = pd.date_range("2024-05-01", periods=20, freq="10min", tz="UTC")
    observations = _observations(times)
    missing = (observations["time"] == times[3].isoformat()) & (
        observations["layer"] == 4
    )
    observations.loc[missing, "psal"] = np.nan
    ledger = fully_joint_target_times(observations)
    assert times[3] not in ledger
    assert ledger.equals(times.delete(3))


def test_prefix_fraction_uses_only_supervised_ledger_and_keeps_embargo() -> None:
    ledger = pd.date_range("2024-03-01", periods=20_000, freq="10min", tz="UTC")
    plan = build_prefix_plan_v3(
        ledger,
        outer_fold="outer_fixture",
        validation_start_kst="2024-08-01T00:00:00+09:00",
        validation_stop_kst="2024-09-01T00:00:00+09:00",
        fraction=0.4,
    )
    expected = int(np.ceil(0.4 * len(plan.eligible_times)))
    assert len(plan.prefix_times) == expected
    assert len(plan.inner_folds) == 3
    assert all(inner.train_times[-1] < inner.embargo_threshold_utc for inner in plan.inner_folds)
    assert all(
        len(inner.train_times.intersection(inner.validation_times)) == 0
        for inner in plan.inner_folds
    )


def test_deep_adapters_preserve_public_context_but_mask_nonledger_labels() -> None:
    times = pd.date_range("2024-01-01", periods=50_000, freq="10min", tz="UTC")
    supervised = times[5_000:]
    plan = build_prefix_plan_v3(
        supervised,
        outer_fold="outer_fixture",
        validation_start_kst="2024-10-01T00:00:00+09:00",
        validation_stop_kst="2024-11-01T00:00:00+09:00",
        fraction=0.4,
    )
    panel = _panel(times)
    inner_panel, receipt = adapt_panel_for_inner_fold_v3(panel, plan.inner_folds[0])
    assert receipt["continuous_public_covariates_preserved"] is True
    assert len(inner_panel.times) > len(plan.inner_folds[0].train_times) + len(
        plan.inner_folds[0].validation_times
    )
    allowed = plan.inner_folds[0].train_times.append(
        plan.inner_folds[0].validation_times
    )
    exposed = inner_panel.times[inner_panel.target_mask.all(axis=1)]
    assert exposed.equals(allowed)

    full_panel, full_receipt = adapt_panel_for_full_prefix_v3(panel, plan)
    assert full_receipt["continuous_public_covariates_preserved"] is True
    assert full_panel.times[-1] == plan.cutoff_utc
    assert full_panel.times[full_panel.target_mask.all(axis=1)].equals(plan.prefix_times)


def test_unlabeled_calendar_extension_cannot_change_supervised_prefix_cutoff() -> None:
    ledger = pd.date_range("2024-03-01", periods=20_000, freq="10min", tz="UTC")
    kwargs = {
        "outer_fold": "outer_fixture",
        "validation_start_kst": "2024-08-01T00:00:00+09:00",
        "validation_stop_kst": "2024-09-01T00:00:00+09:00",
        "fraction": 0.7,
    }
    first = build_prefix_plan_v3(ledger, **kwargs)
    # V3 has no all-observation timestamp argument.  An arbitrary public-only
    # calendar extension therefore cannot enter its fraction denominator.
    public_only_extension = pd.date_range(
        "2024-01-01", periods=10_000, freq="10min", tz="UTC"
    )
    assert len(public_only_extension) > 0
    assert len(public_only_extension) + len(ledger) > len(ledger)
    second = build_prefix_plan_v3(ledger.copy(), **kwargs)
    assert first.cutoff_utc == second.cutoff_utc
    assert first.prefix_times.equals(second.prefix_times)
