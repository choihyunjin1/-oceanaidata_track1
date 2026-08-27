from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

import p3_wave.era5_context_transfer as transfer
from p3_wave.era5_context_transfer import (
    LEADS,
    TARGET_COLUMNS,
    ERA5ContextTransferError,
    FixedContextTransferRegressor,
    build_source_anchors,
    build_source_cases,
    common_feature_columns,
    select_common_cached_features,
    select_source_year_training,
    select_source_year_validation,
    summarize_past_48h,
)


def _canonical_frame(
    times: pd.DatetimeIndex,
    *,
    hs: np.ndarray | None = None,
) -> pd.DataFrame:
    elapsed = (times - times[0]).total_seconds().to_numpy() / 3600.0
    wave_height = 1.5 + elapsed / 100.0 if hs is None else np.asarray(hs, dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "hs": wave_height,
            "tp": 8.0 + elapsed / 200.0,
            "hmax": wave_height * 1.6,
            "wvdir": 90.0,
            "wspd": 3.0,
            "wdir": 90.0,
            "airt": 10.0 + 2.0 * elapsed,
            "relh": 70.0,
            "caph": 1012.0,
        }
    )


def test_common_surface_is_286_identity_and_calendar_free() -> None:
    columns = common_feature_columns()
    assert len(columns) == 286
    assert len(set(columns)) == len(columns)
    assert columns[0:6] == (
        "hs_current",
        "hs_lag_3h",
        "hs_lag_6h",
        "hs_lag_12h",
        "hs_lag_24h",
        "hs_lag_48h",
    )
    assert not any(
        token in column.lower()
        for column in columns
        for token in ("station", "anchor", "timestamp", "calendar", "latitude", "longitude")
    )


def test_summary_uses_elapsed_time_and_ignores_future_rows() -> None:
    anchor = pd.Timestamp("2023-06-03T00:00:00+00:00")
    offsets = [-48, -31, -24, -12, -6, -3, 0, 1]
    times = pd.DatetimeIndex([anchor + pd.Timedelta(hours=hour) for hour in offsets])
    frame = _canonical_frame(times)
    # Make air temperature exactly linear in actual time relative to the anchor.
    frame["airt"] = 10.0 + 2.0 * np.asarray(offsets, dtype=float)
    frame.loc[frame["time"].gt(anchor), ["hs", "airt", "wspd"]] = 9999.0

    first = summarize_past_48h(frame, anchor)
    changed = frame.copy()
    changed.loc[changed["time"].gt(anchor), ["hs", "airt", "wspd"]] = -9999.0
    second = summarize_past_48h(changed, anchor)

    assert tuple(first) == common_feature_columns()
    assert first == second
    assert first["airt_slope_48h"] == pytest.approx(2.0)
    assert first["airt_lag_6h"] == pytest.approx(-2.0)
    assert first["wvdir_sin_current"] == pytest.approx(1.0)
    assert first["wvdir_cos_current"] == pytest.approx(0.0, abs=1e-12)
    assert first["wdir_sin_current"] == pytest.approx(1.0)
    assert first["wave_energy_current"] == pytest.approx(first["hs_current"] ** 2)
    assert first["wind_input_proxy_current"] == pytest.approx(9.0)


def test_source_anchors_use_fixed_grid_storm_gate_and_complete_log_targets() -> None:
    times = pd.date_range("2023-01-01", periods=85, freq="1h", tz="UTC")
    hs = 2.0 + np.arange(len(times), dtype=float) / 100.0
    hs[54] = 1.0
    frame = _canonical_frame(times, hs=hs)

    anchors = build_source_anchors(frame)

    assert anchors["anchor_time"].tolist() == [times[48], times[60]]
    assert anchors["current_hs"].ge(1.5).all()
    assert anchors.loc[:, [f"future_hs_{lead}h" for lead in LEADS]].notna().all().all()
    expected = np.log1p(anchors["future_hs_24h"]) - np.log1p(anchors["current_hs"])
    np.testing.assert_allclose(anchors["target_log_delta_24h"], expected)

    incomplete = frame.copy()
    incomplete.loc[incomplete["time"].eq(times[84]), "hs"] = np.nan
    incomplete_anchors = build_source_anchors(incomplete)
    assert incomplete_anchors["anchor_time"].tolist() == [times[48]]


def test_group_metadata_never_enters_source_features() -> None:
    times = pd.date_range("2023-01-01", periods=79, freq="1h", tz="UTC")
    first = _canonical_frame(times, hs=np.full(len(times), 2.0))
    second = _canonical_frame(times, hs=np.full(len(times), 2.5))
    first.insert(0, "station", "G-ORS")
    second.insert(0, "station", "I-ORS")
    source = pd.concat([first, second], ignore_index=True)

    cases = build_source_cases(source, group_column="station")

    assert len(cases.anchors) == 4
    assert "station" in cases.anchors
    assert "station" not in cases.features
    assert "anchor_time" not in cases.features
    assert tuple(cases.features.columns) == common_feature_columns()
    assert cases.log_delta_targets.shape == (4, 6)


def test_source_cases_canonicalize_each_station_once_and_exactly_match_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = pd.date_range("2020-01-01", periods=205, freq="1h", tz="UTC")
    first = _canonical_frame(times, hs=2.0 + np.sin(np.arange(len(times)) / 15.0) / 5.0)
    second = _canonical_frame(times, hs=2.4 + np.cos(np.arange(len(times)) / 17.0) / 5.0)
    first.insert(0, "station", "G-ORS")
    second.insert(0, "station", "I-ORS")
    source = pd.concat([first, second], ignore_index=True).sample(frac=1.0, random_state=7)
    expected_anchors = build_source_anchors(source, group_column="station")
    expected_rows = []
    for anchor in expected_anchors.itertuples(index=False):
        station_frame = source.loc[source["station"].eq(anchor.station)]
        expected_rows.append(summarize_past_48h(station_frame, anchor.anchor_time))
    expected_features = pd.DataFrame(expected_rows, columns=common_feature_columns())

    original = transfer.canonicalize_era5_hourly
    call_sizes: list[int] = []

    def counted(frame: pd.DataFrame, *, time_column: str = "time") -> pd.DataFrame:
        call_sizes.append(len(frame))
        return original(frame, time_column=time_column)

    monkeypatch.setattr(transfer, "canonicalize_era5_hourly", counted)
    actual = build_source_cases(source, group_column="station")

    assert sorted(call_sizes) == [len(first), len(second)]
    pd.testing.assert_frame_equal(actual.anchors, expected_anchors, check_exact=True)
    pd.testing.assert_frame_equal(actual.features, expected_features, check_exact=True)
    np.testing.assert_array_equal(
        actual.log_delta_targets,
        expected_anchors.loc[:, TARGET_COLUMNS].to_numpy(dtype=float),
    )
    assert tuple(actual.features.columns) == common_feature_columns()


def test_ten_thousand_hour_source_case_runtime_sanity() -> None:
    times = pd.date_range("2014-01-01", periods=10_000, freq="1h", tz="UTC")
    frame = _canonical_frame(times, hs=np.full(len(times), 2.0))

    started = time.perf_counter()
    cases = build_source_cases(frame)
    elapsed = time.perf_counter() - started

    assert len(cases.anchors) > 1_600
    assert cases.features.shape == (len(cases.anchors), 286)
    # This is a regression tripwire for accidental O(anchors * station_hours),
    # not a machine-to-machine microbenchmark.
    assert elapsed < 30.0


def test_cached_helper_selects_exact_common_subset_and_discards_metadata() -> None:
    requested = common_feature_columns()[:4]
    cache = pd.DataFrame(
        {
            "anchor_id": [1, 2],
            "station": ["G-ORS", "I-ORS"],
            "anchor_time": pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True),
            **{column: [1.0, 2.0] for column in requested},
        }
    )

    selected = select_common_cached_features(cache, source_columns=requested)

    assert tuple(selected.columns) == requested
    with pytest.raises(ERA5ContextTransferError, match="missing common columns"):
        select_common_cached_features(cache)


def test_source_year_split_keeps_full_footprints_inside_year_and_78h_apart() -> None:
    rows: list[dict[str, Any]] = []
    identifier = 1
    for station in ("G-ORS", "I-ORS"):
        for timestamp in pd.to_datetime(
            [
                "2021-01-01T00:00:00Z",  # Its 48-hour context crosses the year boundary.
                "2021-01-03T00:00:00Z",
                "2021-01-06T00:00:00Z",  # Only 72 hours after the first eligible ID.
                "2021-01-06T06:00:00Z",  # Exactly 78 hours; eligible.
                "2021-12-31T00:00:00Z",  # Its +24-hour target touches the next year.
                "2021-12-30T23:00:00Z",  # Last possible hourly footprint in 2021.
                "2022-01-03T00:00:00Z",  # Only 73 hours later; globally ineligible.
                "2022-01-06T06:00:00Z",
            ],
            utc=True,
        ):
            rows.append(
                {
                    "anchor_id": identifier,
                    "station": station,
                    "anchor_time": timestamp,
                    "current_hs": 9999.0,
                    "target_log_delta_24h": 9999.0,
                }
            )
            identifier += 1
    anchors = pd.DataFrame(rows)

    selected = select_source_year_validation(anchors, held_years=(2021, 2022))

    assert list(selected.columns) == [
        "year",
        "episode_id",
        "anchor_id",
        "station",
        "anchor_time",
    ]
    counts = selected.groupby(["year", "station"], observed=True).size()
    assert counts.loc[(2021, "G-ORS")] == 3
    assert counts.loc[(2021, "I-ORS")] == 3
    assert counts.loc[(2022, "G-ORS")] == 1
    assert counts.loc[(2022, "I-ORS")] == 1
    assert selected["episode_id"].is_unique
    assert not any(
        column.startswith(("current_", "future_", "target_")) for column in selected
    )
    for _, group in selected.groupby("station", observed=True):
        gaps = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
        assert gaps.ge(pd.Timedelta(hours=78)).all()
    for (year, _), group in selected.groupby(["year", "station"], observed=True):
        start = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC")
        assert (group["anchor_time"] - pd.Timedelta(hours=48)).ge(start).all()
        assert (group["anchor_time"] + pd.Timedelta(hours=24)).lt(end).all()


def test_source_training_selection_assigns_only_year_internal_2014_2020_footprints() -> None:
    rows: list[dict[str, Any]] = []
    identifier = 1
    for year in range(2014, 2021):
        for station in ("G-ORS", "I-ORS"):
            for timestamp in (
                f"{year}-01-01T00:00:00Z",
                f"{year}-01-03T00:00:00Z",
                f"{year}-12-30T23:00:00Z",
                f"{year}-12-31T00:00:00Z",
            ):
                rows.append(
                    {
                        "anchor_id": identifier,
                        "station": station,
                        "anchor_time": timestamp,
                        "future_hs_24h": 9999.0,
                        "target_log_delta_24h": 9999.0,
                    }
                )
                identifier += 1

    selected = select_source_year_training(pd.DataFrame(rows))

    assert list(selected.columns) == ["year", "anchor_id", "station", "anchor_time"]
    assert set(selected["year"]) == set(range(2014, 2021))
    assert selected.groupby(["year", "station"], observed=True).size().eq(2).all()
    assert not any(column.startswith(("future_", "target_")) for column in selected)
    for (year, _), group in selected.groupby(["year", "station"], observed=True):
        start = pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC")
        assert (group["anchor_time"] - pd.Timedelta(hours=48)).ge(start).all()
        assert (group["anchor_time"] + pd.Timedelta(hours=24)).lt(end).all()


class _FakeCatBoostRegressor:
    instances: list[_FakeCatBoostRegressor] = []

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.fit_call: dict[str, Any] | None = None
        self.__class__.instances.append(self)

    def fit(self, matrix: pd.DataFrame, target: np.ndarray, **kwargs: Any) -> None:
        self.fit_call = {"matrix": matrix.copy(), "target": target.copy(), **kwargs}

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        return matrix["lead_h"].to_numpy(dtype=float) / 100.0


def test_fixed_wrapper_continues_from_source_and_accepts_context_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCatBoostRegressor.instances.clear()
    monkeypatch.setattr(
        transfer,
        "_new_catboost_regressor",
        lambda parameters: _FakeCatBoostRegressor(**dict(parameters)),
    )
    columns = common_feature_columns()[:8]
    source = pd.DataFrame(np.arange(24, dtype=float).reshape(3, 8), columns=columns)
    local = source.iloc[:2].copy()
    source_targets = np.zeros((3, len(LEADS)), dtype=float)
    local_targets = np.ones((2, len(LEADS)), dtype=float) / 100.0

    regressor = FixedContextTransferRegressor().fit_transfer(
        source,
        source_targets,
        local,
        local_targets,
    )

    assert len(_FakeCatBoostRegressor.instances) == 2
    pretrain, continuation = _FakeCatBoostRegressor.instances
    assert pretrain.parameters == dict(transfer.SOURCE_CATBOOST_PARAMETERS)
    assert continuation.parameters == dict(transfer.LOCAL_CATBOOST_PARAMETERS)
    assert pretrain.fit_call is not None
    assert continuation.fit_call is not None
    assert pretrain.fit_call["matrix"].shape == (18, 9)
    assert continuation.fit_call["init_model"] is pretrain
    assert continuation.fit_call["sample_weight"].shape == (12,)
    prediction = regressor.predict_log_delta(local)
    assert prediction.shape == (2, 6)
    np.testing.assert_allclose(prediction[0], np.asarray(LEADS) / 100.0)

    prohibited = local.assign(station="G-ORS")
    with pytest.raises(ERA5ContextTransferError, match="prohibited"):
        regressor.predict_log_delta(prohibited)


def test_clone_pretrained_gives_each_fake_fold_independent_source_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCatBoostRegressor.instances.clear()
    monkeypatch.setattr(
        transfer,
        "_new_catboost_regressor",
        lambda parameters: _FakeCatBoostRegressor(**dict(parameters)),
    )
    source = pd.DataFrame({"hs_current": [1.5, 1.8, 2.1]})
    targets = np.arange(18, dtype=float).reshape(3, 6) / 100.0
    pretrained = FixedContextTransferRegressor().fit_pretrain(source, targets)

    first = pretrained.clone_pretrained()
    second = pretrained.clone_pretrained()

    assert first.source_model is not pretrained.source_model
    assert second.source_model is not pretrained.source_model
    assert first.source_model is not second.source_model
    first.source_model.fold_marker = "first"
    assert not hasattr(pretrained.source_model, "fold_marker")
    assert not hasattr(second.source_model, "fold_marker")
    first.continue_local(source.iloc[:2], targets[:2])
    assert pretrained.model is pretrained.source_model
    assert second.model is second.source_model
    assert first.model is not first.source_model


def test_clone_pretrained_is_independent_with_actual_catboost() -> None:
    pytest.importorskip("catboost")
    rows = 18
    source = pd.DataFrame(
        {
            "hs_current": np.linspace(1.5, 2.5, rows),
            "hs_lag_3h": np.linspace(1.4, 2.6, rows),
        }
    )
    targets = np.sin(np.arange(rows * len(LEADS), dtype=float)).reshape(rows, len(LEADS)) / 20.0
    pretrained = FixedContextTransferRegressor().fit_pretrain(source, targets)
    before = pretrained.predict_log_delta(source.iloc[:3])
    before_trees = int(pretrained.source_model.tree_count_)

    fold = pretrained.clone_pretrained()

    assert fold.source_model is not pretrained.source_model
    np.testing.assert_allclose(fold.predict_log_delta(source.iloc[:3]), before)
    fold.continue_local(source.iloc[:9], targets[:9])
    assert int(fold.model.tree_count_) > before_trees
    assert int(pretrained.source_model.tree_count_) == before_trees
    np.testing.assert_allclose(pretrained.predict_log_delta(source.iloc[:3]), before)
    untouched_fold = pretrained.clone_pretrained()
    assert int(untouched_fold.model.tree_count_) == before_trees


def test_target_dataframe_is_reordered_to_frozen_lead_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCatBoostRegressor.instances.clear()
    monkeypatch.setattr(
        transfer,
        "_new_catboost_regressor",
        lambda parameters: _FakeCatBoostRegressor(**dict(parameters)),
    )
    features = pd.DataFrame({"hs_current": [2.0, 2.1]})
    targets = pd.DataFrame(
        np.arange(12, dtype=float).reshape(2, 6),
        columns=reversed(TARGET_COLUMNS),
    )

    FixedContextTransferRegressor().fit_pretrain(features, targets)

    fit_call = _FakeCatBoostRegressor.instances[0].fit_call
    assert fit_call is not None
    expected = targets.loc[:, TARGET_COLUMNS].to_numpy().reshape(-1)
    np.testing.assert_array_equal(fit_call["target"], expected)
