from __future__ import annotations

import numpy as np
import pandas as pd

import p1_qc.r1_experiment as r1
from p1_qc.config import load_config
from p1_qc.features import FeatureBundle
from p1_qc.splits import Fold


class _FakeModel:
    model = None

    def __init__(self) -> None:
        self.model = self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.clip(np.nan_to_num(features[:, 0], nan=0.1), 0.01, 0.99)
        return np.column_stack((1.0 - positive, positive))


def _data() -> tuple[pd.DataFrame, pd.DataFrame, FeatureBundle]:
    count = 24
    train = pd.DataFrame(
        {
            "station": ["S-ORS"] * count,
            "year": [2025] * count,
            "layer": [1] * count,
            "time": pd.date_range(
                "2025-01-01", periods=count, freq="10min", tz="Asia/Seoul"
            ).astype(str),
            "temp": np.arange(count, dtype=float),
            "psal": [30.0] * count,
            "depth": [5.0] * count,
            "label": np.zeros(count, dtype=np.int8),
            "anomaly_type": [""] * count,
        }
    )
    train.loc[[2, 3, 8, 9, 14, 15], "label"] = 1
    train.loc[[2, 3, 8, 9, 14, 15], "anomaly_type"] = "offset"
    score = np.full(count, 0.1, dtype=np.float32)
    score[[2, 3, 8, 9, 14, 15]] = 0.8
    feature_frame = pd.DataFrame(
        {"score": score, "reference_resid_7d": np.arange(count, dtype=np.float32)}
    )
    bundle = FeatureBundle(feature_frame, tuple(feature_frame.columns), ())
    test = train.iloc[:4].drop(columns=["label", "anomaly_type"]).copy()
    test["year"] = 2026
    return train, test, bundle


def test_preregistered_grid_is_exact_and_contains_noop() -> None:
    grid = r1.preregistered_r1_grid()
    assert len(grid) == 37
    assert sum(item.get("enabled") is False for item in grid) == 1


def test_event_crossing_is_dropped_not_expanded() -> None:
    train, _, _ = _data()
    train["label"] = 0
    train.loc[4:6, "label"] = 1
    audit = r1.audit_inner_event_boundaries(train, np.arange(5), np.arange(7, 12))
    assert audit.dropped_runs == 1
    assert audit.dropped_fit_rows == 1
    assert 4 not in audit.fit_indices
    assert audit.calibration_indices.tolist() == list(range(7, 12))


def test_scope_outside_label_flip_cannot_change_boundary_audit() -> None:
    train, _, _ = _data()
    train["label"] = 0
    # The last calibration row is positive and must be dropped conservatively.
    train.loc[11, "label"] = 1
    fit = np.arange(6)
    calibration = np.arange(6, 12)
    scope = np.arange(12)
    stopped = r1.audit_inner_event_boundaries(
        train,
        fit,
        calibration,
        scope_indices=scope,
    )
    continued = train.copy()
    continued.loc[12, "label"] = 1  # immediately outside the permitted scope
    flipped = r1.audit_inner_event_boundaries(
        continued,
        fit,
        calibration,
        scope_indices=scope,
    )
    assert stopped.fit_indices.tolist() == flipped.fit_indices.tolist()
    assert stopped.calibration_indices.tolist() == flipped.calibration_indices.tolist()
    assert stopped.calibration_indices.tolist() == list(range(6, 11))
    assert stopped.dropped_calibration_rows == flipped.dropped_calibration_rows == 1
    assert stopped.crossed_boundaries == flipped.crossed_boundaries


def test_outer_labels_do_not_affect_selection(monkeypatch) -> None:
    train, test, bundle = _data()
    fold = Fold(
        "tiny",
        np.arange(12),
        np.arange(12, 24),
        pd.Timestamp("2025-01-01T01:50:00+09:00"),
        pd.Timestamp("2025-01-01T02:00:00+09:00"),
        pd.Timestamp("2025-01-01T04:00:00+09:00"),
    )
    monkeypatch.setattr(r1, "outer_folds", lambda *args, **kwargs: [fold])
    monkeypatch.setattr(
        r1,
        "_inner_calibration_indices",
        lambda *args, **kwargs: (np.arange(6), np.arange(6, 12)),
    )

    seen_columns: list[set[str]] = []

    def builder(*, frame, features, probabilities, base_prediction, parameters):
        seen_columns.append(set(frame.columns))
        mask = np.zeros(len(frame), dtype=bool)
        if parameters.get("enabled"):
            mask[features.frame["reference_resid_7d"].to_numpy() % 5 == 4] = True
        return mask

    def fake_fit(*args, **kwargs):
        return _FakeModel()

    config = load_config()
    grid = [{"enabled": False}, {"enabled": True}]
    first = r1.run_r1_nested_cv(
        train,
        test,
        bundle,
        config,
        builder,
        parameter_grid=grid,
        fit_model_fn=fake_fit,
    )
    changed = train.copy()
    changed.loc[fold.val_idx, "label"] = 1 - changed.loc[fold.val_idx, "label"]
    second = r1.run_r1_nested_cv(
        changed,
        test,
        bundle,
        config,
        builder,
        parameter_grid=grid,
        fit_model_fn=fake_fit,
    )
    assert first.selection == second.selection
    assert np.array_equal(first.oof["prediction"], second.oof["prediction"])
    assert all("label" not in columns and "anomaly_type" not in columns for columns in seen_columns)
    assert first.metrics["outer_labels_used_for_selection"] is False


def test_noop_does_not_promote_raw_spike_candidates(monkeypatch) -> None:
    train, test, bundle = _data()
    fold = Fold(
        "tiny",
        np.arange(12),
        np.arange(12, 24),
        pd.Timestamp("2025-01-01T01:50:00+09:00"),
        pd.Timestamp("2025-01-01T02:00:00+09:00"),
        pd.Timestamp("2025-01-01T04:00:00+09:00"),
    )
    monkeypatch.setattr(r1, "outer_folds", lambda *args, **kwargs: [fold])
    monkeypatch.setattr(
        r1,
        "_inner_calibration_indices",
        lambda *args, **kwargs: (np.arange(6), np.arange(6, 12)),
    )
    monkeypatch.setattr(
        r1,
        "detect_singleton_spikes",
        lambda frame: pd.Series(np.ones(len(frame), dtype=bool), index=frame.index),
    )

    result = r1.run_r1_nested_cv(
        train,
        test,
        bundle,
        load_config(),
        lambda **kwargs: np.zeros(len(kwargs["frame"]), dtype=bool),
        parameter_grid=[{"enabled": False}],
        fit_model_fn=lambda *args, **kwargs: _FakeModel(),
    )
    assert np.array_equal(result.oof["prediction"], result.oof["base_prediction"])


def test_noop_does_not_promote_raw_plateau_candidates(monkeypatch) -> None:
    train, test, bundle = _data()
    fold = Fold(
        "tiny",
        np.arange(12),
        np.arange(12, 24),
        pd.Timestamp("2025-01-01T01:50:00+09:00"),
        pd.Timestamp("2025-01-01T02:00:00+09:00"),
        pd.Timestamp("2025-01-01T04:00:00+09:00"),
    )
    monkeypatch.setattr(r1, "outer_folds", lambda *args, **kwargs: [fold])
    monkeypatch.setattr(
        r1,
        "_inner_calibration_indices",
        lambda *args, **kwargs: (np.arange(6), np.arange(6, 12)),
    )
    monkeypatch.setattr(
        r1,
        "detect_plateaus",
        lambda frame: pd.Series(np.ones(len(frame), dtype=bool), index=frame.index),
    )
    # Deliberately freeze an all-normal outer baseline so every raw plateau is
    # unconfirmed.  The R1 no-op must not promote any of them through the
    # protected-mask argument.
    monkeypatch.setattr(
        r1,
        "apply_postprocess",
        lambda frame, probabilities, plateau, spike, parameters: np.zeros(
            len(frame), dtype=np.int8
        ),
    )

    result = r1.run_r1_nested_cv(
        train,
        test,
        bundle,
        load_config(),
        lambda **kwargs: np.zeros(len(kwargs["frame"]), dtype=bool),
        parameter_grid=[{"enabled": False}],
        fit_model_fn=lambda *args, **kwargs: _FakeModel(),
    )
    assert not result.oof["base_prediction"].any()
    assert np.array_equal(result.oof["prediction"], result.oof["base_prediction"])


def test_candidate_factory_rejects_target_columns_in_feature_bundle() -> None:
    train, _, bundle = _data()
    poisoned_frame = bundle.frame.copy()
    poisoned_frame["label"] = train["label"].to_numpy()
    poisoned = FeatureBundle(
        poisoned_frame,
        (*bundle.feature_columns, "label"),
        bundle.categorical_columns,
    )
    with np.testing.assert_raises_regex(ValueError, "forbidden target columns"):
        r1._candidate_factory(
            lambda **kwargs: np.zeros(len(kwargs["frame"]), dtype=bool),
            train,
            poisoned,
            np.zeros(len(train)),
            np.zeros(len(train), dtype=np.int8),
        )
