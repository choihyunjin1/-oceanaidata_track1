from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from p1_qc.config import FeatureConfig, P1QCConfig, SplitConfig
from p1_qc.features import FeatureBundle
from p1_qc.stress import run_gors_holdout_stress, run_year_transfer_stress

POSTPROCESS = {
    "high_threshold": 0.5,
    "low_threshold": 0.5,
    "close_gap_rows": 0,
    "minimum_positive_run": 1,
}


def _rows(
    station: str,
    year: int,
    start: str,
    labels: list[int],
    *,
    depth_regime: str,
    peer_available: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = len(labels)
    time = pd.date_range(start, periods=count, freq="10min", tz="Asia/Seoul")
    raw = pd.DataFrame(
        {
            "station": [station] * count,
            "year": [year] * count,
            "layer": [1] * count,
            "time": time.astype(str),
            "temp": np.arange(count, dtype=float) * 0.1 + 10.0,
            "psal": [33.0] * count,
            "depth": [5.0] * count,
            "label": labels,
            "anomaly_type": ["spike" if value else "" for value in labels],
        }
    )
    feature = pd.DataFrame(
        {
            "station": pd.Series([station] * count, dtype="string"),
            "layer_category": pd.Series(["1"] * count, dtype="string"),
            "depth_regime": pd.Series([depth_regime] * count, dtype="string"),
            # The fake wrapper uses this already constructed numeric feature.
            "signal": np.where(np.asarray(labels) == 1, 0.9, 0.1).astype(np.float32),
            "peer_available": np.full(count, peer_available, dtype=np.float32),
            "temp_peer_residual": np.full(
                count, 0.2 if peer_available else np.nan, dtype=np.float32
            ),
            "reference_resid_7d": np.linspace(0.1, 0.2, count, dtype=np.float32),
        }
    )
    return raw, feature


def _fixture() -> tuple[pd.DataFrame, FeatureBundle]:
    pieces = [
        _rows(
            "S-ORS",
            2024,
            "2024-01-01",
            [0, 1, 0, 1, 0, 0, 1, 0],
            depth_regime="S-ORS|d005.0",
            peer_available=1,
        ),
        _rows(
            "S-ORS",
            2025,
            "2025-01-01",
            [0, 1, 0, 0, 1, 0],
            depth_regime="S-ORS|d006.0",
            peer_available=1,
        ),
        _rows(
            "I-ORS",
            2025,
            "2025-02-01",
            [0, 1, 0, 1, 0, 0],
            depth_regime="I-ORS|d005.0",
            peer_available=1,
        ),
        _rows(
            "G-ORS",
            2025,
            "2025-03-01",
            [0, 1, 0, 1, 0, 0],
            depth_regime="G-ORS|unknown|l1",
            peer_available=0,
        ),
    ]
    train = pd.concat([raw for raw, _ in pieces], ignore_index=True)
    features = pd.concat([feature for _, feature in pieces], ignore_index=True)
    feature_columns = tuple(features.columns)
    categorical = ("station", "layer_category", "depth_regime")
    features.attrs["feature_mode"] = "causal"
    return train, FeatureBundle(features, feature_columns, categorical)


def _config(*, with_iteration: bool = True) -> P1QCConfig:
    model = {"learning_rate": 0.04}
    if with_iteration:
        model["n_estimators"] = 11
    return P1QCConfig(
        seed=17,
        mode="causal",
        features=FeatureConfig(mode="causal"),
        splits=SplitConfig(purge_days=7),
        raw={
            "project": {"threads": 1},
            "models": {"lightgbm": model},
        },
    )


@dataclass
class _RecordingModel:
    record: dict[str, Any]

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        **fit_parameters: Any,
    ) -> _RecordingModel:
        self.record["fit_features"] = np.asarray(features).copy()
        self.record["fit_target"] = np.asarray(target).copy()
        self.record["fit_parameters"] = dict(fit_parameters)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features)
        self.record["predict_features"] = matrix.copy()
        positive = matrix[:, 3].astype(float)  # ``signal`` after three categories
        return np.column_stack((1.0 - positive, positive))


def _recording_factory(records: list[dict[str, Any]]):
    def factory(
        backend: str,
        *,
        seed: int,
        n_jobs: int,
        parameters: dict[str, Any],
    ) -> _RecordingModel:
        record: dict[str, Any] = {
            "backend": backend,
            "seed": seed,
            "n_jobs": n_jobs,
            "parameters": dict(parameters),
        }
        records.append(record)
        return _RecordingModel(record)

    return factory


def test_year_transfer_fits_encoder_and_model_on_2024_only() -> None:
    train, bundle = _fixture()
    records: list[dict[str, Any]] = []
    result = run_year_transfer_stress(
        train,
        bundle,
        _config(),
        backend="lightgbm",
        postprocess_selection=POSTPROCESS,
        selected_iteration=7,
        classifier_factory=_recording_factory(records),
    )

    fitted = train.iloc[result.train_idx]
    holdout = train.iloc[result.holdout_idx]
    assert set(fitted["station"]) == {"S-ORS"}
    assert set(fitted["year"]) == {2024}
    assert set(holdout["station"]) == {"S-ORS"}
    assert set(holdout["year"]) == {2025}
    assert pd.to_datetime(holdout["time"], utc=True).max() < pd.Timestamp(
        "2025-07-01T00:00:00+09:00"
    ).tz_convert("UTC")

    record = records[0]
    assert np.array_equal(record["fit_target"], fitted["label"].to_numpy())
    assert len(record["fit_target"]) == len(result.train_idx)
    assert set(record["fit_parameters"]) == {"sample_weight"}
    assert record["parameters"]["n_estimators"] == 7
    assert result.preprocessing["category_maps_fitted_on"] == "stress_train_only"
    assert result.preprocessing["numeric_scaling"] == "none"
    # The 2025 deployment category is unknown because the map saw 2024 only.
    assert result.preprocessing["unseen_category_rows"]["depth_regime"] == len(holdout)
    assert result.metrics.micro.f1 == pytest.approx(1.0)
    assert result.metrics.weighted.f1 == pytest.approx(1.0)
    assert result.metrics.type_recall["spike"] == pytest.approx(1.0)
    assert result.metrics.events.recall == pytest.approx(1.0)


def test_gors_holdout_reports_unseen_station_and_no_peer_fallback() -> None:
    train, bundle = _fixture()
    records: list[dict[str, Any]] = []
    result = run_gors_holdout_stress(
        train,
        bundle,
        _config(),
        backend="lightgbm",
        postprocess_selection=POSTPROCESS,
        classifier_factory=_recording_factory(records),
    )

    assert train.iloc[result.holdout_idx]["station"].eq("G-ORS").all()
    assert not train.iloc[result.train_idx]["station"].eq("G-ORS").any()
    assert result.preprocessing["unseen_category_rates"]["station"] == pytest.approx(1.0)
    # The first encoded column is station; unseen holdout categories map to -1.
    assert np.all(records[0]["predict_features"][:, 0] == -1)
    assert result.fallback["peer_available_rows"] == 0
    assert result.fallback["no_peer_rows"] == len(result.holdout_idx)
    assert result.fallback["peer_metrics"] is None
    assert result.fallback["no_peer_metrics"]["f1"] == pytest.approx(1.0)
    assert result.fallback["peer_residual_missing_rate_no_peer"] == pytest.approx(1.0)
    assert result.fallback["reference_finite_rate_no_peer"]["reference_resid_7d"] == pytest.approx(
        1.0
    )
    assert result.metrics.groups.iloc[0]["station"] == "G-ORS"


def test_holdout_cannot_select_missing_iteration_or_postprocess_values() -> None:
    train, bundle = _fixture()
    records: list[dict[str, Any]] = []
    factory = _recording_factory(records)
    with pytest.raises(ValueError, match="preselected n_estimators"):
        run_gors_holdout_stress(
            train,
            bundle,
            _config(with_iteration=False),
            backend="lightgbm",
            postprocess_selection=POSTPROCESS,
            classifier_factory=factory,
        )
    with pytest.raises(ValueError, match="postprocess_selection is not preselected"):
        run_gors_holdout_stress(
            train,
            bundle,
            _config(),
            backend="lightgbm",
            postprocess_selection={"high_threshold": 0.5},
            classifier_factory=factory,
        )
    # Both validations fail before any model can inspect a holdout.
    assert records == []
