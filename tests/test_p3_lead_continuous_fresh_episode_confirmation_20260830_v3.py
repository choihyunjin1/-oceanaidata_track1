from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.loss_router import OBSERVED_FEATURES
from p3_wave.p3_lead_continuous_fresh_episode_confirmation_20260830_v3 import (
    LEADS,
    build_comparison_frame,
    build_safe_single_design,
    classify_terminal,
    comparison_metrics,
    predict_frozen_reference,
    select_fresh_surface,
    uncertainty_or_insufficient,
)


class EqualRouter:
    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([0.5, 0.5, 0.0]), (len(frame), 1))


def _safe_features_and_anchors() -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = pd.DataFrame(
        {
            "anchor_id": [10, 11],
            "station": ["G-ORS", "S-ORS"],
            "anchor_time": pd.to_datetime(
                ["2025-06-26T00:00:00Z", "2025-06-26T04:00:00Z"], utc=True
            ),
            "current_hs": [1.6, 1.8],
        }
    )
    features = anchors[["anchor_id", "station"]].copy()
    for number, name in enumerate(OBSERVED_FEATURES):
        features[name] = float(number + 1)
    return features, anchors


def test_select_fresh_surface_uses_metadata_only_and_station_global_gap() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": [1, 2, 3, 4],
            "station": ["G-ORS", "G-ORS", "G-ORS", "S-ORS"],
            "anchor_time": pd.to_datetime(
                [
                    "2025-06-25T00:00:00Z",
                    "2025-06-29T00:00:00Z",
                    "2025-06-29T12:00:00Z",
                    "2025-06-29T00:00:00Z",
                ],
                utc=True,
            ),
            "target_3": [99.0, 98.0, 97.0, 96.0],
        }
    )
    exposed = pd.DataFrame(
        {
            "station": ["G-ORS", "S-ORS"],
            "anchor_time": pd.to_datetime(
                ["2025-06-24T00:00:00Z", "2025-06-25T00:00:00Z"], utc=True
            ),
        }
    )
    selected, audit = select_fresh_surface(
        anchors,
        exposed,
        start=pd.Timestamp("2025-06-25T00:00:00Z"),
        end=pd.Timestamp("2025-07-01T00:00:00Z"),
        separation_hours=78,
    )
    assert set(selected["anchor_id"]) == {2, 4}
    assert audit["target_value_columns_used_for_selection"] == 0
    assert audit["prediction_value_columns_used_for_selection"] == 0
    assert audit["minimum_gap_to_exposed_h"] >= 78.0


def test_safe_design_rejects_target_columns_and_preserves_six_leads() -> None:
    features, anchors = _safe_features_and_anchors()
    design, current, stations = build_safe_single_design(
        features, anchors, list(OBSERVED_FEATURES)
    )
    assert len(design) == len(anchors) * len(LEADS)
    assert tuple(design["lead_h"].drop_duplicates().astype(int)) == LEADS
    assert current.tolist() == [1.6, 1.8]
    assert stations.tolist() == ["G-ORS", "S-ORS"]
    with pytest.raises(ValueError, match="target columns forbidden"):
        build_safe_single_design(features, anchors, ["target_3"])
    forbidden = anchors.assign(target_3=2.0)
    with pytest.raises(ValueError, match="must not contain target"):
        build_safe_single_design(features, forbidden, list(OBSERVED_FEATURES))


def test_frozen_reference_is_inference_only_and_six_lead_aligned() -> None:
    features, anchors = _safe_features_and_anchors()
    cases = len(anchors)
    prediction, receipt = predict_frozen_reference(
        features,
        anchors,
        feature_columns=list(OBSERVED_FEATURES),
        single_predict=lambda frame: np.zeros(len(frame), dtype=float),
        multi_predict=lambda frame: np.zeros((cases, len(LEADS)), dtype=float),
        router=EqualRouter(),
        shrink_weight=0.2,
    )
    expected = np.repeat(anchors["current_hs"].to_numpy(dtype=float), len(LEADS))
    assert np.allclose(prediction, expected, rtol=0.0, atol=1.0e-12)
    assert receipt["catboost_fit_count"] == 0
    assert receipt["router_fit_count"] == 0
    assert receipt["target_value_columns_used"] == 0


def test_single_fresh_episode_never_emits_degenerate_ci() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": [1],
            "station": ["S-ORS"],
            "anchor_time": pd.to_datetime(["2025-06-26T00:00:00Z"], utc=True),
            "current_hs": [1.7],
            **{f"target_{lead}": [1.7 + 0.01 * lead] for lead in LEADS},
        }
    )
    incumbent = np.repeat(1.7, len(LEADS))
    candidate = incumbent + 0.01
    frame = build_comparison_frame(anchors, incumbent, candidate)
    metrics = comparison_metrics(frame)
    uncertainty = uncertainty_or_insufficient(frame, minimum_blocks=2)
    assert metrics["overall"]["rows"] == 6
    assert uncertainty["benefit_ci90_m"] is None
    assert uncertainty["bootstrap_replicates_executed"] == 0
    state = classify_terminal(
        integrity_checks={"all": True}, uncertainty=uncertainty
    )
    assert state == "INCONCLUSIVE_FRESH_SINGLE_EPISODE_INSUFFICIENT_DEPENDENCE_UNITS"


def test_preregistered_config_is_unique_and_forbids_official_action() -> None:
    root = Path(__file__).resolve().parents[1]
    config = (
        root
        / "configs/experiments/p3_lead_continuous_fresh_episode_confirmation_20260830_v3.json"
    )
    text = config.read_text(encoding="utf-8")
    assert "maximum_executions\": 1" in text
    assert "official_action_authorized\": false" in text
    assert "legacy_minimum_0_005m_applied\": false" in text
