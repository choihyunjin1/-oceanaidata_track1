from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.causal_forcing_inner_predictive import (
    BLIND_PREDICTION_COLUMNS,
    FoldScope,
    InnerTargetVault,
    apply_fixed_control_shrink,
    attach_validation_targets,
    expand_control_fit_rows,
    expand_control_prediction_rows,
    independently_recalculate_C_metrics,
    validate_blind_predictions,
)
from p3_wave.episode_distinct_analog import LEADS, EpisodeAnalogError


def _scopes() -> tuple[FoldScope, ...]:
    return (
        FoldScope("f1", np.asarray([0, 1]), np.asarray([2])),
        FoldScope("f2", np.asarray([0, 1, 2, 3]), np.asarray([4])),
        FoldScope("f3", np.asarray([0, 1, 2, 3, 4, 5]), np.asarray([6])),
    )


def _targets() -> np.ndarray:
    return np.arange(7 * len(LEADS), dtype=np.float64).reshape(7, len(LEADS))


def _blind_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for anchor_id, (fold, station) in enumerate(
        (("f1", "G-ORS"), ("f2", "I-ORS"), ("f3", "S-ORS"))
    ):
        for lead in LEADS:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "query_mad_scale": 0.2,
                    "analog_applicable": True,
                    "analog_prediction": 1.0,
                    "control_single_prediction": 2.0,
                    "control_final": 2.0,
                    "candidate_final": 2.0 if lead in (3, 6, 9) else 1.8,
                }
            )
    return pd.DataFrame(rows, columns=BLIND_PREDICTION_COLUMNS)


def test_target_vault_requires_blind_seal_before_validation_read() -> None:
    vault = InnerTargetVault(_targets(), _scopes())

    with pytest.raises(PermissionError, match="fsynced blind seal"):
        vault.read_validation("f1", [2])
    with pytest.raises(EpisodeAnalogError, match="lowercase SHA-256"):
        vault.seal_blind_predictions("not-a-sha")

    vault.seal_blind_predictions("a" * 64)
    assert np.array_equal(vault.read_validation("f1", [2]), _targets()[[2]])
    assert vault.access_log[-1]["blind_seal_sha256"] == "a" * 64


def test_target_vault_allows_prior_labels_but_blocks_current_or_future() -> None:
    vault = InnerTargetVault(_targets(), _scopes())

    assert np.array_equal(vault.read_fit("f2", [0, 1, 2, 3]), _targets()[[0, 1, 2, 3]])
    assert vault.access_log[-1]["allowed_prior_validation_overlap_count"] == 1

    unsafe_scopes = (
        FoldScope("f1", np.asarray([0, 1, 4]), np.asarray([2])),
        FoldScope("f2", np.asarray([0, 1, 2, 3]), np.asarray([4])),
        FoldScope("f3", np.asarray([0, 1, 2, 3, 4, 5]), np.asarray([6])),
    )
    unsafe = InnerTargetVault(_targets(), unsafe_scopes)
    with pytest.raises(PermissionError, match="current/future validation labels"):
        unsafe.read_fit("f1", [0, 1, 4])


def test_control_expansion_never_needs_prediction_targets() -> None:
    features = pd.DataFrame(
        {
            "anchor_id": [0, 1],
            "station": ["G-ORS", "I-ORS"],
            "x": [10.0, 20.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "anchor_id": [0, 1],
            "station": ["G-ORS", "I-ORS"],
            "current_hs": [1.5, 2.5],
        }
    )
    targets = np.asarray([[2.0] * 6, [3.0] * 6])

    fit, residual, metadata = expand_control_fit_rows(
        features=features,
        anchors=anchors,
        anchor_ids=[0, 1],
        target_matrix=targets,
        feature_columns=["x"],
    )
    prediction, prediction_metadata = expand_control_prediction_rows(
        features=features,
        anchors=anchors,
        anchor_ids=[0, 1],
        feature_columns=["x"],
    )

    assert len(fit) == len(prediction) == len(metadata) == 12
    assert residual.tolist() == pytest.approx([0.5, 0.5] * len(LEADS))
    assert not any("target" in column for column in prediction.columns)
    assert prediction_metadata["lead_h"].tolist() == [lead for lead in LEADS for _ in range(2)]


def test_fixed_control_shrink_changes_only_12_18_24_hours() -> None:
    prediction = np.full(len(LEADS), 3.0)
    current = np.full(len(LEADS), 1.0)

    actual = apply_fixed_control_shrink(prediction, current, np.asarray(LEADS))

    assert actual[:3] == pytest.approx([3.0, 3.0, 3.0])
    assert actual[3:] == pytest.approx([2.6, 2.6, 2.6])


def test_blind_schema_enforces_short_lead_no_op_and_no_targets() -> None:
    blind = _blind_predictions()
    validate_blind_predictions(blind, expected_cases=3)

    changed = blind.copy()
    changed.loc[changed["lead_h"].eq(3), "candidate_final"] -= 0.01
    with pytest.raises(EpisodeAnalogError, match="protected"):
        validate_blind_predictions(changed, expected_cases=3)

    exposed = blind.copy()
    exposed["target_hs"] = 1.0
    with pytest.raises(EpisodeAnalogError, match="columns changed"):
        validate_blind_predictions(exposed, expected_cases=3)


def test_row_level_targets_and_metrics_are_independently_recalculated() -> None:
    blind = _blind_predictions()
    targets = {
        "f1": (np.asarray([0]), np.ones((1, len(LEADS)))),
        "f2": (np.asarray([1]), np.ones((1, len(LEADS)))),
        "f3": (np.asarray([2]), np.ones((1, len(LEADS)))),
    }

    evaluated = attach_validation_targets(blind, targets)
    metrics = independently_recalculate_C_metrics(evaluated)

    assert metrics["pass"] is True
    assert metrics["rows"] == 18
    assert metrics["cases"] == 3
    assert metrics["analog_applicable_cases"] == 3
    assert metrics["pooled_delta_m"] < -0.005

    corrupted = evaluated.copy()
    corrupted.loc[0, "candidate_squared_error"] += 0.01
    with pytest.raises(EpisodeAnalogError, match="independently reproducible"):
        independently_recalculate_C_metrics(corrupted)
