from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.iors_external_point_residual import (
    KEY_COLUMNS,
    POINT_RESIDUAL_COLUMNS,
    ExternalPointPrediction,
    append_point_residual_matrix,
    apply_point_residual_gate,
    build_p1_iors_panel,
    build_point_residual_features,
    canonical_artifact_paths,
    compose_incumbent_predictions,
    independent_expected_replacement_keys,
    predict_external_q50,
    select_inner_threshold,
)

CONFIG = Path("configs/experiments/p1_iors_external_point_residual_oof_v1.json")


class BaselineModel:
    """Return the fixed depth-linear feature; it cannot read target TEMP."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 14].astype(np.float64)


def _panel_frame() -> pd.DataFrame:
    rows = []
    for minute, values in ((0, [10.0, 12.0, 14.0]), (10, [11.0, 13.0, 15.0])):
        time = pd.Timestamp("2025-01-01T00:00:00+09:00") + pd.Timedelta(minutes=minute)
        for layer, (temp, depth) in enumerate(zip(values, [5.0, 10.0, 15.0], strict=True), start=1):
            rows.append(
                {
                    "station": "I-ORS",
                    "year": 2025,
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": temp,
                    "psal": 33.0 + layer * 0.1,
                    "depth": depth,
                }
            )
    rows.append(
        {
            "station": "S-ORS",
            "year": 2025,
            "layer": 1,
            "time": "2025-01-01T00:00:00+09:00",
            "temp": 99.0,
            "psal": 30.0,
            "depth": 4.0,
        }
    )
    return pd.DataFrame(rows)


def test_external_q50_masks_the_target_temperature() -> None:
    first = _panel_frame()
    changed = first.copy()
    target_position = 1  # first timestamp, layer 2
    changed.loc[target_position, "temp"] = 120.0
    depths = {1: 5.0, 2: 10.0, 3: 15.0}

    first_prediction = predict_external_q50(
        build_p1_iors_panel(first, depths), BaselineModel(), min_peer_temperatures=2
    )
    changed_prediction = predict_external_q50(
        build_p1_iors_panel(changed, depths), BaselineModel(), min_peer_temperatures=2
    )

    assert first_prediction.q50[target_position] == changed_prediction.q50[target_position]
    assert first_prediction.eligible.sum() == 6
    assert first_prediction.eligible[-1] is np.False_
    assert first_prediction.audit["target_temperature_masked"] is True
    assert first_prediction.audit["quantiles_used"] == [0.5]


def test_point_feature_contract_excludes_interval_quantiles() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert tuple(contract["point_feature_contract"]["columns"]) == POINT_RESIDUAL_COLUMNS
    assert contract["point_feature_contract"]["q10_q90_features"] is False
    assert contract["non_virgin_follow_up"]["q10_q90_excluded"] is True
    assert all("q10" not in value and "q90" not in value for value in POINT_RESIDUAL_COLUMNS)


def test_gap_safe_rolling_median_and_slope_do_not_cross_a_gap() -> None:
    times = [
        *(
            pd.Timestamp("2025-01-01T00:00:00+09:00") + pd.Timedelta(minutes=10 * i)
            for i in range(6)
        ),
        *(
            pd.Timestamp("2025-01-01T02:00:00+09:00") + pd.Timedelta(minutes=10 * i)
            for i in range(6)
        ),
    ]
    temp = np.asarray([*range(6), *range(100, 106)], dtype=float)
    frame = pd.DataFrame(
        {
            "station": "I-ORS",
            "year": 2025,
            "layer": 1,
            "time": [value.isoformat() for value in times],
            "temp": temp,
        }
    )
    prediction = ExternalPointPrediction(
        q50=np.zeros(len(frame)),
        peer_count=np.full(len(frame), 3.0),
        eligible=np.ones(len(frame), dtype=bool),
        audit={},
    )

    features, audit = build_point_residual_features(frame, prediction, minimum_fraction=0.01)

    assert features.loc[5, "ext_residual_median_24h"] == pytest.approx(2.5)
    assert features.loc[6, "ext_residual_median_24h"] == pytest.approx(102.5)
    assert features.loc[4, "ext_residual_slope_24h"] == pytest.approx(6.0)
    assert features.loc[7, "ext_residual_slope_24h"] == pytest.approx(6.0)
    assert audit["segments"] == 2
    assert audit["gap_safe"] is True


def test_append_point_features_requires_exact_columns() -> None:
    frame = pd.DataFrame(np.ones((3, len(POINT_RESIDUAL_COLUMNS))), columns=POINT_RESIDUAL_COLUMNS)
    base = np.zeros((2, 4), dtype=np.float32)

    result = append_point_residual_matrix(base, frame, [0, 2])

    assert result.shape == (2, 4 + len(POINT_RESIDUAL_COLUMNS))
    with pytest.raises(ValueError, match="columns differ"):
        append_point_residual_matrix(base, frame.drop(columns=POINT_RESIDUAL_COLUMNS[-1]), [0, 2])


def test_compose_changes_only_expected_iors_rows_byte_exactly() -> None:
    reference = pd.DataFrame(
        {
            "station": ["G-ORS", "I-ORS", "I-ORS", "S-ORS"],
            "year": [2025] * 4,
            "layer": [1, 1, 2, 1],
            "time": [f"2025-04-01T00:{minute:02d}:00+09:00" for minute in (0, 10, 20, 30)],
            "fold": ["q2"] * 4,
            "prediction": np.asarray([0, 0, 1, 1], dtype=np.int8),
            "probability": np.asarray([0.1, 0.2, 0.9, 0.8], dtype=np.float32),
        }
    )
    replacements = reference.iloc[[1]].loc[:, [*KEY_COLUMNS, "fold"]].copy()
    replacements["candidate_prediction"] = np.int8(1)
    replacements["candidate_probability"] = np.float32(0.75)
    expected = replacements.loc[:, [*KEY_COLUMNS, "fold"]].copy()

    output, audit = compose_incumbent_predictions(reference, replacements, expected)

    assert output["candidate_prediction"].tolist() == [0, 1, 1, 1]
    assert (
        output.loc[[0, 2, 3], "candidate_probability"].to_numpy().tobytes()
        == reference.loc[[0, 2, 3], "probability"].to_numpy(dtype=np.float32).tobytes()
    )
    assert audit["sg_byte_identical"] is True
    assert audit["ineligible_iors_byte_identical"] is True
    assert audit["replaced_iors_rows"] == 1
    assert audit["reference_probability_dtype"] == "float32"
    assert len(audit["reference_probability_raw_bytes_sha256"]) == 64


def test_compose_rejects_invalid_replacement_probability() -> None:
    reference = pd.DataFrame(
        {
            "station": ["I-ORS"],
            "year": [2025],
            "layer": [1],
            "time": ["2025-04-01T00:00:00+09:00"],
            "fold": ["q2"],
            "prediction": np.asarray([0], dtype=np.int8),
            "probability": np.asarray([0.1], dtype=np.float32),
        }
    )
    replacements = reference.loc[:, [*KEY_COLUMNS, "fold"]].copy()
    replacements["candidate_prediction"] = np.int8(1)
    replacements["candidate_probability"] = np.float32(np.nan)

    with pytest.raises(ValueError, match="replacement probabilities"):
        compose_incumbent_predictions(
            reference, replacements, replacements.loc[:, [*KEY_COLUMNS, "fold"]]
        )


def test_q2_noop_is_independent_and_byte_identical() -> None:
    reference = pd.DataFrame(
        {
            "station": ["I-ORS", "I-ORS"],
            "year": [2025, 2025],
            "layer": [1, 1],
            "time": [
                "2025-04-01T00:00:00+09:00",
                "2025-07-01T00:00:00+09:00",
            ],
            "fold": ["2025_q2", "2025_q3"],
            "prediction": np.asarray([0, 0], dtype=np.int8),
            "probability": np.asarray([0.125, 0.25], dtype=np.float32),
        }
    )
    expected = independent_expected_replacement_keys(
        reference,
        [0, 1],
        [True, True],
        candidate_folds=["2025_q3"],
    )
    replacements = expected.copy()
    replacements["candidate_prediction"] = np.int8(1)
    replacements["candidate_probability"] = np.float32(0.75)

    output, _ = compose_incumbent_predictions(reference, replacements, expected)

    assert expected["fold"].tolist() == ["2025_q3"]
    assert output.loc[0, "candidate_prediction"] == reference.loc[0, "prediction"]
    assert (
        output.loc[[0], "candidate_probability"].to_numpy().tobytes()
        == reference.loc[[0], "probability"].to_numpy().tobytes()
    )


def test_alternate_one_shot_output_or_status_path_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    canonical = canonical_artifact_paths(
        Path.cwd(),
        contract["artifacts"],
        requested_output_dir=Path(contract["artifacts"]["output_dir"]),
        requested_status_file=Path(contract["artifacts"]["status"]),
    )

    assert canonical.outer_lock == (Path.cwd() / contract["artifacts"]["outer_lock"]).resolve()
    with pytest.raises(ValueError, match="--output-dir"):
        canonical_artifact_paths(
            Path.cwd(),
            contract["artifacts"],
            requested_output_dir=tmp_path,
            requested_status_file=Path(contract["artifacts"]["status"]),
        )
    with pytest.raises(ValueError, match="--status-file"):
        canonical_artifact_paths(
            Path.cwd(),
            contract["artifacts"],
            requested_output_dir=Path(contract["artifacts"]["output_dir"]),
            requested_status_file=tmp_path / "status.json",
        )


def test_threshold_ties_choose_the_higher_value() -> None:
    threshold, prediction, audit = select_inner_threshold(
        [0, 0], [0.1, 0.2], [False, False], [0.4, 0.8]
    )

    assert threshold == 0.8
    assert prediction.tolist() == [0, 0]
    assert audit["tie_break"] == "higher_threshold"


def test_point_gate_is_strict_and_complete() -> None:
    gate_contract = {
        "overall_weighted_f1_delta_min": 0.005,
        "iors_micro_f1_delta_min": 0.010,
        "offset_or_drift_recall_delta_min": 0.05,
        "every_anomaly_type_recall_delta_min": -0.02,
        "normal_fp_day_relative_increase_lt": 0.10,
        "worst_iors_layer_f1_delta_min": -0.01,
        "paired_bootstrap_ci90_lower_gt": 0.0,
        "minimum_improved_folds": 2,
    }
    result = apply_point_residual_gate(
        overall_weighted_f1_delta=0.005,
        iors_micro_f1_delta=0.010,
        anomaly_type_recall_delta={
            "spike": 0.0,
            "noise": 0.0,
            "flatline": 0.0,
            "offset": 0.05,
            "drift": 0.0,
        },
        normal_fp_day_relative_increase=0.099,
        worst_iors_layer_f1_delta=-0.01,
        paired_bootstrap_ci90_lower=0.001,
        improved_folds=2,
        contract=gate_contract,
    )

    assert result["passed"] is True
    failed = apply_point_residual_gate(
        overall_weighted_f1_delta=0.005,
        iors_micro_f1_delta=0.010,
        anomaly_type_recall_delta={
            "spike": 0.0,
            "noise": 0.0,
            "flatline": 0.0,
            "offset": 0.05,
            "drift": 0.0,
        },
        normal_fp_day_relative_increase=0.10,
        worst_iors_layer_f1_delta=-0.01,
        paired_bootstrap_ci90_lower=0.0,
        improved_folds=2,
        contract=gate_contract,
    )
    assert failed["passed"] is False
    assert failed["checks"]["normal_fp_day"] is False
    assert failed["checks"]["paired_bootstrap"] is False


def test_config_has_one_seed_one_model_and_full_external_qc1() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert contract["candidate_model"]["seed"] == 20260821
    assert contract["candidate_model"]["hyperparameter_search"] is False
    assert contract["external_source"]["use_all_eligible_rows"] is True
    assert contract["external_source"]["qc"] == 1
    assert contract["external_source"]["years"] == list(range(2014, 2024))
    assert contract["external_q50_model"]["alpha"] == 0.5
    assert contract["non_virgin_follow_up"]["disclosed"] is True
    amendment = contract["amendments"][0]
    assert amendment["aborted_attempt"]["outer_truth_accessed"] is False
    assert amendment["label_blind_support_evidence"]["2025_q2"]["outer_train"] == 0
    assert amendment["fold_policy"]["incumbent_noop_folds"] == ["2025_q2"]
    assert amendment["fold_policy"]["candidate_folds"] == ["2025_q3", "2025_q4"]
    assert contract["promotion_gate"]["minimum_improved_folds"] == 2
