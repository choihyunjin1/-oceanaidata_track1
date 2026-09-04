from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.kma_calibrated_deployment import (
    DEPLOYMENT_ALPHA,
    EXPECTED_FULL_TRAIN_ANCHORS,
    EXPECTED_GENERATED_META,
    EXPECTED_REUSED_META,
    KMADeploymentError,
    build_full_ridge_frame,
    calibrators_from_payload,
    calibrators_to_payload,
    combine_full_training_meta,
    count_byte_exact_noop_lines,
    extract_relative_test_history,
    feature_columns_sha256,
    load_deployment_config,
    render_submission_preserving_noop_lines,
    validate_candidate_submission,
)
from p3_wave.kma_calibrated_longlead_blend import RidgeAffineCalibrator
from p3_wave.kma_source_meta import (
    LEADS,
    META_COLUMNS,
    compact_source_feature_columns,
    summarize_common_history,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p3_kma_calibrated_longlead_deployment_v1.json"
RUNNER = ROOT / "scripts/run_p3_kma_calibrated_longlead_deployment_v1.py"


def _case(case_id: str = "C0001", station: str = "G-ORS") -> pd.DataFrame:
    steps = np.arange(-2880, 1, 10, dtype=np.int64)
    wave_slot = steps % 20 == 0
    index = np.arange(len(steps), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "case_id": case_id,
            "station": station,
            "step_minute": steps,
            "hs": np.where(wave_slot, 1.5 + index / 1000.0, np.nan),
            "hmax": np.where(wave_slot, 2.5 + index / 1000.0, np.nan),
            "wvdir": np.where(wave_slot, 120.0, np.nan),
            "wspd": 5.0 + index / 1000.0,
            "gust": 7.0 + index / 1000.0,
            "wdir": 140.0,
            "airt": 20.0,
            "relh": 75.0,
            "caph": 1012.0,
        }
    )
    return frame


def _submission_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for case_number in range(1, 201):
        case_id = f"C{case_number:04d}"
        station = ("G-ORS", "I-ORS", "S-ORS")[(case_number - 1) % 3]
        for lead in LEADS:
            rows.append((case_id, station, lead))
    test_index = pd.DataFrame(rows, columns=["case_id", "station", "lead_h"])
    incumbent = test_index.copy()
    incumbent["hs_pred"] = 2.0
    candidate = incumbent.copy()
    candidate.loc[candidate["lead_h"].isin([18, 24]), "hs_pred"] = 2.1
    return test_index, incumbent, candidate


def _calibrator(lead: int) -> RidgeAffineCalibrator:
    return RidgeAffineCalibrator(
        lead_h=lead,
        ridge_alpha=10.0,
        fit_intercept=False,
        solver="cholesky",
        design_columns=(
            "source_residual",
            "station_G-ORS",
            "station_I-ORS",
            "station_S-ORS",
        ),
        coefficients=(0.5, 0.1, -0.1, 0.0),
        fit_rows=EXPECTED_FULL_TRAIN_ANCHORS,
    )


def test_config_freezes_two_per_lead_ridges_and_secondary_status() -> None:
    config = load_deployment_config(CONFIG)
    assert config["full_refit"]["model_count"] == 2
    assert config["full_refit"]["one_per_active_lead"] is True
    assert config["full_refit"]["single_refit_generation"] is True
    assert config["full_refit"]["active_leads"] == [18, 24]
    assert config["full_refit"]["hyperparameter_grid_size"] == 0
    assert config["test_inference"]["deployment_alpha"] == DEPLOYMENT_ALPHA
    assert config["objective"]["promotion_claimed"] is False
    assert config["execution"]["submission_upload_allowed"] is False
    assert config["prohibitions"]["pooled_shared_18h_24h_ridge"] is True


def test_feature_schema_is_frozen_and_excludes_period_identity_and_calendar() -> None:
    config = load_deployment_config(CONFIG)
    columns = compact_source_feature_columns()
    assert len(columns) == 447
    assert feature_columns_sha256() == config["sealed_source_reuse"]["feature_columns_sha256"]
    lowered = [column.lower() for column in columns]
    assert not any(column.startswith("tp_") for column in lowered)
    assert not any("station" in column for column in lowered)
    assert not any("time" in column or "date" in column for column in lowered)
    assert not any("valid" in column or "missing" in column for column in lowered)


def test_relative_history_is_causal_and_uses_previous_wave_on_half_slots() -> None:
    case = _case()
    history = extract_relative_test_history(case)
    assert len(history) == 97
    # Query -2850 is a structural wave gap and must use -2860, never -2840.
    expected_previous = case.loc[case["step_minute"].eq(-2860), "hs"].iloc[0]
    future = case.loc[case["step_minute"].eq(-2840), "hs"].iloc[0]
    assert history.loc[1, "hs"] == expected_previous
    assert history.loc[1, "hs"] != future
    summary = summarize_common_history(history)
    assert list(summary) == list(compact_source_feature_columns())


def test_relative_history_rejects_cross_case_or_bad_grid() -> None:
    crossed = _case()
    crossed.loc[crossed.index[-1], "case_id"] = "C9999"
    with pytest.raises(KMADeploymentError, match="crossed"):
        extract_relative_test_history(crossed)
    missing = _case().iloc[:-1]
    with pytest.raises(KMADeploymentError, match="relative grid"):
        extract_relative_test_history(missing)


def test_full_meta_composition_is_complete_disjoint_and_finite() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(EXPECTED_FULL_TRAIN_ANCHORS, dtype=np.int64),
            "station": "G-ORS",
            "current_hs": 2.0,
            "target_18": 2.1,
            "target_24": 2.2,
        }
    )
    columns = ["anchor_id", *META_COLUMNS]
    reused = pd.DataFrame(np.full((EXPECTED_REUSED_META, len(columns)), 2.0), columns=columns)
    reused["anchor_id"] = np.arange(EXPECTED_REUSED_META, dtype=np.int64)
    generated = pd.DataFrame(np.full((EXPECTED_GENERATED_META, len(columns)), 2.1), columns=columns)
    generated["anchor_id"] = np.arange(
        EXPECTED_REUSED_META, EXPECTED_FULL_TRAIN_ANCHORS, dtype=np.int64
    )
    combined = combine_full_training_meta(anchors, reused, generated)
    assert len(combined) == EXPECTED_FULL_TRAIN_ANCHORS
    ridge = build_full_ridge_frame(anchors, combined)
    assert len(ridge) == EXPECTED_FULL_TRAIN_ANCHORS * 2
    assert ridge.groupby("lead_h", observed=True).size().to_dict() == {
        18: EXPECTED_FULL_TRAIN_ANCHORS,
        24: EXPECTED_FULL_TRAIN_ANCHORS,
    }


def test_saved_two_model_payload_roundtrips_exactly() -> None:
    calibrators = {18: _calibrator(18), 24: _calibrator(24)}
    payload = calibrators_to_payload(calibrators)
    restored = calibrators_from_payload(json.loads(json.dumps(payload)))
    assert restored == calibrators
    assert payload["model_count"] == 2
    assert payload["one_per_active_lead"] is True


def test_candidate_validator_enforces_short_and_unsupported_noops() -> None:
    test_index, incumbent, candidate = _submission_frames()
    supported = set(test_index["case_id"].unique())
    receipt = validate_candidate_submission(
        candidate, test_index, incumbent, supported_cases=supported
    )
    assert receipt["rows"] == 1200
    assert receipt["modified_active_rows"] == 400
    broken = candidate.copy()
    broken.loc[broken["lead_h"].eq(3).idxmax(), "hs_pred"] = 9.0
    with pytest.raises(KMADeploymentError, match="short-lead"):
        validate_candidate_submission(broken, test_index, incumbent, supported_cases=supported)
    unsupported = candidate.copy()
    first_case = test_index["case_id"].iloc[0]
    with pytest.raises(KMADeploymentError, match="unsupported"):
        validate_candidate_submission(
            unsupported,
            test_index,
            incumbent,
            supported_cases=supported - {first_case},
        )


def test_renderer_copies_every_noop_line_byte_exact() -> None:
    test_index = pd.DataFrame(
        [("C0001", "G-ORS", lead) for lead in LEADS],
        columns=["case_id", "station", "lead_h"],
    )
    candidate = test_index.copy()
    candidate["hs_pred"] = [1.0, 1.1, 1.2, 1.3, 1.8, 2.4]
    incumbent = (
        b"case_id,station,lead_h,hs_pred\r\n"
        b"C0001,G-ORS,3,0.900000\r\n"
        b"C0001,G-ORS,6,0.910000\r\n"
        b"C0001,G-ORS,9,0.920000\r\n"
        b"C0001,G-ORS,12,0.930000\r\n"
        b"C0001,G-ORS,18,0.940000\r\n"
        b"C0001,G-ORS,24,0.950000\r\n"
    )
    rendered = render_submission_preserving_noop_lines(
        incumbent, candidate, supported_cases={"C0001"}
    )
    assert rendered.splitlines(keepends=True)[1:5] == incumbent.splitlines(keepends=True)[1:5]
    assert (
        count_byte_exact_noop_lines(incumbent, rendered, candidate, supported_cases={"C0001"}) == 4
    )
    unsupported = render_submission_preserving_noop_lines(
        incumbent, candidate, supported_cases=set()
    )
    assert unsupported == incumbent


def test_runner_has_no_override_upload_or_absolute_test_mapping_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "--output-dir" not in source
    assert "--experiment-config" not in source
    assert "upload" not in source.lower() or "upload_count" in source
    assert "test_absolute" not in source
    assert "source_model.fit" not in source
    assert "fit_ridge_pair" in source
    assert "test_context" in source
    assert (
        "anchor_time"
        not in source[
            source.index("def _predict_test_source") : source.index("def _test_source_long")
        ]
    )


def test_config_paths_do_not_overwrite_incumbent() -> None:
    config = load_deployment_config(CONFIG)
    incumbent = Path(config["frozen_inputs"]["incumbent_submission"]["path"])
    candidate = Path(config["artifacts"]["submission_path"])
    assert incumbent != candidate
    assert candidate.as_posix().endswith(
        "submissions/p3_kma_calibrated_longlead_secondary_v1/submission.csv"
    )


def test_runner_atomic_parquet_roundtrip(tmp_path: Path) -> None:
    specification = importlib.util.spec_from_file_location("deployment_runner_test", RUNNER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    path = tmp_path / "sealed.parquet"
    expected = pd.DataFrame({"case_id": ["C0001"], "value": [1.25]})
    module._atomic_parquet(path, expected)
    assert pd.read_parquet(path).equals(expected)
