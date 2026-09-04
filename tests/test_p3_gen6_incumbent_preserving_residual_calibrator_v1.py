from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave import gen6_incumbent_preserving_residual_calibrator as gen6

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / gen6.CONFIG_RELATIVE
RUNNER = ROOT / "scripts/run_p3_gen6_incumbent_preserving_residual_calibrator_v1.py"


def _frame(cases: int = 18) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stations = ("G-ORS", "I-ORS", "S-ORS")
    for case in range(cases):
        station = stations[case % len(stations)]
        current = 1.5 + 0.02 * case
        for lead in gen6.LEADS:
            incumbent = current + 0.001 * lead
            rows.append(
                {
                    "anchor_id": case,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "persistence": current,
                    "incumbent_prediction": incumbent,
                    "target_hs": incumbent - 0.04,
                }
            )
    return pd.DataFrame(rows)


def test_config_is_one_hypothesis_static_fail_closed() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert len(raw) == gen6.EXPECTED_CONFIG_BYTES
    assert gen6.sha256_file(CONFIG) == gen6.EXPECTED_CONFIG_SHA256
    assert config["hypothesis_count"] == 1
    assert config["status"].endswith("NO_TEST_NO_UPLOAD")
    assert config["official_promotion_allowed"] is False
    assert config["candidate_or_test_prediction_allowed"] is False
    assert config["registry_append_allowed"] is False
    assert config["upload_allowed"] is False
    assert all(value == 0 for value in config["static_counters"].values())
    assert config["execution_policy"]["run_now_authorized"] is False


def test_config_contains_no_search_or_multiple_hypothesis_escape() -> None:
    config = json.loads(CONFIG.read_bytes())
    calibrator = config["calibrator"]
    search_keys = (
        "hyperparameter_search_count",
        "alpha_search_count",
        "threshold_search_count",
        "seed_search_count",
        "blend_weight_search_count",
        "router_search_count",
    )
    assert all(calibrator[key] == 0 for key in search_keys)
    assert calibrator["ridge_alpha"] == 32.0
    assert calibrator["maximum_absolute_correction_m"] == 0.12


def test_failure_diagnosis_requires_incumbent_preservation() -> None:
    diagnosis = json.loads(CONFIG.read_bytes())["failure_diagnosis"]
    assert diagnosis["gen2"]["full_delta_candidate_minus_incumbent_m"] > 0.03
    assert diagnosis["gen3"]["full_delta_candidate_minus_incumbent_m"] > 0.04
    assert diagnosis["gen4"]["full_delta_candidate_minus_incumbent_m"] > 0.04
    assert diagnosis["gen5r4"]["full_delta_candidate_minus_incumbent_m"] > 0.06
    assert min(
        diagnosis["gen5r4"]["prefix_deltas_candidate_minus_incumbent_m"]
    ) > 0.0


def test_failed_inner_gate_identity_is_byte_exact() -> None:
    incumbent = np.ascontiguousarray(
        np.array([0.0, -0.0, 1.25, 2.5, 30.0], dtype=np.float64)
    )
    result = gen6.apply_identity_or_bounded_correction(incumbent, None, enabled=False)
    assert result is not incumbent
    assert result.dtype == np.dtype("float64")
    assert result.flags.c_contiguous
    assert result.tobytes(order="C") == incumbent.tobytes(order="C")


def test_enabled_correction_is_bounded_and_range_clipped() -> None:
    incumbent = np.ascontiguousarray(np.array([0.02, 2.0, 29.95], dtype=np.float64))
    correction = np.array([-0.12, 0.12, 0.12], dtype=np.float64)
    result = gen6.apply_identity_or_bounded_correction(
        incumbent, correction, enabled=True
    )
    assert np.array_equal(result, np.array([0.0, 2.12, 30.0], dtype=np.float64))
    with pytest.raises(gen6.ContractError, match="exceeds"):
        gen6.apply_identity_or_bounded_correction(
            incumbent, np.array([0.0, 0.0, 0.1200001]), enabled=True
        )


def test_design_matrix_never_uses_target_values() -> None:
    left = _frame()
    right = left.copy()
    right["target_hs"] = right["target_hs"] + 1000.0
    left_design, left_mean, left_scale, left_names = gen6._design_matrix(left)
    right_design, right_mean, right_scale, right_names = gen6._design_matrix(right)
    assert np.array_equal(left_design, right_design)
    assert np.array_equal(left_mean, right_mean)
    assert np.array_equal(left_scale, right_scale)
    assert left_names == right_names


def test_fixed_ridge_recovers_small_bias_without_search() -> None:
    frame = _frame()
    model = gen6.fit_residual_calibrator(frame)
    correction = gen6.predict_bounded_correction(model, frame)
    assert len(model.coefficients) == len(model.feature_names)
    assert np.isfinite(model.coefficients).all()
    assert np.max(np.abs(correction)) <= gen6.CORRECTION_LIMIT_M
    assert float(np.mean(correction)) < -0.01


def test_inner_gate_failure_can_drive_exact_identity() -> None:
    frame = _frame()
    frame[gen6.CANDIDATE_COLUMN] = frame["incumbent_prediction"]
    gate = gen6.evaluate_inner_gate(frame, seed=20260823)
    assert gate["passed"] is False
    incumbent = np.ascontiguousarray(
        frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    )
    result = gen6.apply_identity_or_bounded_correction(incumbent, None, enabled=False)
    assert result.tobytes() == incumbent.tobytes()


def test_current_central_v9_anchor_is_exact() -> None:
    config = json.loads(CONFIG.read_bytes())
    verified = gen6.verify_central_ledger(ROOT, config)
    assert verified == {
        "path": "artifacts/meaningful_score_goal_v9/registry.jsonl",
        "bytes": 13030,
        "sha256": "0cc77ee79856c168dabc40d9a5357ead515ce6a34af0bd8e170f6b6be4833afc",
        "physical_event_lines": 2,
        "global_head_seq": 4,
        "head_event_sha256": (
            "db78fc78d8097eddde827253b1627c18cd990cb1bfb3423151a64b650700ba66"
        ),
        "official_uploads_through_anchor": 0,
    }


def test_runner_defaults_to_check_only() -> None:
    spec = importlib.util.spec_from_file_location("p3_gen6_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parser().parse_args([])
    assert args.mode == "check-only"


def test_canonical_static_preflight_is_read_only_when_data_available() -> None:
    value = os.environ.get("P3_DATA_DIR")
    if not value:
        pytest.skip("P3_DATA_DIR is not configured")
    config = json.loads(CONFIG.read_bytes())
    output = ROOT / config["canonical_paths"]["output"]
    control = ROOT / config["canonical_paths"]["control"]
    assert not output.exists()
    assert not control.exists()
    before = {
        "ledger": gen6.file_pin(
            ROOT / config["canonical_paths"]["central_v9_ledger"], root=ROOT
        ),
        "config": gen6.file_pin(CONFIG, root=ROOT),
    }
    report = gen6.static_preflight(ROOT, Path(value), requested_config=CONFIG)
    after = {
        "ledger": gen6.file_pin(
            ROOT / config["canonical_paths"]["central_v9_ledger"], root=ROOT
        ),
        "config": gen6.file_pin(CONFIG, root=ROOT),
    }
    assert report["status"] == "STATIC_PREFLIGHT_PASS_NO_WRITES"
    assert before == after
    assert report["fits"] == report["predictions"] == report["scores"] == 0
    assert report["test_value_reads"] == report["candidate_files_created"] == 0
    assert report["registry_appends"] == report["uploads"] == 0
    assert not output.exists()
    assert not control.exists()
