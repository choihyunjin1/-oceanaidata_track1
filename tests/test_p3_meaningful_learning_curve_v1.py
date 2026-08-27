from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.meaningful_learning_curve import (
    CRITICAL_SLICE_KEYS,
    HYPOTHESES,
    PREFIX_FRACTIONS,
    central_evidence,
    chronological_prefix_ids,
    evaluate_hypothesis_gate,
    hypothesis_predictions,
    next_structural_generation,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p3_meaningful_learning_curve_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_meaningful_curve_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _deep_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _point(delta: float, *, upper: float = -0.001) -> dict[str, object]:
    return {
        "incumbent_rmse_m": 0.78,
        "challenger_rmse_m": 0.78 + delta,
        "delta_candidate_minus_incumbent_m": delta,
        "delta_ci90_m": [upper - 0.01, upper],
        "fold_deltas_candidate_minus_incumbent_m": {
            "2024_h2_storm": -0.04,
            "winter_transition": -0.03,
            "2025_h1": 0.001,
        },
        "slice_deltas_candidate_minus_incumbent_m": {key: 0.001 for key in CRITICAL_SLICE_KEYS},
        "improved_fold_count": 2,
        "worst_critical_slice_regression_m": 0.001,
        "incumbent_seed_metrics": [0.79, 0.78, 0.77],
        "challenger_seed_metrics": [0.75, 0.74, 0.73],
    }


def test_config_byte_and_deep_pins_are_compiled() -> None:
    path = ROOT / runner.CANONICAL_CONFIG_RELATIVE
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.EXPECTED_CONFIG_SHA256
    assert _deep_sha(parsed) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert tuple(item["id"] for item in parsed["hypotheses"]) == HYPOTHESES
    assert parsed["validation"]["training_prefix_fractions"] == list(PREFIX_FRACTIONS)
    assert parsed["model"]["seed_replicates"] == [20260816, 20260817, 20260818]


def test_canonical_authorization_and_arbitrary_paths_fail_closed(tmp_path: Path) -> None:
    paths = runner._canonical_paths(ROOT)
    config, authorized = runner.authorize_entry(
        root=ROOT,
        requested_config=paths["config"],
        requested_cache=paths["cache"],
        requested_output=paths["output"],
    )
    assert config["experiment_id"] == "p3_meaningful_learning_curve_v1"
    assert authorized == paths
    copied = tmp_path / "copied.json"
    copied.write_bytes(paths["config"].read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=ROOT,
            requested_config=copied,
            requested_cache=paths["cache"],
            requested_output=paths["output"],
        )
    with pytest.raises(PermissionError, match="non-canonical output"):
        runner.authorize_entry(
            root=ROOT,
            requested_config=paths["config"],
            requested_cache=paths["cache"],
            requested_output=tmp_path / "new-output",
        )


def test_prefixes_are_chronological_nested_and_exact_counts() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(20, dtype=np.int64),
            "anchor_time": pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
            "station": ["G-ORS", "I-ORS"] * 10,
        }
    )
    train_ids = np.arange(20, dtype=np.int64)
    prefixes = [chronological_prefix_ids(anchors, train_ids, value) for value in PREFIX_FRACTIONS]
    assert [len(value) for value in prefixes] == [8, 11, 14, 17, 20]
    for left, right in zip(prefixes, prefixes[1:], strict=False):
        assert np.isin(left, right).all()
    assert np.array_equal(prefixes[-1], train_ids)


def test_hypotheses_are_structural_and_splice_is_fixed() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [3, 6, 9, 12, 18, 24],
            "single_prediction": [1, 2, 3, 4, 5, 6],
            "multi_prediction": [11, 12, 13, 14, 15, 16],
            "equal_prediction": [6, 7, 8, 9, 10, 11],
        }
    )
    predictions = hypothesis_predictions(frame)
    assert tuple(predictions) == HYPOTHESES
    assert predictions["single_horizon_residual_head"].tolist() == [1, 2, 3, 4, 5, 6]
    assert predictions["multi_trajectory_residual_head"].tolist() == [11, 12, 13, 14, 15, 16]
    assert predictions["fixed_horizon_splice"].tolist() == [11, 12, 13, 14, 5, 6]


def test_meaningful_gate_passes_only_complete_strict_contract() -> None:
    points = {fraction: _point(-0.04) for fraction in PREFIX_FRACTIONS}
    checks = {"leakage": True}
    reproducibility = {"exact_refit": True}
    passed = evaluate_hypothesis_gate(
        points, leakage_checks=checks, reproducibility_checks=reproducibility
    )
    assert passed["passed"] is True
    failed_points = dict(points)
    failed_points[1.0] = _point(-0.029)
    failed = evaluate_hypothesis_gate(
        failed_points, leakage_checks=checks, reproducibility_checks=reproducibility
    )
    assert failed["passed"] is False
    assert failed["checks"]["full_delta_at_most_minus_0p030m"] is False
    failed_reproduction = evaluate_hypothesis_gate(
        points,
        leakage_checks=checks,
        reproducibility_checks={"reference_seed_exact": False},
    )
    assert failed_reproduction["passed"] is False


def test_central_evidence_has_exact_contract_keys() -> None:
    points = {fraction: _point(-0.04) for fraction in PREFIX_FRACTIONS}
    evidence = central_evidence(
        points,
        leakage_checks={"station_global_78h": True},
        reproducibility_checks={"exact_refit": True},
    )
    assert evidence["problem"] == "P3"
    assert [point["fraction"] for point in evidence["points"]] == list(PREFIX_FRACTIONS)
    assert len(evidence["fold_deltas_candidate_minus_incumbent"]) == 3
    assert set(evidence["slice_deltas_candidate_minus_incumbent"]) == set(CRITICAL_SLICE_KEYS)


def test_no_pass_diagnoses_exactly_one_non_micro_tweak_generation() -> None:
    points = {fraction: _point(0.01, upper=0.02) for fraction in PREFIX_FRACTIONS}
    diagnosis = next_structural_generation(points)
    assert diagnosis["count"] == 1
    serialized = json.dumps(diagnosis).lower()
    assert "sequence" in serialized
    assert "not coefficients" in serialized
    assert "shrink" in serialized


def test_cli_exposes_no_config_cache_or_output_override() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--config"' not in source
    assert 'add_argument("--cache"' not in source
    assert 'add_argument("--output"' not in source
    assert "acquire_persistent_attempt_lock" in source
    assert "safe_new_stage_path" in source
    assert '"official_upload_count": 0' in source
    assert "submission_upload" in (ROOT / runner.CANONICAL_CONFIG_RELATIVE).read_text(
        encoding="utf-8"
    )


def test_append_only_targets_are_absent_before_one_shot() -> None:
    paths = runner._canonical_paths(ROOT)
    assert not paths["output"].exists()
    assert not paths["lock"].exists()
