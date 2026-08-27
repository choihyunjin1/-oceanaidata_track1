from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.features import FeatureBundle
from p1_qc.ring_inner_experiment import (
    FIXED_INNER_BLOCKS,
    RING_FEATURES,
    RingInnerContractError,
    assert_safe_audit_path,
    assert_synthetic_holdout_separation,
    audit_runner_ast,
    deny_model_execution,
    fixed_block_indices,
    label_blind_scoped_frame,
    load_ring_inner_contract,
    validate_ring_inner_contract,
    validate_two_arm_bundle,
    verify_coverage_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "experiments" / "p1_ring_residual_inner_only.json"
RUNNER = ROOT / "scripts" / "run_ring_inner_experiment.py"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["I-ORS"] * 8,
            "year": [2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025],
            "layer": [1] * 8,
            "time": [
                "2024-06-23T23:50:00+09:00",
                "2024-07-01T00:00:00+09:00",
                "2024-08-31T23:50:00+09:00",
                "2024-09-23T23:50:00+09:00",
                "2024-10-01T00:00:00+09:00",
                "2025-01-01T00:00:00+09:00",
                "2025-02-28T23:50:00+09:00",
                "2025-04-01T00:00:00+09:00",
            ],
            "temp": np.arange(8, dtype=float),
            "psal": np.arange(8, dtype=float),
            "depth": np.ones(8),
            "label": [0, 0, 1, 0, 1, 0, 1, 0],
            "anomaly_type": ["", "", "offset", "", "drift", "", "noise", ""],
        }
    )


def test_contract_is_permanent_no_go_with_one_fixed_configuration() -> None:
    payload = load_ring_inner_contract(CONTRACT)
    receipt = validate_ring_inner_contract(payload)
    assert receipt["status"] == "valid_permanent_no_go"
    assert receipt["model_fit_allowed"] is False
    assert payload["arms"] == ["frozen_features", "frozen_plus_4_ring"]
    assert payload["model_contract"]["n_estimators"] == 700
    assert payload["model_contract"]["hyperparameter_search"] is False
    assert len(payload["features"]) == 4


def test_corrected_coverage_artifact_is_hash_bound_and_aggregate_only() -> None:
    artifact = ROOT / "artifacts" / "ring_coverage_audit_20260813" / "result.json"
    if not artifact.is_file():
        pytest.skip("ignored local corrected coverage evidence is unavailable")
    receipt = verify_coverage_artifact(artifact)
    assert receipt["both_flanks_coverage"] == 0.056286277224559346
    assert receipt["decision"] == "permanent_no_go"


def test_cutoff_and_post_cutoff_label_flip_are_invariant() -> None:
    source = _frame()
    first = label_blind_scoped_frame(source)
    changed = source.copy()
    changed.loc[changed.index[-1], ["label", "anomaly_type"]] = [1, "spike"]
    second = label_blind_scoped_frame(changed)
    pd.testing.assert_frame_equal(first, second)
    assert "label" not in first and "anomaly_type" not in first
    assert pd.to_datetime(first["time"], utc=True).max() <= pd.Timestamp(
        "2025-03-24T23:50:00+09:00"
    ).tz_convert("UTC")


def test_fixed_blocks_have_seven_day_gap_and_i3_is_locked() -> None:
    scoped = label_blind_scoped_frame(_frame())
    for block in FIXED_INNER_BLOCKS:
        assert block.validation_start - block.train_end >= pd.Timedelta(days=7)
    fit, validation = fixed_block_indices(scoped, "I3")
    assert not np.intersect1d(fit, validation).size
    assert FIXED_INNER_BLOCKS[-1].role == "locked_confirmation"
    payload = load_ring_inner_contract(CONTRACT)
    mutated = copy.deepcopy(payload)
    mutated["inner_blocks"][-1]["role"] = "development"
    with pytest.raises(RingInnerContractError, match="inner blocks"):
        validate_ring_inner_contract(mutated)


def test_exactly_four_ring_columns_are_the_only_arm_change() -> None:
    base_frame = pd.DataFrame({"base": [1.0, 2.0]})
    candidate_frame = base_frame.copy()
    for offset, name in enumerate(RING_FEATURES):
        candidate_frame[name] = [float(offset), float(offset + 1)]
    base = FeatureBundle(base_frame, ("base",), ())
    candidate = FeatureBundle(candidate_frame, ("base", *RING_FEATURES), ())
    validate_two_arm_bundle(base, candidate)
    bad = FeatureBundle(candidate_frame, ("base", *RING_FEATURES[:-1]), ())
    with pytest.raises(RingInnerContractError, match="exactly"):
        validate_two_arm_bundle(base, bad)


def test_synthetic_sources_must_be_held_out_from_fitting() -> None:
    assert_synthetic_holdout_separation([0, 1, 2], [3, 4])
    with pytest.raises(RingInnerContractError, match="overlap"):
        assert_synthetic_holdout_separation([0, 1, 2], [2, 3])


@pytest.mark.parametrize(
    "unsafe",
    [
        "artifacts/runs/candidate.parquet",
        "artifacts/prior_oof.parquet",
        "reports/failure_analysis.json",
        "artifacts/model_metrics.json",
        "submissions/candidate.csv",
    ],
)
def test_outer_or_evaluation_paths_fail_closed(unsafe: str) -> None:
    with pytest.raises(RingInnerContractError, match="unsafe"):
        assert_safe_audit_path(unsafe)


def test_runner_ast_has_no_model_or_evaluation_surface() -> None:
    audit_runner_ast(RUNNER)
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "p1_qc.pipeline" not in imported
    assert "p1_qc.metrics" not in imported
    with pytest.raises(RingInnerContractError, match="permanently"):
        deny_model_execution(execution=True, authorization=True)


def test_any_attempt_to_enable_execution_fails_contract_validation() -> None:
    payload = load_ring_inner_contract(CONTRACT)
    payload["authorization"]["execution"] = True
    payload["execution_gates"]["model_fit_allowed"] = True
    with pytest.raises(RingInnerContractError):
        validate_ring_inner_contract(payload)
