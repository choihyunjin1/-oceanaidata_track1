from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.features import FeatureBundle
from p1_qc.matched_filter import MATCHED_FILTER_FEATURES
from p1_qc.matched_filter_experiment import blocks_from_contract, load_and_validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/experiments/p1_offset_drift_matched_filter_v1.json"


def test_real_contract_is_inner_only_and_hash_locked() -> None:
    payload = load_and_validate_contract(CONTRACT, project_root=ROOT)
    assert payload["authorization"]["inner_comparison"] is True
    assert payload["authorization"]["outer_one_shot"] is False
    blocks = blocks_from_contract(payload)
    assert tuple(block.name for block in blocks) == ("I1", "I2", "I3")


@pytest.mark.parametrize("field", ["outer_one_shot", "test_prediction", "submission"])
def test_expansive_authorization_fails_closed(tmp_path: Path, field: str) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["authorization"][field] = True
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_and_validate_contract(mutated, project_root=ROOT)


def test_added_features_are_exactly_four() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert tuple(payload["single_change"]["added_features"]) == MATCHED_FILTER_FEATURES


def test_bundle_difference_contract_fixture() -> None:
    base_frame = pd.DataFrame({"base": np.array([1, 2], dtype=np.float32)})
    candidate_frame = base_frame.copy()
    for position, column in enumerate(MATCHED_FILTER_FEATURES):
        candidate_frame[column] = np.float32(position)
    baseline = FeatureBundle(base_frame, ("base",), ())
    candidate = FeatureBundle(candidate_frame, ("base", *MATCHED_FILTER_FEATURES), ())
    assert candidate.feature_columns == (*baseline.feature_columns, *MATCHED_FILTER_FEATURES)


def test_runner_source_contains_no_outer_execution_surface() -> None:
    source = (ROOT / "scripts/run_matched_filter_inner.py").read_text(encoding="utf-8")
    assert "outer_folds(" not in source
    assert "predict_submission" not in source
    assert "write_submission" not in source
