from __future__ import annotations

import json
from pathlib import Path

import pytest

from p1_qc.ts_matched_filter_experiment import load_and_validate_ts_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/experiments/p1_ts_matched_filter_v2.json"


def test_real_adaptive_contract_is_honest_and_inner_only() -> None:
    payload = load_and_validate_ts_contract(CONTRACT, project_root=ROOT)
    assert payload["adaptive_provenance"]["previous_inner_labels_exposed"] is True
    assert payload["adaptive_provenance"]["claim_independent_validation"] is False
    assert payload["authorization"]["outer_one_shot"] is False


@pytest.mark.parametrize("field", ["outer_one_shot", "test_prediction", "submission"])
def test_expansive_authorization_fails_closed(tmp_path: Path, field: str) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["authorization"][field] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        load_and_validate_ts_contract(path, project_root=ROOT)


def test_independence_claim_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["adaptive_provenance"]["claim_independent_validation"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independence"):
        load_and_validate_ts_contract(path, project_root=ROOT)


def test_runner_has_no_outer_or_submission_surface() -> None:
    source = (ROOT / "scripts/run_ts_matched_filter_inner.py").read_text(encoding="utf-8")
    assert "outer_folds(" not in source
    assert "write_submission" not in source
    assert "predict_submission" not in source
