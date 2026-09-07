"""Synthetic hash-drift acceptance must not weaken current-run integrity."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p2_smooth7_portable_20260907 as old  # noqa: E402
import p2_smooth7_portable_20260907_v2 as fixed  # noqa: E402


def fixture_qa(tmp_path):
    source = tmp_path / "base.csv"
    source.write_bytes(b"synthetic bytes, not competition predictions\n")
    (tmp_path / "04_logs").mkdir()
    qa = tmp_path / "04_logs/independent-qa.json"
    qa.write_text(json.dumps({"status": "PASS", "answer_sha256": fixed.sha(source),
                              "checks": {"models": True, "keys": True, "replay": True}}))
    return source, qa


def test_historic_mismatch_records_actual_and_continues(tmp_path):
    source, _ = fixture_qa(tmp_path)
    proof = fixed.base_provenance(tmp_path, source)
    assert proof["base_exact_reconstruction"] is False
    assert proof["base_sha256"] == fixed.sha(source)
    assert proof["expected_base_sha256"] == fixed.BASE


def test_historic_match_is_recorded(tmp_path, monkeypatch):
    source, _ = fixture_qa(tmp_path)
    monkeypatch.setattr(fixed, "BASE", fixed.sha(source))
    assert fixed.base_provenance(tmp_path, source)["base_exact_reconstruction"]


def test_changed_after_current_qa_rejected(tmp_path):
    source, _ = fixture_qa(tmp_path)
    source.write_bytes(b"changed after independent QA")
    with pytest.raises(ValueError, match="changed after QA"):
        fixed.base_provenance(tmp_path, source)


@pytest.mark.parametrize("status,checks", [("FAIL", {"models": True}),
                                          ("PASS", {"models": False}), ("PASS", {})])
def test_failed_or_empty_integrity_rejected(tmp_path, status, checks):
    source, qa = fixture_qa(tmp_path)
    qa.write_text(json.dumps({"status": status, "answer_sha256": fixed.sha(source), "checks": checks}))
    with pytest.raises(ValueError):
        fixed.base_provenance(tmp_path, source)


def test_missing_current_qa_rejected(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_bytes(b"synthetic")
    with pytest.raises(FileNotFoundError):
        fixed.base_provenance(tmp_path, source)


def test_smoothing_numerically_unchanged():
    frame = pd.DataFrame({"station": ["S"] * 9, "layer": [2] * 9,
                          "time": pd.date_range("2025-01-01", periods=9, freq="10min", tz="UTC")})
    values = np.arange(9, dtype=float) ** 2
    np.testing.assert_array_equal(fixed.smooth7(frame, values), old.smooth7(frame, values))
