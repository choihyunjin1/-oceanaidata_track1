from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qa_parallel_score_repair_20260905_v1.py"
SPEC = importlib.util.spec_from_file_location("score_repair_qa", SCRIPT)
QA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QA)


def arrays():
    return {
        "key": np.array(["a", "b", "c", "d"]),
        "fold": np.array(["one", "two", "two", "two"]),
        "truth": np.array([1, 0, 1, 0]),
        "reference": np.array([1, 1, 0, 0]),
        "prediction": np.array([1, 0, 1, 0]),
    }


def test_binary_changes_and_micro_f1():
    result = QA.audit_arrays(arrays(), "P1")
    assert result["reference"]["f1"] == 0.5
    assert result["candidate"]["f1"] == 1.0
    assert result["changes"] == {"added_tp": 1, "added_fp": 0, "removed_tp": 0, "removed_fp": 1}


def test_rmse_uses_sum_squares_not_mean_fold_rmse():
    value = arrays()
    value["truth"] = np.zeros(4)
    value["reference"] = np.array([4.0, 0, 0, 0])
    value["prediction"] = np.ones(4)
    result = QA.audit_arrays(value, "P3")
    assert result["reference"]["rmse"] == 2.0
    assert result["candidate"]["rmse"] == 1.0
    assert result["candidate_minus_reference"] == -1.0


@pytest.mark.parametrize("fault", ["duplicate", "nan", "shape", "missing", "object"])
def test_invalid_archives_fail_closed(fault):
    value = arrays()
    if fault == "duplicate":
        value["key"][1] = "a"
    elif fault == "nan":
        value["prediction"] = np.array([1, 0, np.nan, 0])
    elif fault == "shape":
        value["prediction"] = np.ones((2, 2))
    elif fault == "missing":
        del value["truth"]
    else:
        value["key"] = value["key"].astype(object)
    with pytest.raises(ValueError):
        QA.audit_arrays(value, "P2")


def test_missing_run_is_pending_not_pass(tmp_path):
    assert QA.audit_run("P1", tmp_path)["status"] == "PENDING"


def test_followup_is_separate_and_pending_until_available(tmp_path):
    result = QA.audit_run("P1", tmp_path, run_id="p1_score_repair_decoder_20260905_v1")
    assert result["status"] == "PENDING"
    assert "decoder" in result["artifact"]


def test_archive_path_cannot_escape_allowlist(tmp_path):
    with pytest.raises(ValueError, match="allowlist"):
        QA.audit_run("P1", tmp_path, run_id="../../official")
