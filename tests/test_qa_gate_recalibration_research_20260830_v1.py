from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qa_gate_recalibration_research_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("gate_recalibration_qa", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_f1_recomputes_from_counts() -> None:
    counts = {"tp": 3, "fp": 1, "fn": 2}
    assert MODULE._f1(counts) == 6 / 9


def test_integrated_qa_has_no_failed_checks() -> None:
    qa = MODULE.build_qa()
    assert qa["status"] == "PASS"
    assert qa["failed_checks"] == []
    assert qa["recomputed_results"]["legacy_replay"][
        "high_value_p2_challengers"
    ] == 2


def test_access_and_outlier_boundaries_are_explicit() -> None:
    qa = MODULE.build_qa()
    boundary = qa["data_and_action_boundary"]
    assert boundary["qa_raw_training_rows_read"] == 0
    assert boundary["official_test_sample_submission_hidden_rows_read"] == 0
    assert boundary["prediction_csv_created"] == 0
    assert boundary["uploads"] == 0
    assert boundary["hard_deleted_or_masked_source_rows"] == 0
    assert qa["outlier_finding"]["P2_preexisting_reference_extreme_rows"] == 18
    assert qa["outlier_finding"]["P2_new_or_active_extreme_rows"] == 0


def test_integrated_qa_seal_recomputes() -> None:
    qa = MODULE.build_qa()
    seal = qa.pop("seal")
    assert MODULE._seal(qa) == seal["payload_without_seal_sha256"]
