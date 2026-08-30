from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_p1_clean_state_capa_reopen_preflight_20260831_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_clean_state_capa_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_reopens_only_the_unexecuted_scientific_family() -> None:
    result = _module().build_result()
    assert result["decision"] == "READY_TO_PREREGISTER_RESEARCH_ONLY_NOT_READY_TO_FIT"
    assert result["scientific_disposition"] == "REOPEN_UNEXECUTED_FAMILY"
    assert result["stage1_authorized"] is False
    assert result["lineage_evidence"]["v6r2_model_fits"] == 0
    assert result["lineage_evidence"]["v6r4_model_fits"] == 0


def test_preflight_preserves_zero_action_boundary() -> None:
    result = _module().build_result()
    assert all(result["checks"].values())
    assert result["execution_audit"] == {
        "aggregate_json_files_read": 8,
        "raw_training_rows_read": 0,
        "official_test_sample_submission_hidden_rows_read": 0,
        "model_fits": 0,
        "prediction_rows_created": 0,
        "csv_created": 0,
        "uploads": 0,
    }


def test_headroom_is_reported_as_time_stamped_planning_evidence() -> None:
    headroom = _module().build_result()["leaderboard_headroom_snapshot"]
    assert headroom["mapping_is_not_official"] is True
    assert headroom["planning_only_f1_gap"] > 0.1
    assert headroom["point_gap"] > 3.0
