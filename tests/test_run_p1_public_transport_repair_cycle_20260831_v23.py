from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v23 as cycle  # noqa: E402


def test_source_candidate_facts_are_exact() -> None:
    config, candidate, _ = cycle.load_contract()
    internal = candidate["internal"]
    assert config["lineage"]["official_result"] == "TIE"
    assert internal["delta_f1"] == 0.0013809855753390554
    assert internal["by_fold"]["2025_q3"]["delta_f1"] == 0.0
    assert internal["by_fold"]["2025_q4"]["delta_f1"] == 0.003432580085375281
    assert internal["deployment_threshold"] == 0.8875
    assert candidate["additions_vs_champion"] == 4


def test_missing_score_lineage_fails_closed_without_posthoc_k() -> None:
    config, _, _ = cycle.load_contract()
    result = cycle.preflight()
    assert result["status"] == cycle.EXPECTED_STATUS
    assert result["decision"] == "NO_GO"
    assert result["persistence_audit"]["continuous_score_or_model_files"] == []
    assert config["prospective_rank_stability_contract"]["top_k"] is None
    assert config["prohibitions"]["posthoc_k_from_deployment_additions"] is True


def test_penalty_stays_pending_and_minimum_point_delta_is_preserved() -> None:
    config, _, _ = cycle.load_contract()
    assert config["transport"]["penalty_points"] == "PENDING_V3"
    assert config["transport"]["raw_gate"] == "PENDING_V3"
    assert config["transport"]["minimum_calibrated_expected_point_delta_inclusive"] == 0.01


def test_preflight_has_zero_fit_official_hidden_lock_and_submission_access() -> None:
    result = cycle.preflight()
    assert result["fit_count"] == 0
    assert result["candidate_count"] == 0
    assert all(value == 0 for value in result["access"].values())
    assert not (ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v23").exists()
    assert not (
        ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v23.ATTEMPT_LOCK.json"
    ).exists()


def test_execute_is_hard_blocked_before_lock_creation() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(cycle.__file__)), "--execute"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "continuous score lineage" in completed.stderr
    assert not (
        ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v23.ATTEMPT_LOCK.json"
    ).exists()
