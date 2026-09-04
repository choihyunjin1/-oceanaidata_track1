from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v18r2 as recovery  # noqa: E402


def test_metric_recovery_reuses_exact_sealed_prediction() -> None:
    config, _, _ = recovery.load_recovery_contract()
    seal = json.loads(recovery.PREDICTION_SEAL.read_text(encoding="utf-8"))
    assert config["sealed_prediction_npz_sha256"] == seal["npz_sha256"]
    assert config["sealed_candidate_sha256"] == seal["candidate_sha256"]
    assert config["sealed_probability_sha256"] == seal["probability_sha256"]
    assert config["additional_fit_budget"] == 0
    assert config["prediction_changes"] == 0


def test_penalty_alias_is_exact_and_does_not_change_base_file() -> None:
    before = recovery.original.sha256(recovery.BASE_CONFIG)
    contract = recovery.evaluation_contract()
    assert (
        contract["decision_policy"]["transport_penalty_points"]
        == contract["transport_family"]["transport_penalty_points"]
    )
    assert recovery.original.sha256(recovery.BASE_CONFIG) == before


def test_validate_only_is_side_effect_free() -> None:
    before = recovery.ARTIFACT.exists()
    payload = recovery.validate_only()
    assert payload["status"] == "VALID"
    assert all(payload["checks"].values())
    assert recovery.ARTIFACT.exists() is before
