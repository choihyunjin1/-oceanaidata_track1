"""The current supplement must not silently inherit historical eligibility."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_score_use_policy_distinguishes_comparison_and_inverse_fitting():
    policy = json.loads(
        (ROOT / "configs/compliance/organizer_score_use_policy_20260902.json")
        .read_text(encoding="utf-8")
    )
    assert policy["leaderboard_candidate_comparison_allowed"] is True
    assert policy["leaderboard_parameter_inverse_fitting_allowed"] is False
    assert policy["hidden_label_or_public_membership_reconstruction_allowed"] is False
    assert policy["historical_artifacts_must_be_preserved"] is True
    assert len(policy["requires_refitting_from_distributed_data"]) == 2


def test_current_handoff_warns_against_historical_ready_labels():
    for name in (
        "00_ORGANIZER_DATA_POLICY.md",
        "AI_HANDOFF.md",
        "docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md",
    ):
        assert "9월 2일" in (ROOT / name).read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "2026-09-02 score-use supplement" in agents
