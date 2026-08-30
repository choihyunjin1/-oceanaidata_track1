from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_metric_aligned_gates_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("gate_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_directional_classification_uses_primary_benefit() -> None:
    classify = MODULE.classify_benefit
    assert classify(0.01, 0.001, 0.02) == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    assert classify(0.01, -0.001, 0.02) == "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    assert classify(-0.01, -0.02, -0.001) == "PRIMARY_HARM_RESEARCH_ONLY"
    assert classify(-0.01, -0.02, 0.001) == "INCONCLUSIVE_RESEARCH_ONLY"
    assert classify(-0.01, None, None) == "INCONCLUSIVE_RESEARCH_ONLY"
    assert classify(-0.01, None, None, decisive_without_interval=True) == (
        "PRIMARY_HARM_RESEARCH_ONLY"
    )
    assert classify(0.0, -0.001, 0.001) == "INCONCLUSIVE_RESEARCH_ONLY"


def test_replay_changes_only_p2_legacy_no_go_interpretation() -> None:
    payloads = {name: MODULE._read_json(path) for name, path in MODULE.INPUTS.items()}
    replay = MODULE.build_replay(payloads)
    candidates = replay["candidates"]

    changed = {name for name, value in candidates.items() if value["decision_changed"]}
    assert changed == {
        "P2_gaussian_copula_conditional_mean",
        "P2_state_conditioned_copula",
    }
    assert all(
        candidates[name]["new_state"] == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        for name in changed
    )
    assert candidates["P1_event_balanced_supcon"]["new_state"] == (
        "PRIMARY_HARM_RESEARCH_ONLY"
    )
    assert candidates["P3_catboost_confirmation"]["new_state"] == (
        "PRIMARY_HARM_RESEARCH_ONLY"
    )
    assert candidates["P3_selection_matched_masked_ssl"]["new_state"] == (
        "PRIMARY_HARM_RESEARCH_ONLY"
    )


def test_replay_is_zero_fit_and_official_free() -> None:
    payloads = {name: MODULE._read_json(path) for name, path in MODULE.INPUTS.items()}
    summary = MODULE.build_replay(payloads)["summary"]
    assert summary["model_fits"] == 0
    assert summary["prediction_rows_read"] == 0
    assert summary["raw_training_rows_read"] == 0
    assert summary["official_test_sample_submission_hidden_rows_read"] == 0
    assert summary["csv_created"] == 0
    assert summary["uploads"] == 0
