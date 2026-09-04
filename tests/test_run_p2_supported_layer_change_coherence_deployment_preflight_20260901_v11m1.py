from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p2_supported_layer_change_coherence_deployment_preflight_20260901_v11m1.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p2_supported_layer_change_coherence_deployment_preflight_20260901_v11m1.json"
SPEC = importlib.util.spec_from_file_location("p2_v11m1_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_guards_are_exactly_preregistered() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    guard = config["deployment_guards"]
    assert guard["active_share_lte"] == 0.5
    assert guard["absolute_action_p99_C_lte"] == 0.5
    assert guard["absolute_action_max_C_lte"] == 2.5
    assert guard["profile_projection_exact_noop_atol_C"] == 1e-12
    assert guard["on_fail"] == "NO_CSV"


def test_preflight_is_identical_and_reads_zero_official_rows() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OFFICIAL_ROW_PREFLIGHT_PASS"
    assert first["official_rows_read"] == 0
    assert first["hidden_truth_rows_read"] == 0
    assert first["score_file_rows_read"] == 0
    assert first["submission_csv_created"] == 0


def test_no_hidden_or_score_path_exists_in_runner() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "sample_submission" not in text
    assert "score.py" not in text
    assert "requests.post" not in text
    assert "upload(" not in text
