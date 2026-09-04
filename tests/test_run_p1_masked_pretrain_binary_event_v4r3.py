from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_masked_pretrain_binary_event_v4r3.py"
SPEC = importlib.util.spec_from_file_location("p1_masked_pretrain_v4r3_tombstone_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4r3_static_no_go_and_execution_tombstone_are_exact() -> None:
    artifact = ROOT / "artifacts/p1_masked_pretrain_binary_event_v4r3"
    receipt = artifact / "STATIC_QA_NO_GO_SUPERSEDED_20260823.json"
    tombstone = artifact / "EXECUTION_TOMBSTONE.json"
    assert _sha(receipt) == "ea3102c6f5a6e53423db0e807011e454c65158f8c915bac721cbbda9358816da"
    assert _sha(tombstone) == runner.EXECUTION_TOMBSTONE_SHA256
    value = json.loads(tombstone.read_text(encoding="utf-8"))
    assert value["generation"] == "p1_masked_pretrain_binary_event_v4r3"
    assert value["execution_prohibited"] is True
    assert value["authorization_must_fail_before_attempt_lock"] is True
    assert value["attempt_lock_created"] is False
    assert value["curve_model_fits"] == 0
    assert value["test_value_reads"] == 0
    assert value["uploads"] == 0
    assert value["successor_generation"] == "p1_masked_pretrain_binary_event_v4r4"


def test_v4r3_authorization_fails_before_attempt_lock() -> None:
    lock = ROOT / runner.CANONICAL_LOCK
    assert not lock.exists()
    paths = runner._paths(ROOT)
    with pytest.raises(PermissionError, match="superseded and non-executable"):
        runner.authorize_entry(
            root=ROOT,
            data_dir=ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly",
            requested_config=paths["config"],
            requested_artifact=paths["artifact"],
        )
    assert not lock.exists()
