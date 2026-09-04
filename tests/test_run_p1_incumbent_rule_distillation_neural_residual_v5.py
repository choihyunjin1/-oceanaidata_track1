from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5.py"
SPEC = importlib.util.spec_from_file_location("test_p1_gen5_tombstoned_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gen5_static_no_go_and_execution_tombstone_are_exact() -> None:
    artifact = ROOT / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5"
    prereg = artifact / "preregistration.json"
    preseal = artifact / "preseal_static_qa.json"
    receipt = artifact / "STATIC_QA_NO_GO_SUPERSEDED_20260823.json"
    tombstone = artifact / "EXECUTION_TOMBSTONE.json"
    assert _sha(prereg) == "612543d63eeac8f8ed444d1bfadcac6663b67d31c265082a011cb7a7f3221014"
    assert _sha(preseal) == "8e530ab36c61b50b8f5bca14245578230dad37b386c1d71f60726cc300723449"
    assert _sha(receipt) == "1c3da3bd4bc7c14c44319ecefc8c21e5a698671bfa4f49060afc8b97d58245f4"
    assert _sha(tombstone) == runner.EXECUTION_TOMBSTONE_SHA256
    value = json.loads(tombstone.read_text(encoding="utf-8"))
    assert value["generation"] == "p1_incumbent_rule_distillation_neural_residual_v5"
    assert value["successor_generation"] == (
        "p1_incumbent_rule_distillation_neural_residual_v5r2"
    )
    assert value["execution_prohibited"] is True
    assert value["authorization_must_fail_before_attempt_lock"] is True
    assert value["attempt_lock_created"] is False
    assert value["curve_model_fits"] == 0
    assert value["test_value_reads"] == value["uploads"] == 0


def test_gen5_authorization_fails_before_attempt_lock() -> None:
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
