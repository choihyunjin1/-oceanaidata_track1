from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_central_quantile_residual_cycle_20260901_v31r1.py"
SPEC = importlib.util.spec_from_file_location("p3_v31r1", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_failure_is_immutable_and_result_absent() -> None:
    recovery, source_config = MODULE.load_recovery()
    assert recovery["source_result_must_be_absent"]
    assert not MODULE.SOURCE_RESULT.exists()
    assert MODULE.sha256(MODULE.SOURCE_LOCK) == recovery["source_lock_sha256"]
    assert MODULE.sha256(MODULE.SOURCE_FAILURE) == recovery[
        "source_failure_receipt_sha256"
    ]
    assert source_config["model"]["quantiles"] == [0.25, 0.75]


def test_science_receipt_matches_unchanged_source() -> None:
    recovery, source_config = MODULE.load_recovery()
    receipt = MODULE.science_receipt(source_config)
    assert receipt["quantiles"] == [0.25, 0.75]
    assert receipt["l1_alpha"] == 0.10
    assert receipt["blend"] == 0.10
    assert receipt["maximum_total_fits"] == 12
    assert recovery["science_contract"]["tail_gate_changed"] is False


def test_foreign_spec_adapter_registers_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MODULE.source.v28.SPECS
    observed: list[tuple[object, ...]] = []

    def fake_score(frame, prediction, spec):
        del frame, prediction
        observed.append(MODULE.source.v28.SPECS)
        assert spec in MODULE.source.v28.SPECS
        return {"decision": "NO_GO"}

    monkeypatch.setattr(MODULE.source.v28, "score", fake_score)
    result = MODULE.score_with_registered_spec(None, np.zeros((2, 6)))
    assert result == {"decision": "NO_GO"}
    assert observed == [(MODULE.source.SPEC,)]
    assert MODULE.source.v28.SPECS is original


def test_source_feature_and_fit_budget_contract() -> None:
    assert MODULE.source.CASE_FEATURE_COUNT == 108
    assert MODULE.source.ROW_FEATURE_COUNT == 117
    assert MODULE.source.QUANTILES == (0.25, 0.75)
    assert MODULE.source.BLEND == 0.10


def test_official_policy_zero() -> None:
    recovery, source_config = MODULE.load_recovery()
    assert all(value == 0 for value in recovery["official_policy"].values())
    assert all(value == 0 for value in source_config["official_policy"].values())


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE_SCIENCE_NEUTRAL_RECOVERY"
        assert value["maximum_model_fits"] == 12
        assert value["source_result_absent"]
        assert value["official_access"] == 0
