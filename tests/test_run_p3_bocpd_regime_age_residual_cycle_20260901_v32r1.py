from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_bocpd_regime_age_residual_cycle_20260901_v32r1.py"
SPEC = importlib.util.spec_from_file_location("p3_v32r1", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_terminal_source_evidence_is_immutable() -> None:
    recovery, source_config = MODULE.load_recovery()
    assert recovery["source_config_sha256"] == MODULE.sha256(MODULE.SOURCE_CONFIG)
    assert recovery["source_runner_sha256"] == MODULE.sha256(MODULE.SOURCE_RUNNER)
    assert recovery["source_lock_sha256"] == MODULE.sha256(MODULE.SOURCE_LOCK)
    assert recovery["source_failure_receipt_sha256"] == MODULE.sha256(
        MODULE.SOURCE_FAILURE
    )
    assert not MODULE.SOURCE_RESULT.exists()
    assert source_config["experiment_id"] == MODULE.SOURCE_ID


def test_log_domain_agrees_with_ordinary_domain_on_benign_data() -> None:
    values = np.sin(np.linspace(0.0, 20.0, 289))
    ordinary = MODULE.source.bocpd_summary(values)
    stable = MODULE.bocpd_summary_log(values)
    assert np.allclose(ordinary, stable, atol=1e-12, rtol=1e-12)


def test_log_domain_is_finite_for_extreme_normalized_observation() -> None:
    values = np.zeros(289, dtype=np.float64)
    values[-1] = 1e12
    summary = MODULE.bocpd_summary_log(values)
    assert summary.shape == (8,)
    assert np.isfinite(summary).all()
    assert np.all(summary >= 0.0)
    assert np.all(summary <= 1.0 + 1e-12)


def test_log_domain_positive_affine_invariance() -> None:
    values = np.random.default_rng(20260901).normal(size=289)
    assert np.allclose(
        MODULE.bocpd_summary_log(values),
        MODULE.bocpd_summary_log(3.0 * values + 8.0),
        atol=1e-12,
        rtol=1e-12,
    )


def test_execute_changes_only_adapter_and_restores_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MODULE.source.bocpd_summary

    def fake_execute(config):
        assert MODULE.source.bocpd_summary is MODULE.bocpd_summary_log
        return {
            "schema_version": "source",
            "experiment_id": MODULE.SOURCE_ID,
            "execution": {},
        }, {"sentinel": np.asarray([1.0])}

    monkeypatch.setattr(MODULE.source, "execute", fake_execute)
    result, arrays = MODULE.execute({"sealed": True}, {"experiment_id": MODULE.SOURCE_ID})
    assert MODULE.source.bocpd_summary is original
    assert result["execution"]["science_changes"] == 0
    assert result["execution"]["numerical_adapter_changes"] == 1
    assert np.array_equal(arrays["sentinel"], np.asarray([1.0]))


def test_science_contract_is_byte_for_byte_fixed() -> None:
    recovery, _ = MODULE.load_recovery()
    science = recovery["science_contract"]
    assert science == {
        "hazard": 1.0 / 72.0,
        "maximum_run_length": 96,
        "known_observation_variance": 1.0,
        "feature_count": 96,
        "ridge": [512.0, 2048.0],
        "additive_residual_weight": 0.1,
        "purge_hours": 78,
        "maximum_total_fits": 12,
        "tail_gate_changed": False,
        "result_based_tuning": False,
    }
    assert all(value == 0 for value in recovery["official_policy"].values())


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE_SCIENCE_NEUTRAL_RECOVERY"
        assert value["benign_max_abs_difference"] <= 1e-12
        assert value["extreme_finite"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
