from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_pettitt_rank_change_residual_cycle_20260901_v67.py"
SPEC = importlib.util.spec_from_file_location("p3_v67", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] in {"READY_EXACTLY_ONCE", "STOP_SUPPORT_GATE"}
        assert value["prior_outputs_used"] is False
        assert value["official_used_for_features_gates_selection"] is False


def test_step_localization_sign_and_strength() -> None:
    receipt = MODULE.synthetic_receipt()
    assert 0.45 <= receipt["step_split_fraction"] <= 0.55
    assert receipt["step_signed_median_shift"] > 0.8
    assert receipt["step_abs_normalized_pettitt"] > receipt["iid_abs_normalized_pettitt"] + 0.20


def test_average_ties_and_affine_contract() -> None:
    assert np.array_equal(MODULE.average_ranks(np.asarray([2.0, 1.0, 1.0, 3.0])), [3.0, 1.5, 1.5, 4.0])
    rng = np.random.default_rng(11)
    path = np.concatenate([rng.normal(size=64), rng.normal(size=64) + 3.0])
    direct = MODULE.pettitt_statistics(path)
    assert np.allclose(direct, MODULE.pettitt_statistics(7 * path + 3))
    negative = MODULE.pettitt_statistics(-7 * path + 3)
    assert np.allclose(negative, [-direct[0], direct[1], -direct[2], direct[3]])


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.pettitt_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.pettitt_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.pettitt_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert config["encoder"]["rank_ties"] == "deterministic average ranks"
    assert all(value == 0 for value in config["official_policy"].values())
