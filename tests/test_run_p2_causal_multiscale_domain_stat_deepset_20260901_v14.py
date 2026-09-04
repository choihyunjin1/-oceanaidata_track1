from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_causal_multiscale_domain_stat_deepset_20260901_v14.py"
SPEC = importlib.util.spec_from_file_location("p2_v14", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_frame(rows: int = 160) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    payload: dict[str, object] = {"time": time}
    for layer in (1, 5, 6, 7, 8):
        payload[f"temp_{layer}"] = np.linspace(10 + layer, 12 + layer, rows)
        payload[f"psal_{layer}"] = np.linspace(33 + layer / 100, 34 + layer / 100, rows)
    return pd.DataFrame(payload)


def test_contract_is_one_candidate_nine_fits() -> None:
    config = MODULE.load_config()
    assert config["operation_limits"]["maximum_candidate_count"] == 1
    assert config["training"]["maximum_fit_count"] == 9
    assert config["representation"]["half_life_steps_10min"] == [144, 1008, 4320]
    assert config["training"]["row_deletion"] is False


def test_domain_stats_are_causal() -> None:
    frame = synthetic_frame()
    base = np.zeros((len(frame), 5, 8), dtype=np.float32)
    left, _ = MODULE.augment_tokens(frame, base, [12, 24, 48], 6, 0.05, 0.01)
    changed = frame.copy()
    for layer in (1, 5, 6, 7, 8):
        changed.loc[120:, f"temp_{layer}"] += 1000
        changed.loc[120:, f"psal_{layer}"] += 1000
    right, _ = MODULE.augment_tokens(changed, base, [12, 24, 48], 6, 0.05, 0.01)
    np.testing.assert_array_equal(left[:120], right[:120])


def test_shift_one_excludes_current_row_from_domain_statistics() -> None:
    frame = synthetic_frame()
    base = np.zeros((len(frame), 5, 8), dtype=np.float32)
    value, _ = MODULE.augment_tokens(frame, base, [12, 24, 48], 6, 0.05, 0.01)
    current = frame["temp_1"]
    past = current.shift(1)
    mean = past.ewm(halflife=12, adjust=False, min_periods=6).mean()
    variance = past.ewm(halflife=12, adjust=False, min_periods=6).var(bias=True)
    expected = (current.iloc[20] - mean.iloc[20]) / max(np.sqrt(variance.iloc[20]), 0.05)
    assert np.isclose(value[20, 0, 8], np.clip(expected, -12.0, 12.0), atol=1e-6)


def test_public_sensor_channels_do_not_share_state() -> None:
    frame = synthetic_frame()
    base = np.zeros((len(frame), 5, 8), dtype=np.float32)
    left, _ = MODULE.augment_tokens(frame, base, [12, 24, 48], 6, 0.05, 0.01)
    changed = frame.copy()
    changed["temp_1"] += np.linspace(0.0, 500.0, len(changed))
    right, receipt = MODULE.augment_tokens(changed, base, [12, 24, 48], 6, 0.05, 0.01)
    np.testing.assert_array_equal(left[:, 1:, 8:], right[:, 1:, 8:])
    assert receipt["grouping"] == "each_public_layer_x_channel_independently"


def test_representation_width_and_support() -> None:
    frame = synthetic_frame()
    base = np.zeros((len(frame), 5, 8), dtype=np.float32)
    value, receipt = MODULE.augment_tokens(frame, base, [12, 24, 48], 6, 0.05, 0.01)
    assert value.shape == (len(frame), 5, 20)
    assert receipt["causal_shift_rows"] == 1
    assert receipt["added_domain_stat_features"] == 12
    assert max(receipt["support_share"].values()) > 0.9


def test_permutation_invariance_for_augmented_width() -> None:
    receipt = MODULE.permutation_invariance_receipt(20, 11)
    assert receipt["maximum_abs_error"] <= 1e-6


def test_preflight_is_byte_identical_and_zero_operation() -> None:
    first = MODULE.preflight()
    second = MODULE.preflight()
    assert first == second
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["synthetic_representation"]["finite"]
    assert first["synthetic_representation"]["shape"] == [240, 5, 20]
    assert first["semantic_audit_sha256"]
