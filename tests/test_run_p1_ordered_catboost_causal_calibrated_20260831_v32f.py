from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_ordered_catboost_causal_calibrated_20260831_v32f.py"
SPEC = importlib.util.spec_from_file_location("p1_v32f", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/experiments/p1_ordered_catboost_causal_calibrated_20260831_v32f.json").read_text(
            encoding="utf-8"
        )
    )


def test_grid_is_exact_preregistered_surface() -> None:
    grid = MODULE.threshold_grid(load_config())
    assert len(grid) == 91
    assert np.array_equal(grid, np.round(np.arange(0.05, 0.951, 0.01), 2))


def test_threshold_tie_break_prefers_closest_to_half() -> None:
    truth = np.array([1, 0], dtype=np.int8)
    probability = np.array([0.9, 0.1], dtype=float)
    threshold, score = MODULE.select_threshold(
        truth, probability, np.array([0.2, 0.4, 0.5, 0.6, 0.8])
    )
    assert threshold == 0.5
    assert score == 1.0


def test_fit_calibration_split_is_disjoint_and_purged() -> None:
    time = pd.date_range("2024-01-01", "2025-03-25", freq="12h", tz="Asia/Seoul")
    train = pd.DataFrame(
        {"station": "I-ORS", "layer": 1, "time": time.astype(str), "label": 0}
    )
    fold = MODULE.Fold(
        "2025_q2",
        np.arange(len(train)),
        np.array([len(train) - 1]),
        pd.Timestamp("2025-03-24T14:50:00Z"),
        pd.Timestamp("2025-03-31T15:00:00Z"),
        pd.Timestamp("2025-06-30T15:00:00Z"),
    )
    fit, calibration, audit = MODULE.split_fit_calibration(
        train, fold, calibration_days=45, purge_days=14
    )
    assert len(np.intersect1d(fit, calibration)) == 0
    assert audit["internal_purge_days"] == 14
    fit_max = pd.to_datetime(train.loc[fit, "time"], utc=True).max()
    calibration_min = pd.to_datetime(train.loc[calibration, "time"], utc=True).min()
    assert calibration_min - fit_max >= pd.Timedelta(days=14)


def test_contract_is_independent_and_official_zero() -> None:
    config = load_config()
    assert config["calibration"]["trailing_days"] == 45
    assert config["maximum_runtime_seconds"] == 1200
    assert set(config["official_access_budget"].values()) == {0}
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("predict_submission", "write_submission", "validate_submission"):
        assert token not in source
