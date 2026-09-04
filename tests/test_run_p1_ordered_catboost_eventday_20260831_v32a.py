from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_ordered_catboost_eventday_20260831_v32a.py"
SPEC = importlib.util.spec_from_file_location("p1_v32a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_frame() -> tuple[pd.DataFrame, np.ndarray]:
    time = pd.date_range("2025-01-01", periods=24, freq="10min", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {
            "station": ["I-ORS"] * 24,
            "layer": [1] * 24,
            "time": time.astype(str),
        }
    )
    target = np.zeros(24, dtype=np.int8)
    target[6:10] = 1
    target[18:20] = 1
    return frame, target


def test_event_day_weight_is_positive_finite_and_deterministic() -> None:
    frame, target = synthetic_frame()
    first = MODULE.event_day_weight(frame, target)
    second = MODULE.event_day_weight(frame, target)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert (first > 0).all()
    assert first[target == 1].mean() > first[target == 0].mean()


def test_paired_bootstrap_detects_identical_predictions() -> None:
    frame, target = synthetic_frame()
    prediction = target.copy()
    result = MODULE.paired_bootstrap(
        target, prediction, prediction, frame, replicates=100, seed=7
    )
    assert result["difference_ci90"] == [0.0, 0.0]
    assert result["difference_mean"] == 0.0


def test_preregistration_and_official_access_contract() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/p1_ordered_catboost_eventday_20260831_v32a.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["maximum_runtime_seconds"] == 1200
    assert config["validation"]["folds"] == ["2025_q2", "2025_q3", "2025_q4"]
    assert config["model"]["probability_threshold"] == 0.8
    assert set(config["official_access_budget"].values()) == {0}
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_train_test" not in source
    assert "predict_submission" not in source
    assert "write_submission" not in source


def test_metric_known_example() -> None:
    result = MODULE.metric(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["f1"] == 0.5
