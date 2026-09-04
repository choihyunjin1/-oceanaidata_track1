from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v5_within_station_vertical_causal_graph_20260901_v1r1.py"
CONFIG = ROOT / "configs/experiments/p1_v5_within_station_vertical_causal_graph_20260901_v1r1.json"


def _wrapper():
    spec = importlib.util.spec_from_file_location("p1_v5r1_wrapper", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_repair_is_namespace_and_time_unit_only() -> None:
    wrapper = _wrapper()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert wrapper._sha(wrapper.ORIGINAL) == config["repair"]["predecessor_runner_sha256"]
    assert config["repair"]["predecessor_result_use_for_selection"] == 0
    assert config["architecture"]["maximum_fits"] == 9
    assert config["architecture"]["epochs"] == 4
    assert config["architecture"]["hidden_width"] == 16
    assert config["selection"]["outer_tuning"] == 0
    assert '.astype("int64")' not in wrapper._patched_source()


def test_time_conversion_is_epoch_nanoseconds() -> None:
    implementation = _wrapper()._load_impl()
    parsed = pd.Series(pd.to_datetime(["2025-03-24T23:40:00+09:00", "2025-03-24T23:50:00+09:00"], utc=True))
    values = implementation._time_ns(parsed)
    assert values.dtype == np.int64
    assert values[0] > 10**18
    assert values[1] - values[0] == 600_000_000_000
    assert values[1] == pd.Timestamp("2025-03-24T23:50:00+09:00").value


def test_cutoff_specific_boundaries_are_distinct_and_bounded(tmp_path: Path) -> None:
    implementation = _wrapper()._load_impl()
    times = pd.date_range("2024-01-01", "2025-09-23", freq="10min", tz="UTC")
    path = tmp_path / "train.csv"
    pd.DataFrame({"time": times.astype(str)}).to_csv(path, index=False)
    parts = {
        "q2": {"cutoff": "2025-03-24T23:50:00+09:00"},
        "q3": {"cutoff": "2025-06-23T23:50:00+09:00"},
        "q4": {"cutoff": "2025-09-23T23:50:00+09:00"},
    }
    receipt = implementation._boundary_contract(path, parts, 0.85)
    boundaries = [item["boundary_ns"] for item in receipt["folds"].values()]
    assert receipt["status"] == "PASS_NS_CUTOFF_DISTINCT"
    assert len(set(boundaries)) == 3
    for fold, item in receipt["folds"].items():
        assert item["boundary_ns"] <= pd.Timestamp(parts[fold]["cutoff"]).value


def test_invalid_v5_is_explicitly_non_scientific() -> None:
    failure = json.loads(
        (
            ROOT
            / "artifacts/p1_v5_within_station_vertical_causal_graph_20260901_v1/attempt_journal/9999_failed.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["status"] == "INVALID_TECHNICAL_TARGET_LEAKAGE_TIME_UNIT"
    assert failure["scientific_interpretation_allowed"] is False
    assert failure["resume_or_retry_allowed"] is False
