from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("p1_v34a", ROOT / "scripts/run_p1_iors_e150_microfragment_veto_20260901_v34a.py")


def test_microfragment_mask_removes_only_short_complete_i_segments() -> None:
    frame = pd.DataFrame(
        {
            "station": ["I-ORS"] * 6 + ["G-ORS"],
            "year": [2025] * 7,
            "layer": [1, 1, 1, 1, 1, 2, 1],
            "time": [
                "2025-01-01 00:00:00", "2025-01-01 00:10:00",
                "2025-01-01 01:00:00", "2025-01-01 01:10:00",
                "2025-01-01 01:20:00", "2025-01-01 02:00:00",
                "2025-01-01 03:00:00",
            ],
            "fold": ["2025_q3"] * 7,
        }
    )
    incumbent = np.zeros(7, dtype=np.int8)
    raw = np.ones(7, dtype=np.int8)
    mask, inventory = RUNNER.microfragment_mask(frame, incumbent, raw)
    assert mask.tolist() == [True, True, False, False, False, True, False]
    assert [row["length"] for row in inventory] == [2, 3, 1]


def test_microfragment_mask_never_removes_incumbent_positive() -> None:
    frame = pd.DataFrame(
        {"station": ["I-ORS", "I-ORS"], "year": [2025, 2025], "layer": [1, 1], "time": ["2025-01-01 00:00:00", "2025-01-01 00:10:00"], "fold": ["2025_q3", "2025_q3"]}
    )
    incumbent = np.array([1, 0], dtype=np.int8)
    raw = np.array([1, 1], dtype=np.int8)
    mask, _ = RUNNER.microfragment_mask(frame, incumbent, raw)
    assert mask.tolist() == [False, True]


def test_official_geometry_detects_v33a_micro_f1_impossibility() -> None:
    config = __import__("json").loads(RUNNER.CONFIG_PATH.read_text(encoding="utf-8"))
    audit = RUNNER.official_geometry(config)
    assert audit["metric_geometry_consistent"] is False
    assert audit["feasible_removed_true_positive_counts_at_six_decimal_rounding"] == []
    assert audit["observed_drop"] > audit["maximum_possible_drop_if_all_80_removed_rows_are_true_positive"]
