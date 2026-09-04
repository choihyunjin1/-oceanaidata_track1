from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p1_e125_only_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_lengths_respect_time_gaps() -> None:
    module = load_module()
    keys = pd.DataFrame(
        {
            "station": ["I-ORS"] * 6,
            "year": [2026] * 6,
            "layer": [5] * 6,
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00+09:00",
                    "2026-01-01 00:10+09:00",
                    "2026-01-01 00:20+09:00",
                    "2026-01-01 01:00+09:00",
                    "2026-01-01 01:10+09:00",
                    "2026-01-01 01:20+09:00",
                ]
            ).astype(str),
        }
    )
    assert module.component_lengths(keys, np.ones(len(keys), dtype=bool)) == [3, 3]


def test_equal_arrays_requires_bit_identity() -> None:
    module = load_module()
    left = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    same = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    close = {"x": np.asarray([1.0, np.nextafter(np.float32(2.0), np.float32(3.0))])}
    assert module.equal_arrays(left, same)
    assert not module.equal_arrays(left, close)
