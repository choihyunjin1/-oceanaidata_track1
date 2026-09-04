from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_s_ors_layer6_information_probe_20260901_v34a.py"
MATERIALIZER = ROOT / "scripts/materialize_p1_s_ors_layer6_information_probe_20260901_v34a.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_action_removes_only_s_layer6(tmp_path: Path) -> None:
    runner = load(RUNNER, "p1_s6_runner_test")
    frame = pd.DataFrame({
        "station": ["S-ORS", "S-ORS", "S-ORS", "I-ORS"],
        "year": [2025] * 4,
        "layer": [6, 6, 4, 6],
        "time": ["a", "b", "c", "d"],
    })
    raw = tmp_path / "raw.npz"
    np.savez(raw, incumbent=np.array([0, 1, 0, 0], np.int8), raw_e150=np.array([1, 1, 1, 1], np.int8))
    incumbent, reference, candidate, removal = runner.build_action(frame, raw)
    assert incumbent.tolist() == [0, 1, 0, 0]
    assert reference.tolist() == [1, 1, 1, 1]
    assert candidate.tolist() == [0, 1, 1, 1]
    assert removal.tolist() == [True, False, False, False]


def test_materializer_validation_contract() -> None:
    materializer = load(MATERIALIZER, "p1_s6_materializer_test")
    frame = pd.DataFrame({
        "station": ["S-ORS"],
        "year": [2025],
        "layer": [6],
        "time": ["2025-01-01T00:00:00+09:00"],
        "label": pd.Series([1], dtype="int8"),
    })
    checks = materializer.validate(frame, 1)
    assert all(checks.values())
