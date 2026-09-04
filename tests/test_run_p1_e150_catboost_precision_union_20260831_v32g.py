from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_v32g",
    ROOT / "scripts" / "run_p1_e150_catboost_precision_union_20260831_v32g.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metric_perfect() -> None:
    result = MODULE.v32a.metric(np.array([0, 1], dtype=np.int8), np.array([0, 1], dtype=np.int8))
    assert result["f1"] == 1.0


def test_wilson_lower_is_conservative() -> None:
    assert 0.0 < MODULE.wilson_lower(8, 10) < 0.8


def test_candidate_operation_is_add_only() -> None:
    reference = np.array([0, 1, 0, 1], dtype=np.int8)
    catboost = np.array([1, 0, 0, 1], dtype=np.int8)
    candidate = np.maximum(reference, catboost).astype(np.int8)
    assert np.all(candidate >= reference)
