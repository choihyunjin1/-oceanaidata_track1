import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ts_qa", ROOT / "scripts/qa_p1_ts_disagreement_20260905_v4.py"
)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def test_confusion_counts():
    assert q.independent_counts([0, 0, 1, 1, 1], [0, 1, 0, 1, 1]) == {
        "rows": 5,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "f1": 2 / 3,
    }


def test_empty_counts():
    assert q.independent_counts([], []) == {"rows": 0, "tp": 0, "fp": 0, "fn": 0, "f1": 0}


def test_fractional_rejected():
    with pytest.raises(ValueError):
        q.independent_counts([0.5], [1])
