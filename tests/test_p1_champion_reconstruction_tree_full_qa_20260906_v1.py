"""Synthetic-only full4 QA entrance and fixed-probe checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_qa_test", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/qa_tree_full.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def test_no_read_before_complete(tmp_path, monkeypatch):
    q.t.core.write_json(tmp_path / "terminal_result.json", {"status": "RUNNING"})
    monkeypatch.delenv("P1_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="full terminal COMPLETE"):
        q.verify(tmp_path, tmp_path / "historical", tmp_path / "seal")


def test_probe_uses_sorted_training_positions_not_raw_id_sort():
    frame = pd.DataFrame({"row_id": np.arange(10000)[::-1]})
    actual = q.expected_probe_ids(frame)
    assert actual.shape == (2048,) and len(np.unique(actual)) == 2048
    assert actual[0] == 9999 and actual[-1] == 0


def test_no_model_fit_or_official_paths():
    source = Path(q.__file__).read_text()
    assert ".fit(" not in source and "test.csv" not in source and "sample_submission.csv" not in source
    assert "to_csv(" not in source
