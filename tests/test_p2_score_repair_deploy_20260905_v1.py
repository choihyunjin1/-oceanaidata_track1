"""Synthetic deployment contract tests; no competition input reads."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_score_deploy", ROOT / "scripts/run_p2_score_repair_deploy_20260905_v1.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def example():
    return pd.DataFrame({"station": ["S-ORS"] * 3, "layer": [2, 3, 4], "time": ["2025-09-01T00:00:00+09:00"] * 3, "temp": [1.0, 2.0, 3.0]})


def test_schema_keys_order_finite():
    frame = example()
    assert all(deploy.validate_output(frame, frame[deploy.KEYS], frame[deploy.KEYS], 3).values())


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "order"])
def test_corrupt_candidate_rejected(kind):
    frame = example()
    sample = frame[deploy.KEYS].copy()
    if kind == "duplicate":
        frame.loc[1, "layer"] = 2
    elif kind == "nonfinite":
        frame.loc[1, "temp"] = np.nan
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        deploy.validate_output(frame, sample, sample, 3)


def test_fullfit_requires_parent_qa_without_source_read():
    with pytest.raises(RuntimeError, match="root numerical QA"):
        deploy.train(False)


def test_stage_source_boundaries():
    source = (ROOT / "scripts/run_p2_score_repair_deploy_20260905_v1.py").read_text(encoding="utf-8")
    training = source.split("def train(")[1].split("def predict(")[0]
    assert "test_index.csv" not in training and "sample_submission.csv" not in training
    assert 'usecols=KEYS' in source
    assert "baseline_interp.csv" not in source
