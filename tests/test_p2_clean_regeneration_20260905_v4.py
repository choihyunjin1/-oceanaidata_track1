"""Synthetic path and scratch initialization boundary tests; no data reads."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_regeneration_v4", ROOT / "scripts/run_p2_clean_regeneration_20260905_v4.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_train_reads_only_source_no_old_or_new_models():
    source = (ROOT / "synthetic_source/observations.csv").resolve()
    assert M.path_allowed(source, "RUN_TRAINING", False, source)
    assert not M.path_allowed(source, "RUN_TRAINING", True, source)
    for path in [source.parent / "test_index.csv", source.parent / "sample_submission.csv", ROOT / "old/model.pt", ROOT / "old/answer.csv", ROOT / "old/raw_oof.npz", M.MODELS / "model.pt"]:
        assert not M.path_allowed(path, "RUN_TRAINING", False, source)
    assert M.path_allowed(M.MODELS / "model.pt", "RUN_TRAINING", True, source)


def test_inference_restricts_models_and_answer_location():
    source = (ROOT / "synthetic_source/observations.csv").resolve()
    assert M.path_allowed(M.MODELS / "model.pt", "RUN_INFERENCE", False, source)
    assert M.path_allowed(M.ANSWERS / "answer.csv", "RUN_INFERENCE", True, source)
    assert not M.path_allowed(ROOT / "old/model.pt", "RUN_INFERENCE", False, source)
    assert not M.path_allowed(ROOT / "old/answer.csv", "RUN_INFERENCE", False, source)


def test_scratch_training_source_has_no_load_or_answer_chain():
    assert M.lineage_source_contract()
    with pytest.raises(PermissionError):
        M.forbidden_legacy()


def test_output_order_schema_finite_and_denominator():
    keys = pd.DataFrame({"station": ["S-ORS"] * 3, "layer": [2, 3, 4], "time": ["2025-09-01T00:00:00Z"] * 3})
    answer = keys.assign(temp=[19., 20., 21.])
    assert all(M.canonical.validate_output(answer, keys, keys, 3).values())
    with pytest.raises(ValueError):
        M.canonical.validate_output(answer.iloc[::-1], keys, keys, 3)
    answer.loc[0, "temp"] = np.nan
    with pytest.raises(ValueError):
        M.canonical.validate_output(answer, keys, keys, 3)
