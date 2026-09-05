"""Synthetic case/model/schema contracts for clean-only regeneration."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_p3_clean_regeneration_20260905_v4.py"
SPEC = importlib.util.spec_from_file_location("p3_regeneration_test", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def cases():
    frame = pd.DataFrame({"case_id": ["A", "B"], "station": ["G-ORS", "I-ORS"], "hs_current": [1.6, 2.0]})
    for column in M.OBSERVED_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    return frame


class Single:
    def predict(self, matrix, thread_count):
        assert thread_count == 2
        return np.full(len(matrix), .4)


class Multi:
    def predict(self, matrix, thread_count):
        assert thread_count == 2
        return np.full((len(matrix), 6), .2)


class Router:
    def predict_weights(self, matrix):
        return np.tile([.5, .5, 0], (len(matrix), 1))


def test_six_lead_case_order():
    matrix, current, keys = M.rows_for_cases(cases(), ["hs_current"])
    assert keys.case_id.tolist() == ["A"] * 6 + ["B"] * 6
    assert keys.lead_h.tolist() == list(M.LEADS) * 2
    np.testing.assert_array_equal(current, [1.6] * 6 + [2] * 6)
    assert len(matrix) == 12


def test_duplicate_cases_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        M.rows_for_cases(pd.concat([cases(), cases()]), ["hs_current"])


def test_exact_clean_equal_components_then_long_shrink():
    keys, prediction = M.predict_cases(cases(), ["hs_current"], Single(), Multi(), Router())
    expected = np.repeat([1.6, 2], 6) + np.tile([.3, .3, .3, .24, .24, .24], 2)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-15)
    assert len(keys) == 12


def test_router_short_leads_forced_equal_components():
    class PersistenceRouter:
        def predict_weights(self, matrix):
            return np.tile([0, 0, 1], (len(matrix), 1))

    _, prediction = M.predict_cases(cases(), ["hs_current"], Single(), Multi(), PersistenceRouter())
    expected = np.repeat([1.6, 2], 6) + np.tile([.3, .3, .3, 0, 0, 0], 2)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-15)


def test_case_reordering_not_global_lookup():
    keys, prediction = M.predict_cases(cases(), ["hs_current"], Single(), Multi(), Router())
    reverse, reversed_prediction = M.predict_cases(cases().iloc[::-1], ["hs_current"], Single(), Multi(), Router())
    keys["p"] = prediction
    reverse["p"] = reversed_prediction
    pd.testing.assert_frame_equal(keys.sort_values(M.KEYS).reset_index(drop=True), reverse.sort_values(M.KEYS).reset_index(drop=True))


def test_deterministic_csv_bytes_schema():
    frame = pd.DataFrame({"case_id": ["A"], "station": ["G-ORS"], "lead_h": [3], "hs_pred": [1.6]})
    first = M.csv_bytes(frame)
    assert first == M.csv_bytes(frame.copy())
    assert first.splitlines()[0] == b"case_id,station,lead_h,hs_pred"


def test_budget_provenance_and_no_old_assets():
    cfg = json.loads(M.CONFIG.read_text(encoding="utf-8"))
    assert sum(cfg["fit_budget"][k] for k in ["historical_backbone", "full_backbone", "prequential_router", "full_router"]) == 11
    assert cfg["recipe"]["public_inverse"] == cfg["recipe"]["tabpfn"] == 0
    assert cfg["recipe"]["router_alpha"] == 10
    assert cfg["recipe"]["persistence_shrink"] == .2
    text = PATH.read_text(encoding="utf-8")
    assert "base._fit_full_and_infer(" not in text
    assert "make_regressor" not in text
    assert "pd.read_parquet(WORK / \"oof.parquet\")" not in text
    assert '--RUN_TRAINING' in text and '--RUN_INFERENCE' in text


def test_existing_output_aborts_before_source_read(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="already exists"):
        M.prepare({}, tmp_path / "nonexistent_source")


def test_nonempty_model_blocks_train_without_deleting(tmp_path, monkeypatch):
    saved = tmp_path / "existing-model.marker"
    saved.write_bytes(b"synthetic-marker-not-a-model")
    monkeypatch.setattr(M, "MODELS", tmp_path)
    monkeypatch.setattr(M, "verify", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="must be empty"):
        M.train({}, tmp_path / "nonexistent_source")
    assert saved.read_bytes() == b"synthetic-marker-not-a-model"


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_synthetic_native_cpu_fit_reload(kind, tmp_path):
    rng = np.random.default_rng(10)
    x = pd.DataFrame({"station": ["G-ORS"] * 24, "x": rng.normal(size=24)})
    target = rng.normal(size=24) if kind == "single" else rng.normal(size=(24, 6))
    model = CatBoostRegressor(iterations=3, loss_function="RMSE" if kind == "single" else "MultiRMSE", thread_count=2, task_type="CPU", verbose=False, allow_writing_files=False)
    model.fit(x, target, cat_features=[0])
    path = tmp_path / f"{kind}.cbm"
    model.save_model(path)
    np.testing.assert_array_equal(model.predict(x, thread_count=2), CatBoostRegressor().load_model(path).predict(x, thread_count=2))
