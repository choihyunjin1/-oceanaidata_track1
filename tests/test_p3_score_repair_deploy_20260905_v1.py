"""Synthetic tests: independent cases, exact six-lead ordering, frozen 6h-only path."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_score_repair_deploy_20260905_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_clean_deploy_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cases():
    return pd.DataFrame(
        {
            "case_id": ["A", "B"],
            "station": ["G-ORS", "I-ORS"],
            "hs_current": [1.6, 2.0],
            "tp_current": [4.0, np.nan],
        }
    )


def test_case_major_six_lead_order_and_current():
    x, current, keys = MODULE.rows_for_cases(cases(), ["hs_current", "tp_current"])
    assert keys.case_id.tolist() == ["A"] * 6 + ["B"] * 6
    assert keys.lead_h.tolist() == [3, 6, 9, 12, 18, 24] * 2
    np.testing.assert_array_equal(current, [1.6] * 6 + [2.0] * 6)
    assert len(x) == 12


def test_tab_6h_matrix_keeps_station_and_feature_order():
    rows, _, _ = MODULE.rows_for_cases(cases(), ["hs_current", "tp_current"])
    matrix = MODULE.tab_matrix(rows)
    assert matrix.shape == (2, 4)
    np.testing.assert_array_equal(matrix[:, 0], [0, 1])
    np.testing.assert_allclose(matrix[:, 1], [1.6, 2.0])
    assert np.isnan(matrix[1, -1])


def test_duplicate_cases_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.rows_for_cases(pd.concat([cases(), cases()]), ["hs_current"])


def test_case_reordering_preserves_per_case_features():
    _, _, first = MODULE.rows_for_cases(cases(), ["hs_current"])
    _, _, second = MODULE.rows_for_cases(cases().iloc[::-1], ["hs_current"])
    assert set(first.itertuples(index=False, name=None)) == set(
        second.itertuples(index=False, name=None)
    )


def test_no_legacy_fullfit_entrypoint_or_historical_oof_reader():
    text = PATH.read_text(encoding="utf-8")
    assert "base._fit_full_and_infer(" not in text
    assert "p3_refined_public_optimum" not in text.lower()
    assert 'pd.read_parquet(OUT / "oof.parquet")' not in text


def test_utf8_and_replay_time_are_explicit():
    text = PATH.read_text(encoding="utf-8")
    assert ".read_text()" not in text
    assert (
        'prior_seconds = training["elapsed_prepare_plus_train_seconds"] + replay_qa["seconds"]'
        in text
    )
    assert "reread.hs_pred, frame.hs_pred" in text


def test_complete_prediction_changes_six_hour_only():
    frame = cases()
    for col in MODULE.OBSERVED_FEATURES:
        if col not in frame:
            frame[col] = 0.0

    class Single:
        def predict(self, matrix, thread_count):
            assert thread_count == 2
            return np.zeros(len(matrix))

    class Multi:
        def predict(self, matrix, thread_count):
            return np.zeros((len(matrix), 6))

    class Router:
        def predict_weights(self, matrix):
            return np.tile([0.5, 0.5, 0.0], (len(matrix), 1))

    class Tab:
        def predict(self, matrix):
            return np.full(len(matrix), 0.4)

    keys, baseline, candidate = MODULE.predict_cases(
        frame, ["hs_current", "tp_current"], Single(), Multi(), Router(), Tab()
    )
    six = keys.lead_h.eq(6).to_numpy()
    np.testing.assert_array_equal(candidate[~six], baseline[~six])
    np.testing.assert_allclose(candidate[six] - baseline[six], 0.1)
    np.testing.assert_allclose(baseline, [1.6] * 6 + [2.0] * 6)
