"""Synthetic boundary, target-isolation and hard-rule checks for binary decoder."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_decoder", ROOT / "scripts/run_p1_score_repair_decoder_20260905_v1.py"
)
decoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decoder)


def sample():
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": 1,
            "time": pd.date_range("2025-01-01", periods=30, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "label": np.r_[np.zeros(10), np.ones(10), np.zeros(10)].astype(np.int8),
        }
    )


def test_transition_counts_respect_station_layer_gap():
    frame = sample()
    frame.loc[10:19, "station"] = "I-ORS"
    frame.loc[20:, "layer"] = 2
    fitted = decoder.transition_fit(frame)
    assert fitted["valid_adjacent_pairs"] == 27
    assert fitted["counts"][0][1] == 0
    assert fitted["counts"][1][0] == 0
    assert np.allclose(np.asarray(fitted["transition"]).sum(axis=1), 1)


def test_evaluation_label_poisoning_does_not_change_decode():
    training = sample()
    fitted = decoder.transition_fit(training)
    evaluation = sample()
    unary = np.r_[np.full(10, -3.0), np.full(10, 3.0), np.full(10, -3.0)]
    first = decoder.decode_viterbi(evaluation, unary, fitted, np.zeros(30, bool))
    evaluation["label"] = 1 - evaluation.label
    second = decoder.decode_viterbi(evaluation, unary, fitted, np.zeros(30, bool))
    np.testing.assert_array_equal(first, second)


def test_gap_isolation():
    frame = sample().drop(index=14).reset_index(drop=True)
    fitted = decoder.transition_fit(sample())
    unary = np.full(len(frame), -0.1)
    first = decoder.decode_viterbi(frame, unary, fitted, np.zeros(len(frame), bool))
    unary[:14] = 100
    second = decoder.decode_viterbi(frame, unary, fitted, np.zeros(len(frame), bool))
    np.testing.assert_array_equal(first[14:], second[14:])


def test_hard_spike_preserved_against_negative_unary():
    frame = sample()
    hard = np.zeros(30, bool)
    hard[15] = True
    bits = decoder.decode_viterbi(frame, np.full(30, -100.0), decoder.transition_fit(frame), hard)
    assert bits[15] == 1
    assert bits.sum() == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_unary_fails(bad):
    frame = sample()
    unary = np.zeros(30)
    unary[5] = bad
    with pytest.raises(ValueError, match="finite"):
        decoder.decode_viterbi(frame, unary, decoder.transition_fit(frame), np.zeros(30, bool))


def test_transition_training_boundary_is_not_crossed():
    frame = sample().iloc[:10].copy()
    fitted = decoder.transition_fit(frame)
    assert fitted["counts"] == [[9, 0], [0, 0]]
    assert fitted["valid_adjacent_pairs"] == 9


def test_no_backbone_fit_call_and_no_official_input():
    code = (ROOT / "scripts/run_p1_score_repair_decoder_20260905_v1.py").read_text(encoding="utf-8")
    assert ".fit(" not in code
    assert "test.csv" not in code and "sample_submission.csv" not in code


def test_source_runner_resolves_actual_module_and_receipt_hash():
    import json

    runner = decoder.source_runner_path()
    assert runner == ROOT / "scripts/run_p1_score_repair_20260905_v1.py"
    receipt = json.loads(
        (ROOT / "reports/p1_score_repair_20260905_v1/result.json").read_text(encoding="utf-8")
    )
    assert decoder.screen.sha(runner) == receipt["runner_sha256"]
