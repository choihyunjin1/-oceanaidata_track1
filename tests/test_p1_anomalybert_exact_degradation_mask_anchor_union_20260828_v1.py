from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p1_qc.p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1 import (
    ExactDegradationMaskTransformer,
    RelativePositionBias,
    anchor_union,
    coverage_windows,
    decode_components,
    inject_exact_degradation,
    synthetic_family_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = (
        ROOT
        / "scripts/run_p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1.py"
    )
    spec = importlib.util.spec_from_file_location("p1_anomalybert_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_outputs_exact_row_mask_and_bounded_attention() -> None:
    model = ExactDegradationMaskTransformer(
        input_width=11,
        window_rows=64,
        patch_rows=8,
        d_model=32,
        heads=4,
        layers=2,
        feedforward_width=64,
        dropout=0.0,
    )
    logits = model(torch.randn(2, 64, 11))
    bias = RelativePositionBias(heads=4, maximum_tokens=8)(8)
    assert logits.shape == (2, 64)
    assert bias.shape == (4, 8, 8)
    assert torch.isfinite(logits).all()


def test_exact_generator_mask_matches_mutated_interval_for_all_families() -> None:
    clean = np.zeros((1024, 11), dtype=np.float32)
    durations = {
        "spike": (1, 1),
        "noise": (18, 18),
        "flatline": (12, 12),
        "offset": (48, 48),
        "drift": (54, 54),
    }
    amplitudes = {
        "spike": (2.0, 2.0),
        "noise": (1.0, 1.0),
        "flatline": (0.0, 0.0),
        "offset": (1.25, 1.25),
        "drift": (1.5, 1.5),
    }
    for offset, family in enumerate(durations):
        changed, mask, (start, stop) = inject_exact_degradation(
            clean,
            family,
            np.random.default_rng(100 + offset),
            durations,
            amplitudes,
        )
        assert mask.sum() == stop - start == durations[family][0]
        assert mask[start:stop].all()
        assert not mask[:start].any()
        assert not mask[stop:].any()
        assert np.array_equal(changed[:, 1:5], clean[:, 1:5])
        assert np.array_equal(changed[:, 7:], clean[:, 7:])


def test_coverage_windows_cover_every_eligible_row_without_crossing_segments() -> None:
    segments = np.array([1] * 5 + [2] * 13)
    eligible = np.ones(18, dtype=bool)
    windows = coverage_windows(segments, eligible, window_rows=8, stride_rows=4)
    covered = np.zeros(18, dtype=bool)
    for window in windows:
        valid = window.rows[: window.valid_rows]
        covered[valid] = True
        assert len(np.unique(segments[valid])) == 1
    assert covered.all()


def test_raw_decoder_has_no_bridging_and_enforces_component_bounds() -> None:
    scores = np.zeros(60, dtype=np.float64)
    scores[2:20] = 0.9
    scores[21:39] = 0.9
    scores[40:58] = 0.9
    segments = np.zeros(60, dtype=np.int64)
    eligible = np.ones(60, dtype=bool)
    prediction = decode_components(
        scores,
        segments,
        eligible,
        threshold=0.5,
        minimum_rows=18,
        maximum_rows=18,
    )
    assert prediction.sum() == 54
    assert prediction[20] == prediction[39] == 0
    scores[0:19] = 0.9
    rejected = decode_components(
        scores,
        segments,
        eligible,
        threshold=0.5,
        minimum_rows=18,
        maximum_rows=18,
    )
    assert rejected[:20].sum() == 0


def test_anchor_union_never_deletes_anchor_rows() -> None:
    anchor = np.array([1, 0, 1, 0], dtype=np.int8)
    additions = np.array([0, 1, 0, 0], dtype=np.int8)
    candidate = anchor_union(anchor, additions)
    assert candidate.tolist() == [1, 1, 1, 0]
    assert np.all(candidate[anchor == 1] == 1)


def test_synthetic_family_boundary_metric_is_exact() -> None:
    truth = np.r_[np.zeros(5), np.ones(10), np.zeros(5)].astype(np.int8)
    metrics = synthetic_family_metrics({"noise": [truth]}, {"noise": [truth.copy()]})
    assert metrics["noise"]["f1"] == 1.0
    assert metrics["noise"]["boundary_mae_rows"] == 0.0


def test_contract_defers_q2_and_prohibits_post_q2_or_official_access() -> None:
    runner = _runner()
    config = runner._json(runner.CONFIG)
    receipt = runner.validate_contract(config)
    assert receipt["status"] == "PASS"
    assert receipt["inputs"]["q2_truth_and_keys"]["deferred"] is True
    assert receipt["inputs"]["q2_frozen_anchor"]["deferred"] is True
    assert receipt["q2_truth_rows_read"] == 0
    assert receipt["q3_q4_rows_read"] == 0
    assert receipt["official_test_sample_submission_rows_read"] == 0
    assert config["execution_policy"]["official_upload_authorized"] is False
    assert config["execution_policy"]["submission_csv_generation_allowed"] is False
    assert pd.Timedelta(days=config["split"]["purge_days"]) == pd.Timedelta(days=15)
