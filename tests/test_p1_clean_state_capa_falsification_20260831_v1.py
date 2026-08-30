from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.clean_state_capa import (
    INPUT_ONLY_COLUMNS,
    CapaContractError,
    SegmentCandidate,
    apply_clean_state,
    decode_frame,
    fit_clean_state,
    protected_union,
    synthetic_contract_audit,
    weighted_interval_schedule,
)
from scripts import run_p1_clean_state_capa_falsification_20260831_v1 as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "p1_clean_state_capa_falsification_20260831_v1.json"
)


def _synthetic_frame(rows_per_layer: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows_per_layer, freq="10min", tz="UTC")
    frames: list[pd.DataFrame] = []
    phase = np.arange(rows_per_layer, dtype=np.float64)
    for layer, offset in ((1, 0.0), (2, 0.6)):
        frames.append(
            pd.DataFrame(
                {
                    "station": "synthetic_station",
                    "year": 2024,
                    "layer": layer,
                    "time": timestamps.astype(str),
                    "temp": 12.0
                    + offset
                    + 0.3 * np.sin(2.0 * np.pi * phase / 144.0)
                    + 0.01 * np.cos(phase),
                    "psal": 33.0 + 0.02 * np.sin(phase / 11.0),
                    "depth": float(layer * 10),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).loc[:, list(INPUT_ONLY_COLUMNS)]


def test_synthetic_decoder_contract_passes() -> None:
    audit = synthetic_contract_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["offset_recall"] == pytest.approx(1.0)
    assert audit["drift_recall"] == pytest.approx(1.0)


def test_clean_state_fit_apply_and_frame_decode_are_finite_and_deterministic() -> None:
    frame = _synthetic_frame()
    state_a = fit_clean_state(frame)
    state_b = fit_clean_state(frame)
    assert state_a.sha256 == state_b.sha256

    projection_a = apply_clean_state(frame, state_a)
    projection_b = apply_clean_state(frame, state_b)
    assert np.isfinite(projection_a["decoder_signal"]).all()
    numeric_columns = ["seasonal_residual_z", "graph_residual_z", "decoder_signal"]
    np.testing.assert_allclose(projection_a[numeric_columns], projection_b[numeric_columns])
    np.testing.assert_array_equal(
        projection_a[["clean_state_available", "graph_available", "peer_count"]],
        projection_b[["clean_state_available", "graph_available", "peer_count"]],
    )

    mask_a, proposals_a, audit_a = decode_frame(frame, projection_a)
    mask_b, proposals_b, audit_b = decode_frame(frame, projection_b)
    assert mask_a.dtype == np.dtype("bool")
    assert np.array_equal(mask_a, mask_b)
    assert proposals_a == proposals_b
    assert audit_a["proposal_fingerprint"] == audit_b["proposal_fingerprint"]


def test_weighted_interval_schedule_prefers_maximum_nonoverlapping_score() -> None:
    candidates = [
        SegmentCandidate(0, 10, 7.0, 8.0, 1.0, 10, "mean_shift"),
        SegmentCandidate(0, 5, 4.0, 5.0, 1.0, 5, "mean_shift"),
        SegmentCandidate(5, 10, 4.5, 5.5, 1.0, 5, "linear_drift"),
        SegmentCandidate(10, 15, 2.0, 3.0, 1.0, 5, "mean_shift"),
    ]
    selected = weighted_interval_schedule(candidates)
    assert [(item.start, item.end) for item in selected] == [(0, 5), (5, 10), (10, 15)]
    assert sum(item.score for item in selected) == pytest.approx(10.5)


def test_protected_union_preserves_every_incumbent_positive() -> None:
    incumbent = np.asarray([0, 1, 0, 1], dtype=np.int8)
    additions = np.asarray([True, False, False, True], dtype=bool)
    candidate = protected_union(incumbent, additions)
    np.testing.assert_array_equal(candidate, np.asarray([1, 1, 0, 1], dtype=np.int8))
    with pytest.raises(CapaContractError):
        protected_union(incumbent.astype(np.int64), additions)


def test_unseen_station_layer_abstains_instead_of_using_future_validation_state() -> None:
    prefix = _synthetic_frame()
    state = fit_clean_state(prefix)
    unseen = prefix.copy()
    unseen["station"] = "never_seen_in_prefix"
    projection = apply_clean_state(unseen, state)
    assert not projection["clean_state_available"].any()
    assert np.array_equal(projection["decoder_signal"], np.zeros(len(unseen)))
    mask, proposals, audit = decode_frame(unseen, projection)
    assert not mask.any()
    assert proposals == []
    assert audit["abstained_station_layer_groups"] == 2
    assert audit["abstained_rows"] == len(unseen)


def test_public_science_api_has_no_target_or_label_argument() -> None:
    for function in (fit_clean_state, apply_clean_state, decode_frame):
        parameters = set(inspect.signature(function).parameters)
        assert "target" not in parameters
        assert "label" not in parameters
        assert "anomaly_type" not in parameters


def test_config_freezes_exactly_once_research_only_and_semantic_distinctions() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["authorization"]["maximum_executions"] == 1
    assert config["authorization"]["result_based_retry_or_retune"] == 0
    assert config["decision"]["official_submission_authorized"] is False
    assert config["output_policy"]["submission_csv_creation"] == 0
    assert config["output_policy"]["uploads"] == 0
    assert config["resource_ceiling"]["supervised_model_fits"] == 0
    assert set(config["semantic_distinction"]) == {
        "typed_duration_semimarkov",
        "long_event_segment_proposal_rescore",
        "v6_slow_unary",
    }
    assert config["input_contract"][
        "target_columns_open_only_after_all_fold_predictions_are_sealed"
    ] == ["label", "anomaly_type"]


def test_runner_binary_metrics_and_bootstrap_are_exact_and_deterministic() -> None:
    truth = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    incumbent = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int8)
    candidate = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int8)
    metrics = runner._binary_metrics(truth, candidate)
    assert metrics == {
        "rows": 6,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(2 / 3),
    }
    metadata = pd.DataFrame(
        {
            "station": ["s"] * 6,
            "year": [2025] * 6,
            "layer": [1] * 6,
            "time": pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC").astype(str),
        }
    )
    first = runner._paired_cluster_bootstrap(
        truth, incumbent, candidate, metadata, replicates=50, seed=7
    )
    second = runner._paired_cluster_bootstrap(
        truth, incumbent, candidate, metadata, replicates=50, seed=7
    )
    assert first == second
    assert first["replicates"] == 50
    assert first["cluster_count"] >= 2


def test_runner_seals_all_predictions_before_target_read() -> None:
    source = inspect.getsource(runner.execute)
    completion_write = source.index('ARTIFACT_DIR / "predictions_complete.json"')
    target_read = source.index('usecols=["label", "anomaly_type"]')
    assert completion_write < target_read
    assert runner.ATTEMPT_LOCK.name.endswith(".ATTEMPT_LOCK.json")
