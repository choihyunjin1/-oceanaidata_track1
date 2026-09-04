from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v26 as cycle  # noqa: E402


def test_exact_gce_rank_transport_contract() -> None:
    config = cycle.load_contract()
    assert config["model"]["gce_q"] == 0.7
    assert config["model"]["l2"] == 0.001
    assert config["inner_selector"]["choice"] == "minimum eligible action count"
    assert config["outer_selector"]["labels_used"] is False
    assert config["outer_selector"]["minimum_one_override"] is False
    assert config["fit_budget"]["maximum"] == 2


def test_minimum_eligible_action_fraction_is_selected() -> None:
    n = 1000
    probability = np.linspace(1.0, 0.0, n, endpoint=False)
    labels = np.zeros(n, dtype=np.int8)
    anchor = np.zeros(n, dtype=np.int8)
    labels[0] = 1
    labels[900:950] = 1
    anchor[900:950] = 1
    result = cycle.select_minimum_action_fraction(
        probability,
        labels,
        anchor,
        np.arange(n, dtype=np.uint64),
        maximum_changed_fraction=0.005,
    )
    assert result["selected_count"] == 1
    assert result["inner_delta_f1"] > 0
    assert result["precision"] == 1.0
    assert result["changed_fraction"] == 0.001


def test_outer_ecdf_is_label_free_scale_invariant_and_floor_conservative() -> None:
    assert "labels" not in inspect.signature(cycle.select_outer_top_fraction).parameters
    scores = np.array([0.9, 0.4, 0.4, 0.1])
    hashes = np.array([9, 8, 2, 1], dtype=np.uint64)
    first = cycle.select_outer_top_fraction(
        scores, hashes, fraction_numerator=1, fraction_denominator=2
    )
    transformed = cycle.select_outer_top_fraction(
        np.log1p(scores), hashes, fraction_numerator=1, fraction_denominator=2
    )
    assert np.array_equal(first, transformed)
    assert first.tolist() == [0, 2]
    assert len(first) == (1 * len(scores)) // 2


def test_key_hash_is_deterministic_and_uses_station_layer_time() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A", "A", "B"],
            "layer": [1, 2, 1],
            "time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
            ),
        }
    )
    first = cycle.stable_key_hashes(frame)
    second = cycle.stable_key_hashes(frame)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 3


def test_sealed_probability_diagnostic_confirms_absolute_scale_failure_without_labels() -> None:
    diagnostic = cycle.diagnose_probability_drift()
    q3 = diagnostic["v24"]["2025_q3"]
    q4 = diagnostic["v24"]["2025_q4"]
    assert diagnostic["read_contract"]["labels_read"] == 0
    assert diagnostic["read_contract"]["retrospective_candidate_rescored"] is False
    assert q3["outer_action_fraction"] > 0.005
    assert q4["outer_action_fraction"] < q3["outer_action_fraction"] / 10
    assert q4["score_quantiles"]["p99"] < q3["score_quantiles"]["p99"] / 100


def test_preflight_preserves_v3_gate_and_authorizes_exactly_once_execution() -> None:
    config = cycle.load_contract()
    result = cycle.synthetic_preflight(config)
    assert result["status"] == "PASS"
    assert config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248
    assert config["decision_policy"]["minimum_calibrated_expected_point_delta_inclusive"] == 0.01
    assert config["authorization"]["historical_execution"] is True
    assert config["authorization"]["attempt_lock_creation"] is True


def test_terminal_artifact_and_operations_are_exactly_once_and_zero_official() -> None:
    artifact = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v26"
    result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE_INTERNAL_ONLY"
    assert result["fit_count"] == 2
    assert len(result["nested_fit_receipts"]) == 2
    assert result["operations"] == {
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def test_terminal_metrics_independently_recompute_from_sealed_bits() -> None:
    artifact = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v26"
    result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
    frame, anchor, _, _ = cycle.gce.load_feature_surface()
    with np.load(artifact / "sealed_nested_predictions.npz", allow_pickle=False) as sealed:
        candidate = sealed["candidate"]
    recomputed = cycle.evaluation.evaluate(frame, anchor, candidate, cycle.load_contract())
    assert np.isclose(recomputed["delta_f1"], result["candidate"]["delta_f1"])
    assert np.isclose(
        recomputed["raw_expected_points_delta"],
        result["candidate"]["raw_expected_points_delta"],
    )
    assert recomputed["additions"] == result["candidate"]["additions"]
    assert recomputed["anchor_removals"] == 0
