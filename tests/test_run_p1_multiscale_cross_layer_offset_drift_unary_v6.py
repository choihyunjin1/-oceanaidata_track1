from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.multiscale_cross_layer_offset_drift import (  # noqa: E402
    EXPERIMENT_ID,
    GATE_THRESHOLDS,
    GEOMETRY_FEATURES,
    HYPOTHESIS_ID,
    MAX_SLOW_RUN_ROWS,
    MIN_SLOW_RUN_ROWS,
    MULTISCALE_ROWS,
    RobustSeasonalGraphState,
    apply_robust_seasonal_graph_state,
    build_multiscale_geometry,
    exact_gap_safe_segment_ids,
    protected_incumbent_union,
    seasonal_design,
    static_contract_audit,
    strict_inner_gate,
)

CONFIG = ROOT / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6.json"
RUNNER = ROOT / "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6.py"
HELPER = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift.py"
V9 = ROOT / "artifacts/meaningful_score_goal_v9/registry.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_v6_static_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_gate_metrics() -> dict[str, object]:
    return {
        "micro_f1_delta": 0.01,
        "offset_recall_delta": 0.05,
        "drift_recall_delta": 0.05,
        "spike_f1_delta": 0.0,
        "worst_station_layer_f1_delta": 0.0,
        "normal_fp_relative_increase": 0.01,
        "nondegrading_inner_blocks": 3,
        "inner_block_count": 3,
        "both_slow_types_observed": True,
        "spike_observed": True,
        "all_required_station_layers_observed": True,
        "blind_predictions_sealed_before_gate_labels": True,
    }


def test_single_fixed_hypothesis_and_resource_arithmetic() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hypothesis = config["single_hypothesis_contract"]
    assert config["experiment_id"] == EXPERIMENT_ID
    assert hypothesis["hypothesis_count"] == 1
    assert hypothesis["hypothesis_id"] == HYPOTHESIS_ID
    assert hypothesis["alternatives_registered"] == 0
    assert not any(
        hypothesis[key]
        for key in (
            "threshold_sweep",
            "alpha_sweep",
            "seed_sweep",
            "architecture_search",
            "hyperparameter_search",
            "posthoc_subgroup_selection",
        )
    )
    assert config["trajectory_geometry"]["scales_rows"] == list(MULTISCALE_ROWS)
    assert config["trajectory_geometry"]["feature_count"] == len(GEOMETRY_FEATURES) == 29
    assert config["unary_head"]["threshold"] == 0.5
    assert config["unary_head"]["C"] == 0.25
    assert config["unary_head"]["solver"] == "lbfgs"
    assert config["unary_head"]["max_iter"] == 64
    assert config["unary_head"]["random_state"] == 20260823
    assert config["protected_union"]["minimum_run_rows"] == MIN_SLOW_RUN_ROWS
    assert config["protected_union"]["maximum_run_rows"] == MAX_SLOW_RUN_ROWS
    resource = config["resource_ceiling"]
    assert resource["maximum_top_level_fit_calls"] == 120
    assert resource["maximum_seasonal_irls_steps"] == 7680
    assert resource["maximum_unary_lbfgs_iterations"] == 3840
    assert resource["maximum_total_iterative_steps"] == 11520
    assert resource["maximum_wall_clock_seconds"] == 21600
    assert resource["maximum_vram_bytes"] == 0
    assert resource["maximum_artifact_disk_bytes"] == 1073741824
    assert not any(config["static_prohibitions"].values())
    helper_source = HELPER.read_text(encoding="utf-8")
    assert "def fit_fixed_slow_unary_head(" in helper_source
    assert "def predict_fixed_slow_unary_probability(" in helper_source
    assert 'solver="lbfgs"' in helper_source


def test_fixed_geometry_feature_bank_is_finite_and_gap_safe() -> None:
    rows = 2600
    seasonal = np.zeros(rows, dtype=np.float64)
    graph = np.zeros(rows, dtype=np.float64)
    seasonal[700:1000] = 5.0
    graph[700:1000] = 5.0
    seasonal[1450:1800] = np.linspace(0.0, 7.0, 350, dtype=np.float64)
    graph[1450:1800] = seasonal[1450:1800]
    projection = pd.DataFrame(
        {
            "seasonal_residual_z": seasonal,
            "graph_residual_z": graph,
            "peer_consensus_z": np.zeros(rows, dtype=np.float64),
            "graph_available": np.ones(rows, dtype=np.float64),
            "peer_count": np.ones(rows, dtype=np.float64),
        }
    )
    segments = np.zeros(rows, dtype=np.int64)
    segments[2000:] = 1
    features = build_multiscale_geometry(projection, segments)
    assert tuple(features.columns) == GEOMETRY_FEATURES
    assert features.shape == (rows, 29)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[850, "level_abs_z_96"] > 4.0
    assert features.loc[1600, "slope_abs_z_96"] > features.loc[200, "slope_abs_z_96"]
    # A distinct all-zero segment cannot inherit the preceding trajectory.
    assert float(features.loc[2200:, "level_abs_z_576"].max()) == 0.0
    assert float(features.loc[2200:, "coherence_deficit_z_576"].max()) == 0.0


def test_single_layer_graph_fallback_is_seasonal_and_finite() -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul"),
            "temp": np.linspace(9.0, 11.0, rows),
            "psal": np.full(rows, 33.0),
            "depth": np.full(rows, np.nan),
        }
    )
    state = RobustSeasonalGraphState(
        train_ids_sha256="0" * 64,
        seasonal_coefficients={"G-ORS|1": (10.0,) + (0.0,) * 12},
        seasonal_scales={"G-ORS|1": 1.0},
        edge_residual_deltas={},
        edge_residual_scales={},
    )
    projection = apply_robust_seasonal_graph_state(frame, state)
    assert not projection["graph_available"].astype(bool).any()
    assert np.allclose(projection["graph_residual_z"], projection["seasonal_residual_z"])
    features = build_multiscale_geometry(projection, np.zeros(rows, dtype=np.int64))
    assert np.isfinite(features.to_numpy()).all()


def test_gap_segments_and_fourier_design_are_label_free_and_fixed() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 4 + ["I-ORS"],
            "layer": [1, 1, 1, 1, 1],
            "time": [
                "2025-01-01T00:00:00+09:00",
                "2025-01-01T00:10:00+09:00",
                "2025-01-01T00:50:00+09:00",
                "2025-01-01T01:00:00+09:00",
                "2025-01-01T01:10:00+09:00",
            ],
        }
    )
    assert exact_gap_safe_segment_ids(frame).tolist() == [0, 0, 1, 1, 2]
    design = seasonal_design(frame["time"])
    assert design.shape == (5, 13)
    assert np.array_equal(design[:, 0], np.ones(5))
    assert np.isfinite(design).all()


def test_protected_union_is_byte_exact_on_failure_and_preserves_spike() -> None:
    incumbent_probability = np.full(180, 0.1, dtype=np.float64)
    incumbent_prediction = np.zeros(180, dtype=np.int8)
    incumbent_probability[20] = 0.91
    incumbent_prediction[20] = 1
    slow_probability = np.full(180, 0.1, dtype=np.float64)
    slow_probability[15:27] = 0.9  # blocked around the incumbent singleton
    slow_probability[70:130] = 0.8  # valid fixed-duration addition
    segments = np.zeros(180, dtype=np.int64)
    fallback_probability, fallback_prediction, fallback_additions = protected_incumbent_union(
        incumbent_probability,
        incumbent_prediction,
        slow_probability,
        segments,
        gate_passed=False,
    )
    assert fallback_probability.tobytes() == incumbent_probability.tobytes()
    assert fallback_prediction.tobytes() == incumbent_prediction.tobytes()
    assert not fallback_additions.any()
    active_probability, active_prediction, additions = protected_incumbent_union(
        incumbent_probability,
        incumbent_prediction,
        slow_probability,
        segments,
        gate_passed=True,
    )
    assert additions[70:130].all()
    assert not additions[15:27].any()
    assert active_probability[20] == incumbent_probability[20]
    assert active_prediction[20] == incumbent_prediction[20] == 1
    assert np.array_equal(
        active_probability[incumbent_prediction == 1],
        incumbent_probability[incumbent_prediction == 1],
    )
    invalid_slow = slow_probability.copy()
    invalid_slow[90] = np.inf
    invalid_probability, invalid_prediction, invalid_additions = protected_incumbent_union(
        incumbent_probability,
        incumbent_prediction,
        invalid_slow,
        segments,
        gate_passed=True,
    )
    assert invalid_probability.tobytes() == incumbent_probability.tobytes()
    assert invalid_prediction.tobytes() == incumbent_prediction.tobytes()
    assert not invalid_additions.any()


def test_strict_inner_gate_fails_closed_on_any_guard() -> None:
    passing = _passing_gate_metrics()
    outcome = strict_inner_gate(passing)
    assert outcome["passed"] is True
    assert outcome["fallback"] == "APPLY_FIXED_SLOW_UNARY"
    failing = dict(passing)
    failing["worst_station_layer_f1_delta"] = -0.0020000001
    outcome = strict_inner_gate(failing)
    assert outcome["passed"] is False
    assert outcome["checks"]["worst_station_layer_nonregression"] is False
    assert outcome["fallback"] == "EXACT_INCUMBENT_BYTES"
    nonfinite = dict(passing)
    nonfinite["offset_recall_delta"] = np.inf
    outcome = strict_inner_gate(nonfinite)
    assert outcome == {
        "passed": False,
        "checks": {"all_gate_metrics_finite": False},
        "fallback": "EXACT_INCUMBENT_BYTES",
    }
    assert GATE_THRESHOLDS["minimum_mean_offset_drift_recall_delta"] == 0.04
    with pytest.raises(ValueError, match="keys differ"):
        strict_inner_gate({"micro_f1_delta": 1.0})


def test_pure_static_helper_audit_performs_no_model_operation() -> None:
    audit = static_contract_audit()
    assert audit["feature_count"] == 29
    assert audit["fallback_probability_byte_exact"] is True
    assert audit["fallback_prediction_byte_exact"] is True
    assert audit["incumbent_singleton_preserved"] is True
    assert audit["model_fits"] == 0
    assert audit["predictions_generated"] == 0
    assert audit["scores_computed"] == 0
    assert audit["test_value_reads"] == 0


def test_sealed_history_and_v9_pins_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for family, payload in config["sealed_history"].items():
        pins = {"self": payload} if family == "failure_recon" else payload["pins"]
        for pin in pins.values():
            path = ROOT / pin["path"]
            assert path.stat().st_size == pin["bytes"]
            assert _sha(path) == pin["sha256"]
    binding = config["v9_binding"]
    assert V9.stat().st_size == binding["bytes"] == 15812
    assert _sha(V9) == binding["sha256"]
    records = [json.loads(line) for line in V9.read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [3, 4, 5]
    assert records[-1]["event_sha256"] == binding["head_event_sha256"]
    assert records[-1]["payload"]["decision"]["problem"] == "P1"
    assert records[-1]["payload"]["upload_performed"] is False
    assert not V9.with_name(f"{V9.name}.append.lock").exists()


def test_canonical_check_only_is_read_only() -> None:
    data_raw = os.environ.get("P1_DATA_DIR")
    if not data_raw:
        pytest.skip("P1_DATA_DIR is required for the canonical source-boundary check")
    runner = _load_runner()
    future = json.loads(CONFIG.read_text(encoding="utf-8"))["future_output_paths_must_be_absent"]
    before = {
        "v9_bytes": V9.stat().st_size,
        "v9_sha256": _sha(V9),
        "future": {path: (ROOT / path).exists() for path in future},
    }
    result = runner.check_only(
        environ={
            "P1_WORKSPACE_ROOT": str(ROOT),
            "P1_DATA_DIR": str(Path(data_raw).resolve()),
        }
    )
    after = {
        "v9_bytes": V9.stat().st_size,
        "v9_sha256": _sha(V9),
        "future": {path: (ROOT / path).exists() for path in future},
    }
    assert result["status"].endswith("STATIC_CHECK_PASS")
    assert result["verdict"] == "STATIC_OWNER_GO_AWAIT_INDEPENDENT_QA"
    assert result["future_paths_absent"] is True
    assert result["actual_execution_authorized"] is False
    assert not any(result["static_operation_counts"].values())
    assert before == after


def test_runner_exposes_no_actual_execution_mode() -> None:
    runner = _load_runner()
    with pytest.raises(PermissionError, match="only --check-only"):
        runner.main([])
    actions = runner.build_parser()._actions
    option_strings = {value for action in actions for value in action.option_strings}
    assert "--check-only" in option_strings
    assert "--run" not in option_strings
    assert "--execute" not in option_strings
    source = RUNNER.read_text(encoding="utf-8")
    assert 'open("x")' not in source
    assert "O_EXCL" not in source
    assert "to_csv(" not in source
    assert "torch" not in source


def test_implementation_paths_are_append_only_new_family() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    implementation = config["implementation_contract"]
    assert implementation["helper"]["path"] == HELPER.relative_to(ROOT).as_posix()
    assert implementation["helper"]["sha256"] == _sha(HELPER)
    assert implementation["runner"] == RUNNER.relative_to(ROOT).as_posix()
    assert implementation["tests"] == Path(__file__).relative_to(ROOT).as_posix()
    assert implementation["runner_mode"] == "CHECK_ONLY_NO_OUTPUT"
    assert implementation["actual_execution_entrypoint_present"] is False
    assert not any((ROOT / path).exists() for path in config["future_output_paths_must_be_absent"])
