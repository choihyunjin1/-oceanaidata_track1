from datetime import datetime

from scripts import run_p2_boundary_residual_bridge_20260829_v1 as experiment


def test_boundary_windows_are_outside_the_block() -> None:
    start = datetime.fromisoformat("2025-07-01T00:00:00+09:00")
    stop = datetime.fromisoformat("2025-09-01T00:00:00+09:00")
    windows = experiment.boundary_windows(start, stop, flank_hours=72)
    assert windows["left"] == (
        datetime.fromisoformat("2025-06-28T00:00:00+09:00"),
        start,
    )
    assert windows["right"] == (
        stop,
        datetime.fromisoformat("2025-09-04T00:00:00+09:00"),
    )


def test_half_open_overlap_detects_only_positive_width() -> None:
    hidden_start = datetime.fromisoformat("2025-09-01T00:00:00+09:00")
    hidden_stop = datetime.fromisoformat("2025-11-01T00:00:00+09:00")
    overlap = experiment.interval_overlap(
        hidden_start,
        datetime.fromisoformat("2025-09-04T00:00:00+09:00"),
        hidden_start,
        hidden_stop,
    )
    assert overlap == (
        hidden_start,
        datetime.fromisoformat("2025-09-04T00:00:00+09:00"),
    )
    assert (
        experiment.interval_overlap(
            datetime.fromisoformat("2025-08-29T00:00:00+09:00"),
            hidden_start,
            hidden_start,
            hidden_stop,
        )
        is None
    )


def test_contract_audit_fails_closed_without_data_access() -> None:
    config = experiment.load_config()
    result = experiment.audit_contract(config)
    collisions = {(item["block"], item["side"]) for item in result["hidden_target_collisions"]}
    assert result["decision"] == "NO_GO_CONTRACT_LEAKAGE"
    assert result["family_status"] == "CLOSED_NO_RETRY"
    assert collisions == {("2025_jul_aug", "right"), ("2025_nov_dec", "left")}
    assert result["official_hidden_target_rows_read"] == 0
    assert result["source_observation_rows_read"] == 0
    assert result["data_paths_opened"] == []
    assert result["prediction_rows_generated"] == 0
    assert result["metric_gate_evaluated"] is False
    assert all(value is None for value in result["gate_checks"].values())


def test_exact_gate_and_no_grid_contract_is_frozen() -> None:
    config = experiment.load_config()
    bridge = config["bridge"]
    gate = config["gate"]
    assert bridge["flank_hours"] == 72
    assert bridge["interpolation"] == "cubic_smoothstep_3u2_minus_2u3"
    assert bridge["projector_applications"] == 1
    assert sum(bridge[key] for key in bridge if key.endswith("grid_size")) == 0
    assert gate["pooled_delta_rmse_max_c"] == -0.002
    assert gate["2024_sep_oct_delta_rmse_max_c"] == -0.0015
    assert gate["minimum_improved_blocks"] == 2
    assert gate["maximum_worst_block_regression_c"] == 0.0005
    assert gate["maximum_layer_regression_c"] == 0.001
    assert gate["bootstrap_ci90_upper_max_c"] == 0.0
    assert gate["maximum_absolute_axis_cosine"] == 0.3
    assert gate["maximum_correction_p99_c"] == 0.15
