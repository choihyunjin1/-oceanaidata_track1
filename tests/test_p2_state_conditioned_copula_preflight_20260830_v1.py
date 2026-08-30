from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p2_state_conditioned_copula_preflight_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("p2_state_conditioned_copula_preflight", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _config() -> dict[str, object]:
    return json.loads(RUNNER.CONFIG_PATH.read_text(encoding="utf-8"))


def test_preregistration_seals_zero_fit_sources_and_support_gates() -> None:
    config = _config()

    assert config["experiment_id"] == "p2_state_conditioned_copula_preflight_20260830_v1"
    assert config["source_contract"]["allowed_basenames"] == [
        "observations.csv",
        "README.md",
    ]
    assert set(config["source_contract"]["forbidden_name_fragments"]) == {
        "test_index",
        "sample",
        "baseline",
        "score",
        "submission",
    }
    assert config["state_definition"]["degrees_of_freedom"] == 2
    assert config["gates"]["minimum_kendall_tau_span"] == 0.10
    assert config["gates"]["minimum_profiles_per_state_cell"] == 500
    assert config["gates"]["minimum_kst_days_per_state_cell"] == 30
    assert config["gates"]["minimum_chronological_blocks_per_state_cell"] == 2
    assert config["gates"]["minimum_passing_edges"] == 2
    assert config["overlap_policy"]["minimum_supported_state_cells_for_tau_span"] == 2
    assert (
        config["overlap_policy"]["unsupported_state_cell_action"]
        == "EXACT_NO_DEPENDENCE_ADJUSTMENT"
    )
    assert (
        config["heterogeneity_policy"]["comparison_unit"]
        == "same_supported_state_cell_pair_across_blocks"
    )
    assert config["heterogeneity_policy"][
        "require_block_direction_to_match_pooled_direction"
    ]
    estimand = config["scientific_estimand_contract"]
    assert estimand["dependence_estimand"] == (
        "ordinary_unweighted_kendall_tau_b_on_all_finite_eligible_profiles"
    )
    assert estimand["diagnostic_weights_enter_state_cutpoints"] is False
    assert estimand["diagnostic_weights_enter_support_counts"] is False
    assert estimand["diagnostic_weights_enter_kendall_tau"] is False
    assert estimand["physical_extrema_remain_in_estimand"] is True
    assert config["execution_policy"]["model_fit_count"] == 0
    assert config["execution_policy"]["official_input_rows_read"] == 0
    assert config["execution_policy"]["aggregate_json_only"] is True
    assert config["execution_policy"]["real_data_execution_authorized"] is True
    RUNNER._validate_config(config)
    changed = json.loads(json.dumps(config))
    changed["outlier_diagnostic"]["temporal_isolated_robust_z"] = 7.5
    with pytest.raises(RUNNER.PreflightError, match="preregistered config changed"):
        RUNNER._validate_config(changed)


def _supported_synthetic_profiles(config: dict[str, object]) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    blocks = ["2024_may_jun", "2024_jul_aug"]
    starts = [pd.Timestamp("2024-05-01", tz=RUNNER.KST), pd.Timestamp("2024-07-01", tz=RUNNER.KST)]
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(RUNNER._expected_state_cells(config)):
        thermal_label = cell.split("__", maxsplit=1)[0]
        slope = {"thermal_low": 1.0, "thermal_middle": 0.15, "thermal_high": -1.0}[
            thermal_label
        ]
        for block, start in zip(blocks, starts, strict=True):
            x = np.linspace(-2.0, 2.0, 250) + rng.normal(0.0, 0.002, 250)
            response = slope * x + rng.normal(0.0, 0.02, 250)
            for index in range(250):
                timestamp = start + pd.Timedelta(days=index % 30, minutes=index // 30)
                rows.append(
                    {
                        "station": f"synthetic_{cell_index}",
                        "time": timestamp,
                        "kst_day": timestamp.strftime("%Y-%m-%d"),
                        "block": block,
                        "state_cell": cell,
                        "temp_contrast_signed": float(x[index]),
                        "psal_contrast_signed": float(np.sin(x[index]) + 0.01 * index),
                        "thermal_change_24h_signed": float(np.cos(x[index]) + 0.005 * index),
                        "residual_l2": float(response[index]),
                        "residual_l3": float(0.7 * response[index] + 0.01 * x[index]),
                        "residual_l4": float(-0.4 * response[index] + 0.02 * index),
                        "sensor_suspect": False,
                        "joint_ts_discontinuity": False,
                        "coherent_multilayer_temp_event": False,
                        "physical_extreme_any": False,
                        "preserved_physical_extreme": False,
                        "diagnostic_weight": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_repeated_chronological_state_heterogeneity_passes_all_sealed_gates() -> None:
    config = _config()
    all_profiles = _supported_synthetic_profiles(config)
    supported_cells = {
        "thermal_low__dynamic_steady",
        "thermal_high__dynamic_steady",
    }
    profiles = all_profiles.loc[all_profiles["state_cell"].isin(supported_cells)].copy()

    support = RUNNER._state_support_receipts(profiles, config)
    heterogeneity = RUNNER._kendall_heterogeneity(profiles, config, supported_cells)
    checks = RUNNER._gate_checks(support, heterogeneity, config)

    assert len(support) == 6
    assert sum(item["passes_overlap_support_gate"] for item in support) == 2
    assert all(
        item["profiles"] == 500
        and item["kst_days"] >= 30
        and item["chronological_blocks"] == 2
        for item in support
        if item["passes_overlap_support_gate"]
    )
    assert checks == {
        "at_least_two_supported_state_cells": True,
        "evaluated_state_cells_profiles_gte_500": True,
        "evaluated_state_cells_kst_days_gte_30": True,
        "evaluated_state_cells_chronological_blocks_gte_2": True,
        "kendall_tau_span_gte_0_10_in_at_least_two_blocks": True,
    }
    assert set(heterogeneity["evaluated_supported_state_cells"]) == supported_cells
    assert len(heterogeneity["exact_no_op_unsupported_state_cells"]) == 4
    target_edge = next(
        item
        for item in heterogeneity["edge_heterogeneity"]
        if item["edge"] == "temp_contrast_signed__residual_l2"
    )
    assert target_edge["pooled_tau_span"] >= 0.10
    assert target_edge["heterogeneous_block_count"] >= 2
    assert target_edge["passing_state_cell_pairs"] == [
        "thermal_low__dynamic_steady__vs__thermal_high__dynamic_steady"
    ]
    assert target_edge["passes_predeclared_heterogeneity_gate"] is True


def test_opposite_block_direction_kills_same_pair_heterogeneity_gate() -> None:
    config = _config()
    supported_cells = {
        "thermal_low__dynamic_steady",
        "thermal_high__dynamic_steady",
    }
    profiles = _supported_synthetic_profiles(config)
    profiles = profiles.loc[profiles["state_cell"].isin(supported_cells)].copy()
    first_block = profiles.loc[profiles["block"].eq("2024_may_jun")].copy()
    first_block["time"] = first_block["time"] + pd.Timedelta(hours=1)
    first_block["kst_day"] = first_block["time"].dt.strftime("%Y-%m-%d")
    profiles = pd.concat([profiles, first_block], ignore_index=True)
    opposite = profiles["block"].eq("2024_jul_aug")
    profiles.loc[opposite, "residual_l2"] *= -1.0

    heterogeneity = RUNNER._kendall_heterogeneity(profiles, config, supported_cells)
    target_edge = next(
        item
        for item in heterogeneity["edge_heterogeneity"]
        if item["edge"] == "temp_contrast_signed__residual_l2"
    )

    assert target_edge["pooled_tau_span"] >= 0.10
    assert all(
        span is not None and span >= 0.10
        for span in target_edge["block_tau_spans"].values()
        if span is not None
    )
    assert target_edge["passing_state_cell_pairs"] == []
    assert target_edge["heterogeneous_block_count"] == 1
    assert target_edge["passes_predeclared_heterogeneity_gate"] is False


def test_outlier_diagnostic_weights_isolated_spike_and_preserves_extrema() -> None:
    config = _config()
    flags = pd.DataFrame(
        {
            "temp_spike_layer_count": [1, 2, 0],
            "psal_spike_layer_count": [0, 2, 0],
            "joint_ts_discontinuity": [False, True, False],
            "physical_extreme_any": [True, True, True],
        }
    )

    profiles = RUNNER._profile_outlier_policy(flags, config)
    profiles["block"] = "2024_may_jun"
    profiles["state_cell"] = "thermal_low__dynamic_steady"
    receipt = RUNNER._aggregate_outlier_diagnostics(profiles)

    assert profiles["sensor_suspect"].tolist() == [True, False, False]
    assert profiles["coherent_multilayer_temp_event"].tolist() == [False, True, False]
    assert profiles["coherent_physical_extreme"].tolist() == [False, True, True]
    assert profiles["preserved_physical_extreme"].tolist() == [True, True, True]
    assert profiles["diagnostic_weight"].tolist() == [0.25, 1.0, 1.0]
    assert receipt["global"]["hard_deleted_profiles"] == 0
    assert receipt["global"]["coherent_physical_extreme_profiles"] == 2
    assert receipt["global"]["preserved_physical_extreme_profiles"] == 3
    assert receipt["global"]["diagnostic_weight_sum"] == pytest.approx(2.25)
    assert receipt["diagnostic_weights_enter_kendall_tau"] is False
    assert receipt["physical_extrema_remain_in_estimand"] is True


def _write_tiny_training_source(p2_dir: Path) -> None:
    p2_dir.mkdir()
    (p2_dir / "README.md").write_text("Synthetic training-only fixture.\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for day, top_temp in enumerate([10.0, 11.0], start=1):
        timestamp = f"2024-05-{day:02d}T00:00:00+09:00"
        for layer, depth, temp in (
            (1, 1.0, top_temp),
            (2, 7.04, 9.5),
            (3, 9.44, 9.0),
            (4, 14.74, 8.5),
            (5, 20.0, 8.0),
        ):
            rows.append(
                {
                    "station": "synthetic_station",
                    "year": 2024,
                    "layer": layer,
                    "time": timestamp,
                    "temp": temp,
                    "psal": 34.0 + 0.01 * layer,
                    "depth": depth,
                    "nominal_depth": depth,
                }
            )
    pd.DataFrame(rows).to_csv(p2_dir / "observations.csv", index=False)
    for forbidden in (
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
        "score.py",
        "submission.csv",
    ):
        (p2_dir / forbidden).write_text("must-not-open\n", encoding="utf-8")


def test_audit_opens_only_readme_and_observations_from_explicit_p2_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    p2_dir = tmp_path / "explicit-p2-dir"
    _write_tiny_training_source(p2_dir)
    original_open = Path.open
    opened_from_p2: list[str] = []
    source_root = p2_dir.resolve()

    def recording_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.parent.resolve() == source_root:
            opened_from_p2.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    result = RUNNER.audit(p2_dir, config)

    assert set(opened_from_p2) == {"README.md", "observations.csv"}
    assert result["source_basenames_opened"] == ["README.md", "observations.csv"]
    assert result["source_open_counts"] == {"README.md": 1, "observations.csv": 2}
    assert result["status"] == "NO_GO_STATE_CONDITIONED_COPULA_PREFLIGHT"
    assert result["model_fit_count"] == 0
    assert result["official_input_rows_read"] == 0
    assert result["csv_output_count"] == 0
    assert result["submission_generated"] is False


def test_aggregate_output_cannot_enter_source_directory_or_overwrite(
    tmp_path: Path,
) -> None:
    p2_dir = tmp_path / "p2"
    p2_dir.mkdir()
    inside = p2_dir / "aggregate.json"
    with pytest.raises(RUNNER.PreflightError, match="inside --p2-dir"):
        RUNNER._write_aggregate_json({"aggregate": True}, inside, p2_dir)

    output = tmp_path / "aggregate.json"
    RUNNER._write_aggregate_json({"aggregate": True}, output, p2_dir)
    assert json.loads(output.read_text(encoding="utf-8")) == {"aggregate": True}
    with pytest.raises(FileExistsError):
        RUNNER._write_aggregate_json({"aggregate": False}, output, p2_dir)
