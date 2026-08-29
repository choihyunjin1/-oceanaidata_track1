from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_mstcn_sobol_hpo_20260829_v1.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p1_mstcn_sobol_hpo_20260829_v1.json"


def _load_runner():
    name = "p1_mstcn_sobol_hpo_20260829_v1_tested"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_config_fixes_one_shot_historical_contract() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "p1_mstcn_sobol_hpo_20260829_v1"
    assert config["sobol_design"]["point_count"] == 32
    assert config["sobol_design"]["random_base2_m"] == 5
    assert config["training_contract"]["stop_epoch"] == 150
    assert config["training_contract"]["source_schedule_horizon_epochs"] == 300
    assert config["training_contract"]["threshold_grid"] == [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert config["selection_contract"]["surface"] == "2025_q2 historical only"
    assert config["confirmation_contract"]["winner_count"] == 1
    assert all(config["prohibitions"].values())


def test_sobol_design_is_deterministic_balanced_and_bounded() -> None:
    runner = _load_runner()
    config = runner._config()
    first = runner.generate_design(config)
    second = runner.generate_design(config)
    assert runner._json_bytes(first) == runner._json_bytes(second)
    points = first["points"]
    assert len(points) == 32
    assert [row["trial_id"] for row in points] == [f"trial_{index:02d}" for index in range(32)]
    assert sum(row["width"] == 256 for row in points) == 16
    assert sum(row["width"] == 512 for row in points) == 16
    assert points[0]["sobol_u"][0] == pytest.approx(0.3885430870577693)
    assert points[-1]["sobol_u"][-1] == pytest.approx(0.008950674906373024)
    for row in points:
        assert 0.05 <= row["dropout"] <= 0.30
        assert 1e-4 <= row["learning_rate"] <= 8e-4
        assert 1e-6 <= row["weight_decay"] <= 1e-3
        assert 0.5 <= row["row_soft_dice_weight"] <= 2.0
        assert 0.05 <= row["temporal_smoothing_weight"] <= 0.30
        assert 0.10 <= row["boundary_type_weight"] <= 0.40


def test_trial_config_changes_only_registered_hpo_axes() -> None:
    runner = _load_runner()
    config = runner._config()
    base = runner._load_base()
    source = base._canonical_config()
    trial = runner.generate_design(config)["points"][0]
    candidate = runner._trial_config(base, source, trial, 20260827)
    assert candidate["windowing"] == source["windowing"]
    assert candidate["decoder"] == source["decoder"]
    assert candidate["architecture"]["dual_dilations"] == source["architecture"]["dual_dilations"]
    assert candidate["architecture"]["refinement_stages"] == 3
    assert candidate["architecture"]["width"] == trial["width"]
    assert candidate["architecture"]["dropout"] == trial["dropout"]
    assert candidate["training"]["batch_size"] == trial["batch_size"]
    assert candidate["training"]["seed"] == 20260827
    assert candidate["training"]["maximum_epochs"] == 300
    assert candidate["training"]["loss_weights"]["row_bce"] == 1.0


def test_rank_and_gate_are_lexicographic_and_strict() -> None:
    runner = _load_runner()
    records = [
        {
            "trial": {"trial_index": 0, "width": 512},
            "threshold": 0.8,
            "metrics": {"minimum_monthly_delta_f1": 0.001, "pooled_delta_f1": 0.01},
        },
        {
            "trial": {"trial_index": 1, "width": 256},
            "threshold": 0.7,
            "metrics": {"minimum_monthly_delta_f1": 0.002, "pooled_delta_f1": 0.003},
        },
    ]
    assert runner._rank_records(records)[0]["trial"]["trial_index"] == 1
    monthly = {month: {"delta_f1": value} for month, value in zip(
        ("2025-04", "2025-05", "2025-06"), (0.001, 0.002, 0.003), strict=True
    )}
    assert all(row["delta_f1"] > 0.0 for row in monthly.values())
    monthly["2025-04"]["delta_f1"] = 0.0
    assert not all(row["delta_f1"] > 0.0 for row in monthly.values())


def test_check_only_is_read_only_and_caps_threads() -> None:
    runner = _load_runner()
    artifact_existed = runner.ARTIFACT_DIR.exists()
    lock_existed = runner.ATTEMPT_LOCK.exists()
    result = runner.check_only()
    assert result["result"] == "PASS"
    assert result["design_points"] == 32
    assert result["torch_threads"]["intraop"] <= 2
    assert result["torch_threads"]["interop"] <= 1
    assert result["official_interface_rows_read"] == 0
    assert runner.ARTIFACT_DIR.exists() is artifact_existed
    assert runner.ATTEMPT_LOCK.exists() is lock_existed


def test_cli_requires_reviewed_hash_and_no_protected_interface_literals() -> None:
    runner = _load_runner()
    assert runner._parse_args(["--check-only"]).check_only
    assert runner._parse_args(["--smoke"]).smoke
    with pytest.raises(SystemExit):
        runner._parse_args([])
    with pytest.raises(runner.ContractError, match="reviewed runner bytes"):
        runner.execute(expected_runner_sha256="0" * 64)
    source = RUNNER_PATH.read_text(encoding="utf-8").casefold()
    protected = ["sample_" + "submission", "test." + "csv", "submission." + "csv"]
    assert not any(fragment in source for fragment in protected)
    assert "requests." not in source
    assert "selenium" not in source


def test_confirmatory_metric_uses_incumbent_control() -> None:
    runner = _load_runner()
    base = runner._load_base()
    truth = {
        "q3": __import__("pandas").DataFrame({"label": [1, 1, 0, 0]}),
        "q4": __import__("pandas").DataFrame({"label": [1, 0, 1, 0]}),
    }

    class Surface:
        anchor = np.asarray([1, 0, 0, 0], dtype=np.int8)

    class Hold:
        surface = Surface()

    candidates = {
        "q3": np.asarray([1, 1, 0, 0], dtype=np.int8),
        "q4": np.asarray([1, 0, 1, 0], dtype=np.int8),
    }
    controls = {
        "q3": np.asarray([1, 0, 0, 0], dtype=np.int8),
        "q4": np.asarray([1, 0, 0, 0], dtype=np.int8),
    }
    result = runner._evaluate_confirmatory(
        base, truth, {"q3": Hold(), "q4": Hold()}, candidates, controls
    )
    assert result["decision"] == "PASS"
    assert result["folds"]["q3"]["delta_f1"] > 0.0
    assert result["folds"]["q4"]["delta_f1"] > 0.0
