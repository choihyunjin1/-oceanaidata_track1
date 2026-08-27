from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_p2_nasa_power_residual_meta.py"
    spec = importlib.util.spec_from_file_location("run_p2_nasa_power_residual_meta", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_manifest_validation_is_fail_closed(tmp_path: Path) -> None:
    module = _module()
    raw = tmp_path / "power.json"
    raw.write_text("{}", encoding="utf-8")
    transformation = "unmodified fixture"
    entry = {
        "year": 2024,
        "path": str(raw),
        "manifest_path": str(tmp_path / "manifest.json"),
        "file_sha256": _hash(raw),
        "observed_start": "2024-01-01T00:00:00+00:00",
        "observed_end": "2024-01-01T00:00:00+00:00",
        "row_count": 1,
        "transformation_log": transformation,
    }
    manifest = {
        "schema_version": "1.0",
        "source_id": "nasa_power_kors_meteorology",
        "local_file": str(raw),
        "file_sha256": entry["file_sha256"],
        "observed_start": entry["observed_start"],
        "observed_end": entry["observed_end"],
        "row_count": 1,
        "variables": list(module.POWER_PARAMETERS),
        "transformation_log": transformation,
    }
    manifest_path = Path(entry["manifest_path"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    entry["manifest_sha256"] = _hash(manifest_path)

    validated = module._validate_raw_entry(entry)
    assert validated["sha256"] == entry["file_sha256"]

    raw.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="raw NASA POWER SHA changed"):
        module._validate_raw_entry(entry)


def test_alpha_selection_uses_only_fixed_grid() -> None:
    module = _module()
    truth = np.array([1.0, 2.0, 3.0])
    incumbent = np.array([0.0, 1.0, 2.0])
    correction = np.ones(3)
    alpha, scores = module._choose_alpha(
        truth,
        incumbent,
        correction,
        (0.0, 0.25, 0.5, 1.0),
    )
    assert alpha == 1.0
    assert [row["alpha"] for row in scores] == [0.0, 0.25, 0.5, 1.0]


def test_outer_held_truth_cannot_change_model_or_alpha() -> None:
    module = _module()
    rng = np.random.default_rng(17)
    blocks = np.repeat(["a", "b", "c"], 40)
    feature = rng.normal(size=len(blocks))
    incumbent = 10.0 + 0.2 * feature
    residual = 0.6 * feature + np.where(blocks == "b", 0.1, 0.0)
    frame = pd.DataFrame(
        {
            "truth": incumbent + residual,
            "incumbent_prediction": incumbent,
            "block": blocks,
        }
    )
    features = pd.DataFrame({"public_feature": feature})
    parameters = {
        "objective": "regression_l2",
        "n_estimators": 12,
        "learning_rate": 0.08,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 9,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }
    prediction, detail = module._outer_fold_prediction(
        frame,
        features,
        ("public_feature",),
        "c",
        (0.0, 0.5, 1.0),
        parameters,
    )
    tampered = frame.copy()
    tampered.loc[tampered["block"].eq("c"), "truth"] = 1_000_000.0
    changed_prediction, changed_detail = module._outer_fold_prediction(
        tampered,
        features,
        ("public_feature",),
        "c",
        (0.0, 0.5, 1.0),
        parameters,
    )

    np.testing.assert_array_equal(prediction, changed_prediction)
    assert detail["selected_alpha"] == changed_detail["selected_alpha"]
    assert detail["inner_alpha_scores"] == changed_detail["inner_alpha_scores"]


def test_promotion_fails_when_incremental_gain_is_below_preregistered_minimum() -> None:
    module = _module()
    incremental = {
        "delta_rmse": -0.004,
        "by_block": {
            "a": {"delta_rmse": -0.01},
            "b": {"delta_rmse": -0.01},
            "c": {"delta_rmse": 0.0},
        },
        "by_layer": {
            "2": {"delta_rmse": 0.0},
            "3": {"delta_rmse": 0.0},
            "4": {"delta_rmse": 0.0},
        },
    }
    final = {
        "delta_rmse": -0.02,
        "by_layer": {
            "2": {"delta_rmse": -0.02},
            "3": {"delta_rmse": -0.02},
            "4": {"delta_rmse": -0.02},
        },
    }
    gate = {
        "minimum_external_incremental_improvement_c": 0.01,
        "external_bootstrap_ci90_high_max": 0.0,
        "minimum_external_improved_blocks": 2,
        "maximum_external_layer_regression_c": 0.005,
        "maximum_final_layer_regression_vs_incumbent_c": 0.01,
        "pass_action": "GO",
        "fail_action": "NO_GO",
    }
    decision = module._promotion_decision(
        incremental,
        {"ci90_high": -0.001},
        final,
        gate,
    )
    assert decision["passed"] is False
    assert decision["decision"] == "NO_GO"
    assert decision["checks"]["external_incremental_at_least_0_010c"] is False
