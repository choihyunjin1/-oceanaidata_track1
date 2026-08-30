from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p2_state_conditioned_copula_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("p2_state_conditioned_copula", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _config() -> dict[str, object]:
    return RUNNER.load_config()


def test_sealed_config_exposes_stage0_edges_and_excludes_closed_recipe() -> None:
    config = _config()

    assert config["experiment_id"] == "p2_state_conditioned_copula_20260830_v1"
    assert tuple(config["dependence"]["stage0_exposed_edges"]) == RUNNER.EXPOSED_EDGES
    assert config["dependence"]["diagonal_shrinkage"] == 0.8
    assert config["dependence"]["shrinkage_selected_or_tuned"] is False
    assert config["dependence"]["nearest_psd_projection_allowed"] is False
    assert config["closed_family_exclusion"]["exact_closed_recipe_rerun"] is False
    assert config["closed_family_exclusion"]["seasonal_empirical_residual_margins"] is False
    assert config["closed_family_exclusion"]["inner_model_selection"] is False
    assert config["resource_contract"]["outer_dependence_model_fits"] == 3
    assert config["resource_contract"]["inner_selection_fits"] == 0
    assert config["execution_policy"]["maximum_executions"] == 1
    assert config["execution_policy"]["result_based_retry"] is False
    assert config["execution_policy"]["technical_failure_retry"] is False
    assert config["execution_policy"]["official_interface_reads_allowed"] is False
    assert config["execution_policy"]["csv_output_allowed"] is False
    canonical = json.dumps(
        config, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == RUNNER.SEALED_CONFIG_CANONICAL_SHA256


def test_state_thresholds_are_training_only_and_shared_for_assignment() -> None:
    config = _config()
    training = pd.DataFrame(
        {
            "thermal_contrast_abs": np.arange(1.0, 13.0),
            "thermal_change_24h_abs": np.arange(12.0),
        }
    )

    thresholds = RUNNER._state_thresholds(training, config)
    query = pd.DataFrame(
        {
            "thermal_contrast_abs": [0.5, thresholds["thermal_q1"], 100.0, np.nan],
            "thermal_change_24h_abs": [0.0, 100.0, 0.0, 1.0],
        }
    )
    assigned = RUNNER._assign_state_cells(query, thresholds)

    assert thresholds == RUNNER._state_thresholds(training, config)
    assert assigned["state_cell"].tolist()[:3] == [
        "thermal_low__dynamic_steady",
        "thermal_low__dynamic_active",
        "thermal_high__dynamic_steady",
    ]
    assert pd.isna(assigned.loc[3, "state_cell"])


def _synthetic_cell(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    temp = rng.normal(size=rows)
    psal = 0.25 * temp + rng.normal(size=rows)
    change = -0.15 * temp + 0.2 * psal + rng.normal(size=rows)
    residual_l2 = 0.8 * temp + rng.normal(scale=0.4, size=rows)
    residual_l3 = 0.6 * temp + rng.normal(scale=0.5, size=rows)
    residual_l4 = (
        0.4 * temp
        + 0.35 * psal
        + 0.25 * change
        + 0.2 * residual_l2
        + 0.2 * residual_l3
        + rng.normal(scale=0.5, size=rows)
    )
    return pd.DataFrame(
        {
            "temp_contrast_signed": temp,
            "psal_contrast_signed": psal,
            "thermal_change_24h_signed": change,
            "residual_l2": residual_l2,
            "residual_l3": residual_l3,
            "residual_l4": residual_l4,
            "kst_day": [f"2024-05-{index % 30 + 1:02d}" for index in range(rows)],
            "block": np.where(np.arange(rows) % 2 == 0, "block_a", "block_b"),
        }
    )


def test_sparse_fixed_cell_model_uses_only_exposed_and_nuisance_edges_with_ood_noop() -> None:
    config = _config()
    cell = _synthetic_cell()
    model = RUNNER._fit_state_cell("thermal_low__dynamic_steady", cell, config)

    expected_receipt_edges = set(RUNNER.EXPOSED_EDGES) | {
        "temp_contrast_signed__psal_contrast_signed",
        "temp_contrast_signed__thermal_change_24h_signed",
        "psal_contrast_signed__thermal_change_24h_signed",
    }
    assert set(model.receipt["edge_receipts"]) == expected_receipt_edges
    assert model.receipt["diagonal_shrinkage"] == 0.8
    assert model.receipt["nearest_psd_projection_applied"] is False
    assert model.receipt["minimum_eigenvalue"] > 0.0
    inside = cell.loc[:2, list(RUNNER.CONDITIONERS)].to_numpy()
    query = np.vstack([inside, np.asarray([[1e9, 1e9, 1e9]])])
    correction, supported = model.predict(query, maximum_absolute_latent_mean=1.0)

    assert supported.tolist() == [True, True, True, False]
    assert np.isfinite(correction).all()
    assert np.array_equal(correction[-1], np.zeros(3))


def test_worst_season_guard_is_independent_and_fixed() -> None:
    config = _config()
    base_metric = {
        "rows": 100,
        "reference_rmse": 1.0,
        "candidate_rmse": 0.99,
        "delta_rmse": -0.01,
    }
    metrics = {
        "pooled": dict(base_metric),
        "by_window": {"a": dict(base_metric), "b": dict(base_metric), "c": dict(base_metric)},
        "by_layer": {"2": dict(base_metric), "3": dict(base_metric), "4": dict(base_metric)},
        "by_season": {"JJA": dict(base_metric), "SON": dict(base_metric)},
    }
    bootstrap = {"ci90_high": -0.001}
    checks = RUNNER._gate_checks(metrics, bootstrap, np.zeros(300), config)
    assert all(checks.values())

    metrics["by_season"]["SON"]["delta_rmse"] = 0.0031
    checks = RUNNER._gate_checks(metrics, bootstrap, np.zeros(300), config)
    assert checks["worst_season_regression_lte_0_003_c"] is False


def _write_synthetic_source(p2_dir: Path) -> tuple[bytes, bytes]:
    p2_dir.mkdir()
    readme = b"Synthetic training-only source.\n"
    (p2_dir / "README.md").write_bytes(readme)
    observations = pd.DataFrame(
        [
            {
                "station": "synthetic",
                "year": 2024,
                "layer": 1,
                "time": "2024-05-01T00:00:00+09:00",
                "temp": 10.0,
                "psal": 34.0,
                "depth": 1.0,
                "nominal_depth": 1.0,
            }
        ]
    )
    observations_path = p2_dir / "observations.csv"
    observations.to_csv(observations_path, index=False)
    payload = observations_path.read_bytes()
    for forbidden in (
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
        "score.py",
        "submission.csv",
        "query_support.json",
    ):
        (p2_dir / forbidden).write_text("must-not-open\n", encoding="utf-8")
    return readme, payload


def test_source_reader_opens_only_allowlisted_training_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(_config())
    p2_dir = tmp_path / "explicit-p2"
    readme, observations = _write_synthetic_source(p2_dir)
    config["source_contract"]["readme"] = {
        "bytes": len(readme),
        "sha256": hashlib.sha256(readme).hexdigest(),
    }
    config["source_contract"]["observations"] = {
        "bytes": len(observations),
        "sha256": hashlib.sha256(observations).hexdigest(),
    }
    original_open = Path.open
    opened: list[str] = []
    source_root = p2_dir.resolve()

    def recording_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.parent.resolve() == source_root:
            opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    frame, receipt, access = RUNNER._read_training_source(p2_dir, config)

    assert len(frame) == 1
    assert set(opened) == {"README.md", "observations.csv"}
    assert access.open_counts == {"README.md": 1, "observations.csv": 1}
    assert receipt["observations"]["rows"] == 1


def test_exclusive_result_writer_refuses_source_and_overwrite(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    p2_dir = tmp_path / "p2"
    p2_dir.mkdir()
    inside = p2_dir / "result.json"
    config["artifact_path"] = str(inside)
    with pytest.raises(RUNNER.ExperimentContractError, match="inside --p2-dir"):
        RUNNER._write_result({"aggregate": True}, inside, p2_dir, config)

    output = tmp_path / "result.json"
    config["artifact_path"] = str(output)
    RUNNER._write_result({"aggregate": True}, output, p2_dir, config)
    assert json.loads(output.read_text(encoding="utf-8")) == {"aggregate": True}
    with pytest.raises(FileExistsError):
        RUNNER._write_result({"aggregate": False}, output, p2_dir, config)
