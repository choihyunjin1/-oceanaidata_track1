"""Synthetic-only contracts: never open competition data or historical answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_score_repair", ROOT / "scripts/run_p2_score_repair_20260905_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def synthetic_observations():
    records = []
    for time in pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC"):
        for layer in range(1, 9):
            records.append({"station": "S-ORS", "time": time, "layer": layer, "nominal_depth": layer * 5.0, "depth": layer * 5.0 + 0.7, "temp": 20.0 - layer * 0.3, "psal": 30.0 + layer * 0.1})
    return pd.DataFrame(records)


def test_masked_target_temp_and_psal_never_change_model_inputs():
    original = synthetic_observations()
    corrupted = original.copy()
    corrupted.loc[corrupted.layer.isin((2, 3, 4)), ["temp", "psal"]] = np.nan
    first, _ = runner.public_frame(original)
    second, labels = runner.public_frame(corrupted)
    assert np.isnan(labels).all()
    for left, right in zip(runner.arrays(first), runner.arrays(second), strict=True):
        np.testing.assert_array_equal(left, right)
    assert not set(("temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4")) & set(first)


def test_restoration_purge_uses_both_sides_and_excludes_whole_gap():
    times = pd.date_range("2024-08-01", "2024-12-01", tz="Asia/Seoul")
    fold = runner.load_config()["folds"][0]
    train, valid = runner.restoration_masks(times, fold, 7)
    assert train[times < "2024-08-25"].all()
    assert train[times >= "2024-11-08"].all()
    assert not train[(times >= "2024-08-25") & (times < "2024-11-08")].any()
    assert valid.sum() == 61 and not np.any(train & valid)


def test_interpolation_endpoint_contract_not_extrapolation():
    frame, _ = runner.public_frame(synthetic_observations())
    frame["target_depth"] = -10
    np.testing.assert_allclose(runner.nominal_baseline(frame), frame.temp_1)
    frame["target_depth"] = 1000
    np.testing.assert_allclose(runner.nominal_baseline(frame), frame.temp_8)


def test_blockmask_recomputes_baseline_scale_and_preserves_total_weight():
    frame, truth = runner.public_frame(synthetic_observations())
    config = runner.load_config()
    config["blockmask"]["coverage"] = 1.0
    original, _ = runner.training_arrays(frame, truth, "v23", config)
    augmented, receipt = runner.training_arrays(frame, truth, "v23_blockmask", config)
    assert receipt["augmented_rows"] == len(frame)
    assert len(augmented[0]) == 2 * len(frame)
    np.testing.assert_allclose(augmented[4][:len(frame)] + augmented[4][len(frame):], original[4])
    assert not augmented[1][len(frame):, 1].any()
    assert not np.array_equal(augmented[0][:len(frame)], augmented[0][len(frame):])


def test_actual_depth_is_independent_ablation_with_presence_fallback():
    frame, _ = runner.public_frame(synthetic_observations())
    original = runner.arrays(frame)
    altered = frame.copy()
    altered.loc[0, "target_actual_depth"] = np.nan
    actual = runner.arrays(altered, True)
    assert original[2].shape[1] == 11 and actual[2].shape[1] == 13
    np.testing.assert_array_equal(original[0], actual[0])
    np.testing.assert_array_equal(original[2], actual[2][:, :11])
    assert actual[2][0, -1] == 0
    assert actual[2][0, -2] == pytest.approx(frame.target_depth.iloc[0] / 50)


@pytest.mark.parametrize("arm", ["v23", "v52", "v23_actualdepth", "v52_actualdepth"])
def test_reused_models_forward_and_second_order_gradient(arm):
    frame, truth = runner.public_frame(synthetic_observations())
    data, _ = runner.training_arrays(frame, truth, arm, runner.load_config())
    model = runner.make_model(arm, data[2].shape[1])
    tokens = torch.from_numpy(data[0]).requires_grad_(True)
    mask, context, labels, weights = map(torch.from_numpy, data[1:])
    result = model(tokens, mask, context)
    loss = torch.nn.functional.smooth_l1_loss(result, labels, reduction="none")
    penalty = runner.observed_temperature_gradient_penalty(loss, tokens, mask, weights)
    (loss.mean() + 0.01 * penalty).backward()
    assert torch.isfinite(result).all() and torch.isfinite(penalty)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_metric_pooled_sse_not_fold_mean():
    truth = np.zeros(4)
    prediction = np.asarray([1.0, 2.0, 3.0, 4.0])
    result = runner.metrics(truth, prediction)
    assert result["n"] == 4 and result["sse"] == 30
    assert result["rmse"] == pytest.approx(np.sqrt(7.5))


def test_fixed_fit_arithmetic_and_resources():
    config = runner.load_config()
    assert 2 * 3 + 3 + 3 + 2 * 2 * 3 == config["maximum_deepset_fits"]
    assert config["cpu_threads"] == 1 and config["loader_workers"] == 0
    assert config["official_access"] == config["csv_written"] == config["upload"] == 0


def test_no_forbidden_source_reader_or_old_prediction_reuse():
    source = (ROOT / "scripts/run_p2_score_repair_20260905_v1.py").read_text(encoding="utf-8")
    assert source.count("pd.read_csv(") == 1
    assert 'source = Path(source_dir).resolve() / "observations.csv"' in source
    for forbidden in ("load_p2_data(", "test_index.csv", "sample_submission.csv", "baseline_interp.csv", "read_parquet(", "to_csv("):
        assert forbidden not in source


def test_missing_run_summary_no_rows():
    receipt = runner.missing_runs(np.asarray([False, True, True, False, True]))
    assert receipt == {"missing_count": 3, "run_count": 2, "max_run_steps": 2, "runs_ge_144_steps": 0}


def test_oas_heldout_target_temperature_and_salinity_invariance():
    original = synthetic_observations()
    base = original.iloc[:8].copy()
    frames = []
    for day in range(60):
        part = base.copy()
        part["time"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=day)
        part["temp"] += np.sin(day / 9)
        part["psal"] += np.cos(day / 9)
        frames.append(part)
    observations = pd.concat(frames, ignore_index=True)
    panel, _, targets = runner.build_layer_identity_panel(observations)
    left, right = pd.Timestamp("2025-01-25", tz="UTC"), pd.Timestamp("2025-01-28", tz="UTC")
    inside = (panel.index >= left) & (panel.index < right)
    query = observations.loc[observations.time.between(left, right, inclusive="left") & observations.layer.isin((2, 3, 4)), ["time", "layer"]]
    altered = panel.copy()
    altered.loc[inside, targets] = 999999.0
    arguments = dict(train_stop=pd.Timestamp("2025-03-02", tz="UTC"), exclude_start=left - pd.Timedelta(days=7), exclude_stop=right + pd.Timedelta(days=7), minimum_season_rows=3)
    clean, _ = runner.predict_forward_seasonal_oas(panel, query, **arguments)
    poisoned, _ = runner.predict_forward_seasonal_oas(altered, query, **arguments)
    np.testing.assert_array_equal(clean, poisoned)
