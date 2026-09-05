"""Synthetic P2 tree physical-slot and temporal-mask contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("physical_tree", ROOT / "scripts/run_p2_physical_profile_tree_20260905_v2.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sample():
    records = []
    depths = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    for index, time in enumerate(pd.date_range("2024-10-17", periods=80, freq="1h", tz="UTC")):
        for layer, depth in depths.items():
            records.append({"station": "S-ORS", "time": time, "layer": layer, "nominal_depth": depth, "depth": depth + 0.1, "temp": 22 - depth / 10 + index / 100, "psal": 31 + depth / 50})
    return pd.DataFrame(records)


def test_target_poison_does_not_change_any_features():
    obs = sample()
    clean, _ = runner.base.public_frame(obs)
    obs.loc[obs.layer.isin([2, 3, 4]), ["temp", "psal"]] = 999
    dirty, _ = runner.base.public_frame(obs)
    pd.testing.assert_frame_equal(runner.physical_features(clean, runner.load_config()), runner.physical_features(dirty, runner.load_config()))


def test_49m_sensor_slot_stable_across_layer_number():
    obs = sample()
    obs = obs.loc[obs.layer != 7].copy()
    frame, _ = runner.base.public_frame(obs)
    first = runner.physical_features(frame, runner.load_config())
    obs.loc[obs.layer == 8, "layer"] = 7
    remapped, _ = runner.base.public_frame(obs)
    second = runner.physical_features(remapped, runner.load_config())
    pd.testing.assert_frame_equal(first, second)
    assert first.temp_s39.isna().all() and first.temp_s49.notna().all()


def test_outage_is_applied_before_lag_features():
    frame, _ = runner.base.public_frame(sample())
    cfg = runner.load_config()
    times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True)).unique()
    middle = times[20:60]
    altered = runner.masked_frame(frame, middle)
    features = runner.physical_features(altered, cfg)
    inside = pd.to_datetime(frame.time, utc=True).isin(middle[12:-12])
    columns = [name for name in features if "temp_s19" in name and "present" not in name]
    assert features.loc[inside, columns].isna().all().all()
    assert (features.loc[inside, "present_temp_s19"] == 0).all()


def test_original_weight_mass_and_degree_budget():
    frame, labels = runner.base.public_frame(sample())
    cfg = runner.load_config()
    cfg["blockmask"]["coverage"] = 1.0
    x = runner.physical_features(frame, cfg)
    train = np.ones(len(frame), dtype=bool)
    new_x, target, weights, receipt = runner.make_training_data(frame, labels, train, x, cfg)
    assert len(new_x) == len(target) == len(weights)
    assert receipt["weight_sum"] == len(frame)
    assert cfg["maximum_new_historical_fits"] == 3 + 2 * 3
    assert cfg["calibration_fits"] == cfg["official_access_rows"] == cfg["csv_written"] == 0


def test_no_official_source_or_output_code():
    source = (ROOT / "scripts/run_p2_physical_profile_tree_20260905_v2.py").read_text(encoding="utf-8")
    assert source.count("pd.read_csv(") == 1
    for token in ("test_index.csv", "sample_submission.csv", "baseline_interp.csv", "to_csv("):
        assert token not in source


def test_synthetic_tree_feature_names_and_saved_model_smoke(tmp_path):
    frame, labels = runner.base.public_frame(sample())
    features = runner.physical_features(frame, runner.load_config())
    model = runner.lgb.LGBMRegressor(n_estimators=3, num_leaves=3, verbosity=-1, n_jobs=1)
    model.fit(features, labels - frame.baseline)
    path = tmp_path / "synthetic.txt"
    model.booster_.save_model(str(path))
    replay = runner.lgb.Booster(model_file=str(path))
    np.testing.assert_allclose(model.predict(features), replay.predict(features, num_threads=1), rtol=0, atol=1e-12)
