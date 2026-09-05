import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ts_v4", ROOT / "scripts/run_p1_ts_disagreement_20260905_v4.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample(n=180):
    x = np.arange(n)
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": 1,
            "year": 2025,
            "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "temp": 12 + np.sin(x / 8),
            "psal": 32 + np.sin(x / 8) / 4,
            "depth": 4.0,
        }
    )


def test_labels_rejected():
    with pytest.raises(ValueError, match="raw observation"):
        m.ts_features(sample().assign(label=0), 0.01)


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        m.ts_features(sample(), eps)


def test_no_cross_gap_dependency():
    a = sample(60)
    b = sample(60)
    b["time"] = pd.date_range("2025-02-01", periods=60, freq="10min", tz="Asia/Seoul").astype(str)
    together = pd.concat([a, b], ignore_index=True)
    changed = together.copy()
    changed.loc[:59, ["temp", "psal"]] += 1000
    before, after = [m.ts_features(f, 0.01) for f in (together, changed)]
    pd.testing.assert_frame_equal(before.iloc[60:], after.iloc[60:])
    pd.testing.assert_frame_equal(before.iloc[60:].reset_index(drop=True), m.ts_features(b, 0.01))


def test_missing_salinity_is_not_quiet():
    frame = sample()
    frame["psal"] = np.nan
    features = m.ts_features(frame, 0.01)
    assert features.ts_jump_prev.isna().all()
    assert features.ts_rough_ratio_36.isna().all()
    assert features.ts_residual_36.isna().all()
    assert not np.isinf(features.to_numpy()).any()


def test_train_epsilon_label_independent():
    frame = sample()
    assert m.fit_epsilon(frame.assign(label=0)) == m.fit_epsilon(frame.assign(label=1))
    assert m.fit_epsilon(frame.assign(psal=32)) == 1e-6


def test_segment_edge_and_determinism():
    frame = sample()
    a, b = m.ts_features(frame, 0.01), m.ts_features(frame, 0.01)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (180, 19)
    assert pd.isna(a.ts_jump_prev.iloc[0])
    assert pd.isna(a.ts_jump_next.iloc[-1])
    assert not np.isinf(a.to_numpy()).any()


def test_base_features_unchanged():
    frame = sample()
    stats = m.old.stats_fit(frame)
    frozen = {
        "base_config": "configs/p1.toml",
        "flank_outer_hours": 168,
        "flank_inner_hours": 24,
        "flank_min_fraction": 0.25,
    }
    original = m.old.feature_pair(frame, stats, frozen)[0]
    candidate = m.bundle(frame, stats, frozen, 0.01)
    pd.testing.assert_frame_equal(original.frame, candidate.frame[list(original.frame)])


def test_counts_and_shape_guard():
    result = m.counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert result == {"rows": 4, "tp": 1, "fp": 1, "fn": 1, "f1": 0.5}
    with pytest.raises(ValueError):
        m.counts([0, 1], [1])


def test_synthetic_fit_saved_reload(tmp_path):
    x = m.ts_features(sample(), 0.01).fillna(0).to_numpy()
    y = (np.arange(len(x)) % 5 == 0).astype(int)
    model = m.old.lgb.LGBMClassifier(n_estimators=3, n_jobs=1, verbosity=-1, random_state=1)
    model.fit(x, y)
    prediction = model.predict_proba(x)
    path = tmp_path / "model.joblib"
    m.joblib.dump(model, path)
    assert np.array_equal(prediction, m.joblib.load(path).predict_proba(x))
