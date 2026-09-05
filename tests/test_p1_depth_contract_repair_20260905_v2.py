import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "depth_repair", ROOT / "scripts" / "run_p1_depth_contract_repair_20260905_v2.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample():
    n = 48
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * n,
            "layer": [1] * n,
            "year": [2025] * n,
            "time": pd.date_range("2025-03-01", periods=n, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
            "temp": np.sin(np.arange(n) / 3) + 10,
            "psal": np.full(n, 32.0),
            "depth": np.full(n, 10.0),
            "label": np.zeros(n, dtype=np.int8),
        }
    )


def frozen():
    return {"base_config": "configs/p1.toml", "flank_outer_hours": 168}


def test_year_rename_invariant():
    frame = sample()
    stats = m.old.stats_fit(frame)
    before = m.features(frame, stats, frozen(), current_depth=True).frame
    after = m.features(frame.assign(year=2026), stats, frozen(), current_depth=True).frame
    pd.testing.assert_frame_equal(before, after)


def test_missing_and_unseen_not_conflated():
    frame = sample()
    stats = m.old.stats_fit(frame)
    frame.loc[0, "depth"] = np.nan
    frame["year"] = 2026
    bundle = m.features(frame, stats, frozen(), current_depth=True)
    audit = m.depth_audit(frame, bundle, stats)
    assert audit["nominal_depth_missing"] == 1
    assert audit["unseen_year_key_rows"] == len(frame)
    assert audit["observed_depth_but_nominal_missing"] == 0
    assert bundle.frame.depth_regime.iloc[0] == "S-ORS|unknown|l1"


def test_legacy_only_two_features_differ():
    frame = sample().assign(year=2026)
    stats = m.old.stats_fit(sample())
    legacy = m.features(frame, stats, frozen(), current_depth=False).frame
    new = m.features(frame, stats, frozen(), current_depth=True).frame
    columns = [c for c in legacy if c not in ("nominal_depth_m", "depth_regime")]
    pd.testing.assert_frame_equal(legacy[columns], new[columns])
    assert legacy.nominal_depth_m.isna().all()
    assert new.nominal_depth_m.notna().all()


def test_order_and_immutable_input():
    frame = sample().iloc[::-1].reset_index(drop=True)
    saved = frame.copy(deep=True)
    result = m.features(frame, m.old.stats_fit(frame), frozen(), current_depth=True)
    pd.testing.assert_frame_equal(frame, saved)
    np.testing.assert_array_equal(result.frame.temp_raw, frame.temp.astype(np.float32))


@pytest.mark.parametrize("depth,expected", [(9.1, 10.0), (8.9, 8.0), (11.9, 12.0)])
def test_fixed_depth_rounding(depth, expected):
    frame = sample().assign(depth=depth)
    bundle = m.features(frame, m.old.stats_fit(frame), frozen(), current_depth=True)
    assert (bundle.frame.nominal_depth_m == expected).all()


def test_exact_legacy_feature_reimplementation():
    import json

    cfg = json.loads((ROOT / "configs/experiments/p1_score_repair_20260905_v1.json").read_text())
    frame = sample()
    stats = m.old.stats_fit(frame)
    new = m.features(frame, stats, cfg, current_depth=False)
    old = m.old.feature_pair(frame, stats, cfg)[0]
    pd.testing.assert_frame_equal(new.frame, old.frame)


def test_no_official_reader_in_runner():
    source = (ROOT / "scripts/run_p1_depth_contract_repair_20260905_v2.py").read_text()
    assert '"test.csv"' not in source and '"sample_submission.csv"' not in source
    assert ".to_csv(" not in source
