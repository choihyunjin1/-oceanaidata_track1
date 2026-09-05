import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_ocean_forward_support_20260906_v1 as audit


def test_naive_kst_and_duplicate():
    source = pd.DataFrame({"station": ["S"], "layer": [1], "time": ["2024-01-01"]})
    out = audit.normalize(source, "Asia/Seoul", ["station", "layer", "time"])
    assert out.time.iloc[0] == pd.Timestamp("2023-12-31T15:00Z")
    assert source.time.iloc[0] == "2024-01-01"
    with pytest.raises(ValueError, match="duplicate"):
        audit.normalize(pd.concat([source, source]), "Asia/Seoul", ["station", "layer", "time"])


def test_wave_actual_time_targets_and_raw_episodes():
    times = pd.date_range("2024-01-01", periods=600, freq="20min", tz="UTC")
    frame = pd.DataFrame({"station": "S", "time": times, "hs": 2.0})
    frame.loc[170, "hs"] = np.nan
    anchors = audit.wave_anchors(frame)
    # Missing future target removes an anchor but does not split its raw episode.
    assert times[161] not in set(anchors.anchor_time)
    assert times[160] in set(anchors.anchor_time)
    assert anchors.loc[anchors.anchor_time.eq(times[160]), "episode_id"].iloc[0] == 1
    assert anchors.loc[anchors.anchor_time.eq(times[200]), "episode_id"].iloc[0] == 2
    assert anchors.anchor_time.min() >= times[0] + pd.Timedelta(hours=48)
    assert anchors.anchor_time.max() <= times[-1] - pd.Timedelta(hours=24)


def test_p2_preserves_unsupported_outage_rows():
    contract = audit.cv.load_contract()
    contract["P2"]["folds"] = [contract["P2"]["folds"][2]]
    rows = [{"station": "S", "time": t, "layer": layer, "temp": 10.0, "psal": 30.0, "depth": layer}
            for t in ["2024-08-01", "2024-09-10", "2024-10-20", "2024-12-01"]
            for layer in [1, 2, 3, 4, 5]]
    result = audit.audit_p2(pd.DataFrame(rows), contract)["folds"][0]
    assert result["validation"] == 6
    assert result["outage_target_rows"] == 3
    assert result["outage_under_two_public_temp_rows"] == 3
    assert result["outage_support"] == "UNSUPPORTED_ROWS_PRESERVED"


def test_empty_wave_fails():
    frame = pd.DataFrame({"station": ["S"], "time": ["2024-01-01"], "hs": [0.0]})
    with pytest.raises(ValueError, match="no six-lead"):
        audit.wave_anchors(frame)
