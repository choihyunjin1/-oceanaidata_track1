import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p2_l120_s3_smooth7_projection_20260907_v1 import smooth7  # noqa: E402


def frame(minutes):
    return pd.DataFrame({"station":"S", "layer":2,
                         "time":pd.Timestamp("2025-01-01",tz="UTC")+pd.to_timedelta(minutes,unit="min")})


def test_grid_gap_not_row_window():
    np.testing.assert_array_equal(smooth7(frame([0,10,100]),[0,2,30]),[1,1,30])


def test_group_and_order_preserved():
    a=frame([20,0,10])
    b=a.assign(layer=3)
    q=pd.concat([a,b],ignore_index=True)
    np.testing.assert_array_equal(smooth7(q,[0,3,6,30,30,30]),[3,3,3,30,30,30])


def test_exact_seven_steps():
    q=frame(list(range(0,90,10)))
    p=np.arange(9.)**2
    assert smooth7(q,p)[4] == np.mean(p[1:8])


def test_duplicate_rejected():
    with pytest.raises(ValueError):
        smooth7(frame([0,0]),[1,2])
