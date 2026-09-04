from __future__ import annotations

import pandas as pd
import pytest

from p2_restore.p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2 import (
    contiguous_time_groups,
    time_group_sha256,
)


def test_contiguous_time_groups_are_balanced_disjoint_and_deterministic() -> None:
    times = pd.date_range("2024-01-01", periods=303, freq="h", tz="UTC")
    groups = contiguous_time_groups(times, minimum_profiles=100)
    assert [len(group) for group in groups] == [101, 101, 101]
    assert groups[0][-1] < groups[1][0] < groups[2][0]
    assert len(set().union(*(set(group.asi8) for group in groups))) == 303
    assert time_group_sha256(groups[0]) == time_group_sha256(groups[0].copy())


def test_contiguous_time_groups_reject_insufficient_support() -> None:
    times = pd.date_range("2024-01-01", periods=299, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="minimum profile support"):
        contiguous_time_groups(times, minimum_profiles=100)
