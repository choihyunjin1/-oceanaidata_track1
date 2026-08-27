from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.target_covariate_density_ratio import (
    DOMAIN_FEATURE_COLUMNS,
    DOMAIN_FORBIDDEN_COLUMNS,
    build_daily_domain_covariates,
    combined_training_weight,
    effective_sample_fraction,
    estimate_source_daily_density_ratio,
    map_daily_ratio_to_rows,
    square_root_class_weight,
)


def _panel(*, year: int, shift: float, weeks: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(f"{year}-01-01T00:00:00+09:00")
    for week in range(weeks):
        for station in ("A", "B"):
            for layer in (1, 2):
                for step in range(6):
                    time = start + pd.Timedelta(days=week * 7, minutes=step * 10)
                    value = shift + week * 0.02 + layer * 0.1 + step * 0.01
                    rows.append(
                        {
                            "station": station,
                            "year": year,
                            "layer": layer,
                            "time": time.isoformat(),
                            "temp": value,
                            "psal": 30.0 + value,
                            "depth": layer * 10.0,
                            "label": int(step == 5),
                            "anomaly_type": "spike" if step == 5 else "",
                        }
                    )
    return pd.DataFrame(rows)


def test_daily_domain_covariates_exclude_labels_and_break_differences_at_gaps() -> None:
    frame = _panel(year=2024, shift=0.0, weeks=2)
    # Force an observation gap and a large level jump.  The jump must not enter
    # the gap-safe first-difference statistic.
    mask = frame["station"].eq("A") & frame["layer"].eq(1)
    positions = frame.index[mask]
    frame.loc[positions[3], "time"] = (
        pd.Timestamp(frame.loc[positions[2], "time"]) + pd.Timedelta(minutes=30)
    ).isoformat()
    frame.loc[positions[3], "temp"] = 1000.0
    daily = build_daily_domain_covariates(frame)
    assert set(DOMAIN_FEATURE_COLUMNS).issubset(daily.columns)
    assert not set(DOMAIN_FORBIDDEN_COLUMNS).intersection(DOMAIN_FEATURE_COLUMNS)
    assert "label" not in daily.columns
    assert "anomaly_type" not in daily.columns
    assert daily[["station", "layer", "kst_day"]].duplicated().sum() == 0


def test_density_oof_is_group_disjoint_and_maps_every_source_row() -> None:
    source = _panel(year=2024, shift=0.0)
    target = _panel(year=2026, shift=0.2)
    source_daily = build_daily_domain_covariates(source)
    target_daily = build_daily_domain_covariates(target)
    result = estimate_source_daily_density_ratio(
        source_daily,
        target_daily,
        seed=20260813,
        n_splits=5,
    )
    assert result.audit["all_groups_disjoint"] is True
    assert result.audit["forbidden_feature_intersection"] == []
    assert result.audit["missing_target_station_layer_support"] == []
    assert np.isfinite(result.source_daily_ratio).all()
    assert np.all((result.source_daily_ratio >= 0.1) & (result.source_daily_ratio <= 8.0))
    row_ratio = map_daily_ratio_to_rows(source, source_daily, result.source_daily_ratio)
    assert row_ratio.shape == (len(source),)
    assert effective_sample_fraction(row_ratio) > 0


def test_combined_weight_preserves_frozen_base_weight_sum() -> None:
    target = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int8)
    ratio = np.asarray([0.1, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
    base = square_root_class_weight(target)
    combined, audit = combined_training_weight(target, ratio)
    assert np.isclose(combined.sum(dtype=np.float64), base.sum(), rtol=0, atol=1e-6)
    assert abs(audit["sum_difference"]) < 1e-12
    assert np.isfinite(combined).all()
    assert (combined > 0).all()
