from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from p2_restore.dynamic_sigmoid_profile import (
    SigmoidSpec,
    TimeBlock,
    build_public_features,
    closed_form_convex_alpha,
    effective_depth,
    feature_columns,
    fit_latent_ridge,
    fit_public_profile,
    fit_sigmoid_profile,
    joint_mask_target_intervals,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p2_dynamic_sigmoid_profile_v1.json"


def _spec() -> SigmoidSpec:
    return SigmoidSpec(
        center_bounds_m=(4.0, 49.5),
        width_bounds_m=(0.5, 25.0),
        center_start_fractions=(0.25, 0.5, 0.75),
        width_starts_m=(2.0, 6.0, 14.0),
        max_nfev=120,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        boundary_fraction=0.01,
        target_depths_m=(7.04, 9.44, 14.74),
    )


def _synthetic_observations(periods: int = 36) -> pd.DataFrame:
    times = pd.date_range("2024-08-31T22:00:00+09:00", periods=periods, freq="10min")
    nominal = {1: 4.19, 2: 7.04, 3: 9.44, 4: 14.74, 5: 19.59, 6: 30.68, 7: 39.45, 8: 49.35}
    rows: list[dict[str, object]] = []
    for number, current in enumerate(times):
        center = 13.0 + 0.02 * number
        width = 3.0
        for layer, depth in nominal.items():
            temp = 12.0 + 8.0 * expit((center - depth) / width)
            rows.append(
                {
                    "station": "S-ORS",
                    "year": current.year,
                    "layer": layer,
                    "time": current.isoformat(),
                    "temp": temp,
                    "psal": 33.0 + 0.01 * depth,
                    "depth": depth if not (layer == 6 and number == 0) else np.nan,
                    "nominal_depth": depth,
                }
            )
    return pd.DataFrame(rows)


def test_contract_has_three_exact_61_day_blocks_and_all_bans() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocks = [
        TimeBlock.from_strings(name, values)
        for name, values in contract["validation"]["blocks"].items()
    ]
    assert len(blocks) == 3
    assert [block.days for block in blocks] == [61, 61, 61]
    assert blocks[1].start == pd.Timestamp("2025-07-02T00:00:00+09:00")
    problem = contract["problem_contract"]
    assert problem["hidden_target_values_read"] is False
    assert problem["test_index_values_read"] is False
    assert problem["outer_truth_scoring_allowed"] is False
    assert problem["submission_generation_allowed"] is False


def test_joint_mask_masks_temp_and_psal_only_inside_target_block() -> None:
    observations = _synthetic_observations()
    block = TimeBlock.from_strings(
        "current", ("2024-09-01T00:00:00+09:00", "2024-09-02T00:00:00+09:00")
    )
    masked = joint_mask_target_intervals(observations, (block,))
    times = pd.to_datetime(masked["time"], utc=True)
    selected = block.mask(times) & masked["layer"].isin([2, 3, 4]).to_numpy()
    public = block.mask(times) & masked["layer"].isin([1, 5, 6, 7, 8]).to_numpy()
    assert masked.loc[selected, ["temp", "psal"]].isna().all().all()
    assert masked.loc[public, ["temp", "psal"]].notna().all().all()
    outside = ~block.mask(times)
    assert np.allclose(
        masked.loc[outside, "temp"].to_numpy(float),
        observations.loc[outside, "temp"].to_numpy(float),
    )


def test_actual_depth_falls_back_to_nominal() -> None:
    result = effective_depth(
        np.asarray([4.2, np.nan, -1.0, 0.0]),
        np.asarray([4.19, 19.59, 30.68, np.nan]),
    )
    assert np.allclose(result[:3], [4.2, 19.59, 30.68])
    assert np.isnan(result[3])


def test_variable_projection_recovers_synthetic_sigmoid() -> None:
    depth = np.asarray([4.19, 7.04, 9.44, 14.74, 19.59, 30.68, 39.45, 49.35])
    center, width, offset, amplitude = 13.2, 3.4, 11.5, 9.0
    temperature = offset + amplitude * expit((center - depth) / width)
    result = fit_sigmoid_profile(depth, temperature, _spec(), minimum_points=6)
    assert result.success
    assert result.r2 > 0.999999
    assert abs(result.center_m - center) < 1e-3
    assert abs(result.width_m - width) < 1e-3
    assert result.multistart_target_spread_c < 1e-5
    assert np.isfinite(result.scaled_jacobian_condition)


def test_public_features_are_invariant_to_target_mask_and_exclude_targets() -> None:
    observations = _synthetic_observations(periods=180)
    block = TimeBlock.from_strings(
        "current", ("2024-09-01T00:00:00+09:00", "2024-09-02T00:00:00+09:00")
    )
    original = build_public_features(observations)
    masked = build_public_features(joint_mask_target_intervals(observations, (block,)))
    pd.testing.assert_frame_equal(original, masked)
    columns = feature_columns(original)
    assert not any(column in columns for column in ("temp_2", "temp_3", "temp_4"))
    assert not any(column in columns for column in ("psal_2", "psal_3", "psal_4"))


def test_public_profile_fit_is_finite_with_four_or_more_public_depths() -> None:
    depth = np.asarray([4.19, 19.59, 30.68, 39.45, 49.35])
    temperature = 11.5 + 9.0 * expit((13.2 - depth) / 3.4)
    result = fit_public_profile(
        depth,
        temperature,
        np.asarray([7.04, 9.44, 14.74]),
        center_m=13.2,
        log_width=np.log(3.4),
        minimum_points=4,
        minimum_depth_span_m=30.0,
        center_step_m=0.02,
        log_width_step=0.002,
        condition_max=1e4,
    )
    assert result.supported
    assert np.isfinite(result.target_prediction).all()
    assert result.point_count == 5
    assert result.depth_span_m > 40.0


def test_fold_local_ridge_and_closed_form_alpha_have_exact_no_op() -> None:
    observations = _synthetic_observations(periods=360)
    features = build_public_features(observations)
    times = features.index[::6]
    catalog = pd.DataFrame(
        {
            "time": times,
            "center_m": 12.0 + np.linspace(0.0, 2.0, len(times)),
            "log_width": np.log(3.0 + np.linspace(0.0, 0.5, len(times))),
            "sample_weight": np.ones(len(times)),
        }
    )
    model = fit_latent_ridge(
        features,
        catalog,
        columns=feature_columns(features),
        alpha=1.0,
        minimum_feature_coverage=0.05,
        minimum_rows=20,
        center_bounds_m=(4.0, 49.5),
        width_bounds_m=(0.5, 25.0),
    )
    prediction = model.predict(features.iloc[:10])
    assert prediction.shape == (10, 2)
    truth = np.asarray([1.0, 2.0, 3.0])
    incumbent = np.asarray([1.1, 2.1, 3.1])
    assert closed_form_convex_alpha(truth, incumbent, incumbent) == 0.0
    worse = incumbent + 1.0
    assert closed_form_convex_alpha(truth, incumbent, worse) == 0.0


def test_expanded_outer_mask_has_seven_day_purge() -> None:
    block = TimeBlock.from_strings(
        "outer", ("2025-07-02T00:00:00+09:00", "2025-09-01T00:00:00+09:00")
    )
    times = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-06-24T23:50:00+09:00"),
            pd.Timestamp("2025-06-25T00:00:00+09:00"),
            pd.Timestamp("2025-09-07T23:50:00+09:00"),
            pd.Timestamp("2025-09-08T00:00:00+09:00"),
        ]
    )
    assert block.expanded_mask(times, purge_days=7).tolist() == [False, True, True, False]


def test_runner_has_no_submission_or_test_index_reader() -> None:
    source = (ROOT / "scripts/run_p2_dynamic_sigmoid_profile.py").read_text(encoding="utf-8")
    assert 'sample_submission.csv")' not in source
    assert 'test_index.csv")' not in source
    assert "to_csv(" not in source
    assert "requests." not in source
