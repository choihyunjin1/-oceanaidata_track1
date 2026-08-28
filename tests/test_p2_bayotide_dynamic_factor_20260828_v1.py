from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.p2_bayotide_dynamic_factor_20260828_v1 import (
    TARGET_LAYERS,
    build_registered_panel,
    fit_fixed_dynamic_factor,
    fold_masks,
    guarded_temperature_candidate,
    matern32_discretization,
    smooth_dynamic_factor,
)


def synthetic_observations() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=420, freq="10min", tz="Asia/Seoul")
    depth = {1: 4.2, 2: 7.0, 3: 9.4, 4: 14.7, 5: 19.6, 6: 30.7, 7: 39.5, 8: 49.4}
    rows: list[dict[str, object]] = []
    phase = np.arange(len(times), dtype=float)
    for position, timestamp in enumerate(times):
        for layer in range(1, 9):
            rows.append(
                {
                    "year": 2025,
                    "time": timestamp.isoformat(),
                    "layer": layer,
                    "temp": 20.0 - 0.2 * depth[layer] + np.sin(phase[position] / 17.0) + layer * np.cos(phase[position] / 41.0) / 20.0,
                    "psal": 31.0 + 0.03 * depth[layer] + np.cos(phase[position] / 19.0) + layer * np.sin(phase[position] / 37.0) / 30.0,
                    "depth": depth[layer] + 0.01 * np.sin(phase[position] / 13.0),
                    "nominal_depth": depth[layer],
                }
            )
    return pd.DataFrame(rows)


def fit_arguments() -> dict[str, object]:
    return {
        "trend_lengthscales_hours": (6.0, 48.0, 336.0),
        "periods_hours": (12.42, 24.0),
        "completion_iterations": 2,
        "minimum_channel_coverage": 0.2,
        "observation_noise_floor": 0.0025,
        "periodic_damping": 0.9995,
        "posterior_multiplier": 1.25,
        "posterior_absolute_cap_c": 1.0,
    }


def test_matern32_exact_discretization_is_stable_and_psd() -> None:
    transition, process, stationary = matern32_discretization(48.0)
    assert np.max(np.abs(np.linalg.eigvals(transition))) < 1.0
    assert np.linalg.eigvalsh(process).min() > 0.0
    assert np.allclose(stationary, transition @ stationary @ transition.T + process)


def test_target_values_are_excluded_from_fit_and_measurement_update() -> None:
    observations = synthetic_observations()
    panel = build_registered_panel(observations, 2025)
    start = panel.times[140].tz_convert("Asia/Seoul").tz_localize(None).isoformat(sep=" ")
    stop = panel.times[280].tz_convert("Asia/Seoul").tz_localize(None).isoformat(sep=" ")
    training, _, purged = fold_masks(panel.times, start, stop, purge_days=0)
    model_a, registered_a = fit_fixed_dynamic_factor(panel, training, **fit_arguments())

    changed = observations.copy()
    changed_time = pd.to_datetime(changed["time"], utc=True)
    inside = changed_time.isin(panel.times[purged]) & changed["layer"].isin(TARGET_LAYERS)
    changed.loc[inside, "temp"] += 1000.0
    changed.loc[inside, "psal"] -= 1000.0
    panel_b = build_registered_panel(changed, 2025)
    model_b, registered_b = fit_fixed_dynamic_factor(panel_b, training, **fit_arguments())
    assert np.array_equal(model_a.loading, model_b.loading)
    predicted_a, posterior_a, _ = smooth_dynamic_factor(panel, registered_a, model_a, purged)
    predicted_b, posterior_b, _ = smooth_dynamic_factor(panel_b, registered_b, model_b, purged)
    assert np.allclose(predicted_a[purged], predicted_b[purged], rtol=0.0, atol=1e-10)
    assert np.allclose(posterior_a[purged], posterior_b[purged], rtol=0.0, atol=1e-10)


def test_posterior_guard_is_profilewise_exact_incumbent_fallback() -> None:
    incumbent = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dynamic = incumbent + 0.5
    result = guarded_temperature_candidate(
        incumbent=incumbent,
        dynamic_temperature=dynamic,
        posterior_sd_c=np.asarray([[0.1, 0.1, 0.1], [0.1, 1.1, 0.1]]),
        public_observed_counts=np.asarray([8, 8]),
        posterior_guard_c=np.asarray([0.5, 0.5, 0.5]),
        minimum_public_channels=4,
    )
    assert np.array_equal(result.candidate[0], dynamic[0])
    assert np.array_equal(result.candidate[1], incumbent[1])
    assert result.active[0].all()
    assert not result.active[1].any()


def test_fallback_tolerates_unscored_nan_profile_slots() -> None:
    incumbent = np.asarray([[1.0, np.nan, 3.0]])
    result = guarded_temperature_candidate(
        incumbent=incumbent,
        dynamic_temperature=np.asarray([[1.1, 2.1, 3.1]]),
        posterior_sd_c=np.asarray([[2.0, 2.0, 2.0]]),
        public_observed_counts=np.asarray([8]),
        posterior_guard_c=np.asarray([0.5, 0.5, 0.5]),
        minimum_public_channels=4,
    )
    assert np.array_equal(result.candidate, incumbent, equal_nan=True)


def test_runner_has_no_official_or_submission_file_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_p2_bayotide_dynamic_factor_20260828_v1.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("test_index.csv", "sample_submission.csv", "P2_submission.csv"):
        assert forbidden not in source
    assert "observations.csv" in source
    assert "candidate_csv_generated" in source
