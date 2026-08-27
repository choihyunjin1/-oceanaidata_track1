from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.matched_budget_compare import (
    LocalContext,
    build_bootstrap_plan,
    build_local_context,
    complementarity_report,
    materialize_settings,
    metric_report,
    paired_day_bootstrap,
)
from p2_restore.public_layer_causal_residual import CausalResidualSpec


def _observations_one_time() -> pd.DataFrame:
    time = "2025-01-01T00:00:00+09:00"
    actual_depth = {1: 10.0, 2: 90.0, 3: 91.0, 4: 92.0, 5: 100.0,
                    6: 110.0, 7: 120.0, 8: 130.0}
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": range(1, 9),
            "time": time,
            "temp": np.arange(1.0, 9.0),
            "psal": np.linspace(30.0, 31.0, 8),
            "depth": [actual_depth[layer] for layer in range(1, 9)],
            "nominal_depth": np.arange(10.0, 90.0, 10.0),
        }
    )


def test_local_context_reproduces_round_a_nominal_depth_interpolation() -> None:
    context = build_local_context(_observations_one_time(), CausalResidualSpec())
    baseline = context.baseline_depth_interpolation.sort_values("layer")
    np.testing.assert_allclose(
        baseline["local_depth_interpolation"].to_numpy(), [2.0, 3.0, 4.0]
    )
    # Causal correction still evaluates at the observed/effective target depth.
    np.testing.assert_allclose(baseline["target_depth"].to_numpy(), [90.0, 91.0, 92.0])


def _no_correction_context(times: pd.DatetimeIndex) -> LocalContext:
    spec = CausalResidualSpec()
    state: dict[str, object] = {"time": times}
    for layer in spec.public_layers:
        state[f"median_residual_{layer}"] = np.full(len(times), np.nan)
        state[f"depth_{layer}"] = np.full(len(times), float(layer * 10))
    endpoints = pd.DataFrame(
        {"time": times, "temp_1": np.zeros(len(times)), "temp_5": 5.0}
    )
    return LocalContext(
        baseline_depth_interpolation=pd.DataFrame(),
        truth=pd.DataFrame(),
        endpoints=endpoints,
        causal_state=pd.DataFrame(state),
    )


def test_materialized_settings_follow_sealed_formulas_and_noop_gate() -> None:
    times = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"])
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": [2, 3, 4, 2, 3, 4],
            "time": np.repeat(times.astype(str), 3),
            "target_depth": [20.0, 30.0, 40.0] * 2,
            "local_depth_interpolation": [0.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            "base": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    context = _no_correction_context(times)
    means, per_seed, diagnostics = materialize_settings(
        frame, ("base",), context, CausalResidualSpec()
    )
    baseline = frame["local_depth_interpolation"].to_numpy()
    incumbent = frame["base"].to_numpy()
    np.testing.assert_allclose(means["INCUMBENT_NOOP"], incumbent)
    np.testing.assert_allclose(
        means["STACK_W0500"], baseline + 0.5 * (incumbent - baseline)
    )
    np.testing.assert_allclose(
        means["STACK_W0625"], baseline + 0.625 * (incumbent - baseline)
    )
    np.testing.assert_allclose(
        means["FALLBACK_BLEND50_A0625"],
        0.5 * incumbent + 0.5 * means["STACK_W0625"],
    )
    np.testing.assert_allclose(means["CAUSAL_RESIDUAL_SCALE025"], incumbent)
    np.testing.assert_allclose(
        means["CAUSAL_ON_FALLBACK"], means["FALLBACK_BLEND50_A0625"]
    )
    assert diagnostics["causal_correction"]["supported_rows"] == 0
    assert len(per_seed["INCUMBENT_NOOP"]) == 1


def _metric_frame() -> pd.DataFrame:
    rows = []
    for fold_index, fold in enumerate(("fold_a", "fold_b")):
        for day in range(3):
            for layer in (2, 3, 4):
                rows.append(
                    {
                        "fold": fold,
                        "layer": layer,
                        "_time_key": pd.Timestamp(
                            f"2025-01-{fold_index * 3 + day + 1:02d}T00:00:00Z"
                        ),
                        "truth": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_metrics_and_paired_day_bootstrap_match_hand_calculation() -> None:
    frame = _metric_frame()
    reference = np.zeros(len(frame))
    candidate = np.ones(len(frame))
    report = metric_report(frame, candidate)
    assert report["fold_equal_layer_equal_rmse_c"] == 1.0
    assert report["fixed_historical_row_weighted_rmse_c"] == 1.0
    plan = build_bootstrap_plan(frame, replicates=100, seed=7)
    equal = paired_day_bootstrap(
        frame, reference, reference, plan, interval=0.9
    )
    worse = paired_day_bootstrap(
        frame, reference, candidate, plan, interval=0.9
    )
    assert equal["delta_rmse_c"] == 0.0
    assert equal["ci90_c"] == [0.0, 0.0]
    assert worse["delta_rmse_c"] == 1.0
    assert worse["ci90_c"] == [1.0, 1.0]
    assert worse["probability_candidate_improves"] == 0.0


def test_complementarity_projects_repeated_prefix_surfaces_independently() -> None:
    times = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"])
    base = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": [2, 3, 4, 2, 3, 4],
            "time": np.repeat(times.astype(str), 3),
            "fold": ["fold_a"] * 3 + ["fold_b"] * 3,
            "_time_key": np.repeat(times, 3),
            "truth": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    frame = pd.concat(
        [
            base.assign(prefix_fraction=0.4),
            base.assign(prefix_fraction=1.0),
        ],
        ignore_index=True,
    )
    incumbent = frame["truth"].to_numpy() + 0.1
    candidate = frame["truth"].to_numpy() - 0.1
    report = complementarity_report(
        frame,
        incumbent,
        candidate,
        _no_correction_context(times).endpoints,
    )
    assert report["fixed_50_50_blend_primary_rmse_c"] == 0.0
