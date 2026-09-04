from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.target_masked_quantile import (
    QUANTILE_SCORE_COLUMNS,
    TARGET_MASKED_FEATURE_COLUMNS,
    QuantileModelConfig,
    build_quantile_scores,
    build_target_masked_design,
    cross_fitted_quantiles,
    synthetic_offset_smoke,
)


def _frame(*, days: int = 4, station: str = "S-ORS", layers: tuple[int, ...] = (1, 2)):
    rows = []
    start = pd.Timestamp("2024-01-01T00:00:00+09:00")
    for step in range(days * 24 * 6):
        phase = 2.0 * np.pi * step / 144.0
        for layer in layers:
            psal = 32.0 + 0.1 * np.sin(phase + layer * 0.1)
            rows.append(
                {
                    "station": station,
                    "year": 2024,
                    "layer": layer,
                    "time": (start + pd.Timedelta(minutes=10 * step)).isoformat(),
                    "temp": 18.0 - layer + np.sin(phase) - 0.4 * (psal - 32.0),
                    "psal": psal,
                    "depth": float(layer * 10),
                    "label": int(step == 300 and layer == layers[0]),
                    "anomaly_type": "spike" if step == 300 and layer == layers[0] else "",
                }
            )
    return pd.DataFrame(rows)


def test_own_temperature_is_absent_and_perturbation_invariant() -> None:
    frame = _frame(days=1)
    original = build_target_masked_design(frame)
    changed = frame.copy()
    changed.loc[0, "temp"] += 1000.0
    perturbed = build_target_masked_design(changed)

    assert tuple(original.frame.columns) == TARGET_MASKED_FEATURE_COLUMNS
    assert not any("depth" in column for column in original.feature_columns)
    assert not any(
        "temp" in column and not column.startswith("peer_temp_")
        for column in original.feature_columns
    )
    np.testing.assert_allclose(
        original.frame.iloc[0].drop(labels=["station", "layer_category"]).astype(float),
        perturbed.frame.iloc[0].drop(labels=["station", "layer_category"]).astype(float),
        equal_nan=True,
        rtol=0.0,
        atol=1.0e-12,
    )
    # The changed row remains legitimately visible as an other-layer peer.
    assert original.frame.loc[1, "peer_temp_mean"] != perturbed.frame.loc[1, "peer_temp_mean"]


def test_gors_no_peer_and_depth_missing_fallback() -> None:
    frame = _frame(days=1, station="G-ORS", layers=(1,))
    frame["depth"] = np.nan
    missing_depth = build_target_masked_design(frame)
    changed_depth = frame.copy()
    changed_depth["depth"] = 9999.0
    finite_depth = build_target_masked_design(changed_depth)

    assert missing_depth.frame["peer_temp_count"].eq(0).all()
    assert missing_depth.frame["peer_temp_missing"].eq(1).all()
    assert missing_depth.frame["peer_psal_count"].eq(0).all()
    pd.testing.assert_frame_equal(missing_depth.frame, finite_depth.frame)


def test_crossfit_is_disjoint_and_uses_only_normal_targets() -> None:
    frame = _frame(days=4)
    design = build_target_masked_design(frame)
    config = QuantileModelConfig(
        n_estimators=12,
        min_child_samples=8,
        crossfit_folds=2,
        threads=1,
    )
    predictions, audit = cross_fitted_quantiles(
        frame,
        design,
        frame["temp"].to_numpy(),
        frame["label"].to_numpy(),
        np.arange(len(frame)),
        config=config,
    )

    assert np.isfinite(predictions).all()
    assert audit["all_scope_rows_predicted_once"] is True
    assert all(block["fit_predict_overlap_rows"] == 0 for block in audit["blocks"])
    assert all(block["positive_fit_rows_used"] == 0 for block in audit["blocks"])


def test_quantile_score_contract_and_gap_boundary() -> None:
    frame = _frame(days=1)
    frame = pd.concat([frame.iloc[:40], frame.iloc[44:]], ignore_index=True)
    quantiles = np.column_stack(
        (
            frame["temp"].to_numpy() - 0.5,
            frame["temp"].to_numpy(),
            frame["temp"].to_numpy() + 0.5,
        )
    )
    scores = build_quantile_scores(frame, quantiles)
    assert tuple(scores.columns) == QUANTILE_SCORE_COLUMNS
    assert scores["tmq_outside_tail_distance"].eq(0.0).all()
    assert scores["tmq_quantile_available"].eq(1).all()


def test_synthetic_offset_smoke() -> None:
    report = synthetic_offset_smoke(
        QuantileModelConfig(
            n_estimators=24,
            min_child_samples=8,
            crossfit_folds=2,
            threads=1,
        )
    )
    assert report["passed"] is True
    assert report["positive_fit_rows_used"] == 0
    assert report["offset_tail_median"] > report["normal_tail_median"]
