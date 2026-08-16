from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.loss_router import (
    COMPONENTS,
    LEADS,
    OBSERVED_FEATURES,
    ComponentLossRouter,
    RouterConfig,
    build_case_router_data,
    build_inference_router_features,
    expand_case_router_features,
    expand_case_router_rows,
    route_case_predictions,
    route_row_predictions,
    run_prequential_lead_router,
    run_prequential_router,
)


def _fixture(cases: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    feature_rows = []
    anchor_rows = []
    for case in range(cases):
        current = 1.5 + case / 100
        for lead in LEADS:
            truth = current + lead / 100
            rows.append(
                {
                    "fold": "f1",
                    "anchor_id": case,
                    "station": "S-ORS" if case % 2 else "G-ORS",
                    "lead_h": lead,
                    "current_hs": current,
                    "target_hs": truth,
                    "single_prediction": truth + 0.1,
                    "multi_prediction": truth - 0.2,
                    "persistence": current,
                }
            )
        feature_rows.append(
            {"anchor_id": case, **{name: float(case) for name in OBSERVED_FEATURES}}
        )
        anchor_rows.append(
            {
                "anchor_id": case,
                "anchor_time": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=case * 78),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), pd.DataFrame(anchor_rows)


def test_case_builder_uses_six_leads_and_contains_no_target_columns() -> None:
    inputs, metadata, components, losses = build_case_router_data(*_fixture())
    assert len(inputs) == len(metadata) == 12
    assert components.shape == (12, 6, 3)
    assert losses.shape == (12, 3)
    assert not {"target_hs", "truth", "label"}.intersection(inputs.columns)
    np.testing.assert_allclose(losses[0, :2], [0.01, 0.04])
    rebuilt = build_inference_router_features(
        pd.DataFrame({name: inputs[name] for name in OBSERVED_FEATURES}),
        metadata["station"].to_numpy(),
        np.full(len(metadata), 1.5) + np.arange(len(metadata)) / 100,
        components,
    )
    pd.testing.assert_frame_equal(inputs.reset_index(drop=True), rebuilt.reset_index(drop=True))


def test_case_builder_rejects_missing_lead() -> None:
    oof, features, anchors = _fixture()
    with pytest.raises(ValueError, match="six distinct leads"):
        build_case_router_data(oof.iloc[:-1], features, anchors)


def test_router_weights_are_convex_and_no_op_preserves_incumbent() -> None:
    inputs, _, components, losses = build_case_router_data(*_fixture())
    config = RouterConfig(10.0, 1.0, 0.5, "test")
    weights = ComponentLossRouter(config).fit(inputs, losses).predict_weights(inputs)
    assert weights.shape == (12, len(COMPONENTS))
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    no_op = np.broadcast_to(np.array([0.5, 0.5, 0.0]), weights.shape)
    prediction = route_case_predictions(components, no_op)
    np.testing.assert_allclose(prediction, components[:, :, :2].mean(axis=2))


def test_router_rejects_future_target_feature() -> None:
    inputs, _, _, losses = build_case_router_data(*_fixture())
    inputs["target_hs"] = 1.0
    with pytest.raises(ValueError, match="target_hs"):
        ComponentLossRouter(RouterConfig(10.0, 1.0, 0.5, "test")).fit(inputs, losses)


def test_current_fold_truth_cannot_change_router_predictions() -> None:
    oof, observed, anchors = _fixture(60)
    oof.loc[oof["anchor_id"].between(20, 39), "fold"] = "f2"
    oof.loc[oof["anchor_id"].ge(40), "fold"] = "f3"
    inputs, metadata, components, losses = build_case_router_data(oof, observed, anchors)
    ordered = oof.sort_values(["fold", "anchor_id", "lead_h"])
    truth = np.stack(
        [
            group.sort_values("lead_h")["target_hs"].to_numpy(float)
            for _, group in ordered.groupby(["fold", "anchor_id"], sort=False)
        ]
    )
    first = run_prequential_router(
        inputs,
        metadata,
        components,
        losses,
        truth,
        fold_order=("f1", "f2", "f3"),
    )
    changed = truth.copy()
    changed[metadata["fold"].eq("f3").to_numpy()] += 100.0
    second = run_prequential_router(
        inputs,
        metadata,
        components,
        losses,
        changed,
        fold_order=("f1", "f2", "f3"),
    )
    current = metadata["fold"].eq("f3").to_numpy()
    np.testing.assert_allclose(first.weights[current], second.weights[current])
    np.testing.assert_allclose(first.prediction[current], second.prediction[current])


def test_lead_router_expansion_and_current_truth_invariance() -> None:
    oof, observed, anchors = _fixture(60)
    oof.loc[oof["anchor_id"].between(20, 39), "fold"] = "f2"
    oof.loc[oof["anchor_id"].ge(40), "fold"] = "f3"
    inputs, metadata, components, _ = build_case_router_data(oof, observed, anchors)
    ordered = oof.sort_values(["fold", "anchor_id", "lead_h"])
    truth = np.stack(
        [
            group.sort_values("lead_h")["target_hs"].to_numpy(float)
            for _, group in ordered.groupby(["fold", "anchor_id"], sort=False)
        ]
    )
    row_x, row_meta, row_components, row_losses = expand_case_router_rows(
        inputs, metadata, components, truth
    )
    assert len(row_x) == 360
    safe_x, safe_meta, safe_components = expand_case_router_features(inputs, metadata, components)
    pd.testing.assert_frame_equal(row_x, safe_x)
    pd.testing.assert_frame_equal(row_meta, safe_meta)
    np.testing.assert_array_equal(row_components, safe_components)
    result = run_prequential_lead_router(
        row_x,
        row_meta,
        row_components,
        row_losses,
        truth.reshape(-1),
        fold_order=("f1", "f2", "f3"),
        active_leads=(12, 18, 24),
    )
    changed = truth.reshape(-1).copy()
    current = row_meta["fold"].eq("f3").to_numpy()
    changed[current] += 100.0
    rerun = run_prequential_lead_router(
        row_x,
        row_meta,
        row_components,
        row_losses,
        changed,
        fold_order=("f1", "f2", "f3"),
        active_leads=(12, 18, 24),
    )
    np.testing.assert_allclose(result.weights[current], rerun.weights[current])
    np.testing.assert_allclose(result.prediction[current], rerun.prediction[current])
    np.testing.assert_allclose(
        route_row_predictions(row_components, result.weights), result.prediction
    )
    inactive = ~row_meta["lead_h"].isin([12, 18, 24]).to_numpy()
    np.testing.assert_allclose(
        result.weights[inactive],
        np.broadcast_to([0.5, 0.5, 0.0], result.weights[inactive].shape),
    )
