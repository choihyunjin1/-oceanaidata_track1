from __future__ import annotations

import numpy as np
import pytest

from p3_wave.causal_forcing_sequence_checkpoint import (
    InnerSeedCheckpointCurve,
    ids_sha256,
    postprocess_sequence_delta,
    select_earliest_ensemble_epoch,
)


def _curve(seed: int, deltas: list[float]) -> InnerSeedCheckpointCurve:
    predictions = tuple(
        np.full((2, 6), value, dtype=np.float32) for value in deltas
    )
    return InnerSeedCheckpointCurve(
        seed=seed,
        epochs=tuple(range(1, len(deltas) + 1)),
        prediction_delta_by_epoch=predictions,
        model_state_sha256_by_epoch=tuple(f"state-{seed}-{epoch}" for epoch in range(1, len(deltas) + 1)),
        scaler_state_sha256=f"scaler-{seed}",
        train_ids_sha256=ids_sha256(np.asarray([0, 1], dtype=np.int64)),
        validation_ids_sha256=ids_sha256(np.asarray([2, 3], dtype=np.int64)),
        optimizer_steps=len(deltas),
    )


def test_postprocess_is_the_sealed_fixed8_clip_and_long_lead_shrink() -> None:
    result = postprocess_sequence_delta(
        np.ones((1, 6), dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(result[0, :3], 3.0)
    np.testing.assert_allclose(result[0, 3:], 2.8)


def test_seed_ensemble_selects_earliest_exact_minimum() -> None:
    curves = [
        _curve(20260816, [1.0, 1.0, 2.0]),
        _curve(20260817, [1.0, 1.0, 2.0]),
        _curve(20260818, [1.0, 1.0, 2.0]),
    ]
    current = np.full(2, 2.0, dtype=np.float32)
    target = postprocess_sequence_delta(
        np.ones((2, 6), dtype=np.float32),
        current,
    )
    selection = select_earliest_ensemble_epoch(
        curves,
        current_hs=current,
        target_hs=target,
    )
    assert selection.selected_epoch == 1
    assert selection.rmse_by_epoch[0] == selection.rmse_by_epoch[1] == 0.0
    assert selection.rmse_by_epoch[2] > 0.0
    assert selection.seed_ids == (20260816, 20260817, 20260818)
    assert len(selection.selection_prediction_sha256_by_epoch) == 3


def test_seed_ensemble_rejects_duplicate_seed_curves() -> None:
    curves = [_curve(7, [0.0]), _curve(7, [0.0])]
    with pytest.raises(ValueError, match="duplicated"):
        select_earliest_ensemble_epoch(
            curves,
            current_hs=np.ones(2, dtype=np.float32),
            target_hs=np.ones((2, 6), dtype=np.float32),
        )


def test_id_commitment_rejects_empty_or_nonvector_ids() -> None:
    with pytest.raises(ValueError):
        ids_sha256(np.asarray([], dtype=np.int64))
    with pytest.raises(ValueError):
        ids_sha256(np.asarray([[1, 2]], dtype=np.int64))

