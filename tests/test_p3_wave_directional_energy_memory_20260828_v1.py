from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.wave_directional_energy_memory import (
    DIRECTIONAL_FEATURES,
    VALUE_FEATURES,
    DirectionalContextTransferRegressor,
    DirectionalMemoryError,
    apply_directional_increment,
    summarize_directional_energy_memory,
)


def test_directional_features_are_complete_and_global_flip_invariant() -> None:
    offsets = np.linspace(-48.0, 0.0, 289)
    hs = 2.0 + 0.1 * np.sin(offsets)
    direction = np.mod(210.0 + 2.0 * offsets, 360.0)
    original = summarize_directional_energy_memory(hs, direction, offsets)
    flipped = summarize_directional_energy_memory(hs, direction + 180.0, offsets)

    assert tuple(original) == DIRECTIONAL_FEATURES
    assert len(original) == 20
    assert all(original[f"{name}_mask"] == 1.0 for name in VALUE_FEATURES)
    np.testing.assert_allclose(
        [original[name] for name in VALUE_FEATURES],
        [flipped[name] for name in VALUE_FEATURES],
        rtol=0.0,
        atol=1e-12,
    )


def test_missing_context_sets_values_nan_and_masks_zero() -> None:
    offsets = np.arange(-48, 1, dtype=np.float64)
    result = summarize_directional_energy_memory(
        np.full(49, np.nan), np.full(49, np.nan), offsets
    )
    assert all(np.isnan(result[name]) for name in VALUE_FEATURES)
    assert all(result[f"{name}_mask"] == 0.0 for name in VALUE_FEATURES)


def test_increment_is_exact_noop_on_protected_leads() -> None:
    base = np.arange(12, dtype=np.float64).reshape(2, 6)
    enriched = base + 10.0
    candidate = apply_directional_increment(base, enriched)
    np.testing.assert_array_equal(candidate[:, :4], base[:, :4])
    np.testing.assert_allclose(candidate[:, 4:], base[:, 4:] + 2.0)


def test_model_schema_rejects_identity_and_column_reordering() -> None:
    with pytest.raises(DirectionalMemoryError):
        DirectionalContextTransferRegressor(("hs_current", "station"))
    model = DirectionalContextTransferRegressor(("hs_current", "tp_current"))
    with pytest.raises(DirectionalMemoryError):
        model._context(pd.DataFrame({"tp_current": [5.0], "hs_current": [2.0]}))
