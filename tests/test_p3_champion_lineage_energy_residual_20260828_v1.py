from __future__ import annotations

import numpy as np
import pytest

from p3_wave.champion_lineage_energy_residual import (
    apply_champion_energy_residual,
    reconstruct_champion_lineage,
)


def test_champion_reconstruction_keeps_inactive_leads_bit_exact() -> None:
    original = np.asarray([1.0, 2.0, 3.0, 4.0])
    axis_a = np.asarray([1.5, 2.5, 3.5, 4.5])
    leads = np.asarray([3, 12, 18, 24])
    champion, active = reconstruct_champion_lineage(
        original,
        axis_a,
        leads,
        alpha=-2.0,
    )
    assert active.tolist() == [False, True, True, True]
    assert champion[[0]].tobytes() == original[[0]].tobytes()
    assert np.array_equal(champion[active], original[active] - 2.0 * (axis_a[active] - original[active]))


def test_energy_residual_keeps_non_long_leads_bit_exact() -> None:
    champion = np.asarray([1.0, 2.0, 3.0, 4.0])
    transfer = np.asarray([5.0, 5.0, 5.0, 5.0])
    leads = np.asarray([6, 12, 18, 24])
    candidate, active = apply_champion_energy_residual(champion, transfer, leads)
    assert active.tolist() == [False, False, True, True]
    assert candidate[[0, 1]].tobytes() == champion[[0, 1]].tobytes()
    expected = np.sqrt(0.75 * champion[active] ** 2 + 0.25 * transfer[active] ** 2)
    assert np.allclose(candidate[active], expected)


def test_invalid_energy_weight_fails_closed() -> None:
    values = np.ones(1)
    with pytest.raises(ValueError, match="energy_weight"):
        apply_champion_energy_residual(values, values, np.asarray([18]), energy_weight=1.1)
