from __future__ import annotations

import numpy as np

from p3_wave.longlead_energy_residual import apply_longlead_energy_residual


def test_inactive_leads_are_bit_exact() -> None:
    incumbent = np.asarray([1.0, 2.0, 3.0])
    transfer = np.asarray([4.0, 4.0, 4.0])
    leads = np.asarray([6, 18, 12])
    candidate, active = apply_longlead_energy_residual(
        incumbent, transfer, leads, active_leads=(18,), energy_weight=0.25
    )
    assert active.tolist() == [False, True, False]
    assert candidate[[0, 2]].tobytes() == incumbent[[0, 2]].tobytes()
    assert np.isclose(candidate[1], np.sqrt(0.75 * 4.0 + 0.25 * 16.0))


def test_energy_weight_bounds() -> None:
    values = np.ones(1)
    try:
        apply_longlead_energy_residual(values, values, np.asarray([18]), energy_weight=1.5)
    except ValueError as error:
        assert "energy_weight" in str(error)
    else:
        raise AssertionError("invalid energy weight was accepted")
