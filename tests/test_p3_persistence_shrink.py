from __future__ import annotations

import numpy as np
import pytest

from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)


def test_fixed_shrink_is_exact_no_op_for_short_leads() -> None:
    incumbent = np.arange(6, dtype=float) + 1.0
    persistence = np.full(6, 2.0)
    leads = np.array([3, 6, 9, 12, 18, 24])
    prediction = apply_long_lead_persistence_shrink(incumbent, persistence, leads)
    np.testing.assert_array_equal(prediction[:3], incumbent[:3])
    np.testing.assert_allclose(prediction[3:], 0.8 * incumbent[3:] + 0.2 * persistence[3:])


def test_zero_weight_reproduces_all_incumbent_rows() -> None:
    incumbent = np.array([1.0, 2.0, 3.0])
    prediction = apply_long_lead_persistence_shrink(
        incumbent,
        np.array([4.0, 5.0, 6.0]),
        np.array([12, 18, 24]),
        config=LongLeadPersistenceShrink(weight=0.0),
    )
    np.testing.assert_array_equal(prediction, incumbent)


def test_shrink_rejects_bad_contracts() -> None:
    with pytest.raises(ValueError, match="aligned"):
        apply_long_lead_persistence_shrink(np.array([1.0]), np.array([1.0, 2.0]), np.array([12]))
    with pytest.raises(ValueError, match="unexpected lead"):
        apply_long_lead_persistence_shrink(np.array([1.0]), np.array([1.0]), np.array([15]))
    with pytest.raises(ValueError, match="weight"):
        LongLeadPersistenceShrink(weight=1.1)
