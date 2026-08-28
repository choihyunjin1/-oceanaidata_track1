from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.era5_context_transfer import common_feature_columns
from p3_wave.joint_wave_state_multitask import (
    JointWaveStateError,
    JointWaveStateTransferRegressor,
    apply_joint_increment,
)


def test_joint_target_shape_is_strict() -> None:
    model = JointWaveStateTransferRegressor()
    with pytest.raises(JointWaveStateError):
        model._targets(np.zeros((2, 6)), 2)
    valid = model._targets(np.zeros((2, 6, 3)), 2)
    assert valid.shape == (2, 6, 3)


def test_feature_schema_rejects_identity_and_reordering() -> None:
    columns = common_feature_columns()
    frame = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    assert JointWaveStateTransferRegressor._context(frame).shape == (1, 286)
    with pytest.raises(JointWaveStateError):
        JointWaveStateTransferRegressor._context(frame.loc[:, list(reversed(columns))])


def test_increment_is_bit_exact_on_protected_leads() -> None:
    base = np.arange(12, dtype=np.float64).reshape(2, 6)
    joint = base + 5.0
    candidate = apply_joint_increment(base, joint)
    np.testing.assert_array_equal(candidate[:, :4], base[:, :4])
    np.testing.assert_allclose(candidate[:, 4:], base[:, 4:] + 1.0)


def test_multitask_loss_is_forced() -> None:
    model = JointWaveStateTransferRegressor()
    assert model._source_parameters["loss_function"] == "MultiRMSE"
    assert model._local_parameters["loss_function"] == "MultiRMSE"
