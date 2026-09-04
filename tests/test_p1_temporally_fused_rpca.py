from __future__ import annotations

import numpy as np

from p1_qc.temporally_fused_rpca import duration_mask, temporally_fused_rpca


def test_temporally_fused_rpca_separates_persistent_single_sensor_offset() -> None:
    rng = np.random.default_rng(20260828)
    common = np.sin(np.linspace(0.0, 8.0, 180))[:, None]
    values = common @ np.asarray([[1.0, 0.8, 1.2, 0.9]])
    values += rng.normal(0.0, 0.01, values.shape)
    values[60:120, 2] += 1.5
    result = temporally_fused_rpca(values, maximum_iterations=120, tolerance=1e-5)
    signal = np.abs(result.sparse[:, 2])
    assert np.median(signal[70:110]) > 5.0 * np.median(signal[:40])


def test_duration_mask_keeps_only_registered_runs() -> None:
    mask = np.zeros(700, dtype=bool)
    mask[10:30] = True
    mask[100:160] = True
    mask[200:750 if len(mask) >= 750 else 700] = True
    kept = duration_mask(mask, 48, 519)
    assert not kept[10:30].any()
    assert kept[100:160].all()
    assert kept[200:700].all()
