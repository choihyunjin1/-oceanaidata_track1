"""Fixed, saved-model P3 long-lead calibration primitives.

The 0.25 persistence weight is historical sealed evidence from 2026-08-17.  This
module deliberately exposes no coefficient-search API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedLongLeadShrinkCalibrator:
    """Blend routed forecasts toward persistence only on fixed long leads."""

    persistence_weight: float = 0.25
    active_leads: tuple[int, ...] = (12, 18, 24)
    minimum_m: float = 0.0
    maximum_m: float = 30.0

    def __post_init__(self) -> None:
        if self.persistence_weight != 0.25:
            raise ValueError("the sealed calibrator weight must be exactly 0.25")
        if self.active_leads != (12, 18, 24):
            raise ValueError("the sealed calibrator leads must be 12/18/24")
        if (self.minimum_m, self.maximum_m) != (0.0, 30.0):
            raise ValueError("the P3 prediction range must be 0..30 m")

    def predict(
        self,
        routed_prediction: np.ndarray,
        persistence: np.ndarray,
        lead_h: np.ndarray,
    ) -> np.ndarray:
        routed = np.asarray(routed_prediction, dtype=np.float64)
        anchor = np.asarray(persistence, dtype=np.float64)
        leads = np.asarray(lead_h, dtype=np.int64)
        if routed.shape != anchor.shape or routed.shape != leads.shape:
            raise ValueError("calibrator arrays must have identical shape")
        if not np.isfinite(routed).all() or not np.isfinite(anchor).all():
            raise ValueError("calibrator inputs must be finite")
        output = routed.copy()
        active = np.isin(leads, self.active_leads)
        output[active] = (1.0 - self.persistence_weight) * routed[
            active
        ] + self.persistence_weight * anchor[active]
        return np.clip(output, self.minimum_m, self.maximum_m)
