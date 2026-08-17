"""Low-complexity long-lead shrinkage toward the observed current wave height."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OFFICIAL_LEADS = (3, 6, 9, 12, 18, 24)
LONG_LEADS = (12, 18, 24)
FROZEN_LONG_PERSISTENCE_WEIGHT = 0.20


@dataclass(frozen=True)
class LongLeadPersistenceShrink:
    """One fixed scalar correction applied only to the three longest leads."""

    weight: float = FROZEN_LONG_PERSISTENCE_WEIGHT
    active_leads: tuple[int, ...] = LONG_LEADS

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if not self.active_leads or not set(self.active_leads).issubset(OFFICIAL_LEADS):
            raise ValueError("active_leads must be a non-empty subset of official leads")


def apply_long_lead_persistence_shrink(
    incumbent_prediction: np.ndarray,
    persistence_prediction: np.ndarray,
    lead_h: np.ndarray,
    *,
    config: LongLeadPersistenceShrink | None = None,
) -> np.ndarray:
    """Return a convex shrinkage prediction without reading targets or absolute test time."""

    cfg = config or LongLeadPersistenceShrink()
    incumbent = np.asarray(incumbent_prediction, dtype=float)
    persistence = np.asarray(persistence_prediction, dtype=float)
    leads = np.asarray(lead_h)
    if (
        incumbent.ndim != 1
        or persistence.shape != incumbent.shape
        or leads.shape != incumbent.shape
    ):
        raise ValueError("incumbent, persistence, and lead_h must be aligned 1D arrays")
    if not np.isfinite(incumbent).all() or not np.isfinite(persistence).all():
        raise ValueError("predictions must be finite")
    unexpected = set(np.unique(leads.astype(int))).difference(OFFICIAL_LEADS)
    if unexpected:
        raise ValueError(f"unexpected lead values: {sorted(unexpected)}")
    active = np.isin(leads.astype(int), cfg.active_leads)
    output = incumbent.copy()
    output[active] = (1.0 - cfg.weight) * incumbent[active] + cfg.weight * persistence[active]
    return output
