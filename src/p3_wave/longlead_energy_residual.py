"""Incumbent-preserving long-lead wave-energy residual shrinkage."""

from __future__ import annotations

import numpy as np


def apply_longlead_energy_residual(
    incumbent_hs: np.ndarray,
    transfer_hs: np.ndarray,
    lead_h: np.ndarray,
    *,
    active_leads: tuple[int, ...] = (18, 24),
    energy_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink an external expert's Hs-squared residual into the incumbent."""
    if not 0.0 <= energy_weight <= 1.0:
        raise ValueError("energy_weight must be in [0, 1]")
    incumbent = np.asarray(incumbent_hs, dtype=np.float64)
    transfer = np.asarray(transfer_hs, dtype=np.float64)
    leads = np.asarray(lead_h)
    if incumbent.shape != transfer.shape or incumbent.shape != leads.shape:
        raise ValueError("prediction and lead arrays must share shape")
    if not np.isfinite(incumbent).all() or not np.isfinite(transfer).all():
        raise ValueError("predictions must be finite")
    active = np.isin(leads, np.asarray(active_leads))
    candidate = incumbent.copy()
    energy = incumbent[active] ** 2
    energy += energy_weight * (transfer[active] ** 2 - energy)
    candidate[active] = np.sqrt(np.maximum(energy, 0.0))
    return candidate, active


__all__ = ["apply_longlead_energy_residual"]
