"""P3 champion-lineage reconstruction and fixed wave-energy residual correction."""

from __future__ import annotations

import numpy as np


def reconstruct_champion_lineage(
    original_hs: np.ndarray,
    axis_a_hs: np.ndarray,
    lead_h: np.ndarray,
    *,
    alpha: float,
    active_leads: tuple[int, ...] = (12, 18, 24),
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct O + alpha * (A - O) on the fixed champion lead support."""
    original = np.asarray(original_hs, dtype=np.float64)
    axis_a = np.asarray(axis_a_hs, dtype=np.float64)
    leads = np.asarray(lead_h)
    if original.shape != axis_a.shape or original.shape != leads.shape:
        raise ValueError("original, axis A, and lead arrays must share shape")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(original).all() or not np.isfinite(axis_a).all():
        raise ValueError("source predictions must be finite")
    active = np.isin(leads, np.asarray(active_leads))
    champion = original.copy()
    champion[active] = original[active] + alpha * (axis_a[active] - original[active])
    return champion, active


def apply_champion_energy_residual(
    champion_hs: np.ndarray,
    transfer_hs: np.ndarray,
    lead_h: np.ndarray,
    *,
    energy_weight: float = 0.25,
    active_leads: tuple[int, ...] = (18, 24),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one fixed ERA5 residual in Hs-squared space to the champion."""
    if not 0.0 <= energy_weight <= 1.0:
        raise ValueError("energy_weight must be in [0, 1]")
    champion = np.asarray(champion_hs, dtype=np.float64)
    transfer = np.asarray(transfer_hs, dtype=np.float64)
    leads = np.asarray(lead_h)
    if champion.shape != transfer.shape or champion.shape != leads.shape:
        raise ValueError("champion, transfer, and lead arrays must share shape")
    if not np.isfinite(champion).all() or not np.isfinite(transfer).all():
        raise ValueError("source predictions must be finite")
    active = np.isin(leads, np.asarray(active_leads))
    candidate = champion.copy()
    champion_energy = champion[active] ** 2
    candidate_energy = champion_energy + energy_weight * (
        transfer[active] ** 2 - champion_energy
    )
    candidate[active] = np.sqrt(np.maximum(candidate_energy, 0.0))
    return candidate, active


__all__ = ["apply_champion_energy_residual", "reconstruct_champion_lineage"]
