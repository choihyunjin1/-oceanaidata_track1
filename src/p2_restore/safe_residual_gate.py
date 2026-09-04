"""Support-aware shrinkage for a frozen public-state gate correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATES = ("low", "transition", "high", "missing")


@dataclass(frozen=True)
class SafeCell:
    layer: int
    state: str
    alpha: float
    correction_cap: float
    support_blocks: tuple[str, ...]
    days_by_block: dict[str, int]
    slope_by_block: dict[str, float]


@dataclass(frozen=True)
class SafeResidualCalibrator:
    q33: float
    q67: float
    min_days_per_block: int
    min_support_blocks: int
    cells: dict[tuple[int, str], SafeCell]


def state_labels(values: np.ndarray, *, q33: float, q67: float) -> np.ndarray:
    contrast = np.asarray(values, dtype=float)
    result = np.full(len(contrast), "missing", dtype=object)
    finite = np.isfinite(contrast)
    result[finite & (contrast <= q33)] = "low"
    result[finite & (contrast > q33) & (contrast < q67)] = "transition"
    result[finite & (contrast >= q67)] = "high"
    return result


def _day_balanced_terms(error: np.ndarray, adjustment: np.ndarray, day: np.ndarray) -> tuple:
    unique, inverse, counts = np.unique(day, return_inverse=True, return_counts=True)
    if not len(unique):
        return 0.0, 0.0
    weight = 1.0 / counts[inverse]
    weight /= len(unique)
    return float(np.sum(weight * error * adjustment)), float(np.sum(weight * adjustment**2))


def fit_safe_calibrator(
    frame: pd.DataFrame,
    baseline_prediction: np.ndarray,
    raw_gate_prediction: np.ndarray,
    *,
    min_days_per_block: int = 7,
    min_support_blocks: int = 2,
) -> SafeResidualCalibrator:
    """Fit closed-form, state-local shrinkage on cross-fitted corrections.

    A correction is enabled only when the public state has at least
    ``min_days_per_block`` in enough independent blocks and the derivative at
    the exact no-op is beneficial in every supporting block.
    """

    required = {"time", "layer", "truth", "block", "abs_t1_t5"}
    if not required.issubset(frame.columns):
        raise ValueError(f"safe residual frame is missing {sorted(required - set(frame.columns))}")
    baseline = np.asarray(baseline_prediction, dtype=float)
    raw = np.asarray(raw_gate_prediction, dtype=float)
    if len(frame) != len(baseline) or len(frame) != len(raw):
        raise ValueError("safe residual arrays do not match the frame")
    if not np.isfinite(baseline).all() or not np.isfinite(raw).all():
        raise ValueError("safe residual predictions must be finite")
    finite_contrast = frame["abs_t1_t5"].dropna().to_numpy(float)
    if not len(finite_contrast):
        raise ValueError("safe residual calibration has no finite public contrast")
    q33, q67 = np.quantile(finite_contrast, [1 / 3, 2 / 3])
    states = state_labels(frame["abs_t1_t5"].to_numpy(float), q33=q33, q67=q67)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    layers = frame["layer"].to_numpy(int)
    blocks = frame["block"].astype(str).to_numpy()
    error = baseline - frame["truth"].to_numpy(float)
    adjustment = raw - baseline
    cells: dict[tuple[int, str], SafeCell] = {}
    for layer in (2, 3, 4):
        for state in STATES:
            selected = (layers == layer) & (states == state)
            days_by_block: dict[str, int] = {}
            slope_by_block: dict[str, float] = {}
            alpha_by_block: list[float] = []
            supported: list[str] = []
            for block in sorted(np.unique(blocks[selected])):
                current = selected & (blocks == block)
                days_by_block[block] = int(np.unique(day[current]).size)
                if days_by_block[block] < min_days_per_block:
                    continue
                numerator, denominator = _day_balanced_terms(
                    error[current], adjustment[current], day[current]
                )
                slope_by_block[block] = numerator
                if denominator <= 1e-14 or numerator >= 0:
                    continue
                supported.append(block)
                alpha_by_block.append(float(np.clip(-numerator / denominator, 0.0, 1.0)))
            alpha = 0.0
            cap = 0.0
            if len(supported) >= min_support_blocks:
                stable = selected & np.isin(blocks, supported)
                numerator, denominator = _day_balanced_terms(
                    error[stable], adjustment[stable], day[stable]
                )
                combined = float(np.clip(-numerator / max(denominator, 1e-14), 0.0, 1.0))
                alpha = min(combined, min(alpha_by_block))
                if alpha > 0:
                    cap = float(np.quantile(np.abs(alpha * adjustment[stable]), 0.95))
            cells[(layer, state)] = SafeCell(
                layer=layer,
                state=state,
                alpha=alpha,
                correction_cap=cap,
                support_blocks=tuple(supported),
                days_by_block=days_by_block,
                slope_by_block=slope_by_block,
            )
    return SafeResidualCalibrator(
        q33=float(q33),
        q67=float(q67),
        min_days_per_block=min_days_per_block,
        min_support_blocks=min_support_blocks,
        cells=cells,
    )


def apply_safe_calibrator(
    calibrator: SafeResidualCalibrator,
    frame: pd.DataFrame,
    baseline_prediction: np.ndarray,
    raw_gate_prediction: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline_prediction, dtype=float)
    raw = np.asarray(raw_gate_prediction, dtype=float)
    if len(frame) != len(baseline) or len(frame) != len(raw):
        raise ValueError("safe residual apply arrays do not match the frame")
    states = state_labels(
        frame["abs_t1_t5"].to_numpy(float), q33=calibrator.q33, q67=calibrator.q67
    )
    layers = frame["layer"].to_numpy(int)
    adjustment = raw - baseline
    correction = np.zeros(len(frame), dtype=float)
    for key, cell in calibrator.cells.items():
        selected = (layers == key[0]) & (states == key[1])
        if cell.alpha <= 0 or cell.correction_cap <= 0:
            continue
        correction[selected] = np.clip(
            cell.alpha * adjustment[selected], -cell.correction_cap, cell.correction_cap
        )
    result = baseline + correction
    if not np.isfinite(result).all():
        raise ValueError("safe residual gate produced non-finite predictions")
    return result


def calibrator_summary(calibrator: SafeResidualCalibrator) -> list[dict[str, object]]:
    return [
        {
            "layer": cell.layer,
            "state": cell.state,
            "alpha": cell.alpha,
            "correction_cap": cell.correction_cap,
            "support_blocks": list(cell.support_blocks),
            "days_by_block": cell.days_by_block,
            "slope_by_block": cell.slope_by_block,
        }
        for cell in calibrator.cells.values()
    ]
