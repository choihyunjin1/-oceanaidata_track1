from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln


@dataclass(frozen=True)
class StudentTState:
    location: np.ndarray
    scale: np.ndarray
    log_prior: np.ndarray
    degrees_of_freedom: float


def derive_causal_gap_minutes(frame: pd.DataFrame) -> np.ndarray:
    """Previous timestamp distance within station/layer/year, restored to input order."""
    required = ["station", "layer", "year", "time"]
    if any(column not in frame for column in required):
        raise ValueError("gap derivation requires station/layer/year/time keys")
    work = frame[required].copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["time"] = pd.to_datetime(work["time"], utc=True)
    ordered = work.sort_values(["station", "layer", "year", "time", "__position"], kind="stable")
    gaps = ordered.groupby(["station", "layer", "year"], sort=False, observed=True)["time"].diff()
    ordered["__gap"] = gaps.dt.total_seconds().div(60.0).fillna(0.0)
    restored = ordered.sort_values("__position", kind="stable")["__gap"].to_numpy(np.float64)
    if not np.isfinite(restored).all() or (restored < 0).any():
        raise ValueError("derived gaps are not finite nonnegative causal intervals")
    return restored


def fit_student_t(values: np.ndarray, labels: np.ndarray, degrees_of_freedom: float = 4.0) -> StudentTState:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    if x.ndim != 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("two-class 2-D prefix data required")
    pooled_median = np.nanmedian(x, axis=0)
    pooled_mad = np.nanmedian(np.abs(x - pooled_median), axis=0)
    finite_floor = pooled_mad[np.isfinite(pooled_mad) & (pooled_mad > 0)]
    floor = float(np.quantile(finite_floor, 0.1)) if len(finite_floor) else 1.0
    locations, scales = [], []
    for target in (0, 1):
        subset = x[y == target]
        center = np.nanmedian(subset, axis=0)
        mad = np.nanmedian(np.abs(subset - center), axis=0)
        locations.append(np.where(np.isfinite(center), center, pooled_median))
        scales.append(np.maximum(np.where(np.isfinite(mad), 1.4826 * mad, floor), max(floor, 1e-6)))
    counts = np.bincount(y, minlength=2).astype(np.float64)
    prior = (counts + 0.5) / (counts.sum() + 1.0)
    return StudentTState(np.stack(locations), np.stack(scales), np.log(prior), degrees_of_freedom)


def score_llr(state: StudentTState, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    scores = []
    nu = state.degrees_of_freedom
    constant = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    for target in (0, 1):
        z = (x - state.location[target]) / state.scale[target]
        term = constant - np.log(state.scale[target]) - ((nu + 1.0) / 2.0) * np.log1p(z * z / nu)
        scores.append(np.nansum(np.where(np.isfinite(x), term, 0.0), axis=1) + state.log_prior[target])
    return scores[1] - scores[0]


def wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center - margin) / denominator)


def calibrate_threshold(scores: np.ndarray, truth: np.ndarray, anchor: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int8)
    anchor = np.asarray(anchor, dtype=np.int8)
    negative = anchor == 0
    base_tp = int(((anchor == 1) & (truth == 1)).sum())
    base_fp = int(((anchor == 1) & (truth == 0)).sum())
    base_fn = int(((anchor == 0) & (truth == 1)).sum())
    base_f1 = 2 * base_tp / max(2 * base_tp + base_fp + base_fn, 1)
    best: tuple[float, float, int, int, int] | None = None
    for threshold in np.append(np.unique(scores[negative]), np.inf):
        add = negative & (scores >= threshold)
        additions = int(add.sum())
        tp_add = int((add & (truth == 1)).sum())
        fp_add = additions - tp_add
        if additions == 0 or additions / len(scores) > 0.005:
            continue
        if wilson_lower(tp_add, additions) <= base_f1 / 2.0:
            continue
        f1 = 2 * (base_tp + tp_add) / max(2 * (base_tp + tp_add) + base_fp + fp_add + base_fn - tp_add, 1)
        key = (f1, float(threshold), -additions, tp_add, fp_add)
        if best is None or key[:3] > best[:3]:
            best = key
    if best is None:
        return {"threshold": float("inf"), "additions": 0, "tp": 0, "fp": 0}
    return {"threshold": best[1], "additions": -best[2], "tp": best[3], "fp": best[4]}


def calibrate_threshold_central(scores: np.ndarray, truth: np.ndarray, anchor: np.ndarray) -> dict[str, float | int]:
    """Nested central-F1 selector; intended only for a sealed inner calibration block."""
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int8)
    anchor = np.asarray(anchor, dtype=np.int8)
    negative = anchor == 0
    base_tp = int(((anchor == 1) & (truth == 1)).sum())
    base_fp = int(((anchor == 1) & (truth == 0)).sum())
    base_fn = int(((anchor == 0) & (truth == 1)).sum())
    base_f1 = 2 * base_tp / max(2 * base_tp + base_fp + base_fn, 1)
    best: tuple[float, float, int, int, int, float] | None = None
    for threshold in np.append(np.unique(scores[negative]), np.inf):
        add = negative & (scores >= threshold)
        additions = int(add.sum())
        tp_add = int((add & (truth == 1)).sum())
        fp_add = additions - tp_add
        if additions == 0 or additions / len(scores) > 0.005:
            continue
        precision = tp_add / additions
        if precision <= base_f1 / 2.0:
            continue
        f1 = 2 * (base_tp + tp_add) / max(2 * (base_tp + tp_add) + base_fp + fp_add + base_fn - tp_add, 1)
        delta = f1 - base_f1
        if delta <= 0.0:
            continue
        key = (delta, float(threshold), -additions, tp_add, fp_add, precision)
        if best is None or key[:3] > best[:3]:
            best = key
    if best is None:
        return {"threshold": float("inf"), "additions": 0, "tp": 0, "fp": 0, "inner_delta_f1": 0.0, "precision": 0.0}
    return {"threshold": best[1], "additions": -best[2], "tp": best[3], "fp": best[4], "inner_delta_f1": best[0], "precision": best[5]}
