"""Read-only helpers for the sealed P3 matched-budget local comparison."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


KEY_COLUMNS = ("fold", "station", "anchor_id", "lead_h")
CASE_COLUMNS = ("fold", "station", "anchor_id")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(truth: Iterable[float], prediction: Iterable[float]) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if truth_array.shape != prediction_array.shape or truth_array.size == 0:
        raise ValueError("RMSE arrays must be nonempty and have identical shape")
    if not np.isfinite(truth_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("RMSE arrays contain non-finite values")
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def apply_fixed_long_lead_shrink(
    base: Iterable[float],
    persistence: Iterable[float],
    leads: Iterable[int],
    *,
    weight: float,
    active_leads: Iterable[int],
) -> np.ndarray:
    base_array = np.asarray(base, dtype=np.float64)
    persistence_array = np.asarray(persistence, dtype=np.float64)
    lead_array = np.asarray(leads, dtype=np.int64)
    if base_array.shape != persistence_array.shape or base_array.shape != lead_array.shape:
        raise ValueError("Shrink inputs must have identical shapes")
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Shrink weight must be in [0, 1]")
    result = base_array.copy()
    active = np.isin(lead_array, np.asarray(tuple(active_leads), dtype=np.int64))
    result[active] = (
        (1.0 - float(weight)) * base_array[active]
        + float(weight) * persistence_array[active]
    )
    return result


def validate_surface(
    frame: pd.DataFrame,
    *,
    expected_cases: int,
    expected_rows: int,
    expected_leads: Iterable[int],
) -> dict[str, Any]:
    required = set(KEY_COLUMNS) | {"target_hs"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF is missing required columns: {missing}")
    if len(frame) != int(expected_rows):
        raise ValueError(f"OOF row count differs: {len(frame)} != {expected_rows}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("OOF contains duplicate row keys")
    expected = tuple(sorted(int(value) for value in expected_leads))
    grouped = frame.groupby(list(CASE_COLUMNS), sort=False)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if len(grouped) != int(expected_cases):
        raise ValueError(f"OOF case count differs: {len(grouped)} != {expected_cases}")
    if not grouped.map(lambda value: value == expected).all():
        raise ValueError("At least one case is not a complete six-lead case")
    return {
        "rows": int(len(frame)),
        "cases": int(len(grouped)),
        "duplicate_row_keys": 0,
        "complete_case_surface": True,
        "leads_h": list(expected),
        "folds": sorted(str(value) for value in frame["fold"].unique()),
        "stations": sorted(str(value) for value in frame["station"].unique()),
    }


def metric_summary(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    if prediction not in frame:
        raise ValueError(f"Prediction column does not exist: {prediction}")

    def grouped_metric(column: str) -> dict[str, float]:
        return {
            str(key): rmse(group["target_hs"], group[prediction])
            for key, group in frame.groupby(column, sort=True)
        }

    return {
        "rmse_m": rmse(frame["target_hs"], frame[prediction]),
        "rows": int(len(frame)),
        "by_fold": grouped_metric("fold"),
        "by_station": grouped_metric("station"),
        "by_lead": grouped_metric("lead_h"),
    }


def slice_delta(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled": float(candidate["rmse_m"] - reference["rmse_m"])
    }
    for section in ("by_fold", "by_station", "by_lead"):
        result[section] = {
            key: float(candidate[section][key] - reference[section][key])
            for key in sorted(reference[section])
        }
    return result


def complete_case_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    required = {candidate, reference, "target_hs"} | set(CASE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Bootstrap columns are missing: {missing}")
    work = frame.loc[:, list(CASE_COLUMNS) + ["target_hs", candidate, reference]].copy()
    work["candidate_sq"] = np.square(work[candidate] - work["target_hs"])
    work["reference_sq"] = np.square(work[reference] - work["target_hs"])
    case_sse = work.groupby(list(CASE_COLUMNS), sort=True)[
        ["candidate_sq", "reference_sq"]
    ].sum()
    case_count = int(len(case_sse))
    rows_per_case = int(len(frame) // case_count)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, case_count, size=(int(replicates), case_count))
    candidate_sse = case_sse["candidate_sq"].to_numpy(dtype=np.float64)[indices].sum(axis=1)
    reference_sse = case_sse["reference_sq"].to_numpy(dtype=np.float64)[indices].sum(axis=1)
    denominator = float(case_count * rows_per_case)
    delta = np.sqrt(candidate_sse / denominator) - np.sqrt(reference_sse / denominator)
    low, high = np.quantile(delta, [0.05, 0.95])
    return {
        "unit": "complete_six_lead_case",
        "cases": case_count,
        "rows_per_case": rows_per_case,
        "replicates": int(replicates),
        "seed": int(seed),
        "delta_candidate_minus_reference_ci90_m": [float(low), float(high)],
        "median_delta_m": float(np.median(delta)),
        "probability_candidate_improves_descriptive": float(np.mean(delta < 0.0)),
    }


def residual_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    left_residual = np.asarray(frame[left] - frame["target_hs"], dtype=np.float64)
    right_residual = np.asarray(frame[right] - frame["target_hs"], dtype=np.float64)
    if left_residual.size != right_residual.size or left_residual.size < 2:
        raise ValueError("Residual arrays must align and contain at least two rows")
    value = float(np.corrcoef(left_residual, right_residual)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Residual correlation is not finite")
    return value


def exclusive_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    exclusive_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def compare_aligned_surface(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    float_columns: Iterable[str],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    left_sorted = left.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    right_sorted = right.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if not left_sorted.loc[:, KEY_COLUMNS].equals(right_sorted.loc[:, KEY_COLUMNS]):
        raise ValueError("OOF key surfaces do not align")
    maxima: dict[str, float] = {}
    for column in float_columns:
        difference = np.abs(
            left_sorted[column].to_numpy(dtype=np.float64)
            - right_sorted[column].to_numpy(dtype=np.float64)
        )
        maximum = float(difference.max(initial=0.0))
        if maximum > float(tolerance):
            raise ValueError(f"Aligned column differs: {column}, max={maximum}")
        maxima[column] = maximum
    return {"keys_exact": True, "max_abs_difference": maxima}


__all__ = [
    "CASE_COLUMNS",
    "KEY_COLUMNS",
    "apply_fixed_long_lead_shrink",
    "compare_aligned_surface",
    "complete_case_bootstrap_delta",
    "exclusive_write_json",
    "exclusive_write_text",
    "metric_summary",
    "read_json",
    "residual_correlation",
    "rmse",
    "sha256_file",
    "slice_delta",
    "validate_surface",
]
