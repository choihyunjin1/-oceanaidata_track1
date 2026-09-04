"""Fail-closed causal routing between frozen P3 incumbent and ERA5 experts.

The module contains no source-data or official-test loader.  It accepts only the
already-sealed historical expert predictions, frozen past-only feature rows, and
truth rows released after the corresponding blind commitment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

EXPERIMENT_ID: Final = "p3_era5_incumbent_safe_advantage_router_20260828_v1"
LEADS: Final = (3, 6, 9, 12, 18, 24)
ROUTER_BASE_FEATURES: Final = (
    "hs_current",
    "hs_delta_6h",
    "hs_std_12h",
    "hs_slope_24h",
    "tp_current",
    "wave_energy_current",
    "wave_energy_delta_6h",
    "wave_energy_std_12h",
    "wind_input_proxy_current",
    "wind_input_proxy_delta_6h",
    "wind_input_proxy_std_12h",
)
ROUTER_DERIVED_FEATURES: Final = (
    "lead_h_div_24",
    "transfer_minus_incumbent",
    "abs_transfer_minus_incumbent",
)
ROUTER_FEATURES: Final = (*ROUTER_BASE_FEATURES, *ROUTER_DERIVED_FEATURES)
FORBIDDEN_ROUTER_FEATURE_TOKENS: Final = (
    "station",
    "source",
    "fold",
    "calendar",
    "timestamp",
    "anchor_time",
    "month",
    "year",
    "lead_3",
    "lead_6",
    "lead_9",
    "lead_12",
    "lead_18",
    "lead_24",
)


class AdvantageRouterError(RuntimeError):
    """A frozen input, chronology, feature, or blind-seal contract failed."""


@dataclass(frozen=True)
class InnerBlock:
    """One deterministic 60-day inner validation block."""

    outer_fold: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    anchor_ids: tuple[int, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "name": self.name,
            "start_utc": self.start.isoformat(),
            "end_utc": self.end.isoformat(),
            "selected_cases": len(self.anchor_ids),
            "anchor_ids_sha256": sha256_ints(self.anchor_ids),
        }


@dataclass(frozen=True)
class FittedAdvantageRouter:
    """Fixed median-impute, StandardScaler, Ridge(alpha=100) pipeline."""

    medians: np.ndarray
    scaler: StandardScaler
    ridge: Ridge
    feature_names: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        validate_router_feature_names(tuple(frame.columns))
        if tuple(frame.columns) != self.feature_names:
            raise AdvantageRouterError("router inference feature names or order changed")
        matrix = frame.to_numpy(dtype=np.float64, copy=True)
        if matrix.shape[1] != len(self.medians):
            raise AdvantageRouterError("router inference width changed")
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = self.medians[np.where(missing)[1]]
        transformed = self.scaler.transform(matrix)
        prediction = np.asarray(self.ridge.predict(transformed), dtype=np.float64)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise AdvantageRouterError("router produced invalid advantage predictions")
        return prediction

    def public_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "median": self.medians.tolist(),
            "scaler_mean": np.asarray(self.scaler.mean_, dtype=np.float64).tolist(),
            "scaler_scale": np.asarray(self.scaler.scale_, dtype=np.float64).tolist(),
            "ridge_alpha": float(self.ridge.alpha),
            "ridge_intercept": float(self.ridge.intercept_),
            "ridge_coefficients": np.asarray(self.ridge.coef_, dtype=np.float64).tolist(),
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ints(values: Sequence[int]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return sha256_bytes(array.tobytes())


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise AdvantageRouterError(f"exclusive write verification failed: {path}")
    return sha256_bytes(payload)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    return write_bytes_exclusive(path, canonical_json_bytes(value))


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> str:
    return write_bytes_exclusive(path, parquet_bytes(frame))


def write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    # NumPy's writer requires a file-like object with seek/tell; use an in-memory
    # BytesIO while retaining exclusive publication semantics at the path.
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return write_bytes_exclusive(path, buffer.getvalue())


def file_pin(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix() if root else str(resolved)
    return {"path": relative, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def validate_router_feature_names(names: Sequence[str]) -> None:
    resolved = tuple(str(name) for name in names)
    if resolved != ROUTER_FEATURES:
        raise AdvantageRouterError("router feature names differ from the single preregistered surface")
    lowered = tuple(name.lower() for name in resolved)
    leaking = [
        name
        for name in lowered
        if any(token in name for token in FORBIDDEN_ROUTER_FEATURE_TOKENS)
    ]
    if leaking:
        raise AdvantageRouterError(f"forbidden router features present: {leaking}")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def select_spaced_anchor_ids(frame: pd.DataFrame, *, gap_hours: int = 78) -> tuple[int, ...]:
    required = {"anchor_id", "station", "anchor_time"}
    if not required <= set(frame.columns):
        raise AdvantageRouterError("anchor metadata is incomplete")
    working = frame.loc[:, ["anchor_id", "station", "anchor_time"]].copy()
    working["anchor_time"] = pd.to_datetime(working["anchor_time"], utc=True, errors="raise")
    if working["anchor_id"].duplicated().any():
        raise AdvantageRouterError("anchor IDs are not unique")
    selected: list[int] = []
    gap = pd.Timedelta(hours=gap_hours)
    for _station, group in working.groupby("station", sort=True, observed=True):
        previous: pd.Timestamp | None = None
        ordered = group.sort_values(["anchor_time", "anchor_id"], kind="mergesort")
        for row in ordered.itertuples(index=False):
            when = _utc(row.anchor_time)
            if previous is None or when - previous >= gap:
                selected.append(int(row.anchor_id))
                previous = when
    return tuple(selected)


def build_inner_block_plan(
    anchors: pd.DataFrame,
    windows: Sequence[Sequence[str]],
    *,
    outer_embargo_hours: int = 78,
    block_days: int = 60,
    block_gap_hours: int = 78,
) -> dict[str, tuple[InnerBlock, ...]]:
    """Build four latest chronological blocks without reading any target value."""

    required = {"anchor_id", "station", "anchor_time"}
    if not required <= set(anchors.columns):
        raise AdvantageRouterError("inner-plan anchor metadata is incomplete")
    metadata = anchors.loc[:, ["anchor_id", "station", "anchor_time"]].copy()
    metadata["anchor_time"] = pd.to_datetime(metadata["anchor_time"], utc=True, errors="raise")
    if metadata["anchor_id"].duplicated().any():
        raise AdvantageRouterError("inner-plan anchor IDs are not unique")
    result: dict[str, tuple[InnerBlock, ...]] = {}
    for specification in windows:
        if len(specification) != 3:
            raise AdvantageRouterError("outer window specification changed")
        fold, start_text, _end_text = map(str, specification)
        cursor = _utc(start_text) - pd.Timedelta(hours=outer_embargo_hours)
        reverse_spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _ in range(4):
            start = cursor - pd.Timedelta(days=block_days)
            reverse_spans.append((start, cursor))
            cursor = start - pd.Timedelta(hours=block_gap_hours)
        blocks: list[InnerBlock] = []
        for index, (start, end) in enumerate(reversed(reverse_spans), start=1):
            candidates = metadata.loc[
                metadata["anchor_time"].ge(start) & metadata["anchor_time"].lt(end)
            ]
            blocks.append(
                InnerBlock(
                    outer_fold=fold,
                    name=f"I{index}",
                    start=start,
                    end=end,
                    anchor_ids=select_spaced_anchor_ids(candidates, gap_hours=78),
                )
            )
        result[fold] = tuple(blocks)
    return result


def complete_case_ids(frame: pd.DataFrame) -> set[int]:
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "incumbent_prediction",
        "transfer_prediction",
    }
    if not required <= set(frame.columns):
        raise AdvantageRouterError("sealed expert frame schema changed")
    keys = ["fold", "anchor_id", "station", "lead_h"]
    if frame.duplicated(keys).any():
        raise AdvantageRouterError("sealed expert rows are not unique")
    grouped = frame.groupby("anchor_id", sort=False)["lead_h"].agg(["size", "nunique"])
    good = grouped.index[(grouped["size"] == len(LEADS)) & (grouped["nunique"] == len(LEADS))]
    return set(int(value) for value in good)


def prior_fold_support(
    plan: Mapping[str, Sequence[InnerBlock]],
    sealed_experts: pd.DataFrame,
    fold_order: Sequence[str],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return exact expert-supported inner IDs using earlier folds only."""

    available_folds: list[str] = []
    output: dict[str, dict[str, tuple[int, ...]]] = {}
    for fold in fold_order:
        prior = sealed_experts.loc[sealed_experts["fold"].astype(str).isin(available_folds)]
        available = complete_case_ids(prior) if len(prior) else set()
        output[fold] = {
            block.name: tuple(anchor for anchor in block.anchor_ids if anchor in available)
            for block in plan[fold]
        }
        available_folds.append(str(fold))
    return output


def router_support_passes(blocks: Mapping[str, Sequence[int]]) -> bool:
    expected = {"I1", "I2", "I3", "I4"}
    if set(blocks) != expected:
        raise AdvantageRouterError("inner support block names changed")
    return bool(
        all(len(blocks[name]) >= 12 for name in expected)
        and len(blocks["I1"]) + len(blocks["I2"]) >= 24
    )


def build_router_rows(
    feature_rows: pd.DataFrame,
    expert_rows: pd.DataFrame,
    anchor_ids: Sequence[int],
) -> pd.DataFrame:
    """Create the exact label-free long router matrix for selected case IDs."""

    ids = tuple(int(value) for value in anchor_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise AdvantageRouterError("router anchor IDs must be non-empty and unique")
    feature_columns = ["anchor_id", *ROUTER_BASE_FEATURES]
    if not set(feature_columns) <= set(feature_rows.columns):
        raise AdvantageRouterError("frozen feature cache lacks a router input")
    features = feature_rows.loc[feature_rows["anchor_id"].isin(ids), feature_columns].copy()
    if len(features) != len(ids) or features["anchor_id"].duplicated().any():
        raise AdvantageRouterError("router feature rows do not cover selected anchors")
    experts = expert_rows.loc[expert_rows["anchor_id"].isin(ids)].copy()
    if complete_case_ids(experts) != set(ids):
        raise AdvantageRouterError("sealed experts do not cover selected anchors")
    experts = experts.sort_values(["anchor_id", "lead_h"], kind="mergesort")
    merged = experts.merge(features, on="anchor_id", how="left", validate="many_to_one")
    if len(merged) != len(ids) * len(LEADS):
        raise AdvantageRouterError("router long-row count changed")
    lead = merged["lead_h"].to_numpy(dtype=np.float64)
    if set(lead.astype(int)) != set(LEADS):
        raise AdvantageRouterError("router lead set changed")
    incumbent = merged["incumbent_prediction"].to_numpy(dtype=np.float64)
    transfer = merged["transfer_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(incumbent).all() or not np.isfinite(transfer).all():
        raise AdvantageRouterError("sealed expert predictions are non-finite")
    merged["lead_h_div_24"] = lead / 24.0
    merged["transfer_minus_incumbent"] = transfer - incumbent
    merged["abs_transfer_minus_incumbent"] = np.abs(transfer - incumbent)
    validate_router_feature_names(ROUTER_FEATURES)
    return merged.reset_index(drop=True)


def attach_truth(rows: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "anchor_id", "station", "lead_h"]
    required = {*keys, "target_hs"}
    if not required <= set(truth.columns):
        raise AdvantageRouterError("truth release schema changed")
    if truth.duplicated(keys).any():
        raise AdvantageRouterError("truth release keys are not unique")
    merged = rows.merge(
        truth.loc[:, [*keys, "target_hs"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    target = merged["target_hs"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise AdvantageRouterError("truth release does not cover router rows")
    return merged


def squared_error_advantage(rows: pd.DataFrame) -> np.ndarray:
    target = rows["target_hs"].to_numpy(dtype=np.float64)
    incumbent = rows["incumbent_prediction"].to_numpy(dtype=np.float64)
    transfer = rows["transfer_prediction"].to_numpy(dtype=np.float64)
    advantage = np.square(target - incumbent) - np.square(target - transfer)
    if not np.isfinite(advantage).all():
        raise AdvantageRouterError("advantage target is non-finite")
    return advantage


def fit_advantage_router(rows: pd.DataFrame) -> FittedAdvantageRouter:
    frame = rows.loc[:, ROUTER_FEATURES]
    validate_router_feature_names(tuple(frame.columns))
    matrix = frame.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(matrix)
    medians = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[finite[:, column], column]
        if not len(values):
            raise AdvantageRouterError(
                f"router fit column is entirely missing: {ROUTER_FEATURES[column]}"
            )
        medians[column] = float(np.median(values))
        matrix[~finite[:, column], column] = medians[column]
    scaler = StandardScaler()
    transformed = scaler.fit_transform(matrix)
    ridge = Ridge(alpha=100.0)
    ridge.fit(transformed, squared_error_advantage(rows))
    model = FittedAdvantageRouter(medians, scaler, ridge, ROUTER_FEATURES)
    if not np.isfinite(model.predict(frame)).all():
        raise AdvantageRouterError("router fit smoke prediction failed")
    return model


def calibrate_tau(model: FittedAdvantageRouter, rows: pd.DataFrame) -> float:
    residual = np.abs(squared_error_advantage(rows) - model.predict(rows.loc[:, ROUTER_FEATURES]))
    tau = float(np.quantile(residual, 0.90, method="linear"))
    if not np.isfinite(tau) or tau < 0.0:
        raise AdvantageRouterError("I3 tau is invalid")
    return tau


def apply_bounded_router(
    rows: pd.DataFrame,
    predicted_advantage: np.ndarray,
    tau: float,
    *,
    strength: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    if strength != 0.20:
        raise AdvantageRouterError("router blend strength changed")
    incumbent = rows["incumbent_prediction"].to_numpy(dtype=np.float64)
    transfer = rows["transfer_prediction"].to_numpy(dtype=np.float64)
    predicted = np.asarray(predicted_advantage, dtype=np.float64)
    if predicted.shape != incumbent.shape or not np.isfinite(predicted).all():
        raise AdvantageRouterError("predicted advantage shape or values are invalid")
    active = predicted > float(tau)
    candidate = incumbent.copy()
    candidate[active] = incumbent[active] + strength * (transfer[active] - incumbent[active])
    if candidate[~active].tobytes() != incumbent[~active].tobytes():
        raise AdvantageRouterError("inactive router rows are not bit-exact incumbent")
    if not np.isfinite(candidate).all() or (candidate < 0.0).any() or (candidate > 30.0).any():
        raise AdvantageRouterError("bounded router prediction is outside the frozen range")
    return candidate, active


def exact_incumbent_fallback(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    incumbent = rows["incumbent_prediction"].to_numpy(dtype=np.float64)
    candidate = incumbent.copy()
    if candidate.tobytes() != incumbent.tobytes():
        raise AdvantageRouterError("incumbent fallback is not byte-exact")
    return candidate, np.zeros(len(rows), dtype=bool)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AdvantageRouterError("RMSE arrays are invalid")
    return float(np.sqrt(np.mean(np.square(left - right))))


def metric_pair(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    candidate = frame["candidate_prediction"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(truth, incumbent)
    candidate_rmse = rmse(truth, candidate)
    return {
        "rows": int(len(frame)),
        "cases": int(frame["anchor_id"].nunique()),
        "incumbent_rmse_m": baseline_rmse,
        "candidate_rmse_m": candidate_rmse,
        "delta_m": candidate_rmse - baseline_rmse,
    }


def paired_case_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 20260828,
) -> dict[str, float | int | str]:
    grouped = list(frame.groupby("anchor_id", sort=True, observed=True))
    if not grouped:
        raise AdvantageRouterError("bootstrap received no cases")
    incumbent_sse = np.asarray(
        [np.square(group["target_hs"] - group["incumbent_prediction"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    candidate_sse = np.asarray(
        [np.square(group["target_hs"] - group["candidate_prediction"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[selected].sum()
        deltas[draw] = np.sqrt(candidate_sse[selected].sum() / denominator) - np.sqrt(
            incumbent_sse[selected].sum() / denominator
        )
    return {
        "unit": "anchor_id_complete_six_lead_case",
        "cases": int(len(grouped)),
        "replicates": int(replicates),
        "seed": int(seed),
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
    }


def evaluate_gate(frame: pd.DataFrame, *, require_fold_consistency: bool) -> dict[str, Any]:
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "incumbent_prediction",
        "candidate_prediction",
        "router_active",
    }
    if not required <= set(frame.columns):
        raise AdvantageRouterError("gate frame schema changed")
    overall = metric_pair(frame)
    bootstrap = paired_case_bootstrap(frame)
    by_fold = {
        str(name): metric_pair(block)
        for name, block in frame.groupby("fold", sort=True, observed=True)
    }
    by_station = {
        str(name): metric_pair(block)
        for name, block in frame.groupby("station", sort=True, observed=True)
    }
    by_lead = {
        str(int(name)): metric_pair(block)
        for name, block in frame.groupby("lead_h", sort=True, observed=True)
    }
    if set(by_station) != {"G-ORS", "I-ORS", "S-ORS"}:
        raise AdvantageRouterError("gate does not cover all three stations")
    if set(by_lead) != {str(lead) for lead in LEADS}:
        raise AdvantageRouterError("gate does not cover all six leads")
    coverage = float(frame["router_active"].astype(bool).mean())
    maximum_slice = max(
        [float(item["delta_m"]) for item in by_station.values()]
        + [float(item["delta_m"]) for item in by_lead.values()]
    )
    checks: dict[str, bool] = {
        "delta_at_most_minus_0_003": float(overall["delta_m"]) <= -0.003,
        "case_bootstrap_ci90_upper_below_zero": float(bootstrap["ci90_upper_m"]) < 0.0,
        "coverage_between_0_05_and_0_50": 0.05 <= coverage <= 0.50,
        "all_station_and_lead_regressions_at_most_0_0075": maximum_slice <= 0.0075,
    }
    if require_fold_consistency:
        checks.pop("coverage_between_0_05_and_0_50")
        checks["at_least_two_of_three_folds_improve"] = (
            sum(float(item["delta_m"]) < 0.0 for item in by_fold.values()) >= 2
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overall": overall,
        "coverage": coverage,
        "by_fold": by_fold,
        "by_station": by_station,
        "by_lead": by_lead,
        "maximum_station_or_lead_regression_m": maximum_slice,
        "paired_case_bootstrap": bootstrap,
    }


__all__ = [
    "AdvantageRouterError",
    "EXPERIMENT_ID",
    "FittedAdvantageRouter",
    "InnerBlock",
    "LEADS",
    "ROUTER_BASE_FEATURES",
    "ROUTER_DERIVED_FEATURES",
    "ROUTER_FEATURES",
    "apply_bounded_router",
    "attach_truth",
    "build_inner_block_plan",
    "build_router_rows",
    "calibrate_tau",
    "canonical_json_bytes",
    "complete_case_ids",
    "evaluate_gate",
    "exact_incumbent_fallback",
    "file_pin",
    "fit_advantage_router",
    "paired_case_bootstrap",
    "prior_fold_support",
    "router_support_passes",
    "select_spaced_anchor_ids",
    "sha256_bytes",
    "sha256_file",
    "write_json_exclusive",
    "write_npy_exclusive",
    "write_parquet_exclusive",
]
