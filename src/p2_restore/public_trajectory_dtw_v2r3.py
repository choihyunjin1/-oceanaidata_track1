"""Corrected deterministic zero-fit public-trajectory transfer for P2 v2r3.

This module contains numerical code only.  The sealed runner imports it inside
the authorized worker, never during the default read-only preflight.  It does
not know about official test, sample, candidate, or upload paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
KST = "Asia/Seoul"
RAW_STEP_MINUTES = 10
DTW_STEP_MINUTES = 60
DTW_CORRIDOR_HOURS = 12
SOURCE_EMBARGO_DAYS = 7
MIN_PUBLIC_COVERAGE = 0.70
MAX_NORMALIZED_DTW_DISTANCE = 3.0
MIN_FINITE_NEIGHBORS = 3
PREFILTER_CANDIDATES = 16
MAX_DTW_MATRIX_CELLS = 400_000
MAX_WORKING_ARRAY_BYTES = 512 * 1024 * 1024
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260826


@dataclass(frozen=True, order=True)
class CellSpec:
    cell_id: str
    context_days: int
    neighbors: int


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    start_kst: str
    end_exclusive_kst: str
    source_end_lte_kst: str | None = None


@dataclass(frozen=True)
class DtwResult:
    normalized_distance: float
    path: tuple[tuple[int, int], ...]
    paired_channel_fraction: float


CELLS = (
    CellSpec("d1_k3", 1, 3),
    CellSpec("d1_k7", 1, 7),
    CellSpec("d3_k3", 3, 3),
    CellSpec("d3_k7", 3, 7),
    CellSpec("d7_k3", 7, 3),
    CellSpec("d7_k7", 7, 7),
)

INNER_WINDOWS = (
    WindowSpec(
        "inner_2024_mar",
        "2024-03-01T00:00:00+09:00",
        "2024-04-01T00:00:00+09:00",
        "2024-02-22T23:50:00+09:00",
    ),
    WindowSpec(
        "inner_2024_may",
        "2024-05-01T00:00:00+09:00",
        "2024-06-01T00:00:00+09:00",
        "2024-04-23T23:50:00+09:00",
    ),
    WindowSpec(
        "inner_2024_jul",
        "2024-07-01T00:00:00+09:00",
        "2024-08-01T00:00:00+09:00",
        "2024-06-23T23:50:00+09:00",
    ),
)

EXACT_WINDOW = WindowSpec(
    "exact_2024_sep_oct",
    "2024-09-01T00:00:00+09:00",
    "2024-11-01T00:00:00+09:00",
)

P100_FOLDS = (
    "outer_2024_sep_oct",
    "outer_2025_may_jun",
    "outer_2025_jul_aug",
)
P100_FOLD_WINDOWS_KST = {
    "outer_2024_sep_oct": ("2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"),
    "outer_2025_may_jun": ("2025-05-01T00:00:00+09:00", "2025-07-01T00:00:00+09:00"),
    "outer_2025_jul_aug": ("2025-07-01T00:00:00+09:00", "2025-09-01T00:00:00+09:00"),
}

CHANNEL_GROUP_WEIGHTS = {
    "temp_level": 1.0,
    "psal_level": 0.5,
    "delta": 0.5,
    "missing": 0.25,
    "m2": 0.25,
}


def materialization_slots() -> tuple[dict[str, Any], ...]:
    slots: list[dict[str, Any]] = []
    ordinal = 1
    for window in INNER_WINDOWS:
        for cell in CELLS:
            slots.append(
                {
                    "slot": ordinal,
                    "stage": "INNER",
                    "window_id": window.window_id,
                    "cell_id": cell.cell_id,
                    "conditional": False,
                }
            )
            ordinal += 1
    slots.append(
        {
            "slot": ordinal,
            "stage": "EXACT",
            "window_id": EXACT_WINDOW.window_id,
            "cell_id": "INNER_SELECTED",
            "conditional": False,
        }
    )
    ordinal += 1
    for fold in P100_FOLDS:
        slots.append(
            {
                "slot": ordinal,
                "stage": "P100",
                "window_id": fold,
                "cell_id": "INNER_SELECTED",
                "conditional": True,
            }
        )
        ordinal += 1
    assert len(slots) == 22 and ordinal == 23
    return tuple(slots)


def normalize_utc_ns(values: Sequence[Any] | pd.Series | pd.Index) -> np.ndarray:
    """Return UTC epoch nanoseconds without pandas' integer-unit ambiguity."""

    parsed = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if parsed.isna().any():
        raise ValueError("timestamp key contains NaT")
    parsed_ns = parsed.as_unit("ns")
    ns = parsed_ns.asi8.copy()
    roundtrip = pd.DatetimeIndex(pd.to_datetime(ns, unit="ns", utc=True)).as_unit("ns")
    if not np.array_equal(roundtrip.asi8, ns):
        raise ValueError("UTC nanosecond timestamp roundtrip failed")
    return ns


def utc_ns_to_datetime(values: Sequence[int] | np.ndarray) -> pd.DatetimeIndex:
    ns = np.asarray(values, dtype=np.int64)
    result = pd.DatetimeIndex(pd.to_datetime(ns, unit="ns", utc=True)).as_unit("ns")
    if result.isna().any() or not np.array_equal(result.asi8, ns):
        raise ValueError("UTC nanosecond reconstruction failed")
    return result


def exact_time_contract(values: Sequence[Any] | pd.Series | pd.Index) -> dict[str, Any]:
    ns = normalize_utc_ns(values)
    if len(ns) == 0:
        raise ValueError("exact key set is empty")
    unique = np.unique(ns)
    restored = utc_ns_to_datetime(unique)
    kst = restored.tz_convert(KST)
    expected_start = pd.Timestamp(EXACT_WINDOW.start_kst)
    expected_end = pd.Timestamp(EXACT_WINDOW.end_exclusive_kst) - pd.Timedelta(minutes=10)
    if kst.min() != expected_start or kst.max() != expected_end:
        raise ValueError("exact 2024 Sep-Oct timestamp bounds differ from preregistration")
    day_count = int(pd.Index(kst.normalize()).nunique())
    if day_count != 61:
        raise ValueError("exact 2024 Sep-Oct surface must span 61 KST days")
    diffs = np.diff(unique)
    expected_step = RAW_STEP_MINUTES * 60 * 1_000_000_000
    if len(diffs) and not np.all(diffs == expected_step):
        raise ValueError("exact timestamp grid is not continuous 10-minute UTC-ns")
    return {
        "minimum_kst": kst.min().isoformat(),
        "maximum_kst": kst.max().isoformat(),
        "kst_days": day_count,
        "unique_timestamps": int(len(unique)),
        "unit": "ns",
        "roundtrip_identity": True,
        "nat_count": 0,
    }


def align_on_utc_ns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_time: str = "time",
    right_time: str = "time",
    key_columns: Sequence[str] = ("layer",),
) -> pd.DataFrame:
    """One-to-one alignment using explicit UTC-ns integer keys."""

    left_copy = left.copy()
    right_copy = right.copy()
    left_copy["_time_ns"] = normalize_utc_ns(left_copy[left_time])
    right_copy["_time_ns"] = normalize_utc_ns(right_copy[right_time])
    keys = [*key_columns, "_time_ns"]
    if left_copy.duplicated(keys).any() or right_copy.duplicated(keys).any():
        raise ValueError("duplicate UTC-ns alignment key")
    merged = left_copy.merge(
        right_copy,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
        sort=True,
    )
    if len(merged) != len(left_copy) or len(merged) != len(right_copy):
        raise ValueError("UTC-ns alignment lost or added rows")
    merged["time_utc"] = utc_ns_to_datetime(merged["_time_ns"].to_numpy())
    return merged


def key_digest(frame: pd.DataFrame) -> str:
    """Digest sorted (UTC-ns time, layer) keys without serializing row data."""

    if not {"time", "layer"}.issubset(frame.columns):
        raise ValueError("key digest requires time and layer")
    keys = pd.DataFrame(
        {
            "time_ns": normalize_utc_ns(frame["time"]),
            "layer": pd.to_numeric(frame["layer"], errors="raise").astype("int16"),
        }
    ).sort_values(["time_ns", "layer"], kind="mergesort")
    if keys.duplicated(["time_ns", "layer"]).any():
        raise ValueError("duplicate time-layer key")
    payload = keys[["time_ns", "layer"]].to_records(index=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def canonical_key_digest(
    frame: pd.DataFrame,
    *,
    include_fold_station: bool,
) -> str:
    """Digest one explicitly sorted canonical key, including p100 metadata."""

    required = {"time", "layer"}
    if include_fold_station:
        required.update({"fold", "station"})
    if not required.issubset(frame.columns):
        raise ValueError("canonical key columns are incomplete")
    keys = pd.DataFrame(
        {
            "time_ns": normalize_utc_ns(frame["time"]),
            "layer": pd.to_numeric(frame["layer"], errors="raise").astype("int16"),
        }
    )
    sort_columns = ["time_ns", "layer"]
    if include_fold_station:
        keys["fold"] = frame["fold"].astype("string").to_numpy()
        keys["station"] = frame["station"].astype("string").to_numpy()
        sort_columns = ["fold", "station", "time_ns", "layer"]
    keys = keys.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    if keys.duplicated(sort_columns).any():
        raise ValueError("duplicate canonical key")
    payload = b"\n".join(
        "\x1f".join(str(value) for value in row).encode("utf-8")
        for row in keys[sort_columns].itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload).hexdigest()


def verify_p100_anchor_against_observations(
    observations: pd.DataFrame,
    p100_anchor: pd.DataFrame,
    *,
    anchor_prediction_column: str = "INCUMBENT_NOOP",
    expected_rows: int = 78156,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Key-bind p100 fold/station metadata and truth in one canonical merge."""

    anchor_required = {
        "fold",
        "station",
        "time",
        "layer",
        "truth",
        anchor_prediction_column,
    }
    observation_required = {"station", "time", "layer", "temp"}
    if not anchor_required.issubset(p100_anchor.columns) or len(p100_anchor) != expected_rows:
        raise ValueError("p100 anchor schema or registered row count differs")
    if not observation_required.issubset(observations.columns):
        raise ValueError("historical observations cannot verify p100 station truth")

    anchor = p100_anchor[
        ["fold", "station", "time", "layer", "truth", anchor_prediction_column]
    ].copy()
    anchor["_time_ns"] = normalize_utc_ns(anchor["time"])
    anchor["fold"] = anchor["fold"].astype("string")
    anchor["station"] = anchor["station"].astype("string")
    anchor["layer"] = pd.to_numeric(anchor["layer"], errors="raise").astype("int16")
    canonical = ["fold", "station", "_time_ns", "layer"]
    if anchor.duplicated(canonical).any():
        raise ValueError("p100 anchor contains duplicate canonical keys")

    source = observations[["station", "time", "layer", "temp"]].copy()
    source["_time_ns"] = normalize_utc_ns(source["time"])
    source["station"] = source["station"].astype("string")
    source["layer"] = pd.to_numeric(source["layer"], errors="raise").astype("int16")
    source = source.loc[source["layer"].isin(TARGET_LAYERS)].copy()
    observation_key = ["station", "_time_ns", "layer"]
    if source.duplicated(observation_key).any():
        raise ValueError("historical p100 truth contains duplicate station-time-layer keys")

    merged = anchor.merge(
        source[[*observation_key, "temp"]],
        on=observation_key,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    observed = merged["temp"].to_numpy(dtype=np.float64)
    claimed = merged["truth"].to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all() or not np.array_equal(observed, claimed):
        raise ValueError("p100 anchor truth differs from keyed historical observations")
    if tuple(sorted(merged["fold"].astype(str).unique())) != tuple(sorted(P100_FOLDS)):
        raise ValueError("locked p100 fold ids differ from registered contract")
    time_utc = utc_ns_to_datetime(merged["_time_ns"].to_numpy())
    time_kst = time_utc.tz_convert(KST)
    for fold, (start_text, end_text) in P100_FOLD_WINDOWS_KST.items():
        members = merged["fold"].astype(str).to_numpy() == fold
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        if not members.any() or not ((time_kst[members] >= start) & (time_kst[members] < end)).all():
            raise ValueError("p100 fold metadata does not match its registered time window")

    merged = merged.sort_values(canonical, kind="mergesort").reset_index(drop=True)
    verified = pd.DataFrame(
        {
            "fold": merged["fold"].astype(str),
            "station": merged["station"].astype(str),
            "time": utc_ns_to_datetime(merged["_time_ns"].to_numpy()),
            "layer": merged["layer"].astype(int),
            "truth": merged["temp"].to_numpy(dtype=np.float64),
            "anchor": merged[anchor_prediction_column].to_numpy(dtype=np.float64),
        }
    )
    if not np.isfinite(verified[["truth", "anchor"]].to_numpy(dtype=float)).all():
        raise ValueError("verified p100 anchor contains nonfinite score values")
    key_sha = canonical_key_digest(verified, include_fold_station=True)
    metadata_payload = b"\n".join(
        f"{row.fold}\x1f{row.station}\x1f{int(row.time_ns)}\x1f{int(row.layer)}".encode()
        for row in merged[["fold", "station", "_time_ns", "layer"]]
        .rename(columns={"_time_ns": "time_ns"})
        .itertuples(index=False)
    )
    certificate = {
        "rows": int(len(verified)),
        "canonical_key_columns": ["fold", "station", "UTC_ns_time", "layer"],
        "canonical_key_sha256": key_sha,
        "key_sha256": key_sha,
        "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "truth_sha256": hashlib.sha256(verified["truth"].to_numpy().tobytes()).hexdigest(),
        "anchor_truth_equal_keyed_observations": True,
        "positional_metadata_assignment_used": False,
    }
    return verified, certificate


def build_observation_truth_for_keys(
    observations: pd.DataFrame,
    requested_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild truth independently from observations for an exact key set."""

    required_observations = {"time", "layer", "temp"}
    if not required_observations.issubset(observations.columns):
        raise ValueError("historical observations cannot independently establish truth")
    if not {"time", "layer"}.issubset(requested_keys.columns):
        raise ValueError("requested truth keys are incomplete")
    source = observations[["time", "layer", "temp"]].copy()
    source["_time_ns"] = normalize_utc_ns(source["time"])
    source["layer"] = pd.to_numeric(source["layer"], errors="raise").astype("int16")
    source = source.loc[source["layer"].isin(TARGET_LAYERS)].copy()
    if source.duplicated(["_time_ns", "layer"]).any():
        raise ValueError("historical observation truth has duplicate keys")
    requested = requested_keys[["time", "layer"]].copy()
    requested["_time_ns"] = normalize_utc_ns(requested["time"])
    requested["layer"] = pd.to_numeric(requested["layer"], errors="raise").astype("int16")
    if requested.duplicated(["_time_ns", "layer"]).any():
        raise ValueError("requested truth has duplicate keys")
    merged = requested[["_time_ns", "layer"]].merge(
        source[["_time_ns", "layer", "temp"]],
        on=["_time_ns", "layer"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    if len(merged) != len(requested) or not np.isfinite(merged["temp"].to_numpy(dtype=float)).all():
        raise ValueError("independent historical truth is missing or nonfinite")
    merged = merged.rename(columns={"temp": "truth"})
    merged["time"] = utc_ns_to_datetime(merged["_time_ns"].to_numpy())
    return merged[["time", "layer", "truth"]].reset_index(drop=True)


def verify_anchor_against_observations(
    observations: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    anchor_prediction_column: str,
    expected_rows: int,
    require_exact_contract: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify anchor key and truth independently, then return trusted truth."""

    required = {"time", "layer", "truth", anchor_prediction_column}
    if not required.issubset(anchor.columns) or len(anchor) != expected_rows:
        raise ValueError("anchor schema or registered row count differs")
    anchor_work = anchor[list(required)].copy()
    observation_truth = build_observation_truth_for_keys(observations, anchor_work)
    aligned = align_on_utc_ns(
        observation_truth,
        anchor_work,
        key_columns=("layer",),
    )
    observed = aligned["truth_left"].to_numpy(dtype=np.float64)
    claimed = aligned["truth_right"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed, claimed, equal_nan=False):
        raise ValueError("anchor truth differs from independently reconstructed observations")
    trusted = pd.DataFrame(
        {
            "time": aligned["time_utc"],
            "layer": aligned["layer"].astype(int),
            "truth": observed,
            "anchor": aligned[f"{anchor_prediction_column}"].to_numpy(dtype=np.float64),
        }
    ).sort_values(["time", "layer"], kind="mergesort").reset_index(drop=True)
    if not np.isfinite(trusted[["truth", "anchor"]].to_numpy(dtype=float)).all():
        raise ValueError("verified anchor contains nonfinite score values")
    time_contract = exact_time_contract(trusted["time"]) if require_exact_contract else None
    certificate = {
        "rows": int(len(trusted)),
        "key_sha256": key_digest(trusted),
        "truth_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "anchor_truth_equal_observations": True,
        "time_contract": time_contract,
    }
    return trusted, certificate


def verify_candidate_against_truth(
    candidate: pd.DataFrame,
    trusted_anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bind a candidate to independently verified truth without trusting its truth column."""

    include_metadata = {"fold", "station"}.issubset(trusted_anchor.columns)
    required = {"time", "layer", "candidate", "support"}
    if include_metadata:
        required.update({"fold", "station"})
    if not required.issubset(candidate.columns):
        raise ValueError("candidate frame is missing key or prediction fields")
    keep = [name for name in candidate.columns if name not in {"truth", "anchor"}]
    candidate_work = candidate[keep].copy()
    left_digest = canonical_key_digest(
        trusted_anchor,
        include_fold_station=include_metadata,
    )
    right_digest = canonical_key_digest(
        candidate_work,
        include_fold_station=include_metadata,
    )
    if left_digest != right_digest or len(candidate_work) != len(trusted_anchor):
        raise ValueError("candidate keys differ from independently verified anchor keys")
    trusted = trusted_anchor.copy()
    trusted["_time_ns"] = normalize_utc_ns(trusted["time"])
    candidate_work["_time_ns"] = normalize_utc_ns(candidate_work["time"])
    key_columns = ["_time_ns", "layer"]
    if include_metadata:
        key_columns = ["fold", "station", "_time_ns", "layer"]
    if trusted.duplicated(key_columns).any() or candidate_work.duplicated(key_columns).any():
        raise ValueError("candidate canonical key is duplicated")
    aligned = trusted.merge(
        candidate_work,
        on=key_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_trusted", "_candidate"),
        sort=False,
    )
    if len(aligned) != len(trusted):
        raise ValueError("candidate canonical merge lost verified truth rows")
    result = pd.DataFrame(
        {
            "time": utc_ns_to_datetime(aligned["_time_ns"].to_numpy()),
            "layer": aligned["layer"].astype(int),
            "truth": aligned["truth"].to_numpy(dtype=np.float64),
            "anchor": aligned["anchor"].to_numpy(dtype=np.float64),
            "candidate": aligned["candidate"].to_numpy(dtype=np.float64),
            "support": aligned["support"].astype(bool).to_numpy(),
        }
    )
    if include_metadata:
        result["fold"] = aligned["fold"].astype(str).to_numpy()
        result["station"] = aligned["station"].astype(str).to_numpy()
    for name in (
        "dtw_distance",
        "public_missingness",
        "trajectory_coverage",
    ):
        if name in aligned:
            result[name] = aligned[name].to_numpy()
    if not np.isfinite(result[["truth", "anchor", "candidate"]].to_numpy(dtype=float)).all():
        raise ValueError("candidate scoring values are nonfinite")
    certificate = {
        "rows": int(len(result)),
        "key_sha256": left_digest,
        "canonical_key_columns": (
            ["fold", "station", "UTC_ns_time", "layer"]
            if include_metadata
            else ["UTC_ns_time", "layer"]
        ),
        "candidate_key_equal_verified_truth": True,
        "candidate_truth_column_used": False,
    }
    result_sort = ["time", "layer"]
    if include_metadata:
        result_sort = ["fold", "station", "time", "layer"]
    result = result.sort_values(result_sort, kind="mergesort").reset_index(drop=True)
    result.attrs["truth_verification"] = certificate
    return result, certificate


def assert_query_feature_firewall(columns: Iterable[str]) -> None:
    normalized = {str(column).strip().lower() for column in columns}
    forbidden_exact = {
        "target",
        "truth",
        "label",
        "residual",
        "prediction",
        "official_score",
        "official_rank",
    }
    forbidden_exact.update(f"temp_{layer}" for layer in TARGET_LAYERS)
    forbidden_exact.update(f"psal_{layer}" for layer in TARGET_LAYERS)
    bad = sorted(normalized.intersection(forbidden_exact))
    bad.extend(
        sorted(
            column
            for column in normalized
            if any(token in column for token in ("sample_submission", "test_index", "candidate"))
        )
    )
    if bad:
        raise ValueError(f"target leakage firewall rejected query columns: {sorted(set(bad))}")


def validate_continuous_block(
    values: Sequence[Any] | pd.Series | pd.Index,
    *,
    expected_step_minutes: int,
    expected_rows: int | None = None,
) -> np.ndarray:
    ns = normalize_utc_ns(values)
    if expected_rows is not None and len(ns) != expected_rows:
        raise ValueError("trajectory block row count differs from sealed context")
    if len(np.unique(ns)) != len(ns):
        raise ValueError("trajectory block contains duplicate timestamps")
    expected = expected_step_minutes * 60 * 1_000_000_000
    if len(ns) > 1 and not np.all(np.diff(ns) == expected):
        raise ValueError("trajectory block is not a continuous prefix")
    return ns


def validate_source_before_query(
    source_times: Sequence[Any] | pd.Series | pd.Index,
    query_times: Sequence[Any] | pd.Series | pd.Index,
) -> None:
    source_ns = normalize_utc_ns(source_times)
    query_ns = normalize_utc_ns(query_times)
    if not len(source_ns) or not len(query_ns):
        raise ValueError("source and query blocks must be nonempty")
    step_ns = DTW_STEP_MINUTES * 60 * 1_000_000_000
    validate_source_end_before_trajectory_start(
        int(source_ns.max()) + step_ns,
        int(query_ns.min()),
    )


def validate_source_end_before_trajectory_start(
    source_end_exclusive: Any,
    trajectory_start: Any,
) -> dict[str, int]:
    """Prove the seven-day embargo against trajectory start, not prediction day.

    Integer inputs are explicitly UTC nanoseconds. Datetime-like inputs are
    normalized with :func:`normalize_utc_ns`; pandas' ambiguous default unit is
    never used.
    """

    def one_ns(value: Any) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value)
        return int(normalize_utc_ns([value])[0])

    source_end_ns = one_ns(source_end_exclusive)
    trajectory_start_ns = one_ns(trajectory_start)
    embargo_ns = SOURCE_EMBARGO_DAYS * 86_400 * 1_000_000_000
    observed = trajectory_start_ns - source_end_ns
    if observed < embargo_ns:
        raise ValueError("source endpoint violates seven-day embargo before query trajectory start")
    return {
        "source_end_utc_ns": source_end_ns,
        "trajectory_start_utc_ns": trajectory_start_ns,
        "embargo_ns": observed,
        "minimum_embargo_ns": embargo_ns,
    }


def _local_cost(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    paired = np.isfinite(left) & np.isfinite(right) & (weights > 0)
    fraction = float(paired.sum() / max(1, int((weights > 0).sum())))
    if not paired.any():
        return math.inf, fraction
    active = weights[paired]
    delta = left[paired] - right[paired]
    return float(np.sum(active * delta * delta) / np.sum(active)), fraction


def constrained_dtw(
    query: np.ndarray,
    source: np.ndarray,
    channel_weights: Sequence[float] | np.ndarray,
    *,
    corridor_steps: int = DTW_CORRIDOR_HOURS,
) -> DtwResult:
    """Endpoint-anchored DTW with band and local slope in [0.5, 2]."""

    query = np.asarray(query, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    weights = np.asarray(channel_weights, dtype=np.float64)
    if query.ndim != 2 or source.ndim != 2 or query.shape[1] != source.shape[1]:
        raise ValueError("DTW arrays must be two-dimensional with identical channels")
    if weights.shape != (query.shape[1],) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("invalid DTW channel weights")
    n, m = len(query), len(source)
    if n == 0 or m == 0 or n * m > MAX_DTW_MATRIX_CELLS:
        raise MemoryError("DTW block exceeds sealed matrix ceiling")
    if corridor_steps < abs(n - m):
        raise ValueError("DTW corridor cannot connect endpoints")
    estimated = (n + 1) * (m + 1) * (8 + 1)
    if estimated > MAX_WORKING_ARRAY_BYTES:
        raise MemoryError("DTW working arrays exceed sealed byte ceiling")

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    predecessor = np.zeros((n + 1, m + 1), dtype=np.int8)
    coverage = np.zeros((n + 1, m + 1), dtype=np.float32)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        low = max(1, i - corridor_steps)
        high = min(m, i + corridor_steps)
        for j in range(low, high + 1):
            local, paired = _local_cost(query[i - 1], source[j - 1], weights)
            if not math.isfinite(local):
                continue
            options = (
                (cost[i - 1, j - 1], 1, i - 1, j - 1),
                (cost[i - 2, j - 1] if i >= 2 else math.inf, 2, i - 2, j - 1),
                (cost[i - 1, j - 2] if j >= 2 else math.inf, 3, i - 1, j - 2),
            )
            prior_cost, move, pi, pj = min(options, key=lambda item: (item[0], item[1]))
            if math.isfinite(prior_cost):
                cost[i, j] = prior_cost + local
                predecessor[i, j] = move
                coverage[i, j] = coverage[pi, pj] + paired
                steps[i, j] = steps[pi, pj] + 1
    if not math.isfinite(float(cost[n, m])) or steps[n, m] <= 0:
        raise ValueError("no feasible constrained DTW path")

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = int(predecessor[i, j])
        if move == 1:
            i, j = i - 1, j - 1
        elif move == 2:
            i, j = i - 2, j - 1
        elif move == 3:
            i, j = i - 1, j - 2
        else:
            raise ValueError("DTW predecessor chain is torn")
    if i != 0 or j != 0:
        raise ValueError("DTW path did not reach both origins")
    path.reverse()
    return DtwResult(
        normalized_distance=float(math.sqrt(cost[n, m] / steps[n, m])),
        path=tuple(path),
        paired_channel_fraction=float(coverage[n, m] / steps[n, m]),
    )


def align_historical_residual(
    path: Sequence[tuple[int, int]], source_residual: np.ndarray, query_rows: int
) -> np.ndarray:
    residual = np.asarray(source_residual, dtype=np.float64)
    if residual.ndim == 1:
        residual = residual[:, None]
    buckets: list[list[int]] = [[] for _ in range(query_rows)]
    for qi, sj in path:
        if not (0 <= qi < query_rows and 0 <= sj < len(residual)):
            raise ValueError("DTW path points outside residual arrays")
        buckets[qi].append(sj)
    aligned = np.full((query_rows, residual.shape[1]), np.nan, dtype=np.float64)
    for qi, source_rows in enumerate(buckets):
        if source_rows:
            aligned[qi] = np.nanmedian(residual[source_rows], axis=0)
    for column in range(aligned.shape[1]):
        finite = np.flatnonzero(np.isfinite(aligned[:, column]))
        if len(finite):
            aligned[:, column] = np.interp(
                np.arange(query_rows), finite, aligned[finite, column]
            )
    return aligned


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if int(keep.sum()) < MIN_FINITE_NEIGHBORS:
        return math.nan
    ordered = np.argsort(values[keep], kind="mergesort")
    selected_values = values[keep][ordered]
    selected_weights = weights[keep][ordered]
    cutoff = 0.5 * selected_weights.sum()
    return float(selected_values[np.searchsorted(np.cumsum(selected_weights), cutoff, side="left")])


def aggregate_neighbor_residuals(
    aligned_residuals: Sequence[np.ndarray], distances: Sequence[float]
) -> np.ndarray:
    if len(aligned_residuals) != len(distances) or len(aligned_residuals) < MIN_FINITE_NEIGHBORS:
        raise ValueError("insufficient aligned historical neighbors")
    stack = np.stack([np.asarray(value, dtype=np.float64) for value in aligned_residuals])
    weights = 1.0 / (1.0 + np.asarray(distances, dtype=np.float64))
    result = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    for row in range(result.shape[0]):
        for layer in range(result.shape[1]):
            result[row, layer] = weighted_median(stack[:, row, layer], weights)
    return result


def _pava(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    counts: list[int] = []
    for value in np.asarray(values, dtype=np.float64):
        levels.append(float(value))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total = counts[-2] + counts[-1]
            merged = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total
            levels[-2:] = [merged]
            counts[-2:] = [total]
    return np.concatenate(
        [np.full(count, level) for level, count in zip(levels, counts, strict=True)]
    )


def project_public_endpoints(l1: float, internal: Sequence[float], l5: float) -> np.ndarray:
    internal_values = np.asarray(internal, dtype=np.float64)
    if not np.isfinite(l1) or not np.isfinite(l5) or not np.isfinite(internal_values).all():
        return internal_values.copy()
    sign = 1.0 if l5 >= l1 else -1.0
    low, high = sorted((sign * l1, sign * l5))
    transformed = np.clip(sign * internal_values, low, high)
    projected = _pava(np.concatenate(([low], transformed, [high])))[1:-1]
    return sign * np.clip(projected, low, high)


def endpoint_safe_candidate(
    l1: float,
    anchor_internal: Sequence[float],
    candidate_internal: Sequence[float],
    l5: float,
    support: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Project while forcing weak or endpoint-missing rows to exact no-op."""

    anchor_values = np.asarray(anchor_internal, dtype=np.float64)
    candidate_values = np.asarray(candidate_internal, dtype=np.float64)
    support_values = np.asarray(support, dtype=bool)
    if (
        anchor_values.shape != candidate_values.shape
        or anchor_values.shape != support_values.shape
        or not np.isfinite(anchor_values).all()
    ):
        raise ValueError("endpoint-safe arrays differ or anchor is nonfinite")
    output = candidate_values.copy()
    output[~support_values] = anchor_values[~support_values]
    if not math.isfinite(l1) or not math.isfinite(l5):
        return anchor_values.copy(), np.zeros_like(support_values), "MISSING_L1_OR_L5_ENDPOINT"
    if not support_values.all():
        return output, support_values, "PARTIAL_WEAK_SUPPORT_NO_PROJECTION"
    projected = project_public_endpoints(l1, output, l5)
    if not np.isfinite(projected).all():
        return anchor_values.copy(), np.zeros_like(support_values), "NONFINITE_PROJECTION_NOOP"
    return projected, support_values, "PROJECTED"


def select_inner_cell(records: Sequence[Mapping[str, Any]]) -> CellSpec:
    expected = {(window.window_id, cell.cell_id) for window in INNER_WINDOWS for cell in CELLS}
    observed = {(str(row["window_id"]), str(row["cell_id"])) for row in records}
    if observed != expected or len(records) != 18:
        raise ValueError("inner selection requires exactly the preregistered 18 materializations")
    allowed = {"window_id", "cell_id", "layer_equal_rmse_c", "worst_layer_rmse_c"}
    if any(set(row) - allowed for row in records):
        raise ValueError("inner selection record contains non-preregistered selection fields")
    summaries: list[tuple[float, float, int, int, str, CellSpec]] = []
    for cell in CELLS:
        rows = [row for row in records if row["cell_id"] == cell.cell_id]
        scores = np.asarray([row["layer_equal_rmse_c"] for row in rows], dtype=float)
        if len(rows) != 3 or not np.isfinite(scores).all():
            raise ValueError("inner selection metric is incomplete or non-finite")
        summaries.append(
            (
                float(scores.mean()),
                float(scores.max()),
                cell.context_days,
                cell.neighbors,
                cell.cell_id,
                cell,
            )
        )
    return min(summaries)[-1]


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("RMSE arrays must be equal-shape and finite")
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def paired_day_bootstrap(
    frame: pd.DataFrame,
    *,
    truth_column: str = "truth",
    candidate_column: str = "candidate",
    anchor_column: str = "anchor",
) -> dict[str, Any]:
    required = {"time", truth_column, candidate_column, anchor_column}
    if not required.issubset(frame.columns):
        raise ValueError("paired bootstrap columns are incomplete")
    ns = normalize_utc_ns(frame["time"])
    kst_days = utc_ns_to_datetime(ns).tz_convert(KST).normalize()
    work = frame[[truth_column, candidate_column, anchor_column]].copy()
    work["day"] = kst_days
    grouped = work.groupby("day", sort=True)
    days = list(grouped.groups)
    if len(days) < 2:
        raise ValueError("paired day bootstrap requires at least two KST days")
    candidate_sse = np.asarray(
        [np.square(group[candidate_column] - group[truth_column]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    anchor_sse = np.asarray(
        [np.square(group[anchor_column] - group[truth_column]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(days), size=(BOOTSTRAP_REPLICATES, len(days)))
    cand = np.sqrt(candidate_sse[draws].sum(axis=1) / counts[draws].sum(axis=1))
    anchor = np.sqrt(anchor_sse[draws].sum(axis=1) / counts[draws].sum(axis=1))
    delta = cand - anchor
    return {
        "unit": "KST_day",
        "day_count": len(days),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "delta_mean_c": float(delta.mean()),
        "ci90_lower_c": float(np.quantile(delta, 0.05)),
        "ci90_upper_c": float(np.quantile(delta, 0.95)),
    }


def score_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"time", "layer", "truth", "anchor", "candidate", "support"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("prediction frame is incomplete")
    if frame[list(required - {"time"})].isna().any().any():
        raise ValueError("prediction frame contains non-finite score values")
    overall_candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    overall_anchor = rmse(frame["truth"].to_numpy(), frame["anchor"].to_numpy())
    by_layer: dict[str, Any] = {}
    for layer in TARGET_LAYERS:
        subset = frame.loc[frame["layer"] == layer]
        if subset.empty:
            raise ValueError("prediction frame is missing a target layer")
        candidate_rmse = rmse(subset["truth"].to_numpy(), subset["candidate"].to_numpy())
        anchor_rmse = rmse(subset["truth"].to_numpy(), subset["anchor"].to_numpy())
        by_layer[str(layer)] = {
            "rows": int(len(subset)),
            "candidate_rmse_c": candidate_rmse,
            "anchor_rmse_c": anchor_rmse,
            "delta_rmse_c": candidate_rmse - anchor_rmse,
        }
    local = frame.copy()
    local["kst_week"] = (
        utc_ns_to_datetime(normalize_utc_ns(local["time"]))
        .tz_convert(KST)
        .tz_localize(None)
        .to_period("W")
        .astype(str)
    )
    weekly: list[float] = []
    for _, subset in local.groupby("kst_week", sort=True):
        weekly.append(
            rmse(subset["truth"].to_numpy(), subset["candidate"].to_numpy())
            - rmse(subset["truth"].to_numpy(), subset["anchor"].to_numpy())
        )
    return {
        "rows": int(len(frame)),
        "candidate_row_pooled_rmse_c": overall_candidate,
        "anchor_row_pooled_rmse_c": overall_anchor,
        "delta_row_pooled_rmse_c": overall_candidate - overall_anchor,
        "candidate_layer_equal_rmse_c": float(np.mean([v["candidate_rmse_c"] for v in by_layer.values()])),
        "anchor_layer_equal_rmse_c": float(np.mean([v["anchor_rmse_c"] for v in by_layer.values()])),
        "by_layer": by_layer,
        "weekly_delta_rmse_c_p90": float(np.quantile(weekly, 0.90)),
        "support_share": float(frame["support"].astype(bool).mean()),
        "bootstrap": paired_day_bootstrap(frame),
    }


def exact_no_op_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    """Derive (never assert) the exact-anchor fallback invariant."""

    required = {"anchor", "candidate", "support"}
    if not required.issubset(frame.columns):
        raise ValueError("fallback integrity frame is incomplete")
    weak = ~frame["support"].astype(bool).to_numpy()
    anchor = frame["anchor"].to_numpy(dtype=np.float64)
    candidate = frame["candidate"].to_numpy(dtype=np.float64)
    equal = bool(np.array_equal(anchor[weak], candidate[weak], equal_nan=False))
    return {
        "weak_support_rows": int(weak.sum()),
        "weak_support_candidate_equals_anchor": equal,
        "all_score_values_finite": bool(
            np.isfinite(frame[["truth", "anchor", "candidate"]].to_numpy(dtype=float)).all()
        ),
    }


def _decile_labels(values: pd.Series, support: pd.Series, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    supported = support.astype(bool).to_numpy() & np.isfinite(numeric)
    labels = np.full(len(values), "UNSUPPORTED_OR_NONFINITE", dtype=object)
    if supported.any():
        edges = np.unique(np.quantile(numeric[supported], np.linspace(0.0, 1.0, 11)))
        if len(edges) == 1:
            labels[supported] = f"{prefix}00"
        else:
            bins = np.searchsorted(edges[1:-1], numeric[supported], side="right")
            labels[supported] = [f"{prefix}{int(index):02d}" for index in bins]
    return pd.Series(labels, index=values.index, dtype="string")


def _slice_summary(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    grouper: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    for key, subset in frame.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = {
            name: str(value) for name, value in zip(columns, key_tuple, strict=True)
        }
        row["rows"] = int(len(subset))
        finite = np.isfinite(
            subset[["truth", "anchor", "candidate"]].to_numpy(dtype=np.float64)
        ).all(axis=1)
        row["finite_rows"] = int(finite.sum())
        if finite.any():
            finite_subset = subset.loc[finite]
            row["delta_rmse_c"] = rmse(
                finite_subset["truth"].to_numpy(),
                finite_subset["candidate"].to_numpy(),
            ) - rmse(
                finite_subset["truth"].to_numpy(),
                finite_subset["anchor"].to_numpy(),
            )
        else:
            row["delta_rmse_c"] = None
        result.append(row)
    return result


def derive_p100_slice_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute and validate every registered p100 slice from row diagnostics."""

    required = {
        "fold",
        "layer",
        "time",
        "truth",
        "anchor",
        "candidate",
        "support",
        "dtw_distance",
        "public_missingness",
        "trajectory_coverage",
    }
    if not required.issubset(frame.columns) or len(frame) != 78156:
        raise ValueError("locked p100 slice frame is incomplete or has wrong row count")
    work = frame.copy()
    kst = utc_ns_to_datetime(normalize_utc_ns(work["time"])).tz_convert(KST)
    month = kst.month.to_numpy()
    work["season"] = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])],
        ["DJF", "MAM", "JJA"],
        default="SON",
    )
    work["distance_decile"] = _decile_labels(work["dtw_distance"], work["support"], "D")
    work["missingness_decile"] = _decile_labels(
        work["public_missingness"], work["support"], "M"
    )
    work["coverage_decile"] = _decile_labels(
        work["trajectory_coverage"], work["support"], "C"
    )
    families: dict[str, list[dict[str, Any]]] = {
        "fold": _slice_summary(work, ["fold"]),
        "layer": _slice_summary(work, ["layer"]),
        "season": _slice_summary(work, ["season"]),
        "distance_decile": _slice_summary(work, ["distance_decile"]),
        "missingness_decile": _slice_summary(work, ["missingness_decile"]),
        "coverage_decile": _slice_summary(work, ["coverage_decile"]),
        "distance_by_layer": _slice_summary(work, ["distance_decile", "layer"]),
    }
    expected_families = {
        "fold",
        "layer",
        "season",
        "distance_decile",
        "missingness_decile",
        "coverage_decile",
        "distance_by_layer",
    }
    row_identities = {
        name: sum(int(row["rows"]) for row in rows) == len(work)
        for name, rows in families.items()
    }
    nonempty_finite = {
        name: all(row["rows"] > 0 and row["finite_rows"] == row["rows"] for row in rows)
        for name, rows in families.items()
    }
    derived_gate = bool(
        set(families) == expected_families
        and all(row_identities.values())
        and all(nonempty_finite.values())
    )
    inventory_payload = json.dumps(
        families,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "families": families,
        "family_row_count_identity": row_identities,
        "family_all_rows_finite": nonempty_finite,
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "required_slices_logged": derived_gate,
        "derived_not_hardcoded": derived_gate,
    }


def derive_p100_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    base = score_prediction_frame(frame)
    by_fold: dict[str, Any] = {}
    for fold, subset in frame.groupby("fold", sort=True):
        score = score_prediction_frame(subset)
        by_fold[str(fold)] = {
            "rows": score["rows"],
            "delta_rmse_c": score["delta_row_pooled_rmse_c"],
        }
    slices = derive_p100_slice_metrics(frame)
    return {
        **base,
        "by_fold": by_fold,
        "slices": slices,
        "required_slices_logged": bool(slices["required_slices_logged"]),
        "p100_tuning_or_rerun_count": 0,
    }


def exact_research_gate(metrics: Mapping[str, Any]) -> bool:
    layers = metrics["by_layer"]
    return bool(
        metrics["delta_row_pooled_rmse_c"] <= -0.060
        and metrics["bootstrap"]["ci90_upper_c"] <= -0.040
        and all(layers[str(layer)]["delta_rmse_c"] <= 0.003 for layer in TARGET_LAYERS)
        and layers["4"]["delta_rmse_c"] < 0.0
        and metrics["weekly_delta_rmse_c_p90"] <= 0.015
        and metrics.get("weak_support_exact_anchor_fallback", False)
        and metrics.get("exact_key_alignment", False)
        and metrics.get("all_values_finite", False)
    )


def p100_research_gate(metrics: Mapping[str, Any]) -> bool:
    folds = metrics["by_fold"]
    layers = metrics["by_layer"]
    fold_deltas = [float(value["delta_rmse_c"]) for value in folds.values()]
    layer_deltas = [float(value["delta_rmse_c"]) for value in layers.values()]
    return bool(
        metrics["delta_row_pooled_rmse_c"] < 0.0
        and metrics["bootstrap"]["ci90_upper_c"] < 0.0
        and sum(value < 0.0 for value in fold_deltas) >= 2
        and sum(value < 0.0 for value in layer_deltas) >= 2
        and max(fold_deltas) <= 0.010
        and max(layer_deltas) <= 0.005
        and metrics.get("required_slices_logged", False)
        and metrics.get("p100_tuning_or_rerun_count") == 0
    )


def check_deadline(deadline_monotonic: float) -> None:
    if time.monotonic() >= deadline_monotonic:
        raise TimeoutError("sealed deterministic materialization deadline reached")


def summarize_contract() -> dict[str, Any]:
    return {
        "cells": [cell.__dict__ for cell in CELLS],
        "inner_windows": [window.__dict__ for window in INNER_WINDOWS],
        "exact_window": EXACT_WINDOW.__dict__,
        "p100_folds": list(P100_FOLDS),
        "materialization_slots": list(materialization_slots()),
        "physical_fit_calls": 0,
        "deterministic": True,
        "timestamp_key_unit": "UTC_ns",
        "source_embargo_days": SOURCE_EMBARGO_DAYS,
        "max_working_array_bytes": MAX_WORKING_ARRAY_BYTES,
    }


@dataclass(frozen=True)
class TrajectoryPanel:
    station: str
    index: pd.DatetimeIndex
    temp: pd.DataFrame
    psal: pd.DataFrame
    nominal_depth: pd.DataFrame
    baseline: pd.DataFrame
    residual: pd.DataFrame
    public_hourly: pd.DataFrame
    residual_hourly: pd.DataFrame
    channel_columns: tuple[str, ...]
    channel_weights: np.ndarray


def _nearest_public_baseline_arrays(
    public_temp: np.ndarray, public_depth: np.ndarray, target_depth: np.ndarray
) -> np.ndarray:
    result = np.full(len(target_depth), np.nan, dtype=np.float64)
    for row in range(len(target_depth)):
        keep = (
            np.isfinite(public_temp[row])
            & np.isfinite(public_depth[row])
            & np.isfinite(target_depth[row])
        )
        if int(keep.sum()) < 2:
            continue
        depths = public_depth[row, keep]
        values = public_temp[row, keep]
        order = np.argsort(depths, kind="mergesort")
        depths, values = depths[order], values[order]
        lower = np.flatnonzero(depths <= target_depth[row])
        upper = np.flatnonzero(depths >= target_depth[row])
        if len(lower) and len(upper):
            left, right = int(lower[-1]), int(upper[0])
        else:
            nearest = np.sort(np.argsort(np.abs(depths - target_depth[row]), kind="mergesort")[:2])
            left, right = int(nearest[0]), int(nearest[1])
        span = depths[right] - depths[left]
        result[row] = (
            values[left]
            if span == 0
            else values[left]
            + (values[right] - values[left])
            * (target_depth[row] - depths[left])
            / span
        )
    return result


def prepare_trajectory_panel(observations: pd.DataFrame) -> TrajectoryPanel:
    required_columns = (
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    )
    if tuple(observations.columns) != required_columns:
        raise ValueError("historical observations schema differs from sealed contract")
    stations = observations["station"].dropna().astype(str).unique()
    if len(stations) != 1:
        raise ValueError("trajectory implementation requires exactly one historical station")
    work = observations.copy()
    work["_time_ns"] = normalize_utc_ns(work["time"])
    keys = ["station", "layer", "_time_ns"]
    if work.duplicated(keys).any():
        raise ValueError("historical observations contain duplicate station-layer-time keys")
    index_ns = np.sort(work["_time_ns"].unique())
    index = utc_ns_to_datetime(index_ns)
    validate_continuous_block(index, expected_step_minutes=RAW_STEP_MINUTES)

    def wide(value: str) -> pd.DataFrame:
        frame = work.pivot(index="_time_ns", columns="layer", values=value).reindex(index_ns)
        frame.index = index
        return frame.reindex(columns=range(1, 9))

    temp, psal, nominal = wide("temp"), wide("psal"), wide("nominal_depth")
    public_temp = temp.loc[:, PUBLIC_LAYERS].to_numpy(dtype=np.float64)
    public_depth = nominal.loc[:, PUBLIC_LAYERS].to_numpy(dtype=np.float64)
    baseline = pd.DataFrame(index=index, columns=TARGET_LAYERS, dtype=float)
    residual = pd.DataFrame(index=index, columns=TARGET_LAYERS, dtype=float)
    for layer in TARGET_LAYERS:
        base = _nearest_public_baseline_arrays(
            public_temp,
            public_depth,
            nominal[layer].to_numpy(dtype=np.float64),
        )
        baseline[layer] = base
        residual[layer] = temp[layer].to_numpy(dtype=np.float64) - base

    public = pd.DataFrame(index=index)
    channel_groups: dict[str, str] = {}
    for layer in PUBLIC_LAYERS:
        public[f"temp_l{layer}"] = temp[layer]
        public[f"psal_l{layer}"] = psal[layer]
        public[f"temp_missing_l{layer}"] = temp[layer].isna().astype(float)
        public[f"psal_missing_l{layer}"] = psal[layer].isna().astype(float)
        channel_groups[f"temp_l{layer}"] = "temp_level"
        channel_groups[f"psal_l{layer}"] = "psal_level"
        channel_groups[f"temp_missing_l{layer}"] = "missing"
        channel_groups[f"psal_missing_l{layer}"] = "missing"
    hourly = public.resample(
        f"{DTW_STEP_MINUTES}min",
        origin="start_day",
        closed="left",
        label="left",
    ).median()
    for name in [column for column in public if "_missing_" in column]:
        hourly[name] = public[name].resample(
            f"{DTW_STEP_MINUTES}min",
            origin="start_day",
            closed="left",
            label="left",
        ).mean()
    for layer in PUBLIC_LAYERS:
        for lag_hours in (6, 24):
            name = f"temp_delta{lag_hours}h_l{layer}"
            hourly[name] = hourly[f"temp_l{layer}"].diff(lag_hours)
            channel_groups[name] = "delta"
            name = f"psal_delta{lag_hours}h_l{layer}"
            hourly[name] = hourly[f"psal_l{layer}"].diff(lag_hours)
            channel_groups[name] = "delta"
    epoch_seconds = hourly.index.as_unit("ns").asi8 / 1e9
    phase = 2 * np.pi * epoch_seconds / (12.42 * 3600)
    hourly["m2_sin"] = np.sin(phase)
    hourly["m2_cos"] = np.cos(phase)
    hourly["m2_amplitude"] = 1.0
    hourly["m2_phase"] = np.arctan2(hourly["m2_sin"], hourly["m2_cos"])
    for name in ("m2_sin", "m2_cos", "m2_amplitude", "m2_phase"):
        channel_groups[name] = "m2"
    columns = tuple(hourly.columns)
    assert_query_feature_firewall(columns)
    weights = np.asarray([CHANNEL_GROUP_WEIGHTS[channel_groups[name]] for name in columns], dtype=float)
    residual_hourly = residual.resample(
        f"{DTW_STEP_MINUTES}min",
        origin="start_day",
        closed="left",
        label="left",
    ).median()
    return TrajectoryPanel(
        station=str(stations[0]),
        index=index,
        temp=temp,
        psal=psal,
        nominal_depth=nominal,
        baseline=baseline,
        residual=residual,
        public_hourly=hourly,
        residual_hourly=residual_hourly,
        channel_columns=columns,
        channel_weights=weights,
    )


def _prefix_scale(prefix: pd.DataFrame, arrays: Sequence[pd.DataFrame]) -> list[np.ndarray]:
    values = prefix.to_numpy(dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    median[~np.isfinite(median)] = 0.0
    return [(frame.to_numpy(dtype=np.float64) - median) / scale for frame in arrays]


def _summary_vector(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(values)
        first = values[0]
        last = values[-1]
        parts = (
            np.nanmedian(values, axis=0),
            np.nanquantile(values, 0.75, axis=0) - np.nanquantile(values, 0.25, axis=0),
            last - first,
            finite.mean(axis=0),
            1.0 - finite.mean(axis=0),
        )
    return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)


def _daily_bounds(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(start=start, end=end, inclusive="left", freq="1D", tz=start.tz)
    return [(day, min(day + pd.Timedelta(days=1), end)) for day in starts]


def _window_from_anchor(fold: str, frame: pd.DataFrame) -> WindowSpec:
    if frame.empty:
        raise ValueError("locked p100 fold is empty")
    ns = normalize_utc_ns(frame["time"])
    restored = utc_ns_to_datetime(ns).tz_convert(KST)
    start = restored.min().floor("D")
    end = restored.max().floor("D") + pd.Timedelta(days=1)
    return WindowSpec(fold, start.isoformat(), end.isoformat())


class HistoricalTrajectoryMaterializer:
    """In-memory deterministic materializer; it never persists row predictions."""

    def __init__(
        self,
        observations: pd.DataFrame,
        exact_anchor: pd.DataFrame,
    ) -> None:
        self._observations = observations.copy()
        self.panel = prepare_trajectory_panel(observations)
        exact = exact_anchor.copy()
        if "block" in exact:
            exact = exact.loc[exact["block"].astype(str) == "2024_sep_oct"].copy()
        self.exact_anchor, self.exact_truth_certificate = verify_anchor_against_observations(
            observations,
            exact,
            anchor_prediction_column="prediction",
            expected_rows=26273,
            require_exact_contract=True,
        )
        self.p100_anchor: pd.DataFrame | None = None
        self.p100_truth_certificate: dict[str, Any] | None = None

    def load_p100_anchor(self, p100_anchor: pd.DataFrame) -> dict[str, Any]:
        """Attach p100 exactly once, after a durably recorded exact gate."""

        if self.p100_anchor is not None:
            raise RuntimeError("locked p100 anchor was already loaded")
        required_p100 = {"fold", "station", "layer", "time", "truth", "INCUMBENT_NOOP"}
        if not required_p100.issubset(p100_anchor.columns) or len(p100_anchor) != 78156:
            raise ValueError("locked p100 anchor does not contain the sealed 78,156 rows")
        if tuple(sorted(p100_anchor["fold"].astype(str).unique())) != tuple(sorted(P100_FOLDS)):
            raise ValueError("locked p100 fold ids differ from sealed contract")
        verified, certificate = verify_p100_anchor_against_observations(
            self._observations,
            p100_anchor,
            anchor_prediction_column="INCUMBENT_NOOP",
            expected_rows=78156,
        )
        self.p100_anchor = verified
        self.p100_truth_certificate = certificate
        return dict(certificate)

    def _anchor_and_window(self, window: WindowSpec | str) -> tuple[WindowSpec, pd.DataFrame | None]:
        if isinstance(window, WindowSpec):
            if window.window_id == EXACT_WINDOW.window_id:
                return window, self.exact_anchor.copy()
            return window, None
        fold = str(window)
        if fold not in P100_FOLDS:
            raise ValueError("non-preregistered p100 fold")
        if self.p100_anchor is None:
            raise RuntimeError("locked p100 requested before exact RESEARCH_GO lazy load")
        anchor = self.p100_anchor.loc[self.p100_anchor["fold"].astype(str) == fold].copy()
        return _window_from_anchor(fold, anchor), anchor

    def _candidate_source_ends(
        self, context_days: int, source_limit: pd.Timestamp
    ) -> list[pd.Timestamp]:
        hourly = self.panel.public_hourly
        first = hourly.index.min() + pd.Timedelta(days=context_days)
        starts = pd.date_range(
            start=first.tz_convert(KST).ceil("D"),
            end=source_limit.tz_convert(KST).floor("D"),
            freq="1D",
            tz=KST,
        )
        ends: list[pd.Timestamp] = []
        expected = context_days * 24
        for end_kst in starts:
            end = end_kst.tz_convert("UTC")
            block = hourly.loc[end - pd.Timedelta(days=context_days) : end - pd.Timedelta(hours=1)]
            residual = self.panel.residual_hourly.reindex(block.index)
            if len(block) != expected or len(residual) != expected:
                continue
            try:
                validate_continuous_block(block.index, expected_step_minutes=DTW_STEP_MINUTES, expected_rows=expected)
            except ValueError:
                continue
            if int(np.isfinite(residual.to_numpy()).sum(axis=0).min()) >= MIN_FINITE_NEIGHBORS:
                ends.append(end)
        return ends

    def _daily_correction(
        self,
        day_start_kst: pd.Timestamp,
        cell: CellSpec,
        deadline_monotonic: float,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        check_deadline(deadline_monotonic)
        day_end_kst = day_start_kst + pd.Timedelta(days=1)
        prediction_day_start = day_start_kst.tz_convert("UTC")
        trajectory_end = prediction_day_start
        trajectory_start = trajectory_end - pd.Timedelta(days=cell.context_days)
        source_limit = trajectory_start - pd.Timedelta(days=SOURCE_EMBARGO_DAYS)
        target_index = pd.date_range(
            start=day_start_kst.tz_convert("UTC"),
            end=day_end_kst.tz_convert("UTC"),
            inclusive="left",
            freq=f"{DTW_STEP_MINUTES}min",
        )
        query = self.panel.public_hourly.loc[
            trajectory_start : trajectory_end - pd.Timedelta(hours=1), self.panel.channel_columns
        ]
        expected = cell.context_days * 24
        if len(query) != expected:
            return pd.DataFrame(index=target_index, columns=TARGET_LAYERS, dtype=float), {"support": False, "reason": "QUERY_NOT_CONTINUOUS"}
        validate_continuous_block(query.index, expected_step_minutes=DTW_STEP_MINUTES, expected_rows=expected)
        public_level_columns = [name for name in self.panel.channel_columns if name.startswith(("temp_l", "psal_l"))]
        coverage = float(np.isfinite(query[public_level_columns].to_numpy()).mean())
        if coverage < MIN_PUBLIC_COVERAGE:
            return pd.DataFrame(index=target_index, columns=TARGET_LAYERS, dtype=float), {"support": False, "reason": "LOW_QUERY_COVERAGE", "coverage": coverage}
        prefix = self.panel.public_hourly.loc[
            : source_limit - pd.Timedelta(hours=1), self.panel.channel_columns
        ]
        if len(prefix) < expected:
            return pd.DataFrame(index=target_index, columns=TARGET_LAYERS, dtype=float), {"support": False, "reason": "SHORT_SOURCE_PREFIX"}
        source_ends = self._candidate_source_ends(cell.context_days, source_limit)
        if len(source_ends) < cell.neighbors:
            return pd.DataFrame(index=target_index, columns=TARGET_LAYERS, dtype=float), {"support": False, "reason": "INSUFFICIENT_SOURCE_BLOCKS"}
        source_frames = [
            self.panel.public_hourly.loc[
                end - pd.Timedelta(days=cell.context_days) : end - pd.Timedelta(hours=1),
                self.panel.channel_columns,
            ]
            for end in source_ends
        ]
        scaled = _prefix_scale(prefix, [query, *source_frames])
        query_scaled, source_scaled = scaled[0], scaled[1:]
        query_summary = _summary_vector(query_scaled)
        prefilter = sorted(
            (
                float(np.linalg.norm(_summary_vector(values) - query_summary)),
                int(end.as_unit("ns").value),
                index,
            )
            for index, (end, values) in enumerate(
                zip(source_ends, source_scaled, strict=True)
            )
        )[:PREFILTER_CANDIDATES]
        matches: list[tuple[float, int, np.ndarray]] = []
        for _, _, index in prefilter:
            check_deadline(deadline_monotonic)
            end = source_ends[index]
            source_frame = source_frames[index]
            validate_source_end_before_trajectory_start(end, trajectory_start)
            validate_source_before_query(source_frame.index, query.index)
            result = constrained_dtw(
                query_scaled,
                source_scaled[index],
                self.panel.channel_weights,
                corridor_steps=DTW_CORRIDOR_HOURS,
            )
            if (
                result.normalized_distance <= MAX_NORMALIZED_DTW_DISTANCE
                and result.paired_channel_fraction >= MIN_PUBLIC_COVERAGE
            ):
                source_residual = self.panel.residual_hourly.reindex(source_frame.index).to_numpy(dtype=float)
                aligned = align_historical_residual(result.path, source_residual, len(query))
                matches.append((result.normalized_distance, int(end.as_unit("ns").value), aligned))
        matches.sort(key=lambda value: (value[0], value[1]))
        selected = matches[: cell.neighbors]
        if len(selected) < cell.neighbors:
            return pd.DataFrame(index=target_index, columns=TARGET_LAYERS, dtype=float), {"support": False, "reason": "INSUFFICIENT_DTW_SUPPORT", "coverage": coverage}
        correction = aggregate_neighbor_residuals(
            [value[2] for value in selected], [value[0] for value in selected]
        )
        historical = self.panel.residual.loc[: source_limit - pd.Timedelta(minutes=RAW_STEP_MINUTES)]
        lower = historical.quantile(0.01).to_numpy(dtype=float)
        upper = historical.quantile(0.99).to_numpy(dtype=float)
        correction = np.clip(correction, lower, upper)
        daily = pd.DataFrame(correction[-24:], index=target_index, columns=TARGET_LAYERS)
        selected_end_ns = [value[1] for value in selected]
        embargo_ns = int(trajectory_start.as_unit("ns").value) - max(selected_end_ns)
        return daily, {
            "support": bool(np.isfinite(daily.to_numpy()).all()),
            "coverage": coverage,
            "neighbor_count": len(selected),
            "mean_distance": float(np.mean([value[0] for value in selected])),
            "maximum_distance": float(max(value[0] for value in selected)),
            "prediction_day_start_utc_ns": int(prediction_day_start.as_unit("ns").value),
            "trajectory_start_utc_ns": int(trajectory_start.as_unit("ns").value),
            "latest_selected_source_end_utc_ns": int(max(selected_end_ns)),
            "minimum_selected_embargo_seconds": float(embargo_ns / 1_000_000_000),
        }

    def __call__(
        self,
        window: WindowSpec | str,
        cell: CellSpec,
        deadline_monotonic: float,
    ) -> pd.DataFrame:
        spec, anchor = self._anchor_and_window(window)
        start = pd.Timestamp(spec.start_kst).tz_convert(KST)
        end = pd.Timestamp(spec.end_exclusive_kst).tz_convert(KST)
        corrections: list[pd.DataFrame] = []
        diagnostic_by_day: dict[str, Mapping[str, Any]] = {}
        for day_start, _ in _daily_bounds(start, end):
            daily, diagnostics = self._daily_correction(
                day_start, cell, deadline_monotonic
            )
            corrections.append(daily)
            diagnostic_by_day[day_start.date().isoformat()] = diagnostics
        hourly = pd.concat(corrections).sort_index() if corrections else pd.DataFrame(columns=TARGET_LAYERS)

        if anchor is None:
            target_times = self.panel.index[
                (self.panel.index >= start.tz_convert("UTC"))
                & (self.panel.index < end.tz_convert("UTC"))
            ]
            rows: list[pd.DataFrame] = []
            for layer in TARGET_LAYERS:
                part = pd.DataFrame(
                    {
                        "station": self.panel.station,
                        "layer": layer,
                        "time": target_times,
                        "truth": self.panel.temp.loc[target_times, layer].to_numpy(dtype=float),
                        "anchor": self.panel.baseline.loc[target_times, layer].to_numpy(dtype=float),
                    }
                )
                rows.append(part)
            target = pd.concat(rows, ignore_index=True)
            target = target.loc[np.isfinite(target["truth"]) & np.isfinite(target["anchor"])].copy()
        else:
            target = anchor.copy()
            target["time"] = utc_ns_to_datetime(normalize_utc_ns(target["time"]))
            if "station" not in target:
                target["station"] = self.panel.station
        correction_ten = hourly.reindex(
            hourly.index.union(self.panel.index), copy=False
        ).interpolate(method="time", limit_direction="both").reindex(self.panel.index)
        target["_time_ns"] = normalize_utc_ns(target["time"])
        target["candidate"] = target["anchor"].astype(float)
        target["support"] = False
        target["dtw_distance"] = np.nan
        target["public_missingness"] = np.nan
        target["trajectory_coverage"] = np.nan
        target["no_op_reason"] = "WEAK_SUPPORT"
        panel_ns = self.panel.index.as_unit("ns").asi8
        row_lookup = {int(value): index for index, value in enumerate(panel_ns)}
        for row_index, row in target.iterrows():
            ns = int(row["_time_ns"])
            position = row_lookup.get(ns)
            layer = int(row["layer"])
            if position is None or layer not in TARGET_LAYERS:
                continue
            correction = float(correction_ten.iloc[position][layer])
            baseline = float(self.panel.baseline.iloc[position][layer])
            day = self.panel.index[position].tz_convert(KST).date().isoformat()
            diagnostics = diagnostic_by_day.get(day, {})
            coverage = float(diagnostics.get("coverage", math.nan))
            target.at[row_index, "trajectory_coverage"] = coverage
            target.at[row_index, "public_missingness"] = (
                1.0 - coverage if math.isfinite(coverage) else math.nan
            )
            l1 = float(self.panel.temp.iloc[position][1])
            l5 = float(self.panel.temp.iloc[position][5])
            if not math.isfinite(l1) or not math.isfinite(l5):
                target.at[row_index, "candidate"] = target.at[row_index, "anchor"]
                target.at[row_index, "support"] = False
                target.at[row_index, "no_op_reason"] = "MISSING_L1_OR_L5_ENDPOINT"
            elif diagnostics.get("support") and math.isfinite(correction) and math.isfinite(baseline):
                target.at[row_index, "candidate"] = baseline + correction
                target.at[row_index, "support"] = True
                target.at[row_index, "dtw_distance"] = diagnostics.get("mean_distance", math.nan)
                target.at[row_index, "no_op_reason"] = ""
        for ns, group_index in target.groupby("_time_ns", sort=False).groups.items():
            group = target.loc[group_index].sort_values("layer")
            if tuple(group["layer"].astype(int)) != TARGET_LAYERS or not bool(group["support"].all()):
                continue
            position = row_lookup.get(int(ns))
            if position is None:
                continue
            l1 = float(self.panel.temp.iloc[position][1])
            l5 = float(self.panel.temp.iloc[position][5])
            projected, projected_support, projection_reason = endpoint_safe_candidate(
                l1,
                group["anchor"].to_numpy(dtype=float),
                group["candidate"].to_numpy(dtype=float),
                l5,
                group["support"].to_numpy(dtype=bool),
            )
            target.loc[group.index, "candidate"] = projected
            target.loc[group.index, "support"] = projected_support
            if projection_reason != "PROJECTED":
                target.loc[group.index, "no_op_reason"] = projection_reason
        target["time"] = utc_ns_to_datetime(target["_time_ns"].to_numpy())
        target = target.drop(columns=["_time_ns"])
        if not np.isfinite(target[["truth", "anchor", "candidate"]].to_numpy(dtype=float)).all():
            raise ValueError("materialized score frame contains non-finite values")
        if anchor is not None:
            trusted = self.exact_anchor if spec.window_id == EXACT_WINDOW.window_id else anchor
            target, certificate = verify_candidate_against_truth(target, trusted)
            target.attrs["truth_verification"] = certificate
            if spec.window_id == EXACT_WINDOW.window_id:
                target.attrs["anchor_truth_verification"] = self.exact_truth_certificate
            else:
                target.attrs["anchor_truth_verification"] = self.p100_truth_certificate
        if isinstance(window, str):
            target["fold"] = window
        return target.reset_index(drop=True)


MaterializeCallback = Callable[[WindowSpec | str, CellSpec, float], pd.DataFrame]


def execute_zero_fit_protocol(
    materialize: MaterializeCallback,
    *,
    deadline_monotonic: float,
    on_slot: Callable[[int, str, Mapping[str, Any]], None],
    on_exact_research_go: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute the sealed 18 -> 1 -> conditional-3 graph in memory.

    The callback must construct predictions from historical public trajectories.
    It is deliberately injected so the orchestration and accounting can be
    exhaustively tested without opening any project data.
    """

    inner_records: list[dict[str, Any]] = []
    slot = 1
    for window in INNER_WINDOWS:
        for cell in CELLS:
            check_deadline(deadline_monotonic)
            on_slot(slot, "RESERVED", {"stage": "INNER", "window_id": window.window_id, "cell_id": cell.cell_id})
            try:
                frame = materialize(window, cell, deadline_monotonic)
                metrics = score_prediction_frame(frame)
            except BaseException:
                on_slot(slot, "FAILED", {"stage": "INNER", "window_id": window.window_id, "cell_id": cell.cell_id})
                raise
            record = {
                "window_id": window.window_id,
                "cell_id": cell.cell_id,
                "layer_equal_rmse_c": metrics["candidate_layer_equal_rmse_c"],
                "worst_layer_rmse_c": max(value["candidate_rmse_c"] for value in metrics["by_layer"].values()),
            }
            inner_records.append(record)
            on_slot(slot, "COMPLETED", {"stage": "INNER", "window_id": window.window_id, "cell_id": cell.cell_id, "aggregate": record})
            slot += 1
    selected = select_inner_cell(inner_records)
    check_deadline(deadline_monotonic)
    on_slot(slot, "RESERVED", {"stage": "EXACT", "window_id": EXACT_WINDOW.window_id, "cell_id": selected.cell_id})
    try:
        exact_frame = materialize(EXACT_WINDOW, selected, deadline_monotonic)
        exact_contract = exact_time_contract(exact_frame["time"])
        exact_metrics = score_prediction_frame(exact_frame)
        fallback = exact_no_op_integrity(exact_frame)
        truth_certificate = exact_frame.attrs.get("truth_verification", {})
        exact_metrics.update(
            {
                "all_values_finite": fallback["all_score_values_finite"],
                "weak_support_exact_anchor_fallback": fallback[
                    "weak_support_candidate_equals_anchor"
                ],
                "exact_key_alignment": bool(
                    truth_certificate.get("candidate_key_equal_verified_truth", False)
                ),
                "truth_verification": dict(truth_certificate),
                "fallback_integrity": fallback,
                "time_contract": exact_contract,
            }
        )
    except BaseException:
        on_slot(slot, "FAILED", {"stage": "EXACT", "window_id": EXACT_WINDOW.window_id, "cell_id": selected.cell_id})
        raise
    exact_go = exact_research_gate(exact_metrics)
    on_slot(slot, "COMPLETED", {"stage": "EXACT", "window_id": EXACT_WINDOW.window_id, "cell_id": selected.cell_id, "gate": "RESEARCH_GO" if exact_go else "RESEARCH_NO_GO"})
    slot += 1

    p100_frames: list[pd.DataFrame] = []
    if exact_go:
        if on_exact_research_go is None:
            raise RuntimeError("exact RESEARCH_GO has no registered lazy p100 loader")
        on_exact_research_go()
        for fold in P100_FOLDS:
            check_deadline(deadline_monotonic)
            on_slot(slot, "RESERVED", {"stage": "P100", "window_id": fold, "cell_id": selected.cell_id})
            try:
                frame = materialize(fold, selected, deadline_monotonic)
                if "fold" not in frame:
                    frame = frame.assign(fold=fold)
                p100_frames.append(frame)
            except BaseException:
                on_slot(slot, "FAILED", {"stage": "P100", "window_id": fold, "cell_id": selected.cell_id})
                raise
            on_slot(slot, "COMPLETED", {"stage": "P100", "window_id": fold, "cell_id": selected.cell_id, "rows": int(len(frame))})
            slot += 1
        p100 = pd.concat(p100_frames, ignore_index=True)
        p100_metrics = derive_p100_metrics(p100)
        p100_go = p100_research_gate(p100_metrics)
    else:
        for fold in P100_FOLDS:
            on_slot(slot, "SKIPPED_GATE", {"stage": "P100", "window_id": fold, "cell_id": selected.cell_id, "reason": "EXACT_RESEARCH_NO_GO"})
            slot += 1
        p100_metrics = None
        p100_go = False
    if slot != 23:
        raise RuntimeError("deterministic materialization graph did not consume exactly 22 slots")
    return {
        "status": "COMPLETE_LOCAL_RESEARCH_ONLY",
        "selected_cell": selected.__dict__,
        "inner_selection_records": inner_records,
        "exact": exact_metrics,
        "exact_gate": "RESEARCH_GO" if exact_go else "RESEARCH_NO_GO",
        "p100": p100_metrics,
        "p100_gate": "RESEARCH_GO" if p100_go else ("NOT_RUN_EXACT_GATE" if not exact_go else "RESEARCH_NO_GO"),
        "overall_gate": "RESEARCH_GO" if exact_go and p100_go else "RESEARCH_NO_GO",
        "physical_fit_calls": 0,
        "materialization_slots_total": 22,
        "result_driven_reruns": 0,
    }


