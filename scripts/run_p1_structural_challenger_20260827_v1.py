"""One-shot P1 structural challenger on the frozen chronological OOF surface.

This runner deliberately implements a small, label-free proposal bank rather
than another row model.  It fills only internal holes of seven through forty-
eight rows when the two immediate anchor endpoints are positive in all three
registered Round-B seeds and the complete adjacent positive-run hull lies in
the official offset/drift duration envelope (48--519 rows).

The proposal rule is immutable in this file.  There are no threshold or model
arguments.  Labels are introduced only after the candidate has been frozen in
memory, for aggregate evaluation.  No row-level predictions are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_structural_challenger_20260827_v1"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "structural_challenger_20260827_v1" / "p1"

INCUMBENT_DIR = PROJECT_ROOT / "artifacts" / "p1_round_b_nonspike_long_event_residual_v1r6"
INCUMBENT_PREDICTIONS = INCUMBENT_DIR / "predictions.parquet"
INCUMBENT_METRICS = INCUMBENT_DIR / "metrics.json"
INCUMBENT_RESULT = INCUMBENT_DIR / "result.json"
INCUMBENT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r6.json"
)
TRUTH_OOF = PROJECT_ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
P1_CONFIG = PROJECT_ROOT / "configs" / "p1.toml"
P1_PREREGISTERED_DESIGN = (
    PROJECT_ROOT
    / "artifacts"
    / "structural_challenger_20260827_v1"
    / "p1_preregistered_design.json"
)
BOUNDARY_RECEIPT = (
    PROJECT_ROOT / "artifacts" / "structural_challenger_20260827_v1" / "boundary_receipt.json"
)

EXPECTED_P1_DESIGN_SHA256 = "fef64965e6d8672497b5cbfd2a6251c5e93f26aad0c76b5887f8b4700607ecc3"
EXPECTED_SOURCE_SHA256 = "20b656b0cbd524ad9da0bae8ecb6e0bacfc006e05810b37e83f29a5fa8e65cd2"
EXPECTED_TRUTH_OOF_SHA256 = "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a"
EXPECTED_INCUMBENT_PREDICTIONS_SHA256 = (
    "953415097b2f43421cf40ffe98100f305e3fbd2ba1215656acc4d13bf2b8ec93"
)
EXPECTED_ROWS = 421_032
EXPECTED_FOLD_ROWS = {"2025_q2": 133_170, "2025_q3": 176_738, "2025_q4": 111_124}
EXPECTED_INCUMBENT_COUNTS = {"tp": 12_718, "fp": 644, "fn": 3_337, "tn": 404_333}
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
KEY_COLUMNS = ("station", "year", "layer", "time", "fold")
SEED_COLUMNS = (
    "round_b__seed_20260813",
    "round_b__seed_20260829",
    "round_b__seed_20260847",
)
ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")

# Frozen before the one-shot result.  Gaps <= 6 are intentionally excluded
# because they already occur in the prior postprocess search surface.
MIN_GAP_ROWS = 7
MAX_GAP_ROWS = 48
MIN_LONG_HULL_ROWS = 48
MAX_LONG_HULL_ROWS = 519
DRIFT_MIN_HULL_ROWS = 54
CADENCE_MINUTES = 10

MIN_SUPPORT_POSITIVE_ROWS = 100
MIN_SUPPORT_EVENTS = 5
MIN_SUPPORT_KST_DAYS = 10
PROMOTION_F1_DELTA = 0.0015
PROMOTION_MIN_IMPROVED_FOLDS = 2
MAX_ADEQUATE_STATION_LAYER_F1_DROP = 0.015
MAX_SUPPORTED_NON_SPIKE_TYPE_RECALL_DROP = 0.02


@dataclass(frozen=True)
class GapProposal:
    """In-memory half-open proposal; coordinates are never exported."""

    fold: str
    station: str
    layer: int
    gap_start: int
    gap_stop: int
    gap_rows: int
    hull_rows: int
    eligible_types: tuple[str, ...]


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_payload(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ) + "\n"


def _write_text_atomic(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, _json_payload(value))
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(_json_payload(value)):
        raise RuntimeError(f"JSON reload mismatch: {path.name}")


def _checked_binary(values: Sequence[int], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int8)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    return array


def _binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    y = _checked_binary(truth, name="truth")
    p = _checked_binary(prediction, name="prediction")
    if y.shape != p.shape:
        raise ValueError("truth and prediction shapes differ")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = float(2 * tp / denominator) if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _canonical_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "station": frame["station"].astype("string"),
            "year": pd.to_numeric(frame["year"], errors="raise").astype(np.int64),
            "layer": pd.to_numeric(frame["layer"], errors="raise").astype(np.int64),
            "time": pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed"),
            "fold": frame["fold"].astype("string"),
        }
    )
    if result.isna().any().any():
        raise ValueError("ordered key surface contains nulls")
    return result


def _key_digest(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized["time"] = normalized["time"].astype("int64")
    values = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="<u8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _assert_ordered_key_equality(left: pd.DataFrame, right: pd.DataFrame) -> str:
    if len(left) != len(right):
        raise RuntimeError("incumbent and truth row counts differ")
    for column in KEY_COLUMNS:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise RuntimeError(f"ordered incumbent/truth key mismatch: {column}")
    if left.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("ordered validation keys are not unique")
    left_digest = _key_digest(left)
    right_digest = _key_digest(right)
    if left_digest != right_digest:
        raise RuntimeError("ordered validation key digest mismatch")
    return left_digest


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    vector = np.asarray(values, dtype=bool)
    if not vector.any():
        return []
    changes = np.diff(np.pad(vector.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _internal_zero_runs(values: np.ndarray) -> list[tuple[int, int]]:
    zero_runs = _true_runs(np.asarray(values, dtype=np.int8) == 0)
    return [
        (start, stop)
        for start, stop in zero_runs
        if start > 0 and stop < len(values) and values[start - 1] == 1 and values[stop] == 1
    ]


def _proposal_bank(
    metadata: pd.DataFrame,
    anchor_prediction: Sequence[int],
    seed_predictions: np.ndarray,
) -> tuple[np.ndarray, tuple[GapProposal, ...], dict[str, Any]]:
    """Freeze a target-free, non-destructive duration-aware gap proposal bank."""

    required = {"station", "layer", "time", "fold"}
    if not required.issubset(metadata.columns):
        raise KeyError(f"proposal metadata lacks columns: {sorted(required - set(metadata.columns))}")
    forbidden = {"label", "anomaly_type"}.intersection(metadata.columns)
    if forbidden:
        raise ValueError("proposal generator received truth columns")
    anchor = _checked_binary(anchor_prediction, name="anchor_prediction")
    seeds = np.asarray(seed_predictions, dtype=np.int8)
    if seeds.shape != (len(metadata), 3) or not np.isin(seeds, [0, 1]).all():
        raise ValueError("three registered seed predictions are required")
    candidate = anchor.copy()
    parsed_time = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
    work = metadata.loc[:, ["fold", "station", "layer"]].copy()
    work["__time"] = parsed_time
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work.sort_values(
        ["fold", "station", "layer", "__time", "__position"],
        kind="mergesort",
        inplace=True,
    )
    proposals: list[GapProposal] = []
    rejected = Counter()
    considered = 0
    cadence_ns = int(pd.Timedelta(minutes=CADENCE_MINUTES).value)
    for (fold, station, layer), group in work.groupby(
        ["fold", "station", "layer"], sort=False, observed=True
    ):
        ordered = group["__position"].to_numpy(dtype=np.int64)
        # Pandas 3 may retain microsecond-backed datetime storage.  Request
        # nanoseconds explicitly so the cadence constant has one fixed unit.
        times = group["__time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        breaks = np.flatnonzero(np.diff(times) != cadence_ns) + 1
        for segment in np.split(ordered, breaks):
            if len(segment) < 3:
                continue
            local_anchor = anchor[segment]
            local_consensus = np.all(seeds[segment] == 1, axis=1)
            for gap_start, gap_stop in _internal_zero_runs(local_anchor):
                considered += 1
                gap_rows = gap_stop - gap_start
                if gap_rows < MIN_GAP_ROWS:
                    rejected["previous_close_gap_surface_le_6"] += 1
                    continue
                if gap_rows > MAX_GAP_ROWS:
                    rejected["gap_above_frozen_maximum"] += 1
                    continue
                left_start = gap_start - 1
                while left_start > 0 and local_anchor[left_start - 1] == 1:
                    left_start -= 1
                right_stop = gap_stop + 1
                while right_stop < len(local_anchor) and local_anchor[right_stop] == 1:
                    right_stop += 1
                hull_rows = right_stop - left_start
                if not MIN_LONG_HULL_ROWS <= hull_rows <= MAX_LONG_HULL_ROWS:
                    rejected["outside_offset_drift_duration_hull"] += 1
                    continue
                if not (local_consensus[gap_start - 1] and local_consensus[gap_stop]):
                    rejected["endpoint_not_unanimous_across_three_seeds"] += 1
                    continue
                eligible_types = ("offset",)
                if hull_rows >= DRIFT_MIN_HULL_ROWS:
                    eligible_types = ("offset", "drift")
                absolute_gap = segment[gap_start:gap_stop]
                if np.any(anchor[absolute_gap] != 0):
                    raise AssertionError("proposal contains a non-negative anchor row")
                candidate[absolute_gap] = 1
                proposals.append(
                    GapProposal(
                        fold=str(fold),
                        station=str(station),
                        layer=int(layer),
                        gap_start=int(absolute_gap[0]),
                        gap_stop=int(absolute_gap[-1] + 1),
                        gap_rows=int(gap_rows),
                        hull_rows=int(hull_rows),
                        eligible_types=eligible_types,
                    )
                )
    if any(proposal.gap_rows <= 6 for proposal in proposals):
        raise AssertionError("new proposal overlaps the prior close-gap<=6 surface")
    if np.any(candidate < anchor):
        raise AssertionError("the structural challenger removed an incumbent positive")
    audit = {
        "internal_anchor_holes_considered": int(considered),
        "accepted_proposals": len(proposals),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "prior_close_gap_surface_overlap_count": int(
            sum(proposal.gap_rows <= 6 for proposal in proposals)
        ),
        "removed_incumbent_positive_rows": int(np.sum((anchor == 1) & (candidate == 0))),
        "added_rows": int(np.sum((anchor == 0) & (candidate == 1))),
    }
    return candidate, tuple(proposals), audit


def _event_ids(metadata: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    work = metadata.loc[:, ["fold", "station", "layer", "time"]].copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__truth"] = truth
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(
        ["fold", "station", "layer", "__time", "__position"], kind="mergesort", inplace=True
    )
    grouped = work.groupby(["fold", "station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(CADENCE_MINUTES * 60)
    prior_positive = grouped["__truth"].shift(1).fillna(0).eq(1)
    starts = work["__truth"].eq(1) & (~contiguous | ~prior_positive)
    work["__event"] = starts.cumsum().where(work["__truth"].eq(1), -1).astype(np.int64)
    restored = work.sort_values("__position", kind="mergesort")
    return restored["__event"].to_numpy(dtype=np.int64)


def _support(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    event_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int | bool]:
    positive = mask & (truth == 1)
    event_count = int(len(np.unique(event_ids[positive & (event_ids >= 0)])))
    times = pd.to_datetime(metadata.loc[positive, "time"], errors="raise", utc=True, format="mixed")
    if positive.any():
        day_frame = metadata.loc[positive, ["station", "layer"]].reset_index(drop=True).copy()
        day_frame["kst_day"] = times.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
        kst_days = int(day_frame.drop_duplicates().shape[0])
    else:
        kst_days = 0
    positive_rows = int(positive.sum())
    adequate = (
        positive_rows >= MIN_SUPPORT_POSITIVE_ROWS
        and event_count >= MIN_SUPPORT_EVENTS
        and kst_days >= MIN_SUPPORT_KST_DAYS
    )
    return {
        "positive_rows": positive_rows,
        "positive_events": event_count,
        "positive_station_layer_kst_days": kst_days,
        "adequate_support": adequate,
    }


def _slice_metric(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    event_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    incumbent_metrics = _binary_metrics(truth[mask], anchor[mask])
    candidate_metrics = _binary_metrics(truth[mask], candidate[mask])
    support = _support(metadata, truth, event_ids, mask)
    return {
        "rows": int(mask.sum()),
        **support,
        "incumbent": incumbent_metrics,
        "candidate": candidate_metrics,
        "f1_delta": float(candidate_metrics["f1"] - incumbent_metrics["f1"]),
        "precision_delta": float(candidate_metrics["precision"] - incumbent_metrics["precision"]),
        "recall_delta": float(candidate_metrics["recall"] - incumbent_metrics["recall"]),
        "added_rows": int(np.sum(mask & (anchor == 0) & (candidate == 1))),
    }


def _group_metrics(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    event_ids: np.ndarray,
    columns: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped = metadata.groupby(list(columns), sort=True, observed=True).indices
    for raw_key, positions in grouped.items():
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = "|".join(str(value) for value in values)
        mask = np.zeros(len(metadata), dtype=bool)
        mask[np.asarray(positions, dtype=np.int64)] = True
        result[key] = _slice_metric(metadata, truth, anchor, candidate, event_ids, mask)
    return result


def _type_masks(anomaly_type: pd.Series, truth: np.ndarray) -> dict[str, np.ndarray]:
    tokens = anomaly_type.astype("string").fillna("").map(
        lambda value: {item.strip() for item in str(value).split("+") if item.strip()}
    )
    return {
        anomaly: np.asarray(
            [(truth[index] == 1 and anomaly in value) for index, value in enumerate(tokens)],
            dtype=bool,
        )
        for anomaly in ANOMALY_TYPES
    }


def _type_metrics(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    anomaly_type: pd.Series,
    event_ids: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for anomaly, mask in _type_masks(anomaly_type, truth).items():
        support = _support(metadata, truth, event_ids, mask)
        rows = int(mask.sum())
        incumbent_recall = float(anchor[mask].mean()) if rows else None
        candidate_recall = float(candidate[mask].mean()) if rows else None
        recall_delta = (
            None
            if incumbent_recall is None or candidate_recall is None
            else float(candidate_recall - incumbent_recall)
        )
        result[anomaly] = {
            **support,
            "incumbent_recall": incumbent_recall,
            "candidate_recall": candidate_recall,
            "recall_delta": recall_delta,
            "veto_applies": bool(anomaly != "spike" and support["adequate_support"]),
        }
    return result


def _proposal_aggregates(
    proposals: Sequence[GapProposal],
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    per_fold = Counter(proposal.fold for proposal in proposals)
    per_station = Counter(proposal.station for proposal in proposals)
    per_layer = Counter(str(proposal.layer) for proposal in proposals)
    per_eligible_type = Counter(
        anomaly for proposal in proposals for anomaly in proposal.eligible_types
    )
    gap_lengths = np.asarray([proposal.gap_rows for proposal in proposals], dtype=np.int64)
    hull_lengths = np.asarray([proposal.hull_rows for proposal in proposals], dtype=np.int64)
    added = (anchor == 0) & (candidate == 1)
    added_tp = int(np.sum(added & (truth == 1)))
    added_fp = int(np.sum(added & (truth == 0)))
    added_total = added_tp + added_fp
    return {
        "accepted_proposals": len(proposals),
        "by_fold": dict(sorted(per_fold.items())),
        "by_station": dict(sorted(per_station.items())),
        "by_layer": dict(sorted(per_layer.items(), key=lambda item: int(item[0]))),
        "eligible_type_memberships": dict(sorted(per_eligible_type.items())),
        "gap_rows_summary": {
            "minimum": int(gap_lengths.min()) if len(gap_lengths) else None,
            "median": float(np.median(gap_lengths)) if len(gap_lengths) else None,
            "maximum": int(gap_lengths.max()) if len(gap_lengths) else None,
        },
        "hull_rows_summary": {
            "minimum": int(hull_lengths.min()) if len(hull_lengths) else None,
            "median": float(np.median(hull_lengths)) if len(hull_lengths) else None,
            "maximum": int(hull_lengths.max()) if len(hull_lengths) else None,
        },
        "added_rows": added_total,
        "added_true_positive_rows": added_tp,
        "added_false_positive_rows": added_fp,
        "added_row_precision": float(added_tp / added_total) if added_total else None,
    }


def _validate_split_contract(metadata: pd.DataFrame) -> dict[str, Any]:
    config = tomllib.loads(P1_CONFIG.read_text(encoding="utf-8"))
    validation = config["validation"]
    if int(validation["purge_days"]) != 7:
        raise RuntimeError("P1 purge contract is no longer seven days")
    registered = {fold["name"]: fold for fold in validation["folds"]}
    if tuple(registered) != FOLD_ORDER:
        raise RuntimeError("P1 chronological fold order changed")
    parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
    folds: dict[str, Any] = {}
    for name in FOLD_ORDER:
        fold = registered[name]
        train_end = pd.Timestamp(fold["train_end"]).tz_convert("UTC")
        val_start = pd.Timestamp(fold["val_start"]).tz_convert("UTC")
        val_end = pd.Timestamp(fold["val_end"]).tz_convert("UTC")
        mask = metadata["fold"].astype(str).eq(name).to_numpy()
        if int(mask.sum()) != EXPECTED_FOLD_ROWS[name]:
            raise RuntimeError(f"fold row count drift: {name}")
        below_nominal_start = int((parsed[mask] < val_start).sum())
        at_or_above_nominal_end = int((parsed[mask] >= val_end).sum())
        purge = val_start - train_end
        if purge < pd.Timedelta(days=7):
            raise RuntimeError(f"purge shorter than seven days: {name}")
        folds[name] = {
            "rows": int(mask.sum()),
            "train_end_inclusive": str(train_end),
            "validation_start_inclusive": str(val_start),
            "validation_end_exclusive": str(val_end),
            "train_to_validation_gap_seconds": float(purge.total_seconds()),
            "minimum_required_purge_days": 7,
            "purge_passed": True,
            "frozen_membership_is_authoritative": True,
            "rows_below_nominal_start": below_nominal_start,
            "rows_at_or_above_nominal_end": at_or_above_nominal_end,
            "nominal_time_spill_rows": below_nominal_start + at_or_above_nominal_end,
        }
    return {
        "folds": folds,
        "same_registered_row_universe": True,
        "same_frozen_fold_labels": True,
        "purge_days": 7,
        "membership_policy": (
            "Exact frozen ordered keys and incumbent fold labels are authoritative. "
            "Nominal timestamp spill is audited but never dropped or reassigned."
        ),
        "total_nominal_time_spill_rows": int(
            sum(fold["nominal_time_spill_rows"] for fold in folds.values())
        ),
    }


def _prior_family_audit() -> dict[str, Any]:
    paths = {
        "typed_duration_result": PROJECT_ROOT / "artifacts/p1_typed_duration_semimarkov_v2/result.json",
        "typed_factorial_result": PROJECT_ROOT / "artifacts/p1_typed_factorial_semimarkov_v1/result.json",
        "connected_residual_metrics": INCUMBENT_METRICS,
        "long_event_terminal_no_go": PROJECT_ROOT
        / "reports/p1_long_event_segment_proposal_rescore_v8_terminal_no_go_20260826.json",
        "deep_research_report": PROJECT_ROOT
        / "reports/deep_research_breakthrough_20260826/report-source.md",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"prior-family audit input is absent: {path}")
    duration = json.loads(paths["typed_duration_result"].read_text(encoding="utf-8"))
    factorial = json.loads(paths["typed_factorial_result"].read_text(encoding="utf-8"))
    residual = json.loads(paths["connected_residual_metrics"].read_text(encoding="utf-8"))
    terminal = json.loads(paths["long_event_terminal_no_go"].read_text(encoding="utf-8"))
    return {
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in paths.values()
        },
        "typed_duration_semimarkov": {
            "decision": duration["decision"],
            "micro_f1_delta": duration["aggregate"]["micro_f1_delta"],
            "distinction": "same-unary hard per-type decoding; may alter the entire chain",
        },
        "typed_factorial_semimarkov": {
            "decision": factorial["decision"],
            "inner_blocks_executed": factorial["inner_blocks_executed"],
            "distinction": "factorial overlap/order grammar failed identifiability precheck",
        },
        "connected_long_event_residual": {
            "decision": "NO_GO_LOCAL_GATE",
            "rescued_rows": residual["structural"]["rescued_rows"],
            "micro_f1_delta": residual["pooled"]["f1_delta"],
            "distinction": "learned residual, at most 18-row connected growth; produced no rescue",
        },
        "change_point_segment_rescore": {
            "decision": terminal["decision"],
            "physical_fits": terminal["independent_zero_state_check"]["physical_fits"],
            "outer_scores": terminal["independent_zero_state_check"]["outer_scores"],
            "distinction": "raw-channel change-point bank plus learned segment scorer; science never ran",
        },
        "current_nonduplicate_scope": {
            "model_fits": 0,
            "uses_raw_measurement_values": False,
            "uses_truth_for_proposal_generation": False,
            "non_destructive": True,
            "excluded_prior_close_gap_rows": [0, 6],
            "frozen_gap_rows": [MIN_GAP_ROWS, MAX_GAP_ROWS],
            "frozen_offset_drift_hull_rows": [MIN_LONG_HULL_ROWS, MAX_LONG_HULL_ROWS],
            "endpoint_rule": "immediate bounding anchor rows must be positive in all three registered seeds",
        },
    }


def _self_check() -> None:
    def case(gap: int, *, discontinuity: bool = False) -> int:
        left = 21
        right = 20
        rows = left + gap + right
        anchor = np.r_[np.ones(left, dtype=np.int8), np.zeros(gap, dtype=np.int8), np.ones(right, dtype=np.int8)]
        seeds = np.repeat(anchor[:, None], 3, axis=1)
        time = pd.date_range("2025-04-01", periods=rows, freq="10min", tz="UTC")
        if discontinuity:
            time = time.to_series(index=np.arange(rows))
            time.iloc[left:] += pd.Timedelta(minutes=10)
            time = pd.DatetimeIndex(time)
        metadata = pd.DataFrame(
            {
                "fold": "2025_q2",
                "station": "SYNTHETIC",
                "layer": 1,
                "time": time,
            }
        )
        candidate, proposals, _audit = _proposal_bank(metadata, anchor, seeds)
        if np.any(candidate < anchor):
            raise AssertionError("synthetic non-destructive check failed")
        return len(proposals)

    if case(6) != 0:
        raise AssertionError("prior close-gap<=6 surface was not excluded")
    if case(7) != 1:
        raise AssertionError("frozen minimum gap was not admitted")
    if case(49) != 0:
        raise AssertionError("gap above frozen maximum was admitted")
    if case(7, discontinuity=True) != 0:
        raise AssertionError("proposal bridged a non-10-minute time discontinuity")


def _load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = (
        INCUMBENT_PREDICTIONS,
        INCUMBENT_METRICS,
        INCUMBENT_RESULT,
        INCUMBENT_CONFIG,
        TRUTH_OOF,
        P1_CONFIG,
        P1_PREREGISTERED_DESIGN,
        BOUNDARY_RECEIPT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required immutable input is absent: {path}")
    train = data_dir.expanduser().resolve(strict=True) / "train.csv"
    if not train.is_file():
        raise FileNotFoundError("P1 train.csv is absent from --data-dir")
    hashes = {
        "train.csv": _sha256(train),
        str(INCUMBENT_PREDICTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            INCUMBENT_PREDICTIONS
        ),
        str(INCUMBENT_METRICS.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            INCUMBENT_METRICS
        ),
        str(INCUMBENT_RESULT.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            INCUMBENT_RESULT
        ),
        str(INCUMBENT_CONFIG.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            INCUMBENT_CONFIG
        ),
        str(TRUTH_OOF.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(TRUTH_OOF),
        str(P1_CONFIG.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(P1_CONFIG),
        str(P1_PREREGISTERED_DESIGN.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            P1_PREREGISTERED_DESIGN
        ),
        str(BOUNDARY_RECEIPT.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
            BOUNDARY_RECEIPT
        ),
    }
    if hashes["train.csv"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("P1 source train hash drift")
    if hashes[str(TRUTH_OOF.relative_to(PROJECT_ROOT)).replace("\\", "/")] != EXPECTED_TRUTH_OOF_SHA256:
        raise RuntimeError("frozen truth OOF hash drift")
    if (
        hashes[str(INCUMBENT_PREDICTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/")]
        != EXPECTED_INCUMBENT_PREDICTIONS_SHA256
    ):
        raise RuntimeError("frozen incumbent prediction hash drift")
    if (
        hashes[str(P1_PREREGISTERED_DESIGN.relative_to(PROJECT_ROOT)).replace("\\", "/")]
        != EXPECTED_P1_DESIGN_SHA256
    ):
        raise RuntimeError("P1 preregistered design hash drift")
    incumbent_columns = [
        "station",
        "year",
        "layer",
        "time",
        "row_position",
        "fold",
        "round_b_prediction",
        *SEED_COLUMNS,
    ]
    truth_columns = ["station", "year", "layer", "time", "label", "anomaly_type", "fold"]
    incumbent = pd.read_parquet(INCUMBENT_PREDICTIONS, columns=incumbent_columns)
    truth = pd.read_parquet(TRUTH_OOF, columns=truth_columns)
    return incumbent, truth, hashes


def _build_report(result: dict[str, Any]) -> str:
    pooled = result["metrics"]["pooled"]
    proposal = result["proposal_evaluation"]
    lines = [
        "# P1 Structural Challenger 2026-08-27 v1",
        "",
        f"결론: **{result['decision']}** ({result['short_decision']})",
        "",
        "## 핵심 결과",
        "",
        f"- 동일 chronological OOF {pooled['rows']:,}행에서 incumbent F1은 {pooled['incumbent']['f1']:.6f}, candidate F1은 {pooled['candidate']['f1']:.6f}, ΔF1은 {pooled['f1_delta']:+.6f}입니다.",
        f"- 고정 proposal {proposal['accepted_proposals']:,}개가 {proposal['added_rows']:,}행을 추가했고, 추가행 precision은 {proposal['added_row_precision'] if proposal['added_row_precision'] is not None else 'N/A'}입니다.",
        f"- 개선 fold 수는 {result['decision_evidence']['improved_folds_of_3']}/3입니다.",
        f"- incumbent 양성 1→0 변경은 {result['proposal_generation_audit']['removed_incumbent_positive_rows']}행, 기존 close-gap≤6와 겹친 proposal은 {result['proposal_generation_audit']['prior_close_gap_surface_overlap_count']}개입니다.",
        "",
        "## 고정 가설",
        "",
        "기존 morphology가 다룬 0–6행 gap은 제외하고, 7–48행 내부 hole만 봅니다. immediate 양끝이 Round-B 3개 seed 모두에서 양성이고, 양옆 양성 run을 포함한 hull이 공식 offset/drift duration 48–519행 안일 때만 gap을 채웁니다. 원 양성은 절대 제거하지 않으며 원시 측정값·label·anomaly_type은 proposal 생성에 쓰지 않습니다.",
        "",
        "## Preregistered gate",
        "",
    ]
    for name, passed in result["decision_evidence"]["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Fold별 ΔF1",
            "",
            "| Fold | Rows | Incumbent F1 | Candidate F1 | ΔF1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in FOLD_ORDER:
        metric = result["metrics"]["folds"][fold]
        lines.append(
            f"| {fold} | {metric['rows']:,} | {metric['incumbent']['f1']:.6f} | {metric['candidate']['f1']:.6f} | {metric['f1_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 유형별 recall",
            "",
            "| Type | Positive rows | Support | Incumbent | Candidate | Δ | Veto |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for anomaly in ANOMALY_TYPES:
        metric = result["metrics"]["types"][anomaly]
        inc = "N/A" if metric["incumbent_recall"] is None else f"{metric['incumbent_recall']:.6f}"
        cand = "N/A" if metric["candidate_recall"] is None else f"{metric['candidate_recall']:.6f}"
        delta = "N/A" if metric["recall_delta"] is None else f"{metric['recall_delta']:+.6f}"
        lines.append(
            f"| {anomaly} | {metric['positive_rows']:,} | {'adequate' if metric['adequate_support'] else 'sparse/diagnostic'} | {inc} | {cand} | {delta} | {'yes' if metric['veto_applies'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 무결성 및 한계",
            "",
            "- P1 train에서 파생된 frozen OOF와 frozen Round-B prediction만 읽었습니다. 공식 test/sample/submission/candidate는 읽거나 생성하지 않았고 upload는 0회입니다.",
            "- 모델 fit은 0회입니다. 기존 모델·Round E·P3 ERA5 정의 및 산출물은 수정하지 않았습니다.",
            "- 동일 OOF를 한 번 사용한 구조 screen이며 fresh holdout 또는 공식 성능 보장은 아닙니다.",
            "- 이 challenger는 내부 hole만 복구하므로 아직 발견되지 않은 event나 바깥 boundary는 복구할 수 없습니다.",
            "- 결과를 본 뒤 gap, hull, endpoint 또는 gate를 변경하지 않았습니다.",
            "",
            "## 재현 명령",
            "",
            f"`{result['execution']['command']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(data_dir: Path) -> int:
    started = time.perf_counter()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"one-shot artifact already exists: {OUTPUT_DIR}")
    _self_check()
    prior_audit = _prior_family_audit()
    incumbent_frame, truth_frame, input_hashes = _load_inputs(data_dir)
    if len(incumbent_frame) != EXPECTED_ROWS or len(truth_frame) != EXPECTED_ROWS:
        raise RuntimeError("frozen OOF row count drift")
    incumbent_keys = _canonical_key_frame(incumbent_frame)
    truth_keys = _canonical_key_frame(truth_frame)
    ordered_key_sha = _assert_ordered_key_equality(incumbent_keys, truth_keys)
    if incumbent_frame["row_position"].duplicated().any():
        raise RuntimeError("incumbent source row positions are not unique")
    if (pd.to_numeric(incumbent_frame["row_position"], errors="raise") < 0).any():
        raise RuntimeError("incumbent source row position is negative")
    split_audit = _validate_split_contract(incumbent_keys)
    anchor = _checked_binary(incumbent_frame["round_b_prediction"], name="round_b_prediction")
    seed_predictions = np.column_stack(
        [_checked_binary(incumbent_frame[column], name=column) for column in SEED_COLUMNS]
    )
    truth = _checked_binary(truth_frame["label"], name="label")
    incumbent_metrics = _binary_metrics(truth, anchor)
    for name, expected in EXPECTED_INCUMBENT_COUNTS.items():
        if int(incumbent_metrics[name]) != expected:
            raise RuntimeError(f"frozen incumbent metric parity failed: {name}")
    registered_metrics = json.loads(INCUMBENT_METRICS.read_text(encoding="utf-8"))["pooled"][
        "baseline"
    ]
    if any(
        int(incumbent_metrics[name]) != int(registered_metrics[name])
        for name in ("tp", "fp", "fn", "tn")
    ):
        raise RuntimeError("frozen incumbent result parity failed")

    proposal_metadata = incumbent_keys.loc[:, ["fold", "station", "layer", "time"]].copy()
    candidate, proposals, proposal_audit = _proposal_bank(
        proposal_metadata,
        anchor,
        seed_predictions,
    )
    # Freeze is complete.  Only from this point may truth affect summaries.
    proposal_evaluation = _proposal_aggregates(proposals, truth, anchor, candidate)
    event_ids = _event_ids(proposal_metadata, truth)
    all_rows = np.ones(len(truth), dtype=bool)
    pooled = _slice_metric(
        proposal_metadata, truth, anchor, candidate, event_ids, all_rows
    )
    folds = _group_metrics(
        proposal_metadata, truth, anchor, candidate, event_ids, ("fold",)
    )
    stations = _group_metrics(
        proposal_metadata, truth, anchor, candidate, event_ids, ("station",)
    )
    layers = _group_metrics(
        proposal_metadata, truth, anchor, candidate, event_ids, ("layer",)
    )
    station_layers = _group_metrics(
        proposal_metadata, truth, anchor, candidate, event_ids, ("station", "layer")
    )
    station_layer_folds = _group_metrics(
        proposal_metadata,
        truth,
        anchor,
        candidate,
        event_ids,
        ("station", "layer", "fold"),
    )
    types = _type_metrics(
        proposal_metadata,
        truth,
        anchor,
        candidate,
        truth_frame["anomaly_type"],
        event_ids,
    )

    improved_folds = int(sum(folds[name]["f1_delta"] > 0 for name in FOLD_ORDER))
    adequate_station_layers = {
        key: metric for key, metric in station_layers.items() if metric["adequate_support"]
    }
    sparse_station_layers = {
        key: {
            "positive_rows": metric["positive_rows"],
            "positive_events": metric["positive_events"],
            "positive_station_layer_kst_days": metric["positive_station_layer_kst_days"],
        }
        for key, metric in station_layers.items()
        if not metric["adequate_support"]
    }
    worst_adequate_station_layer_delta = (
        min(metric["f1_delta"] for metric in adequate_station_layers.values())
        if adequate_station_layers
        else None
    )
    supported_non_spike_types = {
        name: metric
        for name, metric in types.items()
        if name != "spike" and metric["veto_applies"]
    }
    worst_supported_type_recall_delta = (
        min(float(metric["recall_delta"]) for metric in supported_non_spike_types.values())
        if supported_non_spike_types
        else None
    )
    checks = {
        "delta_micro_f1_gte_0_0015": pooled["f1_delta"] >= PROMOTION_F1_DELTA,
        "at_least_2_of_3_folds_improve": improved_folds >= PROMOTION_MIN_IMPROVED_FOLDS,
        "adequate_station_layer_drop_within_0_015": (
            worst_adequate_station_layer_delta is not None
            and worst_adequate_station_layer_delta >= -MAX_ADEQUATE_STATION_LAYER_F1_DROP
        ),
        "supported_non_spike_type_recall_drop_within_0_02": (
            worst_supported_type_recall_delta is not None
            and worst_supported_type_recall_delta
            >= -MAX_SUPPORTED_NON_SPIKE_TYPE_RECALL_DROP
        ),
        "identical_ordered_row_universe": True,
        "registered_seven_day_purge": True,
        "prior_close_gap_surface_disjoint": proposal_audit[
            "prior_close_gap_surface_overlap_count"
        ]
        == 0,
        "non_destructive_anchor": proposal_audit["removed_incumbent_positive_rows"] == 0,
    }
    promoted = all(checks.values())
    decision = "PROMOTE_TO_CONFIRMATION" if promoted else "REJECT_FAMILY"
    short_decision = "PROMOTE" if promoted else "REJECT"
    result: dict[str, Any] = {
        "schema_version": "ocean_hackathon.p1_structural_challenger.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "short_decision": short_decision,
        "one_shot_screen": True,
        "hypothesis": (
            "A frozen offset/drift-duration proposal bank can recover internal long-event holes "
            "that survived the prior close-gap<=6 postprocess, without deleting any incumbent positive."
        ),
        "frozen_rule": {
            "gap_rows_inclusive": [MIN_GAP_ROWS, MAX_GAP_ROWS],
            "offset_drift_hull_rows_inclusive": [MIN_LONG_HULL_ROWS, MAX_LONG_HULL_ROWS],
            "drift_minimum_hull_rows": DRIFT_MIN_HULL_ROWS,
            "endpoint": "immediate bounding anchor rows unanimous positive across three registered seeds",
            "cadence_minutes": CADENCE_MINUTES,
            "may_remove_incumbent_positive": False,
            "proposal_uses_truth_or_anomaly_type": False,
            "post_result_changes": 0,
        },
        "prior_family_audit": prior_audit,
        "lineage": {
            "ordered_key_sha256": ordered_key_sha,
            "rows": len(truth),
            "fold_order": list(FOLD_ORDER),
            "split_audit": split_audit,
            "incumbent_metric_parity": True,
            "source_train_sha256": EXPECTED_SOURCE_SHA256,
        },
        "proposal_generation_audit": proposal_audit,
        "proposal_evaluation": proposal_evaluation,
        "metrics": {
            "pooled": pooled,
            "folds": {name: folds[name] for name in FOLD_ORDER},
            "stations": stations,
            "layers": layers,
            "station_layers": station_layers,
            "station_layer_folds_diagnostic": station_layer_folds,
            "types": types,
        },
        "decision_evidence": {
            "checks": checks,
            "improved_folds_of_3": improved_folds,
            "adequate_station_layer_count": len(adequate_station_layers),
            "sparse_station_layers_diagnostic_only": sparse_station_layers,
            "worst_adequate_station_layer_f1_delta": worst_adequate_station_layer_delta,
            "supported_non_spike_type_count": len(supported_non_spike_types),
            "worst_supported_non_spike_type_recall_delta": worst_supported_type_recall_delta,
            "promotion_rule": (
                "delta_micro_f1 >= 0.0015 and at least 2/3 folds improve, with all integrity "
                "and supported-slice guardrails passing"
            ),
        },
        "input_sha256": input_hashes,
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(
                Path(__file__).resolve()
            )
        },
        "operation_counters": {
            "physical_model_fits": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
            "model_artifacts_modified": 0,
            "era5_paths_read_or_modified": 0,
            "round_e_paths_read_or_modified": 0,
            "row_level_predictions_persisted": 0,
            "result_driven_reruns": 0,
        },
        "execution": {
            "command": (
                ".venv-p1\\Scripts\\python.exe "
                "scripts\\run_p1_structural_challenger_20260827_v1.py "
                "--data-dir <P1_DATA_DIR>"
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "pre_science_failed_calls": [
                {
                    "count": 1,
                    "reason": (
                        "An over-strict nominal validation-end assertion rejected 119 rows in the "
                        "already frozen Q3 membership. The call stopped before proposal generation, "
                        "truth scoring, artifact creation, or result review."
                    ),
                    "proposal_generation_count": 0,
                    "score_count": 0,
                    "artifact_count": 0,
                    "scientific_hypothesis_or_gate_change": False,
                }
            ],
        },
        "limitations": [
            "This is one frozen train-derived OOF screen, not a fresh holdout or official score.",
            "The proposal can fill only bounded internal holes; it cannot discover disconnected events or extend outside boundaries.",
            "Unanimous binary endpoints discard probability magnitude and may be over-conservative.",
            "Sparse station-layer and anomaly-type slices are diagnostic only and do not trigger regression vetoes.",
            "No post-result threshold, duration, endpoint, split, feature, or model change is permitted.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(OUTPUT_DIR / "result.json", result)
    _write_text_atomic(OUTPUT_DIR / "report_ko.md", _build_report(result))
    manifest = {
        "schema_version": "ocean_hackathon.p1_structural_challenger.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "artifact_sha256": {
            "result.json": _sha256(OUTPUT_DIR / "result.json"),
            "report_ko.md": _sha256(OUTPUT_DIR / "report_ko.md"),
        },
        "implementation_sha256": result["implementation_sha256"],
        "input_sha256": input_hashes,
        "no_row_level_output": True,
        "submission_or_upload_count": 0,
    }
    _write_json_atomic(OUTPUT_DIR / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "decision": decision,
                "rows": pooled["rows"],
                "incumbent_f1": pooled["incumbent"]["f1"],
                "candidate_f1": pooled["candidate"]["f1"],
                "delta_f1": pooled["f1_delta"],
                "improved_folds": improved_folds,
                "accepted_proposals": proposal_evaluation["accepted_proposals"],
                "added_rows": proposal_evaluation["added_rows"],
                "artifact_dir": str(OUTPUT_DIR),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen P1 structural challenger once")
    parser.add_argument("--data-dir", type=Path, required=True, help="P1_qc_anomaly directory")
    arguments = parser.parse_args()
    return run(arguments.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
