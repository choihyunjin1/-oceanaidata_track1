"""Preclaim identifiability and error-provenance guards for P2 DTW v2r6.

The frozen v2r6 diagnosis proved that the first registered inner window has no
finite target-temperature truth rows.  This module therefore contains no
scientific materializer or scorer.  It reproduces an aggregate-only finite-mask
certificate, fails closed before any claim, and provides the bounded sanitized
worker-error envelope that v2r5 lacked.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
TARGET_LAYERS = (2, 3, 4)
CELLS = ("d1_k3", "d1_k7", "d3_k3", "d3_k7", "d7_k3", "d7_k7")
MATERIALIZATION_SLOTS = 22
MAX_ERROR_ENVELOPE_BYTES = 65_536
MAX_ERROR_CHAIN = 8
MAX_ERROR_FRAMES = 32
MAX_ERROR_MESSAGE_CHARS = 512


@dataclass(frozen=True)
class InnerWindow:
    window_id: str
    start_kst: str
    end_exclusive_kst: str


INNER_WINDOWS = (
    InnerWindow(
        "inner_2024_mar",
        "2024-03-01T00:00:00+09:00",
        "2024-04-01T00:00:00+09:00",
    ),
    InnerWindow(
        "inner_2024_may",
        "2024-05-01T00:00:00+09:00",
        "2024-06-01T00:00:00+09:00",
    ),
    InnerWindow(
        "inner_2024_jul",
        "2024-07-01T00:00:00+09:00",
        "2024-08-01T00:00:00+09:00",
    ),
)

EXPECTED_WINDOWS = {
    "inner_2024_mar": {
        "time_keys": 4_464,
        "layers": {
            "2": (0, 4_454, 0),
            "3": (0, 4_454, 0),
            "4": (0, 4_454, 0),
        },
        "rows_after_mask": 0,
        "kst_days_after_mask": 0,
        "identifiable": False,
    },
    "inner_2024_may": {
        "time_keys": 4_464,
        "layers": {
            "2": (3_347, 4_247, 3_332),
            "3": (3_339, 4_247, 3_324),
            "4": (3_344, 4_247, 3_329),
        },
        "rows_after_mask": 9_985,
        "kst_days_after_mask": 24,
        "identifiable": True,
    },
    "inner_2024_jul": {
        "time_keys": 4_464,
        "layers": {
            "2": (4_459, 4_464, 4_459),
            "3": (4_452, 4_464, 4_452),
            "4": (4_447, 4_464, 4_447),
        },
        "rows_after_mask": 13_358,
        "kst_days_after_mask": 31,
        "identifiable": True,
    },
}


class TrajectoryPanelLike(Protocol):
    index: pd.DatetimeIndex
    temp: pd.DataFrame
    baseline: pd.DataFrame


class InnerWindowUnidentifiable(RuntimeError):
    """Raised before claim when a frozen selection window cannot be scored."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_utc_ns(values: Sequence[Any] | pd.Series | pd.Index) -> np.ndarray:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if parsed.isna().any():
        raise ValueError("timestamp key contains NaT")
    parsed_ns = parsed.as_unit("ns")
    ns = parsed_ns.asi8.copy()
    roundtrip = pd.DatetimeIndex(
        pd.to_datetime(ns, unit="ns", utc=True)
    ).as_unit("ns")
    if not np.array_equal(roundtrip.asi8, ns):
        raise ValueError("UTC-ns timestamp roundtrip failed")
    return ns


def _validate_panel_contract(panel: TrajectoryPanelLike) -> None:
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.index.empty:
        raise ValueError("trajectory panel index is empty or not datetime")
    index_ns = normalize_utc_ns(panel.index)
    if len(np.unique(index_ns)) != len(index_ns):
        raise ValueError("trajectory panel contains duplicate timestamps")
    if not np.all(np.diff(index_ns) > 0):
        raise ValueError("trajectory panel timestamps are not strictly increasing")
    if tuple(panel.temp.columns) != tuple(range(1, 9)):
        raise ValueError("trajectory temperature layer surface changed")
    if tuple(panel.baseline.columns) != TARGET_LAYERS:
        raise ValueError("trajectory baseline target layers changed")
    if not panel.temp.index.equals(panel.index) or not panel.baseline.index.equals(
        panel.index
    ):
        raise ValueError("trajectory panel surfaces are not index-aligned")


def build_inner_identifiability_certificate(
    panel: TrajectoryPanelLike,
) -> dict[str, Any]:
    """Return only time/key counts and finite masks; never score or expose values."""

    _validate_panel_contract(panel)
    windows: list[dict[str, Any]] = []
    for window in INNER_WINDOWS:
        start = pd.Timestamp(window.start_kst).tz_convert("UTC")
        end = pd.Timestamp(window.end_exclusive_kst).tz_convert("UTC")
        key_mask = (panel.index >= start) & (panel.index < end)
        times = panel.index[key_mask]
        layer_masks: dict[str, dict[str, int]] = {}
        finite_days: set[str] = set()
        rows_after_mask = 0
        for layer in TARGET_LAYERS:
            truth_finite = np.isfinite(
                panel.temp.loc[times, layer].to_numpy(dtype=np.float64)
            )
            anchor_finite = np.isfinite(
                panel.baseline.loc[times, layer].to_numpy(dtype=np.float64)
            )
            both = truth_finite & anchor_finite
            both_count = int(both.sum())
            rows_after_mask += both_count
            if both_count:
                finite_days.update(
                    times[both]
                    .tz_convert(KST)
                    .normalize()
                    .astype(str)
                    .tolist()
                )
            layer_masks[str(layer)] = {
                "finite_truth": int(truth_finite.sum()),
                "finite_anchor": int(anchor_finite.sum()),
                "finite_truth_and_anchor": both_count,
            }
        every_layer_nonempty = all(
            layer_masks[str(layer)]["finite_truth_and_anchor"] > 0
            for layer in TARGET_LAYERS
        )
        identifiable = bool(
            rows_after_mask > 0
            and every_layer_nonempty
            and len(finite_days) >= 2
        )
        windows.append(
            {
                "window_id": window.window_id,
                "time_keys": int(len(times)),
                "layer_finite_masks": layer_masks,
                "rows_after_finite_truth_anchor_mask": rows_after_mask,
                "kst_days_after_mask": len(finite_days),
                "identifiable_for_scoring": identifiable,
            }
        )
    return {
        "schema_version": "p2_dtw_v2r6.inner_identifiability_runtime.v1",
        "windows": windows,
        "all_three_registered_windows_identifiable": all(
            row["identifiable_for_scoring"] for row in windows
        ),
        "frozen_eighteen_cell_selection_complete": all(
            row["identifiable_for_scoring"] for row in windows
        ),
        "prediction_values_read_or_written": 0,
        "metrics_computed": 0,
        "scores_computed": 0,
        "physical_fit_calls": 0,
        "p100_accesses": 0,
    }


def verify_frozen_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    windows = certificate.get("windows")
    if not isinstance(windows, list) or len(windows) != len(INNER_WINDOWS):
        raise ValueError("inner identifiability window inventory changed")
    observed_ids = [str(row.get("window_id")) for row in windows]
    if observed_ids != [window.window_id for window in INNER_WINDOWS]:
        raise ValueError("inner identifiability window order changed")
    for row in windows:
        window_id = str(row["window_id"])
        expected = EXPECTED_WINDOWS[window_id]
        if int(row["time_keys"]) != int(expected["time_keys"]):
            raise ValueError(f"{window_id} time-key count changed")
        observed_layers = row.get("layer_finite_masks")
        if not isinstance(observed_layers, Mapping) or set(observed_layers) != {
            "2",
            "3",
            "4",
        }:
            raise ValueError(f"{window_id} target-layer inventory changed")
        expected_layers = expected["layers"]
        assert isinstance(expected_layers, Mapping)
        for layer in TARGET_LAYERS:
            observed = observed_layers[str(layer)]
            expected_counts = expected_layers[str(layer)]
            counts = (
                int(observed["finite_truth"]),
                int(observed["finite_anchor"]),
                int(observed["finite_truth_and_anchor"]),
            )
            if counts != expected_counts:
                raise ValueError(f"{window_id} layer {layer} finite-mask count changed")
        if (
            int(row["rows_after_finite_truth_anchor_mask"])
            != int(expected["rows_after_mask"])
            or int(row["kst_days_after_mask"])
            != int(expected["kst_days_after_mask"])
            or bool(row["identifiable_for_scoring"])
            is not bool(expected["identifiable"])
        ):
            raise ValueError(f"{window_id} identifiability result changed")
    if (
        certificate.get("all_three_registered_windows_identifiable") is not False
        or certificate.get("frozen_eighteen_cell_selection_complete") is not False
    ):
        raise ValueError("frozen March blocker was not preserved")
    if any(
        int(certificate.get(name, -1)) != 0
        for name in (
            "prediction_values_read_or_written",
            "metrics_computed",
            "scores_computed",
            "physical_fit_calls",
            "p100_accesses",
        )
    ):
        raise ValueError("identifiability certificate contains forbidden operations")
    return {
        "status": "NO_GO_INNER_WINDOW_UNIDENTIFIABLE",
        "blocking_window": "inner_2024_mar",
        "blocking_rows_after_mask": 0,
        "claim_permitted": False,
        "worker_launch_permitted": False,
        "materialization_permitted": False,
        "score_permitted": False,
        "physical_fit_calls": 0,
        "p100_accesses": 0,
    }


def require_identifiable_or_stop(certificate: Mapping[str, Any]) -> None:
    resolution = verify_frozen_certificate(certificate)
    if resolution["status"] == "NO_GO_INNER_WINDOW_UNIDENTIFIABLE":
        raise InnerWindowUnidentifiable(
            "preclaim stop: frozen inner_2024_mar has zero finite truth-and-anchor "
            "rows; the preregistered 18-cell selection is non-identifiable"
        )


def sanitize_message(message: str) -> str:
    value = message.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"[A-Za-z]:[\\/][^\s]+", "<ABSOLUTE_PATH_REDACTED>", value)
    value = re.sub(r"/(?:[^\s/]+/)+[^\s]+", "<ABSOLUTE_PATH_REDACTED>", value)
    return value[:MAX_ERROR_MESSAGE_CHARS]


def sanitized_error_envelope(
    error: BaseException,
    *,
    phase: str,
) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < MAX_ERROR_CHAIN:
        seen.add(id(current))
        raw_message = str(current)
        chain.append(
            {
                "type": type(current).__name__,
                "module": type(current).__module__,
                "message_sanitized": sanitize_message(raw_message),
                "message_sha256": sha256_bytes(
                    raw_message.encode("utf-8", errors="replace")
                ),
            }
        )
        current = current.__cause__ or current.__context__
    extracted = traceback.extract_tb(error.__traceback__)[-MAX_ERROR_FRAMES:]
    frames = [
        {
            "file": Path(frame.filename).name,
            "module": Path(frame.filename).stem,
            "function": frame.name,
            "line": int(frame.lineno),
        }
        for frame in extracted
    ]
    raw_traceback = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    envelope = {
        "schema_version": "p2_dtw_v2r6.worker_error_envelope.v1",
        "phase": str(phase)[:128],
        "chain": chain,
        "frames": frames,
        "traceback_sha256": sha256_bytes(
            raw_traceback.encode("utf-8", errors="replace")
        ),
        "traceback_frame_count": len(frames),
        "locals_captured": False,
        "raw_traceback_persisted": False,
        "prediction_values_captured": 0,
        "truth_values_captured": 0,
        "metric_values_captured": 0,
    }
    validate_error_envelope(envelope)
    return envelope


def validate_error_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "phase",
        "chain",
        "frames",
        "traceback_sha256",
        "traceback_frame_count",
        "locals_captured",
        "raw_traceback_persisted",
        "prediction_values_captured",
        "truth_values_captured",
        "metric_values_captured",
    }
    if set(envelope) != required:
        raise ValueError("worker error envelope schema changed")
    if envelope["schema_version"] != "p2_dtw_v2r6.worker_error_envelope.v1":
        raise ValueError("worker error envelope version changed")
    chain = envelope["chain"]
    frames = envelope["frames"]
    if not isinstance(chain, list) or not 1 <= len(chain) <= MAX_ERROR_CHAIN:
        raise ValueError("worker error envelope chain is empty or oversized")
    if not isinstance(frames, list) or len(frames) > MAX_ERROR_FRAMES:
        raise ValueError("worker error envelope frames are oversized")
    if int(envelope["traceback_frame_count"]) != len(frames):
        raise ValueError("worker error envelope frame count differs")
    if not re.fullmatch(r"[0-9a-f]{64}", str(envelope["traceback_sha256"])):
        raise ValueError("worker error envelope traceback hash is invalid")
    for item in chain:
        if set(item) != {
            "type",
            "module",
            "message_sanitized",
            "message_sha256",
        }:
            raise ValueError("worker error envelope chain schema changed")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["message_sha256"])):
            raise ValueError("worker error envelope message hash is invalid")
        message = str(item["message_sanitized"])
        if len(message) > MAX_ERROR_MESSAGE_CHARS or "\n" in message or "\r" in message:
            raise ValueError("worker error envelope message is not bounded")
    if any(
        bool(envelope[name])
        for name in ("locals_captured", "raw_traceback_persisted")
    ):
        raise ValueError("worker error envelope captured forbidden traceback content")
    if any(
        int(envelope[name]) != 0
        for name in (
            "prediction_values_captured",
            "truth_values_captured",
            "metric_values_captured",
        )
    ):
        raise ValueError("worker error envelope captured scientific values")
    payload = canonical_json_bytes(envelope)
    if len(payload) > MAX_ERROR_ENVELOPE_BYTES:
        raise ValueError("worker error envelope exceeds byte ceiling")
    return {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "valid": True,
    }


def scientific_contract() -> dict[str, Any]:
    return {
        "cells": list(CELLS),
        "inner_windows": [window.window_id for window in INNER_WINDOWS],
        "materialization_slots": MATERIALIZATION_SLOTS,
        "physical_fit_calls": 0,
        "scientific_logic_changed": False,
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )
