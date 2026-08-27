"""Capability-only execution engine for P3 Gen6r2.

The engine imports the pinned v1 scientific functions unchanged.  It changes only
execution ordering: predictor-only load, fold-major durable blind commitments,
selective prior-fold truth release, predictions-complete, scoring, and sealing.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from . import gen6_incumbent_preserving_residual_calibrator as science
from . import gen6_incumbent_preserving_residual_calibrator_contract_r2 as guard
from .meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    evaluate_hypothesis_gate,
    evaluate_point,
)

STATIONS: Final = ("G-ORS", "I-ORS", "S-ORS")
LEADS: Final = (3, 6, 9, 12, 18, 24)
STATION_TO_CODE: Final = {value: index for index, value in enumerate(STATIONS)}
SOURCE_HEADER: Final = (b"station", b"time", b"hs", b"tp", b"hmax", b"wvdir")
TIME_PATTERN: Final = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00")
MISSING_TOKENS: Final = frozenset({b"", b"nan", b"NaN", b"NA", b"null", b"None"})
PREDICTOR_COLUMNS: Final = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "persistence",
    "incumbent_prediction",
    "prefix_fraction",
)
KEY_COLUMNS: Final = ("prefix_fraction", "fold", "anchor_id", "station", "lead_h")
TARGET_KEY_COLUMNS: Final = ("fold", "anchor_id", "station", "lead_h")


class R2ExecutionError(RuntimeError):
    """A scientific-engine ordering or integrity violation."""


def _key_sha256(frame: pd.DataFrame) -> str:
    ordered = frame[list(KEY_COLUMNS)]
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        prefix, fold, anchor_id, station, lead = row
        digest.update(
            f"{float(prefix):.2f}|{fold}|{int(anchor_id)}|{station}|{int(lead)}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(values, dtype="<f8"), allow_pickle=False)
    return buffer.getvalue()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


class SelectiveOfficialTargetVault:
    """Decode only six official train-wave targets for a committed fold."""

    def __init__(
        self,
        capability: object,
        wave_path: Path,
        output: Path,
        anchors: pd.DataFrame,
        validation_keys: pd.DataFrame,
        *,
        expected_sha256: str,
        expected_bytes: int,
        expected_rows: int = 118152,
        expected_validation_cases: int = 181,
    ) -> None:
        guard.verify_live_phase(
            capability,
            phase="FOLD_0_PREDICT_COMMIT",
            entry_name="engine.target_vault.key_input_only_index",
        )
        self._capability = capability
        self._wave_path = wave_path.resolve(strict=True)
        self._output = output.resolve(strict=True)
        context = guard.capability_context(capability)
        canonical_wave = (
            context["data_dir"]
            / context["config"]["immutable_inputs"]["source/train_wave.csv"]["path"]
        ).resolve(strict=True)
        canonical_output = (
            context["root"] / context["config"]["canonical_paths"]["output"]
        ).resolve(strict=True)
        if self._wave_path != canonical_wave or self._output != canonical_output:
            raise R2ExecutionError("target-vault canonical input/output identity differs")
        if self._wave_path.name != "train_wave.csv":
            raise R2ExecutionError("target vault accepts only train_wave.csv")
        if self._wave_path.stat().st_size != expected_bytes:
            raise R2ExecutionError("target-vault source byte count changed")
        if guard.sha256_file(self._wave_path) != expected_sha256:
            raise R2ExecutionError("target-vault source SHA-256 changed")
        if tuple(anchors.columns) != ("anchor_id", "station", "anchor_time"):
            raise R2ExecutionError("target-vault anchor input-only columns differ")
        ordered = anchors.sort_values("anchor_id").reset_index(drop=True).copy()
        expected_ids = np.arange(len(ordered), dtype=np.int64)
        if not np.array_equal(ordered["anchor_id"].to_numpy(np.int64), expected_ids):
            raise R2ExecutionError("target-vault anchor IDs are not contiguous")
        station = ordered["station"].map(STATION_TO_CODE)
        if station.isna().any():
            raise R2ExecutionError("target-vault anchor station differs")
        times = pd.DatetimeIndex(
            pd.to_datetime(ordered["anchor_time"], utc=True, errors="raise")
        ).as_unit("ns")
        self._anchor_station = station.to_numpy(np.int8)
        self._anchor_time_ns = times.asi8.astype(np.int64, copy=False)

        # Parse only the first two comma-delimited fields.  The suffix beginning
        # with hs stays opaque until release(), so malformed/poisoned target text
        # cannot influence input indexing or blind prediction control flow.
        identity_station: list[str] = []
        identity_time: list[str] = []
        offsets = np.empty(expected_rows, dtype=np.int64)
        with self._wave_path.open("rb") as stream:
            if tuple(stream.readline().rstrip(b"\r\n").split(b",")) != SOURCE_HEADER:
                raise R2ExecutionError("target-vault source header changed")
            row = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                identity = line.rstrip(b"\r\n").split(b",", 2)
                if len(identity) != 3 or row >= expected_rows:
                    raise R2ExecutionError("target-vault source identity row changed")
                station_token, time_token, _opaque_target_suffix = identity
                if not TIME_PATTERN.fullmatch(time_token):
                    raise R2ExecutionError("target-vault timestamp token changed")
                identity_station.append(station_token.decode("ascii"))
                identity_time.append(time_token.decode("ascii"))
                offsets[row] = offset
                row += 1
        if row != expected_rows:
            raise R2ExecutionError("target-vault binary row count changed")
        self._offsets = offsets
        encoded_station_series = pd.Series(identity_station).map(STATION_TO_CODE)
        if encoded_station_series.isna().any():
            raise R2ExecutionError("target-vault source station differs")
        encoded_station = encoded_station_series.to_numpy(np.int8)
        source_time = pd.DatetimeIndex(
            pd.to_datetime(identity_time, utc=True, errors="raise")
        ).as_unit("ns")
        source_ns = source_time.asi8.astype(np.int64, copy=False)
        self._station_rows: dict[int, np.ndarray] = {}
        self._station_times: dict[int, np.ndarray] = {}
        for code in range(len(STATIONS)):
            rows = np.flatnonzero(encoded_station == code).astype(np.int64)
            order = np.argsort(source_ns[rows], kind="stable")
            rows = rows[order]
            values = source_ns[rows]
            if len(values) == 0 or np.any(np.diff(values) <= 0):
                raise R2ExecutionError("target-vault station times are not increasing")
            self._station_rows[code] = rows
            self._station_times[code] = values

        required_keys = ("fold", "anchor_id", "station", "episode_id")
        if (
            tuple(validation_keys.columns) != required_keys
            or len(validation_keys) != expected_validation_cases
        ):
            raise R2ExecutionError("target-vault validation-key surface changed")
        self._case_ids: dict[str, np.ndarray] = {}
        self._target_rows: dict[str, np.ndarray] = {}
        observed_ids: set[int] = set()
        observed_rows: set[int] = set()
        lead_ns = np.asarray(LEADS, dtype=np.int64) * 3600 * 1_000_000_000
        for fold in guard.FOLD_ORDER:
            current = validation_keys.loc[validation_keys["fold"].eq(fold)].sort_values(
                ["anchor_id", "station"]
            )
            ids = current["anchor_id"].to_numpy(dtype=np.int64)
            if len(ids) == 0 or len(np.unique(ids)) != len(ids):
                raise R2ExecutionError("target-vault fold case IDs changed")
            if np.any(ids < 0) or np.any(ids >= len(self._anchor_station)):
                raise R2ExecutionError("target-vault fold case ID is outside anchor metadata")
            expected_station = np.asarray(
                [STATIONS[int(code)] for code in self._anchor_station[ids]],
                dtype=object,
            )
            if not np.array_equal(
                current["station"].astype(str).to_numpy(), expected_station
            ):
                raise R2ExecutionError("target-vault validation station/key differs")
            if observed_ids.intersection(int(value) for value in ids):
                raise R2ExecutionError("target-vault fold IDs overlap")
            observed_ids.update(int(value) for value in ids)
            target_times = self._anchor_time_ns[ids, None] + lead_ns[None, :]
            rows = np.full(target_times.shape, -1, dtype=np.int64)
            for code in range(len(STATIONS)):
                local = np.flatnonzero(self._anchor_station[ids] == code)
                if len(local) == 0:
                    continue
                station_times = self._station_times[code]
                station_rows = self._station_rows[code]
                queries = target_times[local].reshape(-1)
                positions = np.searchsorted(station_times, queries)
                safe = np.minimum(positions, len(station_times) - 1)
                found = (positions < len(station_times)) & (
                    station_times[safe] == queries
                )
                located = np.full(len(queries), -1, dtype=np.int64)
                located[found] = station_rows[positions[found]]
                rows[local] = located.reshape(len(local), len(LEADS))
            if np.any(rows < 0):
                raise R2ExecutionError("target-vault official validation row is unavailable")
            flattened = set(int(value) for value in rows.reshape(-1))
            if len(flattened) != rows.size:
                raise R2ExecutionError("target-vault source targets overlap within a fold")
            if observed_rows.intersection(flattened):
                raise R2ExecutionError("target-vault source targets overlap across folds")
            observed_rows.update(flattened)
            self._case_ids[fold] = ids
            self._target_rows[fold] = rows
        self._next_release = 0
        self._released: dict[str, str] = {}
        self._decoded_source_rows: set[int] = set()
        self._total_scalar_decodes = 0
        self._expected_total_scalar_decodes = int(
            sum(values.size for values in self._target_rows.values())
        )
        self._forbidden_release_attempts = 0

    def release(
        self,
        capability: object,
        fold: str,
        *,
        fold_commitment_path: Path,
        fold_commitment_sha256: str,
    ) -> pd.DataFrame:
        if self._next_release >= len(guard.FOLD_ORDER):
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault release replay rejected")
        if capability is not self._capability:
            self._forbidden_release_attempts += 1
            raise guard.R2CapabilityError("cross-capability target-vault call rejected")
        expected_fold = guard.FOLD_ORDER[self._next_release]
        if fold != expected_fold or fold in self._released:
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault fold release order/replay rejected")
        release_phase = (
            f"FOLD_{self._next_release + 1}_PREDICT_COMMIT"
            if self._next_release < 2
            else "SCORE_AND_WRITE_CORE"
        )
        guard.verify_live_phase(
            capability,
            phase=release_phase,
            entry_name=f"engine.target_vault.release[{self._next_release}]",
        )
        expected_commitment = (
            self._output
            / f"commitments/fold_{self._next_release:02d}_{expected_fold}.json"
        ).resolve(strict=True)
        if fold_commitment_path.resolve(strict=True) != expected_commitment:
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault commitment path differs")
        if not re.fullmatch(r"[0-9a-f]{64}", fold_commitment_sha256):
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault commitment digest is invalid")
        if guard.sha256_file(fold_commitment_path) != fold_commitment_sha256:
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault fold commitment changed")
        commitment = json.loads(fold_commitment_path.read_bytes())
        prediction = self._output / f"blind/fold_{self._next_release:02d}_{fold}.npy"
        if (
            commitment.get("fold_index") != self._next_release
            or commitment.get("fold") != fold
            or commitment.get("truth_attached") is not False
            or commitment.get("blind_prediction")
            != guard.file_pin(prediction, root=self._output)
        ):
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault fold commitment semantics changed")
        rows = self._target_rows[fold]
        requested = [int(value) for value in rows.reshape(-1)]
        if self._decoded_source_rows.intersection(requested):
            self._forbidden_release_attempts += 1
            raise PermissionError("target-vault source target was already decoded")
        values: dict[int, float] = {}
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = os.open(self._wave_path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                for row in requested:
                    stream.seek(int(self._offsets[row]))
                    fields = stream.readline().rstrip(b"\r\n").split(b",")
                    if len(fields) != len(SOURCE_HEADER) or fields[2] in MISSING_TOKENS:
                        raise R2ExecutionError("released target row became malformed")
                    try:
                        value = float(fields[2].decode("ascii"))
                    except (UnicodeDecodeError, ValueError) as exception:
                        raise R2ExecutionError(
                            "released target scalar is not a valid ASCII float"
                        ) from exception
                    if not np.isfinite(value):
                        raise R2ExecutionError("released target scalar is non-finite")
                    values[row] = value
                    self._decoded_source_rows.add(row)
                    self._total_scalar_decodes += 1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        ids = self._case_ids[fold]
        frame = pd.DataFrame(
            {
                "fold": np.repeat(fold, len(ids) * len(LEADS)),
                "anchor_id": np.repeat(ids, len(LEADS)),
                "lead_h": np.tile(np.asarray(LEADS, dtype=np.int64), len(ids)),
                "target_hs": [values[int(row)] for row in requested],
            }
        )
        station_by_id = {
            int(anchor_id): STATIONS[int(self._anchor_station[int(anchor_id)])]
            for anchor_id in ids
        }
        frame.insert(
            2,
            "station",
            frame["anchor_id"].map(station_by_id).astype(str),
        )
        self._released[fold] = fold_commitment_sha256
        self._next_release += 1
        return frame

    def access_audit(self) -> dict[str, Any]:
        return {
            "source": "train_wave.csv_selective_official_six_only",
            "float_target_decodes_during_identity_index": 0,
            "released_folds": list(self._released),
            "release_commitment_sha256": dict(self._released),
            "unique_source_target_scalar_decodes": self._total_scalar_decodes,
            "expected_total_source_target_scalar_decodes_after_all_release": (
                self._expected_total_scalar_decodes
            ),
            "forbidden_release_attempts": self._forbidden_release_attempts,
            "anonymous_test_value_reads": 0,
        }


@dataclass
class _EngineState:
    capability: object
    root: Path
    data_dir: Path
    config: dict[str, Any]
    output: Path
    predictors: pd.DataFrame
    anchors: pd.DataFrame
    validation_keys: pd.DataFrame
    vault: SelectiveOfficialTargetVault
    truth_by_fold: dict[str, pd.DataFrame] = field(default_factory=dict)
    blind_by_fold: dict[str, pd.DataFrame] = field(default_factory=dict)
    fold_commitments: dict[str, dict[str, Any]] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    model_receipts: list[dict[str, Any]] = field(default_factory=list)
    predictions_complete: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    started: float = field(default_factory=time.perf_counter)


_STATE_LOCK = threading.RLock()
_STATE_REGISTRY: dict[int, _EngineState] = {}


def _require_state(capability: object, state: object) -> _EngineState:
    guard.capability_context(capability)
    with _STATE_LOCK:
        expected = _STATE_REGISTRY.get(id(capability))
        if expected is None or expected.capability is not capability or expected is not state:
            raise guard.R2CapabilityError("forged, stale, or cross-capability engine state rejected")
        return expected


def _validate_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != PREDICTOR_COLUMNS or len(frame) != 5430:
        raise R2ExecutionError("predictor-only Gen1 OOF surface changed")
    if "target_hs" in frame.columns:
        raise R2ExecutionError("target_hs was decoded during key/input-only load")
    keys = list(KEY_COLUMNS)
    if frame.duplicated(keys).any():
        raise R2ExecutionError("predictor-only keys are duplicated")
    if tuple(sorted(frame["prefix_fraction"].unique())) != PREFIX_FRACTIONS:
        raise R2ExecutionError("predictor-only prefix values changed")
    if set(frame["fold"].astype(str)) != set(guard.FOLD_ORDER):
        raise R2ExecutionError("predictor-only fold values changed")
    if set(frame["station"].astype(str)) != set(STATIONS):
        raise R2ExecutionError("predictor-only station values changed")
    if set(frame["lead_h"].astype(int)) != set(LEADS):
        raise R2ExecutionError("predictor-only lead values changed")
    numeric = frame[
        ["current_hs", "persistence", "incumbent_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise R2ExecutionError("predictor-only numeric values are non-finite")
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def load_key_input_only(capability: object) -> _EngineState:
    """First engine entry: consume no target column or target scalar."""

    guard.enter_engine_phase(
        capability,
        expected_phase="ENGINE_LOAD_INPUTS",
        next_phase="FOLD_0_PREDICT_COMMIT",
        entry_name="engine.load_key_input_only",
    )
    context = guard.capability_context(capability)
    root: Path = context["root"]
    data_dir: Path = context["data_dir"]
    config: dict[str, Any] = context["config"]
    paths = config["canonical_paths"]
    predictors = pd.read_parquet(
        root / paths["sealed_gen1_oof"], columns=list(PREDICTOR_COLUMNS)
    )
    predictors = _validate_predictors(predictors)
    anchors = pd.read_parquet(
        root / paths["train_anchor_metadata"],
        columns=["anchor_id", "station", "anchor_time"],
    )
    if tuple(anchors.columns) != ("anchor_id", "station", "anchor_time"):
        raise R2ExecutionError("anchor key/input-only load changed")
    validation_keys = pd.read_parquet(root / paths["sealed_validation_keys"])
    source_pin = config["immutable_inputs"]["source/train_wave.csv"]
    output = guard.create_output_directories(
        capability, phase="FOLD_0_PREDICT_COMMIT"
    )
    vault = SelectiveOfficialTargetVault(
        capability,
        data_dir / source_pin["path"],
        output,
        anchors,
        validation_keys,
        expected_sha256=source_pin["sha256"],
        expected_bytes=source_pin["bytes"],
    )
    if vault.access_audit()["unique_source_target_scalar_decodes"] != 0:
        raise R2ExecutionError("target vault decoded a scalar during input-only load")
    state = _EngineState(
        capability=capability,
        root=root,
        data_dir=data_dir,
        config=config,
        output=output,
        predictors=predictors,
        anchors=anchors,
        validation_keys=validation_keys,
        vault=vault,
    )
    with _STATE_LOCK:
        if id(capability) in _STATE_REGISTRY:
            raise guard.R2CapabilityError("engine state already exists for capability")
        _STATE_REGISTRY[id(capability)] = state
    return state


def _historical_truth_for_prefix(
    state: _EngineState, *, prefix: float, fold_index: int
) -> pd.DataFrame:
    prior = guard.FOLD_ORDER[:fold_index]
    if not prior:
        return pd.DataFrame(
            columns=[
                "prefix_fraction",
                "fold",
                "anchor_id",
                "station",
                "lead_h",
                "current_hs",
                "persistence",
                "incumbent_prediction",
                "anchor_time",
                "target_hs",
            ]
        )
    if set(state.truth_by_fold) != set(prior):
        raise R2ExecutionError("prior-fold truth availability order changed")
    predictors = state.predictors.loc[
        state.predictors["prefix_fraction"].eq(prefix)
        & state.predictors["fold"].isin(prior)
    ].copy()
    anchor_time = state.anchors[["anchor_id", "station", "anchor_time"]]
    predictors = predictors.merge(
        anchor_time, on=["anchor_id", "station"], how="left", validate="many_to_one"
    )
    truth = pd.concat([state.truth_by_fold[name] for name in prior], ignore_index=True)
    result = predictors.merge(
        truth,
        on=list(TARGET_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if result["target_hs"].isna().any():
        raise R2ExecutionError("prior-fold truth join is incomplete")
    return result


def _fold_paths(index: int, fold: str) -> tuple[str, str]:
    stem = f"fold_{index:02d}_{fold}"
    return f"blind/{stem}.npy", f"commitments/{stem}.json"


def predict_and_commit_fold(
    capability: object, state: object, *, fold_index: int
) -> dict[str, Any]:
    current = _require_state(capability, state)
    if fold_index < 0 or fold_index >= len(guard.FOLD_ORDER):
        raise R2ExecutionError("fold index is outside the fixed order")
    fold = guard.FOLD_ORDER[fold_index]
    expected = f"FOLD_{fold_index}_PREDICT_COMMIT"
    if fold_index < 2:
        next_phase = f"FOLD_{fold_index}_RELEASE_PRIOR_TRUTH"
    else:
        next_phase = "PREDICTIONS_COMPLETE_COMMIT"
    guard.enter_engine_phase(
        capability,
        expected_phase=expected,
        next_phase=next_phase,
        entry_name=f"engine.predict_and_commit_fold[{fold_index}]",
    )
    if fold in current.blind_by_fold or fold in current.fold_commitments:
        raise R2ExecutionError("fold prediction replay rejected")
    target_decodes_before = current.vault.access_audit()[
        "unique_source_target_scalar_decodes"
    ]
    frames: list[pd.DataFrame] = []
    fold_receipts: list[dict[str, Any]] = []
    for prefix in PREFIX_FRACTIONS:
        prefix_predictors = current.predictors.loc[
            current.predictors["prefix_fraction"].eq(prefix)
        ].copy()
        prefix_predictors = prefix_predictors.merge(
            current.anchors[["anchor_id", "station", "anchor_time"]],
            on=["anchor_id", "station"],
            how="left",
            validate="many_to_one",
        )
        historical = _historical_truth_for_prefix(
            current, prefix=prefix, fold_index=fold_index
        )
        prediction, receipt, model = science._predict_one_outer_fold(
            prefix=prefix,
            outer_fold=fold,
            predictors=prefix_predictors,
            historical_truth=historical,
        )
        outer = prefix_predictors.loc[prefix_predictors["fold"].eq(fold)].copy()
        frame = outer[list(KEY_COLUMNS)].copy()
        frame[science.CANDIDATE_COLUMN] = prediction
        frames.append(frame)
        receipt = {**receipt, "fold_index": fold_index}
        fold_receipts.append(receipt)
        current.receipts.append(receipt)
        current.model_receipts.append(
            {
                "prefix_fraction": float(prefix),
                "fold_index": fold_index,
                "outer_fold": fold,
                "selected_model": None if model is None else model.to_json(),
                "fit_count": int(receipt["fit_count"]),
                "decision": receipt["decision"],
            }
        )
    blind = pd.concat(frames, ignore_index=True).sort_values(
        list(KEY_COLUMNS)
    ).reset_index(drop=True)
    if blind.duplicated(list(KEY_COLUMNS)).any():
        raise R2ExecutionError("blind fold keys are duplicated")
    expected_rows = 5 * int(
        current.validation_keys.loc[current.validation_keys["fold"].eq(fold)].shape[0]
    ) * len(LEADS)
    if len(blind) != expected_rows:
        raise R2ExecutionError("blind fold row count changed")
    prediction_values = np.ascontiguousarray(
        blind[science.CANDIDATE_COLUMN].to_numpy(dtype=np.float64), dtype=np.float64
    )
    prediction_relative, commitment_relative = _fold_paths(fold_index, fold)
    prediction_pin = guard.write_output_exclusive(
        capability,
        phase=next_phase,
        relative_path=prediction_relative,
        payload=_npy_bytes(prediction_values),
    )
    commitment = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.blind_fold_commitment.r2.v1"
        ),
        "fold_index": fold_index,
        "fold": fold,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "case_count": int(expected_rows // (5 * len(LEADS))),
        "row_count": len(blind),
        "key_sha256": _key_sha256(blind),
        "validation_ids_sha256": _ids_sha256(
            np.sort(
                current.validation_keys.loc[
                    current.validation_keys["fold"].eq(fold), "anchor_id"
                ].to_numpy(dtype=np.int64)
            )
        ),
        "blind_prediction": prediction_pin,
        "cell_receipts": fold_receipts,
        "fit_count": int(sum(row["fit_count"] for row in fold_receipts)),
        "target_scalar_decodes_before_fold_commitment": int(target_decodes_before),
        "active_fold_target_scalar_decodes_before_commitment": 0,
        "truth_attached": False,
        "candidate_or_test_prediction": False,
    }
    commitment_pin = guard.write_output_exclusive(
        capability,
        phase=next_phase,
        relative_path=commitment_relative,
        payload=guard.canonical_json_bytes(commitment) + b"\n",
    )
    current.blind_by_fold[fold] = blind
    current.fold_commitments[fold] = {
        "path": commitment_relative,
        "sha256": commitment_pin["sha256"],
        "blind_prediction": prediction_pin,
    }
    return current.fold_commitments[fold]


def release_committed_fold_truth(
    capability: object, state: object, *, fold_index: int
) -> dict[str, Any]:
    current = _require_state(capability, state)
    fold = guard.FOLD_ORDER[fold_index]
    if fold_index < 2:
        expected = f"FOLD_{fold_index}_RELEASE_PRIOR_TRUTH"
        next_phase = f"FOLD_{fold_index + 1}_PREDICT_COMMIT"
    elif fold_index == 2:
        expected = "FOLD_2_RELEASE_SCORING_TRUTH"
        next_phase = "SCORE_AND_WRITE_CORE"
        if current.predictions_complete is None:
            raise R2ExecutionError("fold 2 truth requires predictions-complete first")
    else:
        raise R2ExecutionError("fold release index is outside the fixed order")
    guard.enter_engine_phase(
        capability,
        expected_phase=expected,
        next_phase=next_phase,
        entry_name=f"engine.release_committed_fold_truth[{fold_index}]",
    )
    if fold not in current.fold_commitments or fold in current.truth_by_fold:
        raise R2ExecutionError("truth release lacks one fresh fold commitment")
    commitment = current.fold_commitments[fold]
    truth = current.vault.release(
        capability,
        fold,
        fold_commitment_path=current.output / commitment["path"],
        fold_commitment_sha256=commitment["sha256"],
    )
    expected_rows = int(
        current.validation_keys.loc[current.validation_keys["fold"].eq(fold)].shape[0]
    ) * len(LEADS)
    if len(truth) != expected_rows or not np.isfinite(
        truth["target_hs"].to_numpy(dtype=np.float64)
    ).all():
        raise R2ExecutionError("selectively released fold truth changed")
    current.truth_by_fold[fold] = truth
    return {
        "fold_index": fold_index,
        "fold": fold,
        "rows": len(truth),
        "source_target_scalar_decodes_after_release": current.vault.access_audit()[
            "unique_source_target_scalar_decodes"
        ],
        "commitment_sha256": commitment["sha256"],
    }


def commit_predictions_complete(capability: object, state: object) -> dict[str, Any]:
    current = _require_state(capability, state)
    guard.enter_engine_phase(
        capability,
        expected_phase="PREDICTIONS_COMPLETE_COMMIT",
        next_phase="FOLD_2_RELEASE_SCORING_TRUTH",
        entry_name="engine.commit_predictions_complete",
    )
    if set(current.fold_commitments) != set(guard.FOLD_ORDER):
        raise R2ExecutionError("predictions-complete lacks a fold commitment")
    if set(current.truth_by_fold) != set(guard.FOLD_ORDER[:2]):
        raise R2ExecutionError("predictions-complete truth-release order changed")
    folds = [current.fold_commitments[name] for name in guard.FOLD_ORDER]
    for item in folds:
        if guard.sha256_file(current.output / item["path"]) != item["sha256"]:
            raise R2ExecutionError("fold commitment changed before predictions-complete")
        prediction = item["blind_prediction"]
        if guard.sha256_file(current.output / prediction["path"]) != prediction["sha256"]:
            raise R2ExecutionError("blind prediction changed before predictions-complete")
    payload = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.predictions_complete.r2.v1"
        ),
        "fold_order": list(guard.FOLD_ORDER),
        "fold_commitments": folds,
        "fold_0_and_1_truth_released_only_after_own_commitment": True,
        "fold_2_truth_released": False,
        "fit_count_authorized_upper_bound": 20,
        "fit_count_observed_so_far": int(
            sum(receipt["fit_count"] for receipt in current.receipts)
        ),
        "scoring_truth_attached": False,
        "anonymous_test_value_reads": 0,
        "candidate_or_test_prediction": False,
    }
    pin = guard.write_output_exclusive(
        capability,
        phase="FOLD_2_RELEASE_SCORING_TRUTH",
        relative_path="commitments/predictions_complete.json",
        payload=guard.canonical_json_bytes(payload) + b"\n",
    )
    current.predictions_complete = {**payload, "file": pin}
    return current.predictions_complete


def _durable_blind_surface(current: _EngineState) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, fold in enumerate(guard.FOLD_ORDER):
        prediction_relative, _ = _fold_paths(index, fold)
        saved = np.load(current.output / prediction_relative, allow_pickle=False)
        frame = current.blind_by_fold[fold].copy()
        if saved.dtype != np.dtype("float64") or saved.shape != (len(frame),):
            raise R2ExecutionError("durable blind prediction array shape changed")
        if not np.array_equal(
            saved,
            frame[science.CANDIDATE_COLUMN].to_numpy(dtype=np.float64),
        ):
            raise R2ExecutionError("durable blind prediction reread changed")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(
        list(KEY_COLUMNS)
    ).reset_index(drop=True)
    if len(result) != 5430 or result.duplicated(list(KEY_COLUMNS)).any():
        raise R2ExecutionError("durable blind aggregate surface changed")
    return result


def score_and_write_core(capability: object, state: object) -> dict[str, Any]:
    current = _require_state(capability, state)
    guard.enter_engine_phase(
        capability,
        expected_phase="SCORE_AND_WRITE_CORE",
        next_phase="PUBLISH_MANIFEST_SIDECAR_SEAL",
        entry_name="engine.score_and_write_core",
    )
    if current.predictions_complete is None or set(current.truth_by_fold) != set(
        guard.FOLD_ORDER
    ):
        raise R2ExecutionError("scoring requires predictions-complete and all released truth")
    complete_path = current.output / "commitments/predictions_complete.json"
    if guard.sha256_file(complete_path) != current.predictions_complete["file"]["sha256"]:
        raise R2ExecutionError("predictions-complete changed before scoring")
    durable = _durable_blind_surface(current)
    source_truth = pd.concat(
        [current.truth_by_fold[name] for name in guard.FOLD_ORDER], ignore_index=True
    )
    predictors = current.predictors.copy()
    evaluated = predictors.merge(
        source_truth,
        on=list(TARGET_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    ).merge(durable, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if evaluated[["target_hs", science.CANDIDATE_COLUMN]].isna().any().any():
        raise R2ExecutionError("scoring truth or blind prediction attachment is incomplete")

    sealed_truth = pd.read_parquet(
        current.root / current.config["canonical_paths"]["sealed_gen1_oof"],
        columns=[*KEY_COLUMNS, "target_hs"],
    )
    left = evaluated.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    right = sealed_truth.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if not left[list(KEY_COLUMNS)].equals(right[list(KEY_COLUMNS)]) or not np.array_equal(
        left["target_hs"].to_numpy(dtype=np.float64),
        right["target_hs"].to_numpy(dtype=np.float64),
    ):
        raise R2ExecutionError("selective source truth differs from sealed Gen1 OOF truth")

    points: dict[float, dict[str, Any]] = {}
    for prefix in PREFIX_FRACTIONS:
        frame = evaluated.loc[evaluated["prefix_fraction"].eq(prefix)].copy()
        points[prefix] = evaluate_point(
            frame,
            candidate_column=science.CANDIDATE_COLUMN,
            bootstrap_replicates=5000,
            bootstrap_seed=20260823 + int(round(prefix * 1000)),
        )
    leakage_checks = {
        "sealed_corrected_validation_surface_reused": True,
        "key_input_only_loaded_before_target_scalar_decode": True,
        "fold_major_blind_o_excl_commitment_precedes_own_target_decode": True,
        "prior_fold_truth_only_for_later_fold_calibration": True,
        "predictions_complete_precedes_all_scoring_truth_attachment": True,
        "same_prefix_only_calibration": True,
        "anonymous_test_value_reads_zero": True,
    }
    reproducibility_checks = {
        "v1_science_deep_equal": True,
        "single_preregistered_hypothesis": True,
        "fixed_ridge_alpha_threshold_bound_and_zero_search": True,
        "failed_inner_gate_exact_identity_bytes": all(
            bool(receipt.get("identity_bytes_equal"))
            for receipt in current.receipts
            if "IDENTITY" in str(receipt["decision"])
        ),
        "all_blind_predictions_reread_exact": True,
        "source_truth_exact_to_sealed_gen1_oof_after_predictions_complete": True,
        "fit_count_is_observed_receipt_sum_and_at_most_20": True,
        "candidate_test_registry_and_upload_zero": True,
    }
    gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=leakage_checks,
        reproducibility_checks=reproducibility_checks,
    )
    evidence = central_evidence(
        points,
        leakage_checks=leakage_checks,
        reproducibility_checks=reproducibility_checks,
    )
    evidence.update(
        {
            "comparison_mode": "SEALED_GEN1_OOF_INCUMBENT_PRESERVING_RESEARCH_ONLY",
            "local_numeric_gate": gate,
            "official_promotion": {
                "allowed": False,
                "reason": "SEALED_GEN1_OOF_IS_NOT_AN_EXACT_OFFICIAL_PAIRED_AB",
            },
            "preregistration": {
                "hypothesis_count": 1,
                "science_deep_sha256": guard.EXPECTED_SCIENCE_DEEP_SHA256,
                "alpha_threshold_seed_or_weight_search_count": 0,
            },
        }
    )
    observed_fits = int(sum(receipt["fit_count"] for receipt in current.receipts))
    if observed_fits > 20:
        raise R2ExecutionError("observed fit count exceeds the authorized upper bound")
    status = (
        "LOCAL_CURVE_QUALIFIED_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
        if gate["passed"]
        else "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
    )
    metrics = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.metrics.r2.v1"
        ),
        "experiment_id": guard.EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": status,
        "points": {str(key): value for key, value in points.items()},
        "local_gate": gate,
        "inner_gate_receipts": current.receipts,
        "fit_count_contract": {
            "authorized_upper_bound": 20,
            "observed_exact": observed_fits,
            "observed_equals_cell_receipt_sum": True,
        },
        "target_access_audit": current.vault.access_audit(),
        "predictions_complete": current.predictions_complete["file"],
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "elapsed_seconds": float(time.perf_counter() - current.started),
        "full_fit_performed": False,
        "candidate_created": False,
        "test_prediction_created": False,
        "official_promotion_allowed": False,
        "registry_append_allowed": False,
        "official_upload_count": 0,
    }
    models = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.models.r2.v1"
        ),
        "experiment_id": guard.EXPERIMENT_ID,
        "science_deep_sha256": guard.EXPECTED_SCIENCE_DEEP_SHA256,
        "records": current.model_receipts,
        "fit_count_observed_exact": observed_fits,
        "fit_count_authorized_upper_bound": 20,
    }
    phase = "PUBLISH_MANIFEST_SIDECAR_SEAL"
    guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="calibrator_models.json",
        payload=guard.canonical_json_bytes(models) + b"\n",
    )
    guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="learning_curve_evidence.json",
        payload=guard.canonical_json_bytes(evidence) + b"\n",
    )
    guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="metrics.json",
        payload=guard.canonical_json_bytes(metrics) + b"\n",
    )
    ordered_columns = [
        *KEY_COLUMNS,
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "target_hs",
        science.CANDIDATE_COLUMN,
    ]
    evaluated = evaluated.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="oof.parquet",
        payload=_parquet_bytes(evaluated[ordered_columns]),
    )
    current.metrics = metrics
    current.evidence = evidence
    return {
        "status": status,
        "fit_count_observed_exact": observed_fits,
        "local_gate_passed": bool(gate["passed"]),
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


def publish_manifest_sidecar_seal(capability: object, state: object) -> dict[str, Any]:
    current = _require_state(capability, state)
    if current.metrics is None or current.evidence is None:
        raise R2ExecutionError("publish requires scored core outputs")
    guard.enter_engine_phase(
        capability,
        expected_phase="PUBLISH_MANIFEST_SIDECAR_SEAL",
        next_phase="PUBLISH_IN_PROGRESS",
        entry_name="engine.publish_manifest_sidecar_seal",
    )
    context = guard.capability_context(capability)
    lineage = context["static_lineage"]
    config_raw = (current.root / guard.CONFIG_RELATIVE).read_bytes()
    lock_path = current.root / current.config["canonical_paths"]["attempt_lock"]
    lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw)
    core = {
        relative: guard.file_pin(current.output / relative, root=current.output)
        for relative in guard.CORE_FILES
    }
    manifest_lineage = {
        "config": {
            "path": guard.CONFIG_RELATIVE,
            "bytes": len(config_raw),
            "sha256": guard.sha256_file(current.root / guard.CONFIG_RELATIVE),
            "deep_sha256": guard.deep_sha256(current.config),
        },
        "science": lineage["science"],
        "superseded_v1": lineage["superseded_v1"],
        "qa_sha256": context["qa_sha256"],
        "authorization_sha256": context["authorization_sha256"],
        "attempt_lock": {
            "path": current.config["canonical_paths"]["attempt_lock"],
            "bytes": len(lock_raw),
            "sha256": hashlib.sha256(lock_raw).hexdigest(),
            "deep_sha256": guard.deep_sha256(lock),
        },
        "implementation_pins": lineage["implementation_pins"],
        "immutable_inputs": lineage["immutable_inputs"],
        "runtime": lineage["runtime"],
        "central_v9_anchor": lineage["central_v9_anchor"],
    }
    manifest = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.manifest.r2.v1"
        ),
        "experiment_id": guard.EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": current.metrics["status"],
        "lineage": manifest_lineage,
        "core_files": core,
        "fit_count_authorized_upper_bound": 20,
        "fit_count_observed_exact": current.metrics["fit_count_contract"]["observed_exact"],
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "official_upload_count": 0,
    }
    phase = "PUBLISH_IN_PROGRESS"
    manifest_pin = guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="manifest.json",
        payload=guard.canonical_json_bytes(manifest) + b"\n",
    )
    sidecar_pin = guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="manifest.sha256",
        payload=f"{manifest_pin['sha256']}  manifest.json\n".encode("ascii"),
    )
    seal_lineage = {
        "qa_sha256": context["qa_sha256"],
        "authorization_sha256": context["authorization_sha256"],
        "attempt_lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "implementation_pins": lineage["implementation_pins"],
        "immutable_inputs": lineage["immutable_inputs"],
        "runtime": lineage["runtime"],
        "central_v9_anchor": lineage["central_v9_anchor"],
    }
    seal = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.output_seal.r2.v1"
        ),
        "experiment_id": guard.EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "SEALED_RESEARCH_ONLY",
        "lineage": seal_lineage,
        "manifest": manifest_pin,
        "manifest_sidecar": sidecar_pin,
        "allowed_directories": list(guard.ALLOWED_DIRECTORIES),
        "allowed_files": list(guard.ALLOWED_FILES),
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }
    seal_pin = guard.write_output_exclusive(
        capability,
        phase=phase,
        relative_path="seal.json",
        payload=guard.canonical_json_bytes(seal) + b"\n",
    )
    guard.revoke_capability(capability, expected_phase="PUBLISH_IN_PROGRESS")
    with _STATE_LOCK:
        _STATE_REGISTRY.pop(id(capability), None)
    return {
        "status": current.metrics["status"],
        "manifest": manifest_pin,
        "manifest_sidecar": sidecar_pin,
        "seal": seal_pin,
        "fit_count_observed_exact": current.metrics["fit_count_contract"]["observed_exact"],
        "capability_revoked": True,
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


__all__ = [
    "KEY_COLUMNS",
    "LEADS",
    "PREDICTOR_COLUMNS",
    "R2ExecutionError",
    "SelectiveOfficialTargetVault",
    "commit_predictions_complete",
    "load_key_input_only",
    "predict_and_commit_fold",
    "publish_manifest_sidecar_seal",
    "release_committed_fold_truth",
    "score_and_write_core",
]
