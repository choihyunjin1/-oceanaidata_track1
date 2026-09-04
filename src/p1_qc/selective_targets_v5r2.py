"""Selective target firewall for P1 Gen5r2 rolling-origin evaluation.

The input frame is loaded without target columns.  Target CSV fields are
indexed as opaque byte spans and decoded only for explicitly authorized row
positions.  A row that belongs to an outer-validation fold remains withheld
until all five blind cells for that fold have immutable commitments.  This
allows a completed earlier fold to become legitimate training history for a
later rolling-origin fold without exposing any active fold's targets.
"""

from __future__ import annotations

import csv
import hashlib
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from p1_qc.data import BASE_COLUMNS, TRAIN_COLUMNS

KEY_COLUMNS = ("station", "year", "layer", "time")
TARGET_COLUMNS = ("label", "anomaly_type")


class SelectiveTargetError(ValueError):
    """Raised when the target firewall or pinned CSV layout differs."""


class CommitmentView(Protocol):
    def is_fold_committed(self, fold: str) -> bool: ...

    def is_global_committed(self) -> bool: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_field_spans(
    raw_line: bytes,
    *,
    expected_fields: int,
) -> tuple[bytes, tuple[tuple[int, int], ...]]:
    """Locate one RFC-4180 record's fields without decoding field contents."""

    line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.rstrip(b"\n")
    if b"\r" in line or b"\n" in line:
        raise SelectiveTargetError("P1 CSV contains an unsupported embedded newline")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    in_quotes = False
    while index < len(line):
        value = line[index]
        if value == 34:
            if in_quotes and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            in_quotes = not in_quotes
        elif value == 44 and not in_quotes:
            spans.append((start, index))
            start = index + 1
        index += 1
    if in_quotes:
        raise SelectiveTargetError("P1 CSV contains an unterminated quoted field")
    spans.append((start, len(line)))
    if len(spans) != int(expected_fields):
        raise SelectiveTargetError("P1 CSV row width changed")
    return line, tuple(spans)


def decode_csv_field(raw_line: bytes, span: tuple[int, int]) -> str:
    """Decode exactly one selected field and no adjacent target scalar."""

    start, stop = span
    token = raw_line[start:stop].decode("utf-8")
    if token == "":
        return ""
    parsed = next(csv.reader([token], strict=True), None)
    if parsed is None or len(parsed) != 1:
        raise SelectiveTargetError("selected P1 CSV field did not decode to one scalar")
    return parsed[0]


def _decode_required_integer(raw_line: bytes, span: tuple[int, int], *, column: str) -> int:
    value = decode_csv_field(raw_line, span)
    try:
        return int(value)
    except ValueError as exc:
        raise SelectiveTargetError(f"P1 {column} is not an integer") from exc


def _decode_optional_float(raw_line: bytes, span: tuple[int, int], *, column: str) -> float:
    value = decode_csv_field(raw_line, span)
    if value == "" or value.casefold() in {"na", "nan", "null", "none"}:
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise SelectiveTargetError(f"P1 {column} is not numeric") from exc


def load_input_only_train(path: Path) -> pd.DataFrame:
    """Load base fields from bytes without slicing or decoding either target field."""

    source = path.expanduser().resolve(strict=True)
    before = source.stat()
    station: list[str] = []
    year = array("q")
    layer = array("q")
    time_values: list[str] = []
    temp = array("d")
    psal = array("d")
    depth = array("d")
    with source.open("rb") as stream:
        raw_header = stream.readline()
        if not raw_header:
            raise SelectiveTargetError("train.csv is empty")
        header_line, header_spans = csv_field_spans(
            raw_header,
            expected_fields=len(TRAIN_COLUMNS),
        )
        header = tuple(decode_csv_field(header_line, span) for span in header_spans)
        if header != TRAIN_COLUMNS:
            raise SelectiveTargetError("train.csv schema or column order changed")
        while raw_row := stream.readline():
            row_line, spans = csv_field_spans(
                raw_row,
                expected_fields=len(TRAIN_COLUMNS),
            )
            station.append(decode_csv_field(row_line, spans[0]))
            year.append(_decode_required_integer(row_line, spans[1], column="year"))
            layer.append(_decode_required_integer(row_line, spans[2], column="layer"))
            time_values.append(decode_csv_field(row_line, spans[3]))
            temp.append(_decode_optional_float(row_line, spans[4], column="temp"))
            psal.append(_decode_optional_float(row_line, spans[5], column="psal"))
            depth.append(_decode_optional_float(row_line, spans[6], column="depth"))
    after = source.stat()
    before_signature = (before.st_size, before.st_mtime_ns)
    after_signature = (after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"source file changed while it was being read: {source}")
    frame = pd.DataFrame(
        {
            "station": station,
            "year": np.asarray(year, dtype=np.int64),
            "layer": np.asarray(layer, dtype=np.int64),
            "time": time_values,
            "temp": np.asarray(temp, dtype=np.float64),
            "psal": np.asarray(psal, dtype=np.float64),
            "depth": np.asarray(depth, dtype=np.float64),
        },
        columns=list(BASE_COLUMNS),
    )
    if tuple(frame.columns) != BASE_COLUMNS:
        raise SelectiveTargetError("input-only P1 column order differs")
    if set(TARGET_COLUMNS).intersection(frame.columns):
        raise PermissionError("input-only P1 loader materialized a target column")
    frame.attrs.update(
        {
            "source_path": str(source),
            "source_size": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "input_fields_decoded": len(frame) * len(BASE_COLUMNS),
            "target_fields_decoded": 0,
        }
    )
    return frame


def load_frozen_oof_keys_only(path: Path) -> pd.DataFrame:
    """Project frozen OOF to routing fields without decoding any target column."""

    columns = [*KEY_COLUMNS, "fold", "prediction"]
    frame = pd.read_parquet(path, columns=columns)
    if list(frame.columns) != columns:
        raise SelectiveTargetError("frozen OOF key projection differs")
    if {"label", "anomaly_type"}.intersection(frame.columns):
        raise PermissionError("frozen OOF target column was decoded")
    return frame


@dataclass(frozen=True)
class TargetDecodeEvent:
    purpose: str
    fold: str | None
    row_count: int
    row_ids_sha256: str
    decoded_columns: tuple[str, ...]


class SelectiveTargetAccessor:
    """Opaque-row index plus guarded scalar decoder for train.csv targets."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_rows: int,
        validation_rows_by_fold: Mapping[str, Sequence[int] | np.ndarray],
        fold_order: Sequence[str],
    ) -> None:
        self._path = path.resolve(strict=True)
        if sha256_file(self._path) != expected_sha256:
            raise SelectiveTargetError("pinned train.csv SHA-256 changed")
        self._fold_order = tuple(str(value) for value in fold_order)
        if tuple(validation_rows_by_fold) != self._fold_order:
            raise SelectiveTargetError("outer-validation fold order differs")
        self._validation_rows_by_fold = {
            fold: self._strict_ids(values, size=int(expected_rows), role=f"{fold} validation")
            for fold, values in validation_rows_by_fold.items()
        }
        combined = np.concatenate(list(self._validation_rows_by_fold.values()))
        if np.unique(combined).size != len(combined):
            raise SelectiveTargetError("outer-validation row positions overlap across folds")
        self._row_to_fold = np.full(int(expected_rows), -1, dtype=np.int8)
        for ordinal, fold in enumerate(self._fold_order):
            self._row_to_fold[self._validation_rows_by_fold[fold]] = ordinal
        self._offsets = array("Q")
        self._lengths = array("I")
        self._header: tuple[str, ...]
        self._column_index: dict[str, int]
        self._build_opaque_index(expected_rows=int(expected_rows))
        self._label_cache: dict[int, int] = {}
        self._anomaly_cache: dict[int, str] = {}
        self._events: list[TargetDecodeEvent] = []
        self._decoded_target_scalars = 0

    @staticmethod
    def _strict_ids(
        values: Sequence[int] | np.ndarray,
        *,
        size: int,
        role: str,
    ) -> np.ndarray:
        result = np.asarray(values)
        if result.ndim != 1:
            raise SelectiveTargetError(f"{role} IDs must be a vector")
        if not np.issubdtype(result.dtype, np.integer):
            raise SelectiveTargetError(f"{role} IDs must be integers")
        result = result.astype(np.int64, copy=False)
        if len(result) and (result.min() < 0 or result.max() >= size):
            raise SelectiveTargetError(f"{role} IDs are outside train.csv")
        if np.unique(result).size != len(result):
            raise SelectiveTargetError(f"{role} IDs must be unique")
        return result

    def _build_opaque_index(self, *, expected_rows: int) -> None:
        with self._path.open("rb") as stream:
            raw_header = stream.readline()
            if not raw_header:
                raise SelectiveTargetError("train.csv is empty")
            header_line, header_spans = csv_field_spans(
                raw_header,
                expected_fields=len(TRAIN_COLUMNS),
            )
            self._header = tuple(decode_csv_field(header_line, span) for span in header_spans)
            if self._header != TRAIN_COLUMNS:
                raise SelectiveTargetError("train.csv schema or column order changed")
            self._column_index = {name: index for index, name in enumerate(self._header)}
            while True:
                offset = stream.tell()
                raw_row = stream.readline()
                if not raw_row:
                    break
                csv_field_spans(raw_row, expected_fields=len(self._header))
                self._offsets.append(offset)
                self._lengths.append(len(raw_row))
        if len(self._offsets) != expected_rows:
            raise SelectiveTargetError("train.csv indexed row count differs")

    @property
    def row_count(self) -> int:
        return len(self._offsets)

    @property
    def decoded_label_rows(self) -> int:
        return len(self._label_cache)

    @property
    def decoded_anomaly_rows(self) -> int:
        return len(self._anomaly_cache)

    @property
    def decoded_target_scalars(self) -> int:
        return self._decoded_target_scalars

    @property
    def events(self) -> tuple[TargetDecodeEvent, ...]:
        return tuple(self._events)

    def validation_rows(self, fold: str) -> np.ndarray:
        return self._validation_rows_by_fold[fold].copy()

    def validation_target_decode_counts(self, fold: str) -> dict[str, int]:
        validation = self._validation_rows_by_fold[fold]
        label_ids = np.fromiter(self._label_cache, dtype=np.int64)
        anomaly_ids = np.fromiter(self._anomaly_cache, dtype=np.int64)
        return {
            "label": int(np.intersect1d(validation, label_ids).size),
            "anomaly_type": int(np.intersect1d(validation, anomaly_ids).size),
        }

    def _authorize_rows(
        self,
        ids: np.ndarray,
        *,
        commitment: CommitmentView,
        purpose: str,
        active_fold: str | None,
        require_global: bool,
    ) -> None:
        if require_global and not commitment.is_global_committed():
            raise PermissionError(f"{purpose} requires the global predictions commitment")
        if active_fold is not None:
            if active_fold not in self._validation_rows_by_fold:
                raise SelectiveTargetError("active fold is not registered")
            if np.intersect1d(ids, self._validation_rows_by_fold[active_fold]).size:
                raise PermissionError("active outer-validation target decode is forbidden")
        if commitment.is_global_committed():
            return
        fold_ordinals = np.unique(self._row_to_fold[ids])
        for ordinal in fold_ordinals:
            if ordinal < 0:
                continue
            fold = self._fold_order[int(ordinal)]
            if not commitment.is_fold_committed(fold):
                raise PermissionError(
                    f"outer-validation targets for {fold} remain withheld until its 5/5 commitments"
                )

    def _decode_selected(self, ids: np.ndarray, column: str) -> dict[int, str]:
        index = self._column_index[column]
        result: dict[int, str] = {}
        with self._path.open("rb") as stream:
            for row_id in ids.tolist():
                stream.seek(int(self._offsets[row_id]))
                raw_row = stream.read(int(self._lengths[row_id]))
                row_line, spans = csv_field_spans(
                    raw_row,
                    expected_fields=len(self._header),
                )
                result[row_id] = decode_csv_field(row_line, spans[index])
        self._decoded_target_scalars += len(result)
        return result

    @staticmethod
    def _ids_sha(ids: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes(order="C")).hexdigest()

    def labels_for(
        self,
        row_ids: Sequence[int] | np.ndarray,
        *,
        commitment: CommitmentView,
        purpose: str,
        active_fold: str | None,
        require_global: bool = False,
    ) -> np.ndarray:
        ids = self._strict_ids(row_ids, size=self.row_count, role=purpose)
        self._authorize_rows(
            ids,
            commitment=commitment,
            purpose=purpose,
            active_fold=active_fold,
            require_global=require_global,
        )
        missing = np.asarray(
            [row_id for row_id in ids.tolist() if row_id not in self._label_cache],
            dtype=np.int64,
        )
        if len(missing):
            decoded = self._decode_selected(missing, "label")
            for row_id, value in decoded.items():
                if value not in {"0", "1"}:
                    raise SelectiveTargetError("selected P1 label is not binary")
                self._label_cache[row_id] = int(value)
            self._events.append(
                TargetDecodeEvent(
                    purpose=purpose,
                    fold=active_fold,
                    row_count=len(missing),
                    row_ids_sha256=self._ids_sha(missing),
                    decoded_columns=("label",),
                )
            )
        return np.asarray([self._label_cache[row_id] for row_id in ids], dtype=np.int8)

    def anomaly_types_for(
        self,
        row_ids: Sequence[int] | np.ndarray,
        *,
        commitment: CommitmentView,
        purpose: str,
    ) -> np.ndarray:
        ids = self._strict_ids(row_ids, size=self.row_count, role=purpose)
        self._authorize_rows(
            ids,
            commitment=commitment,
            purpose=purpose,
            active_fold=None,
            require_global=True,
        )
        missing = np.asarray(
            [row_id for row_id in ids.tolist() if row_id not in self._anomaly_cache],
            dtype=np.int64,
        )
        if len(missing):
            self._anomaly_cache.update(self._decode_selected(missing, "anomaly_type"))
            self._events.append(
                TargetDecodeEvent(
                    purpose=purpose,
                    fold=None,
                    row_count=len(missing),
                    row_ids_sha256=self._ids_sha(missing),
                    decoded_columns=("anomaly_type",),
                )
            )
        return np.asarray([self._anomaly_cache[row_id] for row_id in ids], dtype=object)

    def audit(self) -> dict[str, Any]:
        return {
            "source_path": self._path.name,
            "source_rows": self.row_count,
            "input_loader_target_columns": 0,
            "opaque_index_target_fields_decoded": 0,
            "decoded_label_rows": self.decoded_label_rows,
            "decoded_anomaly_type_rows": self.decoded_anomaly_rows,
            "decoded_target_scalars": self.decoded_target_scalars,
            "decode_events": [event.__dict__ for event in self._events],
        }


__all__ = [
    "CommitmentView",
    "KEY_COLUMNS",
    "SelectiveTargetAccessor",
    "SelectiveTargetError",
    "TARGET_COLUMNS",
    "TargetDecodeEvent",
    "csv_field_spans",
    "decode_csv_field",
    "load_frozen_oof_keys_only",
    "load_input_only_train",
    "sha256_file",
]
