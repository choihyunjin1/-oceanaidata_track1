"""Selective train-only dense 72-step target access for P3 Gen5r2.

The accessor separates target *identity and availability* from target scalar
decoding.  Static preflight may index ``station,time`` and whether the ``hs``
field is present, but a floating-point target is decoded only for an explicitly
authorized training anchor.  Target rows belonging to a not-yet-committed
validation fold remain quarantined even when another anchor would address the
same source row.

No anonymous-test file is accepted by this module.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DENSE_TARGET_STEPS = 72
DENSE_TARGET_CADENCE_MINUTES = 20
OFFICIAL_DENSE_INDICES = (8, 17, 26, 35, 53, 71)
EXPECTED_ANCHOR_COUNT = 24_360
EXPECTED_COMPLETE_CASES = 23_527
EXPECTED_INCOMPLETE_CASES = 833
EXPECTED_MISSING_SCALARS = 1_505
EXPECTED_SOURCE_ROWS = 118_152
SOURCE_HEADER = (b"station", b"time", b"hs", b"tp", b"hmax", b"wvdir")
STATION_TO_CODE = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
_TIME_PATTERN = re.compile(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00")
_MISSING_TOKENS = frozenset({b"", b"nan", b"NaN", b"NA", b"null", b"None"})
_STEP_NS = DENSE_TARGET_CADENCE_MINUTES * 60 * 1_000_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(values: Sequence[int] | np.ndarray, *, size: int, role: str) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError(f"{role} IDs must be a non-empty vector")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{role} IDs must be integers")
    result = source.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} IDs must be unique")
    if result.min() < 0 or result.max() >= size:
        raise IndexError(f"{role} IDs are outside the anchor table")
    return result


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Dense72AvailabilityAudit:
    anchor_count: int
    dense_steps: int
    complete_cases: int
    incomplete_cases: int
    missing_scalars: int
    official_six_missing_scalars: int
    mask_sha256: str
    scalar_decodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_count": self.anchor_count,
            "dense_steps": self.dense_steps,
            "complete_cases": self.complete_cases,
            "incomplete_cases": self.incomplete_cases,
            "missing_scalars": self.missing_scalars,
            "official_six_missing_scalars": self.official_six_missing_scalars,
            "mask_sha256": self.mask_sha256,
            "scalar_decodes": self.scalar_decodes,
        }


@dataclass(frozen=True)
class Dense72TrainingPayload:
    case_ids: np.ndarray
    target_delta: np.ndarray
    target_mask: np.ndarray
    current_hs: np.ndarray
    decoded_scalar_count: int
    case_ids_sha256: str
    target_delta_sha256: str
    target_mask_sha256: str
    forbidden_scalar_decodes: int


class Dense72TargetAccessor:
    """Index one immutable P3 ``train_wave.csv`` without decoding target values.

    ``anchors`` must be the canonical 24,360-row train anchor table loaded with
    input-only columns.  ``validation_groups`` registers every outer validation
    case.  A group can be released only after its fold commitment is persisted;
    until then its 72 future source rows are forbidden to scalar decoding.
    """

    def __init__(
        self,
        wave_path: str | Path,
        anchors: pd.DataFrame,
        *,
        validation_groups: Mapping[str, Sequence[int] | np.ndarray],
        expected_source_sha256: str,
        expected_source_bytes: int,
        expected_source_rows: int = EXPECTED_SOURCE_ROWS,
        enforce_canonical_aggregate: bool = True,
    ) -> None:
        self.wave_path = Path(wave_path).resolve(strict=True)
        if self.wave_path.name != "train_wave.csv":
            raise PermissionError("dense target accessor accepts only train_wave.csv")
        if self.wave_path.stat().st_size != int(expected_source_bytes):
            raise PermissionError("train_wave byte count differs")
        if sha256_file(self.wave_path) != str(expected_source_sha256):
            raise PermissionError("train_wave SHA-256 differs")

        required = {"anchor_id", "station", "anchor_time", "current_hs"}
        if set(anchors.columns) != required:
            raise ValueError("anchor accessor surface must contain input-only columns exactly")
        ordered = anchors.sort_values("anchor_id").reset_index(drop=True).copy()
        if len(ordered) == 0 or ordered["anchor_id"].duplicated().any():
            raise ValueError("anchor table is empty or duplicated")
        expected_ids = np.arange(len(ordered), dtype=np.int64)
        if not np.array_equal(ordered["anchor_id"].to_numpy(np.int64), expected_ids):
            raise ValueError("anchor IDs must be a contiguous sequence-row identity")
        station = ordered["station"].map(STATION_TO_CODE)
        if station.isna().any():
            raise ValueError("anchor station lies outside the official set")
        anchor_time = pd.DatetimeIndex(
            pd.to_datetime(ordered["anchor_time"], utc=True, errors="raise")
        ).as_unit("ns")
        current_hs = ordered["current_hs"].to_numpy(np.float64)
        if not np.isfinite(current_hs).all():
            raise ValueError("anchor current_hs must be finite")
        self._anchor_station = station.to_numpy(np.int8)
        self._anchor_time_ns = anchor_time.asi8.astype(np.int64, copy=False)
        self._current_hs = current_hs
        self._anchor_count = len(ordered)

        identities = pd.read_csv(
            self.wave_path,
            usecols=["station", "time"],
            dtype={"station": "string", "time": "string"},
        )
        if len(identities) != int(expected_source_rows):
            raise ValueError("train_wave identity row count differs")
        encoded_station = identities["station"].map(STATION_TO_CODE)
        if encoded_station.isna().any():
            raise ValueError("train_wave station lies outside the official set")
        source_time = pd.DatetimeIndex(
            pd.to_datetime(identities["time"], utc=True, errors="raise")
        ).as_unit("ns")
        source_station = encoded_station.to_numpy(np.int8)
        source_time_ns = source_time.asi8.astype(np.int64, copy=False)
        identity_station = identities["station"].astype(str).to_numpy()
        identity_time = identities["time"].astype(str).to_numpy()

        offsets = np.empty(len(identities), dtype=np.int64)
        available = np.empty(len(identities), dtype=bool)
        with self.wave_path.open("rb") as stream:
            header = tuple(stream.readline().rstrip(b"\r\n").split(b","))
            if header != SOURCE_HEADER:
                raise ValueError("train_wave header differs from the pinned six-field schema")
            row = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                fields = line.rstrip(b"\r\n").split(b",")
                if len(fields) != len(SOURCE_HEADER):
                    raise ValueError("train_wave contains a non-six-field row")
                if row >= len(identities):
                    raise ValueError("train_wave binary rows exceed identity rows")
                if fields[0].decode("ascii") != identity_station[row]:
                    raise ValueError("binary and input-only station identities differ")
                if not _TIME_PATTERN.fullmatch(fields[1]):
                    raise ValueError("train_wave timestamp token differs from the fixed ISO form")
                if fields[1].decode("ascii") != identity_time[row]:
                    raise ValueError("binary and input-only time identities differ")
                offsets[row] = int(offset)
                available[row] = fields[2] not in _MISSING_TOKENS
                row += 1
        if row != len(identities):
            raise ValueError("train_wave binary and identity row counts differ")

        self._source_offsets = offsets
        self._source_available = available
        self._station_times: dict[int, np.ndarray] = {}
        self._station_rows: dict[int, np.ndarray] = {}
        for code in sorted(STATION_TO_CODE.values()):
            rows = np.flatnonzero(source_station == code).astype(np.int64)
            order = np.argsort(source_time_ns[rows], kind="stable")
            rows = rows[order]
            times = source_time_ns[rows]
            if len(times) == 0 or np.any(np.diff(times) <= 0):
                raise ValueError("train_wave station times must be strictly increasing")
            self._station_times[code] = times
            self._station_rows[code] = rows

        self._validation_case_ids: dict[str, np.ndarray] = {}
        self._validation_target_rows: dict[str, frozenset[int]] = {}
        observed_validation_ids: list[np.ndarray] = []
        observed_target_rows: set[int] = set()
        for name, values in validation_groups.items():
            label = str(name)
            if not label or label in self._validation_case_ids:
                raise ValueError("validation group names must be unique and non-empty")
            ids = _ids(values, size=self._anchor_count, role=f"validation group {label}")
            rows, _mask = self._locate(ids)
            present_rows = frozenset(int(value) for value in rows[rows >= 0])
            if observed_target_rows.intersection(present_rows):
                raise PermissionError("validation dense-target source rows overlap across folds")
            observed_target_rows.update(present_rows)
            observed_validation_ids.append(ids)
            self._validation_case_ids[label] = ids
            self._validation_target_rows[label] = present_rows
        if not observed_validation_ids:
            raise ValueError("at least one validation group must be registered")
        merged_validation = np.concatenate(observed_validation_ids)
        if np.unique(merged_validation).size != len(merged_validation):
            raise PermissionError("validation anchor IDs overlap across folds")
        self._released_groups: set[str] = set()
        self._release_commitments: dict[str, str] = {}
        self._decoded_rows: dict[int, float] = {}
        self._total_scalar_decodes = 0
        self._forbidden_scalar_decodes = 0

        audit = self.availability_audit()
        if enforce_canonical_aggregate:
            expected = (
                EXPECTED_ANCHOR_COUNT,
                EXPECTED_COMPLETE_CASES,
                EXPECTED_INCOMPLETE_CASES,
                EXPECTED_MISSING_SCALARS,
                0,
            )
            observed = (
                audit.anchor_count,
                audit.complete_cases,
                audit.incomplete_cases,
                audit.missing_scalars,
                audit.official_six_missing_scalars,
            )
            if observed != expected:
                raise ValueError("canonical dense72 availability aggregate differs")

    @property
    def total_scalar_decodes(self) -> int:
        return int(self._total_scalar_decodes)

    @property
    def forbidden_scalar_decodes(self) -> int:
        return int(self._forbidden_scalar_decodes)

    @property
    def released_groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._released_groups))

    def _locate(self, case_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(case_ids, dtype=np.int64)
        target_time = self._anchor_time_ns[ids, None] + _STEP_NS * np.arange(
            1, DENSE_TARGET_STEPS + 1, dtype=np.int64
        )[None, :]
        rows = np.full(target_time.shape, -1, dtype=np.int64)
        for code in sorted(STATION_TO_CODE.values()):
            local = np.flatnonzero(self._anchor_station[ids] == code)
            if len(local) == 0:
                continue
            times = self._station_times[code]
            station_rows = self._station_rows[code]
            query = target_time[local].reshape(-1)
            position = np.searchsorted(times, query)
            safe = np.minimum(position, len(times) - 1)
            found = (position < len(times)) & (times[safe] == query)
            located = np.full(len(query), -1, dtype=np.int64)
            located[found] = station_rows[position[found]]
            rows[local] = located.reshape(len(local), DENSE_TARGET_STEPS)
        mask = rows >= 0
        valid = rows[mask]
        mask[mask] = self._source_available[valid]
        return rows, mask

    def availability_audit(self) -> Dense72AvailabilityAudit:
        ids = np.arange(self._anchor_count, dtype=np.int64)
        _rows, mask = self._locate(ids)
        complete = mask.all(axis=1)
        return Dense72AvailabilityAudit(
            anchor_count=int(self._anchor_count),
            dense_steps=DENSE_TARGET_STEPS,
            complete_cases=int(complete.sum()),
            incomplete_cases=int((~complete).sum()),
            missing_scalars=int(mask.size - mask.sum()),
            official_six_missing_scalars=int((~mask[:, OFFICIAL_DENSE_INDICES]).sum()),
            mask_sha256=_array_sha256(mask, dtype="|b1"),
            scalar_decodes=int(self._total_scalar_decodes),
        )

    def release_validation_group(self, name: str, *, fold_commitment_sha256: str) -> None:
        label = str(name)
        if label not in self._validation_case_ids:
            raise KeyError("unknown validation group")
        if label in self._released_groups:
            raise PermissionError("validation group was already released")
        digest = str(fold_commitment_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("fold commitment must be a lowercase SHA-256")
        self._released_groups.add(label)
        self._release_commitments[label] = digest

    def _unreleased_target_rows(self) -> set[int]:
        result: set[int] = set()
        for name, rows in self._validation_target_rows.items():
            if name not in self._released_groups:
                result.update(rows)
        return result

    def _decode_rows(self, rows: np.ndarray) -> dict[int, float]:
        requested = sorted(set(int(value) for value in rows))
        missing = [row for row in requested if row not in self._decoded_rows]
        if not missing:
            return {row: self._decoded_rows[row] for row in requested}
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = os.open(self.wave_path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                for row in missing:
                    stream.seek(int(self._source_offsets[row]))
                    fields = stream.readline().rstrip(b"\r\n").split(b",")
                    if len(fields) != len(SOURCE_HEADER) or fields[2] in _MISSING_TOKENS:
                        raise ValueError("selected dense target row became missing or malformed")
                    value = float(fields[2].decode("ascii"))
                    if not np.isfinite(value):
                        raise ValueError("selected dense target scalar is non-finite")
                    self._decoded_rows[row] = value
                    self._total_scalar_decodes += 1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return {row: self._decoded_rows[row] for row in requested}

    def load_training_targets(
        self,
        train_case_ids: Sequence[int] | np.ndarray,
        *,
        active_validation_case_ids: Sequence[int] | np.ndarray,
    ) -> Dense72TrainingPayload:
        train_ids = _ids(train_case_ids, size=self._anchor_count, role="training")
        validation_ids = _ids(
            active_validation_case_ids,
            size=self._anchor_count,
            role="active validation",
        )
        if np.intersect1d(train_ids, validation_ids).size:
            raise PermissionError("training and active validation anchor IDs overlap")
        registered = [
            name
            for name, values in self._validation_case_ids.items()
            if np.array_equal(np.sort(values), np.sort(validation_ids))
        ]
        if len(registered) != 1 or registered[0] in self._released_groups:
            raise PermissionError("active validation group is unregistered or already released")

        rows, mask = self._locate(train_ids)
        selected_rows = rows[mask]
        forbidden_rows = self._unreleased_target_rows()
        overlap = forbidden_rows.intersection(int(value) for value in selected_rows)
        if overlap:
            self._forbidden_scalar_decodes += 0
            raise PermissionError("training target request intersects unreleased validation rows")
        values_by_row = self._decode_rows(selected_rows)
        target = np.zeros(rows.shape, dtype=np.float32)
        for row_index, source_row in zip(np.argwhere(mask), selected_rows, strict=True):
            target[tuple(row_index)] = np.float32(values_by_row[int(source_row)])
        current = self._current_hs[train_ids].astype(np.float32, copy=True)
        delta = target - current[:, None]
        delta[~mask] = 0.0
        if not mask[:, OFFICIAL_DENSE_INDICES].all():
            raise ValueError("official six training targets must all be present")
        if not np.isfinite(delta).all() or not mask.any(axis=1).all():
            raise ValueError("masked dense target payload is invalid")
        return Dense72TrainingPayload(
            case_ids=train_ids.copy(),
            target_delta=delta,
            target_mask=mask.copy(),
            current_hs=current,
            decoded_scalar_count=int(mask.sum()),
            case_ids_sha256=_ids_sha256(train_ids),
            target_delta_sha256=_array_sha256(delta, dtype="<f4"),
            target_mask_sha256=_array_sha256(mask, dtype="|b1"),
            forbidden_scalar_decodes=int(self._forbidden_scalar_decodes),
        )

    def access_audit(self) -> dict[str, Any]:
        return {
            "source": "train_wave.csv_only",
            "anchor_count": int(self._anchor_count),
            "dense_steps": DENSE_TARGET_STEPS,
            "registered_validation_groups": sorted(self._validation_case_ids),
            "released_validation_groups": sorted(self._released_groups),
            "release_commitment_sha256": dict(sorted(self._release_commitments.items())),
            "unique_source_target_scalar_decodes": int(self._total_scalar_decodes),
            "forbidden_validation_target_scalar_decodes": int(
                self._forbidden_scalar_decodes
            ),
            "anonymous_test_value_reads": 0,
        }


__all__ = [
    "DENSE_TARGET_CADENCE_MINUTES",
    "DENSE_TARGET_STEPS",
    "Dense72AvailabilityAudit",
    "Dense72TargetAccessor",
    "Dense72TrainingPayload",
    "EXPECTED_ANCHOR_COUNT",
    "EXPECTED_COMPLETE_CASES",
    "EXPECTED_INCOMPLETE_CASES",
    "EXPECTED_MISSING_SCALARS",
    "EXPECTED_SOURCE_ROWS",
    "OFFICIAL_DENSE_INDICES",
    "SOURCE_HEADER",
    "sha256_file",
]
