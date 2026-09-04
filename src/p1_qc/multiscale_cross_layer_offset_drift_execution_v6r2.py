"""Authenticated one-shot curve engine for P1 multiscale Gen6r2.

The bootstrap loads this module only after an independent GO receipt and an
exact user authorization are validated.  Every public entry requires the live
post-lock capability; importing this file through Python's normal loader is
forbidden.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    _CONTEXT = _P1_V6R2_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R2_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
    science = _P1_V6R2_AUTH_SCIENCE  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r2 engine requires the authenticated bootstrap") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("mode") != "execute"
    or _CONTEXT.get("bootstrap_documents_prevalidated") is not True
    or _CONTEXT.get("all_owner_roles_authenticated") is not True
):
    raise RuntimeError("P1 Gen6r2 engine loaded before execution authorization")

GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r2"
TRAIN_COLUMNS = (
    "station",
    "year",
    "layer",
    "time",
    "temp",
    "psal",
    "depth",
    "label",
    "anomaly_type",
)
TARGET_COLUMNS = ("label", "anomaly_type")
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
_BUNDLE_MAGIC = b"P1_V6R2_ARRAY_BUNDLE_V1\n"


class ExecutionError(RuntimeError):
    """The authenticated execution failed closed."""


def _np_pd() -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    verifier = _CONTEXT.get("verify_numerical_runtime")
    if not callable(verifier):
        raise PermissionError("numerical runtime origin verifier is unavailable")
    verifier()
    return np, pd


def _require(capability: object, entry: str) -> None:
    contract.require_engine_capability(capability, entry)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ids_sha(values: Any) -> str:
    np, _pd = _np_pd()
    array_value = np.asarray(values)
    if array_value.dtype != np.dtype("int64") or array_value.ndim != 1:
        raise ExecutionError("row identity must be exact int64")
    return _sha_bytes(array_value.astype("<i8", copy=False).tobytes(order="C"))


def _csv_field_spans(
    raw_line: bytes, expected_fields: int
) -> tuple[bytes, tuple[tuple[int, int], ...]]:
    line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.rstrip(b"\n")
    if b"\r" in line or b"\n" in line:
        raise ExecutionError("embedded CSV newlines are forbidden")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    quoted = False
    while index < len(line):
        value = line[index]
        if value == 34:
            if quoted and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            quoted = not quoted
        elif value == 44 and not quoted:
            spans.append((start, index))
            start = index + 1
        index += 1
    if quoted:
        raise ExecutionError("unterminated quoted CSV field")
    spans.append((start, len(line)))
    if len(spans) != expected_fields:
        raise ExecutionError("train.csv field count differs")
    return line, tuple(spans)


def _decode_field(line: bytes, span: tuple[int, int]) -> str:
    token = line[span[0] : span[1]].decode("utf-8", errors="strict")
    parsed = next(csv.reader([token], strict=True), None)
    if parsed is None or len(parsed) != 1:
        raise ExecutionError("selected CSV scalar differs")
    return parsed[0]


def _optional_float(line: bytes, span: tuple[int, int]) -> float:
    value = _decode_field(line, span)
    if value == "" or value.casefold() in {"na", "nan", "null", "none"}:
        return float("nan")
    return float(value)


def load_input_only_train(capability: object, raw_train: bytes) -> Any:
    """Decode seven input fields while never slicing either target field."""

    _require(capability, "load_input_only_train")
    if type(raw_train) is not bytes:
        raise ExecutionError("train.csv must be the authenticated immutable byte buffer")
    np, pd = _np_pd()
    station: list[str] = []
    year = array("q")
    layer = array("q")
    times: list[str] = []
    temp = array("d")
    psal = array("d")
    depth = array("d")
    with io.BytesIO(raw_train) as stream:
        header_line, header_spans = _csv_field_spans(stream.readline(), len(TRAIN_COLUMNS))
        header = tuple(_decode_field(header_line, span) for span in header_spans)
        if header != TRAIN_COLUMNS:
            raise ExecutionError("train.csv schema/order differs")
        while raw := stream.readline():
            line, spans = _csv_field_spans(raw, len(TRAIN_COLUMNS))
            station.append(_decode_field(line, spans[0]))
            year.append(int(_decode_field(line, spans[1])))
            layer.append(int(_decode_field(line, spans[2])))
            times.append(_decode_field(line, spans[3]))
            temp.append(_optional_float(line, spans[4]))
            psal.append(_optional_float(line, spans[5]))
            depth.append(_optional_float(line, spans[6]))
            # spans[7] and spans[8] stay opaque.
    frame = pd.DataFrame(
        {
            "station": station,
            "year": np.asarray(year, dtype=np.int64),
            "layer": np.asarray(layer, dtype=np.int64),
            "time": times,
            "temp": np.asarray(temp, dtype=np.float64),
            "psal": np.asarray(psal, dtype=np.float64),
            "depth": np.asarray(depth, dtype=np.float64),
        },
        columns=list(science.INPUT_ONLY_COLUMNS),
    )
    if tuple(frame.columns) != science.INPUT_ONLY_COLUMNS or set(TARGET_COLUMNS) & set(frame):
        raise ExecutionError("input-only projection differs")
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    stable = frame.assign(__time=parsed, __row=np.arange(len(frame), dtype=np.int64))
    sorted_rows = stable.sort_values(
        ["station", "year", "layer", "__time", "__row"], kind="stable"
    )["__row"].to_numpy(dtype=np.int64)
    if not np.array_equal(sorted_rows, np.arange(len(frame), dtype=np.int64)):
        raise ExecutionError("train.csv is not in the exact stable key order")
    if frame.loc[:, list(science.KEY_COLUMNS)].duplicated().any():
        raise ExecutionError("train.csv keys are not unique")
    return frame


@dataclass(frozen=True)
class ReleaseScope:
    kind: str
    fold: str
    cell: int | None
    block: int | None
    row_ids_sha256: str
    commitment_sha256: str


class SelectiveTargetAccessorV6R2:
    """Opaque byte-span index whose releases are checked by the live ledger."""

    def __init__(
        self,
        *,
        capability: object,
        raw_train: bytes,
        expected_rows: int,
        validation_rows: dict[str, Any],
        ledger: BlindCommitmentLedger,
    ) -> None:
        _require(capability, "target_accessor_init")
        np, _pd = _np_pd()
        if type(raw_train) is not bytes:
            raise ExecutionError("target accessor requires authenticated train bytes")
        if tuple(validation_rows) != FOLDS:
            raise ExecutionError("validation fold order differs")
        self._capability = capability
        self._raw_train = raw_train
        self._ledger = ledger
        self._validation = {
            fold: self._strict_ids(values, expected_rows, f"{fold} validation")
            for fold, values in validation_rows.items()
        }
        combined = np.concatenate(list(self._validation.values()))
        if len(np.unique(combined)) != len(combined):
            raise ExecutionError("outer validation folds overlap")
        self._row_fold = np.full(expected_rows, -1, dtype=np.int8)
        for ordinal, fold in enumerate(FOLDS):
            self._row_fold[self._validation[fold]] = ordinal
        self._offsets = array("Q")
        self._lengths = array("I")
        self._header: tuple[str, ...] = ()
        self._columns: dict[str, int] = {}
        self._labels: dict[int, int] = {}
        self._anomalies: dict[int, str] = {}
        self._events: list[dict[str, Any]] = []
        self._build_index(expected_rows)

    @staticmethod
    def _strict_ids(values: Any, size: int, role: str) -> Any:
        np, _pd = _np_pd()
        ids = np.asarray(values)
        if (
            ids.dtype != np.dtype("int64")
            or ids.ndim != 1
            or (len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= size))
            or len(np.unique(ids)) != len(ids)
        ):
            raise ExecutionError(f"{role} IDs differ")
        return np.ascontiguousarray(ids)

    def _build_index(self, expected_rows: int) -> None:
        with io.BytesIO(self._raw_train) as stream:
            line, spans = _csv_field_spans(stream.readline(), len(TRAIN_COLUMNS))
            self._header = tuple(_decode_field(line, span) for span in spans)
            if self._header != TRAIN_COLUMNS:
                raise ExecutionError("opaque target index header differs")
            self._columns = {name: index for index, name in enumerate(self._header)}
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                _csv_field_spans(raw, len(TRAIN_COLUMNS))
                self._offsets.append(offset)
                self._lengths.append(len(raw))
        if len(self._offsets) != expected_rows:
            raise ExecutionError("opaque target index row count differs")

    def _decode(self, ids: Any, column: str) -> dict[int, str]:
        result: dict[int, str] = {}
        index = self._columns[column]
        with io.BytesIO(self._raw_train) as stream:
            for row_id in ids.tolist():
                stream.seek(int(self._offsets[row_id]))
                line, spans = _csv_field_spans(
                    stream.read(int(self._lengths[row_id])), len(TRAIN_COLUMNS)
                )
                result[row_id] = _decode_field(line, spans[index])
        return result

    def _training_allowed(self, ids: Any, active_fold: str) -> None:
        np, _pd = _np_pd()
        if np.intersect1d(ids, self._validation[active_fold]).size:
            raise PermissionError("active outer-validation targets remain withheld")
        for fold in FOLDS:
            if fold == active_fold:
                break
            if np.intersect1d(
                ids, self._validation[fold]
            ).size and not self._ledger.is_fold_committed(fold):
                raise PermissionError("earlier outer fold has not been committed")

    def _release(
        self,
        ids: Any,
        *,
        purpose: str,
        active_fold: str,
        scope: ReleaseScope | None,
        require_global: bool,
    ) -> tuple[Any, Any]:
        _require(self._capability, "target_release")
        np, _pd = _np_pd()
        ids = self._strict_ids(ids, len(self._offsets), purpose)
        if require_global:
            if not self._ledger.is_global_committed():
                raise PermissionError("aggregate target release preceded predictions_complete")
        elif scope is None:
            self._training_allowed(ids, active_fold)
        else:
            self._ledger.validate_release_scope(scope, ids)
        missing_labels = np.asarray(
            [item for item in ids if int(item) not in self._labels], dtype=np.int64
        )
        missing_anomalies = np.asarray(
            [item for item in ids if int(item) not in self._anomalies], dtype=np.int64
        )
        if len(missing_labels):
            decoded = self._decode(missing_labels, "label")
            if any(value not in {"0", "1"} for value in decoded.values()):
                raise ExecutionError("decoded label is not binary")
            self._labels.update({row: int(value) for row, value in decoded.items()})
        if len(missing_anomalies):
            self._anomalies.update(self._decode(missing_anomalies, "anomaly_type"))
        newly_decoded = len(missing_labels) + len(missing_anomalies)
        if newly_decoded:
            contract.bump_counter(self._capability, "target_decodes", newly_decoded)
        self._events.append(
            {
                "purpose": purpose,
                "fold": active_fold,
                "scope_kind": None if scope is None else scope.kind,
                "row_count": len(ids),
                "row_ids_sha256": _ids_sha(ids),
                "decoded_label_rows_new": len(missing_labels),
                "decoded_anomaly_rows_new": len(missing_anomalies),
                "after_commitment": scope is not None or require_global,
            }
        )
        return (
            np.asarray([self._labels[int(item)] for item in ids], dtype=np.int8),
            np.asarray([self._anomalies[int(item)] for item in ids], dtype=object),
        )

    def release_training(self, ids: Any, *, purpose: str, active_fold: str) -> tuple[Any, Any]:
        return self._release(
            ids, purpose=purpose, active_fold=active_fold, scope=None, require_global=False
        )

    def release_committed(
        self, ids: Any, *, purpose: str, active_fold: str, scope: ReleaseScope
    ) -> tuple[Any, Any]:
        return self._release(
            ids, purpose=purpose, active_fold=active_fold, scope=scope, require_global=False
        )

    def release_global(self, ids: Any, *, purpose: str, active_fold: str) -> tuple[Any, Any]:
        return self._release(
            ids, purpose=purpose, active_fold=active_fold, scope=None, require_global=True
        )

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_v6r2_selective_target_audit.v1",
            "opaque_index_rows": len(self._offsets),
            "target_fields_decoded_while_indexing": 0,
            "decoded_label_rows": len(self._labels),
            "decoded_anomaly_rows": len(self._anomalies),
            "events": list(self._events),
        }


def _array_bundle_bytes(arrays: dict[str, Any]) -> bytes:
    np, _pd = _np_pd()
    if type(arrays) is not dict or not arrays:
        raise ExecutionError("array bundle must be a nonempty exact mapping")
    header_arrays: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype not in {np.dtype("int8"), np.dtype("int64"), np.dtype("float32")}:
            raise ExecutionError("array bundle dtype is outside the exact allowlist")
        if value.ndim != 1 or not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        raw = value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        header_arrays.append(
            {
                "name": name,
                "dtype": value.dtype.newbyteorder("<").str,
                "shape": [len(value)],
                "offset": offset,
                "bytes": len(raw),
                "sha256": _sha_bytes(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)
    header = _canonical({"schema_version": "p1_v6r2_array_bundle.v1", "arrays": header_arrays})
    return _BUNDLE_MAGIC + struct.pack("<Q", len(header)) + header + b"".join(chunks)


def _load_array_bundle_bytes(payload: bytes) -> dict[str, Any]:
    np, _pd = _np_pd()
    if not payload.startswith(_BUNDLE_MAGIC) or len(payload) < len(_BUNDLE_MAGIC) + 8:
        raise ExecutionError("array bundle magic differs")
    header_size = struct.unpack("<Q", payload[len(_BUNDLE_MAGIC) : len(_BUNDLE_MAGIC) + 8])[0]
    header_start = len(_BUNDLE_MAGIC) + 8
    header_stop = header_start + header_size
    header = json.loads(payload[header_start:header_stop].decode("utf-8"))
    if header.get("schema_version") != "p1_v6r2_array_bundle.v1":
        raise ExecutionError("array bundle schema differs")
    body = payload[header_stop:]
    result: dict[str, Any] = {}
    expected_offset = 0
    for item in header.get("arrays", []):
        if (
            type(item) is not dict
            or set(item) != {"name", "dtype", "shape", "offset", "bytes", "sha256"}
            or item["offset"] != expected_offset
            or type(item["shape"]) is not list
            or len(item["shape"]) != 1
            or item["bytes"] < 0
        ):
            raise ExecutionError("array bundle header differs")
        stop = item["offset"] + item["bytes"]
        raw = body[item["offset"] : stop]
        if len(raw) != item["bytes"] or _sha_bytes(raw) != item["sha256"]:
            raise ExecutionError("array bundle payload differs")
        dtype = np.dtype(item["dtype"])
        if dtype not in {np.dtype("int8"), np.dtype("int64"), np.dtype("float32")}:
            raise ExecutionError("array bundle reload dtype differs")
        value = np.frombuffer(raw, dtype=dtype).copy()
        if value.shape != tuple(item["shape"]):
            raise ExecutionError("array bundle reload shape differs")
        result[item["name"]] = value
        expected_offset = stop
    if expected_offset != len(body) or list(result) != sorted(result):
        raise ExecutionError("array bundle exact inventory differs")
    return result


class BlindCommitmentLedger:
    """Fold-major, cell-interleaved immutable blind commitment chain."""

    def __init__(self, capability: object) -> None:
        _require(capability, "ledger_init")
        self._capability = capability
        self._chain = _sha_bytes(b"p1_v6r2_blind_commitment_genesis")
        self._next_cell = 1
        self._inner_in_cell = 0
        self._folds: set[str] = set()
        self._global = False
        self._scopes: dict[str, ReleaseScope] = {}
        self._released_scopes: set[str] = set()
        self._pins: list[dict[str, Any]] = []

    @staticmethod
    def _cell_identity(cell: int) -> tuple[str, float]:
        if type(cell) is not int or not 1 <= cell <= 15:
            raise ExecutionError("cell number differs")
        return FOLDS[(cell - 1) // 5], FRACTIONS[(cell - 1) % 5]

    def _event(self, relative: str, body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        event = {
            **body,
            "generation": GENERATION,
            "prior_event_sha256": self._chain,
        }
        event_sha = contract.deep_sha256(event)
        event["event_sha256"] = event_sha
        pin = contract.write_output_exclusive(self._capability, relative, event)
        self._chain = event_sha
        self._pins.append(pin)
        return pin, event_sha

    def commit_inner(
        self,
        *,
        cell: int,
        block: int,
        prediction_ids: Any,
        incumbent_probability: Any,
        incumbent_prediction: Any,
        candidate_probability: Any,
        candidate_prediction: Any,
        model_pin: dict[str, Any],
        split_proof_sha256: str,
    ) -> ReleaseScope:
        np, _pd = _np_pd()
        fold, fraction = self._cell_identity(cell)
        if cell != self._next_cell or block != self._inner_in_cell + 1 or block not in {1, 2, 3}:
            raise ExecutionError("inner commitment sequence differs")
        ids = np.asarray(prediction_ids)
        arrays = {
            "candidate_prediction": np.ascontiguousarray(candidate_prediction, dtype=np.int8),
            "candidate_probability": np.ascontiguousarray(candidate_probability, dtype=np.float32),
            "incumbent_prediction": np.ascontiguousarray(incumbent_prediction, dtype=np.int8),
            "incumbent_probability": np.ascontiguousarray(incumbent_probability, dtype=np.float32),
            "row_ids": np.ascontiguousarray(ids, dtype=np.int64),
        }
        if any(len(value) != len(ids) for value in arrays.values()):
            raise ExecutionError("inner commitment array lengths differ")
        payload = _array_bundle_bytes(arrays)
        reloaded = _load_array_bundle_bytes(payload)
        if any(
            reloaded[name].dtype != value.dtype
            or reloaded[name].shape != value.shape
            or reloaded[name].tobytes(order="C") != value.tobytes(order="C")
            for name, value in arrays.items()
        ):
            raise ExecutionError("inner prediction reload identity differs")
        ordinal = (cell - 1) * 3 + block
        prediction_pin = contract.write_output_exclusive(
            self._capability, f"inner_predictions/inner_{ordinal:02d}.bin", payload
        )
        pin, event_sha = self._event(
            f"blind_commitments/inner_{ordinal:02d}.json",
            {
                "schema_version": "p1_v6r2_inner_commitment.v1",
                "ordinal": ordinal,
                "cell": cell,
                "block": block,
                "fold": fold,
                "fraction": fraction,
                "row_ids_sha256": _ids_sha(arrays["row_ids"]),
                "prediction_bundle": prediction_pin,
                "model": model_pin,
                "split_proof_sha256": split_proof_sha256,
                "target_scalars_decoded_before_commitment": 0,
            },
        )
        contract.bump_counter(self._capability, "inner_commitments")
        self._inner_in_cell += 1
        scope = ReleaseScope(
            kind="inner",
            fold=fold,
            cell=cell,
            block=block,
            row_ids_sha256=_ids_sha(ids),
            commitment_sha256=pin["sha256"],
        )
        self._scopes[event_sha] = scope
        return scope

    def commit_cell(
        self,
        *,
        cell: int,
        validation_ids: Any,
        incumbent_probability: Any,
        incumbent_prediction: Any,
        candidate_probability: Any,
        candidate_prediction: Any,
        model_pin: dict[str, Any],
        gate: dict[str, Any],
    ) -> ReleaseScope:
        np, _pd = _np_pd()
        fold, fraction = self._cell_identity(cell)
        if cell != self._next_cell or self._inner_in_cell != 3:
            raise ExecutionError("cell commitment preceded its three inner commitments")
        ids = np.ascontiguousarray(validation_ids, dtype=np.int64)
        arrays = {
            "candidate_prediction": np.ascontiguousarray(candidate_prediction, dtype=np.int8),
            "candidate_probability": np.ascontiguousarray(candidate_probability, dtype=np.float32),
            "incumbent_prediction": np.ascontiguousarray(incumbent_prediction, dtype=np.int8),
            "incumbent_probability": np.ascontiguousarray(incumbent_probability, dtype=np.float32),
            "row_ids": ids,
        }
        if any(len(value) != len(ids) for value in arrays.values()):
            raise ExecutionError("cell commitment array lengths differ")
        payload = _array_bundle_bytes(arrays)
        reloaded = _load_array_bundle_bytes(payload)
        if any(
            reloaded[name].dtype != value.dtype
            or reloaded[name].tobytes(order="C") != value.tobytes(order="C")
            for name, value in arrays.items()
        ):
            raise ExecutionError("cell prediction reload identity differs")
        prediction_pin = contract.write_output_exclusive(
            self._capability, f"prediction_parts/cell_{cell:02d}.bin", payload
        )
        pin, event_sha = self._event(
            f"blind_commitments/cell_{cell:02d}.json",
            {
                "schema_version": "p1_v6r2_cell_commitment.v1",
                "cell": cell,
                "fold": fold,
                "fraction": fraction,
                "row_ids_sha256": _ids_sha(ids),
                "prediction_bundle": prediction_pin,
                "model": model_pin,
                "train_only_gate": gate,
                "active_outer_target_scalars_decoded_before_commitment": 0,
            },
        )
        contract.bump_counter(self._capability, "cell_commitments")
        scope = ReleaseScope(
            kind="cell",
            fold=fold,
            cell=cell,
            block=None,
            row_ids_sha256=_ids_sha(ids),
            commitment_sha256=pin["sha256"],
        )
        self._scopes[event_sha] = scope
        self._next_cell += 1
        self._inner_in_cell = 0
        return scope

    def commit_fold(self, fold: str) -> None:
        expected = FOLDS[len(self._folds)] if len(self._folds) < len(FOLDS) else None
        expected_next_cell = (FOLDS.index(fold) + 1) * 5 + 1
        if fold != expected or self._next_cell != expected_next_cell or self._inner_in_cell:
            raise ExecutionError("fold commitment sequence differs")
        self._event(
            f"blind_commitments/fold_{fold}.json",
            {
                "schema_version": "p1_v6r2_fold_commitment.v1",
                "fold": fold,
                "cell_count": 5,
                "active_fold_target_scalars_decoded_before_commitment": 0,
            },
        )
        self._folds.add(fold)
        contract.bump_counter(self._capability, "fold_commitments")

    def complete(self) -> dict[str, Any]:
        if self._global or self._next_cell != 16 or self._folds != set(FOLDS):
            raise ExecutionError("predictions_complete sequence differs")
        pin, _event_sha = self._event(
            "blind_commitments/predictions_complete.json",
            {
                "schema_version": "p1_v6r2_predictions_complete.v1",
                "inner_commitments": 45,
                "cell_commitments": 15,
                "fold_commitments": 3,
                "aggregate_target_scalars_decoded_before_completion": 0,
                "candidate_created": False,
                "test_prediction_created": False,
                "ledger_appended": False,
                "uploaded": False,
            },
        )
        contract.bump_counter(self._capability, "predictions_complete")
        self._global = True
        return pin

    def validate_release_scope(self, scope: ReleaseScope, ids: Any) -> None:
        if not isinstance(scope, ReleaseScope) or scope not in self._scopes.values():
            raise PermissionError("forged or stale target release scope")
        if scope.row_ids_sha256 != _ids_sha(ids):
            raise PermissionError("target release IDs differ from commitment")
        if scope.commitment_sha256 in self._released_scopes:
            raise PermissionError("target release scope was already consumed")
        if scope.kind == "cell" and scope.fold not in self._folds:
            raise PermissionError("outer cell release preceded five-cell fold commitment")
        self._released_scopes.add(scope.commitment_sha256)

    def is_fold_committed(self, fold: str) -> bool:
        return fold in self._folds

    def is_global_committed(self) -> bool:
        return self._global

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_v6r2_commitment_audit.v1",
            "inner_commitments": 45
            if self._global
            else (self._next_cell - 1) * 3 + self._inner_in_cell,
            "cell_commitments": self._next_cell - 1,
            "fold_commitments": len(self._folds),
            "predictions_complete": int(self._global),
            "chain_head_sha256": self._chain,
            "pins": list(self._pins),
        }


def _workspace_pin(relative: str, pin: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": f"{relative.rstrip('/')}/{pin['path']}",
        "bytes": pin["bytes"],
        "sha256": pin["sha256"],
    }


def _read_pinned_npy(root: Path, relative: str, expected: dict[str, Any]) -> Any:
    np, _pd = _np_pd()
    path = contract.contained_path(root, relative, must_exist=True, kind="file")
    root_relative = root.relative_to(_CONTEXT["workspace"]).as_posix()
    workspace_relative = f"{root_relative}/{relative}"
    before = contract.file_pin(path, relative=workspace_relative)
    if (
        type(expected) is not dict
        or set(expected) != {"bytes", "sha256"}
        or before["bytes"] != expected["bytes"]
        or before["sha256"] != expected["sha256"]
    ):
        raise ExecutionError(f"pinned numpy artifact differs: {relative}")
    payload = _CONTEXT["authenticated_bytes_for_pin"](
        {
            "path": workspace_relative,
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        },
        f"teacher numpy {relative}",
    )
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    after = contract.file_pin(path, relative=workspace_relative)
    if before != after:
        raise ExecutionError(f"numpy artifact identity changed during load: {relative}")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("float32")
        or value.ndim != 1
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ExecutionError(f"teacher numpy array domain differs: {relative}")
    return value


def _load_incumbent_catalog(capability: object) -> dict[str, Any]:
    _require(capability, "load_incumbent_catalog")
    config = _CONTEXT["config"]
    outer = config["incumbent_binding"]
    inner = config["inner_incumbent_binding"]
    predictions_complete = _CONTEXT["authenticated_json_for_pin"](
        outer["predictions_complete"], "outer incumbent predictions_complete"
    )
    split_audit = _CONTEXT["authenticated_json_for_pin"](
        inner["gen5r6_split_audit"], "inner incumbent split audit"
    )
    manifest = _CONTEXT["authenticated_json_for_pin"](
        inner["gen5r6_manifest"], "inner incumbent manifest"
    )
    parts = {
        (item["fold"], float(item["fraction"])): item for item in predictions_complete["parts"]
    }
    expected = {(fold, fraction) for fold in FOLDS for fraction in FRACTIONS}
    if set(parts) != expected:
        raise ExecutionError("outer incumbent cell catalog differs")
    return {"parts": parts, "split_audit": split_audit, "manifest": manifest}


def _load_outer_cell(
    capability: object,
    *,
    frame: Any,
    catalog: dict[str, Any],
    fold: str,
    fraction: float,
) -> dict[str, Any]:
    _require(capability, "load_outer_cell")
    np, pd = _np_pd()
    item = catalog["parts"][(fold, fraction)]
    relative = str(item["parquet"]).replace("\\", "/")
    path = contract.contained_path(_CONTEXT["workspace"], relative, must_exist=True, kind="file")
    before = contract.file_pin(path, relative=relative)
    if before["sha256"] != item["parquet_sha256"]:
        raise ExecutionError("outer incumbent parquet pin differs")
    parquet_raw = _CONTEXT["authenticated_bytes_for_pin"](
        {
            "path": relative,
            "bytes": before["bytes"],
            "sha256": item["parquet_sha256"],
        },
        f"outer incumbent parquet {fold} {fraction}",
    )
    columns = [
        *science.KEY_COLUMNS,
        "row_position",
        "fold",
        "fraction",
        "baseline_probability",
        "baseline_prediction",
    ]
    part = pd.read_parquet(io.BytesIO(parquet_raw), columns=columns)
    _CONTEXT["verify_numerical_runtime"]()
    after = contract.file_pin(path, relative=relative)
    if before != after or tuple(part.columns) != tuple(columns):
        raise ExecutionError("outer incumbent parquet identity changed during projection")
    ids = np.ascontiguousarray(part["row_position"].to_numpy(dtype=np.int64))
    if len(np.unique(ids)) != len(ids) or (len(ids) and int(ids.max()) >= len(frame)):
        raise ExecutionError("outer incumbent validation IDs differ")
    key_left = part.loc[:, list(science.KEY_COLUMNS)].reset_index(drop=True).astype(str)
    key_right = frame.iloc[ids].loc[:, list(science.KEY_COLUMNS)].reset_index(drop=True).astype(str)
    if not key_left.equals(key_right):
        raise ExecutionError("outer incumbent validation key order differs")
    if (
        not (part["fold"] == fold).all()
        or not (part["fraction"].to_numpy(dtype=np.float64) == fraction).all()
    ):
        raise ExecutionError("outer incumbent fold/fraction differs")
    probability = np.ascontiguousarray(part["baseline_probability"].to_numpy(dtype=np.float32))
    prediction = np.ascontiguousarray(part["baseline_prediction"].to_numpy(dtype=np.int8))
    if (
        not np.isfinite(probability).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
        or not np.isin(prediction, [0, 1]).all()
    ):
        raise ExecutionError("outer incumbent probability/prediction domain differs")
    tag = f"p{int(round(fraction * 100)):03d}"
    prefix = catalog["split_audit"]["prefixes"][tag][fold]
    cutoff = pd.Timestamp(prefix["adjusted_cutoff_utc"])
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    prefix_ids = np.ascontiguousarray(
        np.flatnonzero((parsed <= cutoff).to_numpy()).astype(np.int64, copy=False)
    )
    if (
        len(prefix_ids) != prefix["prefix_rows"]
        or _ids_sha(prefix_ids) != prefix["id_sha256_little_endian_int64"]
        or _ids_sha(ids) != prefix["validation_id_sha256_little_endian_int64"]
    ):
        raise ExecutionError("outer prefix/validation row binding differs")
    return {
        "fold": fold,
        "fraction": fraction,
        "prefix_ids": prefix_ids,
        "validation_ids": ids,
        "incumbent_probability": probability,
        "incumbent_prediction": prediction,
    }


def _load_inner_incumbent(
    capability: object,
    *,
    catalog: dict[str, Any],
    frame: Any,
    fold: str,
    fraction: float,
    block: int,
    prediction_ids: Any,
) -> tuple[Any, Any]:
    _require(capability, "load_inner_incumbent")
    tag = f"p{int(round(fraction * 100)):03d}"
    artifact_root = contract.contained_path(
        _CONTEXT["workspace"],
        "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6",
        must_exist=True,
        kind="directory",
    )
    arrays: list[Any] = []
    for seed in SEEDS:
        relative = f"teacher_blind_predictions/curve/{tag}/{fold}/block_{block}/seed_{seed}.npy"
        arrays.append(
            _read_pinned_npy(artifact_root, relative, catalog["manifest"]["artifacts"][relative])
        )
    probability = science.mean_seed_incumbent_probability(
        capability=capability, seed_probabilities=tuple(arrays)
    )
    if len(probability) != len(prediction_ids):
        raise ExecutionError("inner teacher probability row count differs")
    prediction_frame = frame.iloc[prediction_ids].loc[:, list(science.INPUT_ONLY_COLUMNS)].copy()
    prediction = science.fixed_incumbent_postprocess(
        capability=capability, frame=prediction_frame, probabilities=probability, fold=fold
    )
    return probability, prediction


def _slow_target(labels: Any, anomaly_types: Any) -> Any:
    np, _pd = _np_pd()
    return np.ascontiguousarray(
        (
            (np.asarray(labels, dtype=np.int8) == 1)
            & np.asarray(
                ["offset" in str(value) or "drift" in str(value) for value in anomaly_types],
                dtype=bool,
            )
        ).astype(np.int8)
    )


def _fit_and_predict(
    capability: object,
    *,
    frame: Any,
    segment_ids: Any,
    train_ids: Any,
    prediction_ids: Any,
    labels: Any,
    anomaly_types: Any,
    model_relative: str,
) -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
    """Fit, persist, strictly reload, and reproduce one fixed baseline/unary pair."""

    _require(capability, "fit_predict_unit")
    np, _pd = _np_pd()
    contract.bump_counter(capability, "baseline_fits")
    contract.bump_counter(capability, "unary_fits")
    contract.bump_counter(capability, "top_level_fits", 2)
    split = science.verify_dependency_closed_split(
        capability=capability,
        frame=frame,
        train_ids=train_ids,
        holdout_ids=prediction_ids,
        segment_ids=segment_ids,
    )
    train_frame = frame.iloc[train_ids]
    groups = int(train_frame.loc[:, ["station", "layer"]].drop_duplicates().shape[0])
    contract.bump_counter(capability, "seasonal_subfits", groups)
    contract.bump_counter(capability, "seasonal_irls_steps", groups * 8)
    context_ids = np.ascontiguousarray(
        np.unique(np.concatenate((train_ids, prediction_ids))).astype(np.int64, copy=False)
    )
    context_frame = frame.iloc[context_ids].loc[:, list(science.INPUT_ONLY_COLUMNS)].copy()
    local_segments = science.exact_gap_safe_segment_ids(context_frame)
    model_value: dict[str, Any]
    original_probability: Any | None = None
    unavailable_reason: str | None = None
    try:
        baseline = science.fit_robust_seasonal_graph_state(
            capability=capability,
            input_only_frame=frame,
            train_ids=train_ids,
            split_audit=split,
        )
        contract.bump_counter(capability, "graph_edges", len(baseline.edge_residual_deltas))
        projection = science.apply_robust_seasonal_graph_state(
            capability=capability, input_only_frame=context_frame, state=baseline
        )
        geometry = science.build_multiscale_geometry(
            capability=capability,
            baseline_projection=projection,
            segment_ids=local_segments,
            row_ids=context_ids,
        )
        train_target = _slow_target(labels, anomaly_types)
        unary = science.fit_fixed_slow_unary_head(
            capability=capability,
            train_geometry=geometry.loc[train_ids],
            decoded_train_target=train_target,
            explicit_train_ids=train_ids,
            baseline_state=baseline,
        )
        contract.bump_counter(capability, "unary_lbfgs_iterations", unary.optimizer_iterations)
        contract.bump_counter(
            capability, "iterative_steps", groups * 8 + unary.optimizer_iterations
        )
        original_probability = science.predict_fixed_slow_unary_probability(
            capability=capability, geometry=geometry.loc[prediction_ids], state=unary
        )
        model_value = {
            "schema_version": "p1_v6r2_model_pair.v1",
            "available": True,
            "train_ids_sha256": _ids_sha(train_ids),
            "prediction_ids_sha256": _ids_sha(prediction_ids),
            "split_proof": split,
            "baseline_state": baseline.as_dict(),
            "baseline_state_sha256": baseline.state_sha256,
            "unary_state": unary.as_dict(),
            "unary_state_sha256": unary.state_sha256,
        }
    except (science.ScienceContractError, ValueError, np.linalg.LinAlgError) as exc:
        unavailable_reason = f"{type(exc).__name__}:{exc}"
        model_value = {
            "schema_version": "p1_v6r2_model_pair.v1",
            "available": False,
            "train_ids_sha256": _ids_sha(train_ids),
            "prediction_ids_sha256": _ids_sha(prediction_ids),
            "split_proof": split,
            "unavailable_reason": unavailable_reason,
        }
    pin = contract.write_output_exclusive(capability, model_relative, model_value)
    output_relative = _CONTEXT["config"]["canonical_paths"]["output"]
    reloaded = _CONTEXT["strict_dynamic_json_for_pin"](
        _workspace_pin(output_relative, pin), "persisted model pair"
    )
    if reloaded != model_value:
        raise ExecutionError("persisted model pair JSON differs after reload")
    reproduced: Any | None = None
    reload_exact = False
    if reloaded["available"]:
        baseline_reload = science.RobustSeasonalGraphState.from_dict(reloaded["baseline_state"])
        unary_reload = science.FixedSlowUnaryState.from_dict(reloaded["unary_state"])
        if not (
            baseline_reload.state_sha256 == reloaded["baseline_state_sha256"]
            and unary_reload.state_sha256 == reloaded["unary_state_sha256"]
            and baseline_reload.train_ids_sha256 == reloaded["train_ids_sha256"]
            and unary_reload.train_ids_sha256 == reloaded["train_ids_sha256"]
            and baseline_reload.split_audit_sha256 == contract.deep_sha256(reloaded["split_proof"])
        ):
            raise ExecutionError("reloaded baseline/scaler/unary row or state binding differs")
        projection_reload = science.apply_robust_seasonal_graph_state(
            capability=capability, input_only_frame=context_frame, state=baseline_reload
        )
        geometry_reload = science.build_multiscale_geometry(
            capability=capability,
            baseline_projection=projection_reload,
            segment_ids=local_segments,
            row_ids=context_ids,
        )
        reproduced = science.predict_fixed_slow_unary_probability(
            capability=capability,
            geometry=geometry_reload.loc[prediction_ids],
            state=unary_reload,
        )
        reload_exact = bool(
            original_probability is not None
            and np.asarray(original_probability).dtype == np.asarray(reproduced).dtype
            and np.asarray(original_probability).tobytes(order="C")
            == np.asarray(reproduced).tobytes(order="C")
        )
        if not reload_exact:
            raise ExecutionError("saved model reload inference is not byte-exact")
    contract.bump_counter(capability, "predictions")
    return (
        reproduced,
        pin,
        {
            "available": reloaded["available"],
            "unavailable_reason": unavailable_reason,
            "reload_inference_byte_exact": reload_exact if reloaded["available"] else None,
            "split_proof": split,
        },
    )


def _station_layer(frame: Any, ids: Any) -> Any:
    np, _pd = _np_pd()
    rows = frame.iloc[ids]
    return np.asarray(
        [
            f"{station}|{int(layer)}"
            for station, layer in rows.loc[:, ["station", "layer"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=object,
    )


def _inner_gate(blocks: list[dict[str, Any]], capability: object) -> dict[str, Any]:
    np, _pd = _np_pd()
    truth = np.concatenate([item["truth"] for item in blocks]).astype(np.int8, copy=False)
    anomaly = np.concatenate([item["anomaly"] for item in blocks])
    station_layer = np.concatenate([item["station_layer"] for item in blocks])
    incumbent = np.concatenate([item["incumbent_prediction"] for item in blocks]).astype(
        np.int8, copy=False
    )
    candidate = np.concatenate([item["candidate_prediction"] for item in blocks]).astype(
        np.int8, copy=False
    )
    segments: list[Any] = []
    offset = 0
    for item in blocks:
        local = np.asarray(item["segment_ids"], dtype=np.int64)
        if len(local):
            local = local - int(local.min()) + offset
            offset = int(local.max()) + 1
        segments.append(local)
    score = science.score_candidate_delta(
        capability=capability,
        truth=truth,
        anomaly_type=anomaly,
        station_layer=station_layer,
        segment_ids=np.concatenate(segments).astype(np.int64, copy=False),
        incumbent_prediction=incumbent,
        candidate_prediction=candidate,
    )
    metrics = score["metrics"]
    gate_input = {
        "micro_f1_delta": metrics["micro_f1_delta"],
        "offset_recall_delta": metrics["offset_recall_delta"],
        "drift_recall_delta": metrics["drift_recall_delta"],
        "spike_f1_delta": metrics["spike_f1_delta"],
        "worst_station_layer_f1_delta": metrics["worst_station_layer_f1_delta"],
        "normal_fp_relative_increase": metrics["normal_fp_relative_increase"],
        "nondegrading_inner_blocks": int(
            sum(item["score"]["metrics"]["micro_f1_delta"] >= 0.0 for item in blocks)
        ),
        "inner_block_count": 3,
        "both_slow_types_observed": bool(score["offset_observed"] and score["drift_observed"]),
        "spike_observed": bool(score["spike_observed"]),
        "all_required_station_layers_observed": bool(score["all_required_station_layers_observed"]),
        "blind_predictions_sealed_before_gate_labels": all(
            item["sealed_before_labels"] is True for item in blocks
        ),
    }
    gate = science.strict_inner_gate(gate_input)
    return {**gate, "metrics": gate_input, "station_layer_deltas": score["station_layer_deltas"]}


def _bootstrap_units(truth: Any, frame: Any, ids: Any, segment_ids: Any) -> Any:
    np, pd = _np_pd()
    y = np.asarray(truth, dtype=np.int8)
    segments = np.asarray(segment_ids, dtype=np.int64)
    rows = frame.iloc[ids]
    parsed = pd.to_datetime(rows["time"], errors="raise", utc=True, format="mixed").dt.tz_convert(
        "Asia/Seoul"
    )
    result = np.empty(len(ids), dtype=object)
    event = 0
    active = False
    prior_segment: int | None = None
    for index in range(len(ids)):
        if y[index] == 1:
            if not active or prior_segment != int(segments[index]):
                event += 1
            result[index] = f"event:{event}"
            active = True
        else:
            result[index] = (
                f"normal:{rows.iloc[index]['station']}|{int(rows.iloc[index]['layer'])}|"
                f"{parsed.iloc[index].date().isoformat()}"
            )
            active = False
        prior_segment = int(segments[index])
    return result


def _aggregate_score(capability: object, frame: Any, cells: list[dict[str, Any]]) -> dict[str, Any]:
    np, _pd = _np_pd()
    ids = np.concatenate([item["validation_ids"] for item in cells]).astype(np.int64, copy=False)
    truth = np.concatenate([item["truth"] for item in cells]).astype(np.int8, copy=False)
    anomaly = np.concatenate([item["anomaly"] for item in cells])
    incumbent = np.concatenate([item["incumbent_prediction"] for item in cells]).astype(
        np.int8, copy=False
    )
    candidate = np.concatenate([item["candidate_prediction"] for item in cells]).astype(
        np.int8, copy=False
    )
    full_segments = science.exact_gap_safe_segment_ids(frame)
    segments = full_segments[ids]
    score = science.score_candidate_delta(
        capability=capability,
        truth=truth,
        anomaly_type=anomaly,
        station_layer=_station_layer(frame, ids),
        segment_ids=segments,
        incumbent_prediction=incumbent,
        candidate_prediction=candidate,
    )
    contract.bump_counter(capability, "scores")
    return {
        **score,
        "ids": ids,
        "truth": truth,
        "incumbent_prediction": incumbent,
        "candidate_prediction": candidate,
        "bootstrap_units": _bootstrap_units(truth, frame, ids, segments),
    }


def run_curve(capability: object) -> dict[str, Any]:
    """Run the sole authenticated research curve; never create test/candidate/upload state."""

    _require(capability, "run_curve")
    np, _pd = _np_pd()
    config = _CONTEXT["config"]
    train_raw = _CONTEXT["authenticated_train_bytes_for_pin"](
        config["source_pins"]["train.csv"], "train.csv semantic input"
    )
    frame = load_input_only_train(capability, train_raw)
    segment_ids = science.exact_gap_safe_segment_ids(frame)
    catalog = _load_incumbent_catalog(capability)
    outer_cells = {
        (fold, fraction): _load_outer_cell(
            capability, frame=frame, catalog=catalog, fold=fold, fraction=fraction
        )
        for fold in FOLDS
        for fraction in FRACTIONS
    }
    validation_rows = {fold: outer_cells[(fold, FRACTIONS[0])]["validation_ids"] for fold in FOLDS}
    for fold in FOLDS:
        reference = validation_rows[fold]
        if any(
            not np.array_equal(reference, outer_cells[(fold, fraction)]["validation_ids"])
            for fraction in FRACTIONS
        ):
            raise ExecutionError("validation row IDs differ across fold fractions")
    ledger = BlindCommitmentLedger(capability)
    accessor = SelectiveTargetAccessorV6R2(
        capability=capability,
        raw_train=train_raw,
        expected_rows=len(frame),
        validation_rows=validation_rows,
        ledger=ledger,
    )
    split_audits: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    completed_cells: list[dict[str, Any]] = []
    cell_number = 0
    inner_ordinal = 0
    for fold in FOLDS:
        current_fold: list[dict[str, Any]] = []
        for fraction in FRACTIONS:
            cell_number += 1
            cell = outer_cells[(fold, fraction)]
            planned = science.build_three_block_inner_splits(
                capability=capability,
                metadata=frame,
                outer_prefix_ids=cell["prefix_ids"],
            )
            inner_results: list[dict[str, Any]] = []
            for split in planned:
                inner_ordinal += 1
                train_labels, train_anomaly = accessor.release_training(
                    split.train_ids,
                    purpose=f"cell_{cell_number:02d}_inner_{split.block}_train",
                    active_fold=fold,
                )
                incumbent_probability, incumbent_prediction = _load_inner_incumbent(
                    capability,
                    catalog=catalog,
                    frame=frame,
                    fold=fold,
                    fraction=fraction,
                    block=split.block,
                    prediction_ids=split.prediction_ids,
                )
                slow_probability, model_pin, model_audit = _fit_and_predict(
                    capability,
                    frame=frame,
                    segment_ids=segment_ids,
                    train_ids=split.train_ids,
                    prediction_ids=split.prediction_ids,
                    labels=train_labels,
                    anomaly_types=train_anomaly,
                    model_relative=f"models/inner_{inner_ordinal:02d}.json",
                )
                candidate_probability, candidate_prediction, additions = (
                    science.protected_incumbent_union(
                        capability=capability,
                        incumbent_probability=incumbent_probability,
                        incumbent_prediction=incumbent_prediction,
                        gate_passed=slow_probability is not None,
                        slow_probability=slow_probability,
                        segment_ids=segment_ids[split.prediction_ids],
                    )
                )
                scope = ledger.commit_inner(
                    cell=cell_number,
                    block=split.block,
                    prediction_ids=split.prediction_ids,
                    incumbent_probability=incumbent_probability,
                    incumbent_prediction=incumbent_prediction,
                    candidate_probability=candidate_probability,
                    candidate_prediction=candidate_prediction,
                    model_pin=model_pin,
                    split_proof_sha256=contract.deep_sha256(model_audit["split_proof"]),
                )
                holdout_labels, holdout_anomaly = accessor.release_committed(
                    split.prediction_ids,
                    purpose=f"cell_{cell_number:02d}_inner_{split.block}_gate",
                    active_fold=fold,
                    scope=scope,
                )
                score = science.score_candidate_delta(
                    capability=capability,
                    truth=holdout_labels,
                    anomaly_type=holdout_anomaly,
                    station_layer=_station_layer(frame, split.prediction_ids),
                    segment_ids=segment_ids[split.prediction_ids],
                    incumbent_prediction=incumbent_prediction,
                    candidate_prediction=candidate_prediction,
                )
                contract.bump_counter(capability, "scores")
                inner_results.append(
                    {
                        "truth": holdout_labels,
                        "anomaly": holdout_anomaly,
                        "station_layer": _station_layer(frame, split.prediction_ids),
                        "segment_ids": segment_ids[split.prediction_ids],
                        "incumbent_prediction": incumbent_prediction,
                        "candidate_prediction": candidate_prediction,
                        "score": score,
                        "additions": int(np.count_nonzero(additions)),
                        "sealed_before_labels": True,
                    }
                )
                split_audits.append(
                    {
                        "cell": cell_number,
                        "fold": fold,
                        "fraction": fraction,
                        **split.as_audit(),
                        "dependency_proof": model_audit["split_proof"],
                    }
                )
                model_audits.append({"role": "inner", "ordinal": inner_ordinal, **model_audit})
            gate = _inner_gate(inner_results, capability)
            outer_labels, outer_anomaly = accessor.release_training(
                cell["prefix_ids"],
                purpose=f"cell_{cell_number:02d}_outer_train",
                active_fold=fold,
            )
            slow_probability, model_pin, model_audit = _fit_and_predict(
                capability,
                frame=frame,
                segment_ids=segment_ids,
                train_ids=cell["prefix_ids"],
                prediction_ids=cell["validation_ids"],
                labels=outer_labels,
                anomaly_types=outer_anomaly,
                model_relative=f"models/cell_{cell_number:02d}.json",
            )
            candidate_probability, candidate_prediction, additions = (
                science.protected_incumbent_union(
                    capability=capability,
                    incumbent_probability=cell["incumbent_probability"],
                    incumbent_prediction=cell["incumbent_prediction"],
                    gate_passed=bool(gate["passed"] and slow_probability is not None),
                    slow_probability=slow_probability,
                    segment_ids=segment_ids[cell["validation_ids"]],
                )
            )
            scope = ledger.commit_cell(
                cell=cell_number,
                validation_ids=cell["validation_ids"],
                incumbent_probability=cell["incumbent_probability"],
                incumbent_prediction=cell["incumbent_prediction"],
                candidate_probability=candidate_probability,
                candidate_prediction=candidate_prediction,
                model_pin=model_pin,
                gate=gate,
            )
            current_fold.append(
                {
                    **cell,
                    "candidate_probability": candidate_probability,
                    "candidate_prediction": candidate_prediction,
                    "scope": scope,
                    "gate": gate,
                    "additions": int(np.count_nonzero(additions)),
                }
            )
            split_audits.append(
                {
                    "cell": cell_number,
                    "fold": fold,
                    "fraction": fraction,
                    "block": "outer",
                    "dependency_proof": model_audit["split_proof"],
                }
            )
            model_audits.append({"role": "outer", "ordinal": cell_number, **model_audit})
        ledger.commit_fold(fold)
        for cell in current_fold:
            truth, anomaly = accessor.release_committed(
                cell["validation_ids"],
                purpose=f"cell_{FRACTIONS.index(cell['fraction']) + 1:02d}_{fold}_outer_score",
                active_fold=fold,
                scope=cell["scope"],
            )
            cell["truth"] = truth
            cell["anomaly"] = anomaly
            completed_cells.append(cell)
    completion_pin = ledger.complete()
    all_validation = np.concatenate([item["validation_ids"] for item in completed_cells]).astype(
        np.int64, copy=False
    )
    accessor.release_global(
        np.unique(all_validation),
        purpose="aggregate_after_predictions_complete",
        active_fold=FOLDS[-1],
    )
    split_report = {
        "schema_version": "p1_v6r2_split_audit.v1",
        "maximum_dependency_rows": science.MAXIMUM_DEPENDENCY_ROWS,
        "purge_rows": science.PURGE_ROWS,
        "purge_days": science.PURGE_DAYS,
        "proofs": split_audits,
        "all_passed": all(item["dependency_proof"]["passed"] for item in split_audits),
    }
    contract.write_output_exclusive(capability, "split_audit.json", split_report)
    target_audit = accessor.audit()
    contract.write_output_exclusive(capability, "selective_target_audit.json", target_audit)

    fraction_points: list[dict[str, Any]] = []
    fraction_scores: dict[float, dict[str, Any]] = {}
    for point_index, fraction in enumerate(FRACTIONS):
        aggregate = _aggregate_score(
            capability, frame, [item for item in completed_cells if item["fraction"] == fraction]
        )
        fraction_scores[fraction] = aggregate
        ci90 = science.paired_bootstrap_f1_delta_ci90(
            capability=capability,
            truth=aggregate["truth"],
            incumbent_prediction=aggregate["incumbent_prediction"],
            candidate_prediction=aggregate["candidate_prediction"],
            bootstrap_unit_ids=aggregate["bootstrap_units"],
            replicates=5000,
            seed=20260823 + point_index,
        )
        contract.bump_counter(capability, "bootstrap_replicates", 5000)
        values = aggregate["metrics"]
        fraction_points.append(
            {
                "fraction": fraction,
                "micro_f1_delta": values["micro_f1_delta"],
                "ci90": ci90,
                "offset_recall_delta": values["offset_recall_delta"],
                "drift_recall_delta": values["drift_recall_delta"],
                "spike_f1_delta": values["spike_f1_delta"],
                "worst_station_layer_f1_delta": values["worst_station_layer_f1_delta"],
                "bootstrap_replicates": 5000,
                "offset_observed": aggregate["offset_observed"],
                "drift_observed": aggregate["drift_observed"],
                "spike_observed": aggregate["spike_observed"],
                "all_required_station_layers_observed": aggregate[
                    "all_required_station_layers_observed"
                ],
            }
        )
    fold_full_deltas: dict[str, float] = {}
    for fold in FOLDS:
        aggregate = _aggregate_score(
            capability,
            frame,
            [item for item in completed_cells if item["fold"] == fold and item["fraction"] == 1.0],
        )
        fold_full_deltas[fold] = aggregate["metrics"]["micro_f1_delta"]
    reproducible = all(
        item["available"] is False or item["reload_inference_byte_exact"] is True
        for item in model_audits
    )
    final_input = {
        "fraction_metrics": fraction_points,
        "fold_full_micro_f1_deltas": fold_full_deltas,
        "all_leakage_checks": split_report["all_passed"],
        "all_reproducibility_checks": reproducible,
        "all_commitments_verified": ledger.is_global_committed(),
    }
    final_gate = science.strict_final_curve_gate(final_input)
    metrics = {
        "schema_version": "p1_v6r2_metrics.v1",
        "generation": GENERATION,
        **final_input,
        "final_gate": final_gate,
    }
    evidence = {
        "schema_version": "p1_v6r2_learning_curve_evidence.v1",
        "generation": GENERATION,
        "single_preregistered_hypothesis": _CONTEXT["science_projection"]["statement"],
        "points": fraction_points,
        "fold_full_micro_f1_deltas": fold_full_deltas,
        "train_only_gates": [item["gate"] for item in completed_cells],
        "commitment": ledger.audit(),
        "completion_pin": completion_pin,
        "candidate_creation_allowed": False,
        "test_prediction_allowed": False,
        "ledger_append_allowed": False,
        "upload_allowed": False,
    }
    result = {
        "schema_version": "p1_v6r2_result.v1",
        "generation": GENERATION,
        "status": (
            "RESEARCH_ONLY_CURVE_GATE_PASS_NO_CANDIDATE"
            if final_gate["passed"]
            else "RESEARCH_ONLY_NO_PASS"
        ),
        "passed": bool(final_gate["passed"]),
        "fallback": final_gate["fallback"],
        "candidate": None,
        "test_prediction": None,
        "ledger_event": None,
        "upload": None,
        "identifiability": _CONTEXT["science_projection"]["identifiability"],
    }
    contract.enter_phase(capability, expected="curve", new="publishing")
    metrics_pin = contract.write_output_exclusive(capability, "metrics.json", metrics)
    evidence_pin = contract.write_output_exclusive(
        capability, "learning_curve_evidence.json", evidence
    )
    result_pin = contract.write_output_exclusive(capability, "result.json", result)
    resource = contract.resource_snapshot(capability)
    resource["model_audits"] = model_audits
    resource["maximum_peak_rss_bytes"] = config["resource_ceiling"]["maximum_peak_rss_bytes"]
    resource["vram_bytes"] = 0
    resource_pin = contract.write_output_exclusive(capability, "resource_audit.json", resource)
    pre_manifest = contract.verify_output_inventory(capability, final=False)
    manifest = {
        "schema_version": "p1_v6r2_manifest.v1",
        "generation": GENERATION,
        "execution_closure_sha256": contract.execution_closure_sha256(),
        "pre_manifest_inventory_sha256": pre_manifest["inventory_sha256"],
        "commitment_chain_head_sha256": ledger.audit()["chain_head_sha256"],
        "metrics": metrics_pin,
        "learning_curve_evidence": evidence_pin,
        "result": result_pin,
        "resource_audit": resource_pin,
        "candidate_created": False,
        "test_prediction_created": False,
        "ledger_appended": False,
        "uploaded": False,
    }
    manifest_pin = contract.write_output_exclusive(capability, "manifest.json", manifest)
    contract.write_output_exclusive(
        capability, "manifest.sha256", (manifest_pin["sha256"] + "\n").encode("ascii")
    )
    completed = contract.complete_capability(capability)
    return {
        "status": result["status"],
        "passed": result["passed"],
        "metrics_sha256": metrics_pin["sha256"],
        "evidence_sha256": evidence_pin["sha256"],
        "result_sha256": result_pin["sha256"],
        "manifest_sha256": manifest_pin["sha256"],
        "resource": completed,
        "candidate_created": False,
        "test_prediction_created": False,
        "ledger_appended": False,
        "uploaded": False,
    }


__all__ = ["ExecutionError", "run_curve"]
