"""Private inherited-handle target broker for P1 Gen6r4.

Only the dedicated broker process compiles this module.  Parent and worker
processes never receive its authenticated source or the target-bearing source
buffer.  Requests name a frozen phase, never row IDs; the broker derives IDs
from its independently generated cell plan and rehashes persisted evidence
before every release.
"""

from __future__ import annotations

import csv
import hashlib
import io
from typing import Any

try:
    _CONTEXT = _P1_V6R4_BROKER_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R4_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r4 target broker requires private inherited bootstrap") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("private_inherited_broker_entry") is not True
    or _CONTEXT.get("public_cli_fields_absent") is not True
):
    raise RuntimeError("P1 Gen6r4 target broker rejected non-private entry")

SOURCE_COLUMNS = (
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
SANITIZED_COLUMNS = SOURCE_COLUMNS[:-2]


class BrokerError(RuntimeError):
    """The private target protocol or persisted evidence differs."""


def _strict_source(raw: bytes) -> tuple[list[list[str]], bytes]:
    if type(raw) is not bytes or b"\x00" in raw:
        raise BrokerError("target source is not exact non-NUL bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BrokerError("target source is not strict UTF-8 CSV") from exc
    if not rows or tuple(rows[0]) != SOURCE_COLUMNS:
        raise BrokerError("target source header differs")
    body = rows[1:]
    if not body or any(len(row) != len(SOURCE_COLUMNS) for row in body):
        raise BrokerError("target source row shape differs")
    if any(row[-2] not in {"0", "1"} for row in body):
        raise BrokerError("target source label is not binary")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SANITIZED_COLUMNS)
    writer.writerows(row[:-2] for row in body)
    sanitized = stream.getvalue().encode("utf-8")
    if b"label" in sanitized.splitlines()[0] or b"anomaly_type" in sanitized.splitlines()[0]:
        raise BrokerError("target field escaped sanitization")
    return body, sanitized


def _strict_plan(plan: object, *, cell: int, size: int) -> dict[str, list[int]]:
    expected_names = {
        "inner_1_train",
        "inner_1_gate",
        "inner_2_train",
        "inner_2_gate",
        "inner_3_train",
        "inner_3_gate",
        "outer_train",
        "outer_validation",
    }
    value = contract.require_exact_keys(plan, expected_names, label=f"cell {cell} plan")
    normalized: dict[str, list[int]] = {}
    for name in sorted(expected_names):
        ids = value[name]
        contract.row_ids_sha256(ids, size=size, label=f"cell {cell} {name}")
        normalized[name] = list(ids)
    for block in contract.BLOCKS:
        contract.verify_disjoint(
            normalized[f"inner_{block}_train"],
            normalized[f"inner_{block}_gate"],
            label=f"cell {cell} inner {block}",
        )
    contract.verify_disjoint(
        normalized["outer_train"],
        normalized["outer_validation"],
        label=f"cell {cell} outer",
    )
    return normalized


class TargetBroker:
    """A single-use broker with 15 phase-only cell channels.

    There is intentionally no method accepting caller-selected row IDs.  Each
    channel owns a fixed state machine and an independently derived plan.
    """

    __slots__ = (
        "_rows",
        "_sanitized",
        "_plans",
        "_state",
        "_head",
        "_released",
        "_closed",
        "_source_sha256",
    )

    def __init__(self) -> None:
        source_reader = _CONTEXT.get("consume_authenticated_target_source_once")
        plan_deriver = _CONTEXT.get("derive_exact_cell_plan")
        if not callable(source_reader) or not callable(plan_deriver):
            raise BrokerError("private target source or plan authority is absent")
        raw = source_reader()
        rows, base_sanitized = _strict_source(raw)
        self._source_sha256 = hashlib.sha256(raw).hexdigest()
        del raw
        self._rows = rows
        self._sanitized: dict[int, bytes] = {}
        self._plans: dict[int, dict[str, list[int]]] = {}
        self._state = {cell: "inner_1_train" for cell in range(1, contract.CELL_COUNT + 1)}
        self._head: dict[int, str | None] = {
            cell: None for cell in range(1, contract.CELL_COUNT + 1)
        }
        self._released: dict[int, list[dict[str, Any]]] = {
            cell: [] for cell in range(1, contract.CELL_COUNT + 1)
        }
        self._closed = False
        base_sha = hashlib.sha256(base_sanitized).hexdigest()
        for cell in range(1, contract.CELL_COUNT + 1):
            prefix = f"P1V6R4-CELL:{cell:02d}:{base_sha}\n".encode("ascii")
            payload = prefix + base_sanitized
            self._sanitized[cell] = payload
            plan = plan_deriver(base_sanitized, cell)
            self._plans[cell] = _strict_plan(plan, cell=cell, size=len(rows))

    def sanitized_cell_buffer(self, cell: int) -> dict[str, Any]:
        """Return one target-free buffer only on its dedicated inherited channel."""

        contract.cell_identity(cell)
        channel_cell = _CONTEXT.get("active_sanitized_channel_cell")
        if not callable(channel_cell) or channel_cell() != cell:
            raise PermissionError("sanitized buffer channel is not bound to this cell")
        payload = self._sanitized[cell]
        return {
            "schema_version": "p1_v6r4_sanitized_cell_buffer.v1",
            "generation": contract.GENERATION,
            "cell": cell,
            "rows": len(self._rows),
            "payload": payload,
            "pin": {
                "path": f"inherited://sanitized-cell-{cell:02d}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }

    def bind_predecessor(self, cell: int, predecessor_pin: object) -> str:
        """Bind a new cell channel to the independently rehashed canonical head."""

        contract.cell_identity(cell)
        if self._head[cell] is not None or self._state[cell] != "inner_1_train":
            raise PermissionError("cell predecessor was already bound")
        head = self._rehash_evidence(cell, "predecessor", predecessor_pin)
        expected_head = _CONTEXT.get("canonical_predecessor_for_cell")
        if not callable(expected_head) or expected_head(cell) != head:
            raise PermissionError("broker predecessor is not the canonical persisted cell predecessor")
        self._head[cell] = head
        return head

    def release(self, cell: int, phase: str, evidence_pin: object) -> dict[str, Any]:
        """Release the exact broker-selected values for the current frozen phase.

        The API accepts no ID, slice, predicate, path, session, or prior value.
        """

        contract.cell_identity(cell)
        if self._closed or type(phase) is not str or self._state[cell] != phase:
            raise PermissionError("broker phase is closed, unknown, or out of order")
        if phase not in contract.PHASES or phase == "outer_seal":
            raise PermissionError("broker phase does not release target values")
        head = self._head[cell]
        if not contract.is_sha256(head):
            raise PermissionError("broker cell predecessor is not bound")
        evidence_head = self._rehash_evidence(cell, phase, evidence_pin)
        if phase.endswith("_gate"):
            if evidence_head == head:
                raise PermissionError("gate release did not follow a new persisted commitment")
            self._verify_commitment_semantics(cell, phase, evidence_pin, prior=head)
            self._head[cell] = evidence_head
        elif evidence_head != head:
            raise PermissionError("training release evidence is not the current persisted head")
        ids = self._plans[cell][phase]
        labels = [int(self._rows[row_id][-2]) for row_id in ids]
        anomalies = [self._rows[row_id][-1] for row_id in ids]
        event = {
            "cell": cell,
            "phase": phase,
            "row_count": len(ids),
            "row_ids_sha256": contract.deep_sha256(ids),
            "evidence_event_sha256": evidence_head,
            "label_rows_returned": len(labels),
            "anomaly_rows_returned": len(anomalies),
        }
        self._released[cell].append(event)
        self._state[cell] = self._next_phase(phase)
        return {
            "schema_version": "p1_v6r4_phase_target_release.v1",
            "generation": contract.GENERATION,
            **event,
            "labels": labels,
            "anomaly_types": anomalies,
        }

    def seal_outer(self, cell: int, commitment_pin: object) -> dict[str, Any]:
        contract.cell_identity(cell)
        if self._closed or self._state[cell] != "outer_seal":
            raise PermissionError("outer seal is out of order")
        prior = self._head[cell]
        if not contract.is_sha256(prior):
            raise PermissionError("outer seal predecessor is absent")
        head = self._rehash_evidence(cell, "outer_seal", commitment_pin)
        self._verify_commitment_semantics(cell, "outer_seal", commitment_pin, prior=prior)
        self._head[cell] = head
        self._state[cell] = "sealed"
        return {
            "schema_version": "p1_v6r4_outer_target_nonrelease.v1",
            "generation": contract.GENERATION,
            "cell": cell,
            "outer_validation_values_returned_to_worker": 0,
            "outer_commitment_sha256": head,
        }

    def audit(self) -> dict[str, Any]:
        if self._closed or set(self._state.values()) != {"sealed"}:
            raise BrokerError("target broker audit requested before all cells sealed")
        sanitized = []
        for cell in range(1, contract.CELL_COUNT + 1):
            payload = self._sanitized[cell]
            sanitized.append(
                {
                    "cell": cell,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "target_columns_absent": True,
                }
            )
        return {
            "schema_version": "p1_v6r4_target_broker_audit.v1",
            "generation": contract.GENERATION,
            "broker_processes": 1,
            "raw_target_buffer_processes": ["target_broker"],
            "parent_received_raw_target_bytes": False,
            "workers_received_raw_target_bytes": False,
            "arbitrary_id_api_present": False,
            "sanitized_cell_buffers": sanitized,
            "source_sha256": self._source_sha256,
            "release_events": [
                event
                for cell in range(1, contract.CELL_COUNT + 1)
                for event in self._released[cell]
            ],
            "outer_validation_values_returned_to_workers": 0,
        }

    def close(self) -> None:
        if set(self._state.values()) != {"sealed"}:
            raise BrokerError("target broker cannot close before exact completion")
        for row in self._rows:
            row[-2] = ""
            row[-1] = ""
        self._rows.clear()
        self._plans.clear()
        self._sanitized.clear()
        self._closed = True

    @staticmethod
    def _next_phase(phase: str) -> str:
        transitions = {
            "inner_1_train": "inner_1_gate",
            "inner_1_gate": "inner_2_train",
            "inner_2_train": "inner_2_gate",
            "inner_2_gate": "inner_3_train",
            "inner_3_train": "inner_3_gate",
            "inner_3_gate": "outer_train",
            "outer_train": "outer_seal",
        }
        try:
            return transitions[phase]
        except KeyError as exc:  # pragma: no cover - guarded above
            raise BrokerError("phase transition differs") from exc

    @staticmethod
    def _rehash_evidence(cell: int, phase: str, pin: object) -> str:
        rehash = _CONTEXT.get("rehash_persisted_event_same_handle")
        if not callable(rehash):
            raise BrokerError("persisted same-handle reader is absent")
        payload, actual_pin = rehash(pin)
        validated = contract.validate_pin(actual_pin, label=f"cell {cell} {phase} evidence")
        if validated != pin:
            raise PermissionError("persisted evidence pin changed during same-handle rehash")
        event = contract.parse_json_bytes(payload, label=f"cell {cell} {phase} evidence")
        if type(event) is not dict or not contract.is_sha256(event.get("event_sha256")):
            raise PermissionError("persisted evidence event differs")
        return event["event_sha256"]

    def _verify_commitment_semantics(
        self,
        cell: int,
        phase: str,
        pin: object,
        *,
        prior: str,
    ) -> None:
        rehash = _CONTEXT.get("rehash_persisted_event_same_handle")
        payload, _actual_pin = rehash(pin)
        event = contract.parse_json_bytes(payload, label=f"cell {cell} {phase} commitment")
        block = None
        if phase.startswith("inner_"):
            block = contract.strict_int(
                int(phase.split("_")[1]), label="broker commitment block", minimum=1
            )
            schema = "p1_v6r4_inner_commitment.v1"
            expected_ids = self._plans[cell][f"inner_{block}_gate"]
        else:
            schema = "p1_v6r4_cell_receipt.v1"
            expected_ids = self._plans[cell]["outer_validation"]
        contract.verify_event(
            event,
            schema=schema,
            prior=prior,
            label=f"cell {cell} {phase}",
        )
        if (
            event.get("cell") != cell
            or event.get("block") != block
            or event.get(
                "prediction_row_ids_sha256"
                if block is not None
                else "outer_prediction_row_ids_sha256"
            )
            != contract.deep_sha256(expected_ids)
            or event.get("prediction_rows" if block is not None else "outer_prediction_rows")
            != len(expected_ids)
        ):
            raise PermissionError("persisted commitment does not bind broker-selected IDs")


__all__ = ["BrokerError", "SANITIZED_COLUMNS", "SOURCE_COLUMNS", "TargetBroker"]
