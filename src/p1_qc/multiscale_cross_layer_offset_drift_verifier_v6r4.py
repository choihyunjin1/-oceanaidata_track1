"""Full-semantic persisted cell and final verifier for P1 Gen6r4."""

from __future__ import annotations

import math
import struct
from typing import Any

try:
    _CONTEXT = _P1_V6R4_VERIFIER_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R4_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r4 verifier requires authenticated verifier entry") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("all_owner_roles_authenticated") is not True
    or _CONTEXT.get("execution_authority_present") is not False
):
    raise RuntimeError("P1 Gen6r4 verifier context is not authority-free")

PREDICTION_MAGIC = b"P1V6R4PRED\x00"


class VerificationError(RuntimeError):
    """A persisted byte, row scope, counter, edge, or inventory differs."""


def _reader() -> Any:
    value = _CONTEXT.get("authenticated_output_bytes")
    if not callable(value):
        raise VerificationError("authority-free authenticated output reader is absent")
    return value


def _pin_reader() -> Any:
    value = _CONTEXT.get("output_pin")
    if not callable(value):
        raise VerificationError("authority-free output pin reader is absent")
    return value


def _paths_reader() -> Any:
    value = _CONTEXT.get("list_output_paths")
    if not callable(value):
        raise VerificationError("authority-free output inventory reader is absent")
    return value


def _pin(relative: str) -> dict[str, Any]:
    return contract.validate_pin(
        _pin_reader()(relative), label=f"persisted pin {relative}", expected_path=relative
    )


def _bytes(relative: str) -> tuple[bytes, dict[str, Any]]:
    pin = _pin(relative)
    payload = _reader()(pin, relative)
    if type(payload) is not bytes:
        raise VerificationError(f"persisted reader returned non-bytes for {relative}")
    if len(payload) != pin["bytes"] or contract.bytes_sha256(payload) != pin["sha256"]:
        raise VerificationError(f"persisted bytes changed for {relative}")
    return payload, pin


def _json(relative: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, pin = _bytes(relative)
    value = contract.parse_json_bytes(payload, label=relative)
    if type(value) is not dict:
        raise VerificationError(f"{relative} is not a JSON object")
    return value, pin


def encode_predictions(values: list[float]) -> bytes:
    if type(values) is not list or not values:
        raise VerificationError("prediction encoder requires a non-empty list")
    checked: list[float] = []
    for index, raw in enumerate(values):
        if type(raw) is not float or not math.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise VerificationError(f"prediction {index} is not a finite float probability")
        checked.append(raw)
    return PREDICTION_MAGIC + struct.pack("<Q", len(checked)) + struct.pack(
        f"<{len(checked)}d", *checked
    )


def verify_prediction_bytes(payload: bytes, *, expected_rows: int, label: str) -> dict[str, Any]:
    rows = contract.strict_int(expected_rows, label=f"{label} expected_rows", minimum=1)
    header_size = len(PREDICTION_MAGIC) + 8
    if (
        type(payload) is not bytes
        or not payload.startswith(PREDICTION_MAGIC)
        or len(payload) != header_size + rows * 8
    ):
        raise VerificationError(f"{label} prediction binary shape differs")
    claimed_rows = struct.unpack("<Q", payload[len(PREDICTION_MAGIC) : header_size])[0]
    if claimed_rows != rows:
        raise VerificationError(f"{label} prediction row header differs")
    values = struct.unpack(f"<{rows}d", payload[header_size:])
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise VerificationError(f"{label} contains invalid probabilities")
    return {
        "rows": rows,
        "minimum": min(values),
        "maximum": max(values),
        "sha256": contract.bytes_sha256(payload),
    }


def _expected_plan(cell: int) -> dict[str, list[int]]:
    derive = _CONTEXT.get("derive_expected_cell_plan_from_sanitized_buffer")
    if not callable(derive):
        raise VerificationError("independent sanitized split derivation is absent")
    plan = derive(cell)
    names = {
        "inner_1_train",
        "inner_1_gate",
        "inner_2_train",
        "inner_2_gate",
        "inner_3_train",
        "inner_3_gate",
        "outer_train",
        "outer_validation",
    }
    value = contract.require_exact_keys(plan, names, label=f"expected cell {cell} plan")
    size_reader = _CONTEXT.get("sanitized_row_count")
    size = size_reader(cell) if callable(size_reader) else None
    size = contract.strict_int(size, label=f"cell {cell} sanitized rows", minimum=1)
    result: dict[str, list[int]] = {}
    for name in names:
        ids = value[name]
        contract.row_ids_sha256(ids, size=size, label=f"expected cell {cell} {name}")
        result[name] = list(ids)
    for block in contract.BLOCKS:
        contract.verify_disjoint(
            result[f"inner_{block}_train"],
            result[f"inner_{block}_gate"],
            label=f"cell {cell} inner {block}",
        )
    contract.verify_disjoint(
        result["outer_train"], result["outer_validation"], label=f"cell {cell} outer"
    )
    return result


def _verify_row_scope(
    event: dict[str, Any],
    *,
    prefix: str,
    expected: list[int],
    size: int,
    label: str,
) -> None:
    ids = event.get(f"{prefix}_row_ids")
    digest = contract.row_ids_sha256(ids, size=size, label=f"{label} {prefix}")
    if (
        ids != expected
        or event.get(f"{prefix}_rows") != len(expected)
        or event.get(f"{prefix}_row_ids_sha256") != digest
    ):
        raise VerificationError(f"{label} {prefix} row scope differs")


def _verify_model(
    relative: str,
    *,
    cell: int,
    block: int | None,
    train_ids: list[int],
    sanitized_pin: dict[str, Any],
    teacher_receipts: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    model, pin = _json(relative)
    expected_keys = {
        "schema_version",
        "generation",
        "cell",
        "block",
        "fold",
        "fraction_tag",
        "science_projection_sha256",
        "sanitized_input",
        "train_rows",
        "train_row_ids",
        "train_row_ids_sha256",
        "teacher_receipts",
        "baseline_model",
        "unary_model",
        "nonnegative_residual_guard",
    }
    contract.require_exact_keys(model, expected_keys, label=relative)
    fold, _fraction, fraction_tag = contract.cell_identity(cell)
    _verify_row_scope(
        model,
        prefix="train",
        expected=train_ids,
        size=size,
        label=relative,
    )
    if (
        model["schema_version"] != "p1_v6r4_frozen_science_model.v1"
        or model["generation"] != contract.GENERATION
        or model["cell"] != cell
        or model["block"] != block
        or model["fold"] != fold
        or model["fraction_tag"] != fraction_tag
        or not contract.is_sha256(model["science_projection_sha256"])
        or model["sanitized_input"] != sanitized_pin
        or model["teacher_receipts"] != teacher_receipts
        or type(model["baseline_model"]) is not dict
        or type(model["unary_model"]) is not dict
        or model["nonnegative_residual_guard"] is not True
    ):
        raise VerificationError(f"{relative} model semantics differ")
    return pin


def _verify_teacher_bindings(
    raw: object,
    *,
    fold: str,
    fraction_tag: str,
    block: int,
    prediction_ids_sha256: str,
    train_ids_sha256: str,
    prediction_rows: int,
    train_rows: int,
) -> list[dict[str, Any]]:
    if type(raw) is not list or len(raw) != 3:
        raise VerificationError("inner model does not consume exactly three teacher receipts")
    catalog_reader = _CONTEXT.get("teacher_receipt_catalog")
    catalog = catalog_reader() if callable(catalog_reader) else None
    if type(catalog) is not dict:
        raise VerificationError("authenticated teacher receipt catalog is absent")
    result: list[dict[str, Any]] = []
    for index, seed in enumerate(contract.SEEDS):
        expected = catalog.get((fold, fraction_tag, block, seed))
        if type(expected) is not dict or raw[index] != expected:
            raise VerificationError("teacher receipt identity or order differs")
        if (
            expected.get("prediction_ids_sha256") != prediction_ids_sha256
            or expected.get("train_ids_sha256") != train_ids_sha256
            or expected.get("prediction_rows") != prediction_rows
            or expected.get("train_rows") != train_rows
        ):
            raise VerificationError("teacher receipt split binding differs")
        result.append(expected)
    return result


def verify_cell_artifact_graph(
    *,
    cell: int,
    prior_event_sha256: str,
    sanitized_pin: object,
) -> dict[str, Any]:
    """Recompute every byte/content/split/teacher/predecessor edge for one cell."""

    contract.cell_identity(cell)
    if not contract.is_sha256(prior_event_sha256):
        raise VerificationError("cell predecessor hash differs")
    sanitized = contract.validate_pin(sanitized_pin, label=f"cell {cell} sanitized pin")
    expected_paths = set(contract.expected_cell_paths(cell))
    present = set(_paths_reader()())
    if not expected_paths <= present:
        raise VerificationError(f"cell {cell} output paths are incomplete")
    plan = _expected_plan(cell)
    size_reader = _CONTEXT.get("sanitized_row_count")
    size = contract.strict_int(size_reader(cell), label=f"cell {cell} row count", minimum=1)
    fold, _fraction, fraction_tag = contract.cell_identity(cell)
    prior = prior_event_sha256
    inner_pins: list[dict[str, Any]] = []
    consumed_receipts = 0
    for block in contract.BLOCKS:
        train_ids = plan[f"inner_{block}_train"]
        prediction_ids = plan[f"inner_{block}_gate"]
        train_sha = contract.deep_sha256(train_ids)
        prediction_sha = contract.deep_sha256(prediction_ids)
        relative = f"cells/cell_{cell:02d}/commitments/inner_{block}.json"
        event, event_pin = _json(relative)
        expected_keys = {
            "schema_version",
            "generation",
            "prior_event_sha256",
            "event_sha256",
            "cell",
            "block",
            "fold",
            "fraction_tag",
            "sanitized_input",
            "model",
            "prediction",
            "train_rows",
            "train_row_ids",
            "train_row_ids_sha256",
            "prediction_rows",
            "prediction_row_ids",
            "prediction_row_ids_sha256",
            "teacher_receipts",
            "target_scalars_decoded_before_commitment",
        }
        contract.require_exact_keys(event, expected_keys, label=relative)
        head = contract.verify_event(
            event,
            schema="p1_v6r4_inner_commitment.v1",
            prior=prior,
            label=relative,
        )
        _verify_row_scope(event, prefix="train", expected=train_ids, size=size, label=relative)
        _verify_row_scope(
            event,
            prefix="prediction",
            expected=prediction_ids,
            size=size,
            label=relative,
        )
        teacher = _verify_teacher_bindings(
            event["teacher_receipts"],
            fold=fold,
            fraction_tag=fraction_tag,
            block=block,
            prediction_ids_sha256=prediction_sha,
            train_ids_sha256=train_sha,
            prediction_rows=len(prediction_ids),
            train_rows=len(train_ids),
        )
        model_relative = f"cells/cell_{cell:02d}/models/inner_{block}.json"
        model_pin = _verify_model(
            model_relative,
            cell=cell,
            block=block,
            train_ids=train_ids,
            sanitized_pin=sanitized,
            teacher_receipts=teacher,
            size=size,
        )
        prediction_relative = f"cells/cell_{cell:02d}/inner_predictions/block_{block}.bin"
        prediction_bytes, prediction_pin = _bytes(prediction_relative)
        verify_prediction_bytes(
            prediction_bytes,
            expected_rows=len(prediction_ids),
            label=prediction_relative,
        )
        if (
            event["cell"] != cell
            or event["block"] != block
            or event["fold"] != fold
            or event["fraction_tag"] != fraction_tag
            or event["sanitized_input"] != sanitized
            or event["model"] != model_pin
            or event["prediction"] != prediction_pin
            or event["target_scalars_decoded_before_commitment"] != 0
            or type(event["target_scalars_decoded_before_commitment"]) is not int
        ):
            raise VerificationError(f"{relative} commitment semantics differ")
        inner_pins.append(event_pin)
        consumed_receipts += len(teacher)
        prior = head

    receipt_relative = f"cells/cell_{cell:02d}/cell_receipt.json"
    receipt, receipt_pin = _json(receipt_relative)
    expected_receipt_keys = {
        "schema_version",
        "generation",
        "prior_event_sha256",
        "event_sha256",
        "cell",
        "fold",
        "fraction_tag",
        "sanitized_input",
        "inner_commitments",
        "outer_model",
        "outer_prediction",
        "outer_train_rows",
        "outer_train_row_ids",
        "outer_train_row_ids_sha256",
        "outer_prediction_rows",
        "outer_prediction_row_ids",
        "outer_prediction_row_ids_sha256",
        "outer_validation_target_values_returned_to_worker",
        "broker_release_phases",
        "worker_counters",
    }
    contract.require_exact_keys(receipt, expected_receipt_keys, label=receipt_relative)
    head = contract.verify_event(
        receipt,
        schema="p1_v6r4_cell_receipt.v1",
        prior=prior,
        label=receipt_relative,
    )
    _verify_row_scope(
        receipt,
        prefix="outer_train",
        expected=plan["outer_train"],
        size=size,
        label=receipt_relative,
    )
    _verify_row_scope(
        receipt,
        prefix="outer_prediction",
        expected=plan["outer_validation"],
        size=size,
        label=receipt_relative,
    )
    outer_model_relative = f"cells/cell_{cell:02d}/models/outer.json"
    outer_model_pin = _verify_model(
        outer_model_relative,
        cell=cell,
        block=None,
        train_ids=plan["outer_train"],
        sanitized_pin=sanitized,
        teacher_receipts=[],
        size=size,
    )
    outer_prediction_relative = f"cells/cell_{cell:02d}/outer_prediction.bin"
    outer_bytes, outer_prediction_pin = _bytes(outer_prediction_relative)
    verify_prediction_bytes(
        outer_bytes,
        expected_rows=len(plan["outer_validation"]),
        label=outer_prediction_relative,
    )
    expected_worker_counters = {
        "baseline_fits": 4,
        "unary_fits": 4,
        "top_level_fits": 8,
        "predictions": 4,
        "inner_commitments": 3,
        "cell_commitments": 1,
        "teacher_receipts_consumed": 9,
        "scores": 4,
        "broker_target_release_phases": 7,
    }
    counters = contract.require_exact_keys(
        receipt["worker_counters"], expected_worker_counters, label="worker counters"
    )
    for name, expected in expected_worker_counters.items():
        actual = contract.strict_int(counters[name], label=f"worker counter {name}")
        if actual != expected:
            raise VerificationError(f"worker counter {name} differs")
    if (
        receipt["cell"] != cell
        or receipt["fold"] != fold
        or receipt["fraction_tag"] != fraction_tag
        or receipt["sanitized_input"] != sanitized
        or receipt["inner_commitments"] != inner_pins
        or receipt["outer_model"] != outer_model_pin
        or receipt["outer_prediction"] != outer_prediction_pin
        or receipt["outer_validation_target_values_returned_to_worker"] != 0
        or type(receipt["outer_validation_target_values_returned_to_worker"]) is not int
        or receipt["broker_release_phases"] != list(contract.PHASES[:-1])
        or consumed_receipts != 9
    ):
        raise VerificationError("cell receipt full semantics differ")
    return {
        "cell": cell,
        "fold": fold,
        "fraction_tag": fraction_tag,
        "head_event_sha256": head,
        "receipt": receipt_pin,
        "teacher_receipts_consumed": consumed_receipts,
        "worker_counters": dict(counters),
        "verified_paths": sorted(expected_paths),
    }


def _verify_metric_document(relative: str, *, expected_scope: dict[str, Any]) -> dict[str, Any]:
    value, pin = _json(relative)
    expected_keys = {"schema_version", "generation", "scope", "rows", "precision", "recall", "f1"}
    contract.require_exact_keys(value, expected_keys, label=relative)
    if (
        value["schema_version"] != "p1_v6r4_metric.v1"
        or value["generation"] != contract.GENERATION
        or value["scope"] != expected_scope
    ):
        raise VerificationError(f"{relative} metric identity differs")
    contract.strict_int(value["rows"], label=f"{relative} rows", minimum=1)
    for name in ("precision", "recall", "f1"):
        metric = value[name]
        if type(metric) is not float or not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise VerificationError(f"{relative} {name} differs")
    return pin


def _verify_inventory_rows(
    rows: object,
    *,
    expected_paths: set[str],
    label: str,
) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != len(expected_paths):
        raise VerificationError(f"{label} inventory cardinality differs")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        pin = contract.validate_pin(raw, label=f"{label} row {index}")
        path = pin["path"]
        if path in seen or path not in expected_paths or pin != _pin(path):
            raise VerificationError(f"{label} inventory row differs")
        seen.add(path)
        result.append(pin)
    if seen != expected_paths:
        raise VerificationError(f"{label} inventory paths differ")
    return result


def verify_final_output() -> dict[str, Any]:
    """Verify all 202 paths and every semantic edge independently."""

    expected_order = list(contract.expected_output_paths())
    expected = set(expected_order)
    present_raw = _paths_reader()()
    if type(present_raw) is not list or any(type(path) is not str for path in present_raw):
        raise VerificationError("output inventory reader returned non-paths")
    present = set(present_raw)
    if len(present_raw) != len(present) or present != expected or len(present) != 202:
        raise VerificationError("final output tree is not exactly the canonical 202 paths")

    session, _session_pin = _json("commitments/session.json")
    session_keys = {
        "schema_version",
        "generation",
        "prior_event_sha256",
        "event_sha256",
        "session_sha256",
        "config",
        "qa_receipt",
        "authorization",
        "attempt_lock",
        "encoded_launch_provenance",
        "target_broker_processes",
        "worker_processes",
        "sanitized_cell_buffers",
    }
    contract.require_exact_keys(session, session_keys, label="session")
    head = contract.verify_event(
        session,
        schema="p1_v6r4_session.v1",
        prior=contract.GENESIS_SHA256,
        label="session",
    )
    if (
        not contract.is_sha256(session["session_sha256"])
        or session["target_broker_processes"] != 1
        or type(session["target_broker_processes"]) is not int
        or session["worker_processes"] != 15
        or type(session["worker_processes"]) is not int
        or type(session["sanitized_cell_buffers"]) is not list
        or len(session["sanitized_cell_buffers"]) != 15
    ):
        raise VerificationError("session process/buffer semantics differ")
    sanitized_by_cell: dict[int, dict[str, Any]] = {}
    for cell, raw in enumerate(session["sanitized_cell_buffers"], start=1):
        pin = contract.validate_pin(raw, label=f"session sanitized cell {cell}")
        if pin["path"] != f"inherited://sanitized-cell-{cell:02d}":
            raise VerificationError("session sanitized cell path differs")
        sanitized_by_cell[cell] = pin

    cell_results: list[dict[str, Any]] = []
    fold_pins: list[dict[str, Any]] = []
    for cell in range(1, 16):
        result = verify_cell_artifact_graph(
            cell=cell,
            prior_event_sha256=head,
            sanitized_pin=sanitized_by_cell[cell],
        )
        head = result["head_event_sha256"]
        cell_results.append(result)
        if cell % 5 == 0:
            fold = contract.FOLDS[cell // 5 - 1]
            relative = f"commitments/fold_{fold}.json"
            event, pin = _json(relative)
            keys = {
                "schema_version",
                "generation",
                "prior_event_sha256",
                "event_sha256",
                "fold",
                "cells",
                "cell_receipts",
                "outer_prediction_rows",
                "outer_prediction_row_ids_sha256",
                "outer_validation_targets_released_to_parent",
            }
            contract.require_exact_keys(event, keys, label=relative)
            new_head = contract.verify_event(
                event,
                schema="p1_v6r4_fold_commitment.v1",
                prior=head,
                label=relative,
            )
            expected_cells = list(range(cell - 4, cell + 1))
            expected_receipts = [row["receipt"] for row in cell_results[-5:]]
            expected_plan = _expected_plan(cell - 4)["outer_validation"]
            if any(_expected_plan(item)["outer_validation"] != expected_plan for item in expected_cells):
                raise VerificationError("fraction cells do not share the exact fold validation scope")
            if (
                event["fold"] != fold
                or event["cells"] != expected_cells
                or event["cell_receipts"] != expected_receipts
                or event["outer_prediction_rows"] != len(expected_plan)
                or event["outer_prediction_row_ids_sha256"] != contract.deep_sha256(expected_plan)
                or event["outer_validation_targets_released_to_parent"] != 0
                or type(event["outer_validation_targets_released_to_parent"]) is not int
            ):
                raise VerificationError("fold commitment full semantics differ")
            fold_pins.append(pin)
            head = new_head

    predictions_complete, predictions_pin = _json("commitments/predictions_complete.json")
    complete_keys = {
        "schema_version",
        "generation",
        "prior_event_sha256",
        "event_sha256",
        "fold_commitments",
        "cell_receipts",
        "prediction_files",
        "teacher_receipts_consumed",
    }
    contract.require_exact_keys(predictions_complete, complete_keys, label="predictions_complete")
    head = contract.verify_event(
        predictions_complete,
        schema="p1_v6r4_predictions_complete.v1",
        prior=head,
        label="predictions_complete",
    )
    expected_prediction_pins = [
        _pin(path)
        for path in expected
        if path.endswith("outer_prediction.bin") or "/inner_predictions/" in path
    ]
    expected_prediction_pins.sort(key=lambda item: item["path"])
    if (
        predictions_complete["fold_commitments"] != fold_pins
        or predictions_complete["cell_receipts"] != [row["receipt"] for row in cell_results]
        or predictions_complete["prediction_files"] != expected_prediction_pins
        or predictions_complete["teacher_receipts_consumed"] != 135
        or type(predictions_complete["teacher_receipts_consumed"]) is not int
    ):
        raise VerificationError("predictions_complete graph differs")

    for fold in contract.FOLDS:
        _verify_metric_document(f"metrics/fold_{fold}.json", expected_scope={"fold": fold})
    for tag in contract.FRACTION_TAGS:
        _verify_metric_document(
            f"metrics/fraction_{tag}.json", expected_scope={"fraction_tag": tag}
        )

    split_audit, _split_pin = _json("split_audit.json")
    contract.require_exact_keys(
        split_audit,
        ("schema_version", "generation", "cells"),
        label="split audit",
    )
    expected_split_rows = [
        {
            "cell": cell,
            "fold": contract.cell_identity(cell)[0],
            "fraction_tag": contract.cell_identity(cell)[2],
            "plan": _expected_plan(cell),
        }
        for cell in range(1, 16)
    ]
    if (
        split_audit["schema_version"] != "p1_v6r4_split_audit.v1"
        or split_audit["generation"] != contract.GENERATION
        or split_audit["cells"] != expected_split_rows
    ):
        raise VerificationError("split audit does not reproduce all row scopes")

    broker_audit, _broker_pin = _json("target_broker_audit.json")
    broker_keys = {
        "schema_version",
        "generation",
        "broker_processes",
        "raw_target_buffer_processes",
        "parent_received_raw_target_bytes",
        "workers_received_raw_target_bytes",
        "arbitrary_id_api_present",
        "sanitized_cell_buffers",
        "source_sha256",
        "release_events",
        "outer_validation_values_returned_to_workers",
    }
    contract.require_exact_keys(broker_audit, broker_keys, label="target broker audit")
    expected_sanitized = [
        {
            "cell": cell,
            "bytes": sanitized_by_cell[cell]["bytes"],
            "sha256": sanitized_by_cell[cell]["sha256"],
            "target_columns_absent": True,
        }
        for cell in range(1, 16)
    ]
    if (
        broker_audit["schema_version"] != "p1_v6r4_target_broker_audit.v1"
        or broker_audit["generation"] != contract.GENERATION
        or broker_audit["broker_processes"] != 1
        or type(broker_audit["broker_processes"]) is not int
        or broker_audit["raw_target_buffer_processes"] != ["target_broker"]
        or broker_audit["parent_received_raw_target_bytes"] is not False
        or broker_audit["workers_received_raw_target_bytes"] is not False
        or broker_audit["arbitrary_id_api_present"] is not False
        or broker_audit["sanitized_cell_buffers"] != expected_sanitized
        or not contract.is_sha256(broker_audit["source_sha256"])
        or type(broker_audit["release_events"]) is not list
        or len(broker_audit["release_events"]) != 105
        or broker_audit["outer_validation_values_returned_to_workers"] != 0
        or type(broker_audit["outer_validation_values_returned_to_workers"]) is not int
    ):
        raise VerificationError("target broker audit full semantics differ")

    metrics, _metrics_pin = _json("metrics.json")
    metrics_keys = {"schema_version", "generation", "score_decomposition", "counters"}
    contract.require_exact_keys(metrics, metrics_keys, label="metrics")
    if (
        metrics["schema_version"] != "p1_v6r4_metrics.v1"
        or metrics["generation"] != contract.GENERATION
        or metrics["score_decomposition"] != contract.SCORE_DECOMPOSITION
    ):
        raise VerificationError("metrics score decomposition differs")
    counters = contract.verify_completion_counters(metrics["counters"])

    curve, _curve_pin = _json("learning_curve_evidence.json")
    if (
        curve.get("schema_version") != "p1_v6r4_learning_curve_evidence.v1"
        or curve.get("generation") != contract.GENERATION
        or curve.get("folds") != list(contract.FOLDS)
        or curve.get("fraction_tags") != list(contract.FRACTION_TAGS)
        or curve.get("nonnegative_guards_all_true") is not True
    ):
        raise VerificationError("learning-curve evidence semantics differ")

    result, _result_pin = _json("result.json")
    if (
        result.get("schema_version") != "p1_v6r4_result.v1"
        or result.get("generation") != contract.GENERATION
        or result.get("research_only") is not True
        or result.get("candidate_created") is not False
        or result.get("test_values_read") != 0
        or type(result.get("test_values_read")) is not int
        or result.get("ledger_appended") is not False
        or result.get("uploaded") is not False
        or result.get("predictions_complete") != predictions_pin
    ):
        raise VerificationError("result fail-closed semantics differ")

    before_manifest = expected - {
        "manifest.json",
        "manifest.sha256",
        "preseal.json",
        "final_seal.json",
    }
    manifest, manifest_pin = _json("manifest.json")
    contract.require_exact_keys(
        manifest,
        ("schema_version", "generation", "files"),
        label="manifest",
    )
    if (
        manifest["schema_version"] != "p1_v6r4_manifest.v1"
        or manifest["generation"] != contract.GENERATION
    ):
        raise VerificationError("manifest identity differs")
    _verify_inventory_rows(manifest["files"], expected_paths=before_manifest, label="manifest")

    sidecar_bytes, sidecar_pin = _bytes("manifest.sha256")
    if sidecar_bytes != f"{manifest_pin['sha256']}  manifest.json\n".encode("ascii"):
        raise VerificationError("manifest sidecar does not bind manifest bytes")

    preseal, preseal_pin = _json("preseal.json")
    preseal_keys = {
        "schema_version",
        "generation",
        "prior_event_sha256",
        "event_sha256",
        "manifest",
        "manifest_sidecar",
        "inventory",
        "inventory_count",
        "counters",
    }
    contract.require_exact_keys(preseal, preseal_keys, label="preseal")
    preseal_head = contract.verify_event(
        preseal,
        schema="p1_v6r4_preseal.v1",
        prior=head,
        label="preseal",
    )
    preseal_paths = expected - {"preseal.json", "final_seal.json"}
    # Recompute the complete preseal inventory, including manifest and sidecar.
    _verify_inventory_rows(preseal["inventory"], expected_paths=preseal_paths, label="preseal")
    if (
        preseal["manifest"] != manifest_pin
        or preseal["manifest_sidecar"] != sidecar_pin
        or preseal["inventory_count"] != 200
        or type(preseal["inventory_count"]) is not int
        or preseal["counters"] != counters
    ):
        raise VerificationError("preseal full semantics differ")

    seal, seal_pin = _json("final_seal.json")
    seal_keys = {
        "schema_version",
        "generation",
        "prior_event_sha256",
        "event_sha256",
        "preseal",
        "inventory",
        "inventory_count",
        "final_file_count",
        "counters",
        "candidate_created",
        "test_values_read",
        "ledger_appended",
        "uploaded",
    }
    contract.require_exact_keys(seal, seal_keys, label="final seal")
    final_head = contract.verify_event(
        seal,
        schema="p1_v6r4_final_seal.v1",
        prior=preseal_head,
        label="final seal",
    )
    final_inventory_paths = expected - {"final_seal.json"}
    _verify_inventory_rows(
        seal["inventory"], expected_paths=final_inventory_paths, label="final seal"
    )
    if (
        seal["preseal"] != preseal_pin
        or seal["inventory_count"] != 201
        or type(seal["inventory_count"]) is not int
        or seal["final_file_count"] != 202
        or type(seal["final_file_count"]) is not int
        or seal["counters"] != counters
        or seal["candidate_created"] is not False
        or seal["test_values_read"] != 0
        or type(seal["test_values_read"]) is not int
        or seal["ledger_appended"] is not False
        or seal["uploaded"] is not False
    ):
        raise VerificationError("final seal full semantics differ")
    return {
        "schema_version": "p1_v6r4_full_semantic_verification.v1",
        "generation": contract.GENERATION,
        "verified_paths": len(expected),
        "verified_cells": len(cell_results),
        "teacher_receipts_consumed": sum(
            row["teacher_receipts_consumed"] for row in cell_results
        ),
        "score_calls": counters["scores"],
        "final_event_sha256": final_head,
        "final_seal": seal_pin,
    }


__all__ = [
    "PREDICTION_MAGIC",
    "VerificationError",
    "encode_predictions",
    "verify_cell_artifact_graph",
    "verify_final_output",
    "verify_prediction_bytes",
]
