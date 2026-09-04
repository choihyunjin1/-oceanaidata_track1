"""Brokered, private-entry execution graph for P1 multiscale Gen6r4.

This module never receives target-bearing train bytes.  A worker receives one
cell-bound sanitized buffer and uses a phase-only inherited broker channel.
The parent receives target-free pins, predictions, and aggregate score records;
it never receives labels or anomaly_type values.
"""

from __future__ import annotations

from typing import Any

try:
    _CONTEXT = _P1_V6R4_ENGINE_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R4_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
    verifier = _P1_V6R4_AUTH_VERIFIER  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r4 engine requires authenticated private bootstrap") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("all_owner_roles_authenticated") is not True
    or _CONTEXT.get("raw_target_bytes_available") is not False
    or _CONTEXT.get("public_worker_cli_available") is not False
):
    raise RuntimeError("P1 Gen6r4 engine rejected unsafe bootstrap context")


class ExecutionError(RuntimeError):
    """The private brokered research execution failed closed."""


def _service(name: str) -> Any:
    value = _CONTEXT.get(name)
    if not callable(value):
        raise ExecutionError(f"private execution service {name!r} is absent")
    return value


def _write(relative: str, value: bytes | dict[str, Any]) -> dict[str, Any]:
    payload = value if type(value) is bytes else contract.canonical_json_bytes(value) + b"\n"
    pin = _service("write_output_exclusive")(relative, payload)
    return contract.validate_pin(pin, label=f"write result {relative}", expected_path=relative)


def _worker_session(cell: int) -> dict[str, Any]:
    value = _service("private_session_snapshot")()
    expected_keys = {
        "role",
        "cell",
        "session_sha256",
        "encoded_launch_provenance",
        "inherited_handle_identity",
    }
    session = contract.require_exact_keys(value, expected_keys, label="worker private session")
    if (
        session["role"] != "cell_worker"
        or session["cell"] != cell
        or not contract.is_sha256(session["session_sha256"])
        or not contract.is_sha256(session["inherited_handle_identity"])
        or type(session["encoded_launch_provenance"]) is not dict
    ):
        raise PermissionError("worker private inherited session differs")
    return session


def _model_document(
    *,
    cell: int,
    block: int | None,
    sanitized_pin: dict[str, Any],
    train_ids: list[int],
    teacher_receipts: list[dict[str, Any]],
    fit: dict[str, Any],
) -> dict[str, Any]:
    fold, _fraction, fraction_tag = contract.cell_identity(cell)
    fit_value = contract.require_exact_keys(
        fit,
        (
            "baseline_model",
            "unary_model",
            "science_projection_sha256",
            "nonnegative_residual_guard",
            "probabilities",
        ),
        label="frozen science fit",
    )
    probabilities = fit_value["probabilities"]
    if type(probabilities) is not list or not probabilities:
        raise ExecutionError("frozen science probabilities differ")
    if fit_value["nonnegative_residual_guard"] is not True:
        raise ExecutionError("frozen >=0 residual guard failed")
    return {
        "schema_version": "p1_v6r4_frozen_science_model.v1",
        "generation": contract.GENERATION,
        "cell": cell,
        "block": block,
        "fold": fold,
        "fraction_tag": fraction_tag,
        "science_projection_sha256": fit_value["science_projection_sha256"],
        "sanitized_input": sanitized_pin,
        "train_rows": len(train_ids),
        "train_row_ids": train_ids,
        "train_row_ids_sha256": contract.deep_sha256(train_ids),
        "teacher_receipts": teacher_receipts,
        "baseline_model": fit_value["baseline_model"],
        "unary_model": fit_value["unary_model"],
        "nonnegative_residual_guard": True,
    }


def run_cell(*, cell: int) -> dict[str, Any]:
    """Run one inherited-handle-only worker against one phase-only broker channel."""

    contract.cell_identity(cell)
    session = _worker_session(cell)
    buffer_envelope = _service("receive_cell_sanitized_buffer_once")(cell)
    envelope_keys = {"schema_version", "generation", "cell", "rows", "payload", "pin"}
    envelope = contract.require_exact_keys(
        buffer_envelope, envelope_keys, label="sanitized buffer envelope"
    )
    if (
        envelope["schema_version"] != "p1_v6r4_sanitized_cell_buffer.v1"
        or envelope["generation"] != contract.GENERATION
        or envelope["cell"] != cell
    ):
        raise ExecutionError("sanitized buffer envelope identity differs")
    sanitized = contract.verify_sanitized_buffer(
        envelope["payload"], cell=cell, expected_pin=envelope["pin"]
    )
    if sanitized["rows"] != envelope["rows"]:
        raise ExecutionError("sanitized buffer row count differs")
    sanitized_pin = sanitized["pin"]
    frame = _service("load_target_free_frame")(envelope["payload"], cell)
    plan = _service("derive_exact_cell_plan_from_sanitized_buffer")(cell)
    plan = contract.require_exact_keys(
        plan,
        (
            "inner_1_train",
            "inner_1_gate",
            "inner_2_train",
            "inner_2_gate",
            "inner_3_train",
            "inner_3_gate",
            "outer_train",
            "outer_validation",
        ),
        label="worker exact cell plan",
    )
    for name, ids in plan.items():
        contract.row_ids_sha256(ids, size=sanitized["rows"], label=f"worker {name}")

    predecessor_pin = _service("receive_canonical_predecessor_pin_once")(cell)
    predecessor_event = contract.parse_json_bytes(
        _service("authenticated_output_bytes")(predecessor_pin, "worker predecessor"),
        label="worker predecessor",
    )
    if type(predecessor_event) is not dict or not contract.is_sha256(
        predecessor_event.get("event_sha256")
    ):
        raise ExecutionError("worker predecessor event differs")
    prior = predecessor_event["event_sha256"]
    if _service("validate_persisted_canonical_next_cell")(
        cell, prior, session["session_sha256"]
    ) is not True:
        raise PermissionError("worker is not the canonical persisted next cell")
    if _service("broker_bind_predecessor")(cell, predecessor_pin) != prior:
        raise PermissionError("target broker predecessor binding differs")

    fold, _fraction, fraction_tag = contract.cell_identity(cell)
    inner_pins: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    counters = {
        "baseline_fits": 0,
        "unary_fits": 0,
        "top_level_fits": 0,
        "predictions": 0,
        "inner_commitments": 0,
        "cell_commitments": 0,
        "teacher_receipts_consumed": 0,
        "scores": 0,
        "broker_target_release_phases": 0,
    }
    evidence_pin = predecessor_pin
    for block in contract.BLOCKS:
        train_phase = f"inner_{block}_train"
        gate_phase = f"inner_{block}_gate"
        train_release = _service("broker_release_phase")(
            cell, train_phase, evidence_pin
        )
        if (
            train_release.get("phase") != train_phase
            or train_release.get("row_ids_sha256") != contract.deep_sha256(plan[train_phase])
            or train_release.get("row_count") != len(plan[train_phase])
        ):
            raise ExecutionError("broker-selected inner train scope differs")
        counters["broker_target_release_phases"] += 1
        teacher = _service("consume_exact_teacher_receipts")(
            fold,
            fraction_tag,
            block,
            plan[train_phase],
            plan[gate_phase],
        )
        if type(teacher) is not dict or type(teacher.get("receipts")) is not list:
            raise ExecutionError("teacher receipt service differs")
        receipts = teacher["receipts"]
        if len(receipts) != 3:
            raise ExecutionError("worker did not consume three teacher receipts")
        fit = _service("frozen_science_fit_predict")(
            frame=frame,
            train_ids=plan[train_phase],
            prediction_ids=plan[gate_phase],
            labels=train_release["labels"],
            anomaly_types=train_release["anomaly_types"],
            incumbent_probability=teacher["probabilities"],
            cell=cell,
            block=block,
        )
        model_relative = f"cells/cell_{cell:02d}/models/inner_{block}.json"
        model = _model_document(
            cell=cell,
            block=block,
            sanitized_pin=sanitized_pin,
            train_ids=plan[train_phase],
            teacher_receipts=receipts,
            fit=fit,
        )
        model_pin = _write(model_relative, model)
        prediction_relative = f"cells/cell_{cell:02d}/inner_predictions/block_{block}.bin"
        prediction_pin = _write(
            prediction_relative, verifier.encode_predictions(fit["probabilities"])
        )
        commitment = contract.event_body(
            "p1_v6r4_inner_commitment.v1",
            prior,
            {
                "cell": cell,
                "block": block,
                "fold": fold,
                "fraction_tag": fraction_tag,
                "sanitized_input": sanitized_pin,
                "model": model_pin,
                "prediction": prediction_pin,
                "train_rows": len(plan[train_phase]),
                "train_row_ids": plan[train_phase],
                "train_row_ids_sha256": contract.deep_sha256(plan[train_phase]),
                "prediction_rows": len(plan[gate_phase]),
                "prediction_row_ids": plan[gate_phase],
                "prediction_row_ids_sha256": contract.deep_sha256(plan[gate_phase]),
                "teacher_receipts": receipts,
                "target_scalars_decoded_before_commitment": 0,
            },
        )
        commitment_relative = f"cells/cell_{cell:02d}/commitments/inner_{block}.json"
        commitment_pin = _write(commitment_relative, commitment)
        prior = commitment["event_sha256"]
        gate_release = _service("broker_release_phase")(cell, gate_phase, commitment_pin)
        if (
            gate_release.get("phase") != gate_phase
            or gate_release.get("row_ids_sha256") != contract.deep_sha256(plan[gate_phase])
            or gate_release.get("evidence_event_sha256") != prior
        ):
            raise ExecutionError("broker-selected inner gate scope differs")
        score = _service("frozen_science_inner_score")(
            truth=gate_release["labels"],
            anomaly_types=gate_release["anomaly_types"],
            prediction=fit["predictions"],
            probabilities=fit["probabilities"],
            cell=cell,
            block=block,
        )
        if type(score) is not dict:
            raise ExecutionError("inner score record differs")
        inner_scores.append(score)
        inner_pins.append(commitment_pin)
        evidence_pin = commitment_pin
        counters["baseline_fits"] += 1
        counters["unary_fits"] += 1
        counters["top_level_fits"] += 2
        counters["predictions"] += 1
        counters["inner_commitments"] += 1
        counters["teacher_receipts_consumed"] += 3
        counters["scores"] += 1
        counters["broker_target_release_phases"] += 1

    gate = _service("frozen_science_inner_aggregate_gate")(inner_scores, cell)
    if type(gate) is not dict or type(gate.get("passed")) is not bool:
        raise ExecutionError("three-block inner aggregate gate differs")
    counters["scores"] += 1
    outer_release = _service("broker_release_phase")(cell, "outer_train", evidence_pin)
    if (
        outer_release.get("phase") != "outer_train"
        or outer_release.get("row_ids_sha256") != contract.deep_sha256(plan["outer_train"])
    ):
        raise ExecutionError("broker-selected outer train scope differs")
    counters["broker_target_release_phases"] += 1
    outer_fit = _service("frozen_science_fit_predict")(
        frame=frame,
        train_ids=plan["outer_train"],
        prediction_ids=plan["outer_validation"],
        labels=outer_release["labels"],
        anomaly_types=outer_release["anomaly_types"],
        incumbent_probability=None,
        cell=cell,
        block=None,
        inner_gate=gate,
    )
    outer_model = _model_document(
        cell=cell,
        block=None,
        sanitized_pin=sanitized_pin,
        train_ids=plan["outer_train"],
        teacher_receipts=[],
        fit=outer_fit,
    )
    outer_model_pin = _write(f"cells/cell_{cell:02d}/models/outer.json", outer_model)
    outer_prediction_pin = _write(
        f"cells/cell_{cell:02d}/outer_prediction.bin",
        verifier.encode_predictions(outer_fit["probabilities"]),
    )
    counters["baseline_fits"] += 1
    counters["unary_fits"] += 1
    counters["top_level_fits"] += 2
    counters["predictions"] += 1
    counters["cell_commitments"] += 1
    receipt = contract.event_body(
        "p1_v6r4_cell_receipt.v1",
        prior,
        {
            "cell": cell,
            "fold": fold,
            "fraction_tag": fraction_tag,
            "sanitized_input": sanitized_pin,
            "inner_commitments": inner_pins,
            "outer_model": outer_model_pin,
            "outer_prediction": outer_prediction_pin,
            "outer_train_rows": len(plan["outer_train"]),
            "outer_train_row_ids": plan["outer_train"],
            "outer_train_row_ids_sha256": contract.deep_sha256(plan["outer_train"]),
            "outer_prediction_rows": len(plan["outer_validation"]),
            "outer_prediction_row_ids": plan["outer_validation"],
            "outer_prediction_row_ids_sha256": contract.deep_sha256(
                plan["outer_validation"]
            ),
            "outer_validation_target_values_returned_to_worker": 0,
            "broker_release_phases": list(contract.PHASES[:-1]),
            "worker_counters": counters,
        },
    )
    receipt_pin = _write(f"cells/cell_{cell:02d}/cell_receipt.json", receipt)
    outer_seal = _service("broker_seal_outer")(cell, receipt_pin)
    if (
        outer_seal.get("outer_commitment_sha256") != receipt["event_sha256"]
        or outer_seal.get("outer_validation_values_returned_to_worker") != 0
    ):
        raise ExecutionError("broker outer non-release proof differs")
    if counters != {
        "baseline_fits": 4,
        "unary_fits": 4,
        "top_level_fits": 8,
        "predictions": 4,
        "inner_commitments": 3,
        "cell_commitments": 1,
        "teacher_receipts_consumed": 9,
        "scores": 4,
        "broker_target_release_phases": 7,
    }:
        raise ExecutionError("worker exact completion counters differ")
    _service("close_private_worker_session")()
    return {
        "cell": cell,
        "receipt": receipt_pin,
        "event_sha256": receipt["event_sha256"],
        "worker_counters": counters,
        "sanitized_input": sanitized_pin,
    }


def _metric_document(scope: dict[str, Any], raw: object) -> dict[str, Any]:
    value = contract.require_exact_keys(
        raw, ("rows", "precision", "recall", "f1"), label="broker metric result"
    )
    contract.strict_int(value["rows"], label="metric rows", minimum=1)
    for name in ("precision", "recall", "f1"):
        if type(value[name]) is not float or not 0.0 <= value[name] <= 1.0:
            raise ExecutionError(f"metric {name} differs")
    return {
        "schema_version": "p1_v6r4_metric.v1",
        "generation": contract.GENERATION,
        "scope": scope,
        **value,
    }


def _inventory(paths: set[str]) -> list[dict[str, Any]]:
    return [_service("output_pin")(path) for path in sorted(paths)]


def run_parent() -> dict[str, Any]:
    """Coordinate 15 private workers while never holding target-bearing bytes."""

    session = _service("private_session_snapshot")()
    if (
        type(session) is not dict
        or session.get("role") != "parent"
        or not contract.is_sha256(session.get("session_sha256"))
    ):
        raise PermissionError("parent private inherited session differs")
    sanitized_rows = _service("receive_all_sanitized_pins_once")()
    if type(sanitized_rows) is not list or len(sanitized_rows) != 15:
        raise ExecutionError("parent did not receive exactly 15 target-free buffer pins")
    sanitized_pins: dict[int, dict[str, Any]] = {}
    for cell, raw in enumerate(sanitized_rows, start=1):
        pin = contract.validate_pin(raw, label=f"parent sanitized cell {cell}")
        if pin["path"] != f"inherited://sanitized-cell-{cell:02d}":
            raise ExecutionError("parent sanitized buffer channel identity differs")
        sanitized_pins[cell] = pin
    session_event = contract.event_body(
        "p1_v6r4_session.v1",
        contract.GENESIS_SHA256,
        {
            "session_sha256": session["session_sha256"],
            "config": session["config"],
            "qa_receipt": session["qa_receipt"],
            "authorization": session["authorization"],
            "attempt_lock": session["attempt_lock"],
            "encoded_launch_provenance": session["encoded_launch_provenance"],
            "target_broker_processes": 1,
            "worker_processes": 15,
            "sanitized_cell_buffers": [sanitized_pins[cell] for cell in range(1, 16)],
        },
    )
    session_pin = _write("commitments/session.json", session_event)
    head = session_event["event_sha256"]
    predecessor_pin = session_pin
    cell_results: list[dict[str, Any]] = []
    fold_pins: list[dict[str, Any]] = []
    fold_metrics: dict[str, dict[str, Any]] = {}
    for cell in range(1, 16):
        if _service("validate_persisted_canonical_next_cell")(
            cell, head, session["session_sha256"]
        ) is not True:
            raise ExecutionError("parent canonical chain differs before worker launch")
        result = _service("launch_private_worker_inherited_handles")(
            cell=cell,
            sanitized_pin=sanitized_pins[cell],
            predecessor_pin=predecessor_pin,
        )
        if (
            type(result) is not dict
            or result.get("cell") != cell
            or result.get("sanitized_input") != sanitized_pins[cell]
        ):
            raise ExecutionError("private worker result identity differs")
        verified = verifier.verify_cell_artifact_graph(
            cell=cell,
            prior_event_sha256=head,
            sanitized_pin=sanitized_pins[cell],
        )
        if (
            verified["head_event_sha256"] != result.get("event_sha256")
            or verified["receipt"] != result.get("receipt")
        ):
            raise ExecutionError("independent cell semantic graph differs")
        head = verified["head_event_sha256"]
        predecessor_pin = verified["receipt"]
        cell_results.append(verified)
        if cell % 5 == 0:
            fold = contract.FOLDS[cell // 5 - 1]
            fold_cells = list(range(cell - 4, cell + 1))
            validation_ids = _service("derive_exact_cell_plan_from_sanitized_buffer")(
                fold_cells[0]
            )["outer_validation"]
            fold_event = contract.event_body(
                "p1_v6r4_fold_commitment.v1",
                head,
                {
                    "fold": fold,
                    "cells": fold_cells,
                    "cell_receipts": [row["receipt"] for row in cell_results[-5:]],
                    "outer_prediction_rows": len(validation_ids),
                    "outer_prediction_row_ids_sha256": contract.deep_sha256(validation_ids),
                    "outer_validation_targets_released_to_parent": 0,
                },
            )
            fold_pin = _write(f"commitments/fold_{fold}.json", fold_event)
            metric = _service("broker_score_committed_fold")(
                fold=fold,
                fold_commitment=fold_pin,
                cell_receipts=[row["receipt"] for row in cell_results[-5:]],
            )
            fold_metrics[fold] = _metric_document({"fold": fold}, metric)
            _write(f"metrics/fold_{fold}.json", fold_metrics[fold])
            fold_pins.append(fold_pin)
            head = fold_event["event_sha256"]
            predecessor_pin = fold_pin

    prediction_paths = sorted(
        path
        for path in contract.expected_output_paths()
        if path.endswith("outer_prediction.bin") or "/inner_predictions/" in path
    )
    complete_event = contract.event_body(
        "p1_v6r4_predictions_complete.v1",
        head,
        {
            "fold_commitments": fold_pins,
            "cell_receipts": [row["receipt"] for row in cell_results],
            "prediction_files": [_service("output_pin")(path) for path in prediction_paths],
            "teacher_receipts_consumed": 135,
        },
    )
    complete_pin = _write("commitments/predictions_complete.json", complete_event)
    head = complete_event["event_sha256"]
    fraction_metrics: dict[str, dict[str, Any]] = {}
    for tag in contract.FRACTION_TAGS:
        metric = _service("broker_score_committed_fraction")(
            fraction_tag=tag,
            predictions_complete=complete_pin,
            cell_receipts=[
                row["receipt"] for row in cell_results if row["fraction_tag"] == tag
            ],
        )
        document = _metric_document({"fraction_tag": tag}, metric)
        fraction_metrics[tag] = document
        _write(f"metrics/fraction_{tag}.json", document)
    split_cells = [
        {
            "cell": cell,
            "fold": contract.cell_identity(cell)[0],
            "fraction_tag": contract.cell_identity(cell)[2],
            "plan": _service("derive_exact_cell_plan_from_sanitized_buffer")(cell),
        }
        for cell in range(1, 16)
    ]
    _write(
        "split_audit.json",
        {
            "schema_version": "p1_v6r4_split_audit.v1",
            "generation": contract.GENERATION,
            "cells": split_cells,
        },
    )
    broker_audit = _service("receive_closed_target_broker_audit")()
    _write("target_broker_audit.json", broker_audit)
    counters = dict(contract.EXACT_COMPLETION_COUNTERS)
    _write(
        "metrics.json",
        {
            "schema_version": "p1_v6r4_metrics.v1",
            "generation": contract.GENERATION,
            "score_decomposition": contract.SCORE_DECOMPOSITION,
            "counters": counters,
        },
    )
    nonnegative = all(
        metric["f1"] >= 0.0
        for metric in [*fold_metrics.values(), *fraction_metrics.values()]
    )
    if not nonnegative:
        raise ExecutionError("frozen >=0 aggregate guard failed")
    _write(
        "learning_curve_evidence.json",
        {
            "schema_version": "p1_v6r4_learning_curve_evidence.v1",
            "generation": contract.GENERATION,
            "folds": list(contract.FOLDS),
            "fraction_tags": list(contract.FRACTION_TAGS),
            "nonnegative_guards_all_true": True,
        },
    )
    _write(
        "result.json",
        {
            "schema_version": "p1_v6r4_result.v1",
            "generation": contract.GENERATION,
            "research_only": True,
            "candidate_created": False,
            "test_values_read": 0,
            "ledger_appended": False,
            "uploaded": False,
            "predictions_complete": complete_pin,
        },
    )
    all_paths = set(contract.expected_output_paths())
    before_manifest = all_paths - {
        "manifest.json",
        "manifest.sha256",
        "preseal.json",
        "final_seal.json",
    }
    manifest_pin = _write(
        "manifest.json",
        {
            "schema_version": "p1_v6r4_manifest.v1",
            "generation": contract.GENERATION,
            "files": _inventory(before_manifest),
        },
    )
    sidecar_pin = _write(
        "manifest.sha256", f"{manifest_pin['sha256']}  manifest.json\n".encode("ascii")
    )
    preseal_paths = all_paths - {"preseal.json", "final_seal.json"}
    preseal = contract.event_body(
        "p1_v6r4_preseal.v1",
        head,
        {
            "manifest": manifest_pin,
            "manifest_sidecar": sidecar_pin,
            "inventory": _inventory(preseal_paths),
            "inventory_count": 200,
            "counters": counters,
        },
    )
    preseal_pin = _write("preseal.json", preseal)
    final_inventory_paths = all_paths - {"final_seal.json"}
    seal = contract.event_body(
        "p1_v6r4_final_seal.v1",
        preseal["event_sha256"],
        {
            "preseal": preseal_pin,
            "inventory": _inventory(final_inventory_paths),
            "inventory_count": 201,
            "final_file_count": 202,
            "counters": counters,
            "candidate_created": False,
            "test_values_read": 0,
            "ledger_appended": False,
            "uploaded": False,
        },
    )
    _write("final_seal.json", seal)
    verified = verifier.verify_final_output()
    _service("close_private_parent_session")()
    return {
        "status": "RESEARCH_ONLY_STATIC_CONTRACT_COMPLETE_NO_CANDIDATE",
        "files": verified["verified_paths"],
        "scores": verified["score_calls"],
        "teacher_receipts_consumed": verified["teacher_receipts_consumed"],
        "candidate_created": False,
        "test_values_read": 0,
        "ledger_appended": False,
        "uploaded": False,
    }


__all__ = ["ExecutionError", "run_cell", "run_parent"]
