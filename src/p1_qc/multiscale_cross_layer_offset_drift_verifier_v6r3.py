"""Independent persisted-artifact and causality verifier for P1 Gen6r3."""

from __future__ import annotations

from typing import Any

try:
    _CONTEXT = _P1_V6R3_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R3_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r3 verifier requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r3 verifier loaded before source authentication")


class VerificationError(RuntimeError):
    """A persisted artifact, output tree, or causal chain differs."""


def expected_cell_paths(cell: int) -> set[str]:
    contract.cell_identity(cell)
    prefix = f"cells/cell_{cell:02d}"
    return {
        *(f"{prefix}/models/inner_{block}.json" for block in contract.BLOCKS),
        f"{prefix}/models/outer.json",
        *(f"{prefix}/inner_predictions/block_{block}.bin" for block in contract.BLOCKS),
        *(f"{prefix}/commitments/inner_{block}.json" for block in contract.BLOCKS),
        f"{prefix}/outer_prediction.bin",
        f"{prefix}/cell_receipt.json",
    }


def expected_output_paths() -> set[str]:
    paths = {path for cell in range(1, 16) for path in expected_cell_paths(cell)}
    paths.update(
        {
            "commitments/session.json",
            "commitments/predictions_complete.json",
            *(f"commitments/fold_{fold}.json" for fold in contract.FOLDS),
            *(f"metrics/fold_{fold}.json" for fold in contract.FOLDS),
            *(f"metrics/fraction_{tag}.json" for tag in contract.FRACTION_TAGS),
            "split_audit.json",
            "selective_target_audit.json",
            "metrics.json",
            "learning_curve_evidence.json",
            "result.json",
            "resource_audit.json",
            "manifest.json",
            "manifest.sha256",
            "final_seal.json",
        }
    )
    if len(paths) != 202:
        raise VerificationError("internal expected-output arithmetic differs")
    return paths


def _reader() -> Any:
    value = _CONTEXT.get("authenticated_output_bytes")
    if not callable(value):
        raise VerificationError("authenticated output reader is unavailable")
    return value


def _pin_for(relative: str) -> dict[str, Any]:
    pin = _CONTEXT.get("output_pin")
    if not callable(pin):
        raise VerificationError("output pin provider is unavailable")
    value = pin(relative)
    if type(value) is not dict or value.get("path") != relative:
        raise VerificationError(f"output pin differs: {relative}")
    return value


def _json(relative: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pin = _pin_for(relative)
    value = contract.parse_json_bytes(_reader()(pin, relative), label=relative)
    if type(value) is not dict:
        raise VerificationError(f"output JSON object differs: {relative}")
    return value, pin


def _verify_event(event: dict[str, Any], *, prior: str, schema: str) -> str:
    if (
        event.get("schema_version") != schema
        or event.get("generation") != contract.GENERATION
        or event.get("prior_event_sha256") != prior
        or not contract._is_sha(event.get("event_sha256"))
    ):
        raise VerificationError(f"{schema} identity/causality differs")
    claimed = event["event_sha256"]
    body = dict(event)
    body.pop("event_sha256")
    if contract.deep_sha256(body) != claimed:
        raise VerificationError(f"{schema} event hash differs")
    return claimed


def _assert_pin(reference: Any, relative: str) -> dict[str, Any]:
    observed = _pin_for(relative)
    if reference != observed:
        raise VerificationError(f"artifact pin differs: {relative}")
    _reader()(observed, relative)
    return observed


def verify_cell_artifact_graph(
    capability: object,
    *,
    cell: int,
    worker_result: dict[str, Any],
    prior_event_sha256: str,
    session_sha256: str,
) -> dict[str, Any]:
    live = contract.capability_snapshot(capability)
    if live["role"] != "parent":
        raise PermissionError("only the parent may verify a completed cell graph")
    fold, fraction, fraction_tag = contract.cell_identity(cell)
    listed = _CONTEXT.get("list_output_paths")
    if not callable(listed):
        raise VerificationError("output path enumerator is unavailable")
    prefix = f"cells/cell_{cell:02d}/"
    observed_paths = set(listed(prefix))
    if observed_paths != expected_cell_paths(cell):
        raise VerificationError("cell output tree has missing or extra paths")
    prior = prior_event_sha256
    for block in contract.BLOCKS:
        relative = f"cells/cell_{cell:02d}/commitments/inner_{block}.json"
        event, _pin = _json(relative)
        prior = _verify_event(event, prior=prior, schema="p1_v6r3_inner_commitment.v1")
        if (
            event.get("cell") != cell
            or event.get("block") != block
            or event.get("fold") != fold
            or event.get("fraction") != fraction
            or not contract._is_sha(event.get("row_ids_sha256"))
            or not contract._is_sha(event.get("split_proof_sha256"))
        ):
            raise VerificationError("inner commitment semantics differ")
        _assert_pin(
            event.get("prediction_bundle"),
            f"cells/cell_{cell:02d}/inner_predictions/block_{block}.bin",
        )
        _assert_pin(event.get("model"), f"cells/cell_{cell:02d}/models/inner_{block}.json")
    receipt_relative = f"cells/cell_{cell:02d}/cell_receipt.json"
    receipt, receipt_pin = _json(receipt_relative)
    head = _verify_event(receipt, prior=prior, schema="p1_v6r3_cell_receipt.v1")
    if (
        receipt.get("cell") != cell
        or receipt.get("fold") != fold
        or receipt.get("fraction") != fraction
        or receipt.get("fraction_tag") != fraction_tag
        or receipt.get("session_sha256") != session_sha256
        or worker_result.get("event_sha256") != head
        or worker_result.get("receipt") != receipt_pin
        or receipt.get("outer_precommit_target_proof", {}).get(
            "already_decoded_row_scope_exposures_before_commitment"
        )
        != 0
        or receipt.get("outer_precommit_target_proof", {}).get(
            "active_outer_target_scalars_decoded_before_commitment"
        )
        != 0
        or receipt.get("target_vault_audit", {}).get("fresh_process_empty_at_start") is not True
        or receipt.get("target_vault_audit", {}).get("outer_validation_decoded_in_worker") is not False
    ):
        raise VerificationError("cell receipt target-isolation semantics differ")
    _assert_pin(
        receipt.get("outer_prediction_bundle"), f"cells/cell_{cell:02d}/outer_prediction.bin"
    )
    _assert_pin(receipt.get("outer_model"), f"cells/cell_{cell:02d}/models/outer.json")
    proofs = receipt.get("split_audits")
    if (
        type(proofs) is not list
        or len(proofs) != 3
        or any(
            proof.get("block") != block
            or proof.get("precommit_target_proof", {}).get(
                "already_decoded_row_scope_exposures_before_commitment"
            )
            != 0
            or proof.get("precommit_target_proof", {}).get(
                "target_scalars_decoded_before_commitment"
            )
            != 0
            for block, proof in zip(contract.BLOCKS, proofs, strict=True)
        )
    ):
        raise VerificationError("inner precommit target proofs differ")
    counters = worker_result.get("worker_counters")
    if (
        type(counters) is not dict
        or counters.get("scores") != 4
        or counters.get("baseline_fits") != 4
        or counters.get("unary_fits") != 4
        or counters.get("top_level_fits") != 8
        or counters.get("predictions") != 4
        or counters.get("inner_commitments") != 3
        or counters.get("cell_commitments") != 1
    ):
        raise VerificationError("cell exact science counters differ")
    return {"cell": cell, "head_event_sha256": head, "receipt": receipt_pin, "files": 12}


def verify_final_output(
    capability: object,
    *,
    final_seal_pin: dict[str, Any],
    commitment_chain_head_sha256: str,
) -> dict[str, Any]:
    live = contract.capability_snapshot(capability)
    if live["role"] != "parent":
        raise PermissionError("only the parent may verify the final output")
    listed = _CONTEXT.get("list_output_paths")
    inventory = _CONTEXT.get("output_inventory")
    if not callable(listed) or not callable(inventory):
        raise VerificationError("final output inventory callbacks are unavailable")
    paths = set(listed(""))
    if paths != expected_output_paths():
        raise VerificationError("final output tree has missing or extra paths")
    seal, observed_pin = _json("final_seal.json")
    if (
        observed_pin != final_seal_pin
        or seal.get("schema_version") != "p1_v6r3_final_seal.v1"
        or seal.get("generation") != contract.GENERATION
        or seal.get("pre_seal_files") != 201
        or seal.get("expected_final_files") != 202
        or seal.get("commitment_chain_head_sha256") != commitment_chain_head_sha256
        or seal.get("candidate_created") is not False
        or seal.get("test_prediction_created") is not False
        or seal.get("ledger_appended") is not False
        or seal.get("uploaded") is not False
    ):
        raise VerificationError("final seal semantics differ")
    final = inventory(final=True)
    if (
        final.get("files") != 202
        or final.get("same_handle_final_rehashes") != 202
        or set(final.get("paths", [])) != paths
        or not contract._is_sha(final.get("inventory_sha256"))
    ):
        raise VerificationError("final same-handle inventory differs")
    return final


__all__ = [
    "VerificationError",
    "expected_cell_paths",
    "expected_output_paths",
    "verify_cell_artifact_graph",
    "verify_final_output",
]
