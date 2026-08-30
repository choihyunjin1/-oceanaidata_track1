"""Zero-fit integrity and deployability audit for the frozen P2 Gaussian copula.

The audit intentionally does not load observations, prediction arrays, or official
inputs. Historical prediction files are checked by stat metadata only; their hashes
are accepted only as commitments already sealed in aggregate JSON receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p2_gaussian_copula_frozen_deployability_preflight_20260830_v3"
FORBIDDEN_ROW_FILE_SUFFIXES = {".csv", ".parquet", ".npz"}


class PreflightError(RuntimeError):
    """Raised when a sealed preflight invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"Expected JSON object: {path}")
    return value


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    candidate_text = Path(relative_path)
    if candidate_text.is_absolute():
        raise PreflightError(f"Absolute evidence path is forbidden: {relative_path}")
    root = repo_root.resolve()
    candidate = (root / candidate_text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"Evidence escapes repository root: {relative_path}") from exc
    return candidate


def _nested_value(document: Any, json_path: list[Any]) -> Any:
    value = document
    for key in json_path:
        if isinstance(value, dict) and isinstance(key, str):
            if key not in value:
                raise PreflightError(f"Missing JSON key {key!r} in path {json_path!r}")
            value = value[key]
        elif isinstance(value, list) and isinstance(key, int):
            try:
                value = value[key]
            except IndexError as exc:
                raise PreflightError(f"Missing JSON index {key} in path {json_path!r}") from exc
        else:
            raise PreflightError(f"Invalid JSON traversal at {key!r} in {json_path!r}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise PreflightError("Unexpected experiment_id")
    if config.get("mode") != "ZERO_FIT_HASH_LEDGER_DEPLOYABILITY_PREFLIGHT":
        raise PreflightError("Preflight mode is not sealed")
    policy = config.get("execution_policy", {})
    required_policy = {
        "maximum_executions": 1,
        "maximum_model_fits": 0,
        "maximum_threads": 1,
        "result_based_retry": False,
        "result_based_tuning": False,
        "read_training_rows": False,
        "read_prediction_rows": False,
        "read_official_test_index_sample_baseline_score_query_rows": False,
        "create_csv": False,
        "upload": False,
        "commit": False,
        "push": False,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise PreflightError(f"Execution policy mismatch for {key}")
    frozen = config.get("frozen_candidate", {})
    if frozen.get("local_confirmation_rerun_authorized") is not False:
        raise PreflightError("Frozen v2 local rerun must remain unauthorized")


def _inspect_pinned_files(
    repo_root: Path, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    receipts: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    roles: set[str] = set()
    for item in config["pinned_files"]:
        role = item["role"]
        if role in roles:
            raise PreflightError(f"Duplicate pinned role: {role}")
        roles.add(role)
        path = resolve_repo_path(repo_root, item["path"])
        if not path.is_file():
            raise PreflightError(f"Missing pinned file: {item['path']}")
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"]:
            raise PreflightError(f"SHA-256 mismatch for {role}")
        inspection = item["inspection"]
        if inspection not in {"json_aggregate", "hash_only"}:
            raise PreflightError(f"Unsupported inspection mode for {role}: {inspection}")
        if inspection == "json_aggregate":
            if path.suffix.lower() != ".json":
                raise PreflightError(f"Aggregate JSON role is not JSON: {role}")
            documents[role] = load_json(path)
        receipts[role] = {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "inspection": inspection,
        }
    return receipts, documents


def _check_claims(config: dict[str, Any], documents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for check in config["claim_checks"]:
        role = check["source_role"]
        if role not in documents:
            raise PreflightError(f"Claim source was not parsed as aggregate JSON: {role}")
        actual = _nested_value(documents[role], check["json_path"])
        expected = check["equals"]
        if actual != expected:
            raise PreflightError(
                f"Claim mismatch for {role}:{check['json_path']!r}; "
                f"expected {expected!r}, got {actual!r}"
            )
        receipts.append(
            {
                "source_role": role,
                "json_path": check["json_path"],
                "matched": True,
            }
        )
    return receipts


def _check_prediction_metadata(
    repo_root: Path,
    config: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    commitment = documents["prediction_commitment"]
    committed_outputs = commitment.get("outputs", {})
    receipts: list[dict[str, Any]] = []
    total_rows = 0
    for item in config["historical_prediction_artifacts"]:
        path = resolve_repo_path(repo_root, item["path"])
        if path.suffix.lower() not in FORBIDDEN_ROW_FILE_SUFFIXES:
            raise PreflightError(f"Expected a row-bearing historical artifact: {item['path']}")
        if not path.is_file():
            raise PreflightError(f"Missing historical prediction artifact: {item['path']}")
        actual_bytes = path.stat().st_size
        if actual_bytes != item["committed_bytes"]:
            raise PreflightError(f"Historical prediction byte-size mismatch: {item['block']}")
        committed = committed_outputs.get(item["block"])
        expected_commitment = {
            "path": item["path"].replace("/", "\\"),
            "rows": item["committed_rows"],
            "bytes": item["committed_bytes"],
            "sha256": item["committed_sha256"],
        }
        if committed != expected_commitment:
            raise PreflightError(f"Prediction commitment mismatch: {item['block']}")
        total_rows += item["committed_rows"]
        receipts.append(
            {
                "block": item["block"],
                "path": item["path"],
                "bytes": actual_bytes,
                "committed_rows": item["committed_rows"],
                "committed_sha256": item["committed_sha256"],
                "inspection": "STAT_ONLY_NO_CONTENT_HASH_NO_ROW_READ",
            }
        )
    expected_rows = config["expected_historical_result"]["rows"]
    if total_rows != expected_rows:
        raise PreflightError("Historical prediction row commitments do not sum to expected rows")
    return receipts


def _audit_recorded_duplicates(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    duplicate_config = config["official_submission_duplicate_audit"]
    tokens = [token.casefold() for token in duplicate_config["candidate_tokens"]]
    matches: list[dict[str, str]] = []
    ledgers: list[dict[str, Any]] = []
    for item in duplicate_config["ledgers"]:
        path = resolve_repo_path(repo_root, item["path"])
        if not path.is_file():
            raise PreflightError(f"Missing official aggregate ledger: {item['path']}")
        if path.suffix.lower() in FORBIDDEN_ROW_FILE_SUFFIXES:
            raise PreflightError(f"Row-bearing official file is forbidden: {item['path']}")
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"]:
            raise PreflightError(f"Official aggregate ledger hash mismatch: {item['path']}")
        text = path.read_text(encoding="utf-8").casefold()
        for original, folded in zip(duplicate_config["candidate_tokens"], tokens, strict=True):
            if folded in text:
                matches.append({"ledger": item["path"], "token": original})
        ledgers.append(
            {
                "path": item["path"],
                "bytes": path.stat().st_size,
                "sha256": actual_sha,
            }
        )
    status = (
        "RECORDED_DUPLICATE_IN_PINNED_LOCAL_OFFICIAL_LEDGERS"
        if matches
        else "NO_RECORDED_DUPLICATE_IN_PINNED_LOCAL_OFFICIAL_LEDGERS"
    )
    return {
        "status": status,
        "scope": duplicate_config["scope"],
        "ledger_count": len(ledgers),
        "ledgers": ledgers,
        "matched_token_count": len(matches),
        "matches": matches,
        "platform_current_state_verified": False,
    }


def evaluate_preflight(
    repo_root: Path,
    config: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Evaluate the sealed zero-fit preflight without opening any row data."""

    validate_config(config)
    pinned_receipts, documents = _inspect_pinned_files(repo_root, config)
    claim_receipts = _check_claims(config, documents)
    prediction_receipts = _check_prediction_metadata(repo_root, config, documents)
    duplicate_receipt = _audit_recorded_duplicates(repo_root, config)

    deployability = config["deployability"]
    blockers: list[str] = []
    if duplicate_receipt["matched_token_count"]:
        blockers.append("RECORDED_OFFICIAL_DUPLICATE")
    if not deployability["official_prediction_artifact_materialized"]:
        blockers.append("NO_26061_ROW_OFFICIAL_CANDIDATE_ARTIFACT")
    if not deployability["current_platform_duplicate_state_verified"]:
        blockers.append("CURRENT_PLATFORM_DUPLICATE_STATE_UNVERIFIED")
    if not deployability["current_daily_quota_verified"]:
        blockers.append("CURRENT_DAILY_QUOTA_UNVERIFIED")
    if not deployability["historical_code_hashes_were_sealed_at_v2_execution"]:
        blockers.append("EXECUTION_TIME_SOURCE_HASH_ATTRIBUTION_UNAVAILABLE")

    if duplicate_receipt["matched_token_count"]:
        status = config["predeclared_decision"]["recorded_duplicate"]
    elif blockers:
        status = config["predeclared_decision"][
            "integrity_pass_without_official_artifact_or_live_platform_checks"
        ]
    else:
        status = config["predeclared_decision"][
            "probe_ready_only_if_all_required_before_probe_are_later_satisfied"
        ]

    return {
        "schema_version": "p2.gaussian_copula.frozen_deployability_preflight.result.20260830.v3",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "integrity_pass": True,
        "official_probe_ready": not blockers,
        "frozen_candidate": config["frozen_candidate"],
        "historical_result": config["expected_historical_result"],
        "config_sha256": config_sha256,
        "pinned_file_receipts": pinned_receipts,
        "claim_checks": claim_receipts,
        "historical_prediction_artifact_receipts": prediction_receipts,
        "recorded_submission_duplicate_audit": duplicate_receipt,
        "deployability_blockers": blockers,
        "required_before_probe": deployability["required_before_probe"],
        "limitations": [
            deployability["historical_code_hash_limitation"],
            "Historical NPZ files were stat-checked only; their row values and bytes were not opened or re-hashed.",
            "Absence from pinned local ledgers is not proof of absence from the live platform.",
            "Historical 69850-row evidence uses an exposed proxy comparator and is not official-test confirmation.",
        ],
        "execution_receipt": {
            "model_fits": 0,
            "thread_budget": 1,
            "training_rows_read": 0,
            "historical_prediction_rows_read": 0,
            "official_test_index_sample_baseline_score_query_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
            "commits": 0,
            "pushes": 0,
            "aggregate_or_config_json_files_parsed": len(documents),
            "official_aggregate_ledger_files_read": duplicate_receipt["ledger_count"],
        },
    }


def reserve_attempt(output_dir: Path, config_sha256: str) -> Path:
    """Reserve the sole attempt with exclusive creation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "attempt.lock.json"
    if (output_dir / "result.json").exists():
        raise PreflightError("Result already exists; same-ID rerun is forbidden")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "maximum_executions": 1,
        "attempt_number": 1,
        "config_sha256": config_sha256,
        "result_based_retry": False,
        "reserved_at_kst": datetime.now().astimezone().isoformat(),
        "process_id": os.getpid(),
    }
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise PreflightError("Attempt lock already exists; same-ID rerun is forbidden") from exc
    return lock_path


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise PreflightError(f"Exclusive output already exists: {path}") from exc
