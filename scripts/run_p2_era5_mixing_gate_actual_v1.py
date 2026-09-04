"""Canonical preflight/authorization/one-shot runner for P2 ERA5 mixing gate.

The only mode executed during implementation is ``preflight``.  ``authorize``
requires an independent QA receipt, while ``actual`` additionally consumes the
O_EXCL authorization and permanently reserves generation 1 before value reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ocean_external.p2_era5_scope import validate_p2_era5_scope_amendment
from p2_restore.era5_mixing_gate import (
    ERA5_MIXING_FEATURES,
    ERA5_VALUE_COLUMNS,
    align_mixing_features_to_oof_keys,
    build_hourly_ocean_mixing_features,
    validate_era5_source_frame,
)
from p2_restore.era5_mixing_gate_actual import (
    AppendOnlyLedger,
    FoldLocalTruthVault,
    GrantBoundEra5Reader,
    build_block_masked_public_state_panel,
    load_observations_only,
    metric_summary,
    outer_promotion_decision,
    paired_kst_day_bootstrap,
    run_fold_local_gate,
    sha256_file,
    validate_native_flux_sign_contract,
    validate_truth_shard_key_contract,
    verify_append_only_ledger,
    write_and_seal_blind_predictions,
    write_json_exclusive_fsync,
)
from p2_restore.regime_gate import STATE_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_era5_mixing_gate_actual_v1"
CANONICAL_CONFIG_PATH = REPO_ROOT / "configs/experiments/p2_era5_mixing_gate_actual_v1.json"
CANONICAL_CONFIG_SHA256 = "fe8f167f9f9d0906cf7571f2a2fb0e25d5afda0ebea1553ff907dfc8c47904f7"
CANONICAL_PREFLIGHT_DIRECTORY = REPO_ROOT / "artifacts/p2_era5_mixing_gate_actual_v1_preflight"
CANONICAL_PREFLIGHT_RECEIPT = CANONICAL_PREFLIGHT_DIRECTORY / "preflight_receipt.json"
CANONICAL_QA_RECEIPT = REPO_ROOT / "artifacts/p2_era5_mixing_gate_actual_v1_qa/qa_receipt.json"
CANONICAL_AUTHORIZATION = REPO_ROOT / "artifacts/p2_era5_mixing_gate_actual_v1_authorization.json"
CANONICAL_ACTUAL_DIRECTORY = REPO_ROOT / "artifacts/p2_era5_mixing_gate_actual_v1_run1"
PREFLIGHT_STATUS = REPO_ROOT / "artifacts/status/p2_era5_mixing_gate_actual_v1_preflight.json"
ACTUAL_STATUS = REPO_ROOT / "artifacts/status/p2_era5_mixing_gate_actual_v1.json"
AUTHORIZATION_PHRASE = "AUTHORIZE_P2_ERA5_MIXING_GATE_ACTUAL_V1"
AUTHORIZATION_PHRASE_SHA256 = hashlib.sha256(AUTHORIZATION_PHRASE.encode()).hexdigest()

EXPECTED_DRY_RECEIPT_SHA256 = "da33b72f4bfd7a9aa65c71232c543977e3ffd1fa65883f8376198190d304b24a"
EXPECTED_DRY_MANIFEST_SHA256 = "fcb5db598472915a6479c8b00f79649369b9fb7ec2da17e0e5520c367a578706"
EXPECTED_DRY_SEAL_SHA256 = "00efc2fd455c73c2967cf29338234c21b688e6e87c08c580d23fe39ad3cf7760"
EXPECTED_DRY_CODE_SHA256 = {
    "config": "3b63181debd91d3c6f15be2778f19d982b88c107a5b1b95a6f477d239b05be92",
    "runner": "1078170d0c9b9c45261e0f3bd0bb1fcba77b5a9e566d8da281198eea0084e6a3",
    "feature_helper": "5ef68b6536d8cfd61f1b5017ece16afceb9fcd3df2e7533c2801787266972ecf",
    "tests": "c2f81945c33e0b06e32be6055d378662b9022296dd893861375707faa832432b",
    "policy_validator": "f1902279d01f04d5890e3d2e671d570f4c7675817320c82a5b1849d085567122",
}
EXPECTED_DRY_INPUT_SHA256 = {
    "era5_parquet": "9e84c704e85d77796dbd554986f376d77e838b96de28fd08ad8ab6191560a7a0",
    "era5_manifest": "17591180646a762cfa9a309cf0cc934c8109879c17d10084055133e9aa4f5bee",
    "deep_oof": "76f52261259112d03289226c23712e27a17a17f9320c8565c96a662c1093be8d",
    "physical_oof": "adffb86b43ae69ad3f11863263fe3679cc0195053b46c1ec57a307d2c54eb7cc",
    "public_state_implementation": "d0e384ce7d442a0091a2ac2081df899b7acf78a291bf3a1b311b99b0d3709b45",
}


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("canonical experiment path escaped the repository") from error


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {_relative(path)}")
    return value


def _write_status(
    path: Path,
    *,
    progress: float,
    phase: str,
    detail: str,
    started: float,
    status: str = "running",
) -> None:
    elapsed = time.perf_counter() - started
    bounded = min(max(float(progress), 0.1), 100.0)
    remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100.0 else 0.0
    payload = {
        "title": "P2 ERA5 mixing-gate actual v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "progress": bounded,
        "phase": phase,
        "detail": detail,
        "actual_ready": False,
        "elapsed_seconds": elapsed,
        "eta": (datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))).strftime(
            "%Y-%m-%d %H:%M:%S KST"
        ),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_canonical_config() -> dict[str, Any]:
    if not CANONICAL_CONFIG_PATH.is_file():
        raise FileNotFoundError("canonical actual config is missing")
    if sha256_file(CANONICAL_CONFIG_PATH) != CANONICAL_CONFIG_SHA256:
        raise ValueError("canonical actual config SHA-256 changed")
    contract = _read_json(CANONICAL_CONFIG_PATH)
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("experiment_id") != EXPERIMENT_ID
        or contract.get("generation") != 1
        or contract.get("upload_allowed") is not False
    ):
        raise ValueError("canonical actual config identity changed")
    if contract["gate_model"]["parameter_grid"] != []:
        raise ValueError("actual gate parameter grid must remain empty")
    if contract["features"]["control"] != list(STATE_FEATURES):
        raise ValueError("actual control feature order changed")
    if contract["features"]["era5_increment"] != list(ERA5_MIXING_FEATURES):
        raise ValueError("actual ERA5 feature order changed")
    validate_native_flux_sign_contract(contract)
    deep = contract["frozen_inputs"]["deep_oof"]
    if deep.get("actual_truth_value_reads_allowed") is not False or deep.get(
        "allowed_actual_columns"
    ) != ["time", "layer", "block", "lobo_prediction"]:
        raise ValueError("stacked OOF truth-read ban changed")
    outputs = contract["outputs"]
    canonical_outputs = {
        "preflight_directory": _relative(CANONICAL_PREFLIGHT_DIRECTORY),
        "authorization_file": _relative(CANONICAL_AUTHORIZATION),
        "actual_directory": _relative(CANONICAL_ACTUAL_DIRECTORY),
    }
    for name, expected in canonical_outputs.items():
        if outputs[name] != expected:
            raise ValueError(f"canonical output path changed: {name}")
    return contract


def _assert_dry_generation(contract: Mapping[str, Any]) -> dict[str, Any]:
    dry = contract["canonical_dry_generation"]
    receipt_path = _repo_path(dry["receipt_path"])
    manifest_path = _repo_path(dry["manifest_path"])
    seal_path = _repo_path(dry["seal_path"])
    observed_artifacts = {
        "receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
    }
    expected_artifacts = {
        "receipt_sha256": EXPECTED_DRY_RECEIPT_SHA256,
        "manifest_sha256": EXPECTED_DRY_MANIFEST_SHA256,
        "seal_sha256": EXPECTED_DRY_SEAL_SHA256,
    }
    if observed_artifacts != expected_artifacts:
        raise ValueError("canonical dry artifact SHA-256 equality failed")
    if dry["receipt_sha256"] != EXPECTED_DRY_RECEIPT_SHA256:
        raise ValueError("actual config dry receipt pin differs from hardcoded pin")
    if dry["expected_code_sha256"] != EXPECTED_DRY_CODE_SHA256:
        raise ValueError("actual config dry code pins differ from hardcoded pins")
    if dry["expected_input_sha256"] != EXPECTED_DRY_INPUT_SHA256:
        raise ValueError("actual config dry input pins differ from hardcoded pins")

    receipt = _read_json(receipt_path)
    seal = _read_json(seal_path)
    observed_code = dict(seal["code_sha256"])
    observed_inputs = {
        "era5_parquet": receipt["input_files"]["era5_parquet"]["sha256"],
        "era5_manifest": receipt["input_files"]["era5_manifest"]["sha256"],
        "deep_oof": receipt["input_files"]["deep_oof_opaque_container"]["sha256"],
        "physical_oof": receipt["input_files"]["physical_oof_opaque_container"]["sha256"],
        "public_state_implementation": receipt["input_files"]["public_state_implementation"][
            "sha256"
        ],
    }
    if observed_code != EXPECTED_DRY_CODE_SHA256 or observed_inputs != EXPECTED_DRY_INPUT_SHA256:
        raise ValueError("dry receipt in-memory code/input equality failed")
    if receipt.get("actual_ready") is not False or any(receipt["access_counters"].values()):
        raise ValueError("canonical dry receipt no-access/actual lock changed")
    return {
        "artifacts": observed_artifacts,
        "code_sha256": observed_code,
        "input_sha256": observed_inputs,
        "in_memory_equality": True,
    }


def _scope_grant(contract: Mapping[str, Any]) -> object:
    scope = contract["scope_grant"]
    validator_path = _repo_path(scope["validator_path"])
    amendment_path = _repo_path(scope["amendment_path"])
    if sha256_file(validator_path) != scope["validator_sha256"]:
        raise ValueError("canonical scope validator SHA-256 changed")
    if sha256_file(amendment_path) != scope["amendment_sha256"]:
        raise ValueError("canonical scope amendment SHA-256 changed")
    candidate = _repo_path(contract["frozen_inputs"]["era5_parquet"]["path"])
    return validate_p2_era5_scope_amendment(
        repo_root=REPO_ROOT,
        amendment_path=amendment_path,
        problem=scope["problem"],
        source_id=scope["source_id"],
        purpose=scope["purpose"],
        candidate_parquet_path=candidate,
    )


def _assert_grant_value_contract(contract: Mapping[str, Any], grant: object) -> dict[str, Any]:
    scope = contract["scope_grant"]
    expected_variables = list(scope["expected_allowed_variables"])
    expected = {
        "allowed_variables": expected_variables,
        "units_by_variable": dict(scope["expected_units_by_variable"]),
        "sign_by_variable": dict(scope["expected_sign_by_variable"]),
        "flux_storage_convention": scope["expected_flux_storage_convention"],
        "surface_energy_flux_positive_direction": scope[
            "expected_surface_energy_flux_positive_direction"
        ],
        "native_qnet_formula": scope["expected_native_qnet_formula"],
    }
    observed = {
        "allowed_variables": list(grant.allowed_variables),
        "units_by_variable": dict(grant.units_by_variable),
        "sign_by_variable": dict(grant.sign_by_variable),
        "flux_storage_convention": grant.flux_storage_convention,
        "surface_energy_flux_positive_direction": (grant.surface_energy_flux_positive_direction),
        "native_qnet_formula": grant.native_qnet_formula,
    }
    if expected_variables != list(ERA5_VALUE_COLUMNS) or observed != expected:
        raise ValueError("canonical ERA5 nine-variable unit/sign/native-qnet grant changed")
    if (
        grant.amendment_sha256 != scope["amendment_sha256"]
        or grant.superseded_amendment_sha256 != scope["superseded_v1_sha256"]
        or grant.parquet_sha256 != contract["frozen_inputs"]["era5_parquet"]["sha256"]
        or grant.parquet_path.resolve()
        != _repo_path(contract["frozen_inputs"]["era5_parquet"]["path"])
    ):
        raise ValueError("canonical v2 grant artifact binding changed")
    native = validate_native_flux_sign_contract(contract)
    if native["qnet_definition"] != grant.native_qnet_formula:
        raise ValueError("native-qnet config and canonical v2 grant differ")
    return {
        **expected,
        "amendment_sha256": grant.amendment_sha256,
        "superseded_amendment_sha256": grant.superseded_amendment_sha256,
        "parquet_sha256": grant.parquet_sha256,
        "read_path_source": "grant.parquet_path_only",
        "component_sign_flips_applied": 0,
    }


def _implementation_hashes() -> dict[str, str]:
    files = {
        "config": CANONICAL_CONFIG_PATH,
        "helper": REPO_ROOT / "src/p2_restore/era5_mixing_gate_actual.py",
        "runner": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_p2_era5_mixing_gate_actual.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"actual implementation bundle is incomplete: {missing}")
    return {name: sha256_file(path) for name, path in files.items()}


def _o_excl_probe(path: Path) -> dict[str, Any]:
    payload = {"probe": "o_excl", "experiment_id": EXPERIMENT_ID}
    write_json_exclusive_fsync(path, payload)
    before = sha256_file(path)
    rejected = False
    try:
        write_json_exclusive_fsync(path, payload)
    except FileExistsError:
        rejected = True
    if not rejected or sha256_file(path) != before:
        raise AssertionError("O_EXCL probe did not fail closed")
    return {"second_create_rejected": True, "original_sha256_unchanged": True}


def run_preflight() -> dict[str, Any]:
    started = time.perf_counter()
    _write_status(
        PREFLIGHT_STATUS,
        progress=2,
        phase="canonical_contract",
        detail="hardcoded actual config and dry pins",
        started=started,
    )
    contract = _load_canonical_config()
    dry = _assert_dry_generation(contract)
    if CANONICAL_PREFLIGHT_DIRECTORY.exists():
        raise FileExistsError("canonical actual preflight output already exists")
    CANONICAL_PREFLIGHT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_status(
        PREFLIGHT_STATUS,
        progress=25,
        phase="scope_grant",
        detail="canonical amendment validated before ERA5 value reads",
        started=started,
    )
    grant = _scope_grant(contract)
    if not grant.accepted:
        raise PermissionError("canonical P2 ERA5 scope grant was rejected")
    grant_value_contract = _assert_grant_value_contract(contract, grant)
    schema = pq.ParquetFile(grant.parquet_path).schema_arrow
    expected_era5_columns = {
        "chunk_id",
        "block",
        "time_utc",
        "time_kst",
        "latitude",
        "longitude",
        *ERA5_VALUE_COLUMNS,
    }
    if not expected_era5_columns.issubset(schema.names):
        raise ValueError("grant-bound ERA5 schema changed")
    shard_contract = contract["frozen_inputs"]["truth_vault_shards"]
    truth_shard_keys = validate_truth_shard_key_contract(
        REPO_ROOT,
        shard_contract["shards"],
        expected_union_rows=shard_contract["union_rows"],
        expected_union_key_sha256=shard_contract["union_key_sha256"],
        outer_blocks=contract["oof_contract"]["outer_blocks"],
    )
    if truth_shard_keys["union_key_sha256"] != contract["oof_contract"]["key_sha256"]:
        raise ValueError("truth-vault shard union differs from the frozen OOF keys")
    truth_shard_keys["union_equals_frozen_oof_key_contract"] = True
    expert_keys = _preflight_expert_key_contract(contract)
    if expert_keys["union_key_sha256"] != truth_shard_keys["union_key_sha256"]:
        raise ValueError("truth shard and expert OOF key-only contracts differ")

    o_excl = _o_excl_probe(CANONICAL_PREFLIGHT_DIRECTORY / "o_excl_probe.lock")
    ledger_path = CANONICAL_PREFLIGHT_DIRECTORY / "ledger_probe.jsonl"
    ledger = AppendOnlyLedger(ledger_path, experiment_id=EXPERIMENT_ID)
    ledger.append("preflight_probe", {"raw_values": False})
    ledger_audit = verify_append_only_ledger(ledger_path, experiment_id=EXPERIMENT_ID)
    implementation = _implementation_hashes()
    _write_status(
        PREFLIGHT_STATUS,
        progress=75,
        phase="locks_and_bundle",
        detail="O_EXCL, append-only ledger, implementation hashes",
        started=started,
    )
    qa_expected = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "status": "passed",
        "independent": True,
        "config_sha256": implementation["config"],
        "helper_sha256": implementation["helper"],
        "runner_sha256": implementation["runner"],
        "tests_sha256": implementation["tests"],
        "actual_execution_performed": False,
    }
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "preflight",
        "status": "passed_preflight_actual_locked",
        "actual_ready": False,
        "actual_blockers": [
            "independent_static_qa_receipt_missing",
            "o_excl_authorization_missing",
        ],
        "canonical_config_sha256": CANONICAL_CONFIG_SHA256,
        "dry_generation": dry,
        "scope_grant": grant.public_dict(),
        "grant_value_contract": grant_value_contract,
        "native_flux_semantics": validate_native_flux_sign_contract(contract),
        "truth_shard_key_preflight": truth_shard_keys,
        "expert_oof_key_preflight": expert_keys,
        "era5_schema_metadata_read": True,
        "era5_value_pages_read": 0,
        "o_excl_probe": o_excl,
        "ledger_probe": ledger_audit,
        "implementation_sha256": implementation,
        "qa_receipt_canonical_path": _relative(CANONICAL_QA_RECEIPT),
        "qa_required_fields_before_preflight_sha": qa_expected,
        "authorization_canonical_path": _relative(CANONICAL_AUTHORIZATION),
        "access_counters": {
            "era5_value_reads": 0,
            "expert_prediction_reads": 0,
            "expert_oof_key_only_reads": 2,
            "observation_reads": 0,
            "fold_training_truth_opens": 0,
            "designated_outer_truth_opens": 0,
            "truth_shard_key_only_reads": 3,
            "truth_value_column_reads": 0,
            "model_fits": 0,
            "test_reads": 0,
            "submission_reads_or_writes": 0,
            "uploads": 0,
        },
    }
    receipt_path = CANONICAL_PREFLIGHT_RECEIPT
    write_json_exclusive_fsync(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "actual_ready": False,
        "preflight_receipt_sha256": receipt_sha256,
        "implementation_sha256": implementation,
        "models_written": 0,
        "submissions_written": 0,
    }
    write_json_exclusive_fsync(CANONICAL_PREFLIGHT_DIRECTORY / "manifest.json", manifest)
    _write_status(
        PREFLIGHT_STATUS,
        progress=100,
        phase="complete",
        detail="preflight passed; actual locked for independent QA and authorization",
        started=started,
        status="completed",
    )
    return {
        "status": receipt["status"],
        "actual_ready": False,
        "receipt_path": _relative(receipt_path),
        "receipt_sha256": receipt_sha256,
        "manifest_path": _relative(CANONICAL_PREFLIGHT_DIRECTORY / "manifest.json"),
        "manifest_sha256": sha256_file(CANONICAL_PREFLIGHT_DIRECTORY / "manifest.json"),
    }


def _validate_qa_receipt(
    *,
    qa_path: Path,
    qa_sha256: str,
    preflight_path: Path,
    preflight_sha256: str,
) -> dict[str, Any]:
    if qa_path.resolve() != CANONICAL_QA_RECEIPT.resolve():
        raise PermissionError("only the canonical independent QA receipt path is accepted")
    if preflight_path.resolve() != CANONICAL_PREFLIGHT_RECEIPT.resolve():
        raise PermissionError("only the canonical preflight receipt path is accepted")
    if sha256_file(qa_path) != qa_sha256 or sha256_file(preflight_path) != preflight_sha256:
        raise ValueError("QA or preflight receipt CLI SHA-256 differs")
    qa = _read_json(qa_path)
    preflight = _read_json(preflight_path)
    implementation = _implementation_hashes()
    required = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "status": "passed",
        "independent": True,
        "config_sha256": implementation["config"],
        "helper_sha256": implementation["helper"],
        "runner_sha256": implementation["runner"],
        "tests_sha256": implementation["tests"],
        "preflight_receipt_sha256": preflight_sha256,
        "actual_execution_performed": False,
    }
    mismatches = [name for name, expected in required.items() if qa.get(name) != expected]
    if mismatches or preflight.get("status") != "passed_preflight_actual_locked":
        raise PermissionError(f"independent QA receipt validation failed: {mismatches}")
    return {"qa": qa, "qa_sha256": qa_sha256, "preflight_sha256": preflight_sha256}


def run_authorize(args: argparse.Namespace) -> dict[str, Any]:
    contract = _load_canonical_config()
    dry = _assert_dry_generation(contract)
    if args.authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("explicit canonical authorization phrase is required")
    qa = _validate_qa_receipt(
        qa_path=args.qa_file,
        qa_sha256=args.qa_sha256,
        preflight_path=args.preflight_receipt,
        preflight_sha256=args.preflight_sha256,
    )
    payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "generation": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "authorized": True,
        "maximum_actual_attempts": 1,
        "config_sha256": CANONICAL_CONFIG_SHA256,
        "dry_receipt_sha256": dry["artifacts"]["receipt_sha256"],
        "qa_receipt_sha256": qa["qa_sha256"],
        "preflight_receipt_sha256": qa["preflight_sha256"],
        "authorization_phrase_sha256": AUTHORIZATION_PHRASE_SHA256,
        "test_or_submission_authorized": False,
        "upload_authorized": False,
    }
    write_json_exclusive_fsync(CANONICAL_AUTHORIZATION, payload)
    return {
        "status": "authorized_not_executed",
        "actual_executed": False,
        "authorization_path": _relative(CANONICAL_AUTHORIZATION),
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
    }


def _validate_authorization(
    args: argparse.Namespace,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if args.authorization_file.resolve() != CANONICAL_AUTHORIZATION.resolve():
        raise PermissionError("only the canonical O_EXCL authorization path is accepted")
    if sha256_file(args.authorization_file) != args.authorization_sha256:
        raise ValueError("authorization CLI SHA-256 differs")
    qa = _validate_qa_receipt(
        qa_path=args.qa_file,
        qa_sha256=args.qa_sha256,
        preflight_path=args.preflight_receipt,
        preflight_sha256=args.preflight_sha256,
    )
    authorization = _read_json(args.authorization_file)
    required = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "generation": 1,
        "authorized": True,
        "maximum_actual_attempts": 1,
        "config_sha256": CANONICAL_CONFIG_SHA256,
        "dry_receipt_sha256": EXPECTED_DRY_RECEIPT_SHA256,
        "qa_receipt_sha256": qa["qa_sha256"],
        "preflight_receipt_sha256": qa["preflight_sha256"],
        "authorization_phrase_sha256": AUTHORIZATION_PHRASE_SHA256,
        "test_or_submission_authorized": False,
        "upload_authorized": False,
    }
    mismatches = [
        name for name, expected in required.items() if authorization.get(name) != expected
    ]
    if mismatches or contract["access_bans"]["submission_directory"] is not True:
        raise PermissionError(f"canonical authorization validation failed: {mismatches}")
    return {
        "authorization_sha256": args.authorization_sha256,
        "qa_sha256": qa["qa_sha256"],
        "preflight_sha256": qa["preflight_sha256"],
    }


def _canonical_key_digest(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, ["time", "layer", "block"]].copy()
    normalized["time"] = pd.to_datetime(normalized["time"], utc=True).map(
        lambda value: value.isoformat()
    )
    payload = normalized.to_csv(index=False, header=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _preflight_expert_key_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen expert containers using file hashes and key columns only."""

    frozen = contract["frozen_inputs"]
    audits: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name in ("deep_oof", "physical_oof"):
        spec = frozen[name]
        path = _repo_path(spec["path"])
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"{name} SHA-256 changed during key-only preflight")
        frame = pq.read_table(path, columns=["time", "layer", "block"]).to_pandas()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        frame["layer"] = frame["layer"].astype("int64")
        frame["block"] = frame["block"].astype("string")
        digest = _canonical_key_digest(frame)
        if len(frame) != contract["oof_contract"]["rows"]:
            raise ValueError(f"{name} key-only row count changed")
        if frame.loc[:, ["time", "layer", "block"]].duplicated().any():
            raise ValueError(f"{name} key-only rows are duplicated")
        frames[name] = frame
        audits[name] = {
            "sha256": spec["sha256"],
            "rows": int(len(frame)),
            "key_sha256": digest,
            "columns_requested": ["time", "layer", "block"],
            "prediction_columns_requested": 0,
            "truth_value_columns_requested": 0,
        }
    expected = contract["oof_contract"]["key_sha256"]
    if any(audit["key_sha256"] != expected for audit in audits.values()):
        raise ValueError("frozen expert key digest differs from the OOF contract")
    if not frames["deep_oof"].equals(frames["physical_oof"]):
        raise ValueError("frozen expert key-only rows differ")
    return {
        **audits,
        "deep_physical_keys_exact_equal": True,
        "union_key_sha256": expected,
        "prediction_columns_requested": 0,
        "truth_value_columns_requested": 0,
    }


def _load_expert_design(contract: Mapping[str, Any]) -> pd.DataFrame:
    frozen = contract["frozen_inputs"]
    deep_path = _repo_path(frozen["deep_oof"]["path"])
    physical_path = _repo_path(frozen["physical_oof"]["path"])
    if sha256_file(deep_path) != frozen["deep_oof"]["sha256"]:
        raise ValueError("deep OOF SHA-256 changed")
    if sha256_file(physical_path) != frozen["physical_oof"]["sha256"]:
        raise ValueError("physical OOF SHA-256 changed")
    keys = ["time", "layer", "block"]
    deep = pq.read_table(
        deep_path,
        columns=[*keys, frozen["deep_oof"]["expert_column"]],
    ).to_pandas()
    physical = pq.read_table(
        physical_path,
        columns=[*keys, frozen["physical_oof"]["expert_column"]],
    ).to_pandas()
    deep = deep.rename(columns={frozen["deep_oof"]["expert_column"]: "deep_prediction"})
    physical = physical.rename(
        columns={frozen["physical_oof"]["expert_column"]: "physical_prediction"}
    )
    for frame in (deep, physical):
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    design = deep.merge(physical, on=keys, how="inner", validate="one_to_one")
    oof = contract["oof_contract"]
    if len(design) != oof["rows"] or _canonical_key_digest(design) != oof["key_sha256"]:
        raise ValueError("frozen expert OOF key contract changed")
    if not np.isfinite(design[["deep_prediction", "physical_prediction"]].to_numpy()).all():
        raise ValueError("frozen expert predictions contain non-finite values")
    return design


def _load_era5_features(
    contract: Mapping[str, Any], grant: object, ledger: AppendOnlyLedger
) -> tuple[pd.DataFrame, dict[str, Any], GrantBoundEra5Reader]:
    reader = GrantBoundEra5Reader()
    reader.bind_grant(grant.parquet_path)
    ledger.append("era5_value_read_started", {"grant_sha256": grant.parquet_sha256})
    columns = (
        "chunk_id",
        "block",
        "time_utc",
        "time_kst",
        "latitude",
        "longitude",
        *ERA5_VALUE_COLUMNS,
    )
    frame = reader.read_values(columns=columns)
    quality = validate_era5_source_frame(
        frame,
        expected_rows=contract["frozen_inputs"]["era5_parquet"]["rows"],
        expected_blocks=contract["oof_contract"]["outer_blocks"],
        expected_grid_points=9,
    )
    hourly = build_hourly_ocean_mixing_features(frame, expected_ocean_cells_per_hour=9)
    return hourly, quality, reader


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = _load_canonical_config()
    dry = _assert_dry_generation(contract)
    authorization = _validate_authorization(args, contract=contract)
    grant = _scope_grant(contract)
    if not grant.accepted:
        raise PermissionError("canonical ERA5 scope grant was rejected")
    grant_value_contract = _assert_grant_value_contract(contract, grant)
    if CANONICAL_ACTUAL_DIRECTORY.exists():
        raise FileExistsError("canonical actual generation 1 output already exists")
    CANONICAL_ACTUAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    attempt_lock = CANONICAL_ACTUAL_DIRECTORY / "attempt.lock"
    write_json_exclusive_fsync(
        attempt_lock,
        {
            "experiment_id": EXPERIMENT_ID,
            "generation": 1,
            "authorization_sha256": authorization["authorization_sha256"],
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )
    ledger = AppendOnlyLedger(
        CANONICAL_ACTUAL_DIRECTORY / contract["outputs"]["ledger"],
        experiment_id=EXPERIMENT_ID,
    )
    ledger.append("attempt_reserved", {"attempt_lock_sha256": sha256_file(attempt_lock)})
    _write_status(
        ACTUAL_STATUS,
        progress=5,
        phase="attempt_reserved",
        detail="generation 1 permanently reserved before value reads",
        started=started,
    )

    hourly_era5, era5_quality, era5_reader = _load_era5_features(contract, grant, ledger)
    ledger.append("expert_prediction_read_started", {"truth_columns": False})
    design = _load_expert_design(contract)
    observations, observations_audit = load_observations_only(
        args.data_dir,
        expected_sha256=contract["frozen_inputs"]["observations"]["sha256"],
    )
    public, mask_audits = build_block_masked_public_state_panel(
        observations,
        design.loc[:, ["time", "layer", "block"]],
    )
    era5, join_audit = align_mixing_features_to_oof_keys(
        hourly_era5,
        design.loc[:, ["time", "layer", "block"]],
    )
    if not design.loc[:, ["time", "layer", "block"]].equals(
        public.loc[:, ["time", "layer", "block"]]
    ) or not design.loc[:, ["time", "layer", "block"]].equals(
        era5.loc[:, ["time", "layer", "block"]]
    ):
        raise ValueError("actual public/ERA5/expert design keys differ")
    design = pd.concat(
        [
            design.reset_index(drop=True),
            public.loc[:, STATE_FEATURES].reset_index(drop=True),
            era5.loc[:, ERA5_MIXING_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    _write_status(
        ACTUAL_STATUS,
        progress=28,
        phase="blind_design",
        detail="experts, jointly masked public state, and ERA5 features aligned",
        started=started,
    )

    shard_contract = contract["frozen_inputs"]["truth_vault_shards"]
    truth_column = shard_contract["truth_column"]
    outcomes = []
    fold_vault_audits = []
    for number, outer in enumerate(contract["oof_contract"]["outer_blocks"]):
        lock_path = CANONICAL_ACTUAL_DIRECTORY / "locks" / f"outer_{outer}.lock"
        write_json_exclusive_fsync(
            lock_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "outer_block": outer,
                "current_outer_truth_authorized": False,
            },
        )
        ledger.append(
            "outer_fold_reserved",
            {"outer_block": outer, "lock_sha256": sha256_file(lock_path)},
        )
        fold_vault = FoldLocalTruthVault(
            REPO_ROOT,
            shard_contract["shards"],
            truth_column=truth_column,
            outer_blocks=contract["oof_contract"]["outer_blocks"],
        )
        fold_truth = fold_vault.open_fold_train(outer)
        ledger.append(
            "fold_train_truth_opened",
            {
                "outer_block": outer,
                "allowed_train_blocks": sorted(fold_truth["block"].astype(str).unique()),
                "current_outer_truth_rows": 0,
            },
        )
        outcome = run_fold_local_gate(design, fold_truth, outer_block=outer, purge_hours=168)
        outcomes.append(outcome)
        fold_vault_audit = fold_vault.audit()
        if any(entry["block"] == outer for entry in fold_vault_audit["truth_value_open_log"]):
            raise AssertionError("current outer truth shard appears in the fold open log")
        fold_vault_audits.append(fold_vault_audit)
        del fold_truth
        del fold_vault
        ledger.append(
            "outer_prediction_complete",
            {
                "outer_block": outer,
                "selected_arm": outcome.selected_arm,
                "inner_gate_passed": outcome.inner_gate_passed,
                "rows": len(outcome.predictions),
            },
        )
        _write_status(
            ACTUAL_STATUS,
            progress=35 + 14 * (number + 1),
            phase="fold_local_outer",
            detail=f"{number + 1}/3 outer predictions complete without current truth",
            started=started,
        )

    combined = pd.concat([outcome.predictions for outcome in outcomes], ignore_index=True)
    combined = design.loc[:, ["time", "layer", "block"]].merge(
        combined,
        on=["time", "layer", "block"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    blind_path = CANONICAL_ACTUAL_DIRECTORY / contract["outputs"]["blind_predictions"]
    seal_path = CANONICAL_ACTUAL_DIRECTORY / contract["outputs"]["blind_seal"]
    blind = write_and_seal_blind_predictions(
        combined,
        parquet_path=blind_path,
        seal_path=seal_path,
        seal_metadata={
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": CANONICAL_CONFIG_SHA256,
            "attempt_lock_sha256": sha256_file(attempt_lock),
            "authorization_sha256": authorization["authorization_sha256"],
            "fold_selected_arms": {
                outcome.outer_block: outcome.selected_arm for outcome in outcomes
            },
            "fold_training_truth_opens": len(outcomes),
            "fold_training_truth_shard_value_opens": sum(
                len(audit["truth_value_open_log"]) for audit in fold_vault_audits
            ),
            "fresh_truth_vault_instance_per_outer_fold": True,
            "fold_truth_frame_released_before_next_outer_lock": True,
            "current_outer_truth_rows_seen_during_fit": 0,
            "global_zero_target_exposure_before_seal_claimed": False,
            "designated_scoring_truth_open_count_before_seal": 0,
        },
    )
    ledger.append("blind_outer_predictions_sealed", blind)
    all_no_op = all(outcome.selected_arm == "control" for outcome in outcomes)
    result_common = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "uploaded": False,
        "submission_created": False,
        "models_persisted": False,
        "dry_generation": dry,
        "authorization": authorization,
        "scope_grant": grant.public_dict(),
        "grant_value_contract": grant_value_contract,
        "native_flux_semantics": validate_native_flux_sign_contract(contract),
        "era5_quality": era5_quality,
        "era5_read_events": era5_reader.events,
        "era5_value_read_count": era5_reader.value_read_count,
        "observations": observations_audit,
        "joint_mask_audits": mask_audits,
        "era5_join": join_audit,
        "folds": [outcome.summary for outcome in outcomes],
        "blind": blind,
        "fold_truth_vaults": fold_vault_audits,
        "test_reads": 0,
        "submission_reads_or_writes": 0,
        "uploads": 0,
    }
    result_path = CANONICAL_ACTUAL_DIRECTORY / contract["outputs"]["result"]
    if all_no_op:
        result = {
            **result_common,
            "designated_scoring_truth_vault": None,
            "status": "all_folds_no_op_designated_outer_truth_unopened",
            "decision": "REJECT_ERA5_INCREMENT_KEEP_CONTROL",
            "designated_outer_truth_open_count": 0,
        }
        write_json_exclusive_fsync(result_path, result)
        ledger.append("completed_without_outer_truth", {"result_sha256": sha256_file(result_path)})
    else:
        outer_lock = CANONICAL_ACTUAL_DIRECTORY / "outer_evaluation.lock"
        write_json_exclusive_fsync(
            outer_lock,
            {
                "experiment_id": EXPERIMENT_ID,
                "blind_seal_sha256": blind["seal_sha256"],
            },
        )
        ledger.append(
            "outer_evaluation_reserved",
            {"outer_lock_sha256": sha256_file(outer_lock)},
        )
        scoring_vault = FoldLocalTruthVault(
            REPO_ROOT,
            shard_contract["shards"],
            truth_column=truth_column,
            outer_blocks=contract["oof_contract"]["outer_blocks"],
        )
        outer_truth = scoring_vault.open_designated_outer_once(
            blind_seal_path=seal_path,
            expected_blind_seal_sha256=blind["seal_sha256"],
            completed_outer_blocks=[outcome.outer_block for outcome in outcomes],
        )
        ledger.append("designated_outer_truth_opened", {"semantic_open_count": 1})
        reloaded_blind = pd.read_parquet(blind_path)
        scored = reloaded_blind.merge(
            outer_truth,
            on=["time", "layer", "block"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(scored) != contract["oof_contract"]["rows"] or scored["truth"].isna().any():
            raise ValueError("designated outer truth did not attach to every sealed prediction")
        metrics = metric_summary(scored)
        bootstrap = paired_kst_day_bootstrap(
            scored,
            replicates=contract["outer_promotion"]["paired_kst_day_bootstrap_replicates"],
            seed=contract["outer_promotion"]["bootstrap_seed"],
        )
        promotion = outer_promotion_decision(metrics, bootstrap)
        result = {
            **result_common,
            "designated_scoring_truth_vault": scoring_vault.audit(),
            "status": "outer_evaluated_once",
            "decision": (
                "PROMOTE_LOCAL_RESEARCH_CHALLENGER_NO_SUBMISSION"
                if promotion["promoted"]
                else "REJECT_ERA5_INCREMENT_KEEP_CONTROL"
            ),
            "designated_outer_truth_open_count": 1,
            "metrics": metrics,
            "bootstrap": bootstrap,
            "promotion": promotion,
        }
        write_json_exclusive_fsync(result_path, result)
        ledger.append("completed", {"result_sha256": sha256_file(result_path)})
    ledger_audit = verify_append_only_ledger(
        CANONICAL_ACTUAL_DIRECTORY / contract["outputs"]["ledger"],
        experiment_id=EXPERIMENT_ID,
    )
    _write_status(
        ACTUAL_STATUS,
        progress=100,
        phase="complete",
        detail="one-shot local research evaluation complete; submission/upload banned",
        started=started,
        status="completed",
    )
    return {
        "status": result["status"],
        "decision": result["decision"],
        "result_path": _relative(result_path),
        "result_sha256": sha256_file(result_path),
        "ledger": ledger_audit,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "authorize", "actual"), required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--qa-file", type=Path)
    parser.add_argument("--qa-sha256")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    if args.mode == "preflight":
        result = run_preflight()
    elif args.mode == "authorize":
        required = (
            args.qa_file,
            args.qa_sha256,
            args.preflight_receipt,
            args.preflight_sha256,
            args.authorization_phrase,
        )
        if any(value is None for value in required):
            parser.error("authorize requires QA/preflight paths+SHA and explicit phrase")
        result = run_authorize(args)
    else:
        required = (
            args.data_dir,
            args.qa_file,
            args.qa_sha256,
            args.preflight_receipt,
            args.preflight_sha256,
            args.authorization_file,
            args.authorization_sha256,
        )
        if any(value is None for value in required):
            parser.error("actual requires data-dir, QA, preflight, and authorization paths+SHA")
        result = run_actual(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
