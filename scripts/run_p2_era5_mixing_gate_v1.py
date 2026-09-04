"""Dry-run the preregistered P2 ERA5 mixing-gate generation.

Only independent ERA5 values and frozen OOF key columns are parsed.  This
generation deliberately contains no label access, fitting, test inference, or
submission path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from ocean_external.p2_era5_scope import validate_p2_era5_scope_amendment
from p2_restore.era5_mixing_gate import (
    ERA5_MIXING_FEATURES,
    ERA5_VALUE_COLUMNS,
    align_mixing_features_to_oof_keys,
    build_hourly_ocean_mixing_features,
    validate_era5_source_frame,
    validate_manifest_units_and_signs,
    validate_preregistered_feature_contract,
)
from p2_restore.regime_gate import STATE_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_era5_mixing_gate_v1"
RUNNER_PATH = Path(__file__).resolve()
HELPER_PATH = REPO_ROOT / "src/p2_restore/era5_mixing_gate.py"
TEST_PATH = REPO_ROOT / "tests/test_p2_era5_mixing_gate.py"
POLICY_VALIDATOR_PATH = REPO_ROOT / "src/ocean_external/p2_era5_scope.py"
KEY_COLUMNS = ("time", "layer", "block")
FORBIDDEN_DRY_RUN_COLUMNS = {
    "truth",
    "prediction",
    "lobo_prediction",
    "baseline",
    "baseline_prediction",
    "routed_prediction",
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("experiment paths must remain under the repository root") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
        actual_ready: bool = False,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(float(progress), 0.1), 100.0)
        remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100.0 else 0.0
        payload = {
            "title": "P2 ERA5 mixing-gate v1 dry-run",
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "progress": bounded,
            "phase": phase,
            "detail": detail,
            "actual_ready": bool(actual_ready),
            "elapsed_seconds": elapsed,
            "eta": (datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))).strftime(
                "%Y-%m-%d %H:%M:%S KST"
            ),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        _write_json_atomic(self.path, payload)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {_relative(path)}")
    return value


def _validate_config(path: Path) -> dict[str, Any]:
    contract = _read_json_object(path)
    if contract.get("schema_version") != "1.0":
        raise ValueError("mixing-gate config schema version changed")
    if contract.get("experiment_id") != EXPERIMENT_ID or contract.get("generation") != 1:
        raise ValueError("mixing-gate experiment identity changed")
    authorization = contract.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("dry_run") is not True:
        raise ValueError("dry-run is not authorized")
    forbidden_authorizations = (
        "inner_fit_or_label_access",
        "outer_fit_or_label_access",
        "final_fit",
        "model_read_or_write",
        "test_read",
        "submission_read_or_write",
        "catalog_mutation",
    )
    if any(authorization.get(name) is not False for name in forbidden_authorizations):
        raise ValueError("actual/model/test/submission/catalog authorization must remain false")
    attempt = contract.get("attempt_lock", {})
    if (
        attempt.get("completed_inner_attempts") != 0
        or attempt.get("completed_outer_attempts") != 0
        or attempt.get("maximum_inner_attempts") != 1
        or attempt.get("maximum_outer_attempts") != 1
    ):
        raise ValueError("canonical attempt lock changed")
    validate_preregistered_feature_contract(contract.get("era5_mixing_features", []))
    if tuple(contract.get("public_state_features", [])) != STATE_FEATURES:
        raise ValueError("public-state control feature contract changed")
    model = contract.get("gate_model", {})
    if (
        model.get("parameter_grid") != []
        or model.get("parameter_trials") != 0
        or model.get("regularization") != 10.0
        or model.get("max_iterations") != 1000
        or model.get("seed") != 20260821
        or model.get("extrapolation_allowed") is not False
    ):
        raise ValueError("fixed no-grid gate contract changed")
    if contract.get("outputs", {}).get("collision_policy") != "fail_close":
        raise ValueError("output collision policy must remain fail-close")
    policy = contract.get("policy_reconciliation", {})
    if (
        policy.get("canonical_amendment_sha256")
        != "b2a1fa3059e1ee114b1be4d7f596b02e97ca8354365a3c1ee1bacd6f443940c9"
        or policy.get("canonical_validator_sha256")
        != "f1902279d01f04d5890e3d2e671d570f4c7675817320c82a5b1849d085567122"
        or policy.get("problem") != "P2"
        or policy.get("source_id") != "era5_pre2024"
        or policy.get("purpose") != "feature_design"
        or policy.get("independent_static_modeling_qa_status") != "pending"
    ):
        raise ValueError("canonical P2 ERA5 policy/static-QA lock changed")
    return contract


def _verify_pinned_file(
    specification: Mapping[str, Any], *, granted_path: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    configured_path = _repo_path(str(specification["path"]))
    path = granted_path.resolve() if granted_path is not None else configured_path
    if path != configured_path.resolve():
        raise ValueError("validated grant path differs from the pinned config path")
    if not path.is_file():
        raise FileNotFoundError(f"pinned input is missing: {_relative(path)}")
    observed_hash = _sha256(path)
    if observed_hash != specification["sha256"]:
        raise ValueError(f"pinned SHA-256 changed: {_relative(path)}")
    if "bytes" in specification and path.stat().st_size != int(specification["bytes"]):
        raise ValueError(f"pinned byte size changed: {_relative(path)}")
    return path, {
        "path": _relative(path),
        "sha256": observed_hash,
        "bytes": int(path.stat().st_size),
    }


def _canonical_key_digest(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, KEY_COLUMNS].copy()
    normalized["time"] = pd.to_datetime(normalized["time"], utc=True).map(
        lambda value: value.isoformat()
    )
    payload = normalized.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_oof_keys_only(
    path: Path,
    *,
    expected_rows: int,
    expected_blocks: Mapping[str, int],
    expected_layers: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    schema_columns = tuple(parquet.schema_arrow.names)
    if not set(KEY_COLUMNS).issubset(schema_columns):
        raise ValueError(f"OOF key schema is incomplete: {_relative(path)}")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"OOF row count changed: {_relative(path)}")

    # Deliberately project only the three public key columns.  Prediction and
    # label names may exist in metadata but their pages are never parsed.
    table = pq.read_table(path, columns=list(KEY_COLUMNS))
    if tuple(table.column_names) != KEY_COLUMNS:
        raise AssertionError("Parquet key-only projection changed")
    frame = table.to_pandas()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError(f"OOF keys contain duplicates: {_relative(path)}")
    block_counts = {
        str(key): int(value) for key, value in frame["block"].value_counts().sort_index().items()
    }
    layer_counts = {
        str(key): int(value) for key, value in frame["layer"].value_counts().sort_index().items()
    }
    if block_counts != {str(key): int(value) for key, value in expected_blocks.items()}:
        raise ValueError(f"OOF block counts changed: {_relative(path)}")
    if layer_counts != {str(key): int(value) for key, value in expected_layers.items()}:
        raise ValueError(f"OOF layer counts changed: {_relative(path)}")
    return frame, {
        "rows": int(len(frame)),
        "row_groups": int(parquet.metadata.num_row_groups),
        "schema_column_count": len(schema_columns),
        "schema_contains_label_or_prediction_names": bool(
            FORBIDDEN_DRY_RUN_COLUMNS.intersection(schema_columns)
        ),
        "columns_parsed": list(KEY_COLUMNS),
        "label_or_prediction_columns_parsed": 0,
        "duplicate_key_count": 0,
        "block_counts": block_counts,
        "layer_counts": layer_counts,
        "time_start_utc": frame["time"].min().isoformat(),
        "time_end_utc": frame["time"].max().isoformat(),
        "key_sha256": _canonical_key_digest(frame),
    }


def _validate_policy_reconciliation(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], object]:
    policy = contract["policy_reconciliation"]
    amendment_path = _repo_path(policy["canonical_amendment_path"])
    validator_path = _repo_path(policy["canonical_validator_path"])
    if _sha256(amendment_path) != policy["canonical_amendment_sha256"]:
        raise ValueError("canonical P2 ERA5 amendment SHA changed")
    if _sha256(validator_path) != policy["canonical_validator_sha256"]:
        raise ValueError("canonical P2 ERA5 validator SHA changed")

    frozen = contract["frozen_inputs"]
    configured_parquet = _repo_path(frozen["era5_parquet"]["path"])
    grant = validate_p2_era5_scope_amendment(
        repo_root=REPO_ROOT,
        amendment_path=amendment_path,
        problem=str(policy["problem"]),
        source_id=str(policy["source_id"]),
        purpose=str(policy["purpose"]),
        candidate_parquet_path=configured_parquet,
    )
    if not grant.accepted or grant.parquet_path.resolve() != configured_parquet.resolve():
        raise ValueError("canonical P2 ERA5 scope grant did not accept the pinned Parquet")
    result = {
        "status": "canonical_scope_amendment_validated",
        "passed": True,
        "grant": grant.public_dict(),
        "amendment_path": _relative(amendment_path),
        "amendment_sha256": _sha256(amendment_path),
        "validator_path": _relative(validator_path),
        "validator_sha256": _sha256(validator_path),
        "catalog_mutated": False,
        "independent_static_modeling_qa_required": True,
        "independent_static_modeling_qa_status": policy[
            "independent_static_modeling_qa_status"
        ],
        "actual_ready_by_policy_only": True,
        "actual_ready_overall": False,
    }
    return result, grant


def _code_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "config": config_path,
        "runner": RUNNER_PATH,
        "feature_helper": HELPER_PATH,
        "tests": TEST_PATH,
        "policy_validator": POLICY_VALIDATOR_PATH,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"blind-seal code files are missing: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _blind_seal(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    frozen = contract["frozen_inputs"]
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "generation": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "before_any_designated_truth_or_prediction_parse",
        "research_only": True,
        "actual_ready": False,
        "code_sha256": _code_hashes(config_path),
        "input_sha256_pins": {
            "canonical_scope_amendment": contract["policy_reconciliation"][
                "canonical_amendment_sha256"
            ],
            "canonical_scope_validator": contract["policy_reconciliation"][
                "canonical_validator_sha256"
            ],
            "era5_parquet": frozen["era5_parquet"]["sha256"],
            "era5_manifest": frozen["era5_manifest"]["sha256"],
            "deep_oof_opaque_container": frozen["deep_oof"]["sha256"],
            "physical_oof_opaque_container": frozen["physical_oof"]["sha256"],
        },
        "attempt_lock": contract["attempt_lock"],
        "access_counters": {
            "truth_or_label_columns_parsed": 0,
            "expert_prediction_columns_parsed": 0,
            "public_observations_read": 0,
            "model_files_read_or_written": 0,
            "test_files_read": 0,
            "submission_files_read_or_written": 0,
        },
        "outer_truth_locked": True,
    }


def _load_era5_allowed_columns(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    parquet = pq.ParquetFile(path)
    schema_columns = tuple(parquet.schema_arrow.names)
    allowed = (
        "chunk_id",
        "block",
        "time_utc",
        "time_kst",
        "latitude",
        "longitude",
        *ERA5_VALUE_COLUMNS,
    )
    if not set(allowed).issubset(schema_columns):
        raise ValueError("ERA5 Parquet schema is missing preregistered columns")
    table = pq.read_table(path, columns=list(allowed))
    if tuple(table.column_names) != allowed:
        raise AssertionError("ERA5 allowed-column projection changed")
    return table.to_pandas(), schema_columns


def run_dry_run(
    *,
    config_path: Path,
    output_directory: Path,
    status_path: Path,
) -> dict[str, Any]:
    progress = Progress(status_path)
    progress.update(2, "contract", "dry-run authorization and no-grid contract validation")
    contract = _validate_config(config_path)
    if output_directory.exists():
        raise FileExistsError(
            f"fail-close output collision: {_relative(output_directory)} already exists"
        )
    _relative(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    seal_path = output_directory / contract["outputs"]["blind_seal"]
    seal = _blind_seal(contract, config_path)
    _write_json_atomic(seal_path, seal)
    progress.update(8, "blind_seal", "truth/prediction access counters sealed at zero")

    policy, grant = _validate_policy_reconciliation(contract)
    progress.update(
        15,
        "policy",
        "canonical scope grant valid; catalog unchanged; actual remains static-QA locked",
    )

    frozen = contract["frozen_inputs"]
    era5_path, era5_file = _verify_pinned_file(
        frozen["era5_parquet"], granted_path=grant.parquet_path
    )
    manifest_path, era5_manifest_file = _verify_pinned_file(frozen["era5_manifest"])
    deep_path, deep_file = _verify_pinned_file(frozen["deep_oof"])
    physical_path, physical_file = _verify_pinned_file(frozen["physical_oof"])
    public_state_path, public_state_file = _verify_pinned_file(
        frozen["public_state_implementation"]
    )
    progress.update(28, "hashes", "five pinned inputs match exact SHA-256 values")

    oof = contract["oof_contract"]
    deep_keys, deep_key_audit = _read_oof_keys_only(
        deep_path,
        expected_rows=int(oof["rows"]),
        expected_blocks=oof["block_rows"],
        expected_layers=oof["layer_rows"],
    )
    physical_keys, physical_key_audit = _read_oof_keys_only(
        physical_path,
        expected_rows=int(oof["rows"]),
        expected_blocks=oof["block_rows"],
        expected_layers=oof["layer_rows"],
    )
    exact_key_order_match = deep_keys.equals(physical_keys)
    if not exact_key_order_match:
        raise ValueError("deep and physical frozen OOF key/order contracts differ")
    progress.update(43, "oof_keys", "69,850 frozen keys aligned; labels/predictions not parsed")

    manifest = _read_json_object(manifest_path)
    source = contract["era5_source_contract"]
    validate_manifest_units_and_signs(
        manifest,
        expected_units=source["units"],
        expected_signs=source["sign_semantics"],
    )
    manifest_output = manifest.get("output", {})
    if (
        manifest_output.get("sha256") != frozen["era5_parquet"]["sha256"]
        or manifest_output.get("bytes") != frozen["era5_parquet"]["bytes"]
        or manifest.get("validation", {}).get("chunk_count")
        != frozen["era5_parquet"]["chunk_count"]
    ):
        raise ValueError("ERA5 manifest output pins changed")

    era5_frame, era5_schema = _load_era5_allowed_columns(era5_path)
    era5_audit = validate_era5_source_frame(
        era5_frame,
        expected_rows=int(frozen["era5_parquet"]["rows"]),
        expected_blocks=oof["blocks"],
        expected_grid_points=int(frozen["era5_parquet"]["unique_grid_point_count"]),
    )
    if era5_audit["chunk_count"] != frozen["era5_parquet"]["chunk_count"]:
        raise ValueError("ERA5 observed chunk count changed")
    if era5_audit["unique_hour_count"] != frozen["era5_parquet"]["unique_hour_count"]:
        raise ValueError("ERA5 observed hour count changed")
    cutoff = pd.Timestamp(grant.effective_cutoff_kst)
    observed_end = pd.Timestamp(str(era5_audit["time_end_kst"]))
    if observed_end > cutoff:
        raise ValueError("ERA5 source exceeds the P2 approved cutoff")
    progress.update(61, "era5_quality", "8 mixing variables + LSM validated at 3x3 hourly grain")

    hourly_features = build_hourly_ocean_mixing_features(
        era5_frame,
        ocean_lsm_maximum=float(source["land_sea_mask_maximum"]),
        expected_ocean_cells_per_hour=int(source["expected_ocean_cells_per_hour"]),
    )
    aligned, join_audit = align_mixing_features_to_oof_keys(hourly_features, deep_keys)
    finite = np.isfinite(aligned.loc[:, ERA5_MIXING_FEATURES].to_numpy(float))
    if not finite.all() or join_audit["join_coverage"] != 1.0:
        raise ValueError("ERA5 mixing feature/key coverage is incomplete")
    progress.update(82, "feature_join", "30 preregistered causal features cover every OOF key")

    access_counters = dict(seal["access_counters"])
    if any(access_counters.values()):
        raise AssertionError("dry-run forbidden access counter changed")
    actual_blockers = [
        "inner_fit_or_label_access_not_authorized",
        "outer_fit_or_label_access_not_authorized",
        "independent_modeling_qa_not_recorded",
    ]
    if policy["independent_static_modeling_qa_status"] != "passed":
        actual_blockers.append("independent_static_modeling_qa_pending")
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "generation": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "dry-run",
        "status": "passed_dry_run_actual_locked",
        "research_only": True,
        "actual_ready": False,
        "actual_blockers": actual_blockers,
        "config_sha256": _sha256(config_path),
        "blind_seal": {"path": _relative(seal_path), "sha256": _sha256(seal_path)},
        "input_files": {
            "era5_parquet": era5_file,
            "era5_manifest": era5_manifest_file,
            "deep_oof_opaque_container": deep_file,
            "physical_oof_opaque_container": physical_file,
            "public_state_implementation": public_state_file,
        },
        "policy_reconciliation": policy,
        "era5_schema": {
            "column_count": len(era5_schema),
            "source_variable_count": 8,
            "ancillary_variable_count": 1,
            "allowed_value_columns_parsed": list(ERA5_VALUE_COLUMNS),
            "unexpected_columns_parsed": 0,
        },
        "era5_quality": era5_audit,
        "units": source["units"],
        "sign_semantics": source["sign_semantics"],
        "qnet_formula": source["derived_formulas"]["qnet_native_jm2"],
        "qnet_sign_guard": source["qnet_sign_guard"],
        "oof_key_quality": {
            "deep": deep_key_audit,
            "physical": physical_key_audit,
            "exact_key_and_order_match": exact_key_order_match,
        },
        "feature_quality": {
            "feature_count": len(ERA5_MIXING_FEATURES),
            "feature_names": list(ERA5_MIXING_FEATURES),
            "hourly_feature_rows": int(len(hourly_features)),
            "hourly_feature_duplicate_key_count": int(
                hourly_features[["block", "time_utc"]].duplicated().sum()
            ),
            "aligned": join_audit,
            "all_aligned_values_finite": bool(finite.all()),
            "spatial_aggregate": source["spatial_aggregate"],
            "maximum_dependency_hours": source["alignment"][
                "maximum_feature_dependency_hours"
            ],
            "purge_hours": contract["validation"]["purge_hours"],
        },
        "control_challenger_lock": {
            "control_feature_count": len(STATE_FEATURES),
            "challenger_feature_count": len(STATE_FEATURES) + len(ERA5_MIXING_FEATURES),
            "same_two_frozen_experts": True,
            "same_model_seed_iterations_weighting": True,
            "parameter_grid_size": 0,
            "convex_only": True,
            "extrapolation_allowed": False,
        },
        "validation_locks": {
            "outer_blocks": oof["blocks"],
            "purge_hours": contract["validation"]["purge_hours"],
            "target_layer_temp_psal_joint_mask_required": True,
            "inner_gate": contract["validation"]["inner_minimum_gate"],
            "outer_promotion_gate": contract["validation"]["outer_promotion_gate"],
            "blind_seal_before_truth": True,
        },
        "access_counters": access_counters,
        "opaque_oof_container_hashing_did_not_parse_columns": True,
        "raw_or_row_values_written": False,
        "catalog_mutated": False,
        "models_fit_or_written": 0,
        "labels_parsed": 0,
        "test_files_read": 0,
        "submission_files_read_or_written": 0,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    receipt_path = output_directory / contract["outputs"]["dry_run_receipt"]
    _write_json_atomic(receipt_path, receipt)
    manifest_payload = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "actual_ready": False,
        "blind_seal_sha256": _sha256(seal_path),
        "dry_run_receipt_sha256": _sha256(receipt_path),
        "recorded_payload_file_count": 2,
        "directory_file_count_including_manifest": 3,
        "models_written": 0,
        "submissions_written": 0,
    }
    manifest_path_out = output_directory / "manifest.json"
    _write_json_atomic(manifest_path_out, manifest_payload)
    progress.update(
        100,
        "complete",
        "dry-run passed; actual remains fail-closed",
        status="completed",
        actual_ready=False,
    )
    return {
        "receipt": receipt,
        "receipt_path": _relative(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "manifest_path": _relative(manifest_path_out),
        "manifest_sha256": _sha256(manifest_path_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run",), default="dry-run")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p2_era5_mixing_gate_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()

    config_path = _repo_path(args.config)
    contract = _validate_config(config_path)
    output_directory = _repo_path(
        args.output_dir if args.output_dir is not None else contract["outputs"]["directory"]
    )
    status_path = _repo_path(
        args.status_file if args.status_file is not None else contract["outputs"]["status"]
    )
    try:
        result = run_dry_run(
            config_path=config_path,
            output_directory=output_directory,
            status_path=status_path,
        )
    except Exception as error:
        Progress(status_path).update(
            100,
            "failed",
            f"{type(error).__name__}: {error}",
            status="failed",
            actual_ready=False,
        )
        raise
    print(
        json.dumps(
            {
                "status": result["receipt"]["status"],
                "actual_ready": False,
                "receipt_path": result["receipt_path"],
                "receipt_sha256": result["receipt_sha256"],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
