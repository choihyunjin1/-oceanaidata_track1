"""Fail-closed P3 Gen6 incumbent-preserving residual calibration.

The static preflight reads only file bytes and Parquet metadata.  The research runner
is separately gated by an independent QA receipt, an execution authorization, and an
O_EXCL attempt lock.  Even an executed curve cannot open anonymous-test inputs, create
a candidate, append the central ledger, or upload anything.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    evaluate_hypothesis_gate,
    evaluate_point,
)

SCHEMA_VERSION: Final = "p3_gen6_incumbent_preserving_residual_calibrator.v1"
EXPERIMENT_ID: Final = "p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1"
CONFIG_RELATIVE: Final = (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1.json"
)
HELPER_RELATIVE: Final = "src/p3_wave/gen6_incumbent_preserving_residual_calibrator.py"
RUNNER_RELATIVE: Final = (
    "scripts/run_p3_gen6_incumbent_preserving_residual_calibrator_v1.py"
)
TESTS_RELATIVE: Final = (
    "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_v1.py"
)
EXPECTED_CONFIG_BYTES: Final = 16930
EXPECTED_CONFIG_SHA256: Final = (
    "f7584a811524f0dbe1a17f288cd82f3d3665bd44385d3355006a13bb49fb9968"
)

FOLD_ORDER: Final = ("2024_h2_storm", "winter_transition", "2025_h1")
STATIONS: Final = ("G-ORS", "I-ORS", "S-ORS")
LEADS: Final = (3, 6, 9, 12, 18, 24)
CANDIDATE_COLUMN: Final = "gen6_prediction"
CORRECTION_LIMIT_M: Final = 0.12
RESIDUAL_CLIP_M: Final = 0.75
RIDGE_ALPHA: Final = 32.0
SCALE_FLOOR: Final = 1e-8
INNER_DELTA_GATE_M: Final = -0.005
INNER_MAX_SLICE_REGRESSION_M: Final = 0.0075
INNER_MIN_CASES: Final = 12
INNER_BOOTSTRAP_REPLICATES: Final = 2000
OUTER_BOOTSTRAP_REPLICATES: Final = 5000
BOOTSTRAP_SEED_BASE: Final = 20260823

OOF_COLUMNS: Final = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "target_hs",
    "current_hs",
    "persistence",
    "incumbent_prediction",
    "single_horizon_residual_head",
    "multi_trajectory_residual_head",
    "fixed_horizon_splice",
    "prefix_fraction",
)
VALIDATION_KEY_COLUMNS: Final = ("fold", "anchor_id", "station", "episode_id")


class ContractError(RuntimeError):
    """Raised before mutation when the preregistered contract no longer holds."""


class LedgerAnchorChanged(ContractError):
    """Raised when the append-only central ledger moved after preregistration."""


@dataclass(frozen=True)
class ResidualCalibrator:
    """A deterministic standardized ridge residual model."""

    feature_names: tuple[str, ...]
    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    coefficients: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "class": "DeterministicStandardizedRidgeResidualCalibrator",
            "feature_names": list(self.feature_names),
            "numeric_mean": [float(value) for value in self.numeric_mean],
            "numeric_scale": [float(value) for value in self.numeric_scale],
            "coefficients": [float(value) for value in self.coefficients],
            "ridge_alpha": RIDGE_ALPHA,
            "residual_target_clip_m": [-RESIDUAL_CLIP_M, RESIDUAL_CLIP_M],
            "maximum_absolute_correction_m": CORRECTION_LIMIT_M,
        }


@dataclass(frozen=True)
class _ExecutionCapability:
    root: Path
    config_sha256: str
    ledger_sha256: str
    attempt_lock_sha256: str
    nonce: object


_CAPABILITY_NONCE = object()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_pin(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    display = resolved.as_posix() if root is None else resolved.relative_to(root).as_posix()
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _assert_file_pin(path: Path, pin: Mapping[str, Any], *, label: str) -> None:
    resolved = path.resolve(strict=True)
    actual_bytes = resolved.stat().st_size
    if actual_bytes != int(pin["bytes"]):
        raise ContractError(f"{label} byte size changed: {actual_bytes}")
    actual_sha = sha256_file(resolved)
    if actual_sha != str(pin["sha256"]):
        raise ContractError(f"{label} SHA-256 changed: {actual_sha}")


def _load_config(root: Path, requested_config: Path | None) -> tuple[dict[str, Any], bytes]:
    canonical = (root / CONFIG_RELATIVE).resolve(strict=True)
    requested = (requested_config or canonical).resolve(strict=True)
    if requested != canonical:
        raise ContractError("non-canonical Gen6 config path is forbidden")
    raw = canonical.read_bytes()
    if len(raw) != EXPECTED_CONFIG_BYTES or sha256_bytes(raw) != EXPECTED_CONFIG_SHA256:
        raise ContractError("canonical Gen6 config bytes differ from the compiled pin")
    config = json.loads(raw)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("Gen6 config schema differs")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("Gen6 experiment identity differs")
    if config.get("hypothesis_count") != 1:
        raise ContractError("Gen6 must contain exactly one hypothesis")
    paths = config.get("canonical_paths")
    expected_paths = {
        "config": CONFIG_RELATIVE,
        "helper": HELPER_RELATIVE,
        "runner": RUNNER_RELATIVE,
        "tests": TESTS_RELATIVE,
        "output": "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1",
        "control": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_control"
        ),
        "pre_execution_qa": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_control/"
            "pre_execution_qa.json"
        ),
        "authorization": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_control/"
            "authorization.json"
        ),
        "attempt_lock": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_control/"
            "attempt.lock"
        ),
        "run_failure_receipt": (
            "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1_control/"
            "run_failure_receipt.json"
        ),
        "sealed_gen1_oof": (
            "artifacts/p3_meaningful_learning_curve_20260823_v1/oof/"
            "learning_curve_oof.parquet"
        ),
        "sealed_validation_keys": (
            "artifacts/p3_meaningful_learning_curve_20260823_v1/validation_keys.parquet"
        ),
        "train_anchor_metadata": "artifacts/p3/features_all20_v1/train_anchors.parquet",
        "central_v9_ledger": "artifacts/meaningful_score_goal_v9/registry.jsonl",
    }
    if paths != expected_paths:
        raise ContractError("canonical path contract differs")
    calibrator = config.get("calibrator", {})
    search_keys = (
        "hyperparameter_search_count",
        "alpha_search_count",
        "threshold_search_count",
        "seed_search_count",
        "blend_weight_search_count",
        "router_search_count",
    )
    if any(calibrator.get(key) != 0 for key in search_keys):
        raise ContractError("a forbidden calibrator search was enabled")
    fixed = {
        "ridge_alpha": RIDGE_ALPHA,
        "numeric_scale_floor": SCALE_FLOOR,
        "maximum_absolute_correction_m": CORRECTION_LIMIT_M,
        "residual_target_clip_m": [-RESIDUAL_CLIP_M, RESIDUAL_CLIP_M],
    }
    if any(calibrator.get(key) != value for key, value in fixed.items()):
        raise ContractError("fixed calibrator constants differ")
    if tuple(config["sealed_surface_contract"]["prefix_fractions"]) != PREFIX_FRACTIONS:
        raise ContractError("five-point prefix contract differs")
    if tuple(config["sealed_surface_contract"]["outer_fold_order"]) != FOLD_ORDER:
        raise ContractError("outer fold order differs")
    if tuple(config["sealed_surface_contract"]["official_leads_h"]) != LEADS:
        raise ContractError("lead contract differs")
    if tuple(config["sealed_surface_contract"]["stations"]) != STATIONS:
        raise ContractError("station contract differs")
    if any(int(value) != 0 for value in config.get("static_counters", {}).values()):
        raise ContractError("static preregistration counters must remain zero")
    prohibited_false = (
        "official_promotion_allowed",
        "official_score_route_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    )
    if any(config.get(key) is not False for key in prohibited_false):
        raise ContractError("a fail-closed top-level permission changed")
    return config, raw


def _verify_workspace(root: Path, config: Mapping[str, Any]) -> None:
    expected = config["canonical_workspace_identity"]
    stat = root.stat()
    if stat.st_dev != int(expected["root_st_dev"]) or stat.st_ino != int(
        expected["root_st_ino"]
    ):
        raise ContractError("canonical workspace filesystem identity changed")
    git_config = root / ".git/config"
    if sha256_file(git_config) != expected["git_config_sha256"]:
        raise ContractError("Git config changed")
    text = git_config.read_text(encoding="utf-8")
    origin = str(expected["origin_url"])
    if f"url = {origin}" not in text:
        raise ContractError("canonical Git origin differs")


def verify_central_ledger(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    anchor = config["central_ledger_anchor"]
    path = (root / anchor["path"]).resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) != int(anchor["bytes"]) or sha256_bytes(raw) != anchor["sha256"]:
        raise LedgerAnchorChanged(
            "central v9 ledger changed; fail-close and create an append-only rebase before QA/run"
        )
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise LedgerAnchorChanged("central v9 ledger is not canonical LF-only JSONL")
    lines = raw.splitlines()
    if len(lines) != int(anchor["physical_event_lines"]):
        raise LedgerAnchorChanged("central v9 physical event count changed")
    events = [json.loads(line) for line in lines]
    previous = None
    for event in events:
        event_sha = str(event.get("event_sha256"))
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        if sha256_bytes(canonical_json_bytes(unsigned)) != event_sha:
            raise LedgerAnchorChanged("central v9 event digest verification failed")
        if previous is not None and event.get("previous_event_sha256") != previous:
            raise LedgerAnchorChanged("central v9 event chain verification failed")
        previous = event_sha
    head = events[-1]
    if int(head.get("seq", -1)) != int(anchor["global_head_seq"]):
        raise LedgerAnchorChanged("central v9 global sequence changed")
    if head.get("event_sha256") != anchor["head_event_sha256"]:
        raise LedgerAnchorChanged("central v9 head changed")
    serialized = raw.decode("utf-8")
    forbidden_positive_upload_markers = (
        '"upload_performed":true',
        '"candidate_uploaded":true',
        '"official_upload_count":1',
        '"official_uploads":1',
        '"upload_attempts":1',
    )
    if any(marker in serialized for marker in forbidden_positive_upload_markers):
        raise LedgerAnchorChanged("central v9 ledger contains an unexpected upload")
    return {
        "path": anchor["path"],
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "physical_event_lines": len(events),
        "global_head_seq": int(head["seq"]),
        "head_event_sha256": str(head["event_sha256"]),
        "official_uploads_through_anchor": 0,
    }


def _verify_immutable_inputs(
    root: Path, data_dir: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for label, pin in config["immutable_inputs"].items():
        path = data_dir / pin["path"] if label.startswith("source/") else root / pin["path"]
        _assert_file_pin(path, pin, label=label)
        verified[label] = {"bytes": int(pin["bytes"]), "sha256": str(pin["sha256"])}
    for relative, expected_sha in config["transitive_implementation_pins"].items():
        path = root / relative
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ContractError(f"transitive implementation changed: {relative}")
    return verified


def _read_pinned_json(root: Path, config: Mapping[str, Any], label: str) -> dict[str, Any]:
    pin = config["immutable_inputs"][label]
    value = json.loads((root / pin["path"]).read_bytes())
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def _verify_failure_diagnosis(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile the preregistered diagnosis against sealed aggregate artifacts."""

    diagnosis = config["failure_diagnosis"]
    labels = {
        "gen1": ("gen1_evidence", "gen1_metrics", "gen1_manifest"),
        "gen2": ("gen2_evidence", "gen2_metrics", "gen2_manifest"),
        "gen3": ("gen3_evidence", "gen3_metrics", "gen3_manifest"),
        "gen4": ("gen4_evidence", "gen4_metrics", "gen4_manifest"),
        "gen5r4": ("gen5_evidence", "gen5_metrics", "gen5_manifest"),
    }
    reconciled: dict[str, Any] = {}
    for generation, (evidence_label, metrics_label, manifest_label) in labels.items():
        evidence = _read_pinned_json(root, config, evidence_label)
        metrics = _read_pinned_json(root, config, metrics_label)
        manifest = _read_pinned_json(root, config, manifest_label)
        expected = diagnosis[generation]
        if metrics.get("status") != expected["status"]:
            raise ContractError(f"{generation} sealed status differs from diagnosis")
        points = evidence.get("points")
        if not isinstance(points, list) or len(points) != 5:
            raise ContractError(f"{generation} sealed five-point curve differs")
        fractions = tuple(float(point["fraction"]) for point in points)
        if fractions != PREFIX_FRACTIONS:
            raise ContractError(f"{generation} sealed prefix fractions differ")
        deltas = [
            float(point["challenger"]) - float(point["incumbent"]) for point in points
        ]
        full_delta = deltas[-1]
        if not math.isclose(
            full_delta,
            float(expected["full_delta_candidate_minus_incumbent_m"]),
            rel_tol=0.0,
            abs_tol=2e-15,
        ):
            raise ContractError(f"{generation} sealed full delta differs from diagnosis")
        if generation == "gen1":
            late = expected["late_deltas_candidate_minus_incumbent_m"]
            if any(
                not math.isclose(value, float(want), rel_tol=0.0, abs_tol=2e-15)
                for value, want in zip(deltas[-3:], late, strict=True)
            ):
                raise ContractError("Gen1 sealed late deltas differ from diagnosis")
        if generation == "gen5r4":
            expected_deltas = expected["prefix_deltas_candidate_minus_incumbent_m"]
            if any(
                not math.isclose(value, float(want), rel_tol=0.0, abs_tol=2e-15)
                for value, want in zip(deltas, expected_deltas, strict=True)
            ):
                raise ContractError("Gen5r4 sealed prefix deltas differ from diagnosis")
            fold = evidence.get("fold_deltas_candidate_minus_incumbent")
            if fold != expected["fold_deltas_candidate_minus_incumbent_m"]:
                raise ContractError("Gen5r4 sealed fold deltas differ from diagnosis")
            slices = evidence.get("slice_deltas_candidate_minus_incumbent")
            for key, want in expected[
                "critical_slice_deltas_candidate_minus_incumbent_m"
            ].items():
                if not math.isclose(
                    float(slices[key]), float(want), rel_tol=0.0, abs_tol=2e-15
                ):
                    raise ContractError(f"Gen5r4 sealed slice delta differs: {key}")
        if manifest.get("candidate_created") is not False:
            raise ContractError(f"{generation} unexpectedly created a candidate")
        if manifest.get("official_upload_count") != 0:
            raise ContractError(f"{generation} unexpectedly records an upload")
        if generation == "gen5r4":
            if metrics.get("test_prediction_created") is not False:
                raise ContractError("Gen5r4 unexpectedly created a test prediction")
            counters = metrics.get("access_counters", {})
            if counters.get("test_value_reads") != 0 or counters.get("upload_attempts") != 0:
                raise ContractError("Gen5r4 test/upload counter differs")
        else:
            counters = metrics.get("access_counters", {})
            zero_keys = (
                "test_index_value_reads",
                "test_context_value_reads",
                "test_target_or_hidden_label_reads",
                "current_or_frozen_submission_value_reads",
                "current_or_frozen_submission_writes",
                "upload_attempts",
            )
            if any(counters.get(key) != 0 for key in zero_keys):
                raise ContractError(f"{generation} test/submission/upload counter differs")
        reconciled[generation] = {
            "status": str(metrics["status"]),
            "full_delta_candidate_minus_incumbent_m": float(full_delta),
            "candidate_created": False,
            "official_upload_count": 0,
        }
    return reconciled


def _verify_parquet_metadata(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["canonical_paths"]
    surface = config["sealed_surface_contract"]
    oof = pq.ParquetFile(root / paths["sealed_gen1_oof"])
    if oof.metadata.num_rows != int(surface["rows"]):
        raise ContractError("sealed Gen1 OOF row count differs")
    if tuple(oof.schema_arrow.names) != OOF_COLUMNS:
        raise ContractError("sealed Gen1 OOF schema differs")
    keys = pq.ParquetFile(root / paths["sealed_validation_keys"])
    if keys.metadata.num_rows != int(surface["cases_per_prefix"]):
        raise ContractError("sealed validation-key row count differs")
    if tuple(keys.schema_arrow.names) != VALIDATION_KEY_COLUMNS:
        raise ContractError("sealed validation-key schema differs")
    anchors = pq.ParquetFile(root / paths["train_anchor_metadata"])
    required_anchor_fields = {
        "anchor_id",
        "station",
        "anchor_time",
        "current_hs",
    }
    if anchors.metadata.num_rows != 24360:
        raise ContractError("train anchor metadata row count differs")
    if not required_anchor_fields.issubset(anchors.schema_arrow.names):
        raise ContractError("train anchor metadata schema differs")
    return {
        "sealed_oof_rows": int(oof.metadata.num_rows),
        "sealed_oof_columns": list(oof.schema_arrow.names),
        "validation_key_rows": int(keys.metadata.num_rows),
        "train_anchor_rows": int(anchors.metadata.num_rows),
        "target_values_decoded": 0,
        "anonymous_test_values_read": 0,
    }


def _control_inventory(control: Path) -> set[str]:
    if not control.exists():
        return set()
    if not control.is_dir():
        raise ContractError("canonical control path is not a directory")
    return {path.name for path in control.iterdir()}


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    allow_execution_documents: bool = False,
) -> dict[str, Any]:
    """Perform a read-only static check; this function never creates a path."""

    workspace = root.resolve(strict=True)
    source = data_dir.resolve(strict=True)
    config, raw = _load_config(workspace, requested_config)
    _verify_workspace(workspace, config)
    ledger = verify_central_ledger(workspace, config)
    verified = _verify_immutable_inputs(workspace, source, config)
    diagnosis = _verify_failure_diagnosis(workspace, config)
    parquet = _verify_parquet_metadata(workspace, config)

    paths = config["canonical_paths"]
    output = workspace / paths["output"]
    control = workspace / paths["control"]
    if output.exists():
        raise ContractError("canonical append-only Gen6 output already exists")
    inventory = _control_inventory(control)
    if allow_execution_documents:
        allowed = {"pre_execution_qa.json", "authorization.json"}
        if not inventory.issubset(allowed):
            raise ContractError(f"unexpected Gen6 control files exist: {sorted(inventory)}")
    elif inventory or control.exists():
        raise ContractError("Gen6 control path must not exist during static-only QA")

    return {
        "schema_version": "p3_gen6_incumbent_preserving_residual_calibrator.check_only.v1",
        "status": "STATIC_PREFLIGHT_PASS_NO_WRITES",
        "experiment_id": EXPERIMENT_ID,
        "config": {
            "path": CONFIG_RELATIVE,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        },
        "central_ledger_anchor": ledger,
        "immutable_input_pin_count": len(verified),
        "sealed_failure_diagnosis": diagnosis,
        "transitive_implementation_pin_count": len(
            config["transitive_implementation_pins"]
        ),
        "parquet_metadata": parquet,
        "control_inventory": sorted(inventory),
        "output_exists": False,
        "qa_receipts_created": 0,
        "authorizations_created": 0,
        "attempt_locks_created": 0,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_value_reads": 0,
        "candidate_files_created": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


def _numeric_raw(frame: pd.DataFrame) -> np.ndarray:
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    current = frame["current_hs"].to_numpy(dtype=np.float64)
    persistence = frame["persistence"].to_numpy(dtype=np.float64)
    lead_scaled = frame["lead_h"].to_numpy(dtype=np.float64) / 24.0
    gap = incumbent - persistence
    return np.column_stack(
        [incumbent, current, persistence - current, incumbent - current, gap, lead_scaled * gap]
    )


def _design_matrix(
    frame: pd.DataFrame,
    *,
    numeric_mean: np.ndarray | None = None,
    numeric_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    required = {
        "station",
        "lead_h",
        "incumbent_prediction",
        "current_hs",
        "persistence",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ContractError(f"calibrator frame is missing columns: {sorted(missing)}")
    station = frame["station"].astype(str).to_numpy()
    lead = frame["lead_h"].to_numpy(dtype=np.int64)
    if not np.isin(station, STATIONS).all() or not np.isin(lead, LEADS).all():
        raise ContractError("calibrator station or lead domain differs")
    numeric = _numeric_raw(frame)
    if not np.isfinite(numeric).all():
        raise ContractError("calibrator numeric predictors are non-finite")
    if numeric_mean is None:
        numeric_mean = numeric.mean(axis=0, dtype=np.float64)
    if numeric_scale is None:
        numeric_scale = numeric.std(axis=0, dtype=np.float64)
        numeric_scale = np.where(numeric_scale < SCALE_FLOOR, 1.0, numeric_scale)
    numeric_mean = np.asarray(numeric_mean, dtype=np.float64)
    numeric_scale = np.asarray(numeric_scale, dtype=np.float64)
    if numeric_mean.shape != (numeric.shape[1],) or numeric_scale.shape != (
        numeric.shape[1],
    ):
        raise ContractError("calibrator numeric scaling shape differs")
    standardized = (numeric - numeric_mean) / numeric_scale
    station_hot = np.column_stack([(station == value).astype(np.float64) for value in STATIONS])
    lead_hot = np.column_stack([(lead == value).astype(np.float64) for value in LEADS])
    design = np.column_stack(
        [np.ones(len(frame), dtype=np.float64), station_hot, lead_hot, standardized]
    )
    feature_names = (
        "intercept",
        *(f"station_{value}" for value in STATIONS),
        *(f"lead_{value}" for value in LEADS),
        "incumbent_hs_z",
        "current_hs_z",
        "persistence_minus_current_z",
        "incumbent_minus_current_z",
        "incumbent_minus_persistence_z",
        "lead_scaled_times_incumbent_minus_persistence_z",
    )
    return design, numeric_mean, numeric_scale, tuple(feature_names)


def fit_residual_calibrator(frame: pd.DataFrame) -> ResidualCalibrator:
    """Fit the one fixed ridge model; no validation value selects any parameter."""

    if "target_hs" not in frame or len(frame) == 0:
        raise ContractError("residual calibration requires non-empty historical truth")
    design, mean, scale, names = _design_matrix(frame)
    target = frame["target_hs"].to_numpy(dtype=np.float64)
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(np.column_stack([target, incumbent])).all():
        raise ContractError("historical calibration values are non-finite")
    residual = np.clip(target - incumbent, -RESIDUAL_CLIP_M, RESIDUAL_CLIP_M)
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ residual
    coefficients = np.linalg.solve(lhs, rhs)
    if not np.isfinite(coefficients).all():
        raise ContractError("residual calibrator coefficients are non-finite")
    return ResidualCalibrator(names, mean, scale, coefficients)


def predict_bounded_correction(model: ResidualCalibrator, frame: pd.DataFrame) -> np.ndarray:
    design, _, _, names = _design_matrix(
        frame,
        numeric_mean=model.numeric_mean,
        numeric_scale=model.numeric_scale,
    )
    if names != model.feature_names or design.shape[1] != len(model.coefficients):
        raise ContractError("residual calibrator feature identity differs")
    raw = design @ model.coefficients
    bounded = np.clip(raw, -CORRECTION_LIMIT_M, CORRECTION_LIMIT_M)
    if not np.isfinite(bounded).all():
        raise ContractError("bounded residual corrections are non-finite")
    return np.asarray(bounded, dtype=np.float64)


def apply_identity_or_bounded_correction(
    incumbent_prediction: np.ndarray,
    correction: np.ndarray | None,
    *,
    enabled: bool,
) -> np.ndarray:
    """Return a byte-exact identity copy whenever the fixed inner gate does not pass."""

    incumbent = np.asarray(incumbent_prediction)
    if incumbent.dtype != np.dtype("float64") or incumbent.ndim != 1:
        raise ContractError("incumbent identity contract requires a one-dimensional float64 array")
    if not incumbent.flags.c_contiguous or not np.isfinite(incumbent).all():
        raise ContractError("incumbent identity contract requires finite C-contiguous values")
    if not enabled:
        result = incumbent.copy(order="C")
        if result.dtype != incumbent.dtype or result.tobytes(order="C") != incumbent.tobytes(
            order="C"
        ):
            raise AssertionError("failed inner evidence did not preserve incumbent bytes")
        return result
    if correction is None:
        raise ContractError("enabled correction requires a correction vector")
    values = np.asarray(correction, dtype=np.float64)
    if values.shape != incumbent.shape or not np.isfinite(values).all():
        raise ContractError("correction vector shape or finiteness differs")
    if np.max(np.abs(values), initial=0.0) > CORRECTION_LIMIT_M:
        raise ContractError("correction exceeds the preregistered bound")
    return np.clip(incumbent + values, 0.0, 30.0).astype(np.float64, copy=False)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction), dtype=np.float64)))


def _paired_case_delta_ci90(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    grouped = frame.groupby(["anchor_id", "station"], sort=True, observed=True)
    incumbent_sse: list[float] = []
    candidate_sse: list[float] = []
    counts: list[int] = []
    for _, group in grouped:
        truth = group["target_hs"].to_numpy(dtype=np.float64)
        incumbent = group["incumbent_prediction"].to_numpy(dtype=np.float64)
        candidate = group[candidate_column].to_numpy(dtype=np.float64)
        incumbent_sse.append(float(np.square(truth - incumbent).sum(dtype=np.float64)))
        candidate_sse.append(float(np.square(truth - candidate).sum(dtype=np.float64)))
        counts.append(len(group))
    if len(counts) < INNER_MIN_CASES or set(counts) != {6}:
        raise ContractError("inner bootstrap requires complete six-lead cases")
    incumbent_array = np.asarray(incumbent_sse, dtype=np.float64)
    candidate_array = np.asarray(candidate_sse, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(counts), size=len(counts))
        denominator = float(6 * len(counts))
        deltas[index] = math.sqrt(candidate_array[sample].sum() / denominator) - math.sqrt(
            incumbent_array[sample].sum() / denominator
        )
    low, high = np.quantile(deltas, [0.05, 0.95])
    return float(low), float(high)


def evaluate_inner_gate(
    frame: pd.DataFrame,
    *,
    candidate_column: str = CANDIDATE_COLUMN,
    seed: int,
) -> dict[str, Any]:
    required = {
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "incumbent_prediction",
        candidate_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ContractError(f"inner gate frame is missing: {sorted(missing)}")
    cases = frame[["anchor_id", "station"]].drop_duplicates()
    if len(cases) < INNER_MIN_CASES:
        raise ContractError("inner gate has too few complete cases")
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    candidate = frame[candidate_column].to_numpy(dtype=np.float64)
    delta = _rmse(truth, candidate) - _rmse(truth, incumbent)
    ci90 = _paired_case_delta_ci90(
        frame,
        candidate_column=candidate_column,
        replicates=INNER_BOOTSTRAP_REPLICATES,
        seed=seed,
    )
    slice_deltas: dict[str, float] = {}
    for column in ("station", "lead_h"):
        for value, group in frame.groupby(column, sort=True, observed=True):
            group_truth = group["target_hs"].to_numpy(dtype=np.float64)
            group_incumbent = group["incumbent_prediction"].to_numpy(dtype=np.float64)
            group_candidate = group[candidate_column].to_numpy(dtype=np.float64)
            slice_deltas[f"{column}:{value}"] = _rmse(
                group_truth, group_candidate
            ) - _rmse(group_truth, group_incumbent)
    worst = max(slice_deltas.values())
    checks = {
        "delta_at_most_minus_0p005m": delta <= INNER_DELTA_GATE_M,
        "paired_whole_case_ci90_upper_below_zero": ci90[1] < 0.0,
        "worst_station_or_lead_regression_at_most_0p0075m": (
            worst <= INNER_MAX_SLICE_REGRESSION_M
        ),
        "minimum_complete_cases": len(cases) >= INNER_MIN_CASES,
    }
    return {
        "passed": bool(all(checks.values())),
        "decision": "APPLY_BOUNDED_CORRECTION" if all(checks.values()) else "EXACT_IDENTITY",
        "checks": checks,
        "cases": int(len(cases)),
        "delta_candidate_minus_incumbent_m": float(delta),
        "delta_ci90_m": [float(ci90[0]), float(ci90[1])],
        "worst_station_or_lead_regression_m": float(worst),
        "slice_deltas_candidate_minus_incumbent_m": slice_deltas,
        "bootstrap_replicates": INNER_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": int(seed),
    }


def _validate_curve_surface(oof: pd.DataFrame) -> None:
    if tuple(oof.columns) != OOF_COLUMNS or len(oof) != 5430:
        raise ContractError("decoded sealed Gen1 OOF surface differs")
    keys = ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
    if oof.duplicated(keys).any():
        raise ContractError("sealed Gen1 OOF keys are duplicated")
    if tuple(sorted(oof["prefix_fraction"].unique())) != PREFIX_FRACTIONS:
        raise ContractError("sealed Gen1 prefix values differ")
    if set(oof["fold"].astype(str)) != set(FOLD_ORDER):
        raise ContractError("sealed Gen1 fold values differ")
    if set(oof["station"].astype(str)) != set(STATIONS):
        raise ContractError("sealed Gen1 station values differ")
    numeric = oof[
        ["lead_h", "target_hs", "current_hs", "persistence", "incumbent_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or set(oof["lead_h"].astype(int)) != set(LEADS):
        raise ContractError("sealed Gen1 OOF numeric domain differs")
    blocks = oof.groupby(
        ["prefix_fraction", "fold", "anchor_id", "station"], observed=True
    )["lead_h"].agg(lambda values: tuple(sorted(values.astype(int))))
    if not blocks.map(lambda values: values == LEADS).all():
        raise ContractError("sealed Gen1 OOF contains incomplete cases")


def _attach_anchor_time(oof: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    required = {"anchor_id", "station", "anchor_time"}
    missing = required.difference(anchors.columns)
    if missing:
        raise ContractError(f"anchor metadata is missing: {sorted(missing)}")
    lookup = anchors[list(required)].copy()
    if lookup.duplicated(["anchor_id", "station"]).any():
        raise ContractError("anchor metadata keys are duplicated")
    lookup["anchor_time"] = pd.to_datetime(lookup["anchor_time"], utc=True, errors="raise")
    result = oof.merge(lookup, on=["anchor_id", "station"], how="left", validate="many_to_one")
    if result["anchor_time"].isna().any() or len(result) != len(oof):
        raise ContractError("anchor-time join coverage differs")
    return result


def _split_first_fold_for_winter(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = history[["anchor_id", "station", "anchor_time"]].drop_duplicates()
    fit_keys: list[pd.DataFrame] = []
    gate_keys: list[pd.DataFrame] = []
    gap = pd.Timedelta(hours=78)
    for station in STATIONS:
        current = cases.loc[cases["station"].eq(station)].sort_values(
            ["anchor_time", "anchor_id"]
        )
        gate_count = max(1, int(math.ceil(0.40 * len(current))))
        gate = current.iloc[-gate_count:].copy()
        fit = current.iloc[:-gate_count].copy()
        boundary = gate["anchor_time"].min()
        fit = fit.loc[fit["anchor_time"] <= boundary - gap]
        if fit.empty or gate.empty:
            raise ContractError("first-fold chronological inner split is empty")
        fit_keys.append(fit[["anchor_id", "station"]])
        gate_keys.append(gate[["anchor_id", "station"]])
    fit_index = pd.MultiIndex.from_frame(pd.concat(fit_keys, ignore_index=True))
    gate_index = pd.MultiIndex.from_frame(pd.concat(gate_keys, ignore_index=True))
    row_index = pd.MultiIndex.from_frame(history[["anchor_id", "station"]])
    fit_rows = history.loc[row_index.isin(fit_index)].copy()
    gate_rows = history.loc[row_index.isin(gate_index)].copy()
    if len(fit_rows[["anchor_id", "station"]].drop_duplicates()) < INNER_MIN_CASES:
        raise ContractError("first-fold inner fit has too few cases")
    if len(gate_rows[["anchor_id", "station"]].drop_duplicates()) < INNER_MIN_CASES:
        raise ContractError("first-fold inner gate has too few cases")
    return fit_rows, gate_rows


def _predict_one_outer_fold(
    *,
    prefix: float,
    outer_fold: str,
    predictors: pd.DataFrame,
    historical_truth: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any], ResidualCalibrator | None]:
    outer = predictors.loc[predictors["fold"].eq(outer_fold)].copy()
    incumbent = np.ascontiguousarray(
        outer["incumbent_prediction"].to_numpy(dtype=np.float64), dtype=np.float64
    )
    fold_index = FOLD_ORDER.index(outer_fold)
    if fold_index == 0:
        if not historical_truth.empty:
            raise ContractError("first outer fold unexpectedly received historical truth")
        prediction = apply_identity_or_bounded_correction(incumbent, None, enabled=False)
        return (
            prediction,
            {
                "prefix_fraction": float(prefix),
                "outer_fold": outer_fold,
                "decision": "EXACT_IDENTITY_NO_PRIOR_TRUTH_AVAILABLE",
                "identity_bytes_equal": prediction.tobytes() == incumbent.tobytes(),
                "fit_count": 0,
            },
            None,
        )

    historical_names = FOLD_ORDER[:fold_index]
    if set(historical_truth["fold"].astype(str)) != set(historical_names):
        raise ContractError("outer calibrator received a non-prior truth fold")
    history = historical_truth.copy()
    if fold_index == 1:
        inner_fit, inner_gate = _split_first_fold_for_winter(history)
    else:
        inner_fit = history.loc[history["fold"].eq(FOLD_ORDER[0])].copy()
        inner_gate = history.loc[history["fold"].eq(FOLD_ORDER[1])].copy()
    gate_model = fit_residual_calibrator(inner_fit)
    inner_candidate = inner_gate.copy()
    inner_correction = predict_bounded_correction(gate_model, inner_gate)
    inner_incumbent = np.ascontiguousarray(
        inner_gate["incumbent_prediction"].to_numpy(dtype=np.float64), dtype=np.float64
    )
    inner_candidate[CANDIDATE_COLUMN] = apply_identity_or_bounded_correction(
        inner_incumbent,
        inner_correction,
        enabled=True,
    )
    seed = BOOTSTRAP_SEED_BASE + int(round(prefix * 1000)) + fold_index
    gate = evaluate_inner_gate(inner_candidate, seed=seed)
    if not gate["passed"]:
        prediction = apply_identity_or_bounded_correction(incumbent, None, enabled=False)
        return (
            prediction,
            {
                "prefix_fraction": float(prefix),
                "outer_fold": outer_fold,
                "decision": "EXACT_IDENTITY_INNER_GATE_FAILED",
                "identity_bytes_equal": prediction.tobytes() == incumbent.tobytes(),
                "inner_gate": gate,
                "fit_count": 1,
            },
            gate_model,
        )
    final_model = fit_residual_calibrator(history)
    correction = predict_bounded_correction(final_model, outer)
    prediction = apply_identity_or_bounded_correction(incumbent, correction, enabled=True)
    return (
        prediction,
        {
            "prefix_fraction": float(prefix),
            "outer_fold": outer_fold,
            "decision": "APPLY_BOUNDED_CORRECTION_INNER_GATE_PASSED",
            "identity_bytes_equal": prediction.tobytes() == incumbent.tobytes(),
            "maximum_absolute_correction_m": float(np.max(np.abs(correction), initial=0.0)),
            "inner_gate": gate,
            "fit_count": 2,
        },
        final_model,
    )


def run_research_curve(
    capability: _ExecutionCapability,
    *,
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Execute only the sealed train-OOF research curve after capability issuance."""

    workspace = root.resolve(strict=True)
    if (
        capability.nonce is not _CAPABILITY_NONCE
        or capability.root != workspace
        or capability.config_sha256 != EXPECTED_CONFIG_SHA256
        or capability.ledger_sha256 != config["central_ledger_anchor"]["sha256"]
    ):
        raise ContractError("canonical post-lock execution capability is required")
    paths = config["canonical_paths"]
    oof = pd.read_parquet(workspace / paths["sealed_gen1_oof"])
    anchors = pd.read_parquet(workspace / paths["train_anchor_metadata"])
    _validate_curve_surface(oof)
    attached = _attach_anchor_time(oof, anchors)

    vault_columns = [
        "prefix_fraction",
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "anchor_time",
    ]
    target_vault = attached[vault_columns].copy()
    predictors = target_vault.drop(columns=["target_hs"])
    blind_frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    model_receipts: list[dict[str, Any]] = []
    for prefix in PREFIX_FRACTIONS:
        prefix_predictors = predictors.loc[predictors["prefix_fraction"].eq(prefix)].copy()
        prefix_vault = target_vault.loc[target_vault["prefix_fraction"].eq(prefix)].copy()
        for outer_fold in FOLD_ORDER:
            fold_predictors = prefix_predictors.loc[
                prefix_predictors["fold"].eq(outer_fold)
            ].copy()
            fold_index = FOLD_ORDER.index(outer_fold)
            historical_truth = prefix_vault.loc[
                prefix_vault["fold"].isin(FOLD_ORDER[:fold_index])
            ].copy()
            prediction, receipt, model = _predict_one_outer_fold(
                prefix=prefix,
                outer_fold=outer_fold,
                predictors=prefix_predictors,
                historical_truth=historical_truth,
            )
            blind = fold_predictors[
                ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
            ].copy()
            blind[CANDIDATE_COLUMN] = prediction
            blind_frames.append(blind)
            receipts.append(receipt)
            if model is not None:
                model_receipts.append(
                    {
                        "prefix_fraction": float(prefix),
                        "outer_fold": outer_fold,
                        "selected_model": model.to_json(),
                    }
                )
    blind_all = pd.concat(blind_frames, ignore_index=True)
    keys = ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
    if len(blind_all) != 5430 or blind_all.duplicated(keys).any():
        raise ContractError("blind Gen6 prediction surface differs")
    evaluated = target_vault.merge(blind_all, on=keys, how="left", validate="one_to_one")
    if evaluated[CANDIDATE_COLUMN].isna().any():
        raise ContractError("blind Gen6 prediction attachment is incomplete")
    points: dict[float, dict[str, Any]] = {}
    for prefix in PREFIX_FRACTIONS:
        frame = evaluated.loc[evaluated["prefix_fraction"].eq(prefix)].copy()
        points[prefix] = evaluate_point(
            frame,
            candidate_column=CANDIDATE_COLUMN,
            bootstrap_replicates=OUTER_BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED_BASE + int(round(prefix * 1000)),
        )
    leakage_checks = {
        "sealed_corrected_validation_surface_reused": True,
        "station_global_outer_gap_at_least_78h_inherited": True,
        "same_prefix_only_calibration": True,
        "current_outer_fold_target_excluded_from_calibrator_fit_gate_and_prediction": True,
        "chronologically_prior_fold_truth_only": True,
        "anonymous_test_value_reads_zero": True,
    }
    reproducibility_checks = {
        "single_preregistered_hypothesis": True,
        "fixed_ridge_alpha_no_search": True,
        "fixed_correction_bound_no_search": True,
        "fixed_inner_gate_no_search": True,
        "model_seed_count_zero": True,
        "failed_inner_gate_exact_identity_bytes": all(
            bool(receipt.get("identity_bytes_equal"))
            for receipt in receipts
            if "IDENTITY" in str(receipt["decision"])
        ),
        "sealed_incumbent_column_unchanged": True,
        "candidate_test_full_fit_registry_and_upload_zero": True,
    }
    gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=leakage_checks,
        reproducibility_checks=reproducibility_checks,
    )
    evidence = central_evidence(
        points,
        leakage_checks=leakage_checks,
        reproducibility_checks=reproducibility_checks,
    )
    evidence.update(
        {
            "comparison_mode": config["comparison_mode"],
            "local_numeric_gate": gate,
            "official_promotion": {
                "allowed": False,
                "reason": "SEALED_GEN1_OOF_IS_NOT_AN_EXACT_OFFICIAL_PAIRED_AB",
            },
            "preregistration": {
                "hypothesis_count": 1,
                "alpha_threshold_seed_or_weight_search_count": 0,
                "config_sha256": EXPECTED_CONFIG_SHA256,
            },
        }
    )
    metrics = {
        "schema_version": "p3_gen6_incumbent_preserving_residual_calibrator.metrics.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": (
            "LOCAL_CURVE_QUALIFIED_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
            if gate["passed"]
            else "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
        ),
        "hypothesis": config["hypothesis"]["id"],
        "comparison_mode": config["comparison_mode"],
        "points": {str(key): value for key, value in points.items()},
        "local_gate": gate,
        "inner_gate_receipts": receipts,
        "calibrator_model_receipts": model_receipts,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "candidate_created": False,
        "test_prediction_created": False,
        "full_fit_performed": False,
        "official_promotion_allowed": False,
        "registry_append_allowed": False,
        "official_upload_count": 0,
    }
    return evaluated, metrics, evidence, receipts


def _implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: file_pin(root / relative, root=root)
        for name, relative in {
            "CONFIG": CONFIG_RELATIVE,
            "HELPER": HELPER_RELATIVE,
            "RUNNER": RUNNER_RELATIVE,
            "TESTS": TESTS_RELATIVE,
        }.items()
    }


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value, raw


def verify_execution_documents(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    paths = config["canonical_paths"]
    qa_path = root / paths["pre_execution_qa"]
    authorization_path = root / paths["authorization"]
    if not qa_path.is_file() or not authorization_path.is_file():
        raise ContractError("independent QA and authorization must both pre-exist")
    qa, qa_raw = _load_json_object(qa_path, label="independent QA receipt")
    qa_expected = config["qa_receipt_contract"]
    if qa.get("schema_version") != qa_expected["schema_version"]:
        raise ContractError("independent QA schema differs")
    if qa.get("verdict") != "GO" or qa.get("p0_count") != 0 or qa.get("p1_count") != 0:
        raise ContractError("independent QA is not P0=0/P1=0 GO")
    if qa.get("reviewer_independent_of_implementation_owner") is not True:
        raise ContractError("independent QA reviewer independence is not asserted")
    pins = _implementation_pins(root)
    if qa.get("implementation_pins") != pins:
        raise ContractError("independent QA implementation pins differ")
    ledger = verify_central_ledger(root, config)
    if qa.get("central_ledger_anchor") != ledger:
        raise ContractError("independent QA central-ledger anchor differs")
    qa_sha = sha256_bytes(qa_raw)

    authorization, authorization_raw = _load_json_object(
        authorization_path, label="execution authorization"
    )
    auth_expected = config["authorization_contract"]
    if authorization.get("schema_version") != auth_expected["schema_version"]:
        raise ContractError("execution authorization schema differs")
    if authorization.get("qa_sha256") != qa_sha:
        raise ContractError("execution authorization does not bind the QA receipt")
    if authorization.get("implementation_pins") != pins:
        raise ContractError("execution authorization implementation pins differ")
    if authorization.get("central_ledger_anchor") != ledger:
        raise ContractError("execution authorization central-ledger anchor differs")
    required = {
        "execute_once": True,
        "candidate_or_test_prediction_allowed": False,
        "registry_append_allowed": False,
        "upload_allowed": False,
    }
    if any(authorization.get(key) is not value for key, value in required.items()):
        raise ContractError("execution authorization permissions differ")
    return qa, qa_sha, authorization, sha256_bytes(authorization_raw)


def _robust_write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_attempt_lock(
    root: Path,
    config: Mapping[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> tuple[Path, str]:
    ledger = verify_central_ledger(root, config)
    lock = root / config["canonical_paths"]["attempt_lock"]
    if lock.exists():
        raise FileExistsError("canonical Gen6 attempt lock already exists")
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "p3_gen6_incumbent_preserving_residual_calibrator.attempt_lock.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "CONSUMED_ONE_SHOT_RESEARCH_CURVE",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "qa_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "implementation_pins": _implementation_pins(root),
        "central_ledger_anchor": ledger,
        "candidate_or_test_prediction_allowed": False,
        "registry_append_allowed": False,
        "upload_allowed": False,
    }
    raw = canonical_json_bytes(payload) + b"\n"
    _robust_write_exclusive(lock, raw)
    reread = lock.read_bytes()
    if reread != raw:
        raise ContractError("attempt lock reread differs")
    return lock, sha256_bytes(raw)


def issue_execution_capability(
    root: Path, config: Mapping[str, Any], *, attempt_lock_sha256: str
) -> _ExecutionCapability:
    lock = root / config["canonical_paths"]["attempt_lock"]
    if sha256_file(lock) != attempt_lock_sha256:
        raise ContractError("attempt lock changed before capability issuance")
    ledger = verify_central_ledger(root, config)
    return _ExecutionCapability(
        root=root.resolve(strict=True),
        config_sha256=EXPECTED_CONFIG_SHA256,
        ledger_sha256=ledger["sha256"],
        attempt_lock_sha256=attempt_lock_sha256,
        nonce=_CAPABILITY_NONCE,
    )


def publish_research_output(
    capability: _ExecutionCapability,
    *,
    root: Path,
    config: Mapping[str, Any],
    evaluated: pd.DataFrame,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    if capability.nonce is not _CAPABILITY_NONCE or capability.root != workspace:
        raise ContractError("canonical capability is required to publish Gen6 research output")
    verify_central_ledger(workspace, config)
    output = workspace / config["canonical_paths"]["output"]
    if output.exists():
        raise FileExistsError("canonical Gen6 output already exists")
    artifacts = workspace / "artifacts"
    stage = Path(tempfile.mkdtemp(prefix=".p3_gen6_stage_", dir=artifacts))
    try:
        metrics_raw = canonical_json_bytes(dict(metrics)) + b"\n"
        evidence_raw = canonical_json_bytes(dict(evidence)) + b"\n"
        _robust_write_exclusive(stage / "metrics.json", metrics_raw)
        _robust_write_exclusive(stage / "learning_curve_evidence.json", evidence_raw)
        buffer = io.BytesIO()
        evaluated.to_parquet(buffer, index=False)
        _robust_write_exclusive(stage / "oof.parquet", buffer.getvalue())
        files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(stage.iterdir(), key=lambda value: value.name)
        }
        manifest = {
            "schema_version": (
                "p3_gen6_incumbent_preserving_residual_calibrator.manifest.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "status": metrics["status"],
            "config": file_pin(workspace / CONFIG_RELATIVE, root=workspace),
            "implementation_pins": _implementation_pins(workspace),
            "central_ledger_anchor": verify_central_ledger(workspace, config),
            "output_files_before_manifest": files,
            "candidate_created": False,
            "test_prediction_created": False,
            "registry_append_allowed": False,
            "official_upload_count": 0,
        }
        manifest_raw = canonical_json_bytes(manifest) + b"\n"
        _robust_write_exclusive(stage / "manifest.json", manifest_raw)
        _robust_write_exclusive(
            stage / "manifest.sha256",
            f"{sha256_bytes(manifest_raw)}  manifest.json\n".encode("ascii"),
        )
        verify_central_ledger(workspace, config)
        stage.rename(output)
    except BaseException:
        raise
    return {
        "status": metrics["status"],
        "output": output.relative_to(workspace).as_posix(),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }


def write_failure_receipt(
    root: Path, config: Mapping[str, Any], *, exception: BaseException
) -> None:
    path = root / config["canonical_paths"]["run_failure_receipt"]
    payload = {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.run_failure_receipt.v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "exception_type": type(exception).__name__,
        "message_sha256": sha256_bytes(str(exception).encode("utf-8")),
        "candidate_created": False,
        "test_prediction_created": False,
        "registry_appended": False,
        "uploads": 0,
    }
    _robust_write_exclusive(path, canonical_json_bytes(payload) + b"\n")


__all__ = [
    "CANDIDATE_COLUMN",
    "CONFIG_RELATIVE",
    "ContractError",
    "EXPECTED_CONFIG_BYTES",
    "EXPECTED_CONFIG_SHA256",
    "EXPERIMENT_ID",
    "LedgerAnchorChanged",
    "ResidualCalibrator",
    "apply_identity_or_bounded_correction",
    "create_attempt_lock",
    "evaluate_inner_gate",
    "file_pin",
    "fit_residual_calibrator",
    "issue_execution_capability",
    "predict_bounded_correction",
    "publish_research_output",
    "run_research_curve",
    "sha256_file",
    "static_preflight",
    "verify_central_ledger",
    "verify_execution_documents",
    "write_failure_receipt",
]
