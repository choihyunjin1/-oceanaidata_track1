"""Prepare or run the separately authorized P3 forcing-analog outer v4."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from p3_wave.causal_forcing_analog import FORCING_COLUMNS, ForcingConditionedAnalogIndex
from p3_wave.causal_forcing_outer_research import (
    COMPONENT_BLIND_COLUMNS,
    FINAL_BLIND_COLUMNS,
    PAIR_KEYS,
    FoldLibraryScope,
    FrozenOOFStageVault,
    OuterResearchError,
    TrainingTargetVault,
    attach_designated_targets,
    compose_final_blind,
    evaluate_outer_gate,
    extract_native_20m_histories,
    hash_integer_array,
    read_membership_keys_only,
    sha256_file,
    validate_component_blind,
    validate_final_blind,
    validate_qa_go_receipt,
)
from p3_wave.episode_distinct_analog import LEADS, prepare_histories, project_normalized_residual
from p3_wave.revin_patch import (
    assign_storm_episodes_from_wave,
    build_episode_disjoint_folds_from_ids,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_causal_forcing_analog_outer_research_v4"
CONFIG_PATH = (ROOT / f"configs/experiments/{EXPERIMENT_ID}.json").resolve()
DRY_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/dry_run").resolve()
DRY_RECEIPT = (DRY_DIRECTORY / "receipt.json").resolve()
QA_GO_RECEIPT = (ROOT / f"artifacts/{EXPERIMENT_ID}/qa/QA_GO.json").resolve()
AUTHORIZATION = (ROOT / f"artifacts/{EXPERIMENT_ID}/authorization_amendment.json").resolve()
OUTER_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/outer_one_shot").resolve()
STATUS_PATH = (ROOT / f"artifacts/status/{EXPERIMENT_ID}.json").resolve()
ATTEMPT_LOCK = (
    ROOT / f"artifacts/experiment_locks/{EXPERIMENT_ID}.attempt.lock"
).resolve()
GLOBAL_SCORING_LOCK = (
    ROOT / f"artifacts/experiment_locks/{EXPERIMENT_ID}.outer_target.lock"
).resolve()
CANONICAL_SCORING_LOCK = (OUTER_DIRECTORY / "DESIGNATED_TARGET.lock").resolve()
GLOBAL_OUTER_LEDGER = (ROOT / "artifacts/experiment_locks/p3_outer_truth_ledger.jsonl").resolve()
PARENT_AUTHORIZATION_TOKEN = "ROOT_APPROVED_P3_CAUSAL_FORCING_ANALOG_V4"
OUTER_CONFIRMATION_TOKEN = "RUN_P3_CAUSAL_FORCING_ANALOG_V4_OUTER_ONCE"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _append_outer_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    GLOBAL_OUTER_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if GLOBAL_OUTER_LEDGER.is_file():
        existing = GLOBAL_OUTER_LEDGER.read_text(encoding="utf-8").splitlines()
        for line in existing:
            if not line.strip():
                continue
            prior = json.loads(line)
            if prior.get("experiment_id") == EXPERIMENT_ID:
                raise PermissionError("v4 already appears in the global outer ledger")
    canonical = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with GLOBAL_OUTER_LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        **dict(payload),
        "ledger_path": str(GLOBAL_OUTER_LEDGER.relative_to(ROOT)),
        "ledger_line_number": len(existing) + 1,
        "ledger_entry_sha256": __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest(),
        "ledger_sha256_after_append": sha256_file(GLOBAL_OUTER_LEDGER),
    }


def _status(
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    started: float,
    result: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 causal forcing analog outer research v4",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "upload_count": 0,
        "updated_at": _now(),
    }
    if result is not None:
        payload["result"] = dict(result)
    _atomic_json(STATUS_PATH, payload)


def _load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != CONFIG_PATH:
        raise PermissionError("config override is prohibited for v4")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise OuterResearchError("v4 experiment id changed")
    disclosure = config["adaptive_research_disclosure"]
    if (
        disclosure.get("is_adaptive_research") is not True
        or disclosure.get("outer_membership_and_labels_previously_inspected_in_prior_P3_research")
        is not True
        or disclosure.get("independent_holdout_claimed") is not False
        or disclosure.get("promotion_without_hidden_score") is not False
    ):
        raise OuterResearchError("adaptive outer disclosure changed")
    method = config["sealed_method"]
    if (
        method.get("history_points") != 145
        or method.get("history_cadence_minutes") != 20
        or method.get("history_center") != "current_hs"
        or method.get("history_scale") != "raw_MAD_floor_0.1m"
        or method.get("minimum_history_coverage") != 0.95
        or method.get("sakoe_chiba_radius_steps") != 6
        or method.get("neighbor_count") != 8
        or tuple(method.get("forcing_columns", ())) != FORCING_COLUMNS
    ):
        raise OuterResearchError("sealed v2 retrieval method changed")
    membership = config["outer_membership"]
    if (
        membership.get("allowed_pre_component_seal_columns") != list(PAIR_KEYS)
        or membership.get("expected_rows") != 1092
        or membership.get("expected_cases") != 182
    ):
        raise OuterResearchError("frozen outer membership contract changed")
    candidate = config["candidate"]
    if (
        candidate.get("no_op_leads") != [3, 6, 9]
        or candidate.get("active_leads") != [12, 18, 24]
        or candidate.get("alpha") != 0.2
    ):
        raise OuterResearchError("fixed v4 blend changed")
    gate = config["promotion_gate"]
    expected_gate = {
        "pooled_candidate_minus_incumbent_RMSE_maximum_m": -0.01,
        "paired_case_bootstrap_replicates": 5000,
        "paired_case_bootstrap_seed": 20260822,
        "paired_case_bootstrap_CI": 0.9,
        "minimum_strictly_improved_folds": 2,
        "maximum_any_station_RMSE_degradation_m": 0.01,
        "lead_18_must_not_degrade": True,
        "lead_24_must_not_degrade": True,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise OuterResearchError(f"fixed outer gate changed: {key}")
    if config["row_artifacts"]["component_blind_columns"] != list(COMPONENT_BLIND_COLUMNS):
        raise OuterResearchError("component blind columns changed")
    if config["row_artifacts"]["final_blind_columns"] != list(FINAL_BLIND_COLUMNS):
        raise OuterResearchError("final blind columns changed")
    execution = config["execution"]
    if (
        execution.get("actual_authorized") is not False
        or execution.get("parent_authorization_token") != PARENT_AUTHORIZATION_TOKEN
        or execution.get("outer_one_shot_confirmation_token") != OUTER_CONFIRMATION_TOKEN
    ):
        raise OuterResearchError("v4 authorization boundary changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise OuterResearchError("a v4 prohibition was disabled")
    return config


def _resolve_data_dir(argument: str | None) -> Path:
    raw = argument or os.environ.get("P3_DATA_DIR")
    if not raw:
        raise FileNotFoundError("provide --p3-data-dir or set P3_DATA_DIR")
    path = Path(raw).resolve()
    if not (path / "train_wave.csv").is_file() or not (path / "README.md").is_file():
        raise FileNotFoundError("P3 data directory lacks train_wave.csv or README.md")
    return path


def _implementation_hashes() -> dict[str, str]:
    relative_paths = (
        f"configs/experiments/{EXPERIMENT_ID}.json",
        "src/p3_wave/causal_forcing_outer_research.py",
        f"scripts/run_{EXPERIMENT_ID}.py",
        f"tests/test_{EXPERIMENT_ID}.py",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"v4 implementation file is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _registered_paths(config: Mapping[str, Any], data_dir: Path) -> dict[str, Path]:
    registered = config["registered_inputs"]
    result = {
        "train_wave": data_dir / registered["train_wave"]["source_name"],
        "source_readme": data_dir / registered["source_readme"]["source_name"],
    }
    for role, details in registered.items():
        if role not in result:
            result[role] = ROOT / details["path"]
    return result


def _verify_registered_inputs(config: Mapping[str, Any], data_dir: Path) -> dict[str, str]:
    registered = config["registered_inputs"]
    receipts: dict[str, str] = {}
    for role, path in _registered_paths(config, data_dir).items():
        if not path.is_file():
            raise FileNotFoundError(f"registered v4 input is missing: {role}")
        actual = sha256_file(path)
        if actual != registered[role]["sha256"]:
            raise OuterResearchError(f"registered v4 SHA changed: {role}")
        receipts[role] = actual
    return receipts


def _schema_and_key_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = config["registered_inputs"]
    oof_path = ROOT / registered["incumbent_oof"]["path"]
    anchor_path = ROOT / registered["anchor_metadata_and_target_vault"]["path"]
    feature_path = ROOT / registered["train_features"]["path"]
    oof_schema = parquet.read_schema(oof_path)
    anchor_schema = parquet.read_schema(anchor_path)
    feature_schema = parquet.read_schema(feature_path)
    required_oof = {*PAIR_KEYS, "prediction", "target_hs"}
    required_anchor = {
        "anchor_id",
        "station",
        "anchor_time",
        "grid_position",
        "current_hs",
        *[f"target_{lead}" for lead in LEADS],
    }
    required_feature = {"anchor_id", "station", *FORCING_COLUMNS}
    if not required_oof.issubset(oof_schema.names):
        raise OuterResearchError("frozen incumbent OOF schema changed")
    if not required_anchor.issubset(anchor_schema.names):
        raise OuterResearchError("anchor target vault schema changed")
    if not required_feature.issubset(feature_schema.names):
        raise OuterResearchError("forcing feature cache schema changed")
    keys, _, key_audit = read_membership_keys_only(oof_path, config["outer_membership"])
    v2_result = json.loads(
        (ROOT / registered["v2_result"]["path"]).read_text(encoding="utf-8")
    )
    v3_result = json.loads(
        (ROOT / registered["v3_result"]["path"]).read_text(encoding="utf-8")
    )
    if v2_result.get("decision") != "PASS_B_ADAPTIVE_INNER_ONLY_STOP":
        raise OuterResearchError("sealed v2 B decision changed")
    if v3_result.get("decision") != "PASS_C_ADAPTIVE_INNER_STOP_BEFORE_OUTER":
        raise OuterResearchError("sealed v3 C decision changed")
    return {
        "oof_schema_columns": int(len(oof_schema.names)),
        "anchor_schema_columns": int(len(anchor_schema.names)),
        "feature_schema_columns": int(len(feature_schema.names)),
        "membership": key_audit,
        "key_rows_materialized": int(len(keys)),
        "OOF_columns_materialized": list(PAIR_KEYS),
        "incumbent_prediction_values_read": 0,
        "designated_target_values_read": 0,
        "anchor_target_values_read": 0,
        "model_fit_count": 0,
        "v2_decision": v2_result["decision"],
        "v3_decision": v3_result["decision"],
    }


def _synthetic_contract() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fold_station = (
        ("2024_h2_storm", "G-ORS"),
        ("winter_transition", "I-ORS"),
        ("2025_h1", "S-ORS"),
    )
    for anchor_id, (fold, station) in enumerate(fold_station):
        for lead in LEADS:
            rows.append(
                {
                    "fold": fold,
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "history_eligible": True,
                    "conditioning_used": True,
                    "fallback_reason": "",
                    "query_mad_scale": 0.2,
                    "neighbor_anchor_ids_sha256": "a" * 64,
                    "neighbor_episode_ids_sha256": "b" * 64,
                    "neighbor_distance_mean": 0.5,
                    "neighbor_distance_max": 1.0,
                    "analog_prediction": 1.0,
                }
            )
    component = pd.DataFrame(rows, columns=COMPONENT_BLIND_COLUMNS)
    keys = component.loc[:, PAIR_KEYS].copy()
    validate_component_blind(component, keys)
    incumbent = keys.copy()
    incumbent["incumbent_final"] = 2.0
    final = compose_final_blind(component, incumbent)
    validate_final_blind(final, keys)
    targets = keys.copy()
    targets["target_hs"] = 1.0
    evaluated = attach_designated_targets(final, targets)
    gate = evaluate_outer_gate(evaluated)
    if not gate["pass"]:
        raise AssertionError("synthetic v4 outer gate contract failed")
    return {
        "rows": int(len(final)),
        "cases": 3,
        "component_has_incumbent_or_target": False,
        "final_has_target": False,
        "short_lead_exact_no_op": True,
        "synthetic_gate_pass": True,
        "synthetic_values_are_not_competition_labels": True,
        "model_fit_count": 0,
        "incumbent_prediction_values_read": 0,
        "designated_target_values_read": 0,
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = True
    return {"git_sha": commit, "git_dirty": dirty}


def _dry_run(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    if any(path.exists() for path in (DRY_DIRECTORY, AUTHORIZATION, ATTEMPT_LOCK, OUTER_DIRECTORY)):
        raise PermissionError("v4 dry/authorization/actual state already exists")
    _status(
        state="dry_running",
        phase="hash_schema_key_only_synthetic_preflight",
        progress=20.0,
        detail="key 4열·봉인 SHA·합성 gate 확인; incumbent/target/model/test/submission 0",
        started=started,
    )
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": "dry-run",
        "status": "READY_PENDING_QA_AND_PARENT_AUTHORIZATION",
        "created_at": _now(),
        "adaptive_research": True,
        "independent_holdout": False,
        "registered_input_sha256": _verify_registered_inputs(config, data_dir),
        "implementation_sha256": _implementation_hashes(),
        "schema_and_key_preflight": _schema_and_key_preflight(config),
        "synthetic_contract": _synthetic_contract(),
        "environment": _environment(),
        **_git_provenance(),
        "outer_key_membership_read_count": 1,
        "outer_model_execution_count": 0,
        "training_target_read_count": 0,
        "incumbent_prediction_read_count": 0,
        "designated_target_read_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "upload_count": 0,
    }
    DRY_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _atomic_json(DRY_RECEIPT, receipt)
    receipt_sha = sha256_file(DRY_RECEIPT)
    _status(
        state="dry_ready_waiting_QA_and_parent",
        phase="authorization_boundary_closed",
        progress=100.0,
        detail="dry-run 완료; actual outer/incumbent/target/model/test/submission 0",
        started=started,
        result={"dry_receipt_sha256": receipt_sha},
    )
    print(json.dumps({**receipt, "dry_receipt_sha256": receipt_sha}, ensure_ascii=False, indent=2))
    return 0


def _authorize(
    *,
    qa_receipt_path: str | None,
    parent_token: str | None,
    started: float,
) -> int:
    if parent_token != PARENT_AUTHORIZATION_TOKEN:
        raise PermissionError("authorize mode requires the exact separate parent token")
    if qa_receipt_path is None or Path(qa_receipt_path).resolve() != QA_GO_RECEIPT:
        raise PermissionError("authorize mode requires the canonical QA GO receipt")
    if not DRY_RECEIPT.is_file() or not QA_GO_RECEIPT.is_file():
        raise PermissionError("dry receipt and independent QA GO receipt are both required")
    if any(path.exists() for path in (AUTHORIZATION, ATTEMPT_LOCK, OUTER_DIRECTORY)):
        raise PermissionError("v4 authorization or actual state already exists")
    dry = json.loads(DRY_RECEIPT.read_text(encoding="utf-8"))
    implementation = _implementation_hashes()
    if dry.get("status") != "READY_PENDING_QA_AND_PARENT_AUTHORIZATION":
        raise PermissionError("v4 dry receipt is not authorization-ready")
    if dry.get("implementation_sha256") != implementation:
        raise PermissionError("v4 implementation changed after dry-run")
    qa = json.loads(QA_GO_RECEIPT.read_text(encoding="utf-8"))
    validate_qa_go_receipt(
        qa,
        experiment_id=EXPERIMENT_ID,
        dry_receipt_sha256=sha256_file(DRY_RECEIPT),
        implementation_sha256=implementation,
    )
    amendment = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "authorized": True,
        "authorized_at": _now(),
        "dry_receipt_sha256": sha256_file(DRY_RECEIPT),
        "QA_GO_receipt_sha256": sha256_file(QA_GO_RECEIPT),
        "implementation_sha256": implementation,
        "registered_input_sha256": dry["registered_input_sha256"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "outer_execution_count": 0,
        "incumbent_prediction_read_count": 0,
        "designated_target_read_count": 0,
    }
    _write_exclusive(AUTHORIZATION, amendment)
    _status(
        state="authorized_waiting_explicit_outer_confirmation",
        phase="QA_and_parent_authorization_sealed",
        progress=100.0,
        detail="QA GO+parent 승인 봉인; actual outer/incumbent/target/model 0",
        started=started,
        result={"authorization_sha256": sha256_file(AUTHORIZATION)},
    )
    return 0


def _load_authorization() -> dict[str, Any]:
    if not AUTHORIZATION.is_file():
        raise PermissionError("canonical QA+parent authorization amendment is missing")
    amendment = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if amendment.get("authorized") is not True or amendment.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("v4 authorization amendment is invalid")
    if amendment.get("dry_receipt_sha256") != sha256_file(DRY_RECEIPT):
        raise PermissionError("v4 dry receipt changed after authorization")
    if amendment.get("QA_GO_receipt_sha256") != sha256_file(QA_GO_RECEIPT):
        raise PermissionError("v4 QA GO receipt changed after authorization")
    if amendment.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("v4 implementation changed after authorization")
    if amendment.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise PermissionError("v4 config changed after authorization")
    return amendment


def _read_wave(data_dir: Path) -> pd.DataFrame:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    if not {"station", "time", "hs"}.issubset(wave):
        raise OuterResearchError("train_wave schema changed")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    if wave.duplicated(["station", "time"]).any():
        raise OuterResearchError("train_wave keys are duplicated")
    return wave


def _load_label_free_anchor_surface(
    config: Mapping[str, Any], wave: pd.DataFrame
) -> pd.DataFrame:
    path = ROOT / config["registered_inputs"]["anchor_metadata_and_target_vault"]["path"]
    columns = ["anchor_id", "station", "anchor_time", "grid_position", "current_hs"]
    anchors = pd.read_parquet(path, columns=columns)
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    if (
        len(anchors) != 24360
        or anchors["anchor_id"].duplicated().any()
        or not np.array_equal(
            anchors["anchor_id"].to_numpy(dtype=np.int64), np.arange(len(anchors))
        )
    ):
        raise OuterResearchError("label-free anchor surface changed")
    return assign_storm_episodes_from_wave(anchors, wave)


def _load_forcing(config: Mapping[str, Any], anchors: pd.DataFrame) -> np.ndarray:
    path = ROOT / config["registered_inputs"]["train_features"]["path"]
    frame = pd.read_parquet(path, columns=["anchor_id", "station", *FORCING_COLUMNS])
    if (
        len(frame) != len(anchors)
        or frame["anchor_id"].duplicated().any()
        or not np.array_equal(
            frame["anchor_id"].to_numpy(dtype=np.int64),
            anchors["anchor_id"].to_numpy(dtype=np.int64),
        )
        or not frame["station"].astype(str).equals(anchors["station"].astype(str))
    ):
        raise OuterResearchError("forcing feature surface differs from anchors")
    return frame.loc[:, FORCING_COLUMNS].to_numpy(dtype=np.float64)


def _folds_from_membership(
    anchors: pd.DataFrame,
    membership: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[Any, ...]:
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["folds"]["windows"],
        validation_ids_by_fold=membership,
        embargo_hours=78,
    )
    if len(folds) != 3:
        raise OuterResearchError("outer fold count changed")
    lookup = anchors.set_index("anchor_id")
    for fold in folds:
        cutoff = fold.validation_start - pd.Timedelta(hours=78)
        library = lookup.loc[fold.train_ids]
        if not library["anchor_time"].lt(cutoff).all():
            raise OuterResearchError(f"{fold.name} library crosses the 78h cutoff")
        validation = lookup.loc[fold.validation_ids]
        shared = set(
            zip(library["station"], library["episode_id"], strict=True)
        ).intersection(zip(validation["station"], validation["episode_id"], strict=True))
        if shared:
            raise OuterResearchError(f"{fold.name} library shares a validation episode")
    return folds


def _analog_component_for_fold(
    *,
    fold: Any,
    anchors: pd.DataFrame,
    prepared: Any,
    forcing: np.ndarray,
    library_targets: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = anchors.set_index("anchor_id")
    library_ids = np.asarray(fold.train_ids, dtype=np.int64)
    target_by_id = {
        int(anchor_id): library_targets[row]
        for row, anchor_id in enumerate(library_ids)
    }
    indices: dict[str, tuple[ForcingConditionedAnalogIndex, np.ndarray]] = {}
    for station in sorted(anchors["station"].astype(str).unique()):
        station_ids = library_ids[
            lookup.loc[library_ids, "station"].astype(str).to_numpy() == station
        ]
        station_ids = station_ids[prepared.eligible[station_ids]]
        indices[station] = (
            ForcingConditionedAnalogIndex(
                anchor_ids=station_ids,
                episode_ids=lookup.loc[station_ids, "episode_id"].to_numpy(dtype=np.int64),
                normalized_histories=prepared.normalized[station_ids],
                forcing_state=forcing[station_ids],
                radius_steps=6,
                neighbor_count=8,
                batch_size=1024,
            ),
            station_ids,
        )
    rows: list[dict[str, Any]] = []
    conditioned_cases = 0
    eligible_cases = 0
    for anchor_value in fold.validation_ids:
        anchor_id = int(anchor_value)
        station = str(lookup.loc[anchor_id, "station"])
        current = float(lookup.loc[anchor_id, "current_hs"])
        if not prepared.eligible[anchor_id]:
            analog = np.full(len(LEADS), np.nan)
            fields = {
                "history_eligible": False,
                "conditioning_used": False,
                "fallback_reason": "history_ineligible",
                "query_mad_scale": np.nan,
                "neighbor_anchor_ids_sha256": hash_integer_array([]),
                "neighbor_episode_ids_sha256": hash_integer_array([]),
                "neighbor_distance_mean": np.nan,
                "neighbor_distance_max": np.nan,
            }
        else:
            eligible_cases += 1
            index, station_ids = indices[station]
            selected = index.select_nearest(prepared.normalized[anchor_id], forcing[anchor_id])
            neighbors = selected.neighbors
            neighbor_ids = station_ids[neighbors.indices]
            targets = np.vstack([target_by_id[int(item)] for item in neighbor_ids])
            residual = (
                targets
                - lookup.loc[neighbor_ids, "current_hs"].to_numpy(dtype=np.float64)[:, None]
            ) / prepared.scale[neighbor_ids, None]
            if not np.isfinite(residual).all():
                raise OuterResearchError("analog library target release is incomplete")
            projected = project_normalized_residual(
                residual, neighbors.distances, distance_floor=1e-6
            )
            analog = np.clip(current + prepared.scale[anchor_id] * projected, 0.0, 30.0)
            conditioned_cases += int(selected.conditioning_used)
            fields = {
                "history_eligible": True,
                "conditioning_used": bool(selected.conditioning_used),
                "fallback_reason": "" if selected.fallback_reason is None else selected.fallback_reason,
                "query_mad_scale": float(prepared.scale[anchor_id]),
                "neighbor_anchor_ids_sha256": hash_integer_array(neighbor_ids),
                "neighbor_episode_ids_sha256": hash_integer_array(neighbors.episodes),
                "neighbor_distance_mean": float(neighbors.distances.mean()),
                "neighbor_distance_max": float(neighbors.distances.max()),
            }
        for column, lead in enumerate(LEADS):
            rows.append(
                {
                    "fold": fold.name,
                    "anchor_id": anchor_id,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    **fields,
                    "analog_prediction": float(analog[column]) if np.isfinite(analog[column]) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=COMPONENT_BLIND_COLUMNS), {
        "library_anchors": int(len(library_ids)),
        "library_episodes": int(
            lookup.loc[library_ids].groupby(["station", "episode_id"], observed=True).ngroups
        ),
        "validation_cases": int(len(fold.validation_ids)),
        "history_eligible_cases": int(eligible_cases),
        "forcing_conditioned_cases": int(conditioned_cases),
        "validation_start": fold.validation_start.isoformat(),
        "library_cutoff": (fold.validation_start - pd.Timedelta(hours=78)).isoformat(),
        "same_station": True,
        "episode_distinct": True,
    }


def _assert_roundtrip(expected: pd.DataFrame, path: Path) -> pd.DataFrame:
    actual = pd.read_parquet(path)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    except AssertionError as error:
        raise OuterResearchError(f"fsynced parquet changed after reload: {path.name}") from error
    return actual


def _output_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _outer_one_shot(
    config: Mapping[str, Any],
    data_dir: Path,
    *,
    confirmation: str | None,
    started: float,
) -> int:
    canonical_config = _load_config(CONFIG_PATH)
    if dict(config) != canonical_config:
        raise PermissionError("passed v4 config differs from canonical config")
    config = canonical_config
    if confirmation != OUTER_CONFIRMATION_TOKEN:
        raise PermissionError("outer-one-shot requires the exact confirmation token")
    authorization = _load_authorization()
    if any(
        path.exists()
        for path in (ATTEMPT_LOCK, OUTER_DIRECTORY, GLOBAL_SCORING_LOCK)
    ):
        raise PermissionError("v4 outer one-shot was already attempted")
    receipts = _verify_registered_inputs(config, data_dir)
    if receipts != authorization["registered_input_sha256"]:
        raise PermissionError("registered inputs differ from the authorized dry-run")
    implementation = _implementation_hashes()
    attempt = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "adaptive_outer_research_no_test_submission_or_promotion",
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "implementation_sha256": implementation,
        "designated_target_read_count": 0,
    }
    _write_exclusive(ATTEMPT_LOCK, attempt)
    OUTER_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _status(
        state="outer_running_pre_component_seal",
        phase="key_only_membership_and_causal_component",
        progress=5.0,
        detail="OOF key 4열만 개방; incumbent prediction/target 0",
        started=started,
    )
    oof_path = ROOT / config["registered_inputs"]["incumbent_oof"]["path"]
    keys, membership, key_audit = read_membership_keys_only(
        oof_path, config["outer_membership"]
    )
    wave = _read_wave(data_dir)
    anchors = _load_label_free_anchor_surface(config, wave)
    histories = extract_native_20m_histories(wave, anchors)
    prepared = prepare_histories(histories, minimum_coverage=0.95, mad_floor=0.1)
    forcing = _load_forcing(config, anchors)
    folds = _folds_from_membership(anchors, membership, config)
    scopes = tuple(
        FoldLibraryScope(
            name=fold.name,
            library_ids=np.asarray(fold.train_ids, dtype=np.int64),
            validation_ids=np.asarray(fold.validation_ids, dtype=np.int64),
        )
        for fold in folds
    )
    training_vault = TrainingTargetVault(
        ROOT / config["registered_inputs"]["anchor_metadata_and_target_vault"]["path"],
        scopes,
    )
    component_parts: list[pd.DataFrame] = []
    fold_audit: dict[str, Any] = {}
    for fold_index, fold in enumerate(folds):
        library_targets = training_vault.read_library(fold.name, fold.train_ids)
        part, audit = _analog_component_for_fold(
            fold=fold,
            anchors=anchors,
            prepared=prepared,
            forcing=forcing,
            library_targets=library_targets,
        )
        component_parts.append(part)
        fold_audit[fold.name] = audit
        _status(
            state="outer_running_pre_component_seal",
            phase="causal_component_prediction",
            progress=15.0 + 15.0 * (fold_index + 1),
            detail=f"{fold.name} analog 완료; incumbent prediction/target 0",
            started=started,
        )
    component = (
        pd.concat(component_parts, ignore_index=True)
        .sort_values(list(PAIR_KEYS))
        .reset_index(drop=True)
    )
    component_audit = validate_component_blind(component, keys)
    component_path = OUTER_DIRECTORY / "analog_component_blind.parquet"
    _atomic_parquet(component_path, component)
    component_reloaded = _assert_roundtrip(component, component_path)
    validate_component_blind(component_reloaded, keys)
    component_seal = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "sealed": True,
        "stage": "analog_component_before_incumbent_prediction",
        "created_at": _now(),
        "component_blind_sha256": sha256_file(component_path),
        "membership_key_audit": key_audit,
        "component_audit": component_audit,
        "fold_audit": fold_audit,
        "training_target_access_log": training_vault.access_log,
        "current_or_future_outer_target_overlap_count": 0,
        "incumbent_prediction_read_count": 0,
        "designated_target_read_count": 0,
        "implementation_sha256": implementation,
        "registered_input_sha256": receipts,
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(ATTEMPT_LOCK),
    }
    component_seal_path = OUTER_DIRECTORY / "component_blind_seal.json"
    _atomic_json(component_seal_path, component_seal)
    stage_vault = FrozenOOFStageVault(oof_path, keys)
    stage_vault.register_component_seal(component_seal_path, component_path)
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed before incumbent prediction read")
    if _verify_registered_inputs(config, data_dir) != receipts:
        raise PermissionError("registered input changed before incumbent prediction read")
    incumbent = stage_vault.read_incumbent_once()
    final = compose_final_blind(component_reloaded, incumbent, alpha=0.2)
    final_audit = validate_final_blind(final, keys)
    final_path = OUTER_DIRECTORY / "final_blind_predictions.parquet"
    _atomic_parquet(final_path, final)
    final_reloaded = _assert_roundtrip(final, final_path)
    validate_final_blind(final_reloaded, keys)
    final_seal = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "sealed": True,
        "stage": "all_final_blind_predictions_before_designated_target",
        "created_at": _now(),
        "final_blind_sha256": sha256_file(final_path),
        "component_seal_sha256": sha256_file(component_seal_path),
        "final_audit": final_audit,
        "incumbent_prediction_access_log": stage_vault.access_log,
        "incumbent_prediction_read_count": 1,
        "designated_target_read_count": 0,
        "implementation_sha256": implementation,
        "registered_input_sha256": receipts,
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(ATTEMPT_LOCK),
    }
    final_seal_path = OUTER_DIRECTORY / "final_blind_seal.json"
    _atomic_json(final_seal_path, final_seal)
    stage_vault.register_final_seal(final_seal_path, final_path)
    _status(
        state="outer_blind_sealed_pre_target",
        phase="all_blind_predictions_fsynced_reloaded_and_SHA_sealed",
        progress=82.0,
        detail="final blind seal 완료; designated target 0",
        started=started,
    )
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed after final blind seal")
    if _verify_registered_inputs(config, data_dir) != receipts:
        raise PermissionError("registered input changed after final blind seal")
    scoring_payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "purpose": "one_shot_designated_outer_target_open_after_final_blind_seal",
        "final_seal_sha256": sha256_file(final_seal_path),
        "final_blind_sha256": sha256_file(final_path),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(ATTEMPT_LOCK),
        "designated_target_read_count_before": 0,
    }
    _write_exclusive(CANONICAL_SCORING_LOCK, scoring_payload)
    _write_exclusive(GLOBAL_SCORING_LOCK, scoring_payload)
    ledger_receipt = _append_outer_ledger(scoring_payload)
    target = stage_vault.read_designated_target_once(
        scoring_lock_paths=[CANONICAL_SCORING_LOCK, GLOBAL_SCORING_LOCK],
        ledger_receipt=ledger_receipt,
    )
    evaluated = attach_designated_targets(final_reloaded, target)
    evaluated_path = OUTER_DIRECTORY / "evaluated_outer_predictions.parquet"
    _atomic_parquet(evaluated_path, evaluated)
    evaluated_reloaded = _assert_roundtrip(evaluated, evaluated_path)
    gate = evaluate_outer_gate(evaluated_reloaded)
    expected_incumbent = float(
        config["registered_inputs"]["incumbent_oof"]["expected_incumbent_RMSE_m"]
    )
    if not np.isclose(gate["incumbent_rmse_m"], expected_incumbent, rtol=0.0, atol=1e-12):
        raise OuterResearchError("exact incumbent RMSE did not reconcile")
    metrics_path = OUTER_DIRECTORY / "metrics.json"
    access_path = OUTER_DIRECTORY / "access_log.json"
    _atomic_json(metrics_path, gate)
    _atomic_json(
        access_path,
        {
            "training_target_access_log": training_vault.access_log,
            "staged_OOF_access_log": stage_vault.access_log,
            "current_or_future_outer_target_overlap_count": 0,
            "incumbent_prediction_read_count": stage_vault.incumbent_prediction_read_count,
            "designated_target_read_count": stage_vault.designated_target_read_count,
            "ledger_receipt": ledger_receipt,
        },
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "decision": gate["decision"],
        "outer_gate": gate,
        "adaptive_research": True,
        "independent_holdout": False,
        "promotion_performed": False,
        "required_action": (
            "retain_as_research_candidate_and_require_separate_hidden_score_authorization"
            if gate["pass"]
            else "permanent_stop_keep_frozen_incumbent"
        ),
        "outer_key_membership_read_count": 1,
        "model_fit_count": 0,
        "incumbent_prediction_read_count": 1,
        "designated_target_read_count": 1,
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "upload_count": 0,
        "rerun_prohibited": True,
    }
    _atomic_json(OUTER_DIRECTORY / "result.json", result)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "adaptive_outer_research_no_test_submission_or_promotion",
        "decision": gate["decision"],
        "registered_input_sha256": receipts,
        "implementation_sha256": implementation,
        "dry_receipt_sha256": sha256_file(DRY_RECEIPT),
        "QA_GO_receipt_sha256": sha256_file(QA_GO_RECEIPT),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(ATTEMPT_LOCK),
        "component_seal_sha256": sha256_file(component_seal_path),
        "final_seal_sha256": sha256_file(final_seal_path),
        "canonical_scoring_lock_sha256": sha256_file(CANONICAL_SCORING_LOCK),
        "global_scoring_lock_sha256": sha256_file(GLOBAL_SCORING_LOCK),
        "ledger_receipt": ledger_receipt,
        "output_sha256": _output_hashes(OUTER_DIRECTORY),
        "environment": _environment(),
        **_git_provenance(),
        "adaptive_research": True,
        "independent_holdout": False,
        "promotion_performed": False,
        "model_fit_count": 0,
        "incumbent_prediction_read_count": 1,
        "designated_target_read_count": 1,
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "upload_count": 0,
    }
    _atomic_json(OUTER_DIRECTORY / "manifest.json", manifest)
    _status(
        state="outer_complete_no_promotion",
        phase="one_shot_designated_outer_research_complete",
        progress=100.0,
        detail=f"{gate['decision']}; promotion/test/submission/upload 0",
        started=started,
        result=result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "authorize", "outer-one-shot"), default="dry-run")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--p3-data-dir")
    parser.add_argument("--qa-go-receipt")
    parser.add_argument("--parent-approval-token")
    parser.add_argument("--confirm")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    started = time.perf_counter()
    config = _load_config(arguments.config)
    if arguments.mode == "authorize":
        if arguments.p3_data_dir is not None or arguments.confirm is not None:
            raise PermissionError("authorize mode does not accept data or execution confirmation")
        return _authorize(
            qa_receipt_path=arguments.qa_go_receipt,
            parent_token=arguments.parent_approval_token,
            started=started,
        )
    data_dir = _resolve_data_dir(arguments.p3_data_dir)
    if arguments.mode == "dry-run":
        if any(
            value is not None
            for value in (
                arguments.qa_go_receipt,
                arguments.parent_approval_token,
                arguments.confirm,
            )
        ):
            raise PermissionError("dry-run does not accept authorization or execution tokens")
        return _dry_run(config, data_dir, started)
    if arguments.qa_go_receipt is not None or arguments.parent_approval_token is not None:
        raise PermissionError("outer-one-shot consumes only the sealed authorization amendment")
    return _outer_one_shot(
        config,
        data_dir,
        confirmation=arguments.confirm,
        started=started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
