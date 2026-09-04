"""Run the preregistered point-only external I-ORS P1 OOF experiment once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
QUARANTINE_DIR = PROJECT_ROOT / "external_data" / "quarantine"
OPTIONAL_DEPS = QUARANTINE_DIR / "_deps"
for import_path in (SRC_DIR, OPTIONAL_DEPS):
    if import_path.is_dir() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ocean_external.iors_ctd import (  # noqa: E402
    build_loo_dataset,
    ensure_archive,
    load_json_object,
    read_year_profile,
    validate_source_manifest,
    verify_archive,
    verify_official_record,
)
from ocean_external.iors_precheck import dataset_audit  # noqa: E402
from p1_qc.config import P1QCConfig, load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.experiment import environment_summary, sha256_file  # noqa: E402
from p1_qc.iors_external_point_residual import (  # noqa: E402
    KEY_COLUMNS,
    POINT_RESIDUAL_COLUMNS,
    append_point_residual_matrix,
    apply_point_residual_gate,
    build_p1_iors_panel,
    build_point_residual_features,
    canonical_artifact_paths,
    compose_incumbent_predictions,
    independent_expected_replacement_keys,
    predict_external_q50,
    select_inner_threshold,
)
from p1_qc.metrics import evaluate_predictions, group_row_shares  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    TabularEncoder,
    _best_iteration,
    _fit_model,
    _threads,
    load_or_build_features,
    resolve_data_dir,
)
from p1_qc.rules import detect_plateaus  # noqa: E402
from p1_qc.validation import (  # noqa: E402
    normal_station_layer_day_fp,
    paired_block_bootstrap,
)

EXPERIMENT_ID = "p1_iors_external_point_residual_oof_v1"
BACKEND = "xgboost"
EXTERNAL_INPUT_COLUMNS = (*KEY_COLUMNS, "temp", "psal", "depth")
CANONICAL_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs/experiments/p1_iors_external_point_residual_oof_v1.json"
).resolve()
CANONICAL_OUTPUT_DIR = (PROJECT_ROOT / "artifacts/p1_iors_external_point_residual_oof_v1").resolve()
CANONICAL_STATUS_FILE = (
    PROJECT_ROOT / "artifacts/status/p1_iors_external_point_residual_oof_v1.json"
).resolve()
CANONICAL_OUTER_LOCK = (CANONICAL_OUTPUT_DIR / "outer_exposure.lock").resolve()
GLOBAL_EXPOSURE_LOCK = (PROJECT_ROOT / f"artifacts/one_shot_locks/{EXPERIMENT_ID}.lock").resolve()
GLOBAL_EXPOSURE_LEDGER = (PROJECT_ROOT / "artifacts/one_shot_exposure_ledger.jsonl").resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/p1_iors_external_point_residual_oof_v1.json"),
    )
    parser.add_argument("--p1-config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--quarantine-dir", type=Path, default=Path("external_data/quarantine"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1_iors_external_point_residual_oof_v1"),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("artifacts/status/p1_iors_external_point_residual_oof_v1.json"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-live-record-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_hardcoded_canonical_paths(args: argparse.Namespace) -> None:
    checks = {
        "--experiment-config": (_resolve(args.experiment_config), CANONICAL_EXPERIMENT_CONFIG),
        "--output-dir": (_resolve(args.output_dir), CANONICAL_OUTPUT_DIR),
        "--status-file": (_resolve(args.status_file), CANONICAL_STATUS_FILE),
    }
    mismatches = {
        name: {"requested": str(requested), "required": str(required)}
        for name, (requested, required) in checks.items()
        if requested != required
    }
    if mismatches:
        raise ValueError(f"hardcoded one-shot canonical path mismatch: {mismatches}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _write_parquet_fsync(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _exclusive_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _json_bytes(value)
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exposure_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid exposure ledger row {line_number}")
        rows.append(value)
    return rows


def _assert_not_globally_exposed() -> None:
    if GLOBAL_EXPOSURE_LOCK.exists():
        raise FileExistsError("hardcoded experiment-wide exposure lock already exists")
    rows = _read_exposure_ledger(GLOBAL_EXPOSURE_LEDGER)
    if any(value.get("experiment_id") == EXPERIMENT_ID for value in rows):
        raise FileExistsError("append-only exposure ledger already contains this experiment_id")


def _append_exposure_ledger(value: Mapping[str, Any]) -> None:
    GLOBAL_EXPOSURE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(GLOBAL_EXPOSURE_LEDGER, flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short append to global one-shot exposure ledger")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


class Status:
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
        outer_truth_accessed: bool = False,
    ) -> None:
        bounded = float(np.clip(progress, 0.0, 100.0))
        elapsed = time.perf_counter() - self.started
        if status in {"complete", "failed", "ready"}:
            eta = "완료" if status != "failed" else "중단"
        elif bounded > 0.5:
            remaining = elapsed * (100.0 - bounded) / bounded
            eta = (datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))).strftime(
                "%Y-%m-%d %H:%M:%S KST"
            )
        else:
            eta = "측정 중"
        _atomic_json_fsync(
            self.path,
            {
                "title": "P1 I-ORS external point-residual OOF",
                "experiment": EXPERIMENT_ID,
                "status": status,
                "phase": phase,
                "progress": round(bounded, 2),
                "detail": detail,
                "elapsed_seconds": round(elapsed, 3),
                "eta": eta,
                "updated_at": datetime.now().astimezone().isoformat(),
                "competition_outer_truth_accessed": outer_truth_accessed,
                "competition_upload": False,
                "frozen_submission_mutated": False,
                "frozen_model_mutated": False,
            },
        )


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("experiment config must be a JSON object")
    return value


def _permission_audit(section: Mapping[str, Any]) -> dict[str, Any]:
    permission_path = _resolve(Path(str(section["organizer_permission"])))
    if sha256_file(permission_path) != str(section["organizer_permission_sha256"]):
        raise RuntimeError("organizer permission receipt SHA mismatch")
    receipt = load_json_object(permission_path)
    evidence_path = _resolve(Path(str(receipt["evidence_file"])))
    evidence_sha = sha256_file(evidence_path)
    checks = {
        "approved": receipt.get("status") == "approved",
        "p1_allowed": "P1" in receipt.get("allowed_problems", []),
        "source_allowed": "i_ors_ctd_2014_2023" in receipt.get("allowed_sources", []),
        "feature_design_allowed": "feature_design" in receipt.get("allowed_purposes", []),
        "evidence_receipt_match": evidence_sha == str(receipt["evidence_sha256"]),
        "evidence_contract_match": evidence_sha == str(section["organizer_evidence_sha256"]),
    }
    if not all(checks.values()):
        raise PermissionError(f"external-data permission contract failed: {checks}")
    return {
        "receipt": str(permission_path.relative_to(PROJECT_ROOT)),
        "receipt_sha256": sha256_file(permission_path),
        "evidence": str(evidence_path.relative_to(PROJECT_ROOT)),
        "evidence_sha256": evidence_sha,
        "organizer_channel": receipt.get("organizer_channel"),
        "checks": checks,
    }


def _validate_contract(
    contract: Mapping[str, Any],
    experiment_path: Path,
    p1_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment_id")
    if contract.get("status") != "preregistered_one_shot":
        raise ValueError("experiment must remain preregistered_one_shot")
    authorization = contract["authorization"]
    if not authorization["competition_labels_for_nested_oof"]:
        raise PermissionError("nested P1 OOF was not authorized")
    forbidden = (
        "competition_upload",
        "frozen_submission_mutation",
        "frozen_model_mutation",
        "hyperparameter_search",
    )
    if any(bool(authorization[value]) for value in forbidden):
        raise PermissionError("upload, mutation, and hyperparameter search must remain false")
    external = contract["external_source"]
    hashes = {
        "experiment_config": sha256_file(experiment_path),
        "p1_config": sha256_file(p1_path),
        "source_manifest": sha256_file(source_path),
        "catalog": sha256_file(_resolve(Path(str(external["catalog"])))),
        "incumbent_oof": sha256_file(
            _resolve(Path(str(contract["p1_reference"]["incumbent_oof"])))
        ),
        "v1_result": sha256_file(
            _resolve(Path(str(contract["non_virgin_follow_up"]["v1_result"])))
        ),
        "v2_result": sha256_file(
            _resolve(Path(str(contract["non_virgin_follow_up"]["v2_result"])))
        ),
    }
    expected = {
        "p1_config": str(contract["p1_reference"]["config_sha256"]),
        "source_manifest": str(external["manifest_sha256"]),
        "catalog": str(external["catalog_sha256"]),
        "incumbent_oof": str(contract["p1_reference"]["incumbent_oof_sha256"]),
        "v1_result": str(contract["non_virgin_follow_up"]["v1_result_sha256"]),
        "v2_result": str(contract["non_virgin_follow_up"]["v2_result_sha256"]),
    }
    mismatches = {
        key: (hashes[key], value) for key, value in expected.items() if hashes[key] != value
    }
    if mismatches:
        raise RuntimeError(f"frozen provenance SHA mismatch: {mismatches}")
    if tuple(contract["point_feature_contract"]["columns"]) != POINT_RESIDUAL_COLUMNS:
        raise RuntimeError("point feature columns differ from the code contract")
    if bool(contract["point_feature_contract"]["q10_q90_features"]):
        raise RuntimeError("q10/q90 features are forbidden")
    if any("q10" in value or "q90" in value for value in POINT_RESIDUAL_COLUMNS):
        raise RuntimeError("interval quantile feature entered point-only contract")
    candidate = contract["candidate_model"]
    if candidate["backend"] != BACKEND or bool(candidate["hyperparameter_search"]):
        raise RuntimeError("candidate must be one fixed XGBoost configuration")
    if int(candidate["seed"]) != int(contract["outer_validation"]["bootstrap_seed"]):
        raise RuntimeError("the experiment must use exactly one fixed seed")
    if external["years"] != list(range(2014, 2024)) or int(external["qc"]) != 1:
        raise RuntimeError("external q50 must use 2014-2023 QC1 only")
    if not bool(external["use_all_eligible_rows"]):
        raise RuntimeError("external q50 sampling is forbidden")
    if not bool(contract["non_virgin_follow_up"]["disclosed"]):
        raise RuntimeError("non-virgin follow-up disclosure is missing")
    amendments = contract.get("amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise RuntimeError("exactly one pre-outer amendment must be disclosed")
    amendment = amendments[0]
    if amendment.get("amendment_id") != "A1_q2_label_blind_noop_and_canonical_lock":
        raise RuntimeError("unexpected experiment amendment")
    if amendment["aborted_attempt"].get("outer_truth_accessed") is not False:
        raise RuntimeError("the aborted attempt must remain pre-outer")
    if amendment["fold_policy"].get("incumbent_noop_folds") != ["2025_q2"]:
        raise RuntimeError("Q2 must remain the sole incumbent no-op fold")
    if amendment["fold_policy"].get("candidate_folds") != ["2025_q3", "2025_q4"]:
        raise RuntimeError("only Q3 and Q4 may use the candidate")
    if int(amendment["label_blind_support_evidence"]["2025_q2"]["outer_train"]) != 0:
        raise RuntimeError("Q2 no-op requires zero label-blind outer-train support")
    if int(contract["promotion_gate"]["minimum_improved_folds"]) != 2:
        raise RuntimeError("the two-fold improvement gate must remain unchanged")
    hardening = amendment["one_shot_hardening"]
    if hardening.get("hardcoded_canonical_experiment_config") != str(
        CANONICAL_EXPERIMENT_CONFIG.relative_to(PROJECT_ROOT)
    ).replace("\\", "/"):
        raise RuntimeError("hardcoded canonical experiment path contract changed")
    if hardening.get("experiment_wide_lock") != str(
        GLOBAL_EXPOSURE_LOCK.relative_to(PROJECT_ROOT)
    ).replace("\\", "/"):
        raise RuntimeError("experiment-wide lock contract changed")
    if hardening.get("append_only_exposure_ledger") != str(
        GLOBAL_EXPOSURE_LEDGER.relative_to(PROJECT_ROOT)
    ).replace("\\", "/"):
        raise RuntimeError("global exposure ledger contract changed")
    aborted_dir = _resolve(Path(str(amendment["aborted_attempt"]["run_dir"])))
    aborted_failure = aborted_dir / "failure.json"
    aborted_model = aborted_dir / "external_q50_lightgbm.txt"
    if sha256_file(aborted_failure) != str(amendment["aborted_attempt"]["failure_receipt_sha256"]):
        raise RuntimeError("aborted pre-outer failure receipt changed")
    if sha256_file(aborted_model) != str(amendment["aborted_attempt"]["external_q50_model_sha256"]):
        raise RuntimeError("aborted pre-outer external model changed")
    return hashes


def _load_blind_reference(contract: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve(Path(str(contract["p1_reference"]["incumbent_oof"])))
    columns = [*KEY_COLUMNS, "fold", "prediction", "probability", "plateau"]
    reference = pd.read_parquet(path, columns=columns)
    key = [*KEY_COLUMNS, "fold"]
    if reference.duplicated(key).any():
        raise RuntimeError("incumbent OOF key/fold membership is not unique")
    expected_folds = set(contract["outer_validation"]["folds"])
    if set(reference["fold"].astype(str).unique()) != expected_folds:
        raise RuntimeError("incumbent OOF fold membership differs from the contract")
    if reference["probability"].dtype != np.dtype("float32"):
        raise RuntimeError("incumbent probability dtype changed")
    return reference


def _reference_train_positions(train: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    train_index = pd.MultiIndex.from_frame(train.loc[:, KEY_COLUMNS])
    if not train_index.is_unique:
        raise RuntimeError("P1 train keys must be unique")
    reference_index = pd.MultiIndex.from_frame(reference.loc[:, KEY_COLUMNS])
    positions = train_index.get_indexer(reference_index)
    if (positions < 0).any():
        raise RuntimeError("incumbent OOF contains a key absent from P1 train")
    return positions.astype(np.int64, copy=False)


def _fit_external_q50(
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
    archive: Path,
    run_dir: Path,
    status: Status,
) -> tuple[Any, dict[str, Any]]:
    import lightgbm as lgb

    external = contract["external_source"]
    target_depths = {int(key): float(value) for key, value in external["target_grid_m"].items()}
    profiles = []
    years = [int(value) for value in external["years"]]
    cutoff = pd.Timestamp(external["hard_cutoff_kst"]).tz_convert("UTC").tz_localize(None)
    cutoff_value = np.datetime64(cutoff.to_datetime64()).astype("datetime64[s]")
    for position, year in enumerate(years, start=1):
        status.update(
            10.0 + 22.0 * position / len(years),
            "external_decode",
            f"{year} QC1 profile ({position}/{len(years)})",
        )
        profile = read_year_profile(
            archive,
            source,
            year=year,
            target_depth_by_layer=target_depths,
            max_mapping_distance_m=float(external["maximum_mapping_distance_m"]),
        )
        if profile.time_utc.max().astype("datetime64[s]") > cutoff_value:
            raise RuntimeError(f"external year {year} exceeds hard pre-2024 cutoff")
        profiles.append(profile)
    status.update(34.0, "external_features", "2014-2023 QC1 전체 LOO feature 구성")
    dataset = build_loo_dataset(
        profiles,
        min_peer_temperatures=int(external["minimum_peer_temperatures"]),
        max_rows_per_year_layer=None,
    )
    if set(int(value) for value in np.unique(dataset.year)) != set(years):
        raise RuntimeError("external q50 fit years are incomplete")
    point = contract["external_q50_model"]
    parameters = {
        key: value for key, value in point.items() if key not in {"backend", "objective", "alpha"}
    }
    parameters["random_state"] = int(contract["candidate_model"]["seed"])
    status.update(40.0, "external_model", f"고정 q50 LightGBM · {len(dataset.y):,} rows")
    model = lgb.LGBMRegressor(
        objective=str(point["objective"]),
        alpha=float(point["alpha"]),
        **parameters,
    )
    model.fit(dataset.x, dataset.y)
    if int(model.booster_.num_trees()) != int(point["n_estimators"]):
        raise RuntimeError("external q50 tree count differs from the fixed contract")
    model_path = run_dir / "external_q50_lightgbm.txt"
    model.booster_.save_model(str(model_path))
    importance = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    total = float(importance.sum())
    ranking = np.argsort(-importance)[:20]
    audit = {
        "dataset": dataset_audit(dataset),
        "model": {
            "library": "lightgbm",
            "version": lgb.__version__,
            "objective": "quantile",
            "alpha": 0.5,
            "parameters": parameters,
            "fit_rows": int(dataset.y.size),
            "features": int(dataset.x.shape[1]),
            "feature_names": list(dataset.feature_names),
            "target_temperature_masked": True,
            "p1_label_or_anomaly_type_input": False,
            "q10_q90_models_fit": False,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "model_sha256": sha256_file(model_path),
            "top_gain_features": [
                {
                    "feature": dataset.feature_names[int(index)],
                    "gain_fraction": float(importance[index] / total) if total else 0.0,
                }
                for index in ranking
            ],
        },
        "profiles": [profile.audit for profile in profiles],
        "cutoff_kst": external["hard_cutoff_kst"],
        "all_eligible_rows": True,
    }
    del dataset, profiles
    gc.collect()
    return model, audit


def _fixed_inner_indices(
    parsed_time: pd.Series,
    *,
    train_end: pd.Timestamp,
    calibration_days: int,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_start = train_end - pd.Timedelta(days=calibration_days)
    fit_end = calibration_start - pd.Timedelta(days=purge_days)
    outer_train = np.flatnonzero(parsed_time.le(train_end).to_numpy())
    inner_fit = np.flatnonzero(parsed_time.le(fit_end).to_numpy())
    calibration = np.flatnonzero(
        (parsed_time.ge(calibration_start) & parsed_time.le(train_end)).to_numpy()
    )
    if not len(inner_fit) or not len(calibration):
        raise RuntimeError("fixed inner split is empty")
    return outer_train, inner_fit, calibration


def _fit_candidate_blind(
    train: pd.DataFrame,
    config: P1QCConfig,
    contract: Mapping[str, Any],
    reference: pd.DataFrame,
    reference_positions: np.ndarray,
    point_features: pd.DataFrame,
    external_eligible: np.ndarray,
    run_dir: Path,
    status: Status,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    status.update(53.0, "base_features", "frozen offline P1 feature cache 로드")
    bundle = load_or_build_features(train, config, kind="train", use_cache=True)
    parsed_time = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    station_i = train["station"].astype("string").eq("I-ORS").to_numpy()
    eligible_i = station_i & external_eligible
    target = train["label"].to_numpy(dtype=np.int8)
    candidate_contract = contract["candidate_model"]
    parameters = dict(candidate_contract["parameters"])
    configured_iterations = int(parameters["n_estimators"])
    seed = int(candidate_contract["seed"])
    selection_parameters = dict(parameters)
    selection_parameters["early_stopping_rounds"] = int(
        candidate_contract["inner_early_stopping_rounds"]
    )
    threshold_candidates = [float(value) for value in candidate_contract["threshold_candidates"]]
    fold_specs = {value.name: value for value in config.splits.folds}
    if set(fold_specs) != set(contract["outer_validation"]["folds"]):
        raise RuntimeError("P1 fold specs differ from the frozen outer names")
    amendment = contract["amendments"][-1]
    fold_policy = amendment["fold_policy"]
    candidate_folds = tuple(str(value) for value in fold_policy["candidate_folds"])
    noop_folds = tuple(str(value) for value in fold_policy["incumbent_noop_folds"])
    if set(candidate_folds) | set(noop_folds) != set(contract["outer_validation"]["folds"]):
        raise RuntimeError("amended candidate/no-op fold policy is incomplete")
    if set(candidate_folds) & set(noop_folds):
        raise RuntimeError("candidate and no-op folds overlap")
    fold_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []

    for fold_number, fold_name in enumerate(contract["outer_validation"]["folds"]):
        progress = 57.0 + fold_number * 10.0
        spec = fold_specs[str(fold_name)]
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        outer_train, inner_fit, calibration = _fixed_inner_indices(
            parsed_time,
            train_end=train_end,
            calibration_days=int(contract["outer_validation"]["inner_calibration_days"]),
            purge_days=int(contract["outer_validation"]["purge_days"]),
        )
        outer_i = outer_train[eligible_i[outer_train]]
        inner_i = inner_fit[eligible_i[inner_fit]]
        calibration_i = calibration[eligible_i[calibration]]
        reference_mask = reference["fold"].astype(str).eq(str(fold_name)).to_numpy()
        reference_rows = np.flatnonzero(reference_mask)
        validation_positions_all = reference_positions[reference_rows]
        validation_eligible_mask = eligible_i[validation_positions_all]
        validation_rows = reference_rows[validation_eligible_mask]
        validation_i = validation_positions_all[validation_eligible_mask]
        observed_support = {
            "outer_train": int(len(outer_i)),
            "inner_fit": int(len(inner_i)),
            "inner_calibration": int(len(calibration_i)),
            "validation": int(len(validation_i)),
        }
        expected_support = {
            key: int(value)
            for key, value in amendment["label_blind_support_evidence"][str(fold_name)].items()
        }
        if observed_support != expected_support:
            raise RuntimeError(
                f"{fold_name} label-blind support changed: "
                f"expected={expected_support}, observed={observed_support}"
            )
        if np.intersect1d(outer_train, validation_positions_all).size:
            raise RuntimeError(f"{fold_name} frozen validation overlaps outer train")
        if str(fold_name) in noop_folds:
            if len(outer_i) != 0:
                raise RuntimeError(f"{fold_name} no-op is permitted only for zero outer support")
            status.update(
                progress + 9.0,
                f"{fold_name}:noop",
                "label-blind train support 0 · frozen incumbent bytes 유지",
            )
            audits.append(
                {
                    "fold": fold_name,
                    "action": "incumbent_noop",
                    "reason": "label-blind external eligible outer-train support is zero",
                    "outer_train_rows_iors_eligible": 0,
                    "inner_fit_rows_iors_eligible": int(len(inner_i)),
                    "inner_calibration_rows_iors_eligible": int(len(calibration_i)),
                    "outer_validation_rows_all": int(len(validation_positions_all)),
                    "outer_validation_rows_iors_eligible": int(len(validation_i)),
                    "replacement_rows": 0,
                    "outer_validation_truth_used": False,
                }
            )
            continue
        if str(fold_name) not in candidate_folds:
            raise RuntimeError(f"{fold_name} has no amended fold policy")
        status.update(progress, f"{fold_name}:inner", "outer-train 내부 iteration/threshold 선택")
        if not len(validation_i):
            raise RuntimeError(f"{fold_name} has no eligible I-ORS validation rows")
        for name, values in {
            "inner_fit": inner_i,
            "calibration": calibration_i,
            "outer_train": outer_i,
        }.items():
            if not len(values) or target[values].sum() == 0:
                raise RuntimeError(f"{fold_name} {name} lacks eligible positive rows")
        if np.intersect1d(inner_i, calibration_i).size:
            raise RuntimeError(f"{fold_name} inner fit/calibration overlap")

        encoder_inner = TabularEncoder().fit(bundle, inner_i)
        x_inner = append_point_residual_matrix(
            encoder_inner.transform(bundle, inner_i), point_features, inner_i
        )
        x_calibration = append_point_residual_matrix(
            encoder_inner.transform(bundle, calibration_i), point_features, calibration_i
        )
        selection_model = _fit_model(
            BACKEND,
            selection_parameters,
            seed,
            _threads(config),
            x_inner,
            target[inner_i],
            evaluation=(x_calibration, target[calibration_i]),
        )
        best_iterations = _best_iteration(selection_model, configured_iterations)
        if not 1 <= best_iterations <= configured_iterations:
            raise RuntimeError(f"{fold_name} invalid selected iteration count")
        calibration_probability = selection_model.predict_proba(x_calibration)[:, 1]
        calibration_plateau = detect_plateaus(train.iloc[calibration_i]).to_numpy()
        threshold, _, threshold_audit = select_inner_threshold(
            target[calibration_i],
            calibration_probability,
            calibration_plateau,
            threshold_candidates,
        )

        status.update(
            progress + 5.0, f"{fold_name}:outer", "I-ORS fixed outer model blind prediction"
        )
        encoder = TabularEncoder().fit(bundle, outer_i)
        x_outer = append_point_residual_matrix(
            encoder.transform(bundle, outer_i), point_features, outer_i
        )
        x_validation = append_point_residual_matrix(
            encoder.transform(bundle, validation_i), point_features, validation_i
        )
        outer_parameters = dict(parameters)
        outer_parameters["n_estimators"] = best_iterations
        model = _fit_model(
            BACKEND,
            outer_parameters,
            seed,
            _threads(config),
            x_outer,
            target[outer_i],
        )
        probability = model.predict_proba(x_validation)[:, 1]
        plateau = reference.iloc[validation_rows]["plateau"].to_numpy(dtype=bool)
        prediction = ((probability >= threshold) | plateau).astype(np.int8)
        output = reference.iloc[validation_rows].loc[:, [*KEY_COLUMNS, "fold"]].copy()
        output["candidate_probability"] = probability.astype(np.float32)
        output["candidate_prediction"] = prediction
        fold_parts.append(output)
        model_path = run_dir / f"{fold_name}_candidate.joblib"
        joblib.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "fold": fold_name,
                "action": "candidate_fit_replace",
                "encoder": encoder,
                "model": model,
                "threshold": threshold,
                "iteration_count": best_iterations,
                "point_feature_columns": POINT_RESIDUAL_COLUMNS,
            },
            model_path,
            compress=3,
        )
        audits.append(
            {
                "fold": fold_name,
                "action": "candidate_fit_replace",
                "outer_validation_membership_source": "frozen incumbent OOF key/fold",
                "outer_validation_membership_regenerated": False,
                "outer_train_rows_iors_eligible": int(len(outer_i)),
                "inner_fit_rows_iors_eligible": int(len(inner_i)),
                "inner_calibration_rows_iors_eligible": int(len(calibration_i)),
                "outer_validation_rows_all": int(len(validation_positions_all)),
                "outer_validation_rows_iors_eligible": int(len(validation_i)),
                "selected_iterations": int(best_iterations),
                "selected_threshold": float(threshold),
                "threshold_selection": threshold_audit,
                "selection_scope": "outer-train internal calibration only",
                "outer_validation_truth_used_for_selection": False,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                "model_sha256": sha256_file(model_path),
            }
        )
        del selection_model, model, encoder_inner, encoder
        del x_inner, x_calibration, x_outer, x_validation
        gc.collect()

    replacements = pd.concat(fold_parts, ignore_index=True)
    if replacements.duplicated([*KEY_COLUMNS, "fold"]).any():
        raise RuntimeError("blind candidate replacement keys are duplicated")
    return (
        replacements,
        {
            "folds": audits,
            "seed_values": [seed],
            "hyperparameter_grid": False,
            "candidate_folds": list(candidate_folds),
            "incumbent_noop_folds": list(noop_folds),
            "outer_validation_membership_source": "frozen incumbent OOF key/fold",
            "outer_folds_function_called": False,
            "all_outer_predictions_completed_before_outer_truth_access": True,
        },
    )


def _relative_increase(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - baseline) / baseline


def _noop_fold_byte_audit(
    reference: pd.DataFrame,
    blind: pd.DataFrame,
    noop_folds: Sequence[str],
) -> dict[str, Any]:
    if len(reference) != len(blind):
        raise ValueError("reference and blind frames must have equal rows")
    key = [*KEY_COLUMNS, "fold"]
    if not reference.loc[:, key].equals(blind.loc[:, key]):
        raise ValueError("blind row order differs from the frozen reference")
    result: dict[str, Any] = {}
    for fold in noop_folds:
        mask = reference["fold"].astype(str).eq(str(fold)).to_numpy()
        before_probability = reference.loc[mask, "probability"].to_numpy(copy=False)
        after_probability = blind.loc[mask, "candidate_probability"].to_numpy(copy=False)
        before_prediction = reference.loc[mask, "prediction"].to_numpy(copy=False)
        after_prediction = blind.loc[mask, "candidate_prediction"].to_numpy(copy=False)
        if before_probability.tobytes() != after_probability.tobytes():
            raise AssertionError(f"{fold} no-op probability bytes changed")
        if before_prediction.tobytes() != after_prediction.tobytes():
            raise AssertionError(f"{fold} no-op prediction bytes changed")
        result[str(fold)] = {
            "rows": int(mask.sum()),
            "probability_sha256_before": hashlib.sha256(before_probability.tobytes()).hexdigest(),
            "probability_sha256_after": hashlib.sha256(after_probability.tobytes()).hexdigest(),
            "prediction_sha256_before": hashlib.sha256(before_prediction.tobytes()).hexdigest(),
            "prediction_sha256_after": hashlib.sha256(after_prediction.tobytes()).hexdigest(),
            "byte_identical": True,
        }
    return result


def _evaluate_once(
    blind: pd.DataFrame,
    reference_path: Path,
    test: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth_columns = [*KEY_COLUMNS, "fold", "label", "anomaly_type"]
    truth_frame = pd.read_parquet(reference_path, columns=truth_columns)
    complete = blind.merge(
        truth_frame,
        on=[*KEY_COLUMNS, "fold"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(complete) != len(blind) or len(complete) != len(truth_frame):
        raise RuntimeError("blind prediction and outer truth keys do not align")
    truth = complete["label"].to_numpy(dtype=np.int8)
    candidate = complete["candidate_prediction"].to_numpy(dtype=np.int8)
    incumbent = complete["incumbent_prediction"].to_numpy(dtype=np.int8)
    test_shares = group_row_shares(test)
    candidate_report = evaluate_predictions(
        truth,
        candidate,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    incumbent_report = evaluate_predictions(
        truth,
        incumbent,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    expected = float(contract["p1_reference"]["incumbent_weighted_f1"])
    if abs(incumbent_report.weighted.f1 - expected) > 1.0e-12:
        raise RuntimeError("frozen incumbent weighted F1 did not reproduce")

    i_mask = complete["station"].astype("string").eq("I-ORS").to_numpy()
    i_truth = truth[i_mask]
    i_candidate = candidate[i_mask]
    i_incumbent = incumbent[i_mask]
    i_candidate_report = evaluate_predictions(
        i_truth,
        i_candidate,
        complete.loc[i_mask].reset_index(drop=True),
        anomaly_type=complete.loc[i_mask, "anomaly_type"].reset_index(drop=True),
    )
    i_incumbent_report = evaluate_predictions(
        i_truth,
        i_incumbent,
        complete.loc[i_mask].reset_index(drop=True),
        anomaly_type=complete.loc[i_mask, "anomaly_type"].reset_index(drop=True),
    )
    type_delta = {
        value: float(candidate_report.type_recall[value] - incumbent_report.type_recall[value])
        for value in ("spike", "noise", "flatline", "offset", "drift")
    }
    i_type_delta = {
        value: float(i_candidate_report.type_recall[value] - i_incumbent_report.type_recall[value])
        for value in ("spike", "noise", "flatline", "offset", "drift")
    }

    candidate_groups = candidate_report.groups.set_index(["station", "layer"])
    incumbent_groups = incumbent_report.groups.set_index(["station", "layer"])
    i_layer_rows: list[dict[str, Any]] = []
    for key in candidate_groups.index:
        if str(key[0]) != "I-ORS":
            continue
        candidate_f1 = float(candidate_groups.loc[key, "f1"])
        incumbent_f1 = float(incumbent_groups.loc[key, "f1"])
        i_layer_rows.append(
            {
                "layer": int(key[1]),
                "candidate_f1": candidate_f1,
                "incumbent_f1": incumbent_f1,
                "delta": candidate_f1 - incumbent_f1,
            }
        )
    worst_i_layer = min((value["delta"] for value in i_layer_rows), default=float("-inf"))

    fold_rows: list[dict[str, Any]] = []
    for fold_name, part in complete.groupby("fold", sort=False, observed=True):
        part_truth = part["label"].to_numpy(dtype=np.int8)
        candidate_fold = evaluate_predictions(
            part_truth,
            part["candidate_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=test_shares,
            anomaly_type=part["anomaly_type"],
        )
        incumbent_fold = evaluate_predictions(
            part_truth,
            part["incumbent_prediction"].to_numpy(dtype=np.int8),
            part,
            group_weights=test_shares,
            anomaly_type=part["anomaly_type"],
        )
        delta = float(candidate_fold.weighted.f1 - incumbent_fold.weighted.f1)
        fold_rows.append(
            {
                "fold": str(fold_name),
                "candidate_weighted_f1": candidate_fold.weighted.f1,
                "incumbent_weighted_f1": incumbent_fold.weighted.f1,
                "delta": delta,
                "improved": delta > 0.0,
            }
        )
    improved_folds = sum(bool(value["improved"]) for value in fold_rows)
    bootstrap = paired_block_bootstrap(
        truth,
        candidate,
        incumbent,
        complete,
        replicates=int(contract["outer_validation"]["bootstrap_replicates"]),
        seed=int(contract["outer_validation"]["bootstrap_seed"]),
        normal_day_timezone="Asia/Seoul",
    )
    normal_fp = normal_station_layer_day_fp(truth, candidate, incumbent, complete)
    candidate_fp = float(normal_fp["candidate"]["false_positive_rows_per_normal_station_layer_day"])
    incumbent_fp = float(normal_fp["baseline"]["false_positive_rows_per_normal_station_layer_day"])
    fp_relative = _relative_increase(candidate_fp, incumbent_fp)
    gate = apply_point_residual_gate(
        overall_weighted_f1_delta=float(
            candidate_report.weighted.f1 - incumbent_report.weighted.f1
        ),
        iors_micro_f1_delta=float(i_candidate_report.micro.f1 - i_incumbent_report.micro.f1),
        anomaly_type_recall_delta=type_delta,
        normal_fp_day_relative_increase=fp_relative,
        worst_iors_layer_f1_delta=float(worst_i_layer),
        paired_bootstrap_ci90_lower=float(bootstrap["difference_ci90"][0]),
        improved_folds=improved_folds,
        contract=contract["promotion_gate"],
    )
    metrics = {
        "candidate": candidate_report.to_dict(),
        "incumbent": incumbent_report.to_dict(),
        "iors_candidate": i_candidate_report.to_dict(),
        "iors_incumbent": i_incumbent_report.to_dict(),
        "global_anomaly_type_recall_delta": type_delta,
        "iors_anomaly_type_recall_delta": i_type_delta,
        "iors_layer_comparison": i_layer_rows,
        "fold_comparison": fold_rows,
        "paired_block_bootstrap": bootstrap,
        "normal_station_layer_day_fp": normal_fp,
        "normal_fp_day_relative_increase": fp_relative,
        "gate": gate,
        "outer_is_independent_holdout": False,
        "non_virgin_follow_up": True,
        "official_hidden_test_used": False,
    }
    return complete, metrics


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": bool(status), "changed_path_count": len(status)}


def _implementation_hashes(experiment_path: Path) -> dict[str, str]:
    paths = {
        "experiment_config": experiment_path,
        "runner": Path(__file__).resolve(),
        "point_residual_helper": PROJECT_ROOT / "src/p1_qc/iors_external_point_residual.py",
        "iors_ctd": PROJECT_ROOT / "src/ocean_external/iors_ctd.py",
        "pipeline": PROJECT_ROOT / "src/p1_qc/pipeline.py",
        "models_tabular": PROJECT_ROOT / "src/p1_qc/models_tabular.py",
        "metrics": PROJECT_ROOT / "src/p1_qc/metrics.py",
        "validation": PROJECT_ROOT / "src/p1_qc/validation.py",
        "rules": PROJECT_ROOT / "src/p1_qc/rules.py",
        "data": PROJECT_ROOT / "src/p1_qc/data.py",
        "config_loader": PROJECT_ROOT / "src/p1_qc/config.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"implementation seal files are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _verify_incumbent_oof_sha(contract: Mapping[str, Any]) -> str:
    reference = contract["p1_reference"]
    path = _resolve(Path(str(reference["incumbent_oof"])))
    actual = sha256_file(path)
    expected = str(reference["incumbent_oof_sha256"])
    if actual != expected:
        raise RuntimeError(
            f"incumbent OOF changed during one-shot run: expected={expected}, actual={actual}"
        )
    return actual


def _verify_implementation_seal(
    expected: Mapping[str, str], experiment_path: Path
) -> dict[str, str]:
    actual = _implementation_hashes(experiment_path)
    if dict(expected) != actual:
        changed = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"implementation changed during one-shot run: {changed}")
    return actual


def _environment() -> dict[str, Any]:
    summary = environment_summary()
    try:
        import h5py

        summary["h5py"] = h5py.__version__
    except ImportError:
        summary["h5py"] = None
    try:
        import lightgbm

        summary["lightgbm"] = lightgbm.__version__
    except ImportError:
        summary["lightgbm"] = None
    try:
        import xgboost

        summary["xgboost"] = xgboost.__version__
    except ImportError:
        summary["xgboost"] = None
    summary["numpy"] = np.__version__
    summary["pandas"] = pd.__version__
    summary["platform_detail"] = platform.platform()
    summary["git"] = _git_state()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require_hardcoded_canonical_paths(args)
    experiment_path = CANONICAL_EXPERIMENT_CONFIG
    p1_path = _resolve(args.p1_config)
    contract = _load_contract(experiment_path)
    artifact_paths = canonical_artifact_paths(
        PROJECT_ROOT,
        contract["artifacts"],
        requested_output_dir=args.output_dir,
        requested_status_file=args.status_file,
    )
    if artifact_paths.output_dir != CANONICAL_OUTPUT_DIR:
        raise RuntimeError("config output_dir differs from the hardcoded canonical path")
    if artifact_paths.status_file != CANONICAL_STATUS_FILE:
        raise RuntimeError("config status differs from the hardcoded canonical path")
    if artifact_paths.outer_lock != CANONICAL_OUTER_LOCK:
        raise RuntimeError("config outer_lock differs from the hardcoded canonical path")
    source_path = _resolve(
        args.source_manifest
        if args.source_manifest is not None
        else Path(str(contract["external_source"]["manifest"]))
    )
    canonical_source = _resolve(Path(str(contract["external_source"]["manifest"])))
    if source_path != canonical_source:
        raise ValueError("--source-manifest must equal the canonical preregistered path")
    output_root = artifact_paths.output_dir
    status = Status(artifact_paths.status_file)
    outer_lock = artifact_paths.outer_lock
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    outer_truth_accessed = False
    run_dir: Path | None = None
    try:
        status.update(1.0, "contract", "사전등록·SHA·외부자료 권한 fail-closed 검사")
        hashes = _validate_contract(contract, experiment_path, p1_path, source_path)
        implementation_seal = _implementation_hashes(experiment_path)
        permission = _permission_audit(contract["external_source"])
        source = load_json_object(source_path)
        validate_source_manifest(source)
        if source["license"]["spdx"] != "CC-BY-4.0":
            raise PermissionError("external archive license must remain CC-BY-4.0")
        official = (
            {"skipped": True, "reason": "--skip-live-record-check"}
            if args.skip_live_record_check
            else verify_official_record(source)
        )
        quarantine = _resolve(args.quarantine_dir)
        archive = ensure_archive(source, quarantine, allow_download=args.download)
        archive_audit = verify_archive(archive, source)
        if sha256_file(archive) != str(contract["external_source"]["archive_sha256"]):
            raise RuntimeError("external archive SHA differs from preregistration")
        if outer_lock.exists():
            raise FileExistsError(
                "outer exposure lock exists; this one-shot experiment cannot be rerun"
            )
        _assert_not_globally_exposed()
        reference = _load_blind_reference(contract)
        preflight = {
            "experiment_id": EXPERIMENT_ID,
            "checked_at": datetime.now().astimezone().isoformat(),
            "contract_hashes": hashes,
            "implementation_seal": implementation_seal,
            "canonical_artifacts": {
                "output_dir": str(output_root.relative_to(PROJECT_ROOT)),
                "status_file": str(artifact_paths.status_file.relative_to(PROJECT_ROOT)),
                "outer_lock": str(outer_lock.relative_to(PROJECT_ROOT)),
                "global_exposure_lock": str(GLOBAL_EXPOSURE_LOCK.relative_to(PROJECT_ROOT)),
                "global_exposure_ledger": str(GLOBAL_EXPOSURE_LEDGER.relative_to(PROJECT_ROOT)),
            },
            "permission": permission,
            "official_record": official,
            "archive": archive_audit,
            "incumbent_rows": len(reference),
            "incumbent_folds": reference["fold"].value_counts().sort_index().to_dict(),
            "outer_truth_columns_opened": False,
            "ready": True,
        }
        _atomic_json_fsync(output_root / "preflight.json", preflight)
        if args.preflight_only:
            status.update(
                100.0,
                "ready",
                "READY · outer truth 미접근 · actual one-shot 실행 가능",
                status="ready",
            )
            print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
            return 0

        run_id = started_at.strftime("%Y%m%dT%H%M%S%z")
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        status.update(6.0, "data", "P1 원본 audit 및 frozen OOF membership 고정")
        config = load_config(p1_path)
        data_dir = resolve_data_dir(config, args.data_dir)
        train, test = load_train_test(data_dir, audit=True, strict=True)
        if train.attrs["source_sha256"] != contract["p1_reference"]["train_sha256"]:
            raise RuntimeError("P1 train SHA differs from preregistration")
        if test.attrs["source_sha256"] != contract["p1_reference"]["test_sha256"]:
            raise RuntimeError("P1 test SHA differs from preregistration")
        reference_positions = _reference_train_positions(train, reference)

        external_model, external_audit = _fit_external_q50(
            source, contract, archive, run_dir, status
        )
        status.update(46.0, "p1_external_prediction", "P1 I-ORS target TEMP 완전 마스크 q50 추론")
        external_input = train.loc[:, EXTERNAL_INPUT_COLUMNS].copy()
        if "label" in external_input or "anomaly_type" in external_input:
            raise AssertionError("P1 label entered external q50 feature path")
        panel = build_p1_iors_panel(
            external_input,
            {
                int(key): float(value)
                for key, value in contract["external_source"]["target_grid_m"].items()
            },
        )
        maximum_mapping = max(
            float(value["absolute_difference_m"]) for value in panel.profile.mapping
        )
        if maximum_mapping > float(contract["external_source"]["maximum_mapping_distance_m"]):
            raise RuntimeError("P1 I-ORS depth mapping exceeds the preregistered tolerance")
        point_prediction = predict_external_q50(
            panel,
            external_model,
            min_peer_temperatures=int(contract["external_source"]["minimum_peer_temperatures"]),
        )
        point_features, point_audit = build_point_residual_features(
            external_input,
            point_prediction,
            cadence_minutes=config.data.cadence_minutes,
            minimum_fraction=float(
                contract["point_feature_contract"]["rolling"]["median_min_fraction"]
            ),
        )
        if tuple(point_features.columns) != POINT_RESIDUAL_COLUMNS:
            raise RuntimeError("derived point feature contract changed")
        p1_external_audit = {
            "panel": panel.profile.audit,
            "prediction": point_prediction.audit,
            "derived_features": point_audit,
            "maximum_depth_mapping_difference_m": maximum_mapping,
            "external_input_columns": list(external_input.columns),
            "p1_label_or_anomaly_type_input": False,
        }
        del panel, external_input, external_model
        gc.collect()

        amendment = contract["amendments"][-1]
        candidate_folds = amendment["fold_policy"]["candidate_folds"]
        noop_folds = amendment["fold_policy"]["incumbent_noop_folds"]
        expected = independent_expected_replacement_keys(
            reference,
            reference_positions,
            point_prediction.eligible,
            candidate_folds=candidate_folds,
        )
        replacements, model_audit = _fit_candidate_blind(
            train,
            config,
            contract,
            reference,
            reference_positions,
            point_features,
            point_prediction.eligible,
            run_dir,
            status,
        )
        reference_compose = reference.loc[
            :, [*KEY_COLUMNS, "fold", "prediction", "probability"]
        ].copy()
        blind, compose_audit = compose_incumbent_predictions(
            reference_compose, replacements, expected
        )
        noop_audit = _noop_fold_byte_audit(reference_compose, blind, noop_folds)
        compose_audit["incumbent_noop_folds"] = noop_audit
        compose_audit["expected_replacement_key_source"] = (
            "reference-wide I-ORS external eligibility intersected with amended candidate folds"
        )
        compose_audit["expected_replacement_rows"] = len(expected)
        if len(replacements) != len(expected):
            raise RuntimeError("candidate replacement completeness check failed")
        blind_path = run_dir / "blind_predictions.parquet"
        _write_parquet_fsync(blind_path, blind)
        blind_sha = sha256_file(blind_path)
        _verify_implementation_seal(implementation_seal, experiment_path)
        incumbent_sha_before_lock = _verify_incumbent_oof_sha(contract)
        blind_receipt = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "created_at": datetime.now().astimezone().isoformat(),
            "blind_predictions": str(blind_path.relative_to(PROJECT_ROOT)),
            "blind_predictions_sha256": blind_sha,
            "rows": len(blind),
            "compose_audit": compose_audit,
            "implementation_seal": implementation_seal,
            "implementation_seal_sha256": _hash_json(implementation_seal),
            "incumbent_oof_sha256": incumbent_sha_before_lock,
            "outer_validation_membership_source": "frozen incumbent OOF key/fold",
            "outer_truth_accessed": False,
            "all_outer_predictions_complete": True,
            "non_virgin_follow_up_disclosed": True,
            "competition_upload": False,
        }
        blind_receipt_path = run_dir / "blind_exposure_receipt.json"
        _atomic_json_fsync(blind_receipt_path, blind_receipt)
        blind_receipt_sha = sha256_file(blind_receipt_path)
        exposure_payload = {
            "experiment_id": EXPERIMENT_ID,
            "created_at": datetime.now().astimezone().isoformat(),
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
            "blind_predictions_sha256": blind_sha,
            "blind_exposure_receipt_sha256": blind_receipt_sha,
            "implementation_seal_sha256": _hash_json(implementation_seal),
            "incumbent_oof_sha256": incumbent_sha_before_lock,
            "outer_truth_accessed_at_lock_creation": False,
            "rerun_forbidden": True,
        }
        _assert_not_globally_exposed()
        _exclusive_json_fsync(GLOBAL_EXPOSURE_LOCK, exposure_payload)
        _exclusive_json_fsync(outer_lock, exposure_payload)
        _append_exposure_ledger(exposure_payload)
        status.update(
            89.0,
            "blind_locked",
            f"blind SHA {blind_sha[:12]} · exposure receipt fsync 완료",
        )

        if sha256_file(blind_path) != blind_sha:
            raise RuntimeError("sealed blind parquet changed before reload")
        sealed_blind = pd.read_parquet(blind_path)
        if sha256_file(blind_path) != blind_sha:
            raise RuntimeError("sealed blind parquet changed during reload")
        if len(sealed_blind) != len(reference) or list(sealed_blind.columns) != list(blind.columns):
            raise RuntimeError("reloaded blind parquet schema or row count changed")
        if sha256_file(blind_receipt_path) != blind_receipt_sha:
            raise RuntimeError("blind exposure receipt changed before outer truth access")
        if not GLOBAL_EXPOSURE_LOCK.is_file() or not outer_lock.is_file():
            raise RuntimeError("one-shot exposure locks disappeared before outer truth access")
        ledger_rows = _read_exposure_ledger(GLOBAL_EXPOSURE_LEDGER)
        if sum(value.get("experiment_id") == EXPERIMENT_ID for value in ledger_rows) != 1:
            raise RuntimeError("global one-shot exposure ledger reservation is not unique")
        _verify_implementation_seal(implementation_seal, experiment_path)
        incumbent_sha_before_truth = _verify_incumbent_oof_sha(contract)
        if incumbent_sha_before_truth != incumbent_sha_before_lock:
            raise RuntimeError("incumbent OOF SHA changed between blind lock and truth access")
        del blind
        gc.collect()
        outer_truth_accessed = True
        status.update(
            91.0,
            "outer_evaluation",
            "고정 outer truth 최초 1회 평가·bootstrap gate",
            outer_truth_accessed=True,
        )
        reference_path = _resolve(Path(str(contract["p1_reference"]["incumbent_oof"])))
        evaluated, metrics = _evaluate_once(sealed_blind, reference_path, test, contract)
        incumbent_sha_after_truth = _verify_incumbent_oof_sha(contract)
        if incumbent_sha_after_truth != incumbent_sha_before_truth:
            raise RuntimeError("incumbent OOF SHA changed while outer truth was read")
        evaluated_path = run_dir / "evaluated_oof.parquet"
        _write_parquet_fsync(evaluated_path, evaluated)
        result = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "decision": metrics["gate"]["decision"],
            "scope": {
                "competition_labels_used_for_nested_oof": True,
                "competition_outer_truth_access_count": 1,
                "competition_hidden_test_used": False,
                "competition_submission_created": False,
                "competition_upload": False,
                "frozen_submission_mutated": False,
                "frozen_model_mutated": False,
                "non_virgin_follow_up": True,
            },
            "contracts": {
                "experiment": str(experiment_path.relative_to(PROJECT_ROOT)),
                "experiment_sha256": sha256_file(experiment_path),
                "effective_contract_sha256": _hash_json(contract),
                "p1_config_sha256": sha256_file(p1_path),
                "source_manifest_sha256": sha256_file(source_path),
                "catalog_sha256": sha256_file(
                    _resolve(Path(str(contract["external_source"]["catalog"])))
                ),
                "implementation_seal": implementation_seal,
                "implementation_seal_sha256": _hash_json(implementation_seal),
            },
            "permission": permission,
            "official_record": official,
            "archive": archive_audit,
            "external_q50": external_audit,
            "p1_external_features": p1_external_audit,
            "candidate_training": model_audit,
            "blind_compose": compose_audit,
            "blind_predictions": {
                "path": str(blind_path.relative_to(PROJECT_ROOT)),
                "sha256": blind_sha,
                "receipt_sha256": blind_receipt_sha,
                "persisted_before_outer_truth_access": True,
            },
            "metrics": metrics,
            "environment": _environment(),
        }
        result_path = run_dir / "result.json"
        _atomic_json_fsync(result_path, result)
        receipt = {
            "experiment_id": EXPERIMENT_ID,
            "result": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "blind_predictions_sha256": blind_sha,
            "blind_exposure_receipt_sha256": blind_receipt_sha,
            "implementation_seal_sha256": _hash_json(implementation_seal),
            "incumbent_oof_sha256": incumbent_sha_after_truth,
            "evaluated_oof_sha256": sha256_file(evaluated_path),
            "outer_lock_sha256": sha256_file(outer_lock),
            "global_exposure_lock_sha256": sha256_file(GLOBAL_EXPOSURE_LOCK),
            "global_exposure_ledger_sha256": sha256_file(GLOBAL_EXPOSURE_LEDGER),
            "decision": metrics["gate"]["decision"],
            "competition_upload": False,
        }
        receipt_path = run_dir / "receipt.json"
        _atomic_json_fsync(receipt_path, receipt)
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
            "inputs": {
                "train_sha256": train.attrs["source_sha256"],
                "test_sha256": test.attrs["source_sha256"],
                "incumbent_oof_sha256": sha256_file(reference_path),
                "archive_sha256": sha256_file(archive),
            },
            "artifacts": {
                value.name: {
                    "bytes": value.stat().st_size,
                    "sha256": sha256_file(value),
                }
                for value in sorted(run_dir.iterdir())
                if value.is_file()
            },
            "result_sha256": receipt["result_sha256"],
            "decision": receipt["decision"],
            "git": _git_state(),
        }
        _atomic_json_fsync(run_dir / "manifest.json", manifest)
        status.update(
            100.0,
            "complete",
            f"{receipt['decision']} · result SHA {receipt['result_sha256'][:12]}",
            status="complete",
            outer_truth_accessed=True,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        status.update(
            0.0,
            "failed",
            f"{type(exc).__name__}: {exc}",
            status="failed",
            outer_truth_accessed=outer_truth_accessed,
        )
        if run_dir is not None:
            _atomic_json_fsync(
                run_dir / "failure.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "failed_at": datetime.now().astimezone().isoformat(),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "outer_truth_accessed": outer_truth_accessed,
                    "outer_lock_exists": outer_lock.exists(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
