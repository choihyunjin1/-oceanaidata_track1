"""Run the fixed low-confidence P3 KMA calibrated secondary inference once."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.data import audit_p3_data, load_p3_data  # noqa: E402
from p3_wave.kma_calibrated_deployment import (  # noqa: E402
    DEPLOYMENT_ALPHA,
    EXPECTED_CASES,
    EXPECTED_FULL_TRAIN_ANCHORS,
    EXPECTED_GENERATED_META,
    EXPECTED_REUSED_META,
    EXPERIMENT_ID,
    KEY_COLUMNS,
    SUBMISSION_COLUMNS,
    KMADeploymentError,
    build_full_ridge_frame,
    build_test_source_features,
    calibrators_from_payload,
    calibrators_to_payload,
    combine_full_training_meta,
    count_byte_exact_noop_lines,
    feature_columns_sha256,
    load_deployment_config,
    render_submission_preserving_noop_lines,
    sha256_file,
    validate_candidate_submission,
)
from p3_wave.kma_calibrated_longlead_blend import (  # noqa: E402
    ACTIVE_LEADS,
    NO_OP_LEADS,
    add_calibrated_source,
    blend_long_leads,
    fit_ridge_pair,
)
from p3_wave.kma_source_meta import (  # noqa: E402
    LEADS,
    META_COLUMNS,
    apply_source_median_imputer,
    build_target_source_features,
    compact_source_feature_columns,
    source_predictions_to_meta,
)

CANONICAL_CONFIG = ROOT / "configs/experiments/p3_kma_calibrated_longlead_deployment_v1.json"
HELPER = ROOT / "src/p3_wave/kma_calibrated_deployment.py"
TESTS = ROOT / "tests/test_p3_kma_calibrated_longlead_deployment_v1.py"
RUNNER = Path(__file__).resolve()


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    _atomic_bytes(path, body)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_exclusive_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "output": ROOT / artifacts["output_directory"],
        "submission_directory": ROOT / artifacts["submission_directory"],
        "submission": ROOT / artifacts["submission_path"],
        "status": ROOT / artifacts["status_path"],
        "attempt": ROOT / artifacts["attempt_lock"],
    }


def _status(
    config: dict[str, Any],
    *,
    state: str,
    phase: str,
    progress: int,
    detail: str,
    started: float | None,
    eta_minutes: int | None,
    result: dict[str, Any] | None = None,
) -> None:
    elapsed = None if started is None else round(time.perf_counter() - started, 3)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "updated_at": _now(),
        "state": state,
        "phase": phase,
        "progress_percent": int(progress),
        "elapsed_seconds": elapsed,
        "eta_minutes": eta_minutes,
        "detail": detail,
        "source_model_fit_count": 0,
        "ridge_model_fit_count": 0 if progress < 50 else 2,
        "test_context_inference_generation_count": 0 if progress < 60 else 1,
        "official_upload_count": 0,
        "raw_paths_or_secrets_recorded": False,
    }
    if result is not None:
        payload["result"] = result
    _atomic_json(_paths(config)["status"], payload)


def _implementation_hashes() -> dict[str, str]:
    files = {
        "config": CANONICAL_CONFIG,
        "deployment_helper": HELPER,
        "runner": RUNNER,
        "tests": TESTS,
        "sealed_source_helper": ROOT / "src/p3_wave/kma_source_meta.py",
        "sealed_calibration_helper": ROOT / "src/p3_wave/kma_calibrated_longlead_blend.py",
        "p3_data_helper": ROOT / "src/p3_wave/data.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"implementation files are missing: {missing}")
    return {name: sha256_file(path) for name, path in files.items()}


def _registered_repo_inputs(config: dict[str, Any]) -> dict[str, str]:
    specifications: dict[str, dict[str, Any]] = {
        "validation_v2_result": config["validation_evidence"]["result"],
        "source_model": config["sealed_source_reuse"]["source_model"],
        "source_feature_medians": config["sealed_source_reuse"]["source_feature_medians"],
        "source_meta_predictions": config["sealed_source_reuse"]["source_meta_predictions"],
        "source_meta_seal": config["sealed_source_reuse"]["source_meta_seal"],
        "source_helper": config["sealed_source_reuse"]["source_helper"],
        "train_anchors": config["frozen_inputs"]["train_anchors"],
        "incumbent_submission": config["frozen_inputs"]["incumbent_submission"],
        "calibration_helper": config["frozen_inputs"]["calibration_helper"],
        "external_manifest": config["external_data_attribution"]["manifest"],
    }
    receipts: dict[str, str] = {}
    for name, specification in specifications.items():
        path = (ROOT / specification["path"]).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise KMADeploymentError(f"registered path escapes repository: {name}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != specification["sha256"]:
            raise KMADeploymentError(f"registered input hash changed: {name}")
        receipts[name] = actual
    return receipts


def _registered_p3_inputs(config: dict[str, Any], data_dir: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for filename, expected in config["p3_data_sha256"].items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise KMADeploymentError(f"P3 input hash changed: {filename}")
        receipts[filename] = actual
    return receipts


def _verify_evidence(config: dict[str, Any]) -> dict[str, Any]:
    seal_path = ROOT / config["sealed_source_reuse"]["source_meta_seal"]["path"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = config["sealed_source_reuse"]
    if seal.get("sealed") is not True:
        raise KMADeploymentError("v1 source meta is not sealed")
    for key, field in (
        ("source_model_sha256", "source_model"),
        ("source_meta_predictions_sha256", "source_meta_predictions"),
        ("source_feature_medians_sha256", "source_feature_medians"),
    ):
        if seal.get(key) != expected[field]["sha256"]:
            raise KMADeploymentError(f"source seal differs for {field}")
    if seal.get("p3_target_columns_read_before_source_seal") != 0:
        raise KMADeploymentError("source seal reports P3 target exposure")
    result_path = ROOT / config["validation_evidence"]["result"]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("decision") != config["validation_evidence"]["required_decision"]:
        raise KMADeploymentError("v2 result decision changed")
    promotion = result.get("outer_promotion", {})
    expected_delta = config["validation_evidence"]["outer_candidate_minus_incumbent_rmse_m"]
    if not np.isclose(
        float(promotion.get("candidate_minus_incumbent_rmse")),
        float(expected_delta),
        rtol=0.0,
        atol=1e-15,
    ):
        raise KMADeploymentError("v2 local improvement evidence changed")
    ci = promotion.get("paired_case_bootstrap", {})
    expected_ci = config["validation_evidence"]["outer_bootstrap_ci90"]
    actual_ci = [float(ci.get("ci90_lower")), float(ci.get("ci90_upper"))]
    if actual_ci != [float(value) for value in expected_ci]:
        raise KMADeploymentError("v2 bootstrap caveat changed")
    external_manifest_path = ROOT / config["external_data_attribution"]["manifest"]["path"]
    external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    attribution = config["external_data_attribution"]
    if external_manifest.get("source_id") != attribution["source_id"]:
        raise KMADeploymentError("external source id changed")
    if external_manifest.get("license_name") != attribution["license"]:
        raise KMADeploymentError("external license changed")
    if external_manifest.get("official_sources") != attribution["official_urls"]:
        raise KMADeploymentError("external official URL list changed")
    if external_manifest.get("observed_end") != attribution["observed_end_kst"]:
        raise KMADeploymentError("external cutoff changed")
    return {
        "source_seal_verified": True,
        "v2_no_go_caveat_verified": True,
        "external_attribution_verified": True,
    }


def _ensure_clean_boundary(config: dict[str, Any]) -> None:
    paths = _paths(config)
    if paths["attempt"].exists():
        raise FileExistsError("deployment attempt already exists; rerun is prohibited")
    if paths["output"].exists():
        raise FileExistsError("deployment artifact output already exists")
    if paths["submission_directory"].exists() or paths["submission"].exists():
        raise FileExistsError("secondary submission output already exists")
    incumbent = ROOT / config["frozen_inputs"]["incumbent_submission"]["path"]
    if paths["submission"].resolve() == incumbent.resolve():
        raise KMADeploymentError("secondary output aliases the incumbent")


def _load_source_model(config: dict[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(ROOT / config["sealed_source_reuse"]["source_model"]["path"])
    expected_columns = list(compact_source_feature_columns())
    if list(model.feature_names_) != expected_columns:
        raise KMADeploymentError("sealed source model feature schema changed")
    if int(model.tree_count_) != 1200 or model.get_param("loss_function") != "MultiRMSE":
        raise KMADeploymentError("sealed source model structure changed")
    return model


def _load_source_medians(config: dict[str, Any]) -> pd.Series:
    path = ROOT / config["sealed_source_reuse"]["source_feature_medians"]["path"]
    values = json.loads(path.read_text(encoding="utf-8"))
    medians = pd.Series(values, dtype="float64")
    expected = list(compact_source_feature_columns())
    if list(medians.index) != expected or len(medians) != 447:
        raise KMADeploymentError("sealed source median schema changed")
    if not np.isfinite(medians.to_numpy(dtype=np.float64)).all():
        raise KMADeploymentError("sealed source medians are non-finite")
    return medians


def _generated_missing_training_meta(
    *,
    config: dict[str, Any],
    data: Any,
    anchors: pd.DataFrame,
    reused: pd.DataFrame,
    source_model: Any,
    medians: pd.Series,
    started: float,
) -> pd.DataFrame:
    missing_ids = np.setdiff1d(
        anchors["anchor_id"].to_numpy(dtype=np.int64),
        reused["anchor_id"].to_numpy(dtype=np.int64),
        assume_unique=True,
    )
    if len(missing_ids) != EXPECTED_GENERATED_META:
        raise KMADeploymentError("unexpected number of training anchors needs source inference")
    missing = anchors.loc[
        anchors["anchor_id"].isin(missing_ids),
        ["anchor_id", "station", "anchor_time", "current_hs"],
    ].reset_index(drop=True)

    def progress(done: int, total: int) -> None:
        percent = 8 + int(30 * done / total)
        _status(
            config,
            state="running",
            phase="complete_full_train_source_meta",
            progress=percent,
            detail=f"missing public-train source predictions {done}/{total}",
            started=started,
            eta_minutes=max(2, int((total - done) / 250)),
        )

    features = build_target_source_features(data.wave, data.atmos, missing, progress=progress)
    if not features["anchor_id"].equals(missing["anchor_id"]):
        raise KMADeploymentError("generated training feature order changed")
    matrix = features.loc[:, list(compact_source_feature_columns())]
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = apply_source_median_imputer(matrix, medians)
    residual = np.asarray(source_model.predict(matrix), dtype=np.float64)
    generated = source_predictions_to_meta(
        residual,
        anchor_ids=missing["anchor_id"].to_numpy(dtype=np.int64),
        current_hs=missing["current_hs"].to_numpy(dtype=np.float64),
    )
    return generated


def _predict_test_source(
    *,
    config: dict[str, Any],
    context: pd.DataFrame,
    source_model: Any,
    medians: pd.Series,
    started: float,
) -> tuple[pd.DataFrame, set[str]]:
    def progress(done: int, total: int) -> None:
        _status(
            config,
            state="running",
            phase="same_case_relative_test_source_inference",
            progress=60 + int(15 * done / total),
            detail=f"anonymous relative contexts {done}/{total}; no absolute-time mapping",
            started=started,
            eta_minutes=max(1, int((total - done) / 50)),
        )

    features = build_test_source_features(context, progress=progress)
    columns = list(compact_source_feature_columns())
    raw = features.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    imputed = apply_source_median_imputer(raw, medians)
    post_impute_finite = np.isfinite(imputed.to_numpy(dtype=np.float64)).all(axis=1)
    supported = features["source_supported"].to_numpy(dtype=bool) & post_impute_finite
    residual = np.asarray(source_model.predict(imputed), dtype=np.float64)
    if residual.shape != (EXPECTED_CASES, len(LEADS)):
        raise KMADeploymentError("sealed source test prediction shape changed")
    prediction_finite = np.isfinite(residual).all(axis=1)
    supported &= prediction_finite
    safe_residual = np.where(np.isfinite(residual), residual, 0.0)
    current = features["current_hs"].to_numpy(dtype=np.float64)
    absolute = np.clip(current[:, None] + safe_residual, 0.0, 30.0)
    wide = features.loc[:, ["case_id", "station", "current_hs", "source_supported"]].copy()
    wide["source_supported"] = supported
    for position, column in enumerate(META_COLUMNS):
        wide[column] = absolute[:, position]
    supported_cases = set(wide.loc[wide["source_supported"], "case_id"].astype(str))
    return wide, supported_cases


def _test_source_long(test_index: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    merged = test_index.merge(
        wide,
        on=["case_id", "station"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(merged) != len(test_index) or merged["current_hs"].isna().any():
        raise KMADeploymentError("test source predictions lack an official key")
    lookup = {lead: f"kma_source_hs_pred_{lead}h" for lead in LEADS}
    source = np.empty(len(merged), dtype=np.float64)
    for lead, column in lookup.items():
        mask = merged["lead_h"].eq(lead).to_numpy()
        source[mask] = merged.loc[mask, column].to_numpy(dtype=np.float64)
    result = merged.loc[:, ["case_id", "station", "lead_h", "current_hs"]].copy()
    result["source_prediction"] = source
    result["source_supported"] = merged["source_supported"].to_numpy(dtype=bool)
    return result


def _candidate_from_incumbent(
    *,
    incumbent: pd.DataFrame,
    test_index: pd.DataFrame,
    source_long: pd.DataFrame,
    calibrators: dict[int, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not incumbent[list(KEY_COLUMNS)].equals(test_index):
        raise KMADeploymentError("incumbent does not match test-index keys/order")
    calibrated = add_calibrated_source(source_long, calibrators)
    if not calibrated[list(KEY_COLUMNS)].equals(test_index):
        raise KMADeploymentError("calibrated source order changed")
    base = incumbent["hs_pred"].to_numpy(dtype=np.float64)
    candidate_values = blend_long_leads(
        base,
        calibrated["calibrated_source"].to_numpy(dtype=np.float64),
        calibrated["lead_h"].to_numpy(dtype=np.int64),
        alpha=DEPLOYMENT_ALPHA,
    )
    unsupported = ~calibrated["source_supported"].to_numpy(dtype=bool)
    candidate_values[unsupported] = base[unsupported]
    candidate = test_index.copy()
    candidate["hs_pred"] = candidate_values
    detail = calibrated.copy()
    detail["incumbent_hs_pred"] = base
    detail["candidate_hs_pred"] = candidate_values
    detail["deployment_alpha"] = np.where(
        np.isin(detail["lead_h"].to_numpy(dtype=np.int64), ACTIVE_LEADS) & ~unsupported,
        DEPLOYMENT_ALPHA,
        0.0,
    )
    return candidate.loc[:, list(SUBMISSION_COLUMNS)], detail


def _runtime_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "catboost", "pyarrow")
    result: dict[str, str] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def _preflight(config: dict[str, Any], *, data_dir: Path | None, started: float) -> int:
    _ensure_clean_boundary(config)
    repo_inputs = _registered_repo_inputs(config)
    evidence = _verify_evidence(config)
    p3_inputs = None if data_dir is None else _registered_p3_inputs(config, data_dir)
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "ready": data_dir is not None,
        "registered_repo_input_sha256": repo_inputs,
        "registered_p3_input_sha256": p3_inputs,
        "implementation_sha256": _implementation_hashes(),
        "feature_columns_sha256": feature_columns_sha256(),
        "evidence": evidence,
        "model_fit_count": 0,
        "test_context_value_read_count": 0,
        "submission_write_count": 0,
        "upload_count": 0,
    }
    _status(
        config,
        state="preflight_ready" if data_dir is not None else "preflight_needs_data_dir",
        phase="hash_and_contract_preflight",
        progress=0,
        detail="contracts and hashes verified; model/test inference not started",
        started=started,
        eta_minutes=35 if data_dir is not None else None,
        result={"ready": receipt["ready"], "input_hash_mismatch_count": 0},
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def _run(config: dict[str, Any], *, data_dir: Path, started: float) -> int:
    _ensure_clean_boundary(config)
    repo_inputs = _registered_repo_inputs(config)
    p3_inputs = _registered_p3_inputs(config, data_dir)
    evidence = _verify_evidence(config)
    implementation = _implementation_hashes()
    paths = _paths(config)
    attempt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "run_once": True,
        "official_upload_authorized": False,
        "registered_repo_input_sha256": repo_inputs,
        "registered_p3_input_sha256": p3_inputs,
        "implementation_sha256": implementation,
    }
    _write_exclusive_json(paths["attempt"], attempt)
    paths["output"].mkdir(parents=True, exist_ok=False)
    paths["submission_directory"].mkdir(parents=True, exist_ok=False)
    _status(
        config,
        state="running",
        phase="load_and_audit_public_inputs",
        progress=2,
        detail="run-once lock created; loading immutable P3 inputs",
        started=started,
        eta_minutes=40,
    )
    data = load_p3_data(data_dir)
    audit = audit_p3_data(data)
    if audit["cases"] != EXPECTED_CASES:
        raise KMADeploymentError("P3 structural audit case count changed")
    anchors_path = ROOT / config["frozen_inputs"]["train_anchors"]["path"]
    anchors = pd.read_parquet(anchors_path)
    if len(anchors) != EXPECTED_FULL_TRAIN_ANCHORS:
        raise KMADeploymentError("public training anchor count changed")
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    reused_path = ROOT / config["sealed_source_reuse"]["source_meta_predictions"]["path"]
    reused = pd.read_parquet(reused_path)
    if len(reused) != EXPECTED_REUSED_META:
        raise KMADeploymentError("sealed source-meta count changed")
    source_model = _load_source_model(config)
    medians = _load_source_medians(config)
    generated = _generated_missing_training_meta(
        config=config,
        data=data,
        anchors=anchors,
        reused=reused,
        source_model=source_model,
        medians=medians,
        started=started,
    )
    full_meta = combine_full_training_meta(anchors, reused, generated)
    generated_path = paths["output"] / "generated_missing_train_source_meta.parquet"
    full_meta_path = paths["output"] / "full_train_source_meta.parquet"
    _atomic_parquet(generated_path, generated)
    _atomic_parquet(full_meta_path, full_meta)
    _status(
        config,
        state="running",
        phase="fit_two_fixed_full_train_ridges",
        progress=45,
        detail="fitting exactly one Ridge for 18h and one Ridge for 24h; no search",
        started=started,
        eta_minutes=15,
    )
    ridge_frame = build_full_ridge_frame(anchors, full_meta)
    calibrators = fit_ridge_pair(ridge_frame)
    model_payload = calibrators_to_payload(calibrators)
    model_payload["created_at"] = _now()
    model_payload["deployment_alpha"] = DEPLOYMENT_ALPHA
    model_payload["fit_anchor_count"] = EXPECTED_FULL_TRAIN_ANCHORS
    model_payload["source_model_fit_count"] = 0
    model_payload["ridge_model_fit_count"] = 2
    model_path = paths["output"] / "ridge_models.json"
    _atomic_json(model_path, model_payload)
    reloaded_payload = json.loads(model_path.read_text(encoding="utf-8"))
    reloaded_calibrators = calibrators_from_payload(reloaded_payload)
    original_train_calibrated = add_calibrated_source(ridge_frame, calibrators)[
        "calibrated_source"
    ].to_numpy(dtype=np.float64)
    reloaded_train_calibrated = add_calibrated_source(ridge_frame, reloaded_calibrators)[
        "calibrated_source"
    ].to_numpy(dtype=np.float64)
    if not np.array_equal(original_train_calibrated, reloaded_train_calibrated):
        raise KMADeploymentError("saved Ridge models do not round-trip exactly")

    source_test, supported_cases = _predict_test_source(
        config=config,
        context=data.test_context,
        source_model=source_model,
        medians=medians,
        started=started,
    )
    source_test_path = paths["output"] / "test_source_predictions.parquet"
    _atomic_parquet(source_test_path, source_test)
    source_long = _test_source_long(data.test_index, source_test)
    incumbent_path = ROOT / config["frozen_inputs"]["incumbent_submission"]["path"]
    incumbent = pd.read_csv(incumbent_path, float_precision="round_trip")
    candidate, inference_detail = _candidate_from_incumbent(
        incumbent=incumbent,
        test_index=data.test_index,
        source_long=source_long,
        calibrators=reloaded_calibrators,
    )
    validation = validate_candidate_submission(
        candidate,
        data.test_index,
        incumbent,
        supported_cases=supported_cases,
    )
    detail_path = paths["output"] / "test_inference_detail.parquet"
    _atomic_parquet(detail_path, inference_detail)
    incumbent_bytes = incumbent_path.read_bytes()
    rendered = render_submission_preserving_noop_lines(
        incumbent_bytes,
        candidate,
        supported_cases=supported_cases,
    )
    _atomic_bytes(paths["submission"], rendered)
    reloaded_candidate = pd.read_csv(paths["submission"], float_precision="round_trip")
    validation_reloaded = validate_candidate_submission(
        reloaded_candidate,
        data.test_index,
        incumbent,
        supported_cases=supported_cases,
    )
    if not np.array_equal(
        candidate["hs_pred"].to_numpy(dtype=np.float64),
        reloaded_candidate["hs_pred"].to_numpy(dtype=np.float64),
    ):
        raise KMADeploymentError("candidate CSV prediction values changed after round-trip")
    byte_exact_noop_lines = count_byte_exact_noop_lines(
        incumbent_bytes,
        rendered,
        candidate,
        supported_cases=supported_cases,
    )
    validation.update(
        {
            "reloaded": validation_reloaded,
            "byte_exact_noop_csv_lines": int(byte_exact_noop_lines),
            "short_lead_numeric_exact": True,
            "saved_model_roundtrip_exact": True,
            "schema_and_order_exact": True,
        }
    )
    validation_path = paths["output"] / "validation.json"
    _atomic_json(validation_path, validation)

    if _implementation_hashes() != implementation:
        raise KMADeploymentError("implementation changed during inference")
    if _registered_repo_inputs(config) != repo_inputs:
        raise KMADeploymentError("registered repository input changed during inference")
    if _registered_p3_inputs(config, data_dir) != p3_inputs:
        raise KMADeploymentError("registered P3 input changed during inference")
    outputs = {
        "submission": sha256_file(paths["submission"]),
        "ridge_models": sha256_file(model_path),
        "generated_missing_train_source_meta": sha256_file(generated_path),
        "full_train_source_meta": sha256_file(full_meta_path),
        "test_source_predictions": sha256_file(source_test_path),
        "test_inference_detail": sha256_file(detail_path),
        "validation": sha256_file(validation_path),
    }
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "classification": "low_confidence_secondary_research_candidate",
        "promotion_claimed": False,
        "official_upload_performed": False,
        "official_upload_allowed": False,
        "v2_validation_caveat": {
            "decision": config["validation_evidence"]["required_decision"],
            "candidate_minus_incumbent_rmse_m": config["validation_evidence"][
                "outer_candidate_minus_incumbent_rmse_m"
            ],
            "paired_case_bootstrap_ci90": config["validation_evidence"]["outer_bootstrap_ci90"],
            "reason_not_promoted": config["validation_evidence"]["reason_not_promoted"],
        },
        "deployment_contract": {
            "single_refit_generation": True,
            "model_count": 2,
            "one_per_active_lead": True,
            "active_leads": list(ACTIVE_LEADS),
            "ridge_alpha": 10.0,
            "deployment_alpha": DEPLOYMENT_ALPHA,
            "no_op_leads": list(NO_OP_LEADS),
            "same_case_relative_48h_context_only": True,
            "absolute_timestamp_reconstruction": False,
            "external_test_join": False,
            "unsupported_case_policy": "exact_incumbent_no_op_on_all_leads",
        },
        "model_fit_counts": {"source_catboost": 0, "ridge": 2},
        "training_source_meta": {
            "full_anchor_count": EXPECTED_FULL_TRAIN_ANCHORS,
            "sealed_reused_count": EXPECTED_REUSED_META,
            "new_inference_count": EXPECTED_GENERATED_META,
        },
        "validation": validation,
        "input_sha256": {"repository": repo_inputs, "p3_data": p3_inputs},
        "implementation_sha256": implementation,
        "output_sha256": outputs,
        "attempt_lock_sha256": sha256_file(paths["attempt"]),
        "runtime_versions": _runtime_versions(),
        "external_data_attribution": config["external_data_attribution"],
        "source_period_WP_and_target_tp_used": False,
        "raw_input_copied_to_outputs": False,
        "submission_uploaded": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "evidence_checks": evidence,
    }
    manifest_path = paths["output"] / "manifest.json"
    _atomic_json(manifest_path, manifest)
    result = {
        "decision": "SECONDARY_RESEARCH_CANDIDATE_GENERATED_NOT_PROMOTED_NOT_UPLOADED",
        "submission_sha256": outputs["submission"],
        "manifest_sha256": sha256_file(manifest_path),
        "supported_cases": validation["supported_cases"],
        "unsupported_cases": validation["unsupported_cases"],
        "modified_active_rows": validation["modified_active_rows"],
        "byte_exact_noop_csv_lines": validation["byte_exact_noop_csv_lines"],
        "ridge_model_count": 2,
        "source_model_fit_count": 0,
        "deployment_alpha": DEPLOYMENT_ALPHA,
        "official_upload_performed": False,
        "run_once_lock_retained": True,
    }
    result_path = paths["output"] / "result.json"
    _atomic_json(result_path, result)
    _status(
        config,
        state="complete",
        phase="secondary_inference_validated",
        progress=100,
        detail="secondary candidate generated and validated; not promoted and not uploaded",
        started=started,
        eta_minutes=None,
        result={**result, "result_sha256": sha256_file(result_path)},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--p3-data-dir")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    config = load_deployment_config(CANONICAL_CONFIG)
    data_dir = Path(args.p3_data_dir).resolve() if args.p3_data_dir else None
    if data_dir is not None and not data_dir.is_dir():
        raise FileNotFoundError("P3 data directory does not exist")
    if args.mode == "preflight":
        return _preflight(config, data_dir=data_dir, started=started)
    if data_dir is None:
        raise ValueError("run mode requires --p3-data-dir")
    return _run(config, data_dir=data_dir, started=started)


if __name__ == "__main__":
    raise SystemExit(main())
