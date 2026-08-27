"""Canonical one-shot runner for P3 calibrated KMA long-lead blend v2.

The committed configuration permits dry-run only. Authorization and actual
execution are separate O_EXCL transitions. Dry-run never reads target values,
fits a model, touches test context, or writes a submission.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as pyarrow_parquet

from p3_wave.kma_calibrated_longlead_blend import (
    ACTIVE_LEADS,
    ALPHA_GRID,
    INNER_COLUMNS,
    LEADS,
    OUTER_COLUMNS,
    PAIR_KEYS,
    KMALongLeadError,
    RidgeAffineCalibrator,
    add_calibrated_source,
    apply_fixed_control_shrink,
    blend_long_leads,
    evaluate_inner_gate,
    evaluate_outer_promotion,
    fit_ridge_pair,
    load_preregistration,
    select_fold_alpha,
    sha256_file,
    validate_inner_predictions,
    validate_outer_blind,
)
from p3_wave.kma_source_meta import (
    catboost_frame,
    expand_prediction_rows,
    expand_target_rows,
    read_frozen_outer_key_membership,
    validate_outer_membership_against_anchors,
)
from p3_wave.models import threshold_case_weights
from p3_wave.revin_patch import (
    assign_storm_episodes_from_wave,
    build_episode_disjoint_folds_from_ids,
    build_inner_episode_split,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_kma_calibrated_longlead_blend_v2"
CANONICAL_CONFIG = (ROOT / "configs/experiments/p3_kma_calibrated_longlead_blend_v2.json").resolve()
CANONICAL_OUTPUT = (ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot").resolve()
CANONICAL_DRY_RECEIPT = (
    ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/dry_run/receipt.json"
).resolve()
CANONICAL_STATUS = (ROOT / "artifacts/status/p3_kma_calibrated_longlead_blend_v2.json").resolve()
CANONICAL_AUTHORIZATION = (
    ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/authorization_amendment.json"
).resolve()
CANONICAL_OUTER_LOCK = (CANONICAL_OUTPUT / "OUTER_TRUTH.lock").resolve()
GLOBAL_ATTEMPT_LOCK = (
    ROOT / "artifacts/experiment_locks/p3_kma_calibrated_longlead_blend_v2.attempt.lock"
).resolve()
GLOBAL_OUTER_LOCK = (
    ROOT / "artifacts/experiment_locks/p3_kma_calibrated_longlead_blend_v2.outer.lock"
).resolve()
GLOBAL_LEDGER = (ROOT / "artifacts/experiment_locks/p3_outer_truth_ledger.jsonl").resolve()
AUTHORIZATION_TOKEN = "ROOT_APPROVED_P3_KMA_CALIBRATED_LONGLEAD_BLEND_V2"


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


def _append_global_ledger(payload: Mapping[str, Any]) -> None:
    GLOBAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(GLOBAL_LEDGER, os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _status(
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    started: float,
    eta: str | None,
    result: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 KMA calibrated long-lead blend v2",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "eta": eta,
        "model_fit_count": 0 if state.startswith(("dry", "ready")) else None,
        "target_value_read_count": 0 if state.startswith(("dry", "ready")) else None,
        "outer_designated_scoring_open_count": 0
        if state.startswith(("dry", "ready", "actual_running_pre_outer"))
        else None,
        "updated_at": _now(),
    }
    if result is not None:
        payload["result"] = dict(result)
    _atomic_json(CANONICAL_STATUS, payload)


def _canonical_config(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != CANONICAL_CONFIG:
        raise PermissionError("config override is prohibited for the canonical one-shot runner")
    return resolved


def _implementation_hashes() -> dict[str, str]:
    paths = (
        "configs/experiments/p3_kma_calibrated_longlead_blend_v2.json",
        "src/p3_wave/kma_calibrated_longlead_blend.py",
        "scripts/run_p3_kma_calibrated_longlead_blend_v2.py",
        "tests/test_p3_kma_calibrated_longlead_blend_v2.py",
        "src/p3_wave/kma_source_meta.py",
        "src/p3_wave/revin_patch.py",
        "src/p3_wave/models.py",
    )
    hashes: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"implementation input is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _verify_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {role}")
    actual = sha256_file(path)
    if actual != expected:
        raise KMALongLeadError(f"registered SHA changed: {role}")
    return actual


def _verify_registered_inputs(
    config: Mapping[str, Any], *, p3_data_dir: Path | None
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for section, entries in (
        ("source_reuse", config["source_reuse"]),
        ("v1_provenance", config["v1_provenance"]),
        ("frozen_inputs", config["frozen_inputs"]),
    ):
        for name, item in entries.items():
            if not isinstance(item, Mapping) or "path" not in item or "sha256" not in item:
                continue
            role = f"{section}_{name}"
            hashes[role] = _verify_hash(ROOT / str(item["path"]), str(item["sha256"]), role)
    if p3_data_dir is not None:
        for filename, expected in config["p3_sources"].items():
            role = f"p3_{filename}"
            hashes[role] = _verify_hash(p3_data_dir / filename, str(expected), role)
    return hashes


def _verify_v1_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    result_path = ROOT / config["v1_provenance"]["one_shot_result"]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    requirement = config["v1_provenance"]["one_shot_result"]
    if result.get("decision") != requirement["required_decision"]:
        raise KMALongLeadError("v1 decision changed")
    if (
        result.get("designated_outer_scoring_open_count")
        != requirement["required_outer_scoring_open_count"]
    ):
        raise KMALongLeadError("v1 outer scoring state changed")
    diagnostics_path = ROOT / config["v1_provenance"]["inner_diagnostics"]["path"]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("outer_truth_opened") is not False:
        raise KMALongLeadError("v1 diagnostics unexpectedly opened outer truth")
    pooled = diagnostics["raw_source_meta_diagnostics"]["pooled"]
    evidence = config["objective"]
    checks = (
        np.isclose(
            pooled["source_minus_persistence_rmse"],
            evidence["v1_raw_source_minus_persistence_rmse_m"],
            rtol=0.0,
            atol=1e-15,
        ),
        np.isclose(
            pooled["residual_pearson"],
            evidence["v1_raw_source_residual_pearson"],
            rtol=0.0,
            atol=1e-15,
        ),
        np.isclose(
            pooled["calibration_intercept"],
            evidence["v1_calibration_intercept_m"],
            rtol=0.0,
            atol=1e-15,
        ),
        np.isclose(
            pooled["calibration_scale"],
            evidence["v1_calibration_scale"],
            rtol=0.0,
            atol=1e-15,
        ),
    )
    if not all(checks):
        raise KMALongLeadError("v1 diagnostic evidence changed")
    source_seal_path = ROOT / config["source_reuse"]["source_meta_seal"]["path"]
    source_seal = json.loads(source_seal_path.read_text(encoding="utf-8"))
    model_path = ROOT / config["source_reuse"]["source_model"]["path"]
    meta_path = ROOT / config["source_reuse"]["source_meta_predictions"]["path"]
    if source_seal.get("source_model_sha256") != sha256_file(model_path):
        raise KMALongLeadError("v1 source seal model binding changed")
    if source_seal.get("source_meta_predictions_sha256") != sha256_file(meta_path):
        raise KMALongLeadError("v1 source seal meta binding changed")
    if source_seal.get("p3_target_columns_read_before_source_seal") != 0:
        raise KMALongLeadError("v1 source meta was not target blind")
    return {
        "v1_decision": result["decision"],
        "v1_outer_designated_scoring_open_count": result["designated_outer_scoring_open_count"],
        "raw_source_minus_persistence_rmse": pooled["source_minus_persistence_rmse"],
        "raw_source_residual_pearson": pooled["residual_pearson"],
        "calibration_intercept": pooled["calibration_intercept"],
        "calibration_scale": pooled["calibration_scale"],
    }


def _read_source_meta(config: Mapping[str, Any]) -> pd.DataFrame:
    specification = config["source_reuse"]["source_meta_predictions"]
    path = ROOT / specification["path"]
    expected_columns = ["anchor_id", *[f"kma_source_hs_pred_{lead}h" for lead in LEADS]]
    frame = pd.read_parquet(path, columns=expected_columns)
    if list(frame.columns) != expected_columns:
        raise KMALongLeadError("sealed source meta schema changed")
    if len(frame) != int(specification["expected_rows"]):
        raise KMALongLeadError("sealed source meta row count changed")
    if frame["anchor_id"].duplicated().any():
        raise KMALongLeadError("sealed source meta keys are duplicated")
    values = frame.drop(columns="anchor_id").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 30.0).any():
        raise KMALongLeadError("sealed source meta values are invalid")
    return frame


def _read_incumbent_label_free(config: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / config["frozen_inputs"]["incumbent_oof"]["path"]
    columns = ["fold", "anchor_id", "station", "lead_h", "current_hs", "prediction"]
    frame = pd.read_parquet(path, columns=columns)
    if list(frame.columns) != columns or frame.duplicated(list(PAIR_KEYS)).any():
        raise KMALongLeadError("frozen incumbent label-free schema changed")
    specification = config["frozen_inputs"]["incumbent_oof"]
    if len(frame) != int(specification["expected_rows"]):
        raise KMALongLeadError("frozen incumbent row count changed")
    if frame["anchor_id"].nunique() != int(specification["expected_cases"]):
        raise KMALongLeadError("frozen incumbent case count changed")
    values = frame[["current_hs", "prediction"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (frame["prediction"] < 0.0).any():
        raise KMALongLeadError("frozen incumbent prediction is invalid")
    return frame.rename(columns={"prediction": "incumbent_final"})


def _synthetic_contract() -> dict[str, Any]:
    leads = np.asarray(LEADS, dtype=np.int64)
    control = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    source = np.asarray([9.0, 9.0, 9.0, 9.0, 1.2, 1.3])
    candidate = blend_long_leads(control, source, leads, alpha=0.2)
    if not np.array_equal(candidate[:4], control[:4]):
        raise AssertionError("synthetic short-lead no-op failed")
    calibrator = RidgeAffineCalibrator(
        lead_h=18,
        ridge_alpha=10.0,
        fit_intercept=False,
        solver="cholesky",
        design_columns=(
            "source_residual",
            "station_G-ORS",
            "station_I-ORS",
            "station_S-ORS",
        ),
        coefficients=(1.0, 0.0, 0.0, 0.0),
        fit_rows=3,
    )
    return {
        "alpha_grid": list(ALPHA_GRID),
        "short_lead_no_op_max_abs_difference": float(np.max(np.abs(candidate[:4] - control[:4]))),
        "calibrator_roundtrip": RidgeAffineCalibrator.from_dict(calibrator.to_dict()).to_dict(),
        "ridge_fit_count": 0,
        "catboost_fit_count": 0,
        "target_value_read_count": 0,
    }


def _dry_run(config: Mapping[str, Any], *, p3_data_dir: Path | None, started: float) -> int:
    _status(
        state="dry_running",
        phase="hash_schema_and_key_preflight",
        progress=20,
        detail="v1 seal·frozen input·schema 감사 중 · target/model/test 0",
        started=started,
        eta="약 1분",
    )
    if any(
        path.exists()
        for path in (
            CANONICAL_AUTHORIZATION,
            GLOBAL_ATTEMPT_LOCK,
            GLOBAL_OUTER_LOCK,
            CANONICAL_OUTER_LOCK,
            CANONICAL_OUTPUT,
        )
    ):
        raise PermissionError("v2 one-shot state already exists")
    receipts = _verify_registered_inputs(config, p3_data_dir=p3_data_dir)
    evidence = _verify_v1_evidence(config)
    source_meta = _read_source_meta(config)
    incumbent = _read_incumbent_label_free(config)
    feature_columns = json.loads(
        (ROOT / config["frozen_inputs"]["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    if len(feature_columns) != 591 or len(set(feature_columns)) != 591:
        raise KMALongLeadError("frozen 591-feature surface changed")
    anchor_path = ROOT / config["frozen_inputs"]["anchor_metadata_and_target_vault"]["path"]
    anchor_schema = pyarrow_parquet.read_schema(anchor_path)
    expected_anchor_columns = {
        "anchor_id",
        "station",
        "anchor_time",
        "grid_position",
        "current_hs",
        *[f"target_{lead}" for lead in LEADS],
    }
    if set(anchor_schema.names) != expected_anchor_columns:
        raise KMALongLeadError("target vault schema changed")
    fold_names = [str(row[0]) for row in config["validation"]["windows"]]
    frozen_keys, membership = read_frozen_outer_key_membership(
        ROOT / config["frozen_inputs"]["incumbent_oof"]["path"],
        expected_folds=fold_names,
    )
    if not incumbent[list(PAIR_KEYS)].equals(frozen_keys[list(PAIR_KEYS)]):
        raise KMALongLeadError("label-free incumbent rows differ from frozen membership")
    relevant_ids = np.unique(
        np.concatenate([values for values in membership.values()]).astype(np.int64)
    )
    if not np.isin(relevant_ids, source_meta["anchor_id"].to_numpy(dtype=np.int64)).all():
        raise KMALongLeadError("sealed source meta lacks an outer validation key")
    manifest = json.loads(
        (ROOT / config["frozen_inputs"]["frozen_model_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    proxy = config["inner_control_proxy"]
    for key in (
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "random_seed",
        "loss_function",
        "thread_count",
        "allow_writing_files",
    ):
        if manifest["parameters"]["single"].get(key) != proxy.get(key):
            raise KMALongLeadError(f"inner proxy differs from frozen single manifest: {key}")
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "status": "READY_PENDING_PARENT_ACTUAL_APPROVAL",
        "mode": "dry-run",
        "registered_input_sha256": receipts,
        "implementation_sha256": _implementation_hashes(),
        "v1_evidence": evidence,
        "schema_preflight": {
            "source_meta_rows": int(len(source_meta)),
            "source_meta_columns": int(len(source_meta.columns)),
            "target_feature_count": len(feature_columns),
            "target_vault_columns_seen_by_metadata_only": len(anchor_schema.names),
        },
        "key_preflight": {
            "outer_rows": int(len(frozen_keys)),
            "outer_cases": int(frozen_keys["anchor_id"].nunique()),
            "cases_by_fold": {name: int(len(values)) for name, values in membership.items()},
            "incumbent_columns_read": [
                "fold",
                "anchor_id",
                "station",
                "lead_h",
                "current_hs",
                "prediction",
            ],
            "incumbent_target_columns_read": 0,
        },
        "synthetic_contract": _synthetic_contract(),
        "one_shot_locks_created": 0,
        "one_shot_available": True,
        "model_fit_count": 0,
        "target_value_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(CANONICAL_DRY_RECEIPT, receipt)
    receipt_hash = sha256_file(CANONICAL_DRY_RECEIPT)
    _status(
        state="dry_ready",
        phase="prepared_waiting_parent_and_independent_qa",
        progress=100,
        detail="dry-run 완료 · model/target/outer/test/submission 0",
        started=started,
        eta=None,
        result={
            "decision": receipt["status"],
            "source_meta_rows": len(source_meta),
            "outer_cases": frozen_keys["anchor_id"].nunique(),
            "receipt_sha256": receipt_hash,
            "model_fit_count": 0,
            "target_value_read_count": 0,
            "outer_designated_scoring_open_count": 0,
        },
    )
    print(json.dumps({**receipt, "receipt_sha256": receipt_hash}, ensure_ascii=False, indent=2))
    return 0


@dataclass
class TargetVault:
    path: Path
    access_log: list[dict[str, Any]] = field(default_factory=list)
    designated_scoring_open_count: int = 0

    def _read(self, anchor_ids: Sequence[int]) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise KMALongLeadError("target request must contain unique anchor IDs")
        columns = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id").loc[ids].reset_index()
        if len(frame) != len(ids) or frame["anchor_id"].duplicated().any():
            raise KMALongLeadError("filtered target read is incomplete")
        if not np.isfinite(frame.drop(columns="anchor_id").to_numpy(dtype=np.float64)).all():
            raise KMALongLeadError("filtered target read contains non-finite values")
        return frame

    def read_training(
        self,
        anchor_ids: Sequence[int],
        *,
        forbidden_current_validation_ids: Sequence[int],
        all_outer_validation_ids: Sequence[int],
        allowed_prior_validation_ids: Sequence[int],
        fold: str,
        purpose: str,
    ) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        current_overlap = np.intersect1d(ids, forbidden_current_validation_ids)
        if current_overlap.size:
            raise PermissionError("current-fold validation target requested before blind seal")
        outer_overlap = np.intersect1d(ids, all_outer_validation_ids)
        unexpected = np.setdiff1d(outer_overlap, allowed_prior_validation_ids)
        if unexpected.size:
            raise PermissionError("future-fold validation target requested as training history")
        frame = self._read(ids)
        self.access_log.append(
            {
                "purpose": purpose,
                "fold": fold,
                "rows": int(len(frame)),
                "current_validation_overlap_rows": 0,
                "permitted_prior_validation_history_rows": int(len(outer_overlap)),
            }
        )
        return frame

    def open_designated_scoring_once(
        self,
        anchor_ids: Sequence[int],
        *,
        blind_manifest: Path,
        exposure_receipt: Path,
    ) -> pd.DataFrame:
        if self.designated_scoring_open_count:
            raise PermissionError("designated outer scoring may open only once")
        manifest = json.loads(blind_manifest.read_text(encoding="utf-8"))
        receipt = json.loads(exposure_receipt.read_text(encoding="utf-8"))
        if (
            manifest.get("sealed") is not True
            or manifest.get("designated_outer_scoring_read_performed") is not False
        ):
            raise PermissionError("blind manifest is not sealed pre-scoring")
        if receipt.get("blind_manifest_sha256") != sha256_file(blind_manifest):
            raise PermissionError("exposure receipt is not bound to the blind manifest")
        if receipt.get("fsync_completed_before_designated_scoring") is not True:
            raise PermissionError("blind fsync assertion is absent")
        frame = self._read(anchor_ids)
        self.designated_scoring_open_count += 1
        self.access_log.append(
            {"purpose": "designated_outer_scoring_after_global_blind_seal", "rows": len(frame)}
        )
        return frame


def _read_p3_train(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    atmos = pd.read_csv(data_dir / "train_atmos.csv")
    for frame in (wave, atmos):
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if wave.duplicated(["station", "time"]).any() or atmos.duplicated(["station", "time"]).any():
        raise KMALongLeadError("P3 raw training keys are duplicated")
    return wave, atmos


def _expand_source_rows(
    source_meta: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    *,
    targets: pd.DataFrame | None,
) -> pd.DataFrame:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    meta_lookup = source_meta.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    if not np.isin(ids, meta_lookup.index.to_numpy(dtype=np.int64)).all():
        raise KMALongLeadError("sealed source meta lacks a requested anchor")
    target_lookup = None if targets is None else targets.set_index("anchor_id")
    rows: list[pd.DataFrame] = []
    for lead in LEADS:
        block = pd.DataFrame(
            {
                "anchor_id": ids,
                "station": anchor_lookup.loc[ids, "station"].astype(str).to_numpy(),
                "lead_h": lead,
                "current_hs": anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64),
                "source_prediction": meta_lookup.loc[ids, f"kma_source_hs_pred_{lead}h"].to_numpy(
                    dtype=np.float64
                ),
            }
        )
        if target_lookup is not None:
            block["target_hs"] = target_lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=np.float64)
        rows.append(block)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["anchor_id", "lead_h"]).any():
        raise KMALongLeadError("expanded source rows are duplicated")
    return result


def _control_proxy_model(config: Mapping[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    parameters = config["inner_control_proxy"]
    return CatBoostRegressor(
        iterations=int(parameters["iterations"]),
        learning_rate=float(parameters["learning_rate"]),
        depth=int(parameters["depth"]),
        l2_leaf_reg=float(parameters["l2_leaf_reg"]),
        random_strength=float(parameters["random_strength"]),
        random_seed=int(parameters["random_seed"]),
        loss_function=str(parameters["loss_function"]),
        eval_metric="RMSE",
        task_type=str(parameters["task_type"]),
        thread_count=int(parameters["thread_count"]),
        allow_writing_files=bool(parameters["allow_writing_files"]),
        verbose=False,
    )


def _fit_control_proxy(
    *,
    config: Mapping[str, Any],
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    fit_targets: pd.DataFrame,
    fit_ids: Sequence[int],
    calibration_ids: Sequence[int],
    feature_columns: Sequence[str],
    model_path: Path,
) -> pd.DataFrame:
    x_fit, y_fit, fit_meta = expand_target_rows(
        features, anchors, fit_targets, fit_ids, feature_columns
    )
    x_calibration, calibration_meta = expand_prediction_rows(
        features, anchors, calibration_ids, feature_columns
    )
    model = _control_proxy_model(config)
    model.fit(
        catboost_frame(x_fit),
        y_fit,
        sample_weight=threshold_case_weights(fit_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    model.save_model(model_path)
    residual = np.asarray(model.predict(catboost_frame(x_calibration)), dtype=np.float64)
    prediction = np.clip(
        calibration_meta["current_hs"].to_numpy(dtype=np.float64) + residual,
        0.0,
        30.0,
    )
    result = calibration_meta.copy()
    result["control_single_prediction"] = prediction
    result["control_final"] = apply_fixed_control_shrink(
        prediction,
        result["current_hs"].to_numpy(dtype=np.float64),
        result["lead_h"].to_numpy(dtype=np.int64),
    )
    return result


def _assemble_inner_rows(
    *,
    fold_name: str,
    source_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["anchor_id", "station", "lead_h", "current_hs"]
    merged = source_rows.merge(
        control_rows,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(source_rows) or len(merged) != len(control_rows):
        raise KMALongLeadError("source calibration and control proxy rows differ")
    merged.insert(0, "fold", fold_name)
    return merged


def _coefficient_payload(
    calibrators: Mapping[int, RidgeAffineCalibrator],
) -> dict[str, Any]:
    if set(calibrators) != set(ACTIVE_LEADS):
        raise KMALongLeadError("coefficient payload requires both long leads")
    return {str(lead): calibrators[lead].to_dict() for lead in ACTIVE_LEADS}


def _targets_to_long(targets: pd.DataFrame, blind: pd.DataFrame) -> pd.DataFrame:
    lookup = blind[["fold", "anchor_id", "station"]].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for lead in LEADS:
        block = targets[["anchor_id", f"target_{lead}"]].rename(
            columns={f"target_{lead}": "target_hs"}
        )
        block["lead_h"] = lead
        rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    return lookup.merge(long, on="anchor_id", how="inner", validate="one_to_many")


def _preouter_no_go_result(
    *,
    inner_gate: Mapping[str, Any],
    inner_manifest: Path,
    attempt_lock: Path,
) -> dict[str, Any]:
    if not attempt_lock.is_file():
        raise PermissionError("pre-outer stop is missing the consumed attempt lock")
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "decision": "NO_GO_INNER_CALIBRATED_BLEND",
        "inner_gate": dict(inner_gate),
        "inner_manifest_sha256": sha256_file(inner_manifest),
        "outer_predictions_written": 0,
        "designated_outer_scoring_open_count": 0,
        "one_shot_locks_created": 1,
        "global_attempt_lock_created": True,
        "global_attempt_lock_sha256": sha256_file(attempt_lock),
        "outer_truth_locks_created": 0,
        "rerun_prohibited": True,
    }


def _authorize_actual(*, authorization_token: str | None, started: float) -> int:
    if authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("authorization amendment requires the exact parent token")
    if (
        CANONICAL_AUTHORIZATION.exists()
        or GLOBAL_ATTEMPT_LOCK.exists()
        or CANONICAL_OUTPUT.exists()
    ):
        raise PermissionError("authorization or actual attempt already exists")
    if not CANONICAL_DRY_RECEIPT.is_file():
        raise FileNotFoundError("canonical dry-run receipt is missing")
    receipt = json.loads(CANONICAL_DRY_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("status") != "READY_PENDING_PARENT_ACTUAL_APPROVAL":
        raise PermissionError("dry-run receipt is not approval-ready")
    implementation = _implementation_hashes()
    if receipt.get("implementation_sha256") != implementation:
        raise PermissionError("implementation differs from reviewed dry-run")
    amendment = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "authorized": True,
        "authorized_at": _now(),
        "dry_receipt_sha256": sha256_file(CANONICAL_DRY_RECEIPT),
        "implementation_sha256": implementation,
        "registered_input_sha256": receipt["registered_input_sha256"],
        "config_sha256": sha256_file(CANONICAL_CONFIG),
        "outer_designated_scoring_open_count": 0,
    }
    _write_exclusive(CANONICAL_AUTHORIZATION, amendment)
    _status(
        state="ready_authorized",
        phase="actual_authorization_sealed",
        progress=100,
        detail="exact dry receipt·implementation SHA에 actual 승인 봉인 · model/target/outer 0",
        started=started,
        eta=None,
        result={
            "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
            "dry_receipt_sha256": amendment["dry_receipt_sha256"],
        },
    )
    return 0


def _load_authorization() -> dict[str, Any]:
    if not CANONICAL_AUTHORIZATION.is_file():
        raise PermissionError("canonical actual authorization amendment is missing")
    amendment = json.loads(CANONICAL_AUTHORIZATION.read_text(encoding="utf-8"))
    if amendment.get("authorized") is not True or amendment.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("actual authorization amendment is invalid")
    if amendment.get("dry_receipt_sha256") != sha256_file(CANONICAL_DRY_RECEIPT):
        raise PermissionError("dry-run receipt changed after authorization")
    if amendment.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("implementation changed after authorization")
    if amendment.get("config_sha256") != sha256_file(CANONICAL_CONFIG):
        raise PermissionError("canonical config changed after authorization")
    return amendment


def _actual(
    config: Mapping[str, Any],
    *,
    p3_data_dir: Path,
    authorization_token: str | None,
    started: float,
) -> int:
    if authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("actual execution requires the exact parent authorization token")
    canonical = load_preregistration(CANONICAL_CONFIG)
    if dict(config) != canonical:
        raise PermissionError("in-memory config differs from canonical preregistration")
    config = canonical
    authorization = _load_authorization()
    if any(
        path.exists()
        for path in (GLOBAL_ATTEMPT_LOCK, GLOBAL_OUTER_LOCK, CANONICAL_OUTER_LOCK, CANONICAL_OUTPUT)
    ):
        raise PermissionError("v2 one-shot boundary already exists")
    receipts = _verify_registered_inputs(config, p3_data_dir=p3_data_dir)
    if receipts != authorization["registered_input_sha256"]:
        raise PermissionError("registered inputs differ from authorized dry-run")
    _verify_v1_evidence(config)
    attempt = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "dry_receipt_sha256": authorization["dry_receipt_sha256"],
        "implementation_sha256": authorization["implementation_sha256"],
        "outer_designated_scoring_opened": False,
    }
    _write_exclusive(GLOBAL_ATTEMPT_LOCK, attempt)
    CANONICAL_OUTPUT.mkdir(parents=True, exist_ok=False)
    _status(
        state="actual_running_pre_outer",
        phase="fold_local_inner_fit_and_alpha_selection",
        progress=10,
        detail="sealed source meta 재사용 · ridge/control proxy inner 학습 중 · outer scoring 0",
        started=started,
        eta="약 20~45분",
    )

    source_meta = _read_source_meta(config)
    incumbent = _read_incumbent_label_free(config)
    wave, _ = _read_p3_train(p3_data_dir)
    anchor_path = ROOT / config["frozen_inputs"]["anchor_metadata_and_target_vault"]["path"]
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "grid_position", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    fold_names = [str(row[0]) for row in config["validation"]["windows"]]
    frozen_keys, membership = read_frozen_outer_key_membership(
        ROOT / config["frozen_inputs"]["incumbent_oof"]["path"],
        expected_folds=fold_names,
    )
    if not incumbent[list(PAIR_KEYS)].equals(frozen_keys[list(PAIR_KEYS)]):
        raise KMALongLeadError("incumbent and frozen membership differ")
    validate_outer_membership_against_anchors(frozen_keys, anchors)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["validation"]["windows"],
        validation_ids_by_fold=membership,
        embargo_hours=78,
    )
    all_outer_ids = np.unique(np.concatenate([fold.validation_ids for fold in folds]))
    relevant_ids = np.unique(
        np.concatenate([np.concatenate([fold.train_ids, fold.validation_ids]) for fold in folds])
    )
    if not np.isin(relevant_ids, source_meta["anchor_id"].to_numpy(dtype=np.int64)).all():
        raise KMALongLeadError("sealed source meta does not cover all rolling fold anchors")
    feature_columns = json.loads(
        (ROOT / config["frozen_inputs"]["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    features = pd.read_parquet(
        ROOT / config["frozen_inputs"]["train_features"]["path"],
        columns=["anchor_id", "station", *feature_columns],
    )
    features = features.loc[features["anchor_id"].isin(relevant_ids)].reset_index(drop=True)
    vault = TargetVault(anchor_path)
    prior_validation_ids = np.asarray([], dtype=np.int64)
    inner_rows: list[pd.DataFrame] = []
    inner_coefficients: dict[str, Any] = {}
    grid_scores: dict[str, Any] = {}
    model_hashes: dict[str, str] = {}
    for fold_index, fold in enumerate(folds):
        inner = build_inner_episode_split(
            anchors,
            fold.train_ids,
            validation_days=45,
            embargo_hours=78,
        )
        fit_targets = vault.read_training(
            inner.train_ids,
            forbidden_current_validation_ids=fold.validation_ids,
            all_outer_validation_ids=all_outer_ids,
            allowed_prior_validation_ids=prior_validation_ids,
            fold=fold.name,
            purpose="inner_fit_ridge_and_control_proxy",
        )
        calibration_targets = vault.read_training(
            inner.validation_ids,
            forbidden_current_validation_ids=fold.validation_ids,
            all_outer_validation_ids=all_outer_ids,
            allowed_prior_validation_ids=prior_validation_ids,
            fold=fold.name,
            purpose="inner_calibration_alpha_selection",
        )
        source_fit = _expand_source_rows(source_meta, anchors, inner.train_ids, targets=fit_targets)
        source_calibration = _expand_source_rows(
            source_meta, anchors, inner.validation_ids, targets=calibration_targets
        )
        calibrators = fit_ridge_pair(source_fit)
        calibrated = add_calibrated_source(source_calibration, calibrators)
        model_path = CANONICAL_OUTPUT / f"control_proxy_{fold.name}.cbm"
        control = _fit_control_proxy(
            config=config,
            features=features,
            anchors=anchors,
            fit_targets=fit_targets,
            fit_ids=inner.train_ids,
            calibration_ids=inner.validation_ids,
            feature_columns=feature_columns,
            model_path=model_path,
        )
        assembled = _assemble_inner_rows(
            fold_name=fold.name,
            source_rows=calibrated,
            control_rows=control,
        )
        alpha, selected, scores = select_fold_alpha(assembled)
        selected = selected.loc[:, list(INNER_COLUMNS)]
        validate_inner_predictions(selected)
        inner_rows.append(selected)
        inner_coefficients[fold.name] = {
            "fit_scope": "outer_train_inner_fit_only",
            "fit_anchor_count": int(len(inner.train_ids)),
            "calibration_anchor_count": int(len(inner.validation_ids)),
            "selected_alpha": alpha,
            "calibrators": _coefficient_payload(calibrators),
        }
        grid_scores[fold.name] = scores
        model_hashes[fold.name] = sha256_file(model_path)
        prior_validation_ids = np.union1d(prior_validation_ids, fold.validation_ids)
        _status(
            state="actual_running_pre_outer",
            phase="fold_local_inner_fit_and_alpha_selection",
            progress=25 + 15 * fold_index,
            detail=f"{fold.name} inner 완료 · outer scoring 0",
            started=started,
            eta="약 10~35분",
        )

    inner_predictions = (
        pd.concat(inner_rows, ignore_index=True).sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    )
    validate_inner_predictions(inner_predictions)
    inner_path = CANONICAL_OUTPUT / "inner_predictions.parquet"
    coefficients_path = CANONICAL_OUTPUT / "inner_ridge_coefficients.json"
    grid_path = CANONICAL_OUTPUT / "inner_alpha_grid_scores.json"
    _atomic_parquet(inner_path, inner_predictions)
    _atomic_json(coefficients_path, inner_coefficients)
    _atomic_json(grid_path, grid_scores)
    reloaded_inner = pd.read_parquet(inner_path)
    validate_inner_predictions(reloaded_inner)
    if not reloaded_inner.equals(inner_predictions):
        raise KMALongLeadError("fsynced inner predictions changed after reload")
    inner_gate = evaluate_inner_gate(reloaded_inner)
    inner_gate_path = CANONICAL_OUTPUT / "inner_gate.json"
    _atomic_json(inner_gate_path, inner_gate)
    inner_manifest = {
        "sealed": True,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "row_level_inner_only_before_outer_prediction",
        "inner_predictions_sha256": sha256_file(inner_path),
        "inner_ridge_coefficients_sha256": sha256_file(coefficients_path),
        "inner_alpha_grid_scores_sha256": sha256_file(grid_path),
        "inner_gate_sha256": sha256_file(inner_gate_path),
        "control_proxy_model_sha256_by_fold": model_hashes,
        "target_value_access_log": vault.access_log,
        "current_or_future_fold_validation_overlap_rows": 0,
        "designated_outer_scoring_read_performed": False,
        "global_zero_target_exposure_before_seal_claimed": False,
        "implementation_sha256": _implementation_hashes(),
        "registered_input_sha256": receipts,
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(GLOBAL_ATTEMPT_LOCK),
    }
    inner_manifest_path = CANONICAL_OUTPUT / "inner_manifest.json"
    _atomic_json(inner_manifest_path, inner_manifest)

    if not inner_gate["pass"]:
        result = _preouter_no_go_result(
            inner_gate=inner_gate,
            inner_manifest=inner_manifest_path,
            attempt_lock=GLOBAL_ATTEMPT_LOCK,
        )
        _atomic_json(CANONICAL_OUTPUT / "result.json", result)
        _status(
            state="actual_stopped_pre_outer",
            phase="inner_gate_no_go",
            progress=100,
            detail="inner calibrated blend gate 실패 · outer prediction/scoring 0 · rerun 금지",
            started=started,
            eta=None,
            result=result,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _status(
        state="actual_running_pre_outer",
        phase="outer_train_ridge_refit_and_blind_prediction",
        progress=72,
        detail="inner gate 통과 · full outer-train ridge refit 및 blind prediction 중",
        started=started,
        eta="약 5~15분",
    )
    selected_alphas = {
        str(name): float(value) for name, value in inner_gate["selected_alpha_by_fold"].items()
    }
    prior_validation_ids = np.asarray([], dtype=np.int64)
    outer_rows: list[pd.DataFrame] = []
    outer_coefficients: dict[str, Any] = {}
    for fold in folds:
        outer_train_targets = vault.read_training(
            fold.train_ids,
            forbidden_current_validation_ids=fold.validation_ids,
            all_outer_validation_ids=all_outer_ids,
            allowed_prior_validation_ids=prior_validation_ids,
            fold=fold.name,
            purpose="full_outer_train_fixed_ridge_refit",
        )
        source_train = _expand_source_rows(
            source_meta, anchors, fold.train_ids, targets=outer_train_targets
        )
        calibrators = fit_ridge_pair(source_train)
        validation_source = _expand_source_rows(
            source_meta, anchors, fold.validation_ids, targets=None
        )
        validation_source = add_calibrated_source(validation_source, calibrators)
        base = incumbent.loc[incumbent["fold"].astype(str).eq(fold.name)].copy()
        merged = base.merge(
            validation_source,
            on=["anchor_id", "station", "lead_h", "current_hs"],
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        if len(merged) != len(base):
            raise KMALongLeadError("outer source calibration lacks an incumbent key")
        alpha = selected_alphas[fold.name]
        merged["selected_alpha"] = alpha
        merged["candidate_final"] = blend_long_leads(
            merged["incumbent_final"].to_numpy(dtype=np.float64),
            merged["calibrated_source"].to_numpy(dtype=np.float64),
            merged["lead_h"].to_numpy(dtype=np.int64),
            alpha=alpha,
        )
        merged = merged.loc[:, list(OUTER_COLUMNS)]
        validate_outer_blind(merged)
        outer_rows.append(merged)
        outer_coefficients[fold.name] = {
            "fit_scope": "that_folds_full_outer_train_only",
            "fit_anchor_count": int(len(fold.train_ids)),
            "selected_inner_alpha": alpha,
            "calibrators": _coefficient_payload(calibrators),
        }
        prior_validation_ids = np.union1d(prior_validation_ids, fold.validation_ids)

    blind = (
        pd.concat(outer_rows, ignore_index=True).sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    )
    validate_outer_blind(blind)
    expected = frozen_keys.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    if not blind[list(PAIR_KEYS)].equals(expected[list(PAIR_KEYS)]):
        raise KMALongLeadError("blind prediction keys differ from frozen outer membership")
    outer_coefficients_path = CANONICAL_OUTPUT / "outer_ridge_coefficients.json"
    blind_path = CANONICAL_OUTPUT / "blind_predictions.parquet"
    _atomic_json(outer_coefficients_path, outer_coefficients)
    _atomic_parquet(blind_path, blind)
    blind_reloaded = pd.read_parquet(blind_path)
    validate_outer_blind(blind_reloaded)
    if not blind_reloaded.equals(blind):
        raise KMALongLeadError("fsynced blind predictions changed after reload")
    implementation = _implementation_hashes()
    if implementation != authorization["implementation_sha256"]:
        raise PermissionError("implementation differs from authorization before scoring")
    if _verify_registered_inputs(config, p3_data_dir=p3_data_dir) != receipts:
        raise PermissionError("registered input changed before scoring")
    blind_manifest = {
        "sealed": True,
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "designated_outer_scoring_read_performed": False,
        "blind_predictions_sha256": sha256_file(blind_path),
        "outer_ridge_coefficients_sha256": sha256_file(outer_coefficients_path),
        "inner_manifest_sha256": sha256_file(inner_manifest_path),
        "selected_alpha_by_fold": selected_alphas,
        "deployment_alpha_median": inner_gate["deployment_alpha_median"],
        "incumbent_oof_sha256": receipts["frozen_inputs_incumbent_oof"],
        "implementation_sha256": implementation,
        "registered_input_sha256": receipts,
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(GLOBAL_ATTEMPT_LOCK),
        "fold_local_current_validation_used_by_own_fold_calibrator": False,
        "global_zero_target_exposure_before_blind_seal_claimed": False,
    }
    blind_manifest_path = CANONICAL_OUTPUT / "blind_manifest.json"
    _atomic_json(blind_manifest_path, blind_manifest)
    exposure = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "designated_outer_scoring_read_performed": False,
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
        "blind_predictions_sha256": sha256_file(blind_path),
        "incumbent_oof_sha256": receipts["frozen_inputs_incumbent_oof"],
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "attempt_lock_sha256": sha256_file(GLOBAL_ATTEMPT_LOCK),
        "fsync_completed_before_designated_scoring": True,
    }
    exposure_path = CANONICAL_OUTPUT / "outer_exposure_receipt.json"
    _atomic_json(exposure_path, exposure)
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed after blind seal")
    if sha256_file(CANONICAL_DRY_RECEIPT) != authorization["dry_receipt_sha256"]:
        raise PermissionError("authorized dry receipt changed before scoring")
    if sha256_file(CANONICAL_AUTHORIZATION) != blind_manifest["authorization_sha256"]:
        raise PermissionError("authorization changed before scoring")
    if _verify_registered_inputs(config, p3_data_dir=p3_data_dir) != receipts:
        raise PermissionError("registered input changed after blind seal")
    ledger = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
        "blind_predictions_sha256": sha256_file(blind_path),
        "incumbent_oof_sha256": receipts["frozen_inputs_incumbent_oof"],
        "designated_outer_scoring_open_attempt": True,
    }
    _write_exclusive(GLOBAL_OUTER_LOCK, ledger)
    _write_exclusive(CANONICAL_OUTER_LOCK, ledger)
    _append_global_ledger(ledger)
    outer_targets = vault.open_designated_scoring_once(
        all_outer_ids,
        blind_manifest=blind_manifest_path,
        exposure_receipt=exposure_path,
    )
    if vault.designated_scoring_open_count != 1:
        raise AssertionError("designated outer scoring did not open exactly once")
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed during designated scoring")
    if sha256_file(CANONICAL_AUTHORIZATION) != blind_manifest["authorization_sha256"]:
        raise PermissionError("authorization changed during designated scoring")
    if sha256_file(CANONICAL_DRY_RECEIPT) != authorization["dry_receipt_sha256"]:
        raise PermissionError("dry receipt changed during designated scoring")
    if (
        sha256_file(ROOT / config["frozen_inputs"]["incumbent_oof"]["path"])
        != receipts["frozen_inputs_incumbent_oof"]
    ):
        raise PermissionError("incumbent OOF changed during designated scoring")
    target_long = _targets_to_long(outer_targets, blind_reloaded)
    evaluated = blind_reloaded.merge(
        target_long,
        on=["fold", "anchor_id", "station", "lead_h"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluated) != len(blind_reloaded):
        raise KMALongLeadError("designated outer target coverage is incomplete")
    promotion = evaluate_outer_promotion(evaluated, config)
    expected_incumbent = float(config["outer_candidate"]["expected_exact_incumbent_rmse"])
    if not np.isclose(promotion["incumbent"]["rmse"], expected_incumbent, rtol=0.0, atol=1e-12):
        raise KMALongLeadError("exact incumbent RMSE did not reconcile")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "decision": promotion["decision"],
        "inner_gate": inner_gate,
        "outer_promotion": promotion,
        "designated_outer_scoring_open_count": 1,
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
        "rerun_prohibited": True,
    }
    _atomic_json(CANONICAL_OUTPUT / "result.json", result)
    _status(
        state="actual_complete_one_shot",
        phase="designated_outer_scoring_complete",
        progress=100,
        detail=f"one-shot 완료 · {promotion['decision']}",
        started=started,
        eta=None,
        result={
            "decision": promotion["decision"],
            "candidate_minus_incumbent_rmse": promotion["candidate_minus_incumbent_rmse"],
            "designated_outer_scoring_open_count": 1,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "authorize", "actual"), default="dry-run")
    parser.add_argument("--experiment-config", default=str(CANONICAL_CONFIG))
    parser.add_argument("--p3-data-dir")
    parser.add_argument("--authorization-token")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    config_path = _canonical_config(args.experiment_config)
    config = load_preregistration(config_path)
    p3_data_dir = Path(args.p3_data_dir).resolve() if args.p3_data_dir else None
    if p3_data_dir is not None and not p3_data_dir.is_dir():
        raise FileNotFoundError("P3 data directory does not exist")
    if args.mode == "dry-run":
        return _dry_run(config, p3_data_dir=p3_data_dir, started=started)
    if args.mode == "authorize":
        return _authorize_actual(authorization_token=args.authorization_token, started=started)
    if p3_data_dir is None:
        raise ValueError("actual mode requires --p3-data-dir")
    return _actual(
        config,
        p3_data_dir=p3_data_dir,
        authorization_token=args.authorization_token,
        started=started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
