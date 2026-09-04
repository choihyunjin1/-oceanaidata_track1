"""Prepare or execute the one-shot P3 KMA source-prediction meta ablation.

The committed preregistration currently authorizes ``dry-run`` only.  Dry-run
reads schemas, hashes, and incumbent OOF keys, but never fits CatBoost and never
opens a P3 target column.  The actual path is implemented behind a canonical
config, explicit authorization, prediction sealing, and experiment-wide locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as pyarrow_parquet

from p3_wave.kma_source_meta import (
    LEADS,
    META_COLUMNS,
    PAIR_KEYS,
    KMASourceMetaError,
    append_meta_features,
    apply_source_median_imputer,
    build_source_cases,
    build_target_source_features,
    catboost_frame,
    compact_source_feature_columns,
    evaluate_inner_incremental_signal,
    evaluate_promotion,
    expand_prediction_rows,
    expand_target_rows,
    fit_source_median_imputer,
    integrate_frozen_router,
    load_preregistration,
    paired_comparison,
    read_frozen_outer_key_membership,
    read_frozen_router_components,
    resolve_domain_route,
    sha256_file,
    source_catboost,
    source_predictions_to_meta,
    summarize_common_history,
    target_catboost,
    validate_blind_prediction_frame,
    validate_outer_membership_against_anchors,
)
from p3_wave.models import threshold_case_weights
from p3_wave.revin_patch import (
    assign_storm_episodes_from_wave,
    build_episode_disjoint_folds_from_ids,
    build_inner_episode_split,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_kma_source_prediction_meta_v1"
CANONICAL_CONFIG = (ROOT / "configs/experiments/p3_kma_source_prediction_meta_v1.json").resolve()
CANONICAL_OUTPUT = (ROOT / "artifacts/p3_kma_source_prediction_meta_v1/one_shot").resolve()
CANONICAL_STATUS = (ROOT / "artifacts/status/p3_kma_source_prediction_meta_v1.json").resolve()
CANONICAL_OUTER_LOCK = (CANONICAL_OUTPUT / "OUTER_TRUTH.lock").resolve()
CANONICAL_AUTHORIZATION = (
    ROOT / "artifacts/p3_kma_source_prediction_meta_v1/authorization_amendment.json"
).resolve()
GLOBAL_ATTEMPT_LOCK = (
    ROOT / "artifacts/experiment_locks/p3_kma_source_prediction_meta_v1.attempt.lock"
).resolve()
GLOBAL_OUTER_LOCK = (
    ROOT / "artifacts/experiment_locks/p3_kma_source_prediction_meta_v1.outer.lock"
).resolve()
GLOBAL_LEDGER = (ROOT / "artifacts/experiment_locks/p3_outer_truth_ledger.jsonl").resolve()
AUTHORIZATION_TOKEN = "ROOT_APPROVED_P3_KMA_SOURCE_PREDICTION_META_V1"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
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


def _status(
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    started: float,
    eta: str | None,
    result: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 KMA source-prediction meta v1",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "eta": eta,
        "model_fit_count": 0 if state.startswith(("dry", "ready")) else None,
        "outer_truth_read_count": 0 if state.startswith(("dry", "ready")) else None,
        "updated_at": _now(),
    }
    if result is not None:
        payload["result"] = result
    _atomic_json(CANONICAL_STATUS, payload)


def _canonical_config(path: str | Path) -> Path:
    current = Path(path).resolve()
    if current != CANONICAL_CONFIG:
        raise PermissionError("experiment config override is prohibited for the one-shot runner")
    return current


def _verify_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required {role} input is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise KMASourceMetaError(f"{role} SHA256 changed")
    return actual


def _implementation_hashes() -> dict[str, str]:
    paths = (
        "configs/experiments/p3_kma_source_prediction_meta_v1.json",
        "src/p3_wave/kma_source_meta.py",
        "scripts/run_p3_kma_source_prediction_meta_v1.py",
        "tests/test_p3_kma_source_prediction_meta_v1.py",
        "src/p3_wave/kma_external.py",
        "src/p3_wave/features.py",
        "src/p3_wave/models.py",
        "src/p3_wave/validation.py",
        "src/p3_wave/revin_patch.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"sealed implementation input is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _verify_registered_inputs(
    config: dict[str, Any], *, p3_data_dir: Path | None
) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for section, name in (
        ("source", "observations"),
        ("source", "anchors"),
        ("source", "manifest"),
    ):
        record = config[section][name]
        role = f"{section}_{name}"
        receipts[role] = _verify_hash(ROOT / record["path"], record["sha256"], role)
    for name, record in config["frozen_inputs"].items():
        role = f"frozen_{name}"
        receipts[role] = _verify_hash(ROOT / record["path"], record["sha256"], role)
    for name, record in config["policy_inputs"].items():
        role = f"policy_{name}"
        receipts[role] = _verify_hash(ROOT / record["path"], record["sha256"], role)
    domain = config["domain_shift"]
    receipts["domain_config"] = _verify_hash(
        ROOT / domain["config_path"], domain["config_sha256"], "domain_config"
    )
    result_path = ROOT / domain["result_path"]
    if result_path.is_file():
        receipts["domain_result"] = _verify_hash(
            result_path, domain["result_sha256"], "domain_result"
        )
    elif domain["result_required_for_actual"]:
        receipts["domain_result"] = "pending"
    if p3_data_dir is not None:
        for filename, expected in config["p3_sources"].items():
            receipts[f"p3_{filename}"] = _verify_hash(
                p3_data_dir / filename, expected, f"p3_{filename}"
            )
    return receipts


def _domain_auc(config: dict[str, Any]) -> float | None:
    path = ROOT / config["domain_shift"]["result_path"]
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("does_not_test_transfer_utility") is not True:
        raise KMASourceMetaError("domain result scope no longer states its transfer limitation")
    if payload.get("interpretation_scope", {}).get("kma_source_permanently_rejected") is not False:
        raise KMASourceMetaError("domain result unexpectedly permanently rejects KMA")
    return float(payload["domain_classifier"]["oof_auc"])


def _schema_preflight(config: dict[str, Any]) -> dict[str, Any]:
    source_schema = pyarrow_parquet.read_schema(ROOT / config["source"]["observations"]["path"])
    required_source = {
        "TM",
        "STN",
        "WD1",
        "WS1",
        "WS1_GST",
        "WD2",
        "WS2",
        "WS2_GST",
        "PA",
        "HM",
        "TA",
        "WH_MAX",
        "WH_SIG",
        "WP",
        "WO",
    }
    if not required_source <= set(source_schema.names):
        raise KMASourceMetaError("KMA source parquet schema is incomplete")
    anchor_schema = pyarrow_parquet.read_schema(ROOT / config["source"]["anchors"]["path"])
    if not {"station_id", "anchor_time_kst"} <= set(anchor_schema.names):
        raise KMASourceMetaError("KMA source anchor schema is incomplete")
    frozen_anchor_schema = pyarrow_parquet.read_schema(
        ROOT / config["frozen_inputs"]["anchor_metadata_and_vault"]["path"]
    )
    required_anchor = {
        "anchor_id",
        "station",
        "anchor_time",
        "grid_position",
        "current_hs",
        *[f"target_{lead}" for lead in LEADS],
    }
    if not required_anchor <= set(frozen_anchor_schema.names):
        raise KMASourceMetaError("P3 anchor vault schema is incomplete")
    feature_schema = pyarrow_parquet.read_schema(
        ROOT / config["frozen_inputs"]["train_features"]["path"]
    )
    target_columns = json.loads(
        (ROOT / config["frozen_inputs"]["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    if len(target_columns) != 591 or len(set(target_columns)) != 591:
        raise KMASourceMetaError("frozen target feature list changed")
    if not {"anchor_id", "station", *target_columns} <= set(feature_schema.names):
        raise KMASourceMetaError("frozen target feature parquet is incomplete")
    source_columns = list(compact_source_feature_columns())
    expected_source = [
        column
        for column in target_columns
        if "_valid_" not in column
        and not column.startswith("tp_")
        and not column.startswith("steepness_proxy_")
    ]
    if source_columns != expected_source:
        raise KMASourceMetaError(
            "447-column source surface is not the frozen no-valid/no-period subset"
        )
    columns_hash = hashlib.sha256(
        json.dumps(source_columns, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if columns_hash != config["representation"]["expected_source_feature_columns_sha256"]:
        raise KMASourceMetaError("source feature column hash changed")
    return {
        "source_schema_columns": len(source_schema.names),
        "source_anchor_schema_columns": len(anchor_schema.names),
        "source_feature_count": len(source_columns),
        "target_feature_count": len(target_columns),
        "source_feature_columns_sha256": columns_hash,
    }


def _key_preflight(config: dict[str, Any]) -> dict[str, Any]:
    windows = config["validation"]["windows"]
    fold_names = [str(item[0]) for item in windows]
    oof_path = ROOT / config["frozen_inputs"]["incumbent_oof_keys"]["path"]
    keys, membership = read_frozen_outer_key_membership(oof_path, expected_folds=fold_names)
    router = read_frozen_router_components(oof_path)
    if not router[list(PAIR_KEYS)].equals(keys[list(PAIR_KEYS)]):
        raise KMASourceMetaError("frozen router rows differ from key-only membership rows")
    anchor_path = ROOT / config["frozen_inputs"]["anchor_metadata_and_vault"]["path"]
    anchors = pd.read_parquet(anchor_path, columns=["anchor_id", "station"])
    validate_outer_membership_against_anchors(keys, anchors)
    return {
        "outer_key_rows": int(len(keys)),
        "outer_cases": int(keys[["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "cases_by_fold": {name: int(len(ids)) for name, ids in membership.items()},
        "oof_columns_read": list(PAIR_KEYS),
        "oof_label_free_router_columns_read": 7,
        "oof_exact_incumbent_prediction_read": True,
        "oof_target_columns_read": 0,
    }


def _synthetic_dry_run(target_columns: list[str]) -> dict[str, Any]:
    x = np.arange(97, dtype=np.float64)
    history = pd.DataFrame(
        {
            "hs": 1.5 + 0.01 * x,
            "hmax": 2.2 + 0.012 * x,
            "wspd": 5.0 + 0.02 * x,
            "gust": 7.0 + 0.02 * x,
            "airt": 18.0 - 0.01 * x,
            "relh": 70.0 + 0.01 * x,
            "caph": 1012.0 - 0.02 * x,
            "wvdir": np.mod(210.0 + x, 360.0),
            "wdir": np.mod(180.0 + 0.5 * x, 360.0),
        }
    )
    history.loc[20:21, "relh"] = np.nan
    row = summarize_common_history(history)
    source = pd.DataFrame([row, row], columns=compact_source_feature_columns())
    medians = fit_source_median_imputer(source)
    imputed = apply_source_median_imputer(source, medians)
    meta = source_predictions_to_meta(
        np.zeros((2, 6), dtype=np.float64), anchor_ids=[11, 12], current_hs=[1.8, 2.1]
    )
    base = pd.DataFrame(np.zeros((2, len(target_columns))), columns=target_columns)
    base.insert(0, "station", ["G-ORS", "I-ORS"])
    base.insert(0, "anchor_id", [11, 12])
    challenger = append_meta_features(base, meta, expected_base_columns=target_columns)
    single_blind = pd.DataFrame(
        {
            "fold": ["synthetic"] * 2,
            "anchor_id": [11, 12],
            "station": ["G-ORS", "I-ORS"],
            "lead_h": [3, 12],
            "current_hs": [1.8, 2.1],
            "control_single_prediction": [1.8, 2.1],
            "challenger_single_prediction": [1.8, 2.1],
        }
    )
    router = pd.DataFrame(
        {
            "fold": ["synthetic"] * 2,
            "anchor_id": [11, 12],
            "station": ["G-ORS", "I-ORS"],
            "lead_h": [3, 12],
            "multi_prediction": [1.9, 2.2],
            "persistence": [1.8, 2.1],
            "weight_single": [0.5, 0.5],
            "weight_multi": [0.3, 0.3],
            "weight_persistence": [0.2, 0.2],
            "second_stage_persistence_weight": [0.0, 0.2],
            "prediction": [1.83, 2.13],
        }
    )
    blind = integrate_frozen_router(single_blind, router)
    blind_summary = validate_blind_prediction_frame(blind)
    return {
        "source_feature_shape": list(imputed.shape),
        "challenger_feature_count": int(len(challenger.columns) - 2),
        "meta_column_count": len(META_COLUMNS),
        "blind": blind_summary,
        "catboost_imported": "catboost" in sys.modules,
        "model_fit_count": 0,
        "outer_truth_read_count": 0,
    }


def _dry_run(config: dict[str, Any], *, p3_data_dir: Path | None, started: float) -> int:
    _status(
        state="dry_running",
        phase="hash_and_policy_preflight",
        progress=15,
        detail="등록 SHA·라이선스·cutoff 계약 검증 중",
        started=started,
        eta="약 1분",
    )
    receipts = _verify_registered_inputs(config, p3_data_dir=p3_data_dir)
    manifest = json.loads((ROOT / config["source"]["manifest"]["path"]).read_text(encoding="utf-8"))
    if manifest.get("license_name") != config["source"]["license"]:
        raise KMASourceMetaError("KMA source license changed")
    if manifest.get("observed_end") != config["source"]["maximum_timestamp_kst"]:
        raise KMASourceMetaError("KMA source cutoff receipt changed")
    if (
        manifest.get("precheck", {}).get("independent_anchor_count")
        != config["source"]["anchors"]["expected_count"]
    ):
        raise KMASourceMetaError("KMA source anchor count changed")

    _status(
        state="dry_running",
        phase="schema_and_key_only_audit",
        progress=45,
        detail="값·정답 없이 schema와 frozen OOF key membership만 검증 중",
        started=started,
        eta="1분 미만",
    )
    schema = _schema_preflight(config)
    keys = _key_preflight(config)
    auc = _domain_auc(config)
    route = resolve_domain_route(auc)
    if route.direct_concat_allowed:
        raise AssertionError("direct source-target row concatenation unexpectedly enabled")

    target_columns = json.loads(
        (ROOT / config["frozen_inputs"]["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    synthetic = _synthetic_dry_run(target_columns)
    if synthetic["catboost_imported"]:
        raise AssertionError("dry-run imported CatBoost despite the zero-model-fit contract")
    one_shot_available = not any(
        path.exists()
        for path in (
            CANONICAL_AUTHORIZATION,
            GLOBAL_ATTEMPT_LOCK,
            CANONICAL_OUTER_LOCK,
            GLOBAL_OUTER_LOCK,
        )
    )
    if not one_shot_available:
        raise PermissionError("one-shot outer lock already exists")
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "status": "READY_PENDING_PARENT_ACTUAL_APPROVAL",
        "mode": "dry-run",
        "registered_input_sha256": receipts,
        "implementation_sha256": _implementation_hashes(),
        "schema_preflight": schema,
        "key_preflight": keys,
        "domain_route": {
            "oof_auc": route.auc,
            "route": route.route,
            "direct_concat_allowed": route.direct_concat_allowed,
            "requires_inner_incremental_signal": route.requires_inner_incremental_signal,
            "domain_result_does_not_test_transfer_utility": True,
        },
        "missingness_harmonization": {
            "valid_or_missing_fraction_features": 0,
            "causal_forward_fill_limit_minutes": 60,
            "remaining_nan_imputer_fit_domain": "source_only",
        },
        "synthetic_contract": synthetic,
        "one_shot_locks_created": 0,
        "one_shot_available": one_shot_available,
        "model_fit_count": 0,
        "outer_truth_read_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    dry_dir = ROOT / "artifacts/p3_kma_source_prediction_meta_v1/dry_run"
    _atomic_json(dry_dir / "receipt.json", receipt)
    receipt["receipt_sha256"] = sha256_file(dry_dir / "receipt.json")
    _status(
        state="dry_ready",
        phase="prepared_waiting_parent_approval",
        progress=100,
        detail="dry-run 완료 · model 0 · outer truth 0 · actual 승인 대기",
        started=started,
        eta=None,
        result={
            "decision": receipt["status"],
            "domain_auc": route.auc,
            "domain_route": route.route,
            "source_feature_count": schema["source_feature_count"],
            "target_feature_count": schema["target_feature_count"],
            "model_fit_count": 0,
            "outer_truth_read_count": 0,
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


@dataclass
class TargetVault:
    path: Path
    access_log: list[dict[str, Any]] = field(default_factory=list)
    outer_open_count: int = 0

    def _read(self, anchor_ids: np.ndarray) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise KMASourceMetaError("target request must contain unique anchor ids")
        columns = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id").loc[ids].reset_index()
        if len(frame) != len(ids) or frame["anchor_id"].duplicated().any():
            raise KMASourceMetaError("filtered target read is incomplete")
        if not np.isfinite(frame.drop(columns="anchor_id").to_numpy(dtype=np.float64)).all():
            raise KMASourceMetaError("filtered target read contains a non-finite value")
        return frame

    def read_outer_train(
        self,
        anchor_ids: np.ndarray,
        *,
        forbidden_outer_validation_ids: np.ndarray,
        all_outer_validation_ids: np.ndarray,
        allowed_prior_validation_ids: np.ndarray,
        fold: str,
    ) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if np.intersect1d(ids, forbidden_outer_validation_ids).size:
            raise PermissionError("outer validation target requested before prediction seal")
        outer_overlap = np.intersect1d(ids, all_outer_validation_ids)
        unexpected_outer_overlap = np.setdiff1d(outer_overlap, allowed_prior_validation_ids)
        if unexpected_outer_overlap.size:
            raise PermissionError("future or current outer validation target requested as training")
        frame = self._read(ids)
        self.access_log.append(
            {
                "purpose": "fold_local_outer_train_inner_only",
                "fold": fold,
                "rows": len(frame),
                "current_validation_overlap_rows": 0,
                "permitted_prior_validation_history_rows": int(len(outer_overlap)),
            }
        )
        return frame

    def open_outer_once(
        self,
        anchor_ids: np.ndarray,
        *,
        sealed_manifest: Path,
        exposure_receipt: Path,
    ) -> pd.DataFrame:
        if self.outer_open_count:
            raise PermissionError("outer validation targets may be opened only once")
        manifest = json.loads(sealed_manifest.read_text(encoding="utf-8"))
        receipt = json.loads(exposure_receipt.read_text(encoding="utf-8"))
        if manifest.get("sealed") is not True or manifest.get("outer_truth_opened") is not False:
            raise PermissionError("blind manifest is not in a sealed pre-truth state")
        if receipt.get("blind_manifest_sha256") != sha256_file(sealed_manifest):
            raise PermissionError("outer exposure receipt is not bound to the blind manifest")
        if receipt.get("fsync_completed_before_outer_open") is not True:
            raise PermissionError("pre-open fsync assertion is absent")
        frame = self._read(anchor_ids)
        self.outer_open_count += 1
        self.access_log.append({"purpose": "outer_evaluation_after_seal", "rows": len(frame)})
        return frame


def _read_p3_train(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    atmos = pd.read_csv(data_dir / "train_atmos.csv")
    for frame in (wave, atmos):
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if wave.duplicated(["station", "time"]).any() or atmos.duplicated(["station", "time"]).any():
        raise KMASourceMetaError("P3 raw training input contains duplicate keys")
    return wave, atmos


def _inner_comparison(
    *,
    fold: Any,
    anchors: pd.DataFrame,
    base: pd.DataFrame,
    challenger: pd.DataFrame,
    targets: pd.DataFrame,
    base_columns: list[str],
    target_config: dict[str, Any],
) -> tuple[int, pd.DataFrame]:
    from catboost import Pool

    inner = build_inner_episode_split(
        anchors,
        fold.train_ids,
        validation_days=45,
        embargo_hours=78,
    )
    x_fit, y_fit, fit_meta = expand_target_rows(
        base, anchors, targets, inner.train_ids, base_columns
    )
    x_cal, _, cal_meta = expand_target_rows(
        base, anchors, targets, inner.validation_ids, base_columns
    )
    control = target_catboost(target_config, iterations=int(target_config["maximum_iterations"]))
    control.fit(
        Pool(
            catboost_frame(x_fit),
            y_fit,
            weight=threshold_case_weights(fit_meta["current_hs"].to_numpy()),
            cat_features=[0, 1],
        ),
        eval_set=Pool(
            catboost_frame(x_cal),
            cal_meta["target_hs"].to_numpy() - cal_meta["current_hs"].to_numpy(),
            cat_features=[0, 1],
        ),
        early_stopping_rounds=int(target_config["inner_early_stopping_rounds"]),
        use_best_model=True,
        verbose=False,
    )
    iteration = max(int(control.get_best_iteration()) + 1, 1)
    control_prediction = np.clip(
        cal_meta["current_hs"].to_numpy() + control.predict(catboost_frame(x_cal)), 0.0, 30.0
    )
    x_ch_fit, y_ch_fit, ch_fit_meta = expand_target_rows(
        challenger,
        anchors,
        targets,
        inner.train_ids,
        [*base_columns, *META_COLUMNS],
    )
    x_ch_cal, _, ch_cal_meta = expand_target_rows(
        challenger,
        anchors,
        targets,
        inner.validation_ids,
        [*base_columns, *META_COLUMNS],
    )
    treatment = target_catboost(target_config, iterations=iteration)
    treatment.fit(
        catboost_frame(x_ch_fit),
        y_ch_fit,
        sample_weight=threshold_case_weights(ch_fit_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    challenger_prediction = np.clip(
        ch_cal_meta["current_hs"].to_numpy() + treatment.predict(catboost_frame(x_ch_cal)),
        0.0,
        30.0,
    )
    return iteration, _assemble_inner_gate_frame(
        cal_meta,
        fold_name=fold.name,
        control_prediction=control_prediction,
        challenger_prediction=challenger_prediction,
    )


def _assemble_inner_gate_frame(
    calibration_meta: pd.DataFrame,
    *,
    fold_name: str,
    control_prediction: np.ndarray,
    challenger_prediction: np.ndarray,
) -> pd.DataFrame:
    """Build the exact schema consumed by the preregistered inner utility gate."""

    control = np.asarray(control_prediction, dtype=np.float64)
    challenger = np.asarray(challenger_prediction, dtype=np.float64)
    if len(calibration_meta) != len(control) or len(calibration_meta) != len(challenger):
        raise KMASourceMetaError("inner prediction length differs from calibration metadata")
    if not np.isfinite(control).all() or not np.isfinite(challenger).all():
        raise KMASourceMetaError("inner prediction contains a non-finite value")
    result = calibration_meta.copy()
    result["fold"] = fold_name
    result["control_prediction"] = control
    result["challenger_prediction"] = challenger
    required = {"fold", "target_hs", "control_prediction", "challenger_prediction"}
    if not required <= set(result.columns):
        raise KMASourceMetaError("assembled inner utility frame is incomplete")
    return result


def _outer_blind_predictions(
    *,
    fold: Any,
    anchors: pd.DataFrame,
    base: pd.DataFrame,
    challenger: pd.DataFrame,
    targets: pd.DataFrame,
    base_columns: list[str],
    target_config: dict[str, Any],
    iteration: int,
    output_dir: Path,
) -> pd.DataFrame:
    target_ids = targets["anchor_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(target_ids), np.sort(np.asarray(fold.train_ids))):
        raise KMASourceMetaError("fold target frame differs from the frozen outer-train ids")
    if np.intersect1d(target_ids, np.asarray(fold.validation_ids)).size:
        raise PermissionError("current fold validation target leaked into blind model fit")
    x_train, y_train, train_meta = expand_target_rows(
        base, anchors, targets, fold.train_ids, base_columns
    )
    x_valid, valid_meta = expand_prediction_rows(base, anchors, fold.validation_ids, base_columns)
    control = target_catboost(target_config, iterations=iteration)
    control.fit(
        catboost_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    control_prediction = np.clip(
        valid_meta["current_hs"].to_numpy() + control.predict(catboost_frame(x_valid)),
        0.0,
        30.0,
    )
    x_ch_train, y_ch_train, ch_train_meta = expand_target_rows(
        challenger,
        anchors,
        targets,
        fold.train_ids,
        [*base_columns, *META_COLUMNS],
    )
    x_ch_valid, ch_valid_meta = expand_prediction_rows(
        challenger,
        anchors,
        fold.validation_ids,
        [*base_columns, *META_COLUMNS],
    )
    treatment = target_catboost(target_config, iterations=iteration)
    treatment.fit(
        catboost_frame(x_ch_train),
        y_ch_train,
        sample_weight=threshold_case_weights(ch_train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    challenger_prediction = np.clip(
        ch_valid_meta["current_hs"].to_numpy() + treatment.predict(catboost_frame(x_ch_valid)),
        0.0,
        30.0,
    )
    control.save_model(output_dir / f"control_{fold.name}.cbm")
    treatment.save_model(output_dir / f"challenger_{fold.name}.cbm")
    result = valid_meta.copy()
    result.insert(0, "fold", fold.name)
    result["control_single_prediction"] = control_prediction
    result["challenger_single_prediction"] = challenger_prediction
    result = result[
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "current_hs",
            "control_single_prediction",
            "challenger_single_prediction",
        ]
    ]
    prediction = result[["control_single_prediction", "challenger_single_prediction"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(prediction).all() or (prediction < 0.0).any() or (prediction > 30.0).any():
        raise KMASourceMetaError("single-model blind prediction is invalid")
    if result.duplicated(list(PAIR_KEYS)).any():
        raise KMASourceMetaError("single-model blind prediction contains duplicate keys")
    return result


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _append_global_ledger(payload: dict[str, Any]) -> None:
    GLOBAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(GLOBAL_LEDGER, os.O_CREAT | os.O_APPEND | os.O_WRONLY)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _dry_receipt_path() -> Path:
    return ROOT / "artifacts/p3_kma_source_prediction_meta_v1/dry_run/receipt.json"


def _authorize_actual(*, authorization_token: str | None, started: float) -> int:
    """Create a one-time amendment bound to the exact reviewed dry-run bytes."""

    if authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("authorization amendment requires the exact parent token")
    if CANONICAL_AUTHORIZATION.exists() or GLOBAL_ATTEMPT_LOCK.exists():
        raise PermissionError("authorization or actual attempt already exists")
    receipt_path = _dry_receipt_path()
    if not receipt_path.is_file():
        raise FileNotFoundError("canonical dry-run receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "READY_PENDING_PARENT_ACTUAL_APPROVAL":
        raise PermissionError("dry-run receipt is not approval-ready")
    implementation = _implementation_hashes()
    if receipt.get("implementation_sha256") != implementation:
        raise PermissionError("implementation differs from the reviewed dry-run receipt")
    amendment = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "authorized": True,
        "authorized_at": _now(),
        "dry_receipt_sha256": sha256_file(receipt_path),
        "implementation_sha256": implementation,
        "registered_input_sha256": receipt["registered_input_sha256"],
        "config_sha256": implementation[
            "configs/experiments/p3_kma_source_prediction_meta_v1.json"
        ],
        "outer_truth_opened": False,
    }
    _write_exclusive(CANONICAL_AUTHORIZATION, amendment)
    _status(
        state="ready_actual_authorized",
        phase="authorization_amendment_sealed",
        progress=100,
        detail="exact dry-run SHA에 actual 승인 봉인 · model/outer truth 0",
        started=started,
        eta=None,
        result={
            "decision": "ACTUAL_AUTHORIZED_NOT_STARTED",
            "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
            "dry_receipt_sha256": amendment["dry_receipt_sha256"],
            "model_fit_count": 0,
            "outer_truth_read_count": 0,
        },
    )
    return 0


def _load_authorization() -> dict[str, Any]:
    if not CANONICAL_AUTHORIZATION.is_file():
        raise PermissionError("canonical actual authorization amendment is missing")
    amendment = json.loads(CANONICAL_AUTHORIZATION.read_text(encoding="utf-8"))
    if amendment.get("authorized") is not True or amendment.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("actual authorization amendment is invalid")
    receipt_path = _dry_receipt_path()
    if amendment.get("dry_receipt_sha256") != sha256_file(receipt_path):
        raise PermissionError("dry-run receipt changed after actual authorization")
    if amendment.get("implementation_sha256") != _implementation_hashes():
        raise PermissionError("implementation changed after actual authorization")
    return amendment


def _targets_to_long(targets: pd.DataFrame, blind: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    lookup = blind[["fold", "anchor_id", "station"]].drop_duplicates()
    for lead in LEADS:
        block = targets[["anchor_id", f"target_{lead}"]].rename(
            columns={f"target_{lead}": "target_hs"}
        )
        block["lead_h"] = lead
        rows.append(block)
    long = pd.concat(rows, ignore_index=True)
    return lookup.merge(long, on="anchor_id", how="inner", validate="one_to_many")


def _inner_gate_no_go_result(
    *,
    auc: float,
    domain_route: str,
    inner_gate: dict[str, Any],
    attempt_lock: Path,
) -> dict[str, Any]:
    """Report a pre-outer stop without hiding the consumed one-shot attempt."""

    if not attempt_lock.is_file():
        raise PermissionError("pre-outer stop is missing the global attempt lock")
    return {
        "decision": "NO_GO_HIGH_AUC_INNER_NO_INCREMENTAL_SIGNAL",
        "domain_auc": auc,
        "domain_route": domain_route,
        "inner_gate": inner_gate,
        "outer_predictions_written": 0,
        "designated_outer_scoring_open_count": 0,
        "one_shot_locks_created": 1,
        "global_attempt_lock_created": True,
        "global_attempt_lock_sha256": sha256_file(attempt_lock),
        "outer_truth_locks_created": 0,
        "rerun_prohibited": True,
    }


def _actual(
    config: dict[str, Any], *, p3_data_dir: Path, authorization_token: str | None, started: float
) -> int:
    if authorization_token != AUTHORIZATION_TOKEN:
        raise PermissionError("actual execution requires the exact parent authorization token")
    canonical_config = load_preregistration(CANONICAL_CONFIG)
    if config != canonical_config:
        raise PermissionError("in-memory config differs from the canonical preregistration")
    config = canonical_config
    authorization = _load_authorization()
    if GLOBAL_ATTEMPT_LOCK.exists() or CANONICAL_OUTER_LOCK.exists() or GLOBAL_OUTER_LOCK.exists():
        raise PermissionError("this experiment has already crossed or attempted the outer boundary")
    if CANONICAL_OUTPUT.exists():
        raise PermissionError("canonical one-shot output already exists")
    receipts = _verify_registered_inputs(config, p3_data_dir=p3_data_dir)
    if receipts != authorization["registered_input_sha256"]:
        raise PermissionError("registered inputs differ from the authorized dry-run receipt")
    auc = _domain_auc(config)
    route = resolve_domain_route(auc)
    if auc is None:
        raise PermissionError("domain-shift result must be sealed before actual execution")
    attempt = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "authorization_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "dry_receipt_sha256": authorization["dry_receipt_sha256"],
        "implementation_sha256": authorization["implementation_sha256"],
        "outer_truth_opened": False,
    }
    _write_exclusive(GLOBAL_ATTEMPT_LOCK, attempt)

    _status(
        state="actual_running_pre_outer",
        phase="external_source_fit_and_label_free_target_meta",
        progress=10,
        detail="external-only source model과 label-free target meta 생성 중",
        started=started,
        eta="약 30~65분",
    )
    output = CANONICAL_OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    source_observations = pd.read_parquet(ROOT / config["source"]["observations"]["path"])
    source_anchors = pd.read_parquet(ROOT / config["source"]["anchors"]["path"])
    source_cases = build_source_cases(source_observations, source_anchors)
    source_medians = fit_source_median_imputer(source_cases.features)
    source_x = apply_source_median_imputer(source_cases.features, source_medians)
    source_model = source_catboost(config["source_model"])
    source_model.fit(source_x, source_cases.residual_targets, verbose=False)
    source_model_path = output / "source_model.cbm"
    source_model.save_model(source_model_path)
    _atomic_json(output / "source_feature_medians.json", source_medians.to_dict())

    wave, atmos = _read_p3_train(p3_data_dir)
    anchor_path = ROOT / config["frozen_inputs"]["anchor_metadata_and_vault"]["path"]
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "grid_position", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    fold_names = [str(item[0]) for item in config["validation"]["windows"]]
    frozen_keys, membership = read_frozen_outer_key_membership(
        ROOT / config["frozen_inputs"]["incumbent_oof_keys"]["path"],
        expected_folds=fold_names,
    )
    router_components = read_frozen_router_components(
        ROOT / config["frozen_inputs"]["incumbent_oof_keys"]["path"]
    )
    if not router_components[list(PAIR_KEYS)].equals(frozen_keys[list(PAIR_KEYS)]):
        raise KMASourceMetaError("frozen router rows differ from frozen outer membership")
    validate_outer_membership_against_anchors(frozen_keys, anchors)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["validation"]["windows"],
        validation_ids_by_fold=membership,
        embargo_hours=78,
    )
    relevant_ids = np.unique(
        np.concatenate([np.concatenate([fold.train_ids, fold.validation_ids]) for fold in folds])
    )
    relevant_anchors = anchors.loc[anchors["anchor_id"].isin(relevant_ids)].copy()
    common = build_target_source_features(wave, atmos, relevant_anchors)
    target_x = apply_source_median_imputer(
        common.loc[:, list(compact_source_feature_columns())], source_medians
    )
    source_delta = np.asarray(source_model.predict(target_x), dtype=np.float64)
    current = (
        relevant_anchors.set_index("anchor_id")
        .loc[common["anchor_id"], "current_hs"]
        .to_numpy(dtype=np.float64)
    )
    meta = source_predictions_to_meta(
        source_delta,
        anchor_ids=common["anchor_id"].to_numpy(dtype=np.int64),
        current_hs=current,
    )
    meta_path = output / "source_meta_predictions.parquet"
    _atomic_parquet(meta_path, meta)
    source_seal = {
        "sealed": True,
        "created_at": _now(),
        "source_model_sha256": sha256_file(source_model_path),
        "source_meta_predictions_sha256": sha256_file(meta_path),
        "source_feature_medians_sha256": sha256_file(output / "source_feature_medians.json"),
        "p3_target_columns_read_before_source_seal": 0,
        "source_station_calendar_or_proxy_model_columns": 0,
    }
    _atomic_json(output / "source_meta_seal.json", source_seal)

    base_columns = json.loads(
        (ROOT / config["frozen_inputs"]["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    feature_path = ROOT / config["frozen_inputs"]["train_features"]["path"]
    base = pd.read_parquet(feature_path, columns=["anchor_id", "station", *base_columns])
    base = base.loc[base["anchor_id"].isin(relevant_ids)].reset_index(drop=True)
    treatment_features = append_meta_features(base, meta, expected_base_columns=base_columns)
    vault = TargetVault(anchor_path)
    inner_rows: list[pd.DataFrame] = []
    iterations: dict[str, int] = {}
    train_targets: dict[str, pd.DataFrame] = {}
    all_outer_ids = np.unique(np.concatenate([fold.validation_ids for fold in folds]))
    prior_validation_ids = np.asarray([], dtype=np.int64)
    for fold in folds:
        targets = vault.read_outer_train(
            fold.train_ids,
            forbidden_outer_validation_ids=fold.validation_ids,
            all_outer_validation_ids=all_outer_ids,
            allowed_prior_validation_ids=prior_validation_ids,
            fold=fold.name,
        )
        train_targets[fold.name] = targets
        iteration, inner = _inner_comparison(
            fold=fold,
            anchors=anchors,
            base=base,
            challenger=treatment_features,
            targets=targets,
            base_columns=base_columns,
            target_config=config["target_model"],
        )
        iterations[fold.name] = iteration
        inner_rows.append(inner)
        prior_validation_ids = np.union1d(prior_validation_ids, fold.validation_ids)
    inner = pd.concat(inner_rows, ignore_index=True)
    inner_gate = evaluate_inner_incremental_signal(inner)
    _atomic_json(output / "inner_utility_gate.json", inner_gate)
    if route.requires_inner_incremental_signal and not inner_gate["pass"]:
        result = _inner_gate_no_go_result(
            auc=auc,
            domain_route=route.route,
            inner_gate=inner_gate,
            attempt_lock=GLOBAL_ATTEMPT_LOCK,
        )
        _atomic_json(output / "result.json", result)
        _status(
            state="actual_stopped_pre_outer",
            phase="high_auc_inner_utility_no_go",
            progress=100,
            detail="inner 최소신호 gate 실패 · outer prediction/정답 0",
            started=started,
            eta=None,
            result=result,
        )
        return 0

    blind_rows = [
        _outer_blind_predictions(
            fold=fold,
            anchors=anchors,
            base=base,
            challenger=treatment_features,
            targets=train_targets[fold.name],
            base_columns=base_columns,
            target_config=config["target_model"],
            iteration=iterations[fold.name],
            output_dir=output,
        )
        for fold in folds
    ]
    single_blind = (
        pd.concat(blind_rows, ignore_index=True).sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    )
    router_components = router_components.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    blind = integrate_frozen_router(single_blind, router_components)
    validate_blind_prediction_frame(blind)
    expected = frozen_keys.sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    if not blind[list(PAIR_KEYS)].equals(expected[list(PAIR_KEYS)]):
        raise KMASourceMetaError("blind prediction keys differ from frozen incumbent membership")
    blind_path = output / "blind_predictions.parquet"
    _atomic_parquet(blind_path, blind)
    blind_reloaded = pd.read_parquet(blind_path)
    validate_blind_prediction_frame(blind_reloaded)
    if not blind_reloaded.equals(blind):
        raise KMASourceMetaError("fsynced blind predictions changed after reload")
    implementation = _implementation_hashes()
    if implementation != authorization["implementation_sha256"]:
        raise PermissionError("implementation differs from the authorization amendment")
    manifest = {
        "sealed": True,
        "created_at": _now(),
        "experiment_id": EXPERIMENT_ID,
        "outer_truth_opened": False,
        "domain_auc": auc,
        "domain_route": route.route,
        "inner_gate": inner_gate,
        "blind_predictions_sha256": sha256_file(blind_path),
        "source_meta_seal_sha256": sha256_file(output / "source_meta_seal.json"),
        "frozen_router_components_source_sha256": receipts["frozen_incumbent_oof_keys"],
        "frozen_router_columns": config["frozen_final_integration"]["allowed_pretruth_columns"],
        "incumbent_oof_sha256": receipts["frozen_incumbent_oof_keys"],
        "registered_input_sha256": receipts,
        "implementation_sha256": implementation,
        "authorization_amendment_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "actual_attempt_lock_sha256": sha256_file(GLOBAL_ATTEMPT_LOCK),
        "authorized_dry_receipt_sha256": authorization["dry_receipt_sha256"],
        "validation_target_access_contract": {
            "scope": "fold_local_rolling_origin",
            "current_fold_validation_used_by_own_fold_model": False,
            "prior_fold_validation_allowed_only_in_later_fold_training": True,
            "global_zero_target_exposure_before_blind_seal_claimed": False,
            "designated_scoring_read_performed": False,
            "training_access_log": vault.access_log,
        },
    }
    blind_manifest_path = output / "blind_manifest.json"
    _atomic_json(blind_manifest_path, manifest)
    exposure = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "designated_outer_scoring_read_performed": False,
        "global_zero_target_exposure_before_blind_seal_claimed": False,
        "fold_local_current_validation_used_by_own_fold_model": False,
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
        "blind_predictions_sha256": sha256_file(blind_path),
        "incumbent_oof_sha256": receipts["frozen_incumbent_oof_keys"],
        "authorization_amendment_sha256": sha256_file(CANONICAL_AUTHORIZATION),
        "actual_attempt_lock_sha256": sha256_file(GLOBAL_ATTEMPT_LOCK),
        "fsync_completed_before_outer_open": True,
    }
    exposure_path = output / "outer_exposure_receipt.json"
    _atomic_json(exposure_path, exposure)

    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed after prediction seal")
    if _implementation_hashes() != authorization["implementation_sha256"]:
        raise PermissionError("implementation changed from the authorized dry-run SHA")
    if _verify_registered_inputs(config, p3_data_dir=p3_data_dir) != receipts:
        raise PermissionError("registered input changed after prediction seal")
    ledger_record = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
        "blind_predictions_sha256": sha256_file(blind_path),
        "incumbent_oof_sha256": receipts["frozen_incumbent_oof_keys"],
        "outer_truth_open_attempt": True,
    }
    _write_exclusive(GLOBAL_OUTER_LOCK, ledger_record)
    _write_exclusive(CANONICAL_OUTER_LOCK, ledger_record)
    _append_global_ledger(ledger_record)
    outer_targets = vault.open_outer_once(
        all_outer_ids,
        sealed_manifest=blind_manifest_path,
        exposure_receipt=exposure_path,
    )
    if vault.outer_open_count != 1:
        raise AssertionError("outer target vault did not open exactly once")
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed during outer truth opening")
    if sha256_file(CANONICAL_AUTHORIZATION) != manifest["authorization_amendment_sha256"]:
        raise PermissionError("authorization amendment changed during outer truth opening")
    if sha256_file(_dry_receipt_path()) != authorization["dry_receipt_sha256"]:
        raise PermissionError("authorized dry-run receipt changed during outer truth opening")
    if (
        sha256_file(ROOT / config["frozen_inputs"]["incumbent_oof_keys"]["path"])
        != receipts["frozen_incumbent_oof_keys"]
    ):
        raise PermissionError("incumbent OOF changed during outer truth opening")
    target_long = _targets_to_long(outer_targets, blind_reloaded)
    evaluated = blind_reloaded.merge(
        target_long,
        on=["fold", "anchor_id", "station", "lead_h"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluated) != len(blind_reloaded):
        raise KMASourceMetaError("outer target coverage is incomplete")
    feature_ablation = paired_comparison(
        evaluated,
        config,
        control_column="control_single_prediction",
        challenger_column="challenger_single_prediction",
    )
    final_promotion = evaluate_promotion(
        evaluated,
        config,
        control_column="incumbent_final",
        challenger_column="challenger_final",
    )
    expected_incumbent_rmse = float(
        config["frozen_final_integration"]["expected_exact_incumbent_rmse"]
    )
    if not np.isclose(
        final_promotion["control"]["rmse"],
        expected_incumbent_rmse,
        rtol=0.0,
        atol=1e-12,
    ):
        raise KMASourceMetaError("exact frozen incumbent RMSE did not reconcile after truth open")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "decision": final_promotion["decision"],
        "domain_auc": auc,
        "domain_route": route.route,
        "inner_gate": inner_gate,
        "paired_feature_ablation_591_vs_597": feature_ablation,
        "final_incumbent_promotion": final_promotion,
        "promotion_gate_applied_only_to_final_incumbent_comparison": True,
        "designated_outer_scoring_open_count": vault.outer_open_count,
        "blind_manifest_sha256": sha256_file(blind_manifest_path),
    }
    _atomic_json(output / "result.json", result)
    _status(
        state="actual_complete_one_shot",
        phase="outer_evaluated_once",
        progress=100,
        detail=f"one-shot 완료 · decision {result['decision']}",
        started=started,
        eta=None,
        result={
            "decision": result["decision"],
            "challenger_final_minus_incumbent_rmse": final_promotion[
                "challenger_minus_control_rmse"
            ],
            "designated_outer_scoring_open_count": vault.outer_open_count,
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
