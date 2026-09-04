"""Run fixed external-only I-ORS split-CQR calibration and one 2023 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_QUARANTINE = PROJECT_ROOT / "external_data" / "quarantine"
DEFAULT_OPTIONAL_DEPS = DEFAULT_QUARANTINE / "_deps"
for import_path in (SRC_DIR, DEFAULT_OPTIONAL_DEPS):
    if import_path.is_dir() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np  # noqa: E402

from ocean_external.iors_ctd import (  # noqa: E402
    LooDataset,
    build_loo_dataset,
    ensure_archive,
    load_json_object,
    read_year_profile,
    sha256_file,
    validate_source_manifest,
    verify_archive,
    verify_official_record,
)
from ocean_external.iors_precheck import dataset_audit, evaluate_precheck  # noqa: E402

EXPERIMENT_ID = "p1_iors_external_profile_conformal_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("configs/external_data/i_ors_ctd_v1_1_1.json"),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/p1_iors_external_profile_conformal_v2.json"),
    )
    parser.add_argument("--quarantine-dir", type=Path, default=Path("external_data/quarantine"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p1_iors_external_profile_conformal_v2"),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("artifacts/status/p1_iors_external_profile_conformal_v2.json"),
    )
    parser.add_argument(
        "--download", action="store_true", help="Allow only the pinned ZIP download"
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
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
    ) -> None:
        bounded = min(max(float(progress), 0.0), 100.0)
        elapsed = time.perf_counter() - self.started
        remaining = elapsed * (100.0 - bounded) / bounded if bounded > 0 else None
        eta = (
            datetime.now().astimezone() + timedelta(seconds=max(remaining or 0.0, 0.0))
            if remaining is not None
            else None
        )
        _atomic_json(
            self.path,
            {
                "title": "P1 I-ORS external profile conformal v2",
                "experiment": EXPERIMENT_ID,
                "status": status,
                "phase": phase,
                "progress": round(bounded, 2),
                "detail": detail,
                "elapsed_seconds": round(elapsed, 3),
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if eta is not None else "측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
                "competition_labels_opened": False,
                "competition_oof_opened": False,
                "competition_outer_validation_opened": False,
                "competition_upload": False,
            },
        )


def cqr_widening(
    y: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Return the fixed global non-negative split-CQR order statistic."""

    target = np.asarray(y, dtype=np.float64)
    raw_q10 = np.asarray(q10, dtype=np.float64)
    raw_q90 = np.asarray(q90, dtype=np.float64)
    if not (target.shape == raw_q10.shape == raw_q90.shape) or target.ndim != 1:
        raise ValueError("CQR calibration arrays must be equal one-dimensional shapes")
    if target.size == 0 or not all(
        np.isfinite(value).all() for value in (target, raw_q10, raw_q90)
    ):
        raise ValueError("CQR calibration arrays must be non-empty and finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    lower = np.minimum(raw_q10, raw_q90)
    upper = np.maximum(raw_q10, raw_q90)
    scores = np.maximum.reduce([lower - target, target - upper, np.zeros_like(target)])
    one_based_rank = int(math.ceil((target.size + 1) * (1.0 - alpha)))
    if one_based_rank > target.size:
        raise ValueError("calibration set is too small for the requested finite-sample rank")
    correction = float(np.partition(scores, one_based_rank - 1)[one_based_rank - 1])
    corrected_lower = lower - correction
    corrected_upper = upper + correction
    return {
        "correction": correction,
        "calibration_rows": int(target.size),
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "one_based_rank": one_based_rank,
        "raw_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "corrected_coverage": float(
            np.mean((target >= corrected_lower) & (target <= corrected_upper))
        ),
        "raw_mean_width": float(np.mean(upper - lower)),
        "corrected_mean_width": float(np.mean(corrected_upper - corrected_lower)),
        "crossing_rate_before_reorder": float(np.mean(raw_q10 > raw_q90)),
        "score_definition": "max(lower-y, y-upper, 0)",
        "selection": "ceil((n+1)*(1-alpha)) one-based order statistic",
        "scope": "single global non-negative widening scalar",
    }


def conformal_interval_metrics(
    y: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    *,
    correction: float,
    layer: np.ndarray,
) -> dict[str, Any]:
    """Evaluate fixed corrected bounds without fitting or threshold selection."""

    if correction < 0.0 or not math.isfinite(correction):
        raise ValueError("CQR correction must be finite and non-negative")
    target = np.asarray(y, dtype=np.float64)
    raw_q10 = np.asarray(q10, dtype=np.float64)
    raw_q90 = np.asarray(q90, dtype=np.float64)
    layers = np.asarray(layer)
    if not (target.shape == raw_q10.shape == raw_q90.shape == layers.shape):
        raise ValueError("test arrays must have equal shapes")
    lower = np.minimum(raw_q10, raw_q90) - correction
    upper = np.maximum(raw_q10, raw_q90) + correction
    per_layer: dict[str, Any] = {}
    for layer_value in sorted(int(value) for value in np.unique(layers)):
        mask = layers == layer_value
        per_layer[str(layer_value)] = {
            "rows": int(mask.sum()),
            "coverage": float(
                np.mean((target[mask] >= lower[mask]) & (target[mask] <= upper[mask]))
            ),
            "mean_width": float(np.mean(upper[mask] - lower[mask])),
        }
    return {
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
        "per_layer": per_layer,
    }


def apply_v2_gate(
    metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
    conformal_test: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    source_integrity_verified: bool,
) -> dict[str, Any]:
    """Apply the immutable v2 gate to aggregate external-only metrics."""

    per_layer = list(metrics["per_layer"].values())
    layers_not_worse = sum(
        value["candidate"]["rmse"] <= value["baseline"]["rmse"] for value in per_layer
    )
    maximum_degradation = max(
        (value["candidate"]["rmse"] - value["baseline"]["rmse"]) / value["baseline"]["rmse"]
        for value in per_layer
    )
    checks = {
        "source_integrity": source_integrity_verified
        if bool(gate["all_source_integrity_checks_required"])
        else True,
        "minimum_calibration_rows": int(calibration["calibration_rows"])
        >= int(gate["minimum_calibration_rows"]),
        "minimum_test_rows": int(metrics["rows"]) >= int(gate["minimum_test_rows"]),
        "minimum_evaluated_layers": len(per_layer) >= int(gate["minimum_evaluated_layers"]),
        "minimum_rmse_relative_improvement": float(metrics["rmse_relative_improvement"])
        >= float(gate["minimum_rmse_relative_improvement"]),
        "minimum_mae_relative_improvement": float(metrics["mae_relative_improvement"])
        >= float(gate["minimum_mae_relative_improvement"]),
        "minimum_layers_not_worse": layers_not_worse >= int(gate["minimum_layers_not_worse"]),
        "maximum_single_layer_relative_rmse_degradation": maximum_degradation
        <= float(gate["maximum_single_layer_relative_rmse_degradation"]),
        "conformal_coverage_min": float(conformal_test["coverage"])
        >= float(gate["conformal_coverage_min"]),
        "conformal_coverage_max": float(conformal_test["coverage"])
        <= float(gate["conformal_coverage_max"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "GO_TO_ISOLATED_P1_OOF" if passed else "NO_GO_EXTERNAL_PROFILE",
        "checks": checks,
        "diagnostics": {
            "layers_not_worse": int(layers_not_worse),
            "evaluated_layers": len(per_layer),
            "maximum_single_layer_relative_rmse_degradation": float(maximum_degradation),
        },
        "scope": "external-only follow-up; never submission evidence",
    }


def fit_models_once(
    fit: LooDataset,
    calibration: LooDataset,
    test: LooDataset,
    contract: Mapping[str, Any],
    *,
    progress: Any | None = None,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[str, Any]]:
    """Fit each fixed model once and predict both calibration and test matrices."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM from requirements.txt is required") from exc
    if not (fit.feature_names == calibration.feature_names == test.feature_names):
        raise ValueError("fit/calibration/test feature contracts differ")
    quantiles = tuple(float(value) for value in contract["quantiles"])
    if quantiles != (0.1, 0.5, 0.9):
        raise ValueError("v2 quantiles must remain [0.1, 0.5, 0.9]")
    params = dict(contract["params"])
    calibration_prediction: dict[float, np.ndarray] = {}
    test_prediction: dict[float, np.ndarray] = {}
    audit: dict[str, Any] = {
        "library": "lightgbm",
        "version": str(lgb.__version__),
        "fit_rows": int(fit.y.size),
        "calibration_rows": int(calibration.y.size),
        "test_rows": int(test.y.size),
        "feature_names": list(fit.feature_names),
        "params": params,
        "quantiles": list(quantiles),
        "fit_count": len(quantiles),
        "early_stopping": False,
        "hyperparameter_search": False,
        "models": {},
    }
    for position, quantile in enumerate(quantiles, start=1):
        if progress is not None:
            progress(position - 1, len(quantiles), quantile)
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
        model.fit(fit.x, fit.y)
        cal_value = np.asarray(model.predict(calibration.x), dtype=np.float64)
        test_value = np.asarray(model.predict(test.x), dtype=np.float64)
        if not np.isfinite(cal_value).all() or not np.isfinite(test_value).all():
            raise RuntimeError(f"non-finite prediction for q={quantile}")
        calibration_prediction[quantile] = cal_value
        test_prediction[quantile] = test_value
        importance = np.asarray(
            model.booster_.feature_importance(importance_type="gain"), dtype=float
        )
        total_gain = float(importance.sum())
        ranked = np.argsort(-importance)[:20]
        audit["models"][str(quantile)] = {
            "trees": int(model.booster_.num_trees()),
            "top_gain_features": [
                {
                    "feature": fit.feature_names[int(index)],
                    "gain_fraction": float(importance[index] / total_gain)
                    if total_gain > 0
                    else 0.0,
                }
                for index in ranked
            ],
        }
    if progress is not None:
        progress(len(quantiles), len(quantiles), math.nan)
    return calibration_prediction, test_prediction, audit


def _permission_audit(path: Path, source_id: str) -> dict[str, Any]:
    receipt = load_json_object(path)
    evidence = _resolve(Path(str(receipt["evidence_file"])))
    evidence_sha = sha256_file(evidence)
    checks = {
        "status_approved": receipt.get("status") == "approved",
        "source_allowed": source_id in receipt.get("allowed_sources", []),
        "problem_allowed": "P1" in receipt.get("allowed_problems", []),
        "pretraining_allowed": "pretraining" in receipt.get("allowed_purposes", []),
        "evidence_sha256": evidence_sha == str(receipt["evidence_sha256"]).lower(),
    }
    if not all(checks.values()):
        raise PermissionError(f"official external-data FAQ permission failed: {checks}")
    return {
        "receipt": str(path.relative_to(PROJECT_ROOT)),
        "receipt_sha256": sha256_file(path),
        "organizer_channel": receipt.get("organizer_channel"),
        "evidence": str(evidence.relative_to(PROJECT_ROOT)),
        "evidence_sha256": evidence_sha,
        "checks": checks,
    }


def _verify_v1_provenance(experiment: Mapping[str, Any]) -> dict[str, Any]:
    provenance = experiment["cross_experiment_provenance"]
    result_path = _resolve(Path(provenance["v1_result"]))
    receipt_path = _resolve(Path(provenance["v1_receipt"]))
    result_sha = sha256_file(result_path)
    receipt_sha = sha256_file(receipt_path)
    checks = {
        "result_unchanged": result_sha == provenance["v1_result_sha256"],
        "receipt_unchanged": receipt_sha == provenance["v1_receipt_sha256"],
        "global_2023_first_look_disclosed": provenance["global_2023_first_look"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v1 provenance changed or caveat missing: {checks}")
    return {
        "v1_result": str(result_path),
        "v1_result_sha256": result_sha,
        "v1_receipt": str(receipt_path),
        "v1_receipt_sha256": receipt_sha,
        "v1_pre_format_code_sha256": provenance["v1_pre_format_code_sha256"],
        "checks": checks,
        "caveat": provenance["caveat"],
    }


def _environment() -> dict[str, Any]:
    import h5py
    import lightgbm

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
        git_dirty = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "lightgbm": lightgbm.__version__,
        "git": {"sha": git_sha, "dirty": git_dirty},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = _resolve(args.source_manifest)
    experiment_path = _resolve(args.experiment)
    quarantine = _resolve(args.quarantine_dir)
    output_root = _resolve(args.output_dir)
    status = Status(_resolve(args.status_file))
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    try:
        status.update(2, "contract", "v1 SHA 보존·v2 외부-only 고정 계약 검사")
        source = load_json_object(source_path)
        experiment = load_json_object(experiment_path)
        validate_source_manifest(source)
        if experiment.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("unexpected experiment contract")
        if experiment.get("status") != "preregistered_external_only":
            raise ValueError("v2 must remain preregistered_external_only")
        if experiment["model"].get("hyperparameter_search") is not False:
            raise ValueError("v2 hyperparameter search is forbidden")
        if experiment["conformal"].get("grid_search") is not False:
            raise ValueError("v2 conformal grid search is forbidden")
        catalog_path = _resolve(Path(experiment["external_catalog"]))
        catalog_sha = sha256_file(catalog_path)
        if catalog_sha != experiment["external_catalog_sha256"]:
            raise RuntimeError("external catalog SHA changed after v2 preregistration")
        v1_provenance = _verify_v1_provenance(experiment)
        permission = _permission_audit(
            PROJECT_ROOT / "configs/external_data/official_faq_permission.json",
            str(source["source_id"]),
        )

        status.update(7, "provenance", "KIOST 공식 DOI·CC BY·v1.1.1 실시간 확인")
        official_record = verify_official_record(source)
        archive_file = ensure_archive(source, quarantine, allow_download=args.download)
        archive_audit = verify_archive(archive_file, source)

        split = experiment["split"]
        fit_years = set(int(value) for value in split["fit_years"])
        calibration_year = int(split["calibration_year"])
        test_year = int(split["test_year"])
        if fit_years != set(range(2014, 2022)) or calibration_year != 2022 or test_year != 2023:
            raise ValueError("v2 year split must be 2014-21 / 2022 / 2023")
        if fit_years & {calibration_year, test_year} or calibration_year == test_year:
            raise ValueError("fit/calibration/test years overlap")
        target_depths = {
            int(layer): float(depth) for layer, depth in experiment["target_grid"]["layers"].items()
        }
        max_distance = float(experiment["target_grid"]["max_mapping_distance_m"])
        years = sorted(fit_years | {calibration_year, test_year})
        profiles = []
        for position, year in enumerate(years, start=1):
            status.update(
                12 + 28 * position / len(years),
                "decode",
                f"{year} QC1·실제 수심 profile ({position}/{len(years)})",
            )
            profiles.append(
                read_year_profile(
                    archive_file,
                    source,
                    year=year,
                    target_depth_by_layer=target_depths,
                    max_mapping_distance_m=max_distance,
                )
            )

        status.update(45, "features", "2014–21 fit·2022 calibration·2023 test LOO 분리")
        fit = build_loo_dataset(
            [value for value in profiles if value.year in fit_years],
            min_peer_temperatures=int(split["min_peer_temperatures"]),
            max_rows_per_year_layer=int(split["max_fit_rows_per_year_layer"]),
        )
        calibration = build_loo_dataset(
            [value for value in profiles if value.year == calibration_year],
            min_peer_temperatures=int(split["min_peer_temperatures"]),
            max_rows_per_year_layer=None,
        )
        test = build_loo_dataset(
            [value for value in profiles if value.year == test_year],
            min_peer_temperatures=int(split["min_peer_temperatures"]),
            max_rows_per_year_layer=None,
        )
        if set(int(value) for value in np.unique(fit.year)) != fit_years:
            raise RuntimeError("fit year isolation failed")
        if set(int(value) for value in np.unique(calibration.year)) != {calibration_year}:
            raise RuntimeError("calibration year isolation failed")
        if set(int(value) for value in np.unique(test.year)) != {test_year}:
            raise RuntimeError("test year isolation failed")

        def model_progress(done: int, total: int, quantile: float) -> None:
            if done >= total:
                status.update(79, "model", "고정 q10/q50/q90 모델의 calibration/test 추론 완료")
            else:
                status.update(
                    52 + 26 * done / total,
                    "model",
                    f"q={quantile:.1f} 동일 300-tree 모델 ({done + 1}/{total})",
                )

        calibration_prediction, test_prediction, model_audit = fit_models_once(
            fit,
            calibration,
            test,
            experiment["model"],
            progress=model_progress,
        )

        status.update(82, "calibration", "2022에서 단일 global CQR order statistic 산출")
        alpha = float(experiment["conformal"]["alpha"])
        conformal_calibration = cqr_widening(
            calibration.y,
            calibration_prediction[0.1],
            calibration_prediction[0.9],
            alpha=alpha,
        )
        status.update(88, "test", "고정 correction으로 2023 aggregate test 1회 평가")
        point_metrics = evaluate_precheck(test, test_prediction)
        conformal_test = conformal_interval_metrics(
            test.y,
            test_prediction[0.1],
            test_prediction[0.9],
            correction=float(conformal_calibration["correction"]),
            layer=test.layer,
        )
        source_integrity = bool(archive_audit["integrity_verified"]) and bool(
            official_record["markers_verified"]
        )
        decision = apply_v2_gate(
            point_metrics,
            conformal_calibration,
            conformal_test,
            experiment["stop_gate"],
            source_integrity_verified=source_integrity,
        )

        run_id = started_at.strftime("%Y%m%dT%H%M%S%z")
        run_dir = output_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run output already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        result = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "scope": {
                "competition_labels_opened": False,
                "competition_oof_opened": False,
                "competition_outer_validation_opened": False,
                "competition_submission_created": False,
                "raw_observational_rows_serialized": False,
                "test_2023_used_for_fit_or_calibration": False,
                "global_2023_first_look": False,
            },
            "contracts": {
                "source_manifest": str(source_path.relative_to(PROJECT_ROOT)),
                "source_manifest_sha256": sha256_file(source_path),
                "experiment": str(experiment_path.relative_to(PROJECT_ROOT)),
                "experiment_sha256": sha256_file(experiment_path),
                "external_catalog": str(catalog_path.relative_to(PROJECT_ROOT)),
                "external_catalog_sha256": catalog_sha,
                "effective_contract_sha256": _sha256_json(
                    {"source": source, "experiment": experiment}
                ),
            },
            "v1_provenance": v1_provenance,
            "permission": permission,
            "official_record": official_record,
            "archive": archive_audit,
            "profile_quality": [value.audit for value in profiles],
            "fit_dataset": dataset_audit(fit),
            "calibration_dataset": dataset_audit(calibration),
            "test_dataset": dataset_audit(test),
            "model": model_audit,
            "conformal_calibration": conformal_calibration,
            "point_metrics": point_metrics,
            "conformal_test": conformal_test,
            "gate": decision,
            "environment": _environment(),
        }
        result_path = run_dir / "result.json"
        _atomic_json(result_path, result)
        receipt = {
            "result": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "archive_sha256": archive_audit["sha256"],
            "source_manifest_sha256": sha256_file(source_path),
            "experiment_sha256": sha256_file(experiment_path),
            "v1_result_sha256": v1_provenance["v1_result_sha256"],
            "v1_receipt_sha256": v1_provenance["v1_receipt_sha256"],
            "decision": decision["decision"],
            "competition_upload": False,
        }
        _atomic_json(run_dir / "receipt.json", receipt)
        status.update(
            100,
            "complete",
            f"{decision['decision']} · result SHA {receipt['result_sha256'][:12]}",
            status="complete",
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        status.update(0, "failed", f"{type(exc).__name__}: {exc}", status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
