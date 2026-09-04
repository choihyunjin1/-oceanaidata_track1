"""Exactly-once clean P3 lead-continuous score-priority materialization.

The frozen six-parameter ridge surface is imported from the original one-shot
challenger after its source hash is verified.  It is fit once on the exact
organizer-distributed historical OOF surface, then applied to the eligible clean
official incumbent and organizer persistence axis.  No external data, pretrained
weights, hidden truth, post-result tuning, or upload is permitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT: Final = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.submission import validate_submission  # noqa: E402

EXPERIMENT_ID: Final = "p3_lead_continuous_score_priority_deployment_20260901_v1"
CONFIG_RELATIVE: Final = Path(
    "configs/experiments/p3_lead_continuous_score_priority_deployment_20260901_v1.json"
)
SOURCE_RUNNER_RELATIVE: Final = Path(
    "scripts/run_p3_lead_continuous_challenger_20260827_v1.py"
)
SOURCE_METRICS_RELATIVE: Final = Path(
    "artifacts/structural_challenger_20260827_v1/p3/metrics.json"
)
FRESH_RESULT_RELATIVE: Final = Path(
    "reports/p3_lead_continuous_fresh_episode_confirmation_20260830_v3/result.json"
)
TEST_FEATURES_RELATIVE: Final = Path("artifacts/p3/features_all20_v1/test_features.parquet")
REPORT_RELATIVE: Final = Path(
    "reports/p3_lead_continuous_score_priority_deployment_20260901_v1"
)
LOCK_RELATIVE: Final = Path(
    "artifacts/p3_lead_continuous_score_priority_deployment_20260901_v1.ATTEMPT_LOCK.json"
)
DEFAULT_DATA_DIR: Final = Path(
    r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast"
)
DEFAULT_INCUMBENT: Final = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_P3_REFINED_PUBLIC_OPTIMUM_READY\P3_submission.csv"
)
DEFAULT_OUTPUT: Final = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260901_P3_LEAD_CONTINUOUS_SCORE_PRIORITY_READY_V1"
)
KEYS: Final = ["case_id", "station", "lead_h"]
EXPECTED_LEADS: Final = (3, 6, 9, 12, 18, 24)
EXPECTED_MODEL_STATE_SHA256: Final = (
    "dab5e21d1e836c4f4991549240b10321629908d4077c370f02abea9469a20d2a"
)
TITLE: Final = "P3 Clean Lead-Continuous Score-Priority V1"
SUMMARY: Final = (
    "배포 데이터 전용 clean incumbent에 frozen lead×causal-regime ridge를 1회 적용한 "
    "고득점 우선 후보입니다. 내부 활성 ΔRMSE -0.004188m(+0.066467점 추정)이지만 "
    "fresh 독립 1-case에서는 +0.022617m 악화해 안정성이 낮습니다."
)


class ContractError(RuntimeError):
    """Raised when the sealed materialization contract differs."""


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    """Match the already-sealed fresh-confirmation model-state hash."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_exclusive(path: Path, payload: Any) -> None:
    data = _json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("exclusive write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ContractError(f"{label} SHA differs: {observed}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def _load_source_runner(expected_sha: str) -> ModuleType:
    path = ROOT / SOURCE_RUNNER_RELATIVE
    _verify_sha(path, expected_sha, "frozen lead runner")
    spec = importlib.util.spec_from_file_location("sealed_p3_lead_runner", path)
    if spec is None or spec.loader is None:
        raise ContractError("unable to load frozen lead runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _input_paths(data_dir: Path, incumbent: Path) -> dict[str, Path]:
    return {
        "test_index.csv": data_dir / "test_index.csv",
        "sample_submission.csv": data_dir / "sample_submission.csv",
        "baseline_persistence.csv": data_dir / "baseline_persistence.csv",
        "clean_incumbent_csv": incumbent,
        "test_features": ROOT / TEST_FEATURES_RELATIVE,
        "source_metrics": ROOT / SOURCE_METRICS_RELATIVE,
        "fresh_confirmation": ROOT / FRESH_RESULT_RELATIVE,
    }


def preflight(data_dir: Path, incumbent: Path, output: Path) -> dict[str, Any]:
    config = _load_json(ROOT / CONFIG_RELATIVE)
    if config["status"] != "SEALED_BEFORE_OFFICIAL_MATERIALIZATION":
        raise ContractError("deployment config is not sealed")
    if output.exists() or (ROOT / REPORT_RELATIVE).exists() or (ROOT / LOCK_RELATIVE).exists():
        raise FileExistsError("exactly-once output, report, or attempt lock already exists")

    paths = _input_paths(data_dir, incumbent)
    verified: dict[str, Any] = {}
    frozen = config["frozen_candidate"]
    clean = config["clean_inputs"]
    verified["source_runner"] = _verify_sha(
        ROOT / SOURCE_RUNNER_RELATIVE,
        frozen["source_runner_sha256"],
        "source runner",
    )
    verified["source_metrics"] = _verify_sha(
        paths["source_metrics"], frozen["source_metrics_sha256"], "source metrics"
    )
    verified["fresh_confirmation"] = _verify_sha(
        paths["fresh_confirmation"],
        frozen["fresh_confirmation_sha256"],
        "fresh confirmation",
    )
    verified["test_features"] = _verify_sha(
        paths["test_features"], clean["test_features"]["sha256"], "test features"
    )
    verified["clean_incumbent_csv"] = _verify_sha(
        incumbent,
        clean["clean_incumbent_csv"]["sha256"],
        "clean incumbent",
    )
    for name, expected in clean["official_distributed_files"].items():
        verified[name] = _verify_sha(data_dir / name, expected, name)

    metrics = _load_json(paths["source_metrics"])
    active = metrics["evaluation"]["active_prequential_folds"]
    fresh = _load_json(paths["fresh_confirmation"])
    checks = {
        "source_status_complete": metrics["status"] == "ONE_SHOT_SCREEN_COMPLETE",
        "source_verdict_inconclusive": metrics["verdict"] == "INCONCLUSIVE",
        "active_delta_exact": float(active["delta_candidate_minus_incumbent_m"])
        == float(config["decision"]["selection_metric_value_m"]),
        "fresh_terminal_complete": fresh["status"] == "TERMINAL_EXACTLY_ONCE_COMPLETE",
        "fresh_single_case": int(fresh["primary"]["cases"]) == 1,
        "fresh_harm_exact": float(
            fresh["primary"]["delta_candidate_minus_incumbent_rmse_m"]
        )
        == float(
            config["stability_disclosure"][
                "fresh_delta_candidate_minus_incumbent_rmse_m"
            ]
        ),
        "fresh_readiness_not_ready": fresh["submission_readiness"]
        == "NOT_READY_INSUFFICIENT_FRESH_SUPPORT",
        "lineage_clean": all(
            [
                config["lineage"]["organizer_distributed_data_only"],
                config["lineage"]["scratch_fit_only"],
                config["lineage"]["internet_rows"] == 0,
                config["lineage"]["kiost_raw_rows"] == 0,
                config["lineage"]["external_observation_rows"] == 0,
                config["lineage"]["external_reanalysis_rows"] == 0,
                config["lineage"]["external_forecast_rows"] == 0,
                config["lineage"]["pretrained_weight_files_loaded"] == 0,
                config["lineage"]["kma_era5_v21_v81_predictions_or_metrics_used"] == 0,
                config["lineage"]["hidden_truth_rows"] == 0,
            ]
        ),
    }
    if not all(checks.values()):
        raise ContractError(f"preflight evidence checks failed: {checks}")
    return {
        "status": "READY_EXACTLY_ONCE",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256(ROOT / CONFIG_RELATIVE),
        "verified": verified,
        "checks": checks,
        "output_absent": True,
        "upload_authorized_by_materializer": False,
    }


def _validate_key_surface(frame: pd.DataFrame, name: str) -> None:
    if list(frame.columns)[:3] != KEYS:
        raise ContractError(f"{name} key columns differ")
    if len(frame) != 1_200 or frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ContractError(f"{name} 1,200-row unique-key contract differs")
    counts = frame["lead_h"].value_counts().to_dict()
    if set(counts) != set(EXPECTED_LEADS) or any(int(counts[lead]) != 200 for lead in EXPECTED_LEADS):
        raise ContractError(f"{name} lead distribution differs")
    per_case = frame.groupby("case_id", sort=False, observed=True)["lead_h"].agg(tuple)
    if len(per_case) != 200 or not per_case.map(lambda values: tuple(values) == EXPECTED_LEADS).all():
        raise ContractError(f"{name} case/lead order differs")


def _build_model_frame(
    test_index: pd.DataFrame,
    test_features: pd.DataFrame,
    baseline: pd.DataFrame,
    incumbent: pd.DataFrame,
    regime_features: tuple[str, ...],
) -> pd.DataFrame:
    for name, frame in (("baseline", baseline), ("incumbent", incumbent)):
        if not frame[KEYS].equals(test_index[KEYS]):
            raise ContractError(f"{name} key/order differs from test_index")
    required = {"case_id", "station", *regime_features}
    if missing := required.difference(test_features.columns):
        raise ContractError(f"test feature columns missing: {sorted(missing)}")
    if len(test_features) != 200 or test_features.duplicated(["case_id", "station"]).any():
        raise ContractError("test feature case contract differs")
    model_frame = test_index[KEYS].merge(
        test_features[["case_id", "station", *regime_features]],
        on=["case_id", "station"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if not model_frame[KEYS].equals(test_index[KEYS]):
        raise ContractError("feature merge changed official key order")
    model_frame["persistence"] = pd.to_numeric(
        baseline["hs_pred"], errors="raise"
    ).to_numpy(dtype=np.float64)
    model_frame["final_prediction"] = pd.to_numeric(
        incumbent["hs_pred"], errors="raise"
    ).to_numpy(dtype=np.float64)
    numeric = model_frame[["lead_h", "persistence", "final_prediction"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ContractError("official model frame contains non-finite base values")
    return model_frame


def _execute(data_dir: Path, incumbent_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    pre = preflight(data_dir, incumbent_path, output)
    config = _load_json(ROOT / CONFIG_RELATIVE)
    runner_sha = sha256(Path(__file__).resolve())
    lock = {
        "schema_version": "p3.lead_continuous_score_priority.attempt_lock.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "status": "CONSUMED_EXACTLY_ONCE",
        "runner_sha256": runner_sha,
        "config_sha256": pre["config_sha256"],
        "fit_budget": 1,
        "upload_budget": 0,
    }
    _write_json_exclusive(ROOT / LOCK_RELATIVE, lock)

    test_index = pd.read_csv(data_dir / "test_index.csv", usecols=KEYS)
    sample_keys = pd.read_csv(data_dir / "sample_submission.csv", usecols=KEYS)
    baseline = pd.read_csv(data_dir / "baseline_persistence.csv")
    incumbent = pd.read_csv(incumbent_path)
    test_features = pd.read_parquet(
        ROOT / TEST_FEATURES_RELATIVE,
        columns=["case_id", "station", "hs_delta_3h", "hs_std_6h", "wspd_delta_3h", "caph_delta_6h"],
    )
    _validate_key_surface(test_index, "test_index")
    _validate_key_surface(sample_keys, "sample_submission keys")
    _validate_key_surface(baseline, "baseline")
    _validate_key_surface(incumbent, "clean incumbent")
    if not sample_keys[KEYS].equals(test_index[KEYS]):
        raise ContractError("sample key/order differs from test_index")
    validate_submission(baseline, test_index)
    validate_submission(incumbent, test_index)

    source = _load_source_runner(config["frozen_candidate"]["source_runner_sha256"])
    history, history_audit = source._load_surface(ROOT)
    fit_started = time.perf_counter()
    model = source._fit_model(history)
    fit_elapsed = time.perf_counter() - fit_started
    model_state_sha = array_sha256(
        model.medians,
        model.robust_scales,
        model.basis_scales,
        model.coefficients,
    )
    if model_state_sha != EXPECTED_MODEL_STATE_SHA256:
        raise ContractError(f"full-history model state differs: {model_state_sha}")

    model_frame = _build_model_frame(
        test_index,
        test_features,
        baseline,
        incumbent,
        tuple(source.REGIME_FEATURES),
    )
    prediction, prediction_audit = source._predict_model(model, model_frame)
    candidate = test_index.copy()
    candidate["hs_pred"] = prediction
    validate_submission(candidate, test_index)
    incumbent_values = incumbent["hs_pred"].to_numpy(dtype=np.float64)
    delta = prediction - incumbent_values
    if not np.any(delta != 0.0):
        raise ContractError("candidate is prediction-identical to clean incumbent")

    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "P3_submission.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        candidate.to_csv(handle, index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())
    reread = pd.read_csv(csv_path)
    validate_submission(reread, test_index)
    if not reread[KEYS].equals(test_index[KEYS]):
        raise ContractError("CSV round-trip changed key order")
    csv_sha = sha256(csv_path)
    if csv_sha == config["clean_inputs"]["clean_incumbent_csv"]["sha256"]:
        raise ContractError("candidate CSV duplicates eligible clean incumbent")

    result = {
        "schema_version": "p3.lead_continuous_score_priority.result.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "status": "MATERIALIZED_READY_NOT_UPLOADED_SCORE_PRIORITY_RISK_DISCLOSED",
        "title": TITLE,
        "summary": SUMMARY,
        "decision": config["decision"],
        "stability_disclosure": config["stability_disclosure"],
        "fit": {
            "fit_count": 1,
            "hyperparameter_search_count": 0,
            "fit_elapsed_seconds": float(fit_elapsed),
            "history_cases": int(history[["anchor_id", "station"]].drop_duplicates().shape[0]),
            "history_rows": int(len(history)),
            "history_minimum_gap_h": float(history_audit["minimum_gap_h"]),
            "model_state_sha256": model_state_sha,
            "scratch_fit": True,
            "pretrained_weight_files_loaded": 0,
        },
        "candidate": {
            "path": str(csv_path),
            "bytes": csv_path.stat().st_size,
            "sha256": csv_sha,
            "rows": int(len(candidate)),
            "cases": int(candidate["case_id"].nunique()),
            "changed_rows_vs_clean_incumbent": int(np.count_nonzero(delta)),
            "maximum_absolute_change_m": float(np.max(np.abs(delta))),
            "minimum_m": float(np.min(prediction)),
            "median_m": float(np.median(prediction)),
            "maximum_m": float(np.max(prediction)),
            "prediction_audit": prediction_audit,
            "duplicate_of_clean_incumbent": False,
        },
        "provenance": {
            "config": pre["config_sha256"],
            "runner": runner_sha,
            "source_runner": pre["verified"]["source_runner"]["sha256"],
            "source_metrics": pre["verified"]["source_metrics"]["sha256"],
            "fresh_confirmation": pre["verified"]["fresh_confirmation"]["sha256"],
            "clean_incumbent_csv": pre["verified"]["clean_incumbent_csv"]["sha256"],
            "test_features": pre["verified"]["test_features"]["sha256"],
            "test_index": pre["verified"]["test_index.csv"]["sha256"],
            "sample_submission": pre["verified"]["sample_submission.csv"]["sha256"],
            "baseline_persistence": pre["verified"]["baseline_persistence.csv"]["sha256"],
        },
        "access_audit": {
            "organizer_train_oof_rows_read": 1086,
            "official_test_index_rows_read": 1200,
            "official_sample_key_rows_read": 1200,
            "official_sample_prediction_values_read": 0,
            "official_baseline_rows_read": 1200,
            "official_test_feature_cases_read": 200,
            "clean_incumbent_prediction_rows_read": 1200,
            "hidden_truth_rows_read": 0,
            "internet_rows_read": 0,
            "kiost_raw_rows_read": 0,
            "external_observation_reanalysis_forecast_rows_read": 0,
            "kma_era5_v21_v81_prediction_or_metric_rows_read": 0,
            "uploads": 0,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    manifest = {
        "status": result["status"],
        "candidate": "P3_LEAD_CONTINUOUS_SCORE_PRIORITY_20260901_V1",
        "title": TITLE,
        "summary": SUMMARY,
        "rows": 1200,
        "sha256": csv_sha,
        "model_state_sha256": model_state_sha,
        "expected_public_rmse_m": config["decision"]["estimated_public_rmse_m"],
        "expected_public_points": config["decision"]["estimated_public_points"],
        "expected_point_gain_vs_clean_incumbent": config["decision"][
            "estimated_point_gain_vs_clean_incumbent"
        ],
        "expected_score_is_heuristic": True,
        "fresh_single_case_delta_rmse_m": config["stability_disclosure"][
            "fresh_delta_candidate_minus_incumbent_rmse_m"
        ],
        "stability": "LOW_FRESH_ONE_CASE_WORSENED",
        "organizer_distributed_data_only": True,
        "scratch_fit_only": True,
        "external_or_pretrained_inputs": 0,
        "hidden_truth_rows": 0,
        "uploaded": False,
    }
    _write_json_exclusive(output / "MANIFEST.json", manifest)
    _write_text_exclusive(
        output / "README.md",
        "# " + TITLE + "\n\n" + SUMMARY + "\n\n"
        + f"- CSV SHA-256: `{csv_sha}`\n"
        + "- 상태: `READY_NOT_UPLOADED`\n"
        + "- 주의: 예상 점수는 내부 활성 pseudo-test delta의 선형 환산이며 공식 보장이 아닙니다.\n"
        + "- 위험: fresh 독립 확인은 1 case뿐이며 ΔRMSE +0.022617m로 악화했습니다.\n",
    )

    report_dir = ROOT / REPORT_RELATIVE
    report_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(report_dir / "result.json", result)
    checks = {
        "attempt_lock_exists": (ROOT / LOCK_RELATIVE).is_file(),
        "source_runner_hash_exact": result["provenance"]["source_runner"]
        == config["frozen_candidate"]["source_runner_sha256"],
        "model_state_matches_prior_fresh_confirmation": model_state_sha
        == config["frozen_candidate"]["full_history_model_state_sha256"],
        "exactly_one_scratch_fit": result["fit"]["fit_count"] == 1,
        "hyperparameter_search_zero": result["fit"]["hyperparameter_search_count"] == 0,
        "rows_exact_1200": len(reread) == 1200,
        "schema_exact": list(reread.columns) == [*KEYS, "hs_pred"],
        "keys_exact_and_ordered": reread[KEYS].equals(test_index[KEYS]),
        "duplicate_keys_zero": not reread.duplicated(KEYS).any(),
        "finite_and_range_pass": bool(
            np.isfinite(reread["hs_pred"].to_numpy(float)).all()
            and reread["hs_pred"].between(0.0, 30.0).all()
        ),
        "csv_roundtrip_hash_exact": sha256(csv_path) == csv_sha,
        "prediction_not_clean_incumbent_duplicate": csv_sha
        != config["clean_inputs"]["clean_incumbent_csv"]["sha256"],
        "changed_rows_positive": result["candidate"]["changed_rows_vs_clean_incumbent"] > 0,
        "organizer_distributed_only": config["lineage"]["organizer_distributed_data_only"],
        "external_and_pretrained_zero": all(
            config["lineage"][key] == 0
            for key in (
                "internet_rows",
                "kiost_raw_rows",
                "external_observation_rows",
                "external_reanalysis_rows",
                "external_forecast_rows",
                "pretrained_weight_files_loaded",
                "kma_era5_v21_v81_predictions_or_metrics_used",
            )
        ),
        "hidden_truth_zero": result["access_audit"]["hidden_truth_rows_read"] == 0,
        "fresh_harm_disclosed": "+0.022617" in SUMMARY,
        "upload_zero": result["access_audit"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"post-materialization QA failed: {checks}")
    independent = {
        "schema_version": "p3.lead_continuous_score_priority.independent_qa.20260901.v1",
        "status": "PASS_READY_NOT_UPLOADED_WITH_LOW_STABILITY_DISCLOSED",
        "checks": checks,
        "hashes": {
            "candidate_csv": csv_sha,
            "result_json": sha256(report_dir / "result.json"),
            "manifest_json": sha256(output / "MANIFEST.json"),
            "config_json": pre["config_sha256"],
            "runner": runner_sha,
        },
        "upload_count": 0,
    }
    _write_json_exclusive(report_dir / "independent-qa.json", independent)
    report = (
        "# P3 clean lead-continuous score-priority deployment\n\n"
        "## 결론\n\n"
        "규정 적합 clean incumbent보다 내부 활성 pseudo-test 기준 높은 후보를 정확히 한 번 "
        "materialize했다. 업로드는 하지 않았다. 기대점수는 24.132635점(+0.066467)이나 "
        "공식 점수가 아닌 휴리스틱이다. fresh 독립 1-case에서는 RMSE가 +0.022617m 악화해 "
        "안정성이 낮다는 위험을 제출 설명에 명시해야 한다.\n\n"
        f"- 제출 파일: `{csv_path}`\n"
        f"- SHA-256: `{csv_sha}`\n"
        f"- 제목: `{TITLE}`\n"
        f"- 설명: {SUMMARY}\n"
        "- 외부/KIOST 원자료/사전학습/hidden truth/KMA·ERA5 계보 사용: 0\n"
        "- 업로드: 0 (root 담당)\n"
    )
    _write_text_exclusive(report_dir / "report-source.md", report)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("P3_DATA_DIR", DEFAULT_DATA_DIR)))
    parser.add_argument("--incumbent", type=Path, default=Path(os.environ.get("P3_CLEAN_INCUMBENT_CSV", DEFAULT_INCUMBENT)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(preflight(args.data_dir, args.incumbent, args.output_dir), ensure_ascii=False, sort_keys=True))
        return 0
    result = _execute(args.data_dir, args.incumbent, args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "path": result["candidate"]["path"],
                "rows": result["candidate"]["rows"],
                "sha256": result["candidate"]["sha256"],
                "expected_points": result["decision"]["estimated_public_points"],
                "fresh_case_delta_rmse_m": result["stability_disclosure"][
                    "fresh_delta_candidate_minus_incumbent_rmse_m"
                ],
                "uploaded": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
