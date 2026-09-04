"""Run the adaptive, support-aware P2 safe residual-gate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_p2_public_state_soft_gate import (
    CONTRIBUTORS,
    _load_oof_stack,
    _metrics,
    _paired_day_bootstrap,
)

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.regime_gate import (
    STATE_FEATURES,
    build_public_state_features,
    fit_soft_gate,
    predict_simplex_baseline,
    predict_soft_gate,
)
from p2_restore.safe_residual_gate import (
    apply_safe_calibrator,
    calibrator_summary,
    fit_safe_calibrator,
)
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(self, progress: float, phase: str, detail: str, *, status: str = "running") -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(progress, 0.1), 100.0)
        eta = datetime.now().astimezone() + timedelta(
            seconds=max(0.0, elapsed * (100.0 - bounded) / bounded)
        )
        _write_json(
            self.path,
            {
                "title": "P2 Safe Residual Gate 적응형 로컬 실험",
                "status": status,
                "progress": bounded,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _load_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_safe_residual_gate_v1":
        raise ValueError("unexpected safe residual experiment id")
    if value.get("status") != "authorized_adaptive_local_test":
        raise ValueError("safe residual experiment is not authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("safe residual experiment must remain local-only")
    if value.get("fresh_holdout_claimed") is not False:
        raise ValueError("adaptive experiment cannot claim a fresh holdout")
    return value


def _crossfit_inner(frame: pd.DataFrame, blocks: list[str], regularization: float) -> tuple:
    baseline = np.full(len(frame), np.nan)
    raw = np.full(len(frame), np.nan)
    block_values = frame["block"].astype(str).to_numpy()
    for held in blocks:
        validation = block_values == held
        train = np.isin(block_values, [block for block in blocks if block != held])
        gate = fit_soft_gate(
            frame.loc[train],
            prediction_columns=CONTRIBUTORS,
            regularization=regularization,
        )
        baseline[validation] = predict_simplex_baseline(
            frame.loc[train], frame.loc[validation], CONTRIBUTORS
        )
        raw[validation] = predict_soft_gate(gate, frame.loc[validation])
    selected = np.isin(block_values, blocks)
    if not np.isfinite(baseline[selected]).all() or not np.isfinite(raw[selected]).all():
        raise ValueError("inner cross-fit is incomplete")
    return baseline[selected], raw[selected]


def _active_share(frame: pd.DataFrame, baseline: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(baseline)) > 1e-12))


def _extended_metrics(
    frame: pd.DataFrame, baseline: np.ndarray, raw: np.ndarray, safe: np.ndarray
) -> dict[str, object]:
    result = _metrics(frame, baseline, safe)
    truth = frame["truth"].to_numpy(float)
    result["raw_lambda10_rmse"] = float(np.sqrt(np.mean((raw - truth) ** 2)))
    result["raw_lambda10_delta_rmse"] = result["raw_lambda10_rmse"] - result["baseline_rmse"]
    result["active_row_share"] = _active_share(frame, baseline, safe)
    result["safe_correction_rmse"] = float(np.sqrt(np.mean((safe - baseline) ** 2)))
    result["safe_correction_max_abs"] = float(np.max(np.abs(safe - baseline)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/experiments/p2_safe_residual_gate_v1.json"),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("artifacts/status/p2_safe_residual_gate.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    progress = Progress(args.status_file)
    progress.update(3, "계약 확인", "적응형 연구·no upload·no fresh holdout 고정")
    contract = _load_contract(args.contract)
    regularization = float(contract["raw_gate"]["regularization"])
    safety = contract["safety_calibrator"]
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)
    output_dir = Path(contract["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    progress.update(10, "OOF·공개 상태 정렬", "5개 frozen contributor · 69,850행")
    stack = _load_oof_stack()
    public = build_public_state_features(data.observations, stack[["time", "layer"]])
    stack = pd.concat(
        [stack.reset_index(drop=True), public.loc[:, STATE_FEATURES].reset_index(drop=True)], axis=1
    )
    blocks = sorted(stack["block"].unique())
    block_values = stack["block"].astype(str).to_numpy()
    baseline = np.full(len(stack), np.nan)
    raw = np.full(len(stack), np.nan)
    safe_prediction = np.full(len(stack), np.nan)
    outer_summaries: dict[str, object] = {}

    for number, outer in enumerate(blocks):
        progress.update(
            18 + number * 18,
            "Nested safety calibration",
            f"outer {outer} · λ={regularization:g} · exact no-op fallback",
        )
        validation = block_values == outer
        train = ~validation
        train_blocks = [block for block in blocks if block != outer]
        inner_baseline, inner_raw = _crossfit_inner(
            stack.loc[train].reset_index(drop=True), train_blocks, regularization
        )
        calibrator = fit_safe_calibrator(
            stack.loc[train].reset_index(drop=True),
            inner_baseline,
            inner_raw,
            min_days_per_block=int(safety["min_days_per_block"]),
            min_support_blocks=int(safety["min_support_blocks"]),
        )
        gate = fit_soft_gate(
            stack.loc[train],
            prediction_columns=CONTRIBUTORS,
            regularization=regularization,
        )
        baseline[validation] = predict_simplex_baseline(
            stack.loc[train], stack.loc[validation], CONTRIBUTORS
        )
        raw[validation] = predict_soft_gate(gate, stack.loc[validation])
        safe_prediction[validation] = apply_safe_calibrator(
            calibrator,
            stack.loc[validation],
            baseline[validation],
            raw[validation],
        )
        outer_summaries[outer] = calibrator_summary(calibrator)

    if not all(np.isfinite(values).all() for values in (baseline, raw, safe_prediction)):
        raise ValueError("safe residual OOF is incomplete")
    frozen = pd.read_parquet("artifacts/p2_deep_finalists_v1/stacked_oof.parquet")
    frozen["time"] = pd.to_datetime(frozen["time"], utc=True)
    frozen = stack[["time", "layer", "block"]].merge(
        frozen[["time", "layer", "block", "lobo_prediction"]],
        on=["time", "layer", "block"],
        validate="one_to_one",
    )
    baseline_error = float(np.max(np.abs(baseline - frozen["lobo_prediction"].to_numpy(float))))
    if baseline_error > 1e-6:
        raise ValueError(f"safe residual baseline differs from frozen deep LOBO: {baseline_error}")
    metrics = _extended_metrics(stack, baseline, raw, safe_prediction)
    bootstrap = _paired_day_bootstrap(stack, baseline, safe_prediction)

    progress.update(74, "최종 calibration", "3개 cross-fitted block에서 state support·alpha 동결")
    final_calibrator = fit_safe_calibrator(
        stack,
        baseline,
        raw,
        min_days_per_block=int(safety["min_days_per_block"]),
        min_support_blocks=int(safety["min_support_blocks"]),
    )
    incumbent = pd.read_csv("submissions/p2/P2_DEEP_STACK_V1.csv")
    raw_submission = pd.read_csv("submissions/p2/P2_PUBLIC_STATE_SOFT_GATE_V1.csv")
    if not incumbent[["station", "layer", "time"]].equals(
        raw_submission[["station", "layer", "time"]]
    ):
        raise ValueError("safe residual input submissions have different keys")
    test_public = build_public_state_features(data.observations, data.test_index[["time", "layer"]])
    test_prediction = apply_safe_calibrator(
        final_calibrator,
        test_public,
        incumbent["temp"].to_numpy(float),
        raw_submission["temp"].to_numpy(float),
    )
    submission_path = Path(contract["outputs"]["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    submission_validation = validate_submission(submission_path, data.test_index)

    progress.update(86, "재현 산출물 저장", "OOF·calibrator·submission·SHA256")
    oof = stack[["time", "layer", "truth", "block"]].copy()
    oof["baseline_prediction"] = baseline
    oof["raw_lambda10_prediction"] = raw
    oof["prediction"] = safe_prediction
    oof_path = output_dir / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    model_path = output_dir / "calibrator.joblib"
    joblib.dump(final_calibrator, model_path, compress=3)
    restored = joblib.load(model_path)
    roundtrip = apply_safe_calibrator(
        restored,
        test_public,
        incumbent["temp"].to_numpy(float),
        raw_submission["temp"].to_numpy(float),
    )
    roundtrip_error = float(np.max(np.abs(roundtrip - test_prediction)))
    decision = "KEEP_FROZEN_DEEP_SAFE_GATE_NO_GAIN"
    if metrics["candidate_rmse"] < metrics["baseline_rmse"]:
        decision = (
            "ADAPTIVE_CHALLENGER_REQUIRES_FRESH_VALIDATION"
            if bootstrap["ci90_high"] < 0
            else "RESEARCH_SIGNAL_KEEP_FROZEN_DEEP"
        )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_id": contract["experiment_id"],
        "adaptive_after_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "research_only": True,
        "uploaded": False,
        "external_values_used": False,
        "raw_gate_regularization": regularization,
        "baseline_reproduction_max_abs_error": baseline_error,
        "model_roundtrip_max_abs_error": roundtrip_error,
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "outer_calibrators": outer_summaries,
        "final_calibrator": calibrator_summary(final_calibrator),
        "test_active_row_share": _active_share(
            test_public, incumbent["temp"].to_numpy(float), test_prediction
        ),
        "submission_validation": submission_validation,
        "decision": decision,
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "calibrator": {"path": model_path.as_posix(), "sha256": _sha256(model_path)},
            "submission": {
                "path": submission_path.as_posix(),
                "sha256": _sha256(submission_path),
            },
        },
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "contract_sha256": _sha256(args.contract),
            "oof_sha256": _sha256(oof_path),
            "calibrator_sha256": _sha256(model_path),
            "submission_sha256": _sha256(submission_path),
            "uploaded": False,
        },
    )
    progress.update(
        100,
        "완료",
        f"{decision} · ΔRMSE {metrics['delta_rmse']:+.6f}℃ · 업로드 없음",
        status="complete",
    )
    print(json.dumps({"status": "passed", "decision": decision, "metrics": metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
