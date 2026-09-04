"""Run the local-only P2 public-endpoint profile projection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.profile_projection import project_profiles, public_endpoint_frame
from p2_restore.submission import build_submission, validate_submission

SEED = 20260816


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _status(path: Path, progress: float, phase: str, detail: str, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 0.001)
    remaining = elapsed * max(100.0 - progress, 0.0) / max(progress, 1.0)
    eta = datetime.now().astimezone() + timedelta(seconds=remaining)
    _write_json(
        path,
        {
            "title": "P2 물리 연직 투영 실험",
            "status": "complete" if progress >= 100 else "running",
            "progress": progress,
            "phase": phase,
            "detail": detail,
            "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def _metrics(frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)

    def cut(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(mask.sum()),
            "baseline_rmse": _rmse(truth[mask], baseline[mask]),
            "candidate_rmse": _rmse(truth[mask], candidate[mask]),
            "delta_rmse": _rmse(truth[mask], candidate[mask]) - _rmse(truth[mask], baseline[mask]),
        }

    all_rows = np.ones(len(frame), dtype=bool)
    return {
        **cut(all_rows),
        "by_block": {
            str(block): cut(frame["block"].eq(block).to_numpy())
            for block in frame["block"].drop_duplicates()
        },
        "by_layer": {str(layer): cut(frame["layer"].eq(layer).to_numpy()) for layer in (2, 3, 4)},
    }


def _paired_day_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int = 2000,
) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(float)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(day)
    blocks = [np.flatnonzero(day == value) for value in unique]
    rng = np.random.default_rng(SEED)
    delta = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        chosen = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([blocks[index] for index in chosen])
        delta[number] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], baseline[rows])
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, baseline),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0)),
    }


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_physical_profile_projection_v1":
        raise ValueError("unexpected profile projection experiment id")
    if value.get("status") != "authorized_local_adaptive_research":
        raise ValueError("profile projection experiment is not locally authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("profile projection must remain local-only")
    if value.get("adaptive_after_outer_exposure") is not True:
        raise ValueError("adaptive provenance must not be removed")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_physical_profile_projection_v1.json"),
    )
    parser.add_argument(
        "--incumbent-oof",
        type=Path,
        default=Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
    )
    parser.add_argument(
        "--incumbent-submission",
        type=Path,
        default=Path("submissions/p2/P2_DEEP_STACK_V1.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_physical_profile_projection_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_profile_projection.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    contract = _validate_contract(args.preregistration)
    _status(args.status_file, 5, "입력 검증", "고정 Deep OOF·공개층 endpoint 검사", started)
    data = load_p2_data(resolve_data_dir(args.data_dir))
    endpoints = public_endpoint_frame(data.observations)

    oof = pd.read_parquet(args.incumbent_oof)
    required = {"time", "layer", "truth", "block", "lobo_prediction"}
    if missing := required.difference(oof.columns):
        raise ValueError(f"incumbent OOF is missing columns: {sorted(missing)}")
    if len(oof) != 69_850 or oof.duplicated(["time", "layer"]).any():
        raise ValueError("incumbent OOF grain changed")
    baseline = oof["lobo_prediction"].to_numpy(float)
    _status(
        args.status_file, 35, "OOF 투영", "target label 비사용 연직 envelope·순서 투영", started
    )
    projected_oof = project_profiles(oof, baseline, endpoints)
    metrics = _metrics(oof, baseline, projected_oof.prediction)
    bootstrap = _paired_day_bootstrap(oof, baseline, projected_oof.prediction)

    _status(
        args.status_file, 65, "test 재현", "동결 Deep 제출에 동일 label-blind 변환 적용", started
    )
    incumbent = pd.read_csv(args.incumbent_submission)
    key_columns = ["station", "layer", "time"]
    keys_match = all(
        np.array_equal(
            incumbent[column].astype(str).to_numpy(),
            data.test_index[column].astype(str).to_numpy(),
        )
        for column in key_columns
    )
    if len(incumbent) != 26_061 or not keys_match:
        raise ValueError("incumbent submission keys differ from test_index")
    projected_test = project_profiles(incumbent, incumbent["temp"].to_numpy(float), endpoints)
    output_path = Path(contract["submission"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = build_submission(data.test_index, projected_test.prediction)
    submission.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(output_path, data.test_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof.parquet"
    saved_oof = oof.loc[:, ["time", "layer", "truth", "block", "lobo_prediction"]].copy()
    saved_oof["prediction"] = projected_oof.prediction
    saved_oof["eligible"] = projected_oof.eligible_mask
    saved_oof["active"] = projected_oof.active_mask
    saved_oof.to_parquet(oof_path, index=False, compression="zstd")

    ci_excludes_zero = bootstrap["ci90_high"] < 0
    same_season_gain = metrics["by_block"]["2024_sep_oct"]["delta_rmse"] < 0
    decision = (
        "RESEARCH_CHALLENGER_KEEP_DEEP_PRIMARY_NO_UPLOAD"
        if metrics["delta_rmse"] < 0 and same_season_gain
        else "REJECT_KEEP_DEEP_PRIMARY_NO_UPLOAD"
    )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "adaptive_after_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "ci90_excludes_zero": ci_excludes_zero,
        "oof_projection": {
            **projected_oof.diagnostics(),
            "max_abs_correction": float(np.max(np.abs(projected_oof.prediction - baseline))),
        },
        "test_projection": {
            **projected_test.diagnostics(),
            "max_abs_correction": float(
                np.max(np.abs(projected_test.prediction - incumbent["temp"].to_numpy(float)))
            ),
            "rmse_correction": float(
                np.sqrt(
                    np.mean((projected_test.prediction - incumbent["temp"].to_numpy(float)) ** 2)
                )
            ),
        },
        "submission_validation": validation,
        "decision": decision,
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "submission": {"path": output_path.as_posix(), "sha256": _sha256(output_path)},
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "preregistration_sha256": _sha256(args.preregistration),
            "incumbent_oof_sha256": _sha256(args.incumbent_oof),
            "incumbent_submission_sha256": _sha256(args.incumbent_submission),
            "result_sha256": _sha256(result_path),
            "oof_sha256": _sha256(oof_path),
            "submission_sha256": _sha256(output_path),
            "uploaded": False,
        },
    )
    _status(
        args.status_file,
        100,
        "완료",
        f"{decision} · ΔRMSE {metrics['delta_rmse']:+.6f}℃ · 업로드 없음",
        started,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
