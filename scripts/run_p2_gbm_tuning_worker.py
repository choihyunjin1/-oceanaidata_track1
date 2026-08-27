"""Tune one preregistered P2 GBM family in an isolated worker process."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.gbm_tournament import (
    align_with_deep_stack,
    evaluate_deep_pair,
    paired_day_bootstrap,
)
from p2_restore.gbm_tuning import TUNING_FAMILIES, tune_family
from p2_restore.research import (
    append_public_dynamics,
    append_public_m2_harmonics,
    select_lean_m2_dynamics,
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


def _phase_features(data):
    base = build_training_features(data.observations)
    dynamics = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamics)
    return append_public_m2_harmonics(lean, data.observations)


def _phase_test_features(data):
    base = build_test_features(data)
    dynamics = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamics)
    return append_public_m2_harmonics(lean, data.observations)


def _validate_contract(path: Path, data_dir: Path, family: str, trials: int) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "p2_top3_parallel_tuning_v1":
        raise ValueError("unexpected P2 top-three tuning experiment id")
    if contract.get("status") != "authorized_local_parallel_tuning":
        raise ValueError("P2 top-three tuning is not locally authorized")
    if contract.get("research_only") is not True or contract.get("upload_allowed") is not False:
        raise ValueError("P2 top-three tuning must remain local-only")
    if tuple(contract.get("families", ())) != TUNING_FAMILIES or family not in TUNING_FAMILIES:
        raise ValueError("P2 top-three family contract changed")
    if int(contract["search"]["trials_per_family"]) != trials:
        raise ValueError("worker trial budget differs from preregistration")
    if int(contract["search"]["trials_per_outer_fold"]) * 3 != trials:
        raise ValueError("outer-fold trial partition differs from preregistration")
    if int(contract["parallelism"]["threads_per_worker"]) != 2:
        raise ValueError("worker thread budget changed")
    sources = {
        "observations.csv": data_dir / "observations.csv",
        "test_index.csv": data_dir / "test_index.csv",
        "baseline_interp.csv": data_dir / "baseline_interp.csv",
        "deep_stack_oof": Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
        "deep_stack_submission": Path("submissions/p2/P2_DEEP_STACK_V1.csv"),
    }
    for name, source in sources.items():
        if not source.is_file() or _sha256(source) != contract["sources"][name]:
            raise ValueError(f"P2 tuning source hash mismatch: {name}")
    ranking = Path(contract["ranking_source"]["artifact"])
    if not ranking.is_file() or _sha256(ranking) != contract["ranking_source"]["sha256"]:
        raise ValueError("P2 GBM ranking source hash mismatch")
    return contract


class WorkerProgress:
    def __init__(self, path: Path, family: str) -> None:
        self.path = path
        self.family = family
        self.started = time.perf_counter()

    def update(self, progress: float, phase: str, detail: str, *, status: str = "running") -> None:
        progress = min(max(float(progress), 0.0), 100.0)
        elapsed = time.perf_counter() - self.started
        remaining = elapsed * (100.0 - progress) / progress if progress > 0 else 0.0
        eta = datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
        _write_json(
            self.path,
            {
                "family": self.family,
                "status": status,
                "progress": progress,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if progress else "측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _serializable_pair(pair: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in pair.items() if not isinstance(value, np.ndarray)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=TUNING_FAMILIES)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_top3_parallel_tuning_v1.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/p2_top3_parallel_tuning_v1")
    )
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=36)
    args = parser.parse_args()
    started = time.perf_counter()
    progress = WorkerProgress(args.status_file, args.family)
    progress.update(1, "contract", "사전등록·원본·기준 OOF SHA 검증")
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir, args.family, args.trials)
    preregistration_sha256 = _sha256(args.preregistration)
    family_root = args.output_root / args.family
    family_root.mkdir(parents=True, exist_ok=True)

    progress.update(5, "features", "동일 public-only phase 81개 특징 생성")
    data = load_p2_data(data_dir)
    phase = _phase_features(data)
    if len(phase.feature_columns) != 81:
        raise ValueError(f"expected 81 P2 phase features, got {len(phase.feature_columns)}")

    def tuning_progress(value: dict[str, object]) -> None:
        if value["phase"] == "search":
            trial = int(value["trial"])
            fold = int(value["fold_number"])
            fraction = ((trial - 1) * 3 + fold) / (args.trials * 3)
            progress.update(
                8 + 68 * fraction,
                "nested tuning",
                f"trial {trial}/{args.trials} · inner fold {fold}/3 · {value['fold']}",
            )
        else:
            fold = int(value["fold_number"])
            progress.update(
                76 + 8 * fold / 3,
                "frozen outer",
                f"최적 파라미터 고정 · outer fold {fold}/3 · {value['fold']}",
            )

    progress.update(8, "nested tuning", f"{args.family} · Optuna {args.trials} trials")
    summary, oof, model = tune_family(
        phase,
        args.family,
        family_root / "optuna.sqlite3",
        trials=args.trials,
        threads=int(contract["parallelism"]["threads_per_worker"]),
        progress=tuning_progress,
    )
    oof_path = family_root / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")

    progress.update(86, "pair evaluation", "동결 deep OOF와 층별 convex pair·LOBO 평가")
    deep = pd.read_parquet("artifacts/p2_deep_finalists_v1/stacked_oof.parquet")
    aligned = align_with_deep_stack(deep, oof)
    pair = evaluate_deep_pair(aligned)
    aligned["fitted_pair_prediction"] = pair["fitted_prediction"]
    aligned["lobo_pair_prediction"] = pair["lobo_prediction"]
    fitted_bootstrap = paired_day_bootstrap(
        aligned,
        np.asarray(pair["fitted_prediction"]),
        reference_column="deep_prediction",
        replicates=int(contract["comparison"]["bootstrap"]["replicates"]),
    )
    lobo_bootstrap = paired_day_bootstrap(
        aligned,
        np.asarray(pair["lobo_prediction"]),
        reference_column="deep_lobo_prediction",
        replicates=int(contract["comparison"]["bootstrap"]["replicates"]),
    )
    paired_path = family_root / "paired_oof.parquet"
    aligned.to_parquet(paired_path, index=False, compression="zstd")

    progress.update(90, "full fit", "수렴 라운드 동결 모델 저장·round-trip 재현")
    model_path = family_root / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    test_phase = _phase_test_features(data)
    prediction = model.predict(test_phase)
    restored = joblib.load(model_path)
    roundtrip = restored.predict(test_phase)
    roundtrip_max_abs = float(np.max(np.abs(prediction - roundtrip)))
    if roundtrip_max_abs > 1e-12:
        raise RuntimeError(f"P2 tuned model round-trip changed predictions: {roundtrip_max_abs}")
    submission = build_submission(data.test_index, prediction)
    submission_path = Path("submissions/p2") / f"P2_TUNED_{args.family.upper()}_V1.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False, encoding="utf-8", lineterminator="\n")
    submission_validation = validate_submission(submission_path, data.test_index)

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "family": args.family,
        "research_only": True,
        "uploaded": False,
        "preregistration_sha256": preregistration_sha256,
        "tuning": summary,
        "deep_pair": _serializable_pair(pair),
        "fitted_pair_bootstrap": fitted_bootstrap,
        "lobo_pair_bootstrap": lobo_bootstrap,
        "submission_validation": submission_validation,
        "model_roundtrip_max_abs_error": roundtrip_max_abs,
        "elapsed_seconds": time.perf_counter() - started,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "artifacts": {
            "oof": {"path": str(oof_path), "sha256": _sha256(oof_path)},
            "paired_oof": {"path": str(paired_path), "sha256": _sha256(paired_path)},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "submission": {"path": str(submission_path), "sha256": _sha256(submission_path)},
            "optuna_storage": {
                "path": str(family_root / "optuna.sqlite3"),
                "sha256": _sha256(family_root / "optuna.sqlite3"),
            },
        },
    }
    result_path = family_root / "result.json"
    _write_json(result_path, result)
    _write_json(
        family_root / "manifest.json",
        {
            "family": args.family,
            "result_sha256": _sha256(result_path),
            "artifacts": result["artifacts"],
            "uploaded": False,
        },
    )
    progress.update(
        100,
        "complete",
        f"outer RMSE {summary['outer_rmse']:.6f} · LOBO pair {pair['lobo_blend_rmse']:.6f}",
        status="complete",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
