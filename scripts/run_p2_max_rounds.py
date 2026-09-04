"""Run the authorized 5,000-round P2 convergence screen and freeze candidates."""

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

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.max_rounds import (
    MAX_ROUNDS,
    ROUND_CHECKPOINTS,
    SCORE_LAYER_ARMS,
    fit_max_round_router,
    run_target_round_screen,
)
from p2_restore.research import (
    append_public_dynamics,
    append_public_m2_harmonics,
    select_lean_m2_dynamics,
)
from p2_restore.score_optimization import (
    TARGET_RELEVANT_BLOCKS,
    align_score_oof,
    route_predictions,
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


def _progress(
    path: Path,
    progress: float,
    phase: str,
    detail: str,
    *,
    eta: datetime,
    status: str = "running",
) -> None:
    _write_json(
        path,
        {
            "title": "P2 5,000라운드 수렴 검증",
            "progress": progress,
            "phase": phase,
            "detail": detail,
            "status": status,
            "eta": eta.astimezone().strftime("%Y-%m-%d %H:%M:%S KST"),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _validate_contract(path: Path, data_dir: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "p2_max_round_convergence_v1":
        raise ValueError("unexpected maximum-round experiment id")
    if contract.get("status") != "authorized_local_max_round_run":
        raise ValueError("maximum-round experiment is not authorized")
    if contract.get("upload_allowed") is not False or contract.get("research_only") is not True:
        raise ValueError("maximum-round experiment must remain local-only")
    if int(contract.get("max_rounds", 0)) != MAX_ROUNDS:
        raise ValueError("maximum-round contract changed")
    if tuple(contract.get("checkpoints", ())) != ROUND_CHECKPOINTS:
        raise ValueError("checkpoint grid changed")
    if contract["selection"]["blocks"] != list(TARGET_RELEVANT_BLOCKS):
        raise ValueError("target-relevant block selection changed")
    if {int(key): value for key, value in contract["selection"]["layer_router"].items()} != (
        SCORE_LAYER_ARMS
    ):
        raise ValueError("frozen score layer router changed")
    local_sources = {
        "observations.csv": data_dir / "observations.csv",
        "test_index.csv": data_dir / "test_index.csv",
        "baseline_interp.csv": data_dir / "baseline_interp.csv",
        "phase_oof": Path("artifacts/p2_m2_local_phase_v1/oof.parquet"),
        "state_oof": Path("artifacts/p2_state_conditional_lean_v1/oof.parquet"),
        "score_result": Path("artifacts/p2_score_optimization_v1/result.json"),
    }
    for name, source in local_sources.items():
        if _sha256(source) != contract["sources"][name]:
            raise ValueError(f"source hash mismatch: {name}")
    return contract


def _reference_400() -> pd.DataFrame:
    phase = pd.read_parquet("artifacts/p2_m2_local_phase_v1/oof.parquet")
    state = pd.read_parquet("artifacts/p2_state_conditional_lean_v1/oof.parquet")
    aligned = align_score_oof(phase, state)
    relevant = aligned["block"].isin(TARGET_RELEVANT_BLOCKS)
    result = aligned.loc[relevant].sort_values(["time", "layer"]).reset_index(drop=True)
    result["router"] = route_predictions(result, SCORE_LAYER_ARMS)
    return result


def _validate_400_reproduction(oof: pd.DataFrame) -> float:
    reference = _reference_400()
    candidate = oof.sort_values(["time", "layer"]).reset_index(drop=True)
    if not reference[["time", "layer", "truth", "block"]].equals(
        candidate[["time", "layer", "truth", "block"]]
    ):
        raise RuntimeError("maximum-round OOF keys/truth do not reproduce the frozen 400 run")
    error = float(
        np.max(
            np.abs(reference["router"].to_numpy(float) - candidate["router_400"].to_numpy(float))
        )
    )
    if error > 1e-12:
        raise RuntimeError(f"400-round frozen prediction reproduction failed: {error}")
    return error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_max_round_convergence_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_max_round_convergence_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_max_rounds.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    eta = datetime.now().astimezone() + timedelta(minutes=35)
    _progress(args.status_file, 2, "contract", "원본·OOF·400라운드 기준 해시 검증", eta=eta)
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir)
    data = load_p2_data(data_dir)

    _progress(args.status_file, 10, "feature", "base·lean-M2·local-phase 특징 생성", eta=eta)
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, data.observations)

    def block_progress(position: int, total: int, name: str) -> None:
        fraction = (position - 1) / total
        _progress(
            args.status_file,
            20 + 48 * fraction,
            "validation",
            f"목표계절 블록 {position}/{total} · {name} · 각 모델 5,000라운드",
            eta=eta,
        )

    screen, oof = run_target_round_screen(
        base,
        lean,
        phase,
        checkpoints=ROUND_CHECKPOINTS,
        max_rounds=MAX_ROUNDS,
        progress=block_progress,
    )
    reproduction_error = _validate_400_reproduction(oof)
    selected_round = int(screen["selected_round"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")

    elapsed = time.perf_counter() - started
    eta = datetime.now().astimezone() + timedelta(seconds=max(elapsed / 3.0, 120))
    _progress(
        args.status_file,
        72,
        "full_train",
        f"전체 공개 학습자료로 4개 모델 × 5,000라운드 학습 · 선택 {selected_round}",
        eta=eta,
    )
    model = fit_max_round_router(base, lean, phase, rounds=MAX_ROUNDS)
    test_base = build_test_features(data)
    test_dynamic = append_public_dynamics(test_base, data.observations)
    test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
    test_phase = append_public_m2_harmonics(test_lean, data.observations)
    selected_prediction = model.predict_components_at(
        test_base, test_lean, test_phase, selected_round
    )["router"]
    maximum_prediction = model.predict_components_at(test_base, test_lean, test_phase, MAX_ROUNDS)[
        "router"
    ]

    _progress(
        args.status_file,
        90,
        "candidate",
        "최적 체크포인트와 5,000라운드 제출 형식·재현성 검증",
        eta=datetime.now().astimezone() + timedelta(minutes=3),
    )
    outputs = {
        "selected": (
            Path(f"submissions/p2/P2_SCORE_ROUTER_ROUND{selected_round}.csv"),
            selected_prediction,
            selected_round,
        ),
        "maximum": (
            Path("submissions/p2/P2_SCORE_ROUTER_5000.csv"),
            maximum_prediction,
            MAX_ROUNDS,
        ),
    }
    candidate_manifests: dict[str, object] = {}
    for name, (path, prediction, round_number) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_submission(data.test_index, prediction).to_csv(
            path, index=False, encoding="utf-8", lineterminator="\n"
        )
        candidate_manifests[name] = {
            "path": path.as_posix(),
            "round": round_number,
            "sha256": _sha256(path),
            **validate_submission(path, data.test_index),
        }

    model_path = args.output_dir / "model_5000.joblib"
    joblib.dump(model, model_path, compress=3)
    restored = joblib.load(model_path)
    restored_selected = restored.predict_components_at(
        test_base, test_lean, test_phase, selected_round
    )["router"]
    restored_maximum = restored.predict_components_at(test_base, test_lean, test_phase, MAX_ROUNDS)[
        "router"
    ]
    roundtrip_errors = {
        "selected": float(np.max(np.abs(restored_selected - selected_prediction))),
        "maximum": float(np.max(np.abs(restored_maximum - maximum_prediction))),
    }
    if max(roundtrip_errors.values()) > 1e-12:
        raise RuntimeError(f"saved maximum-round model reproduction failed: {roundtrip_errors}")

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "single_change": contract["single_change"],
        "preregistration_sha256": _sha256(args.preregistration),
        "elapsed_seconds": time.perf_counter() - started,
        "screen": screen,
        "frozen_400_prediction_max_abs_error": reproduction_error,
        "selected_round": selected_round,
        "candidates": candidate_manifests,
        "model_path": model_path.as_posix(),
        "model_sha256": _sha256(model_path),
        "model_roundtrip_max_abs_errors": roundtrip_errors,
        "decision": "FREEZE_SELECTED_AND_MAXIMUM_CANDIDATES_NO_UPLOAD",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "oof_sha256": _sha256(oof_path),
            "model_sha256": _sha256(model_path),
            "candidates": candidate_manifests,
            "uploaded": False,
        },
    )
    _progress(
        args.status_file,
        100,
        "complete",
        f"완료 · 최적 {selected_round}라운드 · 5,000라운드 비교 후보 동결 · 업로드 없음",
        eta=datetime.now().astimezone(),
        status="complete",
    )
    print(
        json.dumps(
            {
                "selected_round": selected_round,
                "round_400_rmse": screen["round_400_router_rmse"],
                "selected_rmse": screen["selected_router_rmse"],
                "round_5000_rmse": screen["round_5000_router_rmse"],
                "candidates": candidate_manifests,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
