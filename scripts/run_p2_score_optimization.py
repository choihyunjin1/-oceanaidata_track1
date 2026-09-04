"""Optimize P2 candidates for the official row-level RMSE objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.research import (
    append_public_dynamics,
    append_public_m2_harmonics,
    paired_rmse_bootstrap,
    select_lean_m2_dynamics,
)
from p2_restore.score_optimization import (
    TARGET_RELEVANT_BLOCKS,
    align_score_oof,
    fit_score_router,
    leave_one_relevant_block_out,
    route_predictions,
    score_diagnostics,
    select_layer_router,
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
    path: Path, progress: float, phase: str, detail: str, status: str = "running"
) -> None:
    _write_json(
        path,
        {
            "title": "P2 공식 RMSE 점수 최적화",
            "progress": progress,
            "phase": phase,
            "detail": detail,
            "status": status,
            "eta": "자동 계산",
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _validate_contract(path: Path, data_dir: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "p2_score_optimization_v1":
        raise ValueError("unexpected P2 score experiment id")
    if contract.get("status") != "authorized_local_score_optimization":
        raise ValueError("P2 score experiment is not authorized")
    if contract.get("upload_allowed") is not False:
        raise ValueError("score experiment must remain local-only")
    for name, expected in contract["sources"].items():
        if _sha256(data_dir / name) != expected:
            raise ValueError(f"source hash mismatch for {name}")
    oof = contract["oof_sources"]
    for key in ("phase", "phase_result", "state", "state_result"):
        if _sha256(Path(oof[f"{key}_path"])) != oof[f"{key}_sha256"]:
            raise ValueError(f"frozen {key} artifact hash changed")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_score_optimization_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_score_optimization_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_score_optimization.json")
    )
    args = parser.parse_args()
    _progress(args.status_file, 2, "contract", "원본·OOF 해시와 공식 RMSE 선택 계약 검증")
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir)
    phase_oof = pd.read_parquet(contract["oof_sources"]["phase_path"])
    state_oof = pd.read_parquet(contract["oof_sources"]["state_path"])
    oof = align_score_oof(phase_oof, state_oof)

    _progress(args.status_file, 18, "selection", "8개 층별 router와 관련계절 LOBO 비교")
    selection = select_layer_router(oof)
    selected_arms = {int(layer): arm for layer, arm in selection["selected"]["layer_arms"].items()}
    diagnostics = score_diagnostics(oof, selected_arms)
    lobo = leave_one_relevant_block_out(oof)
    relevant = oof["block"].isin(TARGET_RELEVANT_BLOCKS).to_numpy()
    router_oof = oof.loc[relevant].reset_index(drop=True).copy()
    router_oof["router"] = route_predictions(router_oof.reset_index(drop=True), selected_arms)
    bootstrap_vs_phase = paired_rmse_bootstrap(
        router_oof.assign(day=pd.to_datetime(router_oof["time"]).dt.floor("D").astype(str)),
        reference="phase",
        candidate="router",
        replicates=2000,
        seed=20260816,
    )

    _progress(args.status_file, 35, "feature", "전체 공개층 V0·M2·phase 특징 생성")
    data = load_p2_data(data_dir)
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, data.observations)

    _progress(args.status_file, 55, "train", "400-round base·phase·상태전문가 전체 재학습")
    model = fit_score_router(base, lean, phase, selected_arms)
    test_base = build_test_features(data)
    test_dynamic = append_public_dynamics(test_base, data.observations)
    test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
    test_phase = append_public_m2_harmonics(test_lean, data.observations)
    predictions = model.predict_components(test_base, test_lean, test_phase)

    _progress(args.status_file, 82, "candidate", "phase·state·layer-router 제출 CSV 3개 검증")
    names = {
        "phase": Path("submissions/p2/P2_SCORE_PHASE400.csv"),
        "state": Path("submissions/p2/P2_SCORE_STATE400.csv"),
        "router": Path("submissions/p2/P2_SCORE_LAYER_ROUTER.csv"),
    }
    candidate_manifests: dict[str, object] = {}
    for arm, output in names.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        submission = build_submission(data.test_index, predictions[arm])
        submission.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
        validated = validate_submission(output, data.test_index)
        candidate_manifests[arm] = {
            "path": output.as_posix(),
            "sha256": _sha256(output),
            **validated,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.joblib"
    joblib.dump(model, model_path)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "official_objective": contract["official_objective"],
        "selection_policy": contract["selection_policy"],
        "preregistration_sha256": _sha256(args.preregistration),
        "oof_rows": len(oof),
        "router_selection": selection,
        "target_relevant_lobo": lobo,
        "diagnostics": diagnostics,
        "target_relevant_bootstrap_router_vs_phase": bootstrap_vs_phase,
        "selected_layer_arms": {str(key): value for key, value in selected_arms.items()},
        "candidates": candidate_manifests,
        "model_sha256": _sha256(model_path),
        "decision": "FREEZE_SCORE_CANDIDATES_NO_UPLOAD",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "model_sha256": _sha256(model_path),
            "candidates": candidate_manifests,
            "uploaded": False,
        },
    )
    _progress(
        args.status_file,
        100,
        "complete",
        "공식 RMSE 후보 3개 동결 · 업로드 없음",
        status="complete",
    )
    print(
        json.dumps(
            {
                "selected_layer_arms": result["selected_layer_arms"],
                "target_relevant": diagnostics["target_relevant"],
                "lobo": lobo,
                "bootstrap": bootstrap_vs_phase,
                "candidates": candidate_manifests,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
