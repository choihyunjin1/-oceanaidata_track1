"""Run the preregistered bounded P2 LightGBM structure/parameter search."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import joblib

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.research import (
    append_public_dynamics,
    comparison_diagnostics,
    paired_rmse_bootstrap,
    select_lean_m2_dynamics,
)
from p2_restore.submission import build_submission, validate_submission
from p2_restore.tuning import (
    evaluate_guard_blocks,
    fit_final_tuned_blend,
    optimize_parameters,
    screen_structures,
)


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
    status: str = "running",
) -> None:
    _write_json(
        path,
        {
            "title": "P2 LightGBM 구조·파라미터 탐색",
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
    if contract.get("experiment_id") != "p2_lgbm_nested_tuning_v1":
        raise ValueError("unexpected P2 tuning experiment id")
    if (
        contract.get("status") != "authorized_one_shot"
        or contract.get("upload_allowed") is not False
    ):
        raise ValueError("P2 tuning experiment is not authorized as a local-only one-shot")
    if contract["search"]["trials"] != 40 or contract["fixed_model_surface"]["blend_weight"] != 0.5:
        raise ValueError("P2 tuning budget or blend contract changed")
    if contract["fixed_model_surface"]["blend_weight_search"] is not False:
        raise ValueError("blend-weight search is forbidden")
    for name, expected in contract["sources"].items():
        if _sha256(data_dir / name) != expected:
            raise ValueError(f"source hash mismatch for {name}")
    reference = Path("artifacts/p2_method_scout/result.json")
    if _sha256(reference) != contract["frozen_reference"]["result_sha256"]:
        raise ValueError("frozen P2 method result changed")
    return contract


def _gates(
    contract: dict[str, object],
    reports: dict[str, object],
    diagnostics: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    limits = contract["promotion_gates"]
    block_delta = {
        name: value["tuned_blend50"]["rmse"] - value["current_blend50"]["rmse"]
        for name, value in reports.items()
    }
    layer_delta = {layer: value["delta_rmse"] for layer, value in diagnostics["by_layer"].items()}
    checks = {
        "aggregate_delta": diagnostics["aggregate"]["delta_rmse"]
        <= limits["aggregate_delta_rmse_lte"],
        "bootstrap_ci90": bootstrap["ci90_high"] < limits["bootstrap_ci90_high_lt"],
        "improved_guard_blocks": sum(value < 0 for value in block_delta.values())
        >= limits["minimum_improved_guard_blocks"],
        "worst_guard_block": max(block_delta.values()) <= limits["maximum_guard_block_regression"],
        "worst_layer": max(layer_delta.values()) <= limits["maximum_layer_regression"],
        "same_season": block_delta["2024_sep_oct"]
        <= limits["same_season_2024_sep_oct_max_regression"],
        "pre_target": block_delta["2025_jul_aug"]
        <= limits["pre_target_2025_jul_aug_max_regression"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "block_deltas": block_delta,
        "layer_deltas": layer_delta,
        "limits": limits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_lgbm_nested_tuning_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_lgbm_nested_tuning_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_lgbm_tuning.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    _progress(args.status_file, 2, "contract", "사전등록·원본·기준 후보 SHA 검증")
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir)
    data = load_p2_data(data_dir)
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)

    _progress(args.status_file, 14, "structure", "개발 score-month에서 shared 대 layerwise 비교")
    structure = screen_structures(base, lean, contract["development_blocks"])
    winner = structure["winner"]

    _progress(args.status_file, 24, "tuning", f"{winner} 구조 Optuna 40 trials 시작")

    def trial_progress(completed: int, total: int, best: float) -> None:
        percent = 24 + 51 * completed / total
        _progress(
            args.status_file,
            percent,
            "tuning",
            f"{winner} · trial {completed}/{total} · dev best RMSE {best:.6f}°C",
        )

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tuning = optimize_parameters(
        base,
        lean,
        contract["development_blocks"],
        structure=winner,
        trials=contract["search"]["trials"],
        progress=trial_progress,
    )

    _progress(
        args.status_file, 78, "guard", "최적 파라미터·median boosting round 동결 후 4개 guard 평가"
    )
    reports, oof = evaluate_guard_blocks(
        base,
        lean,
        contract["guard_blocks"],
        structure=winner,
        parameters=tuning["best_parameters"],
        iterations=tuning["frozen_iterations"],
    )
    previous = json.loads(Path("artifacts/p2_method_scout/result.json").read_text(encoding="utf-8"))
    reproduction_error = max(
        abs(
            value["current_blend50"]["rmse"] - previous["stability_blocks"][name]["blend50"]["rmse"]
        )
        for name, value in reports.items()
    )
    if reproduction_error > 1e-12:
        raise RuntimeError(f"frozen Blend50 guard reproduction failed: {reproduction_error}")
    diagnostics = comparison_diagnostics(
        oof, reference="current_blend50", candidate="tuned_blend50"
    )
    bootstrap = paired_rmse_bootstrap(
        oof,
        reference="current_blend50",
        candidate="tuned_blend50",
        replicates=contract["bootstrap"]["replicates"],
        seed=contract["bootstrap"]["seed"],
    )
    gates = _gates(contract, reports, diagnostics, bootstrap)
    decision = "PROMOTE" if gates["passed"] else "REJECT_AND_CLOSE_GENERATION"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "guard_is_virgin": False,
        "guard_status": contract["guard_status"],
        "preregistration_sha256": _sha256(args.preregistration),
        "source_hashes": contract["sources"],
        "frozen_reference_reproduction_max_abs_rmse_error": reproduction_error,
        "structure_screen": structure,
        "tuning": tuning,
        "guard_blocks": reports,
        "guard_diagnostics": diagnostics,
        "paired_day_bootstrap": bootstrap,
        "promotion_gates": gates,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    oof.to_parquet(args.output_dir / "guard_oof.parquet", index=False, compression="zstd")

    candidate_manifest: dict[str, object] | None = None
    if gates["passed"]:
        _progress(args.status_file, 90, "candidate", "전체 재학습·26,061행 후보 제출 검증")
        model = fit_final_tuned_blend(
            base,
            lean,
            structure=winner,
            parameters=tuning["best_parameters"],
            iterations=tuning["frozen_iterations"],
        )
        test_base = build_test_features(data)
        test_dynamic = append_public_dynamics(test_base, data.observations)
        test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
        prediction = model.predict(test_base, test_lean)
        submission = build_submission(data.test_index, prediction)
        model_path = args.output_dir / "model.joblib"
        submission_path = Path("submissions/p2/P2_TUNED_BLEND50.csv")
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        submission.to_csv(submission_path, index=False, encoding="utf-8", lineterminator="\n")
        validation = validate_submission(submission_path, data.test_index)
        candidate_manifest = {
            "submission": str(submission_path),
            "rows": validation["rows"],
            "minimum": validation["minimum"],
            "maximum": validation["maximum"],
            "model_sha256": _sha256(model_path),
            "submission_sha256": _sha256(submission_path),
        }
        result["candidate"] = candidate_manifest
        _write_json(result_path, result)

    manifest = {
        "decision": decision,
        "result_sha256": _sha256(result_path),
        "guard_oof_sha256": _sha256(args.output_dir / "guard_oof.parquet"),
        "candidate": candidate_manifest,
        "uploaded": False,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _progress(
        args.status_file,
        100,
        "complete",
        f"{decision} · 플랫폼 업로드 없음",
        status="complete",
    )
    print(
        json.dumps({"structure": structure, "tuning": tuning, "gates": gates, **manifest}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
