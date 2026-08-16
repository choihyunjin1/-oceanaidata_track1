"""Run the preregistered P2 mixed/stratified lean-M2 expert experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from p2_restore.state_conditional import (
    fit_full_state_blend,
    run_state_conditional_stability_screen,
    state_bin_diagnostics,
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
            "title": "P2 혼합·성층 상태조건 전문가",
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
    if contract.get("experiment_id") != "p2_state_conditional_lean_v1":
        raise ValueError("unexpected state experiment id")
    if (
        contract.get("status") != "authorized_one_shot"
        or contract.get("upload_allowed") is not False
    ):
        raise ValueError("state experiment is not authorized as a local-only one-shot")
    if contract["feature_contract"].get("parameter_search") is not False:
        raise ValueError("parameter search is forbidden for the state experiment")
    expert = contract["expert_contract"]
    if expert["quantile_low"] != 0.4 or expert["quantile_high"] != 0.6:
        raise ValueError("state quantiles differ from the preregistered values")
    for name, expected in contract["sources"].items():
        if _sha256(data_dir / name) != expected:
            raise ValueError(f"source hash mismatch for {name}")
    frozen = contract["frozen_reference"]
    if _sha256(Path(frozen["result_path"])) != frozen["result_sha256"]:
        raise ValueError("frozen P2 result hash changed")
    if _sha256(Path(frozen["submission_path"])) != frozen["submission_sha256"]:
        raise ValueError("frozen P2 submission hash changed")
    return contract


def _evaluate_gates(
    contract: dict[str, object],
    reports: dict[str, object],
    diagnostics: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    limits = contract["promotion_gates"]
    block_deltas = {
        name: values["state_blend50"]["rmse"] - values["current_blend50"]["rmse"]
        for name, values in reports.items()
    }
    layer_deltas = {
        layer: values["delta_rmse"] for layer, values in diagnostics["by_layer"].items()
    }
    checks = {
        "aggregate_delta": diagnostics["aggregate"]["delta_rmse"]
        <= limits["aggregate_delta_rmse_lte"],
        "bootstrap_ci90": bootstrap["ci90_high"] < limits["bootstrap_ci90_high_lt"],
        "improved_blocks": sum(delta < 0 for delta in block_deltas.values())
        >= limits["minimum_improved_blocks"],
        "worst_block": max(block_deltas.values()) <= limits["maximum_block_regression"],
        "worst_layer": max(layer_deltas.values()) <= limits["maximum_layer_regression"],
        "same_season": block_deltas["2024_sep_oct"]
        <= limits["same_season_2024_sep_oct_max_regression"],
        "pre_gap": block_deltas["2025_jul_aug"] <= limits["pre_gap_2025_jul_aug_max_regression"],
        "post_gap": block_deltas["2025_nov_dec"] <= limits["post_gap_2025_nov_dec_max_regression"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "block_deltas": block_deltas,
        "layer_deltas": layer_deltas,
        "limits": limits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_state_conditional_lean_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_state_conditional_lean_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_state_conditional.json")
    )
    args = parser.parse_args()
    _progress(args.status_file, 2, "contract", "사전등록·원본·동결 후보 SHA 검증")
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_contract(args.preregistration, data_dir)
    data = load_p2_data(data_dir)

    _progress(args.status_file, 15, "feature", "공개층 기반 V0·lean M2 특징 생성")
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)

    _progress(args.status_file, 30, "validation", "고정 8개 계절 블록 one-shot 학습")
    reports, oof = run_state_conditional_stability_screen(base, lean)
    diagnostics = comparison_diagnostics(
        oof, reference="current_blend50", candidate="state_blend50"
    )
    bootstrap = paired_rmse_bootstrap(
        oof,
        reference="current_blend50",
        candidate="state_blend50",
        replicates=2000,
        seed=20260816,
    )
    gates = _evaluate_gates(contract, reports, diagnostics, bootstrap)

    previous = json.loads(Path(contract["frozen_reference"]["result_path"]).read_text("utf-8"))
    reproduction_error = max(
        abs(
            values["current_blend50"]["rmse"]
            - previous["stability_blocks"][name]["blend50"]["rmse"]
        )
        for name, values in reports.items()
    )
    if reproduction_error > 1e-12:
        raise RuntimeError(f"frozen Blend50 reproduction failed: {reproduction_error}")

    _progress(args.status_file, 80, "decision", "bootstrap·블록·층 승격 게이트 계산")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "preregistration_sha256": _sha256(args.preregistration),
        "frozen_reference_reproduction_max_abs_rmse_error": reproduction_error,
        "source_hashes": contract["sources"],
        "feature_contract": contract["feature_contract"],
        "expert_contract": contract["expert_contract"],
        "blocks": reports,
        "diagnostics": diagnostics,
        "state_bins": state_bin_diagnostics(oof),
        "paired_day_bootstrap": bootstrap,
        "promotion_gates": gates,
        "decision": "PROMOTE" if gates["passed"] else "REJECT_AND_CLOSE_FAMILY",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    oof_path = args.output_dir / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")

    candidate_manifest: dict[str, object] | None = None
    if gates["passed"]:
        _progress(args.status_file, 88, "candidate", "전체 재학습·26,061행 제출 후보 검증")
        model = fit_full_state_blend(base, lean)
        test_base = build_test_features(data)
        test_dynamic = append_public_dynamics(test_base, data.observations)
        test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
        prediction = model.predict(test_base, test_lean)
        submission = build_submission(data.test_index, prediction)
        model_path = args.output_dir / "model.joblib"
        submission_path = Path("submissions/p2/P2_STATE_BLEND50.csv")
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        submission.to_csv(submission_path, index=False, encoding="utf-8", lineterminator="\n")
        validated = validate_submission(submission_path, data.test_index)
        candidate_manifest = {
            "submission": submission_path.as_posix(),
            "rows": validated["rows"],
            "minimum": validated["minimum"],
            "maximum": validated["maximum"],
            "model_sha256": _sha256(model_path),
            "submission_sha256": _sha256(submission_path),
        }
        result["candidate"] = candidate_manifest
        _write_json(result_path, result)

    manifest = {
        "result_sha256": _sha256(result_path),
        "oof_sha256": _sha256(oof_path),
        "candidate": candidate_manifest,
        "decision": result["decision"],
        "uploaded": False,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _progress(
        args.status_file,
        100,
        "complete",
        f"결정 {result['decision']} · 업로드 없음",
        status="complete",
    )
    print(json.dumps({"decision": result["decision"], "gates": gates, **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
