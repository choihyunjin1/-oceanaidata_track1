"""Run the preregistered one-shot P2 local-M2 phase experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_test_features, build_training_features
from p2_restore.model import fit_model
from p2_restore.research import (
    P2ResearchBlendModel,
    append_public_dynamics,
    append_public_m2_harmonics,
    comparison_diagnostics,
    paired_rmse_bootstrap,
    run_m2_phase_stability_screen,
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
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _progress(
    path: Path, progress: float, phase: str, detail: str, status: str = "running"
) -> None:
    _write_json(
        path,
        {
            "title": "P2 M2 진폭·위상 단일 가설",
            "progress": progress,
            "phase": phase,
            "detail": detail,
            "status": status,
            "eta": "자동 계산",
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _validate_preregistration(preregistration: Path, data_dir: Path) -> dict[str, object]:
    contract = json.loads(preregistration.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "p2_m2_local_phase_v1":
        raise ValueError("unexpected P2 M2 experiment id")
    if (
        contract.get("status") != "authorized_one_shot"
        or contract.get("upload_allowed") is not False
    ):
        raise ValueError("P2 M2 experiment is not authorized as a local-only one-shot")
    if contract["feature_contract"].get("parameter_search") is not False:
        raise ValueError("parameter search is forbidden for the P2 M2 experiment")
    for name, expected in contract["sources"].items():
        actual = _sha256(data_dir / name)
        if actual != expected:
            raise ValueError(f"source hash mismatch for {name}")
    reference = Path("artifacts/p2_method_scout/result.json")
    if _sha256(reference) != contract["frozen_reference"]["result_sha256"]:
        raise ValueError("frozen P2 method-scout result hash changed")
    return contract


def _evaluate_gates(
    contract: dict[str, object],
    reports: dict[str, object],
    diagnostics: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    limits = contract["promotion_gates"]
    block_deltas = {
        name: values["phase_blend50"]["rmse"] - values["current_blend50"]["rmse"]
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
        default=Path("configs/experiments/p2_m2_local_phase_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p2_m2_local_phase_v1"))
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_m2_phase.json")
    )
    args = parser.parse_args()
    _progress(args.status_file, 2, "contract", "사전등록·원본 SHA 검증")
    data_dir = resolve_data_dir(args.data_dir)
    contract = _validate_preregistration(args.preregistration, data_dir)
    data = load_p2_data(data_dir)

    _progress(args.status_file, 18, "feature", "공개층 전용 M2 조화 특징 20개 생성")
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, data.observations)

    _progress(args.status_file, 38, "validation", "동일 8개 계절 블록 one-shot 학습·검증")
    reports, oof = run_m2_phase_stability_screen(base, lean, phase)
    diagnostics = comparison_diagnostics(
        oof, reference="current_blend50", candidate="phase_blend50"
    )
    bootstrap = paired_rmse_bootstrap(
        oof,
        reference="current_blend50",
        candidate="phase_blend50",
        replicates=2000,
        seed=20260816,
    )
    gates = _evaluate_gates(contract, reports, diagnostics, bootstrap)
    previous = json.loads(Path("artifacts/p2_method_scout/result.json").read_text(encoding="utf-8"))
    reproduction_error = max(
        abs(
            values["current_blend50"]["rmse"]
            - previous["stability_blocks"][name]["blend50"]["rmse"]
        )
        for name, values in reports.items()
    )
    if reproduction_error > 1e-12:
        raise RuntimeError(f"frozen Blend50 reproduction failed: {reproduction_error}")

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
        "blocks": reports,
        "diagnostics": diagnostics,
        "paired_day_bootstrap": bootstrap,
        "promotion_gates": gates,
        "decision": "PROMOTE" if gates["passed"] else "REJECT_AND_CLOSE_FAMILY",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    oof.to_parquet(args.output_dir / "oof.parquet", index=False, compression="zstd")

    candidate_manifest: dict[str, object] | None = None
    if gates["passed"]:
        _progress(args.status_file, 82, "candidate", "통과 후보 전체 학습·26,061행 제출 검증")
        base_model = fit_model(base, seed=20260816)
        phase_model = fit_model(phase, seed=20260816)
        model = P2ResearchBlendModel(base_model=base_model, lean_model=phase_model)
        test_base = build_test_features(data)
        test_dynamic = append_public_dynamics(test_base, data.observations)
        test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
        test_phase = append_public_m2_harmonics(test_lean, data.observations)
        prediction = model.predict(test_base, test_phase)
        submission = build_submission(data.test_index, prediction)
        model_path = args.output_dir / "model.joblib"
        submission_path = Path("submissions/p2/P2_M2_PHASE_BLEND50.csv")
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        submission.to_csv(submission_path, index=False, encoding="utf-8", lineterminator="\n")
        validated = validate_submission(submission_path, data.test_index)
        candidate_manifest = {
            "submission": str(submission_path),
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
        "oof_sha256": _sha256(args.output_dir / "oof.parquet"),
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
