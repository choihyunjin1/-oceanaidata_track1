from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v23.json"
SOURCE_RESULT = ROOT / "artifacts/full_internal_submission_cycle_20260831_v3/p1_result.json"
SOURCE_RUNNER = ROOT / "scripts/run_full_internal_submission_cycle_20260831_v2.py"
SOURCE_ARTIFACT_DIR = ROOT / "artifacts/full_internal_submission_cycle_20260831_v3"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v23/preflight-report.json"

EXPECTED_STATUS = "NO_GO_MISSING_CONTINUOUS_SCORE_LINEAGE_AND_POSTHOC_K"
EXPECTED_CANDIDATE = "P1_2_HIST_GBDT_OOF_STACK_UNION"
FORBIDDEN_PERSISTED_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_record(result: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in result["outputs"] if record["name"] == EXPECTED_CANDIDATE]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one {EXPECTED_CANDIDATE} record, got {len(records)}")
    return records[0]


def _persisted_score_or_model_files() -> list[str]:
    matches: list[str] = []
    for path in SOURCE_ARTIFACT_DIR.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_PERSISTED_SUFFIXES or any(
            token in name for token in ("probability", "score_table", "checkpoint", "model")
        ):
            matches.append(path.relative_to(ROOT).as_posix())
    return sorted(matches)


def load_contract() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    source = SOURCE_RUNNER.read_text(encoding="utf-8")
    candidate = _candidate_record(result)
    internal = candidate["internal"]
    checks = {
        "experiment_id": config["experiment_id"]
        == "p1_public_transport_repair_cycle_20260831_v23",
        "preflight_only": config["authorization"]["preflight_only"] is True,
        "historical_disabled": config["prohibitions"]["historical_truth_execution"] is True,
        "rerun_disabled": config["prohibitions"]["exact_model_rerun"] is True
        and config["prohibitions"]["exact_threshold_rerun"] is True,
        "top_k_unset": config["prospective_rank_stability_contract"]["top_k"] is None,
        "pending_v3": config["transport"]["penalty_points"] == "PENDING_V3",
        "minimum_delta": config["transport"]["minimum_calibrated_expected_point_delta_inclusive"]
        == 0.01,
        "source_delta": internal["delta_f1"] == 0.0013809855753390554,
        "q3_zero": internal["by_fold"]["2025_q3"]["delta_f1"] == 0.0,
        "q4_positive": internal["by_fold"]["2025_q4"]["delta_f1"]
        == 0.003432580085375281,
        "deployment_additions": candidate["additions_vs_champion"] == 4,
        "hgb_structure_present": "HistGradientBoostingClassifier(" in source
        and '"P1_2_HIST_GBDT_OOF_STACK_UNION"' in source,
        "threshold_grid_present": "np.linspace(0.10, 0.95, 35)" in source,
        "models_memory_only": "deployment_models: list[tuple[P1ModelSpec, Any, float]] = []"
        in source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v23 contract mismatch: {checks}")
    return config, candidate, source


def preflight() -> dict[str, Any]:
    config, candidate, source = load_contract()
    internal = candidate["internal"]
    persisted = _persisted_score_or_model_files()
    checks = {
        "only_aggregate_evidence_read": True,
        "source_internal_positive": internal["delta_f1"] > 0.0,
        "q3_not_positive": internal["by_fold"]["2025_q3"]["delta_f1"] == 0.0,
        "q4_positive": internal["by_fold"]["2025_q4"]["delta_f1"] > 0.0,
        "continuous_score_or_checkpoint_missing": len(persisted) == 0,
        "deployment_model_not_serialized": "joblib.dump" not in source
        and "pickle.dump" not in source,
        "top_k_not_registered": config["prospective_rank_stability_contract"]["top_k"]
        is None,
        "posthoc_k_forbidden": config["prohibitions"]["posthoc_k_from_deployment_additions"]
        is True,
        "exact_rerun_forbidden": config["prohibitions"]["exact_model_rerun"] is True,
        "historical_fit_count_zero": config["authorization"]["historical_fits"] == 0,
        "minimum_calibrated_delta_preserved": config["transport"][
            "minimum_calibrated_expected_point_delta_inclusive"
        ]
        == 0.01,
        "official_and_hidden_zero": config["authorization"]["official_rows_read"] == 0
        and config["authorization"]["hidden_truth_reads"] == 0,
        "no_lock_or_submission_authorized": config["authorization"]["attempt_locks"] == 0
        and config["authorization"]["submission_csv_writes"] == 0
        and config["authorization"]["uploads"] == 0,
    }
    evidence_complete = all(checks.values())
    return {
        "schema_version": "p1.v23.no-go-preflight.1",
        "status": EXPECTED_STATUS if evidence_complete else "INVALID_PREFLIGHT_EVIDENCE",
        "decision": "NO_GO",
        "reason": (
            "The exposed HGB candidate cannot be converted to a prospective stable top-k rule: "
            "continuous prefix/deployment scores and a reproducible checkpoint were not persisted, "
            "and k=4 would be selected after observing the deployment output and Public tie."
        ),
        "checks": checks,
        "source_metrics": {
            "internal_delta_f1": internal["delta_f1"],
            "q3_delta_f1": internal["by_fold"]["2025_q3"]["delta_f1"],
            "q4_delta_f1": internal["by_fold"]["2025_q4"]["delta_f1"],
            "q3_threshold": internal["by_fold"]["2025_q3"]["threshold"],
            "q4_threshold": internal["by_fold"]["2025_q4"]["threshold"],
            "deployment_threshold": internal["deployment_threshold"],
            "deployment_additions": candidate["additions_vs_champion"],
            "official_result": config["lineage"]["official_result"],
            "official_score_delta": config["lineage"]["official_score_delta"],
        },
        "persistence_audit": {
            "artifact_directory": SOURCE_ARTIFACT_DIR.relative_to(ROOT).as_posix(),
            "files": sorted(
                path.relative_to(ROOT).as_posix()
                for path in SOURCE_ARTIFACT_DIR.iterdir()
                if path.is_file()
            ),
            "continuous_score_or_model_files": persisted,
            "top_k": None,
        },
        "transport": {
            "penalty_points": "PENDING_V3",
            "minimum_calibrated_expected_point_delta_inclusive": 0.01,
            "raw_gate": "PENDING_V3",
            "gate_evaluated": False,
        },
        "fit_count": 0,
        "candidate_count": 0,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "source_result_sha256": sha256(SOURCE_RESULT),
            "source_runner_sha256": sha256(SOURCE_RUNNER),
        },
        "access": {
            "historical_truth_reads": 0,
            "historical_fits": 0,
            "attempt_locks": 0,
            "official_input_reads": 0,
            "official_rows_read": 0,
            "hidden_truth_reads": 0,
            "submission_csv_reads": 0,
            "submission_csv_writes": 0,
            "uploads": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(
            "v23 historical execution is prohibited: continuous score lineage and a prospective k are missing"
        )
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    started = time.perf_counter()
    result = preflight()
    result["runtime_seconds"] = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != EXPECTED_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
