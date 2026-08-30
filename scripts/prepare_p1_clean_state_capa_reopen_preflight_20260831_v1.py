"""Build the zero-fit P1 clean-state/CAPA lineage-reopen preflight.

This script reads aggregate experiment records only.  It does not read the
official P1 test, sample submission, hidden labels, prediction rows, or raw
observation values, and it never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_clean_state_capa_reopen_preflight_20260831_v1.json"
SOURCES = {
    "leaderboard": ROOT
    / "reports/leaderboard_headroom_double_research_20260829_v1/leaderboard_snapshot.json",
    "validation": ROOT / "artifacts/validation_system_audit_20260822/p1.json",
    "v6r2_disposition": ROOT
    / "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r2_disposition/OWNER_STATIC_QA_NO_GO_20260823.json",
    "v6r4_config": ROOT
    / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4.json",
    "science_projection": ROOT
    / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2_science_projection.json",
    "historical_ledger": ROOT
    / "reports/historical_model_reaudit_20260831_v1/candidate-ledger.json",
    "official_probe": ROOT
    / "reports/official_information_probe_cycle_20260830_v1/independent-qa.json",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_result() -> dict[str, Any]:
    missing = [
        str(path.relative_to(ROOT)) for path in [CONFIG, *SOURCES.values()] if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"required aggregate sources missing: {missing}")

    config = _load(CONFIG)
    source = {name: _load(path) for name, path in SOURCES.items()}
    validation = source["validation"]
    v6r2 = source["v6r2_disposition"]
    v6r4 = source["v6r4_config"]
    leaderboard = source["leaderboard"]
    official = source["official_probe"]

    perf = validation["frozen_oof_performance"]
    type_recall = perf["type_recall"]
    hard_names = ("I-ORS|1", "I-ORS|5", "S-ORS|2")
    station_layers = {item["group"]: item for item in perf["station_layer_aggregates"]}
    hard_cells = {
        name: {
            "f1": float(station_layers[name]["f1"]),
            "test_share_proxy": float(station_layers[name]["test_share"]),
        }
        for name in hard_names
    }
    hard_share = sum(item["test_share_proxy"] for item in hard_cells.values())

    v6r2_fits = int(v6r2["counters"]["model_fits"])
    v6r2_ran = bool(v6r2["reviewed_state"]["actual_run_performed"])
    v6r4_ran = bool(v6r4["execution_gate"]["current_actual_run_performed"])
    v6r4_fits = 0 if not v6r4_ran else int(v6r4["exact_completion_counters"]["top_level_fits"])

    checks = {
        "all_required_source_files_exist": True,
        "v6r2_actual_run_false": not v6r2_ran,
        "v6r2_model_fits_zero": v6r2_fits == 0,
        "v6r4_actual_run_false": not v6r4_ran,
        "v6r4_model_fits_zero": v6r4_fits == 0,
        "offset_and_drift_headroom_present": (
            float(type_recall["offset"]) < 0.75 and float(type_recall["drift"]) < 0.75
        ),
        "hard_cell_headroom_present": all(hard_cells[name]["f1"] < 0.75 for name in hard_names),
        "segment_decoder_materially_distinct": config["lineage"]["decoder_change"]
        == "POINTWISE_LOGISTIC_FIXED_RUN_FILTER_TO_PENALIZED_SEGMENT_LIKELIHOOD",
        "official_test_sample_submission_hidden_rows_read_zero": True,
        "model_fits_zero": True,
        "csv_created_zero": True,
        "uploads_zero": True,
    }
    decision = (
        "READY_TO_PREREGISTER_RESEARCH_ONLY_NOT_READY_TO_FIT"
        if all(checks.values())
        else "STOP_STAGE0_CONTRACT_FAILURE"
    )

    return {
        "schema_version": "p1.clean_state_capa_reopen_preflight.result.v1",
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "scientific_disposition": "REOPEN_UNEXECUTED_FAMILY",
        "stage1_authorized": False,
        "why_not_stage1": config["stage1_authorization"]["reason"],
        "leaderboard_headroom_snapshot": {
            "observed_at_kst": leaderboard["observed_at_kst"],
            "our_public_f1": float(leaderboard["our_team"]["official_metrics"]["P1_F1"]),
            "inferred_problem_best_f1": float(
                leaderboard["inferred_problem_best_metrics"]["P1_F1"]
            ),
            "planning_only_f1_gap": float(leaderboard["inferred_problem_best_metrics"]["P1_F1"])
            - float(leaderboard["our_team"]["official_metrics"]["P1_F1"]),
            "point_gap": float(leaderboard["problem_best"]["P1"]["gap_from_us"]),
            "mapping_is_not_official": True,
        },
        "local_mechanism_headroom": {
            "offset_recall": float(type_recall["offset"]),
            "drift_recall": float(type_recall["drift"]),
            "june_f1": float(perf["monthly_f1"]["6"]),
            "hard_cells": hard_cells,
            "hard_cell_test_share_proxy_sum": hard_share,
        },
        "lineage_evidence": {
            "v6r2_actual_run_performed": v6r2_ran,
            "v6r2_model_fits": v6r2_fits,
            "v6r2_technical_p0_count": int(v6r2["p0_count"]),
            "v6r2_technical_p1_count": int(v6r2["p1_count"]),
            "v6r4_actual_run_performed": v6r4_ran,
            "v6r4_model_fits": v6r4_fits,
            "old_representation": source["science_projection"]["hypothesis_id"],
            "old_decoder": "pointwise_logistic_plus_fixed_run_filter",
            "new_decoder": "joint_point_and_collective_penalized_segment_likelihood",
            "semantic_relation": "HIGH_REPRESENTATION_OVERLAP_MATERIALLY_DISTINCT_DECODER",
        },
        "official_champion_context": {
            "status": official["p1"]["status"],
            "public_f1": float(official["p1"]["champion_public_f1"]),
            "hidden_truth_reads": int(official["official_action_summary"]["hidden_truth_reads"]),
        },
        "checks": checks,
        "source_files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in {"config": CONFIG, **SOURCES}.items()
        },
        "execution_audit": {
            "aggregate_json_files_read": len(SOURCES) + 1,
            "raw_training_rows_read": 0,
            "official_test_sample_submission_hidden_rows_read": 0,
            "model_fits": 0,
            "prediction_rows_created": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_result()
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
