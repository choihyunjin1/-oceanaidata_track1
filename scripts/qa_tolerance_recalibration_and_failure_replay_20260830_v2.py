from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/tolerance_recalibration_and_failure_replay_20260830_v2"
REPLAY = REPORT / "failure-replay.json"
OUTPUT = REPORT / "independent-qa.json"
POLICY = ROOT / "configs/goals/tolerance_recalibration_and_failure_replay_20260830_v2.json"
RUNNER = ROOT / "scripts/audit_tolerance_recalibration_and_failure_replay_20260830_v2.py"
REQUIRED_DOCS = [REPORT / "report-source.md", REPORT / "claim-source-ledger.md", REPORT / "gap-matrix.md"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def build_qa() -> dict[str, Any]:
    policy = _read_json(POLICY)
    replay = _read_json(REPLAY)
    provenance = replay["provenance"]
    cases = {row["candidate"]: row for row in replay["key_case_replay"]}
    official = replay["official_false_negative_cases"]

    p1_evidence = _read_json(ROOT / provenance["inputs"]["p1_official_evidence"]["path"])
    p2_before = _read_json(ROOT / provenance["inputs"]["p2_alpha50_receipt"]["path"])
    leaderboard = _read_json(ROOT / provenance["inputs"]["leaderboard"]["path"])
    official_text = (ROOT / provenance["inputs"]["official_results"]["path"]).read_text(encoding="utf-8")
    p3_20 = re.search(r"KMA 장기보정 20% \| RMSE ([0-9.]+) \| ([0-9.]+)", official_text)
    p3_40 = re.search(r"KMA 장기보정 40% \| RMSE ([0-9.]+) \| ([0-9.]+)", official_text)
    assert p3_20 is not None and p3_40 is not None

    p1_official_delta = (
        float(p1_evidence["official_score_evidence"]["e150_plus_gi2_public_f1"])
        - float(p1_evidence["official_score_evidence"]["e150_union_all_public_f1"])
    )
    p2_official_delta = (
        float(p2_before["official_result"]["public_rmse"])
        - float(leaderboard["our_team"]["official_metrics"]["P2_RMSE_C"])
    )
    p3_official_delta = float(p3_20.group(1)) - float(p3_40.group(1))

    paths_in_replay = [ROOT / item["path"] for item in provenance["inputs"].values()]
    forbidden_artifact_suffixes = {".csv", ".parquet", ".npz", ".pt", ".pth", ".ckpt"}
    generated_files = [POLICY, RUNNER, REPLAY, *REQUIRED_DOCS]
    generated_text = "\n".join(path.read_text(encoding="utf-8") for path in generated_files)

    checks = {
        "policy_hash_matches": provenance["policy"]["sha256"] == _sha256(POLICY),
        "runner_hash_matches": provenance["runner"]["sha256"] == _sha256(RUNNER),
        "all_input_hashes_match": all(
            _sha256(ROOT / item["path"]) == item["sha256"]
            for item in provenance["inputs"].values()
        ),
        "all_input_paths_exist": all(path.is_file() for path in paths_in_replay),
        "no_forbidden_row_artifact_in_provenance": all(
            path.suffix.lower() not in forbidden_artifact_suffixes for path in paths_in_replay
        ),
        "all_required_docs_exist": all(path.is_file() for path in REQUIRED_DOCS),
        "all_48_family_rows_unique": len(replay["historical_family_replay"]) == 48
        and len({row["family_id"] for row in replay["historical_family_replay"]}) == 48,
        "all_35_canonical_groups_unique": len(replay["canonical_group_replay"]) == 35
        and len({(row["problem"], row["group"]) for row in replay["canonical_group_replay"]}) == 35,
        "official_false_negative_p1_recomputed": _close(
            official["P1"]["observed_official_metric_improvement"], p1_official_delta
        )
        and 0.0 < p1_official_delta < official["P1"]["old_gate_raw_metric"],
        "official_false_negative_p2_recomputed": _close(
            official["P2"]["observed_official_metric_improvement"], p2_official_delta
        )
        and 0.0 < p2_official_delta < official["P2"]["old_gate_raw_metric"],
        "official_false_negative_p3_recomputed": _close(
            official["P3"]["observed_official_metric_improvement"], p3_official_delta
        )
        and 0.0 < p3_official_delta < official["P3"]["old_gate_raw_metric"],
        "five_p2_interval_positive_cases": sum(
            row["problem"] == "P2"
            and row["new_state"] == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
            and row["benefit_ci90"][0] > 0.0
            for row in replay["key_case_replay"]
        )
        == 5,
        "p3_lead_is_exploratory": cases["lead_continuous"]["new_state"]
        == "EXPLORATORY_CHALLENGER_RESEARCH_ONLY",
        "clear_harms_preserved": all(
            cases[name]["new_state"] == "PRIMARY_HARM_RESEARCH_ONLY"
            for name in (
                "group_dro_fixed_objective",
                "event_balanced_supcon",
                "availability_aware_copula_v2",
                "catboost_repaired_confirmation",
                "selection_matched_masked_ssl",
            )
        ),
        "numerical_and_scientific_tolerance_separated": policy["tolerance_layers"]["numerical_replay"]["deterministic_metric_absolute_epsilon"] == 1e-12
        and policy["tolerance_layers"]["scientific_effect"]["default_directional_margin_when_no_justified_sesoi_exists"] == 0.0
        and policy["tolerance_layers"]["scientific_effect"]["universal_nonzero_raw_metric_margin"] is None,
        "outlier_hard_delete_is_not_default": policy["outlier_policy"]["default_hard_delete"] is False,
        "zero_fit_raw_official_csv_upload": replay["summary"]["model_fits"] == 0
        and replay["summary"]["raw_training_or_prediction_rows_read"] == 0
        and replay["summary"]["official_test_sample_submission_hidden_or_query_rows_read"] == 0
        and replay["summary"]["csv_created"] == 0
        and replay["summary"]["uploads"] == 0,
        "no_credential_assignment_in_generated_text": re.search(
            r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+",
            generated_text,
        )
        is None,
    }

    return {
        "schema_version": "tolerance_failure_replay.independent_qa.20260830.v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "P1_official_improvement_f1": p1_official_delta,
            "P2_official_improvement_c": p2_official_delta,
            "P3_official_improvement_m": p3_official_delta,
            "family_rows": len(replay["historical_family_replay"]),
            "canonical_groups": len(replay["canonical_group_replay"]),
            "key_cases": len(replay["key_case_replay"]),
        },
        "access_and_execution": {
            "model_fits": 0,
            "raw_training_or_prediction_rows_read": 0,
            "official_test_sample_submission_hidden_or_query_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
            "commit_or_push": 0,
        },
        "hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in generated_files},
    }


def main() -> None:
    qa = build_qa()
    OUTPUT.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": qa["status"], "checks": len(qa["checks"])}))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
