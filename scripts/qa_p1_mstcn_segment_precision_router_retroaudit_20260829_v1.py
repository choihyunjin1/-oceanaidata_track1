"""Independent aggregate QA for the P1 MSTCN segment-router retroaudit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_segment_precision_router_retroaudit_20260829_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
RESULT = ROOT / "artifacts" / EXPERIMENT_ID / "result.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    require(config["experiment_id"] == result["experiment_id"] == EXPERIMENT_ID, "id")
    require(result["status"] == "NO_GO_RETROSPECTIVE_GATE", "status")
    require(result["passed_all_diagnostic_gates"] is False, "gate result")
    require(result["input_hashes"]["config"] == sha256(CONFIG), "config hash")
    require(result["operation_counters"]["official_test_rows_read"] == 0, "official reads")
    require(result["operation_counters"]["submission_files_created"] == 0, "submission")
    require(result["operation_counters"]["uploads"] == 0, "upload")
    require(result["operation_counters"]["model_fits"] == 3, "fit count")
    by_key = {
        (row["arm"], row["evaluation_fold"]): row for row in result["fold_results"]
    }
    q3_core = by_key[("core", "2025_q3")]
    q4_core = by_key[("core", "2025_q4")]
    q4_type = by_key[("type_augmented", "2025_q4")]
    require(q3_core["delta_f1_vs_incumbent"] > 0.0, "q3 core sign")
    require(q3_core["delta_f1_vs_raw_e150"] < 0.0, "q3 raw comparator")
    require(q4_core["delta_f1_vs_incumbent"] < 0.0, "q4 core sign")
    require(q4_type["accepted_segments"] == 0, "type abstention")
    require(q4_type["candidate"] == q4_type["incumbent"], "type exact fallback")
    require(q4_type["delta_f1_vs_raw_e150"] > 0.0, "type raw comparator")

    scheduled_tp = int(q3_core["candidate"]["tp"] + q4_type["candidate"]["tp"])
    scheduled_fp = int(q3_core["candidate"]["fp"] + q4_type["candidate"]["fp"])
    scheduled_fn = int(q3_core["candidate"]["fn"] + q4_type["candidate"]["fn"])
    scheduled_f1 = f1(scheduled_tp, scheduled_fp, scheduled_fn)
    pooled = result["pooled_q3_q4"]
    qa = {
        "schema_version": "p1.mstcn_segment_precision_router.retroaudit.qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_QA_OF_NO_GO_RESULT",
        "result_sha256": sha256(RESULT),
        "checks": {
            "config_hash": True,
            "operation_counters": True,
            "formal_no_go_reproduced": True,
            "type_q4_exact_incumbent_fallback": True,
        },
        "derived_posthoc_diagnostic_not_promotion_evidence": {
            "definition": "Q3 core candidate plus Q4 type-augmented exact fallback",
            "tp": scheduled_tp,
            "fp": scheduled_fp,
            "fn": scheduled_fn,
            "f1": scheduled_f1,
            "delta_f1_vs_pooled_incumbent": scheduled_f1 - pooled["incumbent"]["f1"],
            "delta_f1_vs_pooled_raw_e150": scheduled_f1 - pooled["raw_e150"]["f1"],
        },
        "official_test_rows_read": 0,
        "submission_files_created": 0,
        "uploads": 0,
    }
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
