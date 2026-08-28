"""Independently verify the P1 lower-bound veto report and shadow contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "p1_mstcn_lower_bound_veto_20260829_v2"
EVIDENCE_PATH = REPORT_DIR / "evidence.json"
REPORT_PATH = REPORT_DIR / "report-source.md"
SHADOW_DIR = ROOT / "artifacts" / "p1_mstcn_official_shadow_lower_bound_veto_20260829_v1"
SHADOW_RESULT_PATH = SHADOW_DIR / "result.json"
HISTORICAL_RESULT_PATH = ROOT / "artifacts" / "p1_mstcn_bootstrap_lower_bound_veto_20260829_v1" / "result.json"
DEPLOYMENT_PREFLIGHT_PATH = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1" / "preflight.json"
E150_PATH = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1" / "P1_MSTCN_E150_ROUTER_UNION_ALL.csv"
CHAMPION_PATH = Path(r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_DEADLINE_INFORMATION_PROBES_READY\P1_1_E150_PLUS_GI_SPIKE2\P1_submission.csv")
OUTPUT_PATH = REPORT_DIR / "qa.json"
KEYS = ["station", "year", "layer", "time"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execute() -> dict:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    historical = json.loads(HISTORICAL_RESULT_PATH.read_text(encoding="utf-8"))
    shadow = json.loads(SHADOW_RESULT_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(DEPLOYMENT_PREFLIGHT_PATH.read_text(encoding="utf-8"))
    report = REPORT_PATH.read_text(encoding="utf-8")
    require(evidence["decision"] == "PASS_LABEL_FREE_SHADOW_RELEVANCE_NOT_PROMOTED", "evidence decision")
    require(historical["status"] == "PASS_SHADOW_AUDIT_ELIGIBLE", "historical status")
    require(shadow["status"] == "PASS_LABEL_FREE_SHADOW_RELEVANCE", "shadow status")
    require(all(historical["gate_checks"].values()), "historical gate")
    require(all(shadow["gate_checks"].values()), "shadow gate")
    require(sha256(HISTORICAL_RESULT_PATH) == evidence["historical_lower_bound_result"]["sha256"], "historical hash")
    require(sha256(SHADOW_RESULT_PATH) == evidence["official_shadow_result"]["sha256"], "shadow hash")
    counters = shadow["operation_counters"]
    require(counters["submission_files_created"] == 0 and counters["uploads"] == 0, "mutation counter")
    require(counters["official_test_feature_values_read"] == 0 and counters["official_truth_values_read"] == 0, "official value contract")
    require(not list(SHADOW_DIR.glob("*.csv")), "shadow candidate CSV exists")
    anchor_path = Path(preflight["external_inputs"]["current_router"]["path"])
    anchor = pd.read_csv(anchor_path)
    e150 = pd.read_csv(E150_PATH)
    champion = pd.read_csv(CHAMPION_PATH)
    require(len(anchor) == len(e150) == len(champion) == 169011, "official row count")
    require(anchor[KEYS].astype(str).equals(e150[KEYS].astype(str)), "anchor/e150 keys")
    require(anchor[KEYS].astype(str).equals(champion[KEYS].astype(str)), "anchor/champion keys")
    anchor_label = anchor["label"].astype(int)
    e150_label = e150["label"].astype(int)
    champion_label = champion["label"].astype(int)
    require(int(((e150_label == 1) & (anchor_label == 0)).sum()) == 333, "e150 additions")
    require(int(((champion_label == 1) & (e150_label == 0)).sum()) == 2, "GI-only rows")
    observed = shadow["official_shadow"]
    require(observed["accepted_segments"] == 3 and observed["accepted_e150_rows"] == 8, "accepted counts")
    require(observed["anchor_rows_removed"] == 0 and observed["gi_only_rows_removed"] == 0, "preservation")
    require(observed["shadow_positive_rows"] == observed["anchor_positive_rows"] + observed["official_champion_gi_only_rows"] + observed["accepted_e150_rows"], "shadow arithmetic")
    require(observed["accepted_frequency_minimum"] >= 0.9, "frequency threshold")
    for marker in ("공식 검증 가치가 있는 무라벨 shadow 후보", "CSV 생성과 업로드는 이번 사이클에서 모두 0회", "적응적 회고 증거"):
        require(marker in report, f"report marker: {marker}")
    return {
        "schema_version": "p1.mstcn_lower_bound_veto.report_qa.v2",
        "checked_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS",
        "historical_status": historical["status"],
        "shadow_status": shadow["status"],
        "accepted_segments": observed["accepted_segments"],
        "accepted_rows": observed["accepted_e150_rows"],
        "anchor_rows_removed": observed["anchor_rows_removed"],
        "gi_only_rows_removed": observed["gi_only_rows_removed"],
        "submission_files_created": counters["submission_files_created"],
        "uploads": counters["uploads"],
        "verified_hashes": {
            "evidence": sha256(EVIDENCE_PATH),
            "report": sha256(REPORT_PATH),
            "historical_result": sha256(HISTORICAL_RESULT_PATH),
            "shadow_result": sha256(SHADOW_RESULT_PATH),
            "anchor": sha256(anchor_path),
            "e150": sha256(E150_PATH),
            "official_champion": sha256(CHAMPION_PATH),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "READY", "qa_path": str(OUTPUT_PATH)}, indent=2))
        return
    result = execute()
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
