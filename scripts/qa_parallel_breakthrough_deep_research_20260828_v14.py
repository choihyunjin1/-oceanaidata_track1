"""Independent QA for the three-problem breakthrough research cycle."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "parallel_breakthrough_deep_research_20260828_v14"
P1_RESULT = (
    ROOT
    / "artifacts"
    / "p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1"
    / "result.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    p2_path = REPORT_DIR / "p2_metric_geometry_after_alpha50.json"
    p3_path = REPORT_DIR / "p3_orthogonal_basis_audit.json"
    p1_retro_path = REPORT_DIR / "p1_checkpoint_disagreement_retroaudit.json"
    report_path = REPORT_DIR / "report-source.md"
    ledger_path = REPORT_DIR / "claim-source-ledger.md"
    gap_path = REPORT_DIR / "gap-matrix.md"
    for path in (
        p1_retro_path,
        p2_path,
        p3_path,
        report_path,
        ledger_path,
        gap_path,
        P1_RESULT,
    ):
        require(path.is_file(), f"missing artifact: {path}")

    p1 = json.loads(P1_RESULT.read_text(encoding="utf-8"))
    p1_retro = json.loads(p1_retro_path.read_text(encoding="utf-8"))
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    p3 = json.loads(p3_path.read_text(encoding="utf-8"))

    require(p1["status"] in {"PASS_READY_NOT_UPLOADED", "NO_GO_EXACT_NO_OUTPUT"}, "P1 status")
    require(p1["test_labels_read"] is False and p1["upload_performed"] is False, "P1 leakage")
    if p1["status"] == "NO_GO_EXACT_NO_OUTPUT":
        require(p1["added_rows"] == 1, "P1 expected one-row full-test support")
        require(p1["component_lengths"] == [1], "P1 expected isolated component")
        require(p1["gate_pass"] is False, "P1 no-go gate unexpectedly passed")
        require(p1["epoch150_decoded_exact"] is True, "P1 decoded replay mismatch")
        require(
            p1["gate_checks"]["e150_prediction_array_exact"] is True,
            "P1 prediction replay mismatch",
        )
    require(p1_retro["status"] == "PASS_GO_BOUNDED_FULL_REPLAY", "P1 retrospective status")
    require(p1_retro["pooled"]["signature_rows"] == 44, "P1 signature rows")
    require(p1_retro["pooled"]["signature_true_positive_rows"] == 44, "P1 signature TP")
    require(p2["selected"]["gate_pass"] is False, "P2 gate unexpectedly passed")
    require(p2["submission_csv_written"] is False, "P2 wrote a submission")
    require(p2["decision"]["upload_performed"] is False, "P2 upload")
    require(p3["status"] == "NO_GO_NO_OFFICIAL_CANDIDATE", "P3 status")
    require(not p3["decision"]["promoted"], "P3 promoted an existing basis")
    require(p3["decision"]["official_candidate_created"] is False, "P3 candidate")
    require(p3["decision"]["upload_performed"] is False, "P3 upload")

    p1_candidate = None
    if p1["status"] == "PASS_READY_NOT_UPLOADED":
        candidate_path = Path(p1["candidate"]["ready"])
        require(candidate_path.is_file(), "P1 ready candidate missing")
        require(sha256(candidate_path) == p1["candidate"]["sha256"], "P1 candidate hash")
        data_value = os.environ.get("P1_DATA_DIR")
        require(bool(data_value), "set P1_DATA_DIR for independent candidate QA")
        sample_path = Path(str(data_value)).expanduser().resolve() / "sample_submission.csv"
        sample = pd.read_csv(sample_path)
        candidate = pd.read_csv(candidate_path)
        keys = ["station", "year", "layer", "time"]
        require(list(candidate.columns) == keys + ["label"], "P1 schema")
        require(len(candidate) == 169011, "P1 row count")
        require(candidate[keys].astype(str).equals(sample[keys].astype(str)), "P1 key/order")
        labels = candidate["label"].to_numpy()
        require(np.isin(labels, [0, 1]).all(), "P1 label domain")
        require(not candidate[keys].isna().any().any(), "P1 missing keys")
        require(not candidate.duplicated(keys).any(), "P1 duplicate keys")
        p1_candidate = {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256(candidate_path),
            "rows": len(candidate),
            "positive_rows": int(labels.sum()),
        }

    result = {
        "schema_version": "parallel_breakthrough_deep_research.qa.20260828.v14",
        "status": "PASS",
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "p1_status": p1["status"],
        "p1_candidate": p1_candidate,
        "p2_status": p2["status"],
        "p2_gate_pass": p2["selected"]["gate_pass"],
        "p3_status": p3["status"],
        "p3_promoted": p3["decision"]["promoted"],
        "official_uploads_in_cycle": 0,
        "artifact_hashes": {
            path.name: sha256(path)
            for path in (
                P1_RESULT,
                p1_retro_path,
                p2_path,
                p3_path,
                report_path,
                ledger_path,
                gap_path,
            )
        },
    }
    output = REPORT_DIR / "independent_qa.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
