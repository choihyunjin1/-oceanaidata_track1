"""Read-only post-materialization QA for P1 v30."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import materialize_p1_public_transport_repair_cycle_20260831_v30 as materializer  # noqa: E402
import run_p1_parallel_candidate_cycle_20260831_v4 as official_source  # noqa: E402

REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v30/postrun-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    receipt = json.loads((materializer.ARTIFACT / "result.json").read_text(encoding="utf-8"))
    output = pd.read_csv(
        materializer.OUTPUT,
        dtype={"station": "string", "time": "string", "label": "int8"},
    )
    raw_test = pd.read_csv(
        official_source.P1_DATA / "test.csv",
        usecols=materializer.P1_KEYS,
        dtype={"station": "string", "time": "string"},
    )
    champion = pd.read_csv(
        official_source.P1_CHAMPION,
        usecols=[*materializer.P1_KEYS, "label"],
        dtype={"station": "string", "time": "string", "label": "int8"},
    )
    contract_checks = materializer.validate_output_frame(
        output,
        raw_test[materializer.P1_KEYS],
    )
    label = output["label"].to_numpy(np.int8)
    anchor = champion["label"].to_numpy(np.int8)
    additions = (label == 1) & (anchor == 0)
    removals = (label == 0) & (anchor == 1)
    local_day = (
        pd.to_datetime(raw_test["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    day = pd.DataFrame({"day": local_day, "addition": additions}).groupby(
        "day",
        observed=True,
    )["addition"].agg(["sum", "size"])
    maximum_day_share = float((day["sum"] / day["size"]).max())
    checks = {
        **contract_checks,
        "receipt_status_materialized": receipt["status"]
        == "MATERIALIZED_NOT_UPLOADED",
        "csv_hash_matches_receipt": sha256(materializer.OUTPUT)
        == receipt["output"]["sha256"],
        "csv_bytes_match_receipt": materializer.OUTPUT.stat().st_size
        == receipt["output"]["bytes"],
        "positive_rows_match_receipt": int(label.sum())
        == receipt["output"]["positive_rows"],
        "additions_match_receipt": int(additions.sum())
        == receipt["output"]["additions_vs_anchor"],
        "anchor_removals_zero": int(removals.sum()) == 0,
        "day_cap_recomputed": maximum_day_share <= 0.005
        and np.isclose(
            maximum_day_share,
            receipt["output"]["maximum_kst_day_addition_fraction"],
        ),
        "internal_result_hash_frozen": sha256(materializer.INTERNAL_RESULT)
        == materializer.EXPECTED_INTERNAL_RESULT_SHA256,
        "internal_strict_pass": receipt["internal_pass"]["calibrated_expected_points_delta"]
        >= 0.01,
        "em_converged": receipt["deployment_fit"]["em_converged"] is True,
        "official_labels_read_zero": receipt["deployment_fit"][
            "outer_official_labels_read"
        ]
        == 0,
        "hidden_truth_reads_zero": receipt["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": receipt["operations"]["uploads"] == 0,
    }
    qa = native({
        "schema_version": "p1.v30.postrun-qa.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "internal": receipt["internal_pass"],
        "output": {
            "path": str(materializer.OUTPUT.resolve()),
            "rows": len(output),
            "bytes": materializer.OUTPUT.stat().st_size,
            "sha256": sha256(materializer.OUTPUT),
            "positive_rows": int(label.sum()),
            "additions_vs_anchor": int(additions.sum()),
            "anchor_removals": int(removals.sum()),
            "maximum_kst_day_addition_fraction": maximum_day_share,
        },
        "access": {
            "materializer_official_test_covariate_reads": receipt["operations"][
                "official_test_covariate_reads"
            ],
            "qa_official_test_key_reads": 1,
            "validator_official_test_key_reads": 1,
            "materializer_champion_prediction_reads": receipt["operations"][
                "official_champion_prediction_reads"
            ],
            "qa_champion_prediction_reads": 1,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        "hashes": {
            "internal_result_sha256": sha256(materializer.INTERNAL_RESULT),
            "materialization_result_sha256": sha256(
                materializer.ARTIFACT / "result.json"
            ),
            "materializer_sha256": sha256(
                Path(__file__).with_name(
                    "materialize_p1_public_transport_repair_cycle_20260831_v30.py"
                )
            ),
            "csv_sha256": sha256(materializer.OUTPUT),
        },
    })
    REPORT.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, sort_keys=True))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
