"""Algorithm-only official materializer preflight for P1 v34a.

The preflight reads no official CSV. Execute remains blocked until the v33a
official metric/diff geometry is reconciled and a separate authorization is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_iors_e150_microfragment_veto_20260901_v34a"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RESULT_PATH = ROOT / "artifacts" / EXPERIMENT_ID / "result.json"


class ContractError(RuntimeError):
    """Raised when materialization is not safely authorized."""


def preflight() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    output = Path(config["official_materializer"]["output"])
    checks = {
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "terminal_internal_result": result["status"] in {"INTERNAL_PASS_OFFICIAL_GEOMETRY_BLOCKED", "TERMINAL_NO_GO"},
        "fit0": result["fit_count"] == 0,
        "official_reads_zero": result["operations"]["official_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "output_absent": not output.exists(),
        "algorithm_pinned": config["decoder"]["maximum_segment_length_inclusive"] == 2,
    }
    if not all(checks.values()):
        raise ContractError(f"preflight checks failed: {checks}")
    return {
        "schema_version": "p1.iors_e150_microfragment_veto.materializer_preflight.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": "BLOCKED_NO_GO" if result["status"] == "TERMINAL_NO_GO" else "BLOCKED_PENDING_OFFICIAL_METRIC_GEOMETRY_RECONCILIATION",
        "checks": checks,
        "algorithm": "Start from champion; find contiguous I-ORS rows where anchor=0 and E150=1; reset only complete segments of length <=2; preserve GI2 and every incumbent positive.",
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
        "execute_implemented": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        raise SystemExit("--preflight required; execute intentionally unavailable")
    print(json.dumps(preflight(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
