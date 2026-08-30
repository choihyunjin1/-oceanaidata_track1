"""Independent aggregate-only QA for the P3 lead-continuous v3 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiments/p3_lead_continuous_fresh_episode_confirmation_20260830_v3.json"
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: Any) -> str:
    data = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["canonical_paths"]
    result_path = ROOT / paths["result"]
    lock_path = ROOT / paths["attempt_lock"]
    blind_path = ROOT / paths["blind_seal"]
    for path in (result_path, lock_path, blind_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    unsealed = dict(result)
    seal = unsealed.pop("seal")
    checks = {
        "terminal_exactly_once_complete": result["status"]
        == "TERMINAL_EXACTLY_ONCE_COMPLETE",
        "state_is_inconclusive_for_one_fresh_block": result["terminal_evidence_state"]
        == "INCONCLUSIVE_FRESH_SINGLE_EPISODE_INSUFFICIENT_DEPENDENCE_UNITS",
        "submission_not_ready_and_no_official_action": bool(
            result["submission_readiness"] == "NOT_READY_INSUFFICIENT_FRESH_SUPPORT"
            and result["official_action_authorized"] is False
        ),
        "one_fresh_case_six_rows": bool(
            result["primary"]["cases"] == 1 and result["primary"]["rows"] == 6
        ),
        "no_degenerate_interval_or_bootstrap": bool(
            result["uncertainty"]["benefit_ci90_m"] is None
            and result["uncertainty"]["bootstrap_replicates_executed"] == 0
        ),
        "all_runner_integrity_checks_pass": bool(
            result["integrity_checks"] and all(result["integrity_checks"].values())
        ),
        "fit_budget_exact": bool(
            result["execution"]["candidate_fit_count"] == 1
            and result["execution"]["catboost_fit_count"] == 0
            and result["execution"]["router_fit_count"] == 0
            and result["execution"]["hyperparameter_search_count"] == 0
        ),
        "zero_official_csv_upload_or_raw_prediction_output": bool(
            result["data_access"][
                "official_test_context_index_sample_baseline_score_submission_hidden_rows_read"
            ]
            == 0
            and result["execution"]["csv_output_count"] == 0
            and result["execution"]["upload_count"] == 0
            and result["data_access"]["raw_prediction_rows_persisted"] == 0
        ),
        "attempt_lock_consumed_no_retry": bool(
            lock["status"] == "ATTEMPT_CONSUMED_NO_RETRY"
            and lock["maximum_executions"] == 1
            and lock["result_based_retry"] is False
        ),
        "blind_seal_hash_matches_result": bool(
            _file_sha256(blind_path) == result["blind_seal"]["file_sha256"]
            and blind["joint_prediction_sha256"]
            == result["blind_seal"]["joint_prediction_sha256"]
        ),
        "result_payload_seal_valid": bool(
            seal["payload_without_seal_sha256"] == _payload_sha256(unsealed)
        ),
    }
    payload = {
        "schema_version": "p3.lead_continuous_fresh_episode_confirmation.independent_qa.v3",
        "experiment_id": config["experiment_id"],
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "files": {
            "config_sha256": _file_sha256(config_path),
            "result_sha256": _file_sha256(result_path),
            "attempt_lock_sha256": _file_sha256(lock_path),
            "blind_seal_sha256": _file_sha256(blind_path),
        },
        "raw_or_official_rows_read": 0,
    }
    qa_path = ROOT / paths["qa_result"]
    payload["qa_file_sha256"] = _payload_sha256(payload)
    written_sha256 = _write_exclusive(qa_path, payload)
    return {
        "status": payload["status"],
        "qa_path": qa_path.relative_to(ROOT).as_posix(),
        "qa_sha256": written_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
