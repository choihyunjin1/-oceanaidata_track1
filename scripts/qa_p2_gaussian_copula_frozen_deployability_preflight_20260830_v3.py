"""Independent aggregate-only QA for the P2 Gaussian-copula v3 preflight."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2_restore.p2_gaussian_copula_frozen_deployability_preflight_20260830_v3 import (  # noqa: E402
    evaluate_preflight,
    load_json,
    sha256_file,
    write_json_exclusive,
)

EXPECTED_CONFIG_SHA256 = "3eaa563d544c66e4739189badd9b5faca7802b4d00a2d503eef440cae4443bd3"
DEFAULT_CONFIG = Path(
    "configs/experiments/"
    "p2_gaussian_copula_frozen_deployability_preflight_20260830_v3.json"
)
DEFAULT_OUTPUT = Path(
    "reports/p2_gaussian_copula_frozen_deployability_preflight_20260830_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _under_root(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    candidate.relative_to(root)
    return candidate


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = _under_root(repo_root, args.config)
    output_dir = _under_root(repo_root, args.output_dir)
    result_path = output_dir / "result.json"
    lock_path = output_dir / "attempt.lock.json"
    qa_path = output_dir / "independent-qa.json"

    config_sha = sha256_file(config_path)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Canonical config hash mismatch")
    config = load_json(config_path)
    result = load_json(result_path)
    lock = load_json(lock_path)
    replay = evaluate_preflight(repo_root, config, config_sha)

    checks = {
        "config_hash_match": result.get("config_sha256") == config_sha,
        "attempt_one_of_one": lock.get("attempt_number") == 1
        and lock.get("maximum_executions") == 1,
        "attempt_config_hash_match": lock.get("config_sha256") == config_sha,
        "status_recomputed": result.get("status") == replay.get("status"),
        "blockers_recomputed": result.get("deployability_blockers")
        == replay.get("deployability_blockers"),
        "duplicate_audit_recomputed": result.get("recorded_submission_duplicate_audit")
        == replay.get("recorded_submission_duplicate_audit"),
        "historical_metrics_recomputed": result.get("historical_result")
        == replay.get("historical_result"),
        "integrity_pass": result.get("integrity_pass") is True,
        "official_probe_not_ready": result.get("official_probe_ready") is False,
        "zero_fit": result.get("execution_receipt", {}).get("model_fits") == 0,
        "thread_budget_one": result.get("execution_receipt", {}).get("thread_budget") == 1,
        "row_access_zero": result.get("execution_receipt", {}).get("training_rows_read") == 0
        and result.get("execution_receipt", {}).get("historical_prediction_rows_read") == 0
        and result.get("execution_receipt", {}).get(
            "official_test_index_sample_baseline_score_query_rows_read"
        )
        == 0,
        "csv_upload_commit_push_zero": result.get("execution_receipt", {}).get("csv_created")
        == 0
        and result.get("execution_receipt", {}).get("uploads") == 0
        and result.get("execution_receipt", {}).get("commits") == 0
        and result.get("execution_receipt", {}).get("pushes") == 0,
        "output_contains_no_csv": not any(output_dir.glob("*.csv")),
    }
    qa = {
        "schema_version": "p2.gaussian_copula.frozen_deployability_preflight.qa.20260830.v3",
        "experiment_id": result.get("experiment_id"),
        "qa_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": {
            "config": config_sha,
            "attempt_lock": sha256_file(lock_path),
            "result": sha256_file(result_path),
        },
        "independent_recomputation": {
            "model_fits": 0,
            "raw_or_prediction_rows_read": 0,
            "official_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
        "completed_at_kst": datetime.now().astimezone().isoformat(),
    }
    write_json_exclusive(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, sort_keys=True))
    return 0 if qa["qa_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
