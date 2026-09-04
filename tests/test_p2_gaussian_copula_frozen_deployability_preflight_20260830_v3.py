from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from p2_restore import p2_gaussian_copula_frozen_deployability_preflight_20260830_v3 as preflight

EXPECTED_CONFIG_SHA256 = "3eaa563d544c66e4739189badd9b5faca7802b4d00a2d503eef440cae4443bd3"


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_bundle(tmp_path: Path, *, duplicate: bool = False) -> dict[str, object]:
    prediction = tmp_path / "artifacts" / "synthetic.npz"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"synthetic metadata-only artifact")
    commitment = {
        "outputs": {
            "block": {
                "path": "artifacts\\synthetic.npz",
                "rows": 7,
                "bytes": prediction.stat().st_size,
                "sha256": "committed-not-recomputed",
            }
        }
    }
    commitment_path = tmp_path / "receipts" / "commitment.json"
    commitment_sha = _write_json(commitment_path, commitment)
    ledger_path = tmp_path / "receipts" / "official-ledger.json"
    ledger_sha = _write_json(
        ledger_path,
        {"submitted": "candidate-token" if duplicate else "different-candidate"},
    )
    return {
        "experiment_id": preflight.EXPERIMENT_ID,
        "mode": "ZERO_FIT_HASH_LEDGER_DEPLOYABILITY_PREFLIGHT",
        "frozen_candidate": {"local_confirmation_rerun_authorized": False},
        "expected_historical_result": {"rows": 7},
        "pinned_files": [
            {
                "role": "prediction_commitment",
                "path": "receipts/commitment.json",
                "sha256": commitment_sha,
                "inspection": "json_aggregate",
            }
        ],
        "historical_prediction_artifacts": [
            {
                "block": "block",
                "path": "artifacts/synthetic.npz",
                "committed_rows": 7,
                "committed_bytes": prediction.stat().st_size,
                "committed_sha256": "committed-not-recomputed",
            }
        ],
        "claim_checks": [],
        "official_submission_duplicate_audit": {
            "scope": "synthetic local ledger",
            "ledgers": [
                {
                    "path": "receipts/official-ledger.json",
                    "sha256": ledger_sha,
                }
            ],
            "candidate_tokens": ["candidate-token"],
        },
        "deployability": {
            "official_prediction_artifact_materialized": False,
            "current_platform_duplicate_state_verified": False,
            "current_daily_quota_verified": False,
            "historical_code_hashes_were_sealed_at_v2_execution": False,
            "historical_code_hash_limitation": "synthetic limitation",
            "required_before_probe": ["synthetic requirement"],
        },
        "execution_policy": {
            "maximum_executions": 1,
            "maximum_model_fits": 0,
            "maximum_threads": 1,
            "result_based_retry": False,
            "result_based_tuning": False,
            "read_training_rows": False,
            "read_prediction_rows": False,
            "read_official_test_index_sample_baseline_score_query_rows": False,
            "create_csv": False,
            "upload": False,
            "commit": False,
            "push": False,
        },
        "predeclared_decision": {
            "recorded_duplicate": "BLOCKED_RECORDED_OFFICIAL_DUPLICATE",
            "integrity_pass_without_official_artifact_or_live_platform_checks": (
                "PREFLIGHT_INTEGRITY_PASS_DEPLOYMENT_BLOCKED"
            ),
            "probe_ready_only_if_all_required_before_probe_are_later_satisfied": "READY",
        },
    }


def test_zero_fit_preflight_blocks_missing_deployment_inputs_and_does_not_hash_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_bundle(tmp_path)
    hashed: list[Path] = []
    original = preflight.sha256_file

    def record_hash(path: Path) -> str:
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(preflight, "sha256_file", record_hash)
    result = preflight.evaluate_preflight(tmp_path, config, "synthetic-config-sha")

    assert result["status"] == "PREFLIGHT_INTEGRITY_PASS_DEPLOYMENT_BLOCKED"
    assert result["official_probe_ready"] is False
    assert result["execution_receipt"]["model_fits"] == 0
    assert result["execution_receipt"]["historical_prediction_rows_read"] == 0
    assert not any(path.suffix == ".npz" for path in hashed)


def test_recorded_duplicate_fails_closed(tmp_path: Path) -> None:
    config = _synthetic_bundle(tmp_path, duplicate=True)
    result = preflight.evaluate_preflight(tmp_path, config, "synthetic-config-sha")

    assert result["status"] == "BLOCKED_RECORDED_OFFICIAL_DUPLICATE"
    assert result["recorded_submission_duplicate_audit"]["matched_token_count"] == 1


def test_repo_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(preflight.PreflightError, match="escapes repository root"):
        preflight.resolve_repo_path(tmp_path, "../outside.json")


def test_attempt_reservation_is_exclusive(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    preflight.reserve_attempt(output_dir, "config-sha")
    with pytest.raises(preflight.PreflightError, match="Attempt lock already exists"):
        preflight.reserve_attempt(output_dir, "config-sha")


def test_canonical_repository_config_hash() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "configs"
        / "experiments"
        / "p2_gaussian_copula_frozen_deployability_preflight_20260830_v3.json"
    )
    assert preflight.sha256_file(config_path) == EXPECTED_CONFIG_SHA256
