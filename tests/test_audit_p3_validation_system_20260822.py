from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/validation_system_audit_20260822/p3.json"


def _load() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_p3_validation_audit_is_aggregate_only_and_fail_closed() -> None:
    payload = _load()
    assert payload["decision"] == (
        "KEEP_FROZEN_INCUMBENT_AS_RISK_CONTROL_ONLY__HIDDEN_GENERALIZATION_UNVERIFIED"
    )
    assert (
        payload["official_index_and_context_fidelity"]["hidden_target_values_available_or_read"]
        is False
    )
    operations = payload["operation_counters"]
    assert operations["model_fits"] == 0
    assert operations["prediction_generations"] == 0
    assert operations["prediction_writes"] == 0
    assert operations["official_test_target_reads"] == 0
    assert operations["submission_reads"] == 0
    assert operations["submission_writes"] == 0
    assert operations["uploads"] == 0
    assert operations["canonical_run_anchor_cache_target_columns_materialized"] == 0
    assert operations["canonical_run_non_oof_local_target_rows_selected"] == 0
    assert operations["superseded_failed_draft_run_target_columns_materialized"] == 6
    serialized = ARTIFACT.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in serialized
    assert '"raw_rows"' not in serialized
    assert '"predictions"' not in serialized


def test_p3_validation_audit_reconciles_grain_sampling_and_metrics() -> None:
    payload = _load()
    frozen = payload["frozen_oof_fidelity"]
    assert frozen["rows"] == 1092
    assert frozen["cases"] == 182
    assert frozen["duplicate_keys"] == 0
    assert frozen["anchor_target_max_abs_error_m"] == 0.0
    sampling = payload["sampling_and_event_audit"]
    assert sampling["implementation"]["fold_membership_exactly_recomputed"] is True
    assert sampling["spacing"]["within_fold_station_min_gap_hours"] == 78.0
    assert sampling["spacing"]["all_folds_station_min_gap_hours"] < 72.0
    assert sampling["spacing"]["cross_window_adjacent_pairs_below_78h"] == 1
    assert sampling["episode_structure"]["unique_station_episode_count"] == 181
    local = payload["local_performance"]
    assert abs(local["incumbent"]["rmse_m"] - 0.7801609198910191) < 1e-14
    assert abs(local["persistence"]["rmse_m"] - 0.8629766294655163) < 1e-14
    calibration = payload["official_baseline_calibration"]
    assert calibration["official_T"]["is_hidden_model_score"] is False
    assert calibration["local_minus_official_B_m"] > 0.09


def test_p3_validation_audit_payload_digest_is_reproducible() -> None:
    payload = _load()
    expected = payload.pop("payload_sha256_before_integrity_field")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == expected
