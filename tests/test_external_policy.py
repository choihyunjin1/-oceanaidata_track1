from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ocean_external.policy import PolicyError, audit_catalog, preflight_external_use

CATALOG = Path("configs/external_data/catalog.toml")
OFFICIAL_PERMISSION = Path("configs/external_data/official_faq_permission.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _approval(
    tmp_path: Path,
    *,
    source_id: str,
    problem: str,
    purpose: str,
    with_holder_permission: bool = False,
) -> Path:
    organizer_evidence = tmp_path / "organizer_approval.txt"
    organizer_evidence.write_text("approved for isolated test", encoding="utf-8")
    rights_evidence: dict[str, dict[str, str]] = {}
    if with_holder_permission:
        holder_evidence = tmp_path / "rights_holder_permission.txt"
        holder_evidence.write_text("rights holder permission", encoding="utf-8")
        rights_evidence[source_id] = {
            "evidence_file": str(holder_evidence),
            "evidence_sha256": _sha256(holder_evidence),
        }
    return _write_json(
        tmp_path / "approval.json",
        {
            "schema_version": "1.0",
            "status": "approved",
            "received_at": "2026-08-17T12:00:00+09:00",
            "organizer_channel": "test fixture",
            "evidence_file": str(organizer_evidence),
            "evidence_sha256": _sha256(organizer_evidence),
            "allowed_sources": [source_id],
            "allowed_problems": [problem],
            "allowed_purposes": [purpose],
            "cutoff_by_problem": {problem: "2023-12-31T23:59:59+09:00"},
            "rights_evidence": rights_evidence,
        },
    )


def _manifest(
    tmp_path: Path,
    *,
    source_id: str,
    observed_end: str = "2023-12-31T23:50:00+09:00",
    candidate_exists: bool = True,
) -> Path:
    candidate = tmp_path / "external_candidate.bin"
    if candidate_exists:
        candidate.write_bytes(b"fixture-not-observational-data")
        candidate_sha = _sha256(candidate)
    else:
        candidate_sha = "0" * 64
    return _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "1.0",
            "source_id": source_id,
            "local_file": str(candidate),
            "file_sha256": candidate_sha,
            "observed_start": "2023-01-01T00:00:00+09:00",
            "observed_end": observed_end,
            "row_count": 1,
            "variables": ["fixture"],
            "transformation_log": "none; unit-test fixture",
        },
    )


def test_catalog_is_metadata_only_and_has_expected_sources() -> None:
    audit = audit_catalog(CATALOG)

    assert audit.accepted is True
    assert audit.source_count == 10
    assert audit.ready_open_source_count == 5
    assert audit.rights_blocked_source_count == 5
    assert audit.value_accessed_count == 3
    assert audit.unauthorized_value_access_count == 0


def test_official_faq_permission_is_bound_to_public_evidence() -> None:
    permission = json.loads(OFFICIAL_PERMISSION.read_text(encoding="utf-8"))
    evidence = Path(permission["evidence_file"])

    assert permission["status"] == "approved"
    assert permission["organizer_channel"] == "official public FAQ API, id=9"
    assert evidence.is_file()
    assert _sha256(evidence) == permission["evidence_sha256"]
    assert set(permission["allowed_problems"]) == {"P1", "P2", "P3"}


def test_missing_approval_fails_before_manifest_or_candidate(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="official competition permission"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=tmp_path / "missing-approval.json",
            manifest_path=tmp_path / "missing-manifest.json",
            problem="P3",
            source_id="kma_ocean_buoy_pre2024",
            purpose="pretraining",
        )


def test_organizer_evidence_sha_mismatch_is_fail_closed(tmp_path: Path) -> None:
    approval = _approval(
        tmp_path,
        source_id="kma_ocean_buoy_pre2024",
        problem="P3",
        purpose="pretraining",
    )
    raw = json.loads(approval.read_text(encoding="utf-8"))
    raw["evidence_sha256"] = "0" * 64
    _write_json(approval, raw)

    with pytest.raises(PolicyError, match="competition permission evidence SHA256 mismatch"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=approval,
            manifest_path=tmp_path / "missing-manifest.json",
            problem="P3",
            source_id="kma_ocean_buoy_pre2024",
            purpose="pretraining",
        )


def test_open_source_passes_all_provenance_checks(tmp_path: Path) -> None:
    source_id = "kma_ocean_buoy_pre2024"
    approval = _approval(tmp_path, source_id=source_id, problem="P3", purpose="pretraining")
    manifest = _manifest(tmp_path, source_id=source_id)

    result = preflight_external_use(
        catalog_path=CATALOG,
        approval_receipt_path=approval,
        manifest_path=manifest,
        problem="P3",
        source_id=source_id,
        purpose="pretraining",
    )

    assert result["accepted"] is True
    assert result["source_id"] == source_id
    assert result["candidate_sha256"] == _sha256(tmp_path / "external_candidate.bin")


def test_sors_requires_independent_rights_holder_permission(tmp_path: Path) -> None:
    source_id = "s_ors_ctd_2015_2023"
    approval = _approval(tmp_path, source_id=source_id, problem="P2", purpose="pretraining")

    with pytest.raises(PolicyError, match="separate rights-holder permission"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=approval,
            manifest_path=tmp_path / "missing-manifest.json",
            problem="P2",
            source_id=source_id,
            purpose="pretraining",
        )


def test_sors_passes_with_both_permissions_and_valid_manifest(tmp_path: Path) -> None:
    source_id = "s_ors_ctd_2015_2023"
    approval = _approval(
        tmp_path,
        source_id=source_id,
        problem="P2",
        purpose="pretraining",
        with_holder_permission=True,
    )
    manifest = _manifest(tmp_path, source_id=source_id)

    result = preflight_external_use(
        catalog_path=CATALOG,
        approval_receipt_path=approval,
        manifest_path=manifest,
        problem="P2",
        source_id=source_id,
        purpose="pretraining",
    )

    assert result["accepted"] is True


def test_cutoff_rejects_before_candidate_file_is_touched(tmp_path: Path) -> None:
    source_id = "era5_pre2024"
    approval = _approval(tmp_path, source_id=source_id, problem="P3", purpose="pretraining")
    manifest = _manifest(
        tmp_path,
        source_id=source_id,
        observed_end="2024-01-01T00:00:00+09:00",
        candidate_exists=False,
    )

    with pytest.raises(PolicyError, match="exceed the approved cutoff"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=approval,
            manifest_path=manifest,
            problem="P3",
            source_id=source_id,
            purpose="pretraining",
        )


def test_review_required_source_cannot_reach_manifest(tmp_path: Path) -> None:
    source_id = "argo_gdac_pre2024"
    approval = _approval(tmp_path, source_id=source_id, problem="P2", purpose="pretraining")

    with pytest.raises(PolicyError, match="requires rights/provenance review"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=approval,
            manifest_path=tmp_path / "missing-manifest.json",
            problem="P2",
            source_id=source_id,
            purpose="pretraining",
        )


def test_candidate_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    source_id = "i_ors_ctd_2014_2023"
    approval = _approval(tmp_path, source_id=source_id, problem="P2", purpose="feature_design")
    manifest = _manifest(tmp_path, source_id=source_id)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["file_sha256"] = "0" * 64
    _write_json(manifest, raw)

    with pytest.raises(PolicyError, match="external candidate SHA256 mismatch"):
        preflight_external_use(
            catalog_path=CATALOG,
            approval_receipt_path=approval,
            manifest_path=manifest,
            problem="P2",
            source_id=source_id,
            purpose="feature_design",
        )


@pytest.mark.parametrize(
    ("path", "problem", "source_ids"),
    [
        (
            Path("configs/experiments/p2_external_depth_query_v1.json"),
            "P2",
            {"s_ors_ctd_2015_2023", "i_ors_ctd_2014_2023"},
        ),
        (
            Path("configs/experiments/p3_external_storm_pretrain_v1.json"),
            "P3",
            {"kma_ocean_buoy_pre2024", "era5_pre2024"},
        ),
    ],
)
def test_external_experiments_are_preregistered_and_permission_unblocked(
    path: Path,
    problem: str,
    source_ids: set[str],
) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    preflight = raw["external_preflight"]

    assert raw["status"] == "ready_for_open_source_preflight"
    assert raw["execution_authorized"] is True
    assert raw["objective"]["problem"] == problem
    assert preflight["required"] is True
    assert preflight["organizer_approval_required"] is False
    assert preflight["competition_permission_receipt"] == (
        "configs/external_data/official_faq_permission.json"
    )
    assert preflight["raw_values_accessed"] is False
    assert raw["execution"] == {
        "downloaded": False,
        "trained": False,
        "evaluated": False,
        "submitted": False,
    }
    assert source_ids <= set(
        [preflight.get("primary_source"), preflight.get("fallback_source")]
        + preflight.get("primary_sources", [])
    )
