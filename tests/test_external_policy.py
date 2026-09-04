from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ocean_external.policy import PolicyError, audit_catalog, preflight_external_use

CATALOG = Path("configs/external_data/catalog.toml")
HISTORICAL_PERMISSION = Path("configs/external_data/official_faq_permission.json")
ACTIVE_POLICY = Path("configs/compliance/organizer_data_policy_20260901.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_catalog_remains_readable_as_historical_audit_evidence() -> None:
    audit = audit_catalog(CATALOG)

    assert audit.source_count == 10
    assert audit.value_accessed_count == 3
    assert audit.unauthorized_value_access_count == 0


def test_old_faq_receipt_is_preserved_but_explicitly_superseded() -> None:
    receipt = json.loads(HISTORICAL_PERMISSION.read_text(encoding="utf-8"))
    active = json.loads(ACTIVE_POLICY.read_text(encoding="utf-8"))
    evidence = Path(receipt["evidence_file"])

    assert receipt["status"] == "approved"
    assert evidence.is_file()
    assert _sha256(evidence) == receipt["evidence_sha256"]
    assert active["status"] == "ACTIVE_HIGHEST_PRECEDENCE"
    assert str(HISTORICAL_PERMISSION).replace("\\", "/") in active[
        "superseded_authorization_receipts"
    ]


@pytest.mark.parametrize(
    ("problem", "source_id", "purpose"),
    [
        ("P1", "i_ors_ctd_2014_2023", "feature_design"),
        ("P2", "nasa_power_pre2024", "feature_design"),
        ("P2", "era5_pre2024", "pretraining"),
        ("P3", "kma_ocean_buoy_pre2024", "pretraining"),
        ("P3", "era5_pre2024", "pretraining"),
    ],
)
def test_every_external_observation_use_fails_before_path_access(
    problem: str,
    source_id: str,
    purpose: str,
) -> None:
    with pytest.raises(PolicyError, match="forbidden by the active 2026-09-01"):
        preflight_external_use(
            catalog_path=Path("definitely-missing-catalog.toml"),
            approval_receipt_path=Path("definitely-missing-receipt.json"),
            manifest_path=Path("definitely-missing-manifest.json"),
            problem=problem,
            source_id=source_id,
            purpose=purpose,
        )


def test_active_policy_synthetic_only_exception_is_all_conditions_required() -> None:
    active = json.loads(ACTIVE_POLICY.read_text(encoding="utf-8"))
    pretrained = active["pretrained_weights"]
    exception = pretrained["synthetic_only_exception"]

    assert active["distributed_data_only"] is True
    assert pretrained["real_observation_trained"] == "FORBIDDEN"
    assert pretrained["default_when_provenance_is_unclear"] == "FORBIDDEN"
    assert exception["allowed_only_if_all_conditions_hold"] is True
    assert len(exception["conditions"]) == 4
