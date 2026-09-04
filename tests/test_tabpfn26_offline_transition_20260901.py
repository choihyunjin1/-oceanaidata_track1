from __future__ import annotations

import json
from pathlib import Path

from ocean_tabpfn26.offline import (
    CLASSIFIER_FILENAME,
    REGRESSOR_FILENAME,
    inspect_preflight,
)


def _receipt(path: Path, **updates: object) -> Path:
    value: dict[str, object] = {
        "schema_version": "ocean.tabpfn26.user_license_receipt.v1",
        "license_accepted_by_user": True,
        "model_version": "v2.6",
        "synthetic_only_provenance_reviewed": True,
        "competition_use_terms_reviewed": True,
        "accepted_at_utc": "2026-09-01T12:00:00+00:00",
        "source_url": "https://platform.priorlabs.ai",
    }
    value.update(updates)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_missing_local_license_and_weights_fail_closed(tmp_path: Path) -> None:
    result = inspect_preflight(workspace=tmp_path, env={})
    assert result["status"] == "BLOCKED"
    assert result["auto_download_allowed"] is False
    assert any("license receipt" in blocker for blocker in result["blockers"])
    assert sum("missing local checkpoint" in blocker for blocker in result["blockers"]) == 2


def test_local_receipt_does_not_allow_wrong_checkpoint_bytes(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path / "receipt.json")
    classifier = tmp_path / CLASSIFIER_FILENAME
    regressor = tmp_path / REGRESSOR_FILENAME
    classifier.write_bytes(b"classifier")
    regressor.write_bytes(b"regressor")
    result = inspect_preflight(
        workspace=tmp_path,
        env={
            "TABPFN26_LICENSE_RECEIPT_PATH": str(receipt),
            "TABPFN26_CLASSIFIER_PATH": str(classifier),
            "TABPFN26_REGRESSOR_PATH": str(regressor),
        },
    )
    assert not any("license receipt" in blocker for blocker in result["blockers"])
    assert sum("checkpoint byte count mismatch" in blocker for blocker in result["blockers"]) == 2
    assert result["weights"]["classifier"] is None
    assert result["weights"]["regressor"] is None


def test_license_receipt_rejects_credentials(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path / "receipt.json", token="must-not-be-stored")
    result = inspect_preflight(
        workspace=tmp_path,
        env={"TABPFN26_LICENSE_RECEIPT_PATH": str(receipt)},
    )
    assert "license receipt must not contain credentials" in result["blockers"]
