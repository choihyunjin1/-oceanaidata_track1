from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_historical_model_reaudit_20260831_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("historical_reaudit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exhaustive_counts_and_statuses() -> None:
    module = load_module()
    output = module.build()
    assert output["coverage"] == {
        "historical_family": 48,
        "canonical_group": 35,
        "key_case": 20,
        "workflow_exception": 4,
    }
    records = output["records"]
    assert all(row["primary_status"] in module.STATUSES for row in records)
    assert all(
        row["model_card_id"] in module.CARD_SPECS
        for row in records
        if row["grain"] in {"historical_family", "key_case"}
    )


def test_later_evidence_overrides_stale_reopen_labels() -> None:
    module = load_module()
    output = module.build()
    records = {row["record_id"]: row for row in output["records"] if row["grain"] == "key_case"}
    assert records["sobol_trial18_threshold08"]["primary_status"] == "CLOSED_EXACT"
    assert "CHECKPOINT_PEAK" in records["sobol_trial18_threshold08"]["status_tags"]
    assert "PROXY_EXPOSED" in records["sobol_trial18_threshold08"]["status_tags"]
    assert records["gaussian_copula_v2"]["primary_status"] == "CLOSED_EXACT"
    assert "PROXY_EXPOSED" in records["gaussian_copula_v2"]["status_tags"]
    assert records["lead_continuous"]["primary_status"] == "DISCOVERY_ONLY"


def test_technical_failures_are_not_scientific_harm() -> None:
    module = load_module()
    output = module.build()
    families = {
        row["record_id"]: row for row in output["records"] if row["grain"] == "historical_family"
    }
    assert families["P3-F12"]["primary_status"] == "INVALID_TECHNICAL"
    workflows = [row for row in output["records"] if row["grain"] == "workflow_exception"]
    assert workflows
    assert all(row["primary_status"] == "INVALID_TECHNICAL" for row in workflows)


def test_fingerprints_are_unique_within_each_grain() -> None:
    module = load_module()
    output = module.build()
    for grain in output["coverage"]:
        rows = [row for row in output["records"] if row["grain"] == grain]
        assert len({row["fingerprint_sha256"] for row in rows}) == len(rows)
