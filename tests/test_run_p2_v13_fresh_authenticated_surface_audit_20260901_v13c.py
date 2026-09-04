from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_p2_v13_fresh_authenticated_surface_audit_20260901_v13c.py"
SPEC = importlib.util.spec_from_file_location("p2_v13c", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_contract_is_zero_fit_and_no_fresh_surface() -> None:
    config = MODULE.load_config()
    assert config["decision"] == "HOLD_NO_FRESH_AUTHENTICATED_SURFACE"
    assert config["source_contract"]["fresh_authenticated_scoring_frames"] == []
    assert config["operation_limits"]["model_fits"] == 0
    assert config["operation_limits"]["candidate_predictions"] == 0


def test_raw_labels_cannot_be_repartitioned_after_v13() -> None:
    rule = MODULE.load_config()["freshness_rule"]
    assert rule["raw_label_presence_is_not_freshness"]
    assert rule["forbid_repartitioning_existing_observations_after_v13_result"]


def test_month_coverage_is_metadata_only_and_deterministic() -> None:
    values = pd.Series(["2025-02-01T00:00:00Z", "2024-12-01T00:00:00Z", "2025-02-02T00:00:00Z"])
    assert MODULE.month_coverage(values) == ["2024-12", "2025-02"]


def test_preflight_has_no_operations() -> None:
    receipt = MODULE.preflight()
    assert receipt["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert receipt["model_fits"] == 0
    assert receipt["candidate_predictions"] == 0
    assert receipt["official_rows_read"] == 0
