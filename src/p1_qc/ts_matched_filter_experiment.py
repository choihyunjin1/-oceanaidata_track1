"""Validation for the single adaptive temperature-salinity inner comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment import sha256_file
from .ts_matched_filter import TS_MATCHED_FILTER_FEATURES

EXPECTED_MODULE_SHA = "28a2834b475e70f4a885ffe187bc23ea9eec64f69cfa7ddf80c27715a83d0a4c"


def load_and_validate_ts_contract(path: str | Path, *, project_root: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0-adaptive-inner-only":
        raise ValueError("temperature-salinity contract schema drift")
    if tuple(payload.get("single_change", {}).get("added_features", ())) != (
        TS_MATCHED_FILTER_FEATURES
    ):
        raise ValueError("temperature-salinity feature contract drift")
    provenance = payload.get("adaptive_provenance", {})
    if provenance.get("previous_inner_labels_exposed") is not True:
        raise ValueError("adaptive exposure must remain explicit")
    if provenance.get("claim_independent_validation") is not False:
        raise ValueError("adaptive inner comparison cannot claim independence")
    if provenance.get("maximum_followup_runs_in_same_inner_blocks") != 1:
        raise ValueError("only one adaptive follow-up is allowed")
    if provenance.get("no_further_same_family_retry") is not True:
        raise ValueError("same-family stop rule drift")
    authorization = payload.get("authorization", {})
    expected = {
        "inner_comparison": True,
        "outer_one_shot": False,
        "test_prediction": False,
        "submission": False,
        "commit": False,
        "push": False,
    }
    if authorization != expected:
        raise ValueError("temperature-salinity authorization drift")
    module = project_root / "src/p1_qc/ts_matched_filter.py"
    if sha256_file(module) != EXPECTED_MODULE_SHA:
        raise ValueError("temperature-salinity module hash drift")
    if payload.get("hashes", {}).get("ts_module_sha256") != EXPECTED_MODULE_SHA:
        raise ValueError("declared temperature-salinity module hash drift")
    return payload


__all__ = ["load_and_validate_ts_contract"]
