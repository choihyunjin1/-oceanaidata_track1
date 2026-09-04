from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r2.py"
CONFIG = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r2.json"
EXECUTION = ROOT / "src/p1_qc/long_event_segment_proposal_rescore_execution_v6.py"


def _load():
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation_v1r2", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_contract_is_exact_three_alias_non_scientific_repair() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    bridge = payload["pinned_compatibility_bridge"]
    assert payload["experiment_id"].endswith("_v1r2")
    assert bridge["preexisting_required_aliases"] == ["TabularEncoder"]
    assert bridge["exact_missing_aliases"] == [
        "apply_postprocess",
        "detect_plateaus",
        "detect_singleton_spikes",
    ]
    assert bridge["other_aliases_added"] == bridge["callable_wrappers"] == 0
    assert bridge["scientific_change"] is False
    assert payload["frozen_operation_graph"]["maximum_lifetime_physical_fits"] == 72
    assert payload["unchanged_science"]["decision_gates"] is True


def test_static_required_attribute_enumeration_is_exact() -> None:
    source = EXECUTION.read_text(encoding="utf-8")
    enumerated = set(re.findall(r"\bnumerical\.([A-Za-z_][A-Za-z0-9_]*)", source))
    assert enumerated == {
        "TabularEncoder",
        "apply_postprocess",
        "detect_plateaus",
        "detect_singleton_spikes",
    }


def test_bridge_uses_exact_pinned_callable_identities() -> None:
    module = _load()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from p1_qc import pipeline, rules

        numerical = SimpleNamespace(TabularEncoder=pipeline.TabularEncoder)
        execution = SimpleNamespace(__file__=str(EXECUTION))
        receipt = module._bridge(ROOT, {"numerical": numerical, "execution": execution})
        assert numerical.apply_postprocess is pipeline.apply_postprocess
        assert numerical.detect_plateaus is rules.detect_plateaus
        assert numerical.detect_singleton_spikes is rules.detect_singleton_spikes
        assert receipt["other_aliases_added"] == 0
        assert all(
            value["callable_identity_exact"] and not value["wrapper_used"]
            for value in receipt["callable_receipts"].values()
        )
    finally:
        if sys.path[0] == str(ROOT / "src"):
            sys.path.pop(0)
