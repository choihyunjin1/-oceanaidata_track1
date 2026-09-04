from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1r1.py"
CONFIG = ROOT / "configs/experiments/p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1r1.json"


def _module():
    spec = importlib.util.spec_from_file_location("test_p1_v47r1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corrected_support_gate_path_and_old_model_path_fail_fast() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(KeyError):
        _ = config["model"]["minimum_rows_per_supported_environment"]
    assert module.resolve_support_thresholds(config) == (4096, 2)
    contaminated = json.loads(json.dumps(config))
    contaminated["model"]["minimum_rows_per_supported_environment"] = 4096
    with pytest.raises(RuntimeError, match="must not be injected"):
        module.resolve_support_thresholds(contaminated)


def test_recovery_preserves_frozen_science() -> None:
    current = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / "configs/experiments/p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1.json").read_text(encoding="utf-8")
    )
    for key in ["representation", "representation_support_gate", "model", "selection", "anchor", "diagnostics", "parts", "decision", "operations"]:
        assert current[key] == parent[key]
    assert current["experiment_id"].endswith("v1r1")
    assert current["recovery"]["science_changes"] == 0


def test_v47_parent_namespace_remains_consumed() -> None:
    parent_lock = ROOT / "artifacts/p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1.ATTEMPT_LOCK.json"
    lock = json.loads(parent_lock.read_text(encoding="utf-8"))
    assert lock["status"] == "CONSUMED_EXACTLY_ONCE"
    assert lock["experiment_id"] == "p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1"
