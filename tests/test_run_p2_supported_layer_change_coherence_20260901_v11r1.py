from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p2_supported_layer_change_coherence_20260901_v11r1.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p2_supported_layer_change_coherence_20260901_v11r1.json"
SPEC = importlib.util.spec_from_file_location("p2_v11r1_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_overlay_pins_predecessor_and_only_mutability_change() -> None:
    recovery = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert recovery["predecessor"]["model_fits"] == 0
    assert recovery["only_change"].endswith("before in-place support masking")
    assert recovery["scientific_contract_changed"] is False
    assert recovery["candidate_changed"] is False
    assert recovery["support_changed"] is False
    assert recovery["folds_changed"] is False
    assert recovery["huber_changed"] is False
    assert recovery["gate_changed"] is False


def test_merged_science_is_exact_predecessor_contract() -> None:
    merged = runner.load_config()
    predecessor_path = ROOT / merged["contract_repair"]["predecessor"]["config_path"]
    original = json.loads(predecessor_path.read_text(encoding="utf-8"))
    for key in (
        "semantic_fingerprint",
        "source_contract",
        "negative_registry_audit",
        "training_only_influence",
        "candidate",
        "evaluation",
        "operation_limits",
    ):
        assert merged[key] == original[key]


def test_preflight_is_byte_identical_and_writeable_contract_passes() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["writeable_buffer_contract"] is True
    assert first["scientific_contract_changed"] is False
    assert first["data_rows_read"] == 0
    assert first["official_rows_read"] == 0


def test_explicit_copy_is_writeable() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    score = frame.max(axis=1).to_numpy(dtype=float, copy=True)
    score[0] = np.nan
    assert score.flags.writeable
    assert np.isnan(score[0])


def test_runner_contains_the_single_repair_expression() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert ".to_numpy(dtype=float, copy=True)" in text
    assert "test_index.csv" not in text
    assert "to_csv(" not in text
