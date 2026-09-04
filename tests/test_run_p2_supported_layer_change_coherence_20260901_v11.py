from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p2_supported_layer_change_coherence_20260901_v11.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "p2_supported_layer_change_coherence_20260901_v11.json"
SPEC = importlib.util.spec_from_file_location("p2_v11_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_contract_fixes_supported_layers_and_zero_fit() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    influence = config["training_only_influence"]
    assert influence["public_layers"] == [1, 5, 6, 7]
    assert influence["minimum_available_public_layers"] == 3
    assert influence["row_deletion"] is False
    assert influence["parameter_inherited_without_v10_metrics"] is True
    assert config["candidate"]["candidate_count"] == 1
    assert config["operation_limits"]["maximum_fit_count"] == 0


def test_preflight_is_byte_identical_zero_operation() -> None:
    first = runner.preflight()
    second = runner.preflight()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ZERO_OPERATION_PREFLIGHT_PASS"
    assert first["supported_public_layers"] == [1, 5, 6, 7]
    assert first["data_rows_read"] == 0
    assert first["model_fits"] == 0
    assert first["official_rows_read"] == 0
    assert first["submission_csv_created"] == 0


def test_cross_layer_deviation_flags_isolated_not_coherent_change() -> None:
    coherent = np.asarray([5.0, 5.2, 4.8, 5.1])
    isolated = np.asarray([5.0, 5.2, 4.8, 20.0])
    coherent_score = np.max(np.abs(coherent - np.median(coherent)))
    isolated_score = np.max(np.abs(isolated - np.median(isolated)))
    assert coherent_score < 1.0
    assert isolated_score > 6.0


def test_huber_weight_is_noop_below_cutoff_and_bounded_above() -> None:
    score = np.asarray([5.9, 12.0, 120.0])
    weight = np.ones(3)
    active = score > 6.0
    weight[active] = np.maximum(0.5, 6.0 / score[active])
    assert weight.tolist() == [1.0, 0.5, 0.5]


def test_semantic_audit_marks_v10_and_learned_gate_closed() -> None:
    audit = runner.engine.semantic_audit(runner.engine.load_config())
    closed = set(audit["closed_or_excluded"])
    assert "single_layer_absolute_change_outlier_v10" in closed
    assert "learned_soft_gate_or_benefit_gate" in closed
    assert "row_deletion_or_truth_selected_outlier_filter" in closed


def test_runner_has_no_materialization_or_upload_code() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "test_index.csv" not in text
    assert "sample_submission" not in text
    assert "to_csv(" not in text
    assert "requests.post" not in text
