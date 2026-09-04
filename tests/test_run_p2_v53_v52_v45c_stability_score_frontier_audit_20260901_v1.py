from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts"
    / "run_p2_v53_v52_v45c_stability_score_frontier_audit_20260901_v1.py"
)
RESULT = (
    ROOT
    / "reports"
    / "p2_v53_v52_v45c_stability_score_frontier_audit_20260901_v1"
    / "result.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("p2_v53_frontier", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_is_zero_fit_and_policy_is_sealed() -> None:
    module = _module()
    receipt = module.preflight()
    assert receipt["status"] == "ZERO_FIT_FRONTIER_PREFLIGHT_READY"
    assert receipt["model_fits"] == 0
    assert receipt["official_rows_read"] == 0
    assert receipt["hidden_rows_read"] == 0
    assert receipt["submission_csv_created"] == 0
    assert receipt["uploads"] == 0
    policy = receipt["selection_policy"]
    assert policy["v52_weight_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert policy["heldout_outcomes_visible_during_selection"] is False


def test_lofo_choice_does_not_use_its_heldout_truth() -> None:
    module = _module()
    config, _ = module.load_config()
    folds = np.repeat(np.asarray(module.FOLDS), 6)
    layers = np.tile(np.repeat(np.asarray(module.LAYERS), 2), 3)
    blind = pd.DataFrame({"fold": folds, "layer": layers})
    stable = np.full(len(blind), 0.10, dtype=float)
    reference = stable.copy()
    score = np.linspace(-0.05, 0.20, len(blind), dtype=float)
    truth_a = np.linspace(-0.10, 0.10, len(blind), dtype=float)
    truth_b = truth_a.copy()
    changed = blind["fold"].eq(module.FOLDS[0]).to_numpy()
    truth_b[changed] += 100.0
    _, receipts_a, _ = module._select_lofo(
        config, blind, truth_a, reference, stable, score
    )
    _, receipts_b, _ = module._select_lofo(
        config, blind, truth_b, reference, stable, score
    )
    selected_a = {
        (item["heldout_fold"], item["layer"]): item["selected_v52_weight"]
        for item in receipts_a
    }
    selected_b = {
        (item["heldout_fold"], item["layer"]): item["selected_v52_weight"]
        for item in receipts_b
    }
    for layer in module.LAYERS:
        assert selected_a[(module.FOLDS[0], layer)] == selected_b[
            (module.FOLDS[0], layer)
        ]


def test_terminal_result_fails_closed_and_preserves_official_boundary() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == (
        "NO_GO_NO_LOFO_FRONTIER_CANDIDATE_BEATS_V23_AND_SAFETY"
    )
    assert result["fit_count"] == 0
    assert result["promotion_gate"]["pass"] is False
    assert result["promotion_gate"]["non_harm_cells"] == 7
    assert result["promotion_gate"]["total_cells"] == 9
    counters = result["operation_counters"]
    for name in (
        "model_fits",
        "external_observation_rows_read",
        "external_reanalysis_rows_read",
        "external_forecast_rows_read",
        "pretrained_weight_files_loaded",
        "official_test_index_rows_read",
        "sample_rows_read",
        "baseline_file_rows_read",
        "query_support_rows_read",
        "hidden_truth_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        assert counters[name] == 0


def test_terminal_payload_hash_recomputes() -> None:
    module = _module()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["result_payload_sha256"] == module._result_hash(result)
