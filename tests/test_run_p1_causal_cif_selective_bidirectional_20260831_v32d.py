from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_causal_cif_selective_bidirectional_20260831_v32d.py"
SPEC = importlib.util.spec_from_file_location("p1_v32d", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return {
        "decoder": {
            "probability_threshold_inclusive": 0.5,
            "maximum_changed_fraction_per_fold": 0.5,
        },
        "validation": {"outer_folds": ["2025_q3", "2025_q4"]},
    }


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["B", "A", "A", "B", "A", "B", "A", "B"],
            "year": [2025] * 8,
            "layer": [1] * 8,
            "time": [f"2025-01-01 00:{i:02d}:00+00:00" for i in range(8)],
            "fold": ["2025_q3"] * 4 + ["2025_q4"] * 4,
            "current_router_prediction": np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8),
        }
    )


def test_bidirectional_top_fraction_and_margin_order() -> None:
    probability = np.array([0.99, 0.8, 0.01, 0.4, 0.9, 0.6, 0.1, 0.49], dtype=np.float32)
    candidate, action, receipts = MODULE.build_action_plan(frame(), probability, config())
    assert [item["selected_changes"] for item in receipts] == [2, 2]
    assert action.tolist() == [True, False, True, False, True, False, True, False]
    assert candidate.tolist() == [1, 0, 0, 1, 1, 0, 0, 1]


def test_ties_use_station_then_key_order() -> None:
    probability = np.array(
        [0.75, 0.75, 0.25, 0.25, 0.75, 0.75, 0.25, 0.25], dtype=np.float32
    )
    _, action, _ = MODULE.build_action_plan(frame(), probability, config())
    assert action.tolist() == [False, True, True, False, True, False, True, False]


def test_source_contract_is_frozen_and_metric_only() -> None:
    loaded, paths = MODULE.load_contract()
    assert loaded["authorization"]["metric_only"] is True
    assert loaded["validation"]["model_fits"] == 0
    assert loaded["authorization"]["official_reads"] == 0
    assert loaded["authorization"]["hidden_truth_reads"] == 0
    assert set(paths) == {"config", "runner", "result", "predictions", "anchor"}


def test_terminal_no_go_artifact_is_hash_bound_and_access_clean() -> None:
    artifact = ROOT / "artifacts/p1_causal_cif_selective_bidirectional_20260831_v32d"
    result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
    seal = json.loads((artifact / "action-seal.json").read_text(encoding="utf-8"))
    with np.load(artifact / "sealed_action_mask.npz", allow_pickle=False) as frozen:
        action = frozen["action"]
        candidate = frozen["candidate"]
    assert result["status"] == "NO_GO"
    assert result["fit_count"] == 0 and result["pass_count"] == 0
    assert all(value == 0 for value in result["operations"].values())
    assert seal["truth_reads_before_action_seal"] == 0
    assert seal["changes"] == int(action.sum()) == result["candidate"]["changes"]
    assert seal["action_sha256"] == MODULE.sha256_array(action.astype(np.uint8))
    assert seal["candidate_sha256"] == MODULE.sha256_array(candidate)
    assert set(result["candidate"]["by_fold"]) == {"2025_q3", "2025_q4"}
    assert result["candidate"]["strict_internal_pass"] is False
    assert not all(result["candidate"]["gates"].values())
