from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "materialize_p1_public_transport_repair_cycle_20260831_v31m1.py"
)
SPEC = importlib.util.spec_from_file_location("p1_v31m1_materializer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_contract_is_exact_and_clean() -> None:
    deployment, base, result = MODULE.load_contract()
    assert base["candidate"] == "P1_1_PREFIX_LOGIT_SHRUNK_LABEL_SHIFT_EM"
    assert result["fit_count"] == 2
    assert result["operations"] == {
        "historical_reads": 1,
        "official_reads": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    assert deployment["data_policy"]["organizer_distributed_data_only"] is True


def test_output_validator_accepts_exact_binary_frame() -> None:
    rows = 169_011
    keys = pd.DataFrame(
        {
            "station": np.repeat("G-ORS", rows),
            "year": np.repeat(2025, rows),
            "layer": np.repeat(1, rows),
            "time": np.arange(rows).astype(str),
        }
    )
    submission = keys.copy()
    submission["label"] = (np.arange(rows) % 2).astype(np.int8)
    checks = MODULE.validate_output_frame(submission, keys)
    assert all(checks.values())


def test_deployability_guard_accepts_small_nonduplicate_addition() -> None:
    champion = np.array([1, 0, 0, 0], dtype=np.int8)
    additions = np.array([0, 1, 0, 0], dtype=bool)
    label = np.maximum(champion, additions).astype(np.int8)
    checks = MODULE.deployability_checks(label, champion, additions, 0.30)
    assert all(checks.values())


def test_deployability_guard_blocks_constant_positive() -> None:
    champion = np.array([1, 0, 0, 0], dtype=np.int8)
    additions = np.array([0, 1, 1, 1], dtype=bool)
    label = np.ones(4, dtype=np.int8)
    checks = MODULE.deployability_checks(label, champion, additions, 0.30)
    assert checks["binary_nonconstant"] is False
    assert checks["positive_fraction_within_historical_multiplier"] is False
