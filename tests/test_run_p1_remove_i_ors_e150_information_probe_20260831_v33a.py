from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("v33a_runner", ROOT / "scripts/run_p1_remove_i_ors_e150_information_probe_20260831_v33a.py")
MATERIALIZER = load_script("v33a_materializer", ROOT / "scripts/materialize_p1_remove_i_ors_e150_information_probe_20260831_v33a.py")


def fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "station": ["G-ORS", "I-ORS", "I-ORS", "S-ORS", "I-ORS"],
        "year": [2026] * 5,
        "layer": [1, 1, 2, 1, 3],
        "time": [f"2026-01-01 00:{minute:02d}:00" for minute in (0, 10, 20, 30, 40)],
        "label": [0, 0, 1, 0, 0]
    })
    anchor = base.copy()
    e150 = base.copy()
    e150["label"] = [1, 1, 1, 0, 0]
    champion = e150.copy()
    champion.loc[[3, 4], "label"] = 1
    return champion, e150, anchor


def test_official_builder_removes_only_i_ors_e150_addition() -> None:
    champion, e150, anchor = fixture_frames()
    candidate, removal = MATERIALIZER.build_candidate(champion, e150, anchor)
    assert removal.tolist() == [False, True, False, False, False]
    assert candidate["label"].tolist() == [1, 0, 1, 1, 1]


def test_metric_block_counts_removed_tp_and_fp() -> None:
    truth = np.array([1, 0, 1, 0], dtype=np.int8)
    reference = np.array([1, 1, 1, 1], dtype=np.int8)
    candidate = np.array([1, 0, 0, 1], dtype=np.int8)
    removal = reference != candidate
    block = RUNNER.metric_block(truth, reference, candidate, np.ones(4, dtype=bool), removal)
    assert block["removed_true_positives"] == 1
    assert block["removed_false_positives"] == 1
    assert block["removed_e150_additions"] == 2


def test_validate_frame_rejects_duplicate_keys() -> None:
    champion, _, _ = fixture_frames()
    duplicate = pd.concat([champion.iloc[[0]], champion.iloc[[0]]], ignore_index=True)
    try:
        MATERIALIZER.validate_frame(duplicate, 2)
    except MATERIALIZER.ContractError:
        pass
    else:
        raise AssertionError("duplicate keys must fail")
