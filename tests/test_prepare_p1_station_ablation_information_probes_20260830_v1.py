from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_p1_station_ablation_information_probes_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("station_probes", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "station": ["G-ORS", "I-ORS", "S-ORS", "G-ORS", "S-ORS", "I-ORS"],
        "year": [2026] * 6,
        "layer": ["L1"] * 6,
        "time": [f"2026-01-01 00:{value:02d}:00" for value in (0, 10, 20, 30, 40, 50)],
        "label": [0, 0, 0, 1, 0, 0],
    })
    anchor = base.copy()
    e150 = base.copy()
    e150["label"] = [1, 1, 1, 1, 0, 0]
    champion = e150.copy()
    champion.loc[[4, 5], "label"] = 1
    return champion, e150, anchor


def test_station_ablation_removes_only_selected_e150_additions() -> None:
    champion, e150, anchor = fixture_frames()
    candidate, removed = MODULE.build_candidate(champion, e150, anchor, {"G-ORS"})
    assert removed.tolist() == [True, False, False, False, False, False]
    assert candidate["label"].tolist() == [0, 1, 1, 1, 1, 1]


def test_anchor_and_gi_rows_are_preserved() -> None:
    champion, e150, anchor = fixture_frames()
    candidate, _ = MODULE.build_candidate(champion, e150, anchor, {"G-ORS", "S-ORS"})
    assert candidate.loc[3, "label"] == 1
    assert candidate.loc[4, "label"] == 1
    assert candidate.loc[5, "label"] == 1
