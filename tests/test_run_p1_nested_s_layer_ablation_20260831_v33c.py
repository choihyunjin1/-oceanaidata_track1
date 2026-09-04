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


RUNNER = load_script("v33c_runner", ROOT / "scripts/run_p1_nested_s_layer_ablation_20260831_v33c.py")
MATERIALIZER = load_script("v33c_materializer", ROOT / "scripts/materialize_p1_nested_s_layer_ablation_20260831_v33c.py")


def selection_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    incumbent: list[int] = []
    raw: list[int] = []
    minute = 0
    for fold in RUNNER.FOLDS:
        for layer in (1, 2):
            for index in range(12):
                rows.append({
                    "station": "S-ORS",
                    "year": 2025,
                    "layer": layer,
                    "time": pd.Timestamp("2025-01-01", tz="Asia/Seoul") + pd.Timedelta(minutes=minute),
                    "fold": fold,
                    "label_base": 0 if layer == 1 else int(index < 10),
                })
                incumbent.append(0)
                raw.append(1)
                minute += 10
        for index in range(12):
            rows.append({
                "station": "G-ORS",
                "year": 2025,
                "layer": 1,
                "time": pd.Timestamp("2025-01-01", tz="Asia/Seoul") + pd.Timedelta(minutes=minute),
                "fold": fold,
                "label_base": int(index < 6),
            })
            incumbent.append(int(index < 6))
            raw.append(int(index < 6))
            minute += 10
    return pd.DataFrame(rows), np.asarray(incumbent, dtype=np.int8), np.asarray(raw, dtype=np.int8)


def test_select_layers_uses_strict_f1_half_and_support_floor() -> None:
    frame, incumbent, raw = selection_frame()
    receipt = RUNNER.select_layers(frame, incumbent, raw, ["2025_q2"], 10)
    assert receipt["selected_layers"] == [1]
    assert receipt["layer_statistics"]["1"]["support"] == 12
    assert receipt["layer_statistics"]["2"]["selected"] is False


def test_nested_candidate_abstains_q2_and_preserves_other_stations() -> None:
    frame, incumbent, raw = selection_frame()
    candidate, removal, selections, deployment = RUNNER.build_nested_candidate(frame, incumbent, raw, 10)
    q2 = frame["fold"].eq("2025_q2").to_numpy()
    not_s = ~frame["station"].eq("S-ORS").to_numpy()
    assert not removal[q2].any()
    assert np.array_equal(candidate[q2], raw[q2])
    assert np.array_equal(candidate[not_s], raw[not_s])
    assert selections["2025_q3"]["prefix_folds"] == ["2025_q2"]
    assert selections["2025_q4"]["prefix_folds"] == ["2025_q2", "2025_q3"]
    assert deployment["prefix_folds"] == RUNNER.FOLDS


def official_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "station": ["S-ORS", "S-ORS", "G-ORS", "I-ORS"],
        "year": [2026] * 4,
        "layer": [1, 2, 1, 1],
        "time": [f"2026-01-01 00:{minute:02d}:00+09:00" for minute in (0, 10, 20, 30)],
        "label": [0, 0, 0, 0],
    })
    anchor = base.copy()
    e150 = base.copy()
    e150["label"] = [1, 1, 1, 0]
    champion = e150.copy()
    champion.loc[3, "label"] = 1
    return champion, e150, anchor


def test_materializer_removes_selected_s_layer_only_and_preserves_gi2() -> None:
    champion, e150, anchor = official_frames()
    candidate, removal = MATERIALIZER.build_candidate(champion, e150, anchor, [1])
    assert removal.tolist() == [True, False, False, False]
    assert candidate["label"].tolist() == [0, 1, 1, 1]


def test_materializer_frame_contract_rejects_duplicates() -> None:
    champion, _, _ = official_frames()
    duplicate = pd.concat([champion.iloc[[0]], champion.iloc[[0]]], ignore_index=True)
    try:
        MATERIALIZER.validate_frame(duplicate, 2)
    except MATERIALIZER.ContractError:
        pass
    else:
        raise AssertionError("duplicate keys must fail")
