from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p3_causal_forcing_sequence_checkpoint_nested_v2.py"
SPEC = importlib.util.spec_from_file_location("p3_checkpoint_nested_v2_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_explicit_input_allowlist_is_train_only_and_has_no_official_or_era5_path() -> None:
    root = Path("workspace")
    paths = {
        "compact_cache": root / "compact",
        "sequence_cache": root / "sequence",
        "reference": root / "fixed8",
    }
    inputs = runner._input_paths(Path("source"), paths)
    assert tuple(inputs) == runner.EXPLICIT_INPUT_KEYS
    assert tuple(path.name for path in inputs.values() if path.parent == Path("source")) == runner.EXPLICIT_SOURCE_NAMES
    flattened = "\n".join([*inputs, *(path.as_posix() for path in inputs.values())]).lower()
    for forbidden in ("test_context", "test_index", "sample_submission", "submission.csv", "era5"):
        assert forbidden not in flattened


def test_training_target_loader_uses_only_explicit_train_cache(monkeypatch) -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(5, dtype=np.int64),
            "current_hs": np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
        }
    )
    target = pd.DataFrame({"anchor_id": np.arange(5, dtype=np.int64)})
    for lead in runner.OFFICIAL_LEADS:
        target[f"target_{lead}"] = anchors["current_hs"] + 0.5
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *_args, **_kwargs: target.copy())
    delta = runner._load_training_targets(
        preflight={"anchors": anchors, "outer_union": np.asarray([1, 3], dtype=np.int64)},
        inputs={"compact_cache/train_anchors.parquet": Path("unused.parquet")},
    )
    np.testing.assert_allclose(delta, 0.5)


def test_blind_frame_contains_no_target_column() -> None:
    anchors = pd.DataFrame(
        {
            "anchor_id": np.asarray([0, 1], dtype=np.int64),
            "station": ["G-ORS", "I-ORS"],
        }
    )
    frame = runner._blind_frame(
        anchors,
        np.asarray([0, 1], dtype=np.int64),
        np.ones((2, 6), dtype=float),
        fold="outer",
    )
    assert len(frame) == 12
    assert "target_hs" not in frame
    assert frame.duplicated(runner.KEYS).sum() == 0
