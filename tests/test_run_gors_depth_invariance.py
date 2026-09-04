from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from p1_qc.config import FoldWindowConfig, P1QCConfig, SplitConfig
from scripts.run_gors_depth_invariance import (
    _append_exposure_event,
    _label_blind_fold_indices,
    _relative_increase,
)


def test_frozen_fold_indices_use_keys_and_dates_without_reading_label_values() -> None:
    frame = pd.DataFrame(
        {
            "station": ["G-ORS"] * 5,
            "year": [2025] * 5,
            "layer": [1] * 5,
            "time": pd.date_range("2025-01-01", periods=5, freq="D", tz="Asia/Seoul").astype(str),
            "label": [0, 1, 0, 1, 0],
        }
    )
    reference = frame.loc[[3, 2], ["station", "year", "layer", "time"]].copy()
    reference["fold"] = "fold_1"
    reference["label"] = [99, 99]  # ignored even though deliberately invalid
    config = P1QCConfig(
        splits=SplitConfig(
            folds=(
                FoldWindowConfig(
                    "fold_1",
                    "2025-01-02T23:50:00+09:00",
                    "2025-01-03T00:00:00+09:00",
                    "2025-01-05T00:00:00+09:00",
                ),
            )
        )
    )

    folds = _label_blind_fold_indices(frame, reference, config)
    assert len(folds) == 1
    name, train_idx, val_idx = folds[0]
    assert name == "fold_1"
    np.testing.assert_array_equal(train_idx, [0, 1])
    np.testing.assert_array_equal(val_idx, [3, 2])

    relabelled = frame.copy()
    relabelled["label"] = [1, 0, 1, 0, 1]
    second = _label_blind_fold_indices(relabelled, reference, config)
    np.testing.assert_array_equal(second[0][1], train_idx)
    np.testing.assert_array_equal(second[0][2], val_idx)


def test_exposure_event_is_append_only_jsonl(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event":"preregistered"}\n', encoding="utf-8")
    _append_exposure_event(
        ledger,
        event="outer_evaluated",
        run_id="unit-run",
        preregistration_sha256="a" * 64,
    )
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"event": "preregistered"}
    appended = json.loads(lines[1])
    assert appended["event"] == "outer_evaluated"
    assert appended["outer_result_count"] == 1
    assert appended["run_id"] == "unit-run"


def test_fp_relative_increase_zero_denominator_rule() -> None:
    assert _relative_increase(0.0, 0.0) == 0.0
    assert _relative_increase(1.0, 0.0) is None
    assert _relative_increase(1.1, 1.0) == 0.10000000000000009
