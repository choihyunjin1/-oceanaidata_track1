from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_p1_window_phase_consistency_20260829_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_window_phase_test_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@dataclass
class _Segment:
    segment_id: int
    row_ids: np.ndarray

    @property
    def size(self) -> int:
        return len(self.row_ids)


@dataclass
class _Layout:
    segments: tuple[_Segment, ...]


def test_phase_zero_matches_default_start_geometry() -> None:
    layout = _Layout((_Segment(0, np.arange(10, dtype=np.int64)),))
    windows = runner.build_phase_windows(layout, window_size=8, stride=4, phase_rows=0)
    assert [window.start for window in windows] == [0, 4, 8]
    assert [window.left_pad for window in windows] == [0, 0, 0]
    assert [window.row_ids.tolist() for window in windows] == [
        list(range(8)),
        list(range(4, 10)),
        [8, 9],
    ]


def test_plus_phase_prepends_invalid_context_and_covers_every_row() -> None:
    layout = _Layout((_Segment(0, np.arange(10, dtype=np.int64)),))
    windows = runner.build_phase_windows(layout, window_size=8, stride=4, phase_rows=2)
    assert [window.start for window in windows] == [-2, 2, 6]
    assert [window.left_pad for window in windows] == [2, 0, 0]
    covered = np.unique(np.concatenate([window.row_ids for window in windows]))
    np.testing.assert_array_equal(covered, np.arange(10))


def test_phase_materialize_and_stitch_preserve_aligned_rows() -> None:
    layout = _Layout((_Segment(0, np.arange(10, dtype=np.int64)),))
    windows = runner.build_phase_windows(layout, window_size=8, stride=4, phase_rows=2)
    features = np.arange(10, dtype=np.float32)[:, None]
    values, valid = runner._materialize_phase_features(features, windows)
    assert valid[0, :2].tolist() == [0.0, 0.0]
    predictions = values[:, :, 0]
    stitched = runner.stitch_phase_predictions(predictions, windows, n_rows=10)
    np.testing.assert_allclose(stitched, np.arange(10, dtype=np.float32), rtol=0.0, atol=0.0)


def test_symmetric_js_is_zero_for_identical_logits_and_positive_otherwise() -> None:
    logits = torch.tensor([-2.0, 0.0, 3.0])
    zero = runner.bernoulli_symmetric_js(logits, logits)
    shifted = runner.bernoulli_symmetric_js(logits, -logits)
    assert float(zero) == pytest.approx(0.0, abs=1.0e-12)
    assert float(shifted) > 0.0
    assert float(shifted) == pytest.approx(
        float(runner.bernoulli_symmetric_js(-logits, logits)), rel=0.0, abs=1.0e-12
    )


def test_exclusive_npz_round_trip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "blind.npz"
    expected = np.arange(7, dtype=np.float32)
    digest = runner._exclusive_npz(path, probability=expected)
    assert digest == runner._sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["probability"], expected)
    with pytest.raises(FileExistsError):
        runner._exclusive_npz(path, probability=expected)


def test_config_has_one_recipe_and_one_warm_start_run() -> None:
    config = runner._load_config()
    assert config["fixed_recipe"]["alternate_phase_rows"] == 256
    assert config["paired_view_warm_start"]["run_count"] == 1
    assert config["paired_view_warm_start"]["epochs"] == 5
    assert config["q2_preflight"]["kill_if_any"] == {
        "default_replay_not_bitwise_identical": True,
        "q99_absolute_probability_difference_below": 0.05,
        "proposal_xor_rows_below": 50,
        "fixed_average_anchor_union_delta_f1_below": 0.001,
    }


def test_read_only_preflight_pins_runtime_and_six_states() -> None:
    result = runner.check_only()
    assert result["result"] == "PASS"
    assert result["frozen_checkpoint_count"] == 6
    assert result["official_interface_reads"] == 0
