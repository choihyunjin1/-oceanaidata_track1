from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1_qc.typed_duration_semimarkov import (  # noqa: E402
    DurationDecoderConfig,
    decode_binary_segment,
    decode_independent_types,
    same_unary_control,
)


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "layer": [2] * rows,
            "time": pd.date_range("2024-01-01T00:00:00+09:00", periods=rows, freq="10min").astype(
                str
            ),
        }
    )


def _config() -> DurationDecoderConfig:
    return DurationDecoderConfig(
        duration_rows={
            "spike": (1, 1),
            "noise": (2, 4),
            "flatline": (2, 4),
            "offset": (3, 5),
            "drift": (3, 5),
        },
        start_penalty=1.0,
        stop_penalty=0.0,
    )


def _valid_run_lengths(mask: tuple[int, ...], minimum: int, maximum: int) -> bool:
    padded = (0, *mask, 0)
    starts = [
        index for index in range(len(mask) + 1) if padded[index] == 0 and padded[index + 1] == 1
    ]
    stops = [
        index for index in range(len(mask) + 1) if padded[index] == 1 and padded[index + 1] == 0
    ]
    return all(
        minimum <= stop - start <= maximum for start, stop in zip(starts, stops, strict=True)
    )


class TypedDurationSemiMarkovTests(unittest.TestCase):
    def test_binary_decoder_matches_exhaustive_map(self) -> None:
        probabilities = np.asarray([0.1, 0.8, 0.9, 0.7, 0.2, 0.95], dtype=float)
        decoded = decode_binary_segment(
            probabilities,
            minimum_duration=2,
            maximum_duration=3,
            start_penalty=1.0,
        )
        logits = np.log(probabilities) - np.log1p(-probabilities)
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for mask in itertools.product((0, 1), repeat=len(probabilities)):
            if not _valid_run_lengths(mask, 2, 3):
                continue
            starts = sum(
                value == 1 and (index == 0 or mask[index - 1] == 0)
                for index, value in enumerate(mask)
            )
            score = float(np.dot(mask, logits) - starts)
            candidates.append((score, mask))
        expected = max(candidates, key=lambda item: (item[0], tuple(-x for x in item[1])))[1]
        np.testing.assert_array_equal(decoded, expected)

    def test_spike_is_exactly_one_row(self) -> None:
        probabilities = np.asarray([0.01, 0.999, 0.01, 0.999, 0.01])
        decoded = decode_binary_segment(
            probabilities,
            minimum_duration=1,
            maximum_duration=1,
            start_penalty=1.0,
        )
        np.testing.assert_array_equal(decoded, [0, 1, 0, 1, 0])

    def test_minimum_duration_is_hard_and_end_censored_start_is_rejected(self) -> None:
        probabilities = np.asarray([0.01, 0.999, 0.999, 0.000001, 0.999])
        decoded = decode_binary_segment(
            probabilities,
            minimum_duration=2,
            maximum_duration=4,
            start_penalty=1.0,
        )
        np.testing.assert_array_equal(decoded, [0, 1, 1, 0, 0])

    def test_maximum_duration_forces_a_normal_separator_before_restart(self) -> None:
        probabilities = np.full(7, 0.999)
        decoded = decode_binary_segment(
            probabilities,
            minimum_duration=2,
            maximum_duration=3,
            start_penalty=0.1,
        )
        self.assertEqual(int(decoded.sum()), 6)
        self.assertTrue(_valid_run_lengths(tuple(decoded), 2, 3))

    def test_independent_chains_can_overlap_without_pairwise_model(self) -> None:
        probabilities = np.full((8, 5), 0.01, dtype=float)
        probabilities[1:5, 1] = 0.999
        probabilities[3:7, 2] = 0.999
        union, typed, no_op = decode_independent_types(_frame(8), probabilities, _config())
        self.assertGreater(int((typed.sum(axis=1) == 2).sum()), 0)
        self.assertEqual(int(union.sum()), 6)
        self.assertFalse(no_op.any())

    def test_gap_breaks_duration_chain(self) -> None:
        frame = _frame(6)
        frame.loc[3:, "time"] = pd.date_range(
            "2024-01-01T01:00:00+09:00", periods=3, freq="10min"
        ).astype(str)
        probabilities = np.full((6, 5), 0.01, dtype=float)
        probabilities[:, 1] = 0.999
        _, typed, _ = decode_independent_types(frame, probabilities, _config())
        self.assertEqual(int(typed[:, 1].sum()), 6)

    def test_invalid_unary_is_exact_type_specific_control_noop(self) -> None:
        probabilities = np.full((5, 5), 0.01, dtype=float)
        probabilities[2, 3] = np.nan
        union, typed, no_op = decode_independent_types(_frame(5), probabilities, _config())
        self.assertEqual(int(union[2]), 0)
        self.assertEqual(int(typed[2, 3]), 0)
        self.assertTrue(no_op[2, 3])

    def test_same_unary_control_uses_fixed_half_threshold(self) -> None:
        values = np.asarray([[0.49] * 5, [0.1, 0.5, 0.1, 0.1, 0.1]])
        np.testing.assert_array_equal(same_unary_control(values), [0, 1])

    def test_preregistration_is_adaptive_duration_only_and_forbids_outer_paths(self) -> None:
        path = ROOT / "configs" / "experiments" / "p1_typed_duration_semimarkov_v2.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(config["adaptive_research_disclosure"]["adaptive"])
        self.assertFalse(config["grammar"]["overlap_interaction_modeled"])
        self.assertFalse(config["grammar"]["raw_token_order_modeled"])
        self.assertFalse(config["grammar"]["raw_token_multiplicity_modeled"])
        self.assertEqual(len(config["historical_inner_blocks"]), 2)
        self.assertEqual(config["grammar"]["embargo_days"], 8)
        self.assertEqual(config["rowwise_control"]["threshold"], 0.5)
        for key in (
            "outer_validation_or_scoring",
            "test_read_or_prediction",
            "submission_generation_or_mutation",
            "upload",
            "existing_file_mutation",
            "hyperparameter_threshold_or_penalty_search",
        ):
            self.assertFalse(config["authorization"][key])

    def test_runner_static_surface_and_atomic_receipt(self) -> None:
        runner_path = ROOT / "scripts" / "run_p1_typed_duration_semimarkov.py"
        if not runner_path.exists():
            self.skipTest("runner is added after decoder unit tests")
        source = runner_path.read_text(encoding="utf-8")
        for forbidden in ("test.csv", "sample_submission.csv", "write_submission(", "upload("):
            self.assertNotIn(forbidden, source)
        spec = importlib.util.spec_from_file_location("duration_semimarkov_runner", runner_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            module._write_json_fsync(receipt, {"dry": True, "count": 0})
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["count"], 0)


if __name__ == "__main__":
    unittest.main()
