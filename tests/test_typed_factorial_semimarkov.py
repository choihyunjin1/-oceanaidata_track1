from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1_qc.typed_factorial_semimarkov import (  # noqa: E402
    DecoderConfig,
    build_grammar_audit,
    chronological_split_masks,
    decode_frame,
    decode_segment,
    duration_is_decomposable,
    parse_raw_anomaly_type,
    parse_raw_anomaly_types,
    rowwise_union,
)


def _frame(types: list[str], *, start: str = "2024-01-01T00:00:00+09:00") -> pd.DataFrame:
    time = pd.date_range(start, periods=len(types), freq="10min")
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * len(types),
            "layer": [2] * len(types),
            "time": time.astype(str),
            "label": [int(bool(item)) for item in types],
            "anomaly_type": types,
        }
    )


def _small_decoder() -> DecoderConfig:
    return DecoderConfig(
        duration_rows={
            "spike": (1, 1),
            "noise": (2, 4),
            "flatline": (2, 4),
            "offset": (3, 5),
            "drift": (3, 5),
        },
        beam_width=32,
        start_penalty=1.0,
        overlap_penalty=0.2,
        duplicate_type_start_penalty=1.0,
    )


class TypedFactorialSemiMarkovTests(unittest.TestCase):
    def test_parser_retains_order_and_multiplicity(self) -> None:
        parsed = parse_raw_anomaly_type("flatline+flatline")
        self.assertEqual(parsed.raw, "flatline+flatline")
        self.assertEqual(parsed.tokens, ("flatline", "flatline"))
        self.assertEqual(parsed.counts, (0, 0, 2, 0, 0))
        self.assertEqual(parsed.membership, (False, False, True, False, False))
        reversed_pair = parse_raw_anomaly_type("drift+offset")
        self.assertEqual(reversed_pair.tokens, ("drift", "offset"))
        self.assertNotEqual(reversed_pair.tokens, ("offset", "drift"))
        empty = parse_raw_anomaly_type(pd.NA)
        self.assertEqual(empty.tokens, ())
        with self.assertRaises(ValueError):
            parse_raw_anomaly_type("spike+unknown")

    def test_vector_parser_exposes_counts_without_deduplication(self) -> None:
        parsed, counts = parse_raw_anomaly_types(
            pd.Series(["", "offset+drift", "flatline+flatline"])
        )
        self.assertEqual(parsed[1].tokens, ("offset", "drift"))
        self.assertEqual(int(counts[2, 2]), 2)
        self.assertEqual(int(counts[2].sum()), 2)

    def test_super_events_connect_touching_and_overlapping_runs(self) -> None:
        types = [""] * 12
        types[1] = "spike"
        types[2:5] = ["noise"] * 3
        types[3:5] = ["noise+offset"] * 2
        types[5:8] = ["offset"] * 3
        types[10] = "spike"
        audit = build_grammar_audit(
            _frame(types),
            duration_rows={
                "spike": (1, 1),
                "noise": (3, 3),
                "flatline": (2, 8),
                "offset": (5, 5),
                "drift": (2, 8),
            },
        )
        # The spike at row 1 touches noise starting at row 2, which overlaps
        # offset, so all three belong to one super-event. Row 10 is separate.
        self.assertEqual(len(audit.super_events), 2)
        self.assertEqual(audit.row_super_event_ids[1], audit.row_super_event_ids[7])
        self.assertNotEqual(audit.row_super_event_ids[1], audit.row_super_event_ids[10])
        self.assertTrue(audit.duration_decomposable_positive_rows[1:8].all())

    def test_gap_splits_super_events_even_if_positive_rows_are_adjacent(self) -> None:
        frame = _frame(["spike", "spike"])
        frame.loc[1, "time"] = "2024-01-01 00:20:00+09:00"
        audit = build_grammar_audit(frame)
        self.assertEqual(len(audit.super_events), 2)

    def test_duration_decomposition_supports_adjacent_events(self) -> None:
        self.assertTrue(duration_is_decomposable(96, 48, 519))
        self.assertTrue(duration_is_decomposable(1040, 48, 519))
        self.assertFalse(duration_is_decomposable(47, 48, 519))
        self.assertTrue(duration_is_decomposable(4, 1, 1))

    def test_eight_day_embargo_and_super_event_disjointness(self) -> None:
        frame = _frame([""] * 3000)
        frame.loc[100, ["label", "anomaly_type"]] = [1, "spike"]
        audit = build_grammar_audit(frame)
        fit, validation = chronological_split_masks(
            frame,
            audit,
            fit_end_inclusive="2024-01-05T00:00:00+09:00",
            validation_start_inclusive="2024-01-14T00:10:00+09:00",
            validation_end_inclusive="2024-01-20T00:00:00+09:00",
            embargo_days=8,
        )
        self.assertGreater(int(fit.sum()), 0)
        self.assertGreater(int(validation.sum()), 0)
        self.assertFalse((fit & validation).any())
        with self.assertRaises(ValueError):
            chronological_split_masks(
                frame,
                audit,
                fit_end_inclusive="2024-01-05T00:00:00+09:00",
                validation_start_inclusive="2024-01-13T00:00:00+09:00",
                validation_end_inclusive="2024-01-20T00:00:00+09:00",
                embargo_days=8,
            )

    def test_rowwise_control_is_union_of_exact_same_unaries(self) -> None:
        probabilities = np.asarray(
            [[0.49, 0.1, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1, 0.1]], dtype=float
        )
        np.testing.assert_array_equal(rowwise_union(probabilities), [0, 1])

    def test_decoder_enforces_spike_and_slow_duration_support(self) -> None:
        probabilities = np.full((9, 5), 0.01, dtype=float)
        probabilities[1, 0] = 0.999
        probabilities[4:7, 3] = 0.999
        union, typed = decode_segment(probabilities, _small_decoder())
        self.assertEqual(int(typed[:, 0].sum()), 1)
        self.assertEqual(int(typed[:, 3].sum()), 3)
        self.assertEqual(int(union.sum()), 4)

    def test_decoder_is_deterministic_and_never_exceeds_two_active_types(self) -> None:
        rng = np.random.default_rng(20260822)
        probabilities = rng.uniform(0.05, 0.99, size=(24, 5))
        first_union, first_types = decode_segment(probabilities, _small_decoder())
        second_union, second_types = decode_segment(probabilities, _small_decoder())
        np.testing.assert_array_equal(first_union, second_union)
        np.testing.assert_array_equal(first_types, second_types)
        self.assertLessEqual(int(first_types.sum(axis=1).max()), 2)

    def test_decode_frame_breaks_at_gap_and_fails_closed_on_invalid_unary(self) -> None:
        frame = _frame([""] * 5)
        frame.loc[3:, "time"] = pd.date_range(
            "2024-01-01T01:00:00+09:00", periods=2, freq="10min"
        ).astype(str)
        probabilities = np.full((5, 5), 0.01, dtype=float)
        probabilities[0, 0] = 0.999
        probabilities[3, 0] = 0.999
        probabilities[4, 2] = np.nan
        union, typed, no_op = decode_frame(frame, probabilities, _small_decoder())
        self.assertEqual(int(union[0]), 1)
        self.assertEqual(int(union[3]), 1)
        self.assertTrue(no_op[4])
        self.assertLessEqual(int(typed.sum(axis=1).max()), 2)

    def test_preregistration_has_no_outer_test_or_submission_authority(self) -> None:
        config_path = ROOT / "configs" / "experiments" / "p1_typed_factorial_semimarkov_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertFalse(config["authorization"]["outer_validation_or_scoring"])
        self.assertFalse(config["authorization"]["test_read_or_prediction"])
        self.assertFalse(config["authorization"]["submission_generation_or_mutation"])
        self.assertFalse(config["authorization"]["upload"])
        self.assertFalse(config["authorization"]["hyperparameter_or_threshold_search"])
        self.assertEqual(config["grammar"]["maximum_concurrent_events"], 2)
        self.assertEqual(config["grammar"]["embargo_days"], 8)
        self.assertEqual(config["rowwise_control"]["threshold"], 0.5)
        self.assertEqual(len(config["historical_inner_blocks"]), 3)

    def test_runner_exposes_only_canonical_historical_entrypoint(self) -> None:
        runner_path = ROOT / "scripts" / "run_p1_typed_factorial_semimarkov.py"
        source = runner_path.read_text(encoding="utf-8")
        for forbidden in ("test.csv", "sample_submission.csv", "write_submission(", "upload("):
            self.assertNotIn(forbidden, source)
        spec = importlib.util.spec_from_file_location("typed_semimarkov_runner", runner_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module._read_config()
        self.assertEqual(config["experiment_id"], "p1_typed_factorial_semimarkov_v1")
        self.assertEqual(
            set(module._implementation_hashes()),
            {
                "configs/experiments/p1_typed_factorial_semimarkov_v1.json",
                "src/p1_qc/typed_factorial_semimarkov.py",
                "scripts/run_p1_typed_factorial_semimarkov.py",
                "tests/test_typed_factorial_semimarkov.py",
            },
        )

    def test_runner_atomic_json_roundtrip_is_finite(self) -> None:
        runner_path = ROOT / "scripts" / "run_p1_typed_factorial_semimarkov.py"
        spec = importlib.util.spec_from_file_location("typed_semimarkov_runner_json", runner_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            module._write_json_fsync(path, {"finite": 1.0, "count": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"finite": 1.0, "count": 2}
            )
            with self.assertRaises(ValueError):
                module._write_json_fsync(path, {"not_finite": float("nan")})


if __name__ == "__main__":
    unittest.main()
