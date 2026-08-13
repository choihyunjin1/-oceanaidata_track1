from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.audit import audit_frame
from p1_qc.config import load_config
from p1_qc.data import (
    add_depth_regime,
    load_dataset,
    parse_anomaly_types,
    segment_timeseries,
)


def small_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * 4,
            "year": [2025] * 4,
            "layer": [1] * 4,
            "time": [
                "2025-01-01T00:00:00+09:00",
                "2025-01-01T00:10:00+09:00",
                "2025-01-01T00:30:00+09:00",
                "2025-01-01T00:40:00+09:00",
            ],
            "temp": [10.0, 10.1, 10.2, 10.3],
            "psal": [33.0, 33.0, np.nan, 33.1],
            "depth": [4.1, 4.2, 4.1, 4.2],
            "label": [0, 1, 0, 1],
            "anomaly_type": [None, "spike", None, "flatline+flatline"],
        }
    )


class DataTests(unittest.TestCase):
    def test_multilabel_parser_deduplicates_tokens(self) -> None:
        values = pd.Series([None, "noise+drift", "flatline+flatline", "spike"])
        parsed = parse_anomaly_types(values, strict=True)
        self.assertFalse(parsed.iloc[0].any())
        self.assertTrue(parsed.loc[1, "noise"])
        self.assertTrue(parsed.loc[1, "drift"])
        self.assertEqual(int(parsed.loc[2].sum()), 1)
        with self.assertRaises(ValueError):
            parse_anomaly_types(pd.Series(["unknown"]), strict=True)

    def test_segmentation_preserves_order_and_resets_on_gap(self) -> None:
        frame = small_train().iloc[[2, 0, 3, 1]].copy()
        original = frame.copy(deep=True)
        result = segment_timeseries(frame)
        pd.testing.assert_frame_equal(frame, original)
        # Restore chronological view to check two two-row segments.
        ordered = result.sort_values("parsed_time")
        self.assertEqual(ordered["segment_id"].nunique(), 2)
        self.assertEqual(ordered["position_in_segment"].tolist(), [0, 1, 0, 1])
        self.assertEqual(ordered["segment_size"].tolist(), [2, 2, 2, 2])
        self.assertTrue(np.isnan(ordered.iloc[0]["gap_minutes"]))
        self.assertEqual(float(ordered.iloc[2]["gap_minutes"]), 20.0)

    def test_depth_regime_uses_year_deployment_not_layer_identity(self) -> None:
        frame = pd.DataFrame(
            {
                "station": ["S-ORS"] * 4,
                "year": [2024, 2024, 2025, 2025],
                "layer": [7, 7, 7, 8],
                "depth": [49.0, 49.2, 39.5, 49.4],
            }
        )
        enriched = add_depth_regime(frame, width_m=2.5)
        self.assertEqual(enriched.loc[0, "depth_regime"], enriched.loc[1, "depth_regime"])
        self.assertEqual(enriched.loc[0, "depth_regime"], enriched.loc[3, "depth_regime"])
        self.assertNotEqual(enriched.loc[0, "depth_regime"], enriched.loc[2, "depth_regime"])

    def test_audit_detects_duplicate_key_and_label_semantics(self) -> None:
        frame = small_train()
        good = audit_frame(frame, kind="train")
        self.assertTrue(good.ok)
        broken = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        broken.loc[0, "label"] = 2
        report = audit_frame(broken, kind="train")
        codes = {issue.code for issue in report.errors}
        self.assertIn("duplicate_keys", codes)
        self.assertIn("invalid_label", codes)
        with self.assertRaises(ValueError):
            report.raise_for_errors()

    def test_load_dataset_fingerprints_without_changing_file(self) -> None:
        frame = small_train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame.to_csv(path, index=False)
            before = path.read_bytes()
            loaded = load_dataset(path, kind="train")
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(len(loaded.attrs["source_sha256"]), 64)
            self.assertTrue(loaded.attrs["audit_report"]["ok"])

    def test_toml_and_environment_config_preserve_unknown_mapping(self) -> None:
        content = """
seed = 7
[features]
rolling_hours = [1, 2]
[custom]
note = "kept"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1.toml"
            path.write_text(content, encoding="utf-8")
            config = load_config(
                path,
                env={"P1QC_FEATURES__MODE": '"causal"', "P1QC_SEED": "9"},
            )
        self.assertEqual(config.seed, 9)
        self.assertEqual(config.features.mode, "causal")
        self.assertEqual(config.features.rolling_hours, (1, 2))
        self.assertEqual(config.to_dict()["custom"]["note"], "kept")


if __name__ == "__main__":
    unittest.main()
