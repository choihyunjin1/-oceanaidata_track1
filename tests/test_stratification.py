from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p1_qc.features import build_features
from p1_qc.stratification import (
    PEER_GATE_FEATURES,
    PeerGateConfig,
    append_stratification_peer_gate,
    build_stratification_peer_gate,
)


def layered_frame(
    first: np.ndarray,
    second: np.ndarray,
    *,
    gap_at: int | None = None,
) -> pd.DataFrame:
    if len(first) != len(second):
        raise ValueError("layers must have equal length")
    timestamps: list[str] = []
    current = pd.Timestamp("2025-05-01T00:00:00+09:00")
    for index in range(len(first)):
        if index:
            current += pd.Timedelta(minutes=20 if gap_at == index else 10)
        timestamps.append(current.isoformat())
    rows: list[dict[str, object]] = []
    for layer, values, depth in ((1, first, 5.0), (2, second, 25.0)):
        for timestamp, value in zip(timestamps, values, strict=True):
            rows.append(
                {
                    "station": "SYNTH",
                    "year": 2025,
                    "layer": layer,
                    "time": timestamp,
                    "temp": float(value),
                    "psal": 33.0,
                    "depth": depth,
                }
            )
    return pd.DataFrame(rows)


class StratificationPeerGateTests(unittest.TestCase):
    def test_coherent_changes_receive_high_trust(self) -> None:
        phase = np.linspace(0.0, 4.0 * np.pi, 36)
        frame = layered_frame(np.sin(phase), np.sin(phase) + 3.0)
        features = build_stratification_peer_gate(
            frame,
            config=PeerGateConfig(mode="offline", window_hours=2, min_period_fraction=0.5),
        )
        middle = features.iloc[12:24]
        self.assertGreater(float(middle["peer_change_corr_24h"].median()), 0.99)
        self.assertGreater(float(middle["peer_trust_gate_24h"].median()), 0.9)

    def test_opposing_layer_changes_are_not_trusted(self) -> None:
        phase = np.linspace(0.0, 4.0 * np.pi, 36)
        frame = layered_frame(np.sin(phase), -np.sin(phase) + 3.0)
        features = build_stratification_peer_gate(
            frame,
            config=PeerGateConfig(mode="offline", window_hours=2, min_period_fraction=0.5),
        )
        middle = features.iloc[12:24]
        self.assertLess(float(middle["peer_change_corr_24h"].median()), -0.99)
        self.assertEqual(float(middle["peer_trust_gate_24h"].max()), 0.0)
        self.assertEqual(float(middle["temp_abs_peer_residual_gated_24h"].max()), 0.0)

    def test_single_layer_has_explicit_zero_trust_fallback(self) -> None:
        phase = np.linspace(0.0, 2.0 * np.pi, 24)
        frame = layered_frame(np.sin(phase), np.sin(phase) + 1.0)
        frame = frame.loc[frame["layer"] == 1].reset_index(drop=True)
        features = build_stratification_peer_gate(frame)
        self.assertTrue(features["peer_change_corr_24h"].isna().all())
        self.assertTrue((features["peer_pair_coverage_24h"] == 0).all())
        self.assertTrue((features["peer_trust_gate_24h"] == 0).all())
        self.assertTrue(features["temp_abs_peer_residual_gated_24h"].isna().all())

    def test_causal_prefix_and_gap_boundaries(self) -> None:
        phase = np.linspace(0.0, 6.0 * np.pi, 48)
        prefix = layered_frame(np.sin(phase[:36]), np.sin(phase[:36]) + 2.0, gap_at=24)
        full = layered_frame(np.sin(phase), np.sin(phase) + 2.0, gap_at=24)
        config = PeerGateConfig(mode="causal", window_hours=1, min_period_fraction=0.5)
        prefix_features = build_stratification_peer_gate(prefix, config=config)
        full_features = build_stratification_peer_gate(full, config=config)
        full_prefix = full_features.loc[full["time"].isin(prefix["time"])].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            prefix_features.reset_index(drop=True),
            full_prefix,
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        gap_rows = prefix.groupby("layer", sort=False).nth(24).index
        self.assertTrue(prefix_features.loc[gap_rows, "peer_change_corr_24h"].isna().all())

    def test_append_is_exactly_four_feature_ablation(self) -> None:
        phase = np.linspace(0.0, 2.0 * np.pi, 24)
        frame = layered_frame(np.sin(phase), np.sin(phase) + 1.0)
        base = build_features(frame, mode="causal")
        augmented = append_stratification_peer_gate(
            base,
            frame,
            config=PeerGateConfig(mode="causal", window_hours=1),
        )
        self.assertEqual(augmented.feature_columns[-4:], PEER_GATE_FEATURES)
        self.assertEqual(len(augmented.feature_columns), len(base.feature_columns) + 4)
        self.assertEqual(augmented.categorical_columns, base.categorical_columns)
        self.assertNotIn("label", augmented.feature_columns)


if __name__ == "__main__":
    unittest.main()
