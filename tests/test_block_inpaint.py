from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from p1_qc.block_inpaint import (
    SCORE_COLUMNS,
    BlockInpaintConfig,
    apply_additive_gate,
    assert_mask_invariance,
    assert_target_safe_contract,
    build_model,
    build_safe_design,
    coverage_audit,
    enumerate_blocks,
    fit_additive_gate,
    fit_covariate_scaler,
    materialize_example,
    prepare_series,
)


def _frame(rows: int = 1000, *, gap_at: int | None = None) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    if gap_at is not None:
        time = time.to_series(index=np.arange(rows))
        time.iloc[gap_at:] += pd.Timedelta(minutes=10)
        time = pd.DatetimeIndex(time)
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "station": "S-ORS",
            "year": 2024,
            "layer": 1,
            "time": time.astype(str),
            "temp": 15.0 + np.sin(2.0 * np.pi * phase / 74.52),
            "psal": 32.0 + 0.1 * np.cos(2.0 * np.pi * phase / 144.0),
            "depth": 5.0,
            "label": np.zeros(rows, dtype=np.int8),
            "anomaly_type": "",
        }
    )


class BlockInpaintTests(unittest.TestCase):
    def test_frozen_context_contract_is_six_days(self) -> None:
        config = BlockInpaintConfig()
        self.assertEqual(config.mask_rows, (48, 144, 288, 576))
        self.assertEqual(config.stride_rows, 36)
        self.assertEqual(config.left_flank_rows, 144)
        self.assertEqual(config.maximum_context_rows, 864)

    def test_peer_mean_excludes_own_target(self) -> None:
        time = "2024-01-01 00:00:00+09:00"
        frame = pd.DataFrame(
            {
                "station": ["S-ORS", "S-ORS", "G-ORS"],
                "layer": [1, 2, 1],
                "time": [time, time, time],
                "temp": [10.0, 20.0, 30.0],
                "psal": [31.0, 33.0, 34.0],
                "depth": [5.0, 20.0, np.nan],
            }
        )
        design = build_safe_design(frame)
        self.assertAlmostEqual(design.continuous[0, 1], 20.0)
        self.assertAlmostEqual(design.continuous[1, 1], 10.0)
        self.assertTrue(np.isnan(design.continuous[2, 1]))
        changed = frame.copy()
        changed.loc[0, "temp"] = 999.0
        changed_design = build_safe_design(changed)
        self.assertAlmostEqual(changed_design.continuous[0, 1], 20.0)

    def test_blocks_never_cross_a_physical_gap(self) -> None:
        frame = _frame(1200, gap_at=600)
        design = build_safe_design(frame)
        scaler = fit_covariate_scaler(design, np.arange(len(frame)))
        prepared = prepare_series(frame, design, scaler, np.arange(len(frame)))
        specs = enumerate_blocks(prepared, BlockInpaintConfig(), normal_only=False)
        self.assertGreater(len(specs), 0)
        for spec in specs:
            lower = spec.start - 144
            upper = spec.stop + 144
            self.assertTrue(np.all(prepared.segment_ids[lower:upper] == spec.segment_id))

    def test_masked_target_is_not_a_model_input(self) -> None:
        frame = _frame(1000)
        design = build_safe_design(frame)
        assert_target_safe_contract(frame, design)
        scaler = fit_covariate_scaler(design, np.arange(len(frame)))
        prepared = prepare_series(frame, design, scaler, np.arange(len(frame)))
        spec = next(
            item
            for item in enumerate_blocks(prepared, BlockInpaintConfig(), normal_only=True)
            if item.length == 48
        )
        assert_mask_invariance(prepared, spec, BlockInpaintConfig())
        example = materialize_example(prepared, spec, BlockInpaintConfig())
        self.assertEqual(example["target_covariates"].shape[0], 48)
        self.assertEqual(example["left"].shape[0], 144)

    def test_normal_only_enumeration_rejects_anomaly_overlap(self) -> None:
        frame = _frame(1000)
        frame.loc[450:520, "label"] = 1
        design = build_safe_design(frame)
        scaler = fit_covariate_scaler(design, np.arange(len(frame)))
        prepared = prepare_series(frame, design, scaler, np.arange(len(frame)))
        specs = enumerate_blocks(prepared, BlockInpaintConfig(), normal_only=True)
        for spec in specs:
            window = frame.iloc[spec.start - 144 : spec.stop + 144]
            self.assertEqual(int(window["label"].sum()), 0)

    def test_coverage_and_no_peer_depth_fallback_are_finite(self) -> None:
        frame = _frame(1000)
        frame["station"] = "G-ORS"
        frame["depth"] = np.nan
        design = build_safe_design(frame)
        scaler = fit_covariate_scaler(design, np.arange(len(frame)))
        prepared = prepare_series(frame, design, scaler, np.arange(len(frame)))
        covered, audit = coverage_audit(prepared, BlockInpaintConfig())
        self.assertGreater(covered.sum(), 0)
        self.assertGreater(audit["covered_fraction"], 0)
        self.assertTrue(np.isfinite(prepared.covariates).all())

    def test_additive_gate_has_exact_uncovered_fallback(self) -> None:
        rng = np.random.default_rng(7)
        rows = 200
        labels = np.r_[np.zeros(170, dtype=np.int8), np.ones(30, dtype=np.int8)]
        probability = np.clip(0.05 + labels * 0.4 + rng.normal(0, 0.01, rows), 0.001, 0.999)
        scores = pd.DataFrame(
            {
                SCORE_COLUMNS[0]: labels + rng.normal(0, 0.1, rows),
                SCORE_COLUMNS[1]: labels + 0.1,
                SCORE_COLUMNS[2]: labels * 0.5,
                SCORE_COLUMNS[3]: np.full(rows, 0.2),
            }
        )
        gate = fit_additive_gate(probability, scores, labels)
        self.assertEqual(gate.coefficients.shape, (4,))
        scores.loc[0, list(SCORE_COLUMNS)] = np.nan
        candidate = apply_additive_gate(probability, scores, gate)
        self.assertAlmostEqual(candidate[0], probability[0], places=12)

    def test_model_forward_shape(self) -> None:
        import torch

        config = BlockInpaintConfig(use_bfloat16=False)
        model = build_model(14, 12, config)
        left = torch.zeros((2, 144, 14), dtype=torch.float32)
        right = torch.zeros((2, 144, 14), dtype=torch.float32)
        covariates = torch.zeros((2, 144, 12), dtype=torch.float32)
        lengths = torch.tensor([48, 144], dtype=torch.int64)
        forward, backward = model(left, right, covariates, lengths)
        self.assertEqual(tuple(forward.shape), (2, 144))
        self.assertEqual(tuple(backward.shape), (2, 144))
        self.assertTrue(torch.equal(forward[0, 48:], torch.zeros(96)))


if __name__ == "__main__":
    unittest.main()
