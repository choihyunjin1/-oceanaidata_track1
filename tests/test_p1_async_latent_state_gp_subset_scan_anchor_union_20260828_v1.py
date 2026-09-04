from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1 import (
    best_subset_interval,
    block_proposals,
    conformal_threshold,
    loo_matern_smoother,
    matern32_transition,
)


def test_matern_transition_has_positive_semidefinite_innovation() -> None:
    transition, innovation = matern32_transition(600.0, 6.0 * 3600.0, 0.5)
    assert transition.shape == (2, 2)
    assert np.max(np.abs(innovation - innovation.T)) < 1e-12
    assert np.linalg.eigvalsh(innovation).min() >= -1e-12


def test_loo_state_uses_peer_observation_and_never_target_values() -> None:
    count = 288
    times = np.arange(count, dtype=np.int64) * int(pd.Timedelta(minutes=10).value)
    peers = np.sin(np.arange(count) / 30.0)
    peer_count = np.full(count, 2, dtype=np.int16)
    mean_a, variance_a, observed_a = loo_matern_smoother(
        times,
        peers,
        peer_count,
        minimum_peers=2,
        lengthscale_hours=(6.0, 48.0),
        factor_variance=(0.5, 0.5),
        observation_variance=0.25,
    )
    mean_b, variance_b, observed_b = loo_matern_smoother(
        times,
        peers.copy(),
        peer_count,
        minimum_peers=2,
        lengthscale_hours=(6.0, 48.0),
        factor_variance=(0.5, 0.5),
        observation_variance=0.25,
    )
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(variance_a, variance_b)
    np.testing.assert_array_equal(observed_a, observed_b)
    assert np.isfinite(mean_a).all()
    assert np.isfinite(variance_a).all()


def test_fixed_subset_scan_recovers_long_offset() -> None:
    residual = np.zeros(800, dtype=np.float64)
    residual[240:336] = 3.0
    result = best_subset_interval(
        residual,
        {
            "noise": [18, 36, 72, 144, 288, 353],
            "offset": [48, 96, 192, 384, 519],
            "drift": [54, 108, 216, 432, 519],
        },
    )
    assert result is not None
    score, anomaly_type, duration, start, stop = result
    assert score > 800.0
    assert anomaly_type == "offset"
    assert duration == 96
    assert (start, stop) == (240, 336)


def test_block_decoder_emits_at_most_one_interval_per_block() -> None:
    count = 1008
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=count, freq="10min", tz="Asia/Seoul"),
            "residual": np.r_[np.zeros(300), np.full(96, 3.0), np.zeros(count - 396)],
        }
    )
    proposals = block_proposals(
        frame,
        {"noise": [18], "offset": [96], "drift": [108]},
        block_days=7,
    )
    assert len(proposals) <= 2
    assert sum(proposal.score > 0 for proposal in proposals) >= 1
    assert len({proposal.block_id for proposal in proposals}) == len(proposals)


def test_conformal_threshold_uses_sealed_order_statistic() -> None:
    scores = np.arange(100, dtype=np.float64)
    assert conformal_threshold(scores, 0.01) == 99.0
