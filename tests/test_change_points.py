from __future__ import annotations

import inspect

import numpy as np

from p1_qc.change_points import (
    ChangePointConfig,
    best_per_seed,
    filter_proposals,
    proposals_to_mask,
    propose_change_intervals,
)


def _epidemic_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.r_[np.zeros(12), np.full(10, 4.0), np.zeros(12)]
    probability = np.r_[np.full(12, 0.05), np.full(10, 0.9), np.full(12, 0.05)]
    segment = np.zeros(len(residual), dtype=int)
    return residual, probability, segment


def test_offline_proposal_is_global_half_open_bounded_and_label_free() -> None:
    residual, probability, segment = _epidemic_signal()
    result = propose_change_intervals(
        residual,
        probability,
        segment,
        station=np.array(["S-ORS"] * len(residual)),
        layer=np.ones(len(residual), dtype=int),
        config=ChangePointConfig(max_flank_rows=12),
    )
    assert "label" not in inspect.signature(propose_change_intervals).parameters
    assert result.provenance["labels_used"] is False
    assert result.seed_runs == 1
    assert 1 <= len(result.proposals) <= result.config.max_candidates_per_seed_run
    proposal = best_per_seed(result)[0]
    assert proposal.start <= 12 < 22 <= proposal.stop
    assert proposal.context_start >= 0
    assert proposal.context_stop <= len(residual)
    assert proposal.has_return
    assert proposal.sources
    assert proposal.duration_soft_score > 0
    mask = proposals_to_mask(result, len(residual), top_k_per_seed=1)
    assert mask.shape == residual.shape
    assert mask[proposal.start : proposal.stop].all()


def test_proposals_never_cross_segment_station_layer_or_nan_boundary() -> None:
    one_residual = np.r_[np.zeros(5), np.full(4, 5.0), np.zeros(5)]
    one_probability = np.r_[np.zeros(5), np.full(4, 0.9), np.zeros(5)]
    residual = np.r_[one_residual, np.nan, one_residual]
    probability = np.r_[one_probability, 0.9, one_probability]
    segment = np.r_[np.zeros(15, dtype=int), np.ones(14, dtype=int)]
    station = np.array(["A"] * 15 + ["B"] * 14)
    layer = np.r_[np.ones(15, dtype=int), np.full(14, 2, dtype=int)]
    result = propose_change_intervals(
        residual,
        probability,
        segment,
        station=station,
        layer=layer,
        config=ChangePointConfig(
            min_baseline_rows=3,
            min_return_rows=3,
            max_flank_rows=6,
            mean_gain_threshold=0.1,
        ),
    )
    assert result.proposals
    for proposal in result.proposals:
        positions = np.arange(proposal.start, proposal.stop)
        assert np.isfinite(residual[positions]).all()
        assert len(set(station[positions])) == 1
        assert len(set(layer[positions])) == 1
        assert len(set(segment[positions])) == 1


def test_nan_no_seed_and_short_segments_are_safe() -> None:
    empty = propose_change_intervals([], [], [], config=ChangePointConfig())
    assert empty.proposals == ()
    no_seed = propose_change_intervals(
        [0.0, np.nan, 0.0],
        [0.1, 0.9, np.nan],
        [0, 0, 0],
    )
    assert no_seed.proposals == ()
    assert proposals_to_mask(no_seed, 3).tolist() == [False, False, False]


def test_causal_prefix_is_invariant_to_future_values() -> None:
    residual = np.r_[np.zeros(10), 4.0, 4.0, np.zeros(8)]
    probability = np.r_[np.full(10, 0.05), 0.9, 0.8, np.full(8, 0.05)]
    config = ChangePointConfig(
        mode="causal",
        min_baseline_rows=4,
        max_flank_rows=8,
        mean_gain_threshold=0.1,
    )
    prefix = propose_change_intervals(residual[:11], probability[:11], np.zeros(11), config=config)
    full = propose_change_intervals(residual, probability, np.zeros(len(residual)), config=config)
    assert prefix.to_dict() == {
        **full.to_dict(),
        "proposals": [
            proposal.to_dict() for proposal in full.proposals if proposal.decision_stop <= 11
        ],
    }
    assert all(proposal.context_stop == proposal.decision_stop for proposal in full.proposals)
    assert all(not proposal.has_return for proposal in full.proposals)


def test_filter_and_best_per_seed_control_union_expansion() -> None:
    residual, probability, segment = _epidemic_signal()
    result = propose_change_intervals(
        residual,
        probability,
        segment,
        config=ChangePointConfig(max_flank_rows=12, max_candidates_per_seed_run=8),
    )
    best = best_per_seed(result, top_k=1)
    assert len(best) == 1
    assert best[0].total_score == max(item.total_score for item in result.proposals)
    thresholded = filter_proposals(result, min_total_score=best[0].total_score)
    assert thresholded
    assert all(item.total_score >= best[0].total_score for item in thresholded)
    assert not filter_proposals(result, sources=("mean",), min_total_score=np.inf)


def test_return_filter_uses_python_bool_without_dropping_numpy_finite_results() -> None:
    residual, probability, segment = _epidemic_signal()
    offline = propose_change_intervals(
        residual,
        probability,
        segment,
        config=ChangePointConfig(max_flank_rows=12),
    )
    returning = filter_proposals(offline, require_return=True)
    assert returning
    assert len(returning) == len(offline.proposals)
    assert all(type(proposal.has_return) is bool for proposal in offline.proposals)
    assert not filter_proposals(offline, require_return=False)

    causal = propose_change_intervals(
        residual,
        probability,
        segment,
        config=ChangePointConfig(
            mode="causal",
            max_flank_rows=12,
            mean_gain_threshold=0.1,
        ),
    )
    without_return = filter_proposals(causal, require_return=False)
    assert without_return
    assert len(without_return) == len(causal.proposals)
    assert all(type(proposal.has_return) is bool for proposal in causal.proposals)
    assert not filter_proposals(causal, require_return=True)


def test_variance_and_continuous_slope_experts_are_exposed() -> None:
    probability = np.r_[np.full(12, 0.05), np.full(10, 0.9), np.full(12, 0.05)]
    segment = np.zeros(len(probability), dtype=int)
    noise = np.r_[np.zeros(12), np.tile([-4.0, 4.0], 5), np.zeros(12)]
    noise_result = propose_change_intervals(
        noise,
        probability,
        segment,
        config=ChangePointConfig(max_flank_rows=12),
    )
    assert any("variance" in proposal.sources for proposal in noise_result.proposals)

    ramp = np.r_[np.zeros(12), np.linspace(0.0, 6.0, 10), np.zeros(12)]
    ramp_result = propose_change_intervals(
        ramp,
        probability,
        segment,
        config=ChangePointConfig(max_flank_rows=12),
    )
    assert any("slope" in proposal.sources for proposal in ramp_result.proposals)


def test_duration_landmarks_are_soft_and_do_not_clip_candidates() -> None:
    residual, probability, segment = _epidemic_signal()
    short = propose_change_intervals(
        residual,
        probability,
        segment,
        config=ChangePointConfig(
            max_flank_rows=12,
            min_interval_rows=100,
            max_interval_rows=120,
        ),
    )
    assert short.proposals
    assert all(proposal.duration_rows < 100 for proposal in short.proposals)
    assert all(0 < proposal.duration_soft_score < 1 for proposal in short.proposals)

    long_residual = np.r_[np.zeros(12), np.full(30, 4.0), np.zeros(12)]
    long_probability = np.r_[np.full(12, 0.05), np.full(30, 0.9), np.full(12, 0.05)]
    long = propose_change_intervals(
        long_residual,
        long_probability,
        np.zeros(len(long_residual), dtype=int),
        config=ChangePointConfig(
            max_flank_rows=12,
            min_interval_rows=6,
            max_interval_rows=10,
        ),
    )
    assert long.proposals
    assert any(proposal.duration_rows > 10 for proposal in long.proposals)
