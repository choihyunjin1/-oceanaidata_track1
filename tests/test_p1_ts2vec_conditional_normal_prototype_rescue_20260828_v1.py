from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from p1_qc.ts2vec_conditional_normal_prototype import (
    HierarchicalContrastiveEncoder,
    anchor_union,
    contiguous_windows,
    contrastive_matrix_bound,
    day_block_bootstrap_probability,
    decode_components,
    finite_normal_tail_threshold,
    fit_conditional_prototype,
    hierarchical_contrastive_loss,
    score_conditional_prototype,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "scripts/run_p1_ts2vec_conditional_normal_prototype_rescue_20260828_v1.py"
    spec = importlib.util.spec_from_file_location("p1_ts2vec_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_and_hierarchical_loss_are_finite() -> None:
    model = HierarchicalContrastiveEncoder(3, hidden_width=8, embedding_width=6, dilations=(1, 2))
    first = model(torch.randn(3, 32, 3))
    second = model(torch.randn(3, 32, 3))
    loss = hierarchical_contrastive_loss(first, second, timestamp_cap=16)
    assert first.shape == (3, 32, 6)
    assert torch.isfinite(loss)


def test_full_shape_contrastive_matrices_are_bounded() -> None:
    bound = contrastive_matrix_bound(batch=64, rows=512, timestamp_cap=128)
    assert bound["instance_matrix_side"] == 64
    assert bound["temporal_matrix_side"] == 128
    assert bound["maximum_single_matrix_elements"] == 16_384
    assert bound["maximum_single_matrix_elements"] < (64 * 512) ** 2


def test_windows_do_not_cross_gap_or_ineligible_rows() -> None:
    segments = np.array([1] * 9 + [2] * 9)
    eligible = np.ones(18, dtype=bool)
    eligible[4] = False
    windows = contiguous_windows(segments, eligible, window_rows=4, stride_rows=3)
    assert windows.tolist() == [[0, 4], [5, 9], [9, 13], [12, 16], [14, 18]]


def test_conditional_prototype_scores_reference_lower_than_shift() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(0, 0.1, size=(80, 4)).astype(np.float32)
    day_sin = np.sin(np.linspace(0, 2 * np.pi, 80))
    day_cos = np.cos(np.linspace(0, 2 * np.pi, 80))
    cells = np.array(["G/1"] * 40 + ["I/1"] * 40)
    state = fit_conditional_prototype(
        embeddings,
        day_sin,
        day_cos,
        cells,
        np.ones(80, dtype=bool),
        shrinkage_rows=8,
        bank_per_cell=20,
    )
    reference = score_conditional_prototype(state, embeddings, day_sin, day_cos, cells)
    shifted = score_conditional_prototype(
        state, embeddings + 3.0, day_sin, day_cos, cells
    )
    assert float(np.nanmedian(shifted)) > float(np.nanmedian(reference))


def test_finite_tail_and_decoder_are_fixed_not_prevalence_targeted() -> None:
    assert finite_normal_tail_threshold(np.arange(100, dtype=float), alpha=0.1) == 90.0
    score = np.r_[np.zeros(3), np.ones(4) * 3, np.zeros(2), np.ones(4) * 3]
    prediction = decode_components(
        score, np.zeros(len(score)), 2.0, minimum_rows=10, bridge_rows=2
    )
    assert prediction.sum() == 10


def test_anchor_union_never_deletes_anchor_rows() -> None:
    anchor = np.array([1, 0, 1, 0], dtype=np.int8)
    additions = np.array([0, 1, 0, 0], dtype=np.int8)
    candidate = anchor_union(anchor, additions)
    assert candidate.tolist() == [1, 1, 1, 0]
    assert np.all(candidate[anchor == 1] == 1)


def test_day_block_bootstrap_is_deterministic_and_directional() -> None:
    truth = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    candidate = truth.copy()
    anchor = np.zeros(6, dtype=np.int8)
    days = np.array([0, 0, 1, 1, 2, 2])
    first = day_block_bootstrap_probability(
        truth, candidate, anchor, days, block_days=1, replicates=50, seed=7
    )
    second = day_block_bootstrap_probability(
        truth, candidate, anchor, days, block_days=1, replicates=50, seed=7
    )
    assert first == second == 1.0


def test_check_only_defers_q2_and_reads_no_official_rows() -> None:
    runner = _runner()
    config = runner._json(runner.CONFIG)
    receipt = runner.validate_contract(config)
    assert receipt["status"] == "PASS"
    assert receipt["inputs"]["q2_e150_anchor"]["deferred"] is True
    assert receipt["q2_truth_rows_read"] == 0
    assert receipt["q3_q4_rows_read"] == 0
    assert receipt["official_rows_read"] == 0
